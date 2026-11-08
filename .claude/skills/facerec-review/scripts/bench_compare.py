#!/usr/bin/env python3
"""Benchmark base vs head facerec code on the same synthetic clip and flag regressions.

Usage (from the repo root, with the project's Python):
    python .claude/skills/facerec-review/scripts/bench_compare.py --base main
    python .claude/skills/facerec-review/scripts/bench_compare.py --base facerec/02-core --frames 300

How it works:
  1. Builds a 1280x720 clip of 3 moving faces from data/faces (mirrored copies if fewer people).
  2. Runs `python -m facerec run --benchmark N` on the head (working tree) code.
  3. Checks out the base into a temporary git worktree, copies models/ into it, and runs the same.
  4. Prints both side by side and flags regressions.

Each side runs --repeat times (default 2); the run with the lowest total time is kept,
which reduces noise.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def git(root: Path, *args: str) -> str:
    res = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout.strip()


def make_clip(root: Path, out: Path, frames: int = 300) -> int:
    """Synthetic clip; returns the number of faces placed per frame."""
    sys.path.insert(0, str(root / "src"))
    import cv2
    import numpy as np
    from facerec.config import load_config
    from facerec.registry import create_detector

    cfg = load_config(root / "config.yaml")
    detector = create_detector(cfg)
    faces_dir = cfg.resolve(cfg.paths.faces_dir)
    people = sorted(p for p in faces_dir.iterdir() if p.is_dir()) if faces_dir.exists() else []
    crops = []
    for person in people:
        imgs = sorted(f for f in person.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        for f in imgs:
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
            dets = detector.detect(img) if img is not None else []
            if not dets:
                continue
            d = max(dets, key=lambda d: d.area)
            x, y, w, h = d.bbox
            s = max(w, h) * 1.8
            x0, y0 = int(max(0, x + w / 2 - s / 2)), int(max(0, y + h / 2 - s / 2))
            crop = img[y0:int(y0 + s), x0:int(x0 + s)]
            crops.append(cv2.resize(crop, (260, 260)))
            break
        if len(crops) == 3:
            break
    if not crops:
        raise RuntimeError(f"No detectable enrolment photos in {faces_dir}; cannot build a clip.")
    i = 0
    while len(crops) < 3:
        crops.append(cv2.flip(crops[i], 1))
        i += 1

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720))
    for n in range(frames):
        frame = np.full((720, 1280, 3), 90, np.uint8)
        for k, c in enumerate(crops):
            fx = int(80 + k * 400 + 60 * math.sin(n / 25 + k))
            fy = int(230 + 80 * math.cos(n / 30 + k))
            frame[fy:fy + 260, fx:fx + 260] = c
        writer.write(frame)
    writer.release()
    return len(crops)


def run_bench(code_root: Path, clip: Path, frames: int, repeat: int, every_frame: bool) -> dict | None:
    env = dict(os.environ, PYTHONPATH=str(code_root / "src"))
    probe = subprocess.run([sys.executable, "-c", "import facerec; print(facerec.__file__)"],
                           capture_output=True, text=True, env=env, cwd=code_root)
    loaded = Path(probe.stdout.strip()).resolve() if probe.returncode == 0 else None
    if loaded is None or (code_root / "src").resolve() not in loaded.parents:
        print(f"  ! could not import facerec from {code_root / 'src'} (got {loaded})")
        return None
    cmd = [sys.executable, "-m", "facerec", "-c", str(code_root / "config.yaml"), "run",
           "--source", str(clip), "--benchmark", str(frames)]
    if every_frame:
        cmd.append("--every-frame")
    best = None
    for _ in range(repeat):
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=code_root)
        if res.returncode != 0:
            print(f"  ! benchmark failed in {code_root}:\n{(res.stderr or res.stdout)[-1500:]}")
            return None
        parsed = parse(res.stdout)
        if parsed and (best is None or parsed["total"] < best["total"]):
            best = parsed
    return best


def parse(out: str) -> dict | None:
    r: dict[str, float] = {}
    for stage in ("detect", "track", "embed", "match", "total"):
        m = re.search(rf"^{stage}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", out, re.M)
        if m:
            r[stage] = float(m.group(1))
            if stage == "total":
                r["total_p95"] = float(m.group(3))
    m = re.search(r"faces/frame ([\d.]+) \| embeddings/frame ([\d.]+)", out)
    if m:
        r["faces"], r["embeds"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"pipeline capacity ([\d.]+) fps", out)
    if m:
        r["fps"] = float(m.group(1))
    return r if "total" in r else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="main", help="base branch/commit (default: main)")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--every-frame", action="store_true", help="benchmark without tracking")
    args = ap.parse_args()

    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if not (root / "models").exists() or not any((root / "models").glob("*.onnx")):
        print("Models missing: run `facerec download-models` first.")
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="facerec-bench-"))
    worktree = tmp / "base"
    try:
        clip = tmp / "clip.avi"
        n_faces = make_clip(root, clip, frames=max(args.frames, 150))
        print(f"Clip: 1280x720, {n_faces} moving faces, {clip}")

        print("Benchmarking head (working tree) ...")
        head = run_bench(root, clip, args.frames, args.repeat, args.every_frame)

        base = None
        base_sha = git(root, "rev-parse", "--short", args.base)
        git(root, "worktree", "add", "--detach", str(worktree), args.base)
        if not (worktree / "src" / "facerec").exists():
            print(f"Base {args.base} ({base_sha}) has no src/facerec (pre-rewrite code); "
                  "head numbers only.")
        else:
            shutil.copytree(root / "models", worktree / "models", dirs_exist_ok=True)
            print(f"Benchmarking base {args.base} ({base_sha}) ...")
            base = run_bench(worktree, clip, args.frames, args.repeat, args.every_frame)
    finally:
        if worktree.exists():
            subprocess.run(["git", "-C", str(root), "worktree", "remove", "--force", str(worktree)],
                           capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)

    if head is None:
        print("Head benchmark failed.")
        return 1
    rows = [("detect ms", "detect"), ("track ms", "track"), ("embed ms", "embed"),
            ("match ms", "match"), ("total ms", "total"), ("total p95 ms", "total_p95"),
            ("FPS capacity", "fps"), ("embeddings/frame", "embeds"), ("faces/frame", "faces")]
    print(f"\n{'metric':<18}{'base':>10}{'head':>10}{'change':>10}")
    for label, key in rows:
        h = head.get(key)
        b = base.get(key) if base else None
        if h is None:
            continue
        change = f"{(h - b) / b * 100:+.0f}%" if b not in (None, 0) else ""
        bs = f"{b:.2f}" if b is not None else "-"
        print(f"{label:<18}{bs:>10}{h:>10.2f}{change:>10}")

    flags = []
    if base:
        if head["total"] > base["total"] * 1.15 and head["total"] - base["total"] > 0.5:
            flags.append(f"total time up {(head['total'] / base['total'] - 1) * 100:.0f}%")
        if head.get("embeds", 0) > base.get("embeds", 0) + 0.1:
            flags.append("more embeddings per frame (tracker scheduling changed?)")
        if head.get("faces", 0) < base.get("faces", 0) - 0.05:
            flags.append("fewer faces found per frame (detection recall dropped)")
    if not args.every_frame and head.get("embeds", 0) > 0.3:
        flags.append(f"embeddings/frame {head['embeds']:.2f} > 0.3 in tracked mode")
    if head.get("faces", 0) < n_faces - 0.05:
        flags.append(f"head finds {head.get('faces', 0):.2f} of {n_faces} faces per frame")
    print("\nREGRESSION: " + "; ".join(flags) if flags else "\nNo regression detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
