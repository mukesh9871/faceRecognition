#!/usr/bin/env python3
"""Heuristic scan of a facerec diff for repo-specific review leads.

Usage (from the repo root):
    python .claude/skills/facerec-review/scripts/review_scan.py --base main
    python .claude/skills/facerec-review/scripts/review_scan.py --base facerec/01-foundation --worktree

Only *added* lines are scanned. Every hit is a lead to verify by reading the code, not a
finding. The exit code is always 0 unless git itself fails.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PKG = "src/facerec/"


def in_pkg(path: str, *exclude: str) -> bool:
    return path.startswith(PKG) and PurePosixPath(path).name not in exclude


@dataclass(frozen=True)
class Rule:
    rule_id: str
    pattern: re.Pattern
    applies: callable
    why: str


RULES = [
    Rule("gui-off-main",
         re.compile(r"cv2\.(imshow|waitKey|namedWindow|destroyAllWindows|resizeWindow)\b"),
         lambda p: in_pkg(p, "app.py"),
         "OpenCV GUI call outside app.py: GUI calls must stay on the main thread."),
    Rule("embed-outside-scheduler",
         re.compile(r"\.embed\("),
         lambda p: in_pkg(p, "pipeline.py", "gallery.py", "sface.py", "base.py"),
         "recognizer.embed outside pipeline/gallery: bypasses tracker scheduling (cost per frame)."),
    Rule("sleep-on-hot-path",
         re.compile(r"\btime\.sleep\("),
         lambda p: in_pkg(p) and not p.endswith(("capture.py", "app.py")),
         "sleep in library code: blocks the inference / writer thread."),
    Rule("sqlite-outside-attendance",
         re.compile(r"sqlite3\.connect\("),
         lambda p: in_pkg(p, "attendance.py"),
         "sqlite connection outside attendance.py: bypasses the queued single writer."),
    Rule("sql-string-format",
         re.compile(r"""\.execute(many)?\(\s*(f["']|["'][^"']*["']\s*%|["'][^"']*["']\s*\.format)"""),
         lambda p: p.endswith(".py"),
         "SQL built with string formatting: use ? parameters."),
    Rule("unsafe-pickle",
         re.compile(r"allow_pickle\s*=\s*True|pickle\.loads?\("),
         lambda p: p.endswith(".py"),
         "pickle loading: the gallery cache must load with allow_pickle=False."),
    Rule("imread-unicode",
         re.compile(r"cv2\.imread\("),
         lambda p: in_pkg(p),
         "cv2.imread fails on non-ASCII Windows paths: use gallery.read_image."),
    Rule("videocapture-outside-capture",
         re.compile(r"cv2\.VideoCapture\("),
         lambda p: in_pkg(p, "capture.py"),
         "VideoCapture outside capture.py: skips DirectShow/MJPG/buffer setup and error handling."),
    Rule("rgb-conversion",
         re.compile(r"COLOR_BGR2RGB"),
         lambda p: in_pkg(p),
         "BGR->RGB conversion: YuNet/SFace expect BGR (RGB was a dlib requirement)."),
    Rule("hardcoded-scale",
         re.compile(r"\*\s*4\b|/\s*4\b|\b0\.25\b"),
         lambda p: in_pkg(p, "overlay.py"),
         "Hard-coded scale factor: derive it from the actual resize (old '* 4' bug)."),
    Rule("hardcoded-tunable",
         re.compile(r"\b(threshold|max_side|cooldown_s|interval|min_face_size)\s*=\s*\d"),
         lambda p: in_pkg(p, "config.py"),
         "Literal tunable in library code: should come from Config (check it's a default, not an override)."),
    Rule("print-in-library",
         re.compile(r"^\s*print\("),
         lambda p: in_pkg(p, "cli.py", "app.py", "model_zoo.py"),
         "print() in library module: use logging (and never in per-frame code)."),
    Rule("swallowed-exception",
         re.compile(r"except\s*(Exception|BaseException)?\s*:\s*(pass)?\s*$"),
         lambda p: p.endswith(".py"),
         "Broad except: in worker threads this hides crashes. Make sure the error is surfaced or logged."),
    Rule("gpu-provider-no-fallback",
         re.compile(r"CUDAExecutionProvider"),
         lambda p: in_pkg(p),
         "CUDA provider: make sure there is a CPU fallback and the chosen provider is logged."),
    Rule("network-call",
         re.compile(r"\b(urllib\.request|requests\.|http\.client|socket\.)"),
         lambda p: in_pkg(p, "model_zoo.py"),
         "Network access outside model_zoo: privacy (no telemetry or uploads)."),
]

ARTEFACT_SUFFIXES = {".onnx", ".db", ".npz", ".pt", ".pth", ".pkl", ".csv", ".avi", ".mp4"}
FACE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LARGE_BYTES = 5 * 1024 * 1024
HOT_PATH = ("detectors/", "recognizers/", "tracker.py", "pipeline.py", "capture.py",
            "gallery.py", "registry.py")


def git(*args: str) -> str:
    res = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    if res.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{res.stderr.strip()}")
    return res.stdout


def added_lines(diff_args: list[str]) -> dict[str, list[tuple[int, str]]]:
    out = git("diff", "-U0", "--no-color", "-M", *diff_args)
    files: dict[str, list[tuple[int, str]]] = {}
    path, line_no = None, 0
    for raw in out.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:]
            path = None if target == "/dev/null" else target[2:] if target.startswith("b/") else target
            if path:
                files.setdefault(path, [])
        elif raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            line_no = int(m.group(1)) if m else 0
        elif raw.startswith("+") and path:
            files[path].append((line_no, raw[1:]))
            line_no += 1
    return files


def config_keys(root: Path) -> list[tuple[str, str, int]]:
    """(section, key, line) for every config field, read from the dataclasses in config.py.

    Only fields of @dataclass classes count (not locals such as ``kwargs`` in load_config).
    Section fields come from the class that the top-level ``Config`` maps to that section,
    e.g. ``detector: DetectorConfig`` -> ("detector", "max_side", n). Top-level scalars
    such as ``num_threads`` have section "". ``base_dir`` is internal and skipped.
    """
    path = root / "src" / "facerec" / "config.py"
    if not path.exists():
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    def is_dataclass(node: ast.ClassDef) -> bool:
        for d in node.decorator_list:
            target = d.func if isinstance(d, ast.Call) else d
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
            if name == "dataclass":
                return True
        return False

    classes = {
        n.name: [(s.target.id, s.annotation, s.lineno) for s in n.body
                 if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
        for n in tree.body if isinstance(n, ast.ClassDef) and is_dataclass(n)
    }
    keys: list[tuple[str, str, int]] = []
    for field_name, annotation, lineno in classes.get("Config", []):
        section_cls = annotation.id if isinstance(annotation, ast.Name) else None
        if section_cls in classes:
            keys += [(field_name, k, no) for k, _, no in classes[section_cls]]
        elif field_name != "base_dir":
            keys.append(("", field_name, lineno))
    return keys


def load_yaml(path: Path) -> dict | None:
    """config.yaml as a dict; None if it can't be read (then the yaml check is skipped)."""
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # missing file, PyYAML not installed, invalid YAML
        print(f"  (cannot read {path.name}: {exc}; skipping the config.yaml check)")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="main", help="base branch/commit (default: main)")
    ap.add_argument("--worktree", action="store_true",
                    help="include uncommitted and untracked changes")
    args = ap.parse_args()

    root = Path(git("rev-parse", "--show-toplevel").strip())
    merge_base = git("merge-base", args.base, "HEAD").strip()
    diff_args = [merge_base] if args.worktree else [f"{merge_base}...HEAD"]

    status = git("diff", "--name-status", "-M", *diff_args).splitlines()
    changes: list[tuple[str, str]] = []  # (status letter, path)
    for line in status:
        parts = line.split("\t")
        changes.append((parts[0][0], parts[-1]))
    files = added_lines(diff_args)
    if args.worktree:
        for p in git("ls-files", "--others", "--exclude-standard").splitlines():
            changes.append(("?", p))
            try:
                text = (root / p).read_text(encoding="utf-8")
                files[p] = list(enumerate(text.splitlines(), start=1))
            except (UnicodeDecodeError, OSError):
                files.setdefault(p, [])

    head = "working tree" if args.worktree else "HEAD"
    print(f"facerec review scan: {args.base} ({merge_base[:7]}) ... {head}")
    print(f"{len(changes)} changed file(s)")
    for st, p in changes:
        print(f"  {st} {p}")

    # 1. Pattern leads on added lines
    print("\n== Pattern leads (verify each in context)")
    hits = 0
    for path, lines in sorted(files.items()):
        if path.startswith(".claude/") or not path.endswith(".py"):
            continue
        for no, text in lines:
            if text.lstrip().startswith("#"):
                continue
            for rule in RULES:
                if rule.applies(path) and rule.pattern.search(text):
                    hits += 1
                    print(f"  [{rule.rule_id}] {path}:{no}: {text.strip()[:110]}")
                    print(f"      -> {rule.why}")
    if not hits:
        print("  none")

    # 2. Artefacts / personal data
    print("\n== Committed artefacts and personal data")
    art = 0
    for st, p in changes:
        if st == "D":
            continue
        suffix = PurePosixPath(p).suffix.lower()
        full = root / p
        size = full.stat().st_size if full.exists() else 0
        if suffix in ARTEFACT_SUFFIXES:
            art += 1
            print(f"  {p}: generated/model/data file ({suffix}). Should it be git-ignored?")
        elif p.startswith("data/faces/") and suffix in FACE_IMAGE_SUFFIXES and st != "R":
            art += 1
            print(f"  {p}: new face photo (personal data). Confirm the user intends to commit it.")
        if size > LARGE_BYTES:
            art += 1
            print(f"  {p}: {size / 1e6:.1f} MB, large file")
    if not art:
        print("  none")

    # 3. Config wiring
    print("\n== Config wiring")
    cfg_issues = 0
    added_cfg_lines = {no for no, _ in files.get("src/facerec/config.py", [])}
    new_keys = [k for k in config_keys(root) if k[2] in added_cfg_lines]
    yaml_cfg = load_yaml(root / "config.yaml")
    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").exists() else ""
    for section, key, no in new_keys:
        dotted = f"{section}.{key}" if section else key
        if yaml_cfg is None:
            in_yaml = True  # unknown: already reported that config.yaml couldn't be read
        elif section:
            in_yaml = key in (yaml_cfg.get(section) or {})
        else:
            in_yaml = key in yaml_cfg
        if not in_yaml:
            cfg_issues += 1
            print(f"  config.py:{no}: new key '{dotted}' missing from config.yaml")
        elif not re.search(rf"\b{re.escape(key)}\b", readme):
            print(f"  config.py:{no}: new key '{dotted}' not mentioned in README (fine if not user-facing)")
    if not cfg_issues and not new_keys:
        print("  no new config keys")
    elif not cfg_issues:
        print("  new keys present in config.yaml")

    # 4. Tests and benchmark hints
    print("\n== Tests and benchmark")
    src_changed = [p for st, p in changes if p.startswith(PKG) and p.endswith(".py") and st != "D"
                   and not p.endswith("__init__.py")]
    tests_changed = [p for _, p in changes if p.startswith("tests/")]
    if src_changed and not tests_changed:
        print("  source changed but no tests changed:")
        for p in src_changed:
            print(f"    {p}")
    elif src_changed:
        print(f"  {len(src_changed)} source file(s), {len(tests_changed)} test file(s) changed")
    hot = [p for p in src_changed if any(h in p for h in HOT_PATH)]
    if hot:
        print("  hot-path files changed, so run bench_compare.py:")
        for p in hot:
            print(f"    {p}")
    if not src_changed:
        print("  no package source changes")

    # 5. Commit messages
    print("\n== Commit messages")
    log = git("log", "--format=%h%x1f%s%x1f%B%x1e", f"{merge_base}..HEAD")
    commit_issues = 0
    for entry in filter(None, (e.strip() for e in log.split("\x1e"))):
        sha, subject, body = (entry.split("\x1f") + ["", ""])[:3]
        if re.search(r"co-authored-by:.*(claude|anthropic)|generated with \[?claude", body, re.I):
            commit_issues += 1
            print(f"  {sha}: AI attribution line present (user does not want it)")
        if len(subject) > 72:
            commit_issues += 1
            print(f"  {sha}: subject longer than 72 chars")
        if re.match(r"(?i)^(wip|fixup!|squash!|tmp)\b", subject):
            commit_issues += 1
            print(f"  {sha}: '{subject}': WIP/fixup commit")
    if not commit_issues:
        print("  ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
