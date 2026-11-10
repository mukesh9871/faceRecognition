"""High-level use cases wired from config: live view, benchmark, enrolment."""

from __future__ import annotations

import logging
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2

from facerec.attendance import AttendanceLogger, AttendanceStore
from facerec.capture import ThreadedCamera, is_file_source, open_capture, parse_source
from facerec.config import Config
from facerec.gallery import EnrollStats, Gallery, build_gallery
from facerec.overlay import draw_hud, draw_tracks
from facerec.pipeline import STAGES, AsyncPipeline, RecognitionEngine
from facerec.registry import backend_id, create_detector, create_recognizer

log = logging.getLogger(__name__)

QUIT_KEYS = {27, ord("q"), ord("Q")}


def load_gallery(cfg: Config, detector=None, recognizer=None) -> tuple[Gallery, EnrollStats]:
    detector = detector or create_detector(cfg)
    recognizer = recognizer or create_recognizer(cfg)
    return build_gallery(
        cfg.resolve(cfg.paths.faces_dir),
        detector,
        recognizer,
        cfg.resolve(cfg.paths.gallery_cache),
        backend_id(cfg),
    )


def print_enroll_stats(gallery: Gallery, stats: EnrollStats) -> None:
    print(
        f"Gallery: {gallery.num_people} people, {len(gallery)} images "
        f"({stats.reused} from cache, {stats.embedded} newly embedded, "
        f"{len(stats.skipped)} skipped)"
    )
    for path, reason in stats.skipped:
        print(f"  skipped {path}: {reason}")


def build_engine(
    cfg: Config, attendance: AttendanceLogger | None = None, always_embed: bool = False
) -> tuple[RecognitionEngine, Gallery, EnrollStats]:
    if cfg.num_threads:
        cv2.setNumThreads(cfg.num_threads)
    detector = create_detector(cfg)
    recognizer = create_recognizer(cfg)
    gallery, stats = load_gallery(cfg, detector, recognizer)
    engine = RecognitionEngine(detector, recognizer, gallery, cfg, attendance, always_embed)
    return engine, gallery, stats


# --------------------------------------------------------------------------- live


def run_live(cfg: Config, record_attendance: bool = True) -> int:
    attendance = (
        AttendanceLogger(cfg.resolve(cfg.attendance.db_path), cfg.attendance.cooldown_s)
        if record_attendance
        else None
    )
    engine, gallery, stats = build_engine(cfg, attendance)
    print_enroll_stats(gallery, stats)
    if len(gallery) == 0:
        print("Warning: no known faces enrolled - everyone will show as Unknown.")

    cam_cfg = cfg.camera
    camera = ThreadedCamera(
        cam_cfg.source, cam_cfg.width, cam_cfg.height, cam_cfg.fps, cam_cfg.backend
    )
    w, h = camera.resolution
    print(f"Video source {camera.source!r} opened at {w}x{h}. Press Esc or q to quit.")
    camera.start()
    pipeline = AsyncPipeline(camera, engine).start()

    window = "Face Recognition"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, w, h)
    show_hud = True
    last_id = 0
    display_fps = 0.0
    last_shown = time.perf_counter()
    try:
        while True:
            if pipeline.error is not None:
                raise pipeline.error
            frame_id, frame = camera.read(last_id, timeout=0.5)
            if frame is None:
                if not camera.running:
                    break
                if (cv2.waitKey(1) & 0xFF) in QUIT_KEYS:
                    break
                continue
            last_id = frame_id

            canvas = frame.copy()  # the camera's array is shared with the inference thread
            result, infer_fps, timings = pipeline.snapshot()
            if result is not None:
                draw_tracks(canvas, result.tracks)
            if show_hud:
                draw_hud(canvas, _hud_lines(display_fps, infer_fps, timings, result, engine))
            cv2.imshow(window, canvas)

            now = time.perf_counter()
            inst = 1.0 / max(now - last_shown, 1e-6)
            display_fps = inst if display_fps == 0 else 0.9 * display_fps + 0.1 * inst
            last_shown = now

            key = cv2.waitKey(1) & 0xFF
            if key in QUIT_KEYS:
                break
            if key in (ord("h"), ord("H")):
                show_hud = not show_hud
            elif key in (ord("r"), ord("R")):
                # Separate model instances: the inference thread is using the others.
                new_gallery, new_stats = load_gallery(cfg)
                engine.gallery = new_gallery
                print_enroll_stats(new_gallery, new_stats)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        pipeline.stop()
        camera.stop()
        cv2.destroyAllWindows()
        if attendance is not None:
            attendance.close()
            print_attendance(cfg, datetime.now().date().isoformat())
    return 0


def _hud_lines(display_fps, infer_fps, timings, result, engine: RecognitionEngine) -> list[str]:
    faces = len(result.tracks) if result else 0
    return [
        f"display {display_fps:4.1f} fps | inference {infer_fps:4.1f} fps",
        f"detect {timings.get('detect', 0):4.1f} ms  embed {timings.get('embed', 0):4.1f} ms"
        f"  match {timings.get('match', 0):4.2f} ms",
        f"faces {faces} | gallery {engine.gallery.num_people} people",
        "Esc/q quit   h HUD   r reload faces",
    ]


# --------------------------------------------------------------------------- benchmark


def run_benchmark(
    cfg: Config, frames: int = 300, always_embed: bool = False, warmup: int = 10
) -> dict[str, dict[str, float]]:
    """Process frames as fast as possible (no display) and report per-stage latency."""
    engine, gallery, stats = build_engine(cfg, None, always_embed)
    print_enroll_stats(gallery, stats)
    source = parse_source(cfg.camera.source)
    cap = open_capture(source, cfg.camera.width, cfg.camera.height, cfg.camera.fps, cfg.camera.backend)
    is_file = is_file_source(source)

    samples = {s: [] for s in STAGES}
    embeds = faces = processed = 0
    seen = failures = 0
    start = None
    shape = None  # of the last good frame; `frame` is None after a failed read
    try:
        while processed < frames:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if is_file and seen > 0 and failures == 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop short clips
                    continue
                if not is_file and failures <= 30:
                    time.sleep(0.01)  # webcams/streams drop the odd frame, e.g. at startup
                    continue
                log.warning("Source stopped delivering frames after %d frame(s).", seen)
                break
            failures = 0
            seen += 1
            shape = frame.shape
            result = engine.process(frame)
            if seen <= warmup:
                continue
            if start is None:
                start = time.perf_counter()
            processed += 1
            for s in STAGES:
                samples[s].append(result.timings_ms[s])
            embeds += result.embeddings
            faces += len(result.tracks)
    finally:
        cap.release()
    if processed == 0:
        raise RuntimeError("No frames could be read from the source.")
    wall = time.perf_counter() - start

    mode = "embed every frame (old behaviour)" if always_embed else "tracked (embed on demand)"
    print(f"\nBenchmark: {processed} frames, {shape[1]}x{shape[0]}, mode = {mode}")
    print(f"{'stage':<8}{'mean ms':>10}{'p50 ms':>10}{'p95 ms':>10}")
    report: dict[str, dict[str, float]] = {}
    for s in STAGES:
        vals = sorted(samples[s])
        row = {
            "mean": statistics.fmean(vals),
            "p50": vals[len(vals) // 2],
            "p95": vals[min(len(vals) - 1, int(len(vals) * 0.95))],
        }
        report[s] = row
        print(f"{s:<8}{row['mean']:>10.2f}{row['p50']:>10.2f}{row['p95']:>10.2f}")
    print(
        f"faces/frame {faces / processed:.2f} | embeddings/frame {embeds / processed:.2f}\n"
        f"pipeline capacity {1000 / report['total']['mean']:.1f} fps | "
        f"end-to-end incl. decode {processed / wall:.1f} fps"
    )
    report["summary"] = {
        "fps_pipeline": 1000 / report["total"]["mean"],
        "fps_end_to_end": processed / wall,
        "embeddings_per_frame": embeds / processed,
        "faces_per_frame": faces / processed,
    }
    return report


# --------------------------------------------------------------------------- enrolment

_NAME_RE = re.compile(r"^[\w][\w .\-]{0,63}$", re.UNICODE)


def validate_person_name(name: str) -> str:
    name = name.strip()
    if not _NAME_RE.match(name) or name in {".", ".."}:
        raise ValueError(
            f"Invalid name {name!r}: use letters, digits, spaces, '.', '-' or '_' (max 64)."
        )
    return name


def enroll_from_camera(cfg: Config, name: str, count: int = 5, interval_s: float = 0.8) -> int:
    """Auto-capture ``count`` photos of a single face from the camera into faces_dir/<name>/."""
    name = validate_person_name(name)
    out_dir = cfg.resolve(cfg.paths.faces_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = create_detector(cfg)
    cam_cfg = cfg.camera
    camera = ThreadedCamera(
        cam_cfg.source, cam_cfg.width, cam_cfg.height, cam_cfg.fps, cam_cfg.backend
    ).start()
    window = f"Enroll {name}"
    saved, last_id, last_save = 0, 0, 0.0
    min_side = cfg.tracker.min_face_size * 2
    try:
        while saved < count:
            frame_id, frame = camera.read(last_id, timeout=1.0)
            if frame is None:
                if not camera.running:
                    break
                continue
            last_id = frame_id
            faces = detector.detect(frame)
            canvas = frame.copy()
            if len(faces) == 1 and faces[0].min_side >= min_side:
                hint = "Hold still, turn your head slightly between shots"
                color = (60, 200, 60)
                if time.perf_counter() - last_save >= interval_s:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = out_dir / f"{name}_{stamp}_{saved}.jpg"
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if ok:
                        buf.tofile(str(path))  # tofile handles non-ASCII paths on Windows
                        saved += 1
                        last_save = time.perf_counter()
            else:
                hint = "Exactly one face, close to the camera" if faces else "No face found"
                color = (0, 200, 230)
            for f in faces:
                x, y, w, h = (int(v) for v in f.bbox)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
            draw_hud(canvas, [f"Enrolling {name}: {saved}/{count}", hint, "Esc to cancel"])
            cv2.imshow(window, canvas)
            if (cv2.waitKey(1) & 0xFF) in QUIT_KEYS:
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    print(f"Saved {saved} image(s) to {out_dir}")
    return saved


# --------------------------------------------------------------------------- reports


def print_attendance(cfg: Config, day: str | None) -> None:
    db = cfg.resolve(cfg.attendance.db_path)
    if not Path(db).exists():
        print("No attendance recorded yet.")
        return
    store = AttendanceStore(db)
    try:
        rows = store.rows(day)
    finally:
        store.close()
    title = f"Attendance for {day}" if day else "Attendance (all days)"
    print(f"\n{title}")
    if not rows:
        print("  (none)")
        return
    width = max(6, max(len(r.person) for r in rows))
    print(f"  {'Person':<{width}}  {'Day':<10}  {'First seen':<10}  {'Last seen':<10}  Score")
    for r in rows:
        print(
            f"  {r.person:<{width}}  {r.day:<10}  {r.first_seen[11:]:<10}  "
            f"{r.last_seen[11:]:<10}  {r.best_score:.2f}"
        )
