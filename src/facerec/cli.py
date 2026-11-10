"""Command line interface: `facerec run | enroll | list | export | download-models`."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from facerec.config import Config, load_config


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if getattr(args, "camera", None) is not None:
        cfg.camera.source = args.camera
    if getattr(args, "source", None) is not None:
        cfg.camera.source = args.source
    for attr, section, key in (
        ("width", cfg.camera, "width"),
        ("height", cfg.camera, "height"),
        ("threshold", cfg.recognizer, "threshold"),
        ("max_side", cfg.detector, "max_side"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(section, key, value)
    return cfg


def _add_source_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--camera", type=int, help="webcam index (overrides config)")
    g.add_argument("--source", help="video file / stream URL / webcam index")
    p.add_argument("--width", type=int, help="requested camera width")
    p.add_argument("--height", type=int, help="requested camera height")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="facerec", description=__doc__)
    parser.add_argument("-c", "--config", help="path to config.yaml (default: ./config.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="live recognition window with attendance logging")
    _add_source_args(run)
    run.add_argument("--threshold", type=float, help="cosine similarity needed for a match")
    run.add_argument("--max-side", type=int, help="detection resolution (long side, px)")
    run.add_argument("--no-attendance", action="store_true", help="do not write attendance")
    run.add_argument(
        "--benchmark", type=int, metavar="N", help="process N frames headless and print timings"
    )
    run.add_argument(
        "--every-frame",
        action="store_true",
        help="benchmark only: embed every face on every frame (old behaviour) for comparison",
    )

    enroll = sub.add_parser("enroll", help="(re)build the known-face gallery")
    enroll.add_argument("--name", help="person to capture from the camera")
    enroll.add_argument("--from-camera", action="store_true", help="capture photos from camera")
    enroll.add_argument("--count", type=int, default=5, help="photos to capture (default 5)")
    _add_source_args(enroll)

    lst = sub.add_parser("list", help="show attendance")
    lst.add_argument("--day", help="YYYY-MM-DD (default: today)")
    lst.add_argument("--all", action="store_true", help="all days")

    exp = sub.add_parser("export", help="export attendance to CSV")
    exp.add_argument("--csv", required=True, help="output file")
    exp.add_argument("--day", help="YYYY-MM-DD (default: today)")
    exp.add_argument("--all", action="store_true", help="all days")

    dl = sub.add_parser("download-models", help="download the ONNX models")
    dl.add_argument("--force", action="store_true", help="re-download even if present")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        cfg = _apply_overrides(load_config(args.config), args)
        return _dispatch(cfg, args)
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        if args.verbose:
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _dispatch(cfg: Config, args: argparse.Namespace) -> int:
    # Imported lazily so `facerec --help` stays fast.
    from facerec import app

    if args.command == "run":
        if args.benchmark:
            app.run_benchmark(cfg, args.benchmark, always_embed=args.every_frame)
            return 0
        return app.run_live(cfg, record_attendance=not args.no_attendance)

    if args.command == "enroll":
        if args.from_camera:
            if not args.name:
                raise ValueError("--from-camera needs --name")
            app.enroll_from_camera(cfg, args.name, args.count)
        elif args.name:
            raise ValueError("--name is only used with --from-camera")
        gallery, stats = app.load_gallery(cfg)
        app.print_enroll_stats(gallery, stats)
        for person in gallery.names:
            print(f"  {person}: {int((gallery.labels == person).sum())} image(s)")
        return 0

    if args.command == "list":
        app.print_attendance(cfg, None if args.all else (args.day or date.today().isoformat()))
        return 0

    if args.command == "export":
        from facerec.attendance import AttendanceStore, export_csv

        db = cfg.resolve(cfg.attendance.db_path)
        if not db.exists():
            raise FileNotFoundError(f"No attendance database at {db}")
        store = AttendanceStore(db)
        try:
            rows = store.rows(None if args.all else (args.day or date.today().isoformat()))
        finally:
            store.close()
        n = export_csv(rows, args.csv)
        print(f"Wrote {n} row(s) to {args.csv}")
        return 0

    if args.command == "download-models":
        from facerec.model_zoo import download_models

        print(f"Models directory: {cfg.resolve(cfg.paths.models_dir)}")
        download_models(cfg.resolve(cfg.paths.models_dir), force=args.force)
        print("Done.")
        return 0

    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
