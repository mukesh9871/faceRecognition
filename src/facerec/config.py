"""Typed configuration loaded from YAML, with relative paths resolved against the YAML file."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CameraConfig:
    source: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "auto"


@dataclass
class DetectorConfig:
    name: str = "yunet"
    model: str = "models/face_detection_yunet_2023mar.onnx"
    score_threshold: float = 0.8
    nms_threshold: float = 0.3
    top_k: int = 50
    max_side: int = 640


@dataclass
class RecognizerConfig:
    name: str = "sface"
    model: str = "models/face_recognition_sface_2021dec.onnx"
    threshold: float = 0.363


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.3
    max_misses: int = 15
    min_face_size: int = 40
    vote_frames: int = 5
    confirm_votes: int = 3
    reidentify_interval: int = 30
    unknown_retry_interval: int = 10


@dataclass
class AttendanceConfig:
    db_path: str = "data/attendance.db"
    cooldown_s: float = 10.0


@dataclass
class PathsConfig:
    faces_dir: str = "data/faces"
    gallery_cache: str = "data/gallery.npz"
    models_dir: str = "models"


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    recognizer: RecognizerConfig = field(default_factory=RecognizerConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    attendance: AttendanceConfig = field(default_factory=AttendanceConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    num_threads: int = 0
    base_dir: Path = field(default_factory=Path.cwd)

    def resolve(self, path: str | Path) -> Path:
        """Resolve a configured path relative to the config file's directory."""
        p = Path(path)
        return p if p.is_absolute() else (self.base_dir / p)


def _build(cls: type, values: dict[str, Any], section: str) -> Any:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown key(s) in config section '{section}': {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path | None = None) -> Config:
    """Load config from ``path``; fall back to ./config.yaml, then to built-in defaults."""
    if path is None:
        default = Path.cwd() / "config.yaml"
        if not default.exists():
            return Config()
        path = default
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    section_classes = {
        "camera": CameraConfig,
        "detector": DetectorConfig,
        "recognizer": RecognizerConfig,
        "tracker": TrackerConfig,
        "attendance": AttendanceConfig,
        "paths": PathsConfig,
    }
    kwargs: dict[str, Any] = {"base_dir": path.parent}
    for key, value in raw.items():
        if key in section_classes:
            kwargs[key] = _build(section_classes[key], value or {}, key)
        elif key == "num_threads":
            kwargs[key] = int(value)
        else:
            raise ValueError(f"Unknown top-level config key: '{key}'")
    return Config(**kwargs)
