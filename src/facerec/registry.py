"""Backend factories. Add a backend by registering a builder here; nothing else changes."""

from __future__ import annotations

from typing import Callable

from facerec.config import Config
from facerec.detectors.base import Detector
from facerec.recognizers.base import Recognizer


def _yunet(cfg: Config) -> Detector:
    from facerec.detectors.yunet import YuNetDetector

    d = cfg.detector
    return YuNetDetector(
        cfg.resolve(d.model),
        score_threshold=d.score_threshold,
        nms_threshold=d.nms_threshold,
        top_k=d.top_k,
        max_side=d.max_side,
    )


def _sface(cfg: Config) -> Recognizer:
    from facerec.recognizers.sface import SFaceRecognizer

    return SFaceRecognizer(cfg.resolve(cfg.recognizer.model))


DETECTORS: dict[str, Callable[[Config], Detector]] = {"yunet": _yunet}
RECOGNIZERS: dict[str, Callable[[Config], Recognizer]] = {"sface": _sface}


def create_detector(cfg: Config) -> Detector:
    try:
        builder = DETECTORS[cfg.detector.name]
    except KeyError:
        raise ValueError(
            f"Unknown detector '{cfg.detector.name}'. Available: {sorted(DETECTORS)}"
        ) from None
    return builder(cfg)


def create_recognizer(cfg: Config) -> Recognizer:
    try:
        builder = RECOGNIZERS[cfg.recognizer.name]
    except KeyError:
        raise ValueError(
            f"Unknown recognizer '{cfg.recognizer.name}'. Available: {sorted(RECOGNIZERS)}"
        ) from None
    return builder(cfg)


def backend_id(cfg: Config) -> str:
    """Identifies the embedding space; cached gallery embeddings are only valid for the same id."""
    return f"{cfg.recognizer.name}:{cfg.resolve(cfg.recognizer.model).name}"
