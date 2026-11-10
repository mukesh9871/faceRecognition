"""Plain data types shared between modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """A detected face in full-frame pixel coordinates.

    ``raw`` is the 15-value YuNet-style row ``[x, y, w, h, 5 x (lx, ly), score]`` that
    landmark-based aligners (e.g. SFace ``alignCrop``) consume.
    """

    bbox: np.ndarray  # (4,) float32: x, y, w, h
    landmarks: np.ndarray  # (5, 2) float32: right eye, left eye, nose, right mouth, left mouth
    score: float
    raw: np.ndarray  # (15,) float32

    @classmethod
    def from_yunet_row(cls, row: np.ndarray) -> "Detection":
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        return cls(
            bbox=row[0:4].copy(),
            landmarks=row[4:14].reshape(5, 2).copy(),
            score=float(row[14]),
            raw=row[:15].copy(),
        )

    @property
    def area(self) -> float:
        return float(self.bbox[2] * self.bbox[3])

    @property
    def min_side(self) -> float:
        return float(min(self.bbox[2], self.bbox[3]))


@dataclass(frozen=True)
class MatchResult:
    """Best gallery match for one embedding. ``name`` is None when below threshold."""

    name: str | None
    score: float  # cosine similarity in [-1, 1]; higher is more similar


@dataclass(frozen=True)
class TrackView:
    """Immutable snapshot of a track for drawing / reporting (safe to hand across threads)."""

    track_id: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    name: str | None
    score: float
    status: str  # "pending" | "known" | "unknown"
