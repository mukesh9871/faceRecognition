from __future__ import annotations

from typing import Protocol

import numpy as np

from facerec.structures import Detection


class Recognizer(Protocol):
    """Turns a detected face into an L2-normalised embedding vector."""

    name: str
    dim: int

    def embed(self, frame: np.ndarray, detection: Detection) -> np.ndarray: ...
