from __future__ import annotations

from typing import Protocol

import numpy as np

from facerec.structures import Detection


class Detector(Protocol):
    """Finds faces in a BGR frame and returns them in full-frame coordinates."""

    name: str

    def detect(self, frame: np.ndarray) -> list[Detection]: ...
