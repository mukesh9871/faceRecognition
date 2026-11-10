"""OpenCV SFace recognizer (cv2.FaceRecognizerSF): landmark alignment + 128-D embedding."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facerec.structures import Detection


class SFaceRecognizer:
    name = "sface"
    dim = 128

    def __init__(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"SFace model not found at {model_path}. Run `facerec download-models`."
            )
        self._net = cv2.FaceRecognizerSF.create(str(model_path), "")

    def embed(self, frame: np.ndarray, detection: Detection) -> np.ndarray:
        # Crop from the full-resolution frame, aligned on the 5 landmarks, to 112x112.
        aligned = self._net.alignCrop(frame, detection.raw)
        feature = self._net.feature(aligned).reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(feature))
        return feature / norm if norm > 0 else feature
