"""OpenCV YuNet face detector (cv2.FaceDetectorYN): fast CNN detector with 5 landmarks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facerec.structures import Detection


class YuNetDetector:
    name = "yunet"

    def __init__(
        self,
        model_path: str | Path,
        score_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        max_side: int = 640,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found at {model_path}. Run `facerec download-models`."
            )
        self.max_side = max_side
        self._net = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, nms_threshold, top_k
        )
        self._input_size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        # YuNet is trained on small inputs: very large frames are slower AND miss big faces,
        # so detect on a downscaled copy and map the results back to full resolution.
        scale = min(1.0, self.max_side / max(h, w)) if self.max_side > 0 else 1.0
        if scale < 1.0:
            small = cv2.resize(
                frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
            )
        else:
            small = frame
        size = (small.shape[1], small.shape[0])
        if size != self._input_size:
            self._net.setInputSize(size)
            self._input_size = size

        _, faces = self._net.detect(small)
        if faces is None or len(faces) == 0:
            return []
        faces = faces.astype(np.float32, copy=True)
        if scale < 1.0:
            faces[:, :14] /= scale
        return [Detection.from_yunet_row(row) for row in faces]
