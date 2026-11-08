"""Drawing helpers: face boxes with labels and a performance HUD."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from facerec.structures import TrackView

COLORS = {
    "known": (60, 200, 60),  # green
    "unknown": (40, 40, 230),  # red
    "pending": (0, 200, 230),  # amber: still collecting votes
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label_for(t: TrackView) -> str:
    if t.status == "pending":
        return "..."
    if t.status == "unknown":
        return "Unknown"
    return f"{t.name} {t.score:.2f}"


def draw_tracks(frame: np.ndarray, tracks: Iterable[TrackView]) -> None:
    scale = max(0.5, frame.shape[0] / 1080)
    thickness = max(1, int(round(2 * scale)))
    for t in tracks:
        x, y, w, h = t.bbox
        color = COLORS.get(t.status, COLORS["pending"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        text = label_for(t)
        (tw, th), base = cv2.getTextSize(text, FONT, 0.7 * scale, thickness)
        top = max(0, y - th - base - 6)
        cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + base + 6), color, cv2.FILLED)
        cv2.putText(
            frame, text, (x + 4, top + th + 3), FONT, 0.7 * scale, (255, 255, 255), thickness,
            cv2.LINE_AA,
        )


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    if not lines:
        return
    scale = max(0.45, frame.shape[0] / 1500)
    line_h = int(26 * scale) + 6
    width = max(cv2.getTextSize(s, FONT, scale, 1)[0][0] for s in lines) + 16
    height = line_h * len(lines) + 8
    roi = frame[0:height, 0:width]
    roi[:] = (roi * 0.4).astype(np.uint8)  # darken background for readability
    for i, s in enumerate(lines):
        cv2.putText(
            frame, s, (8, (i + 1) * line_h), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA
        )
