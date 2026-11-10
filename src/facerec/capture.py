"""Camera / video capture on a background thread that always exposes the newest frame.

Reading in a separate thread means the slow parts of the app never stall the camera, and
dropping stale frames removes the multi-frame lag that OpenCV's internal buffer causes.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "any": cv2.CAP_ANY,
}


def parse_source(source: int | str) -> int | str:
    if isinstance(source, str) and source.strip().isdigit():
        return int(source)
    return source


def is_file_source(source: int | str) -> bool:
    """True for a local video file; False for live sources (webcam index or stream URL).

    Live sources are read as fast as they deliver and tolerate transient read failures;
    files are paced at their own FPS and end on the first failed read.
    """
    source = parse_source(source)
    return not isinstance(source, int) and "://" not in source


def open_capture(
    source: int | str, width: int = 0, height: int = 0, fps: int = 0, backend: str = "auto"
) -> cv2.VideoCapture:
    source = parse_source(source)
    if isinstance(source, int):
        if backend == "auto":
            # MSMF (Windows default) can take seconds to open and adds latency; DirectShow doesn't.
            api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        else:
            api = _BACKENDS[backend]
        cap = cv2.VideoCapture(source, api)
        if cap.isOpened():
            # MJPG lets USB webcams deliver 720p/1080p at full frame rate.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps:
                cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        if "://" not in source and not Path(source).exists():
            raise FileNotFoundError(f"Video source not found: {source}")
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source {source!r}. "
            "Check the camera index (try 0 or 1) or the file path."
        )
    return cap


class ThreadedCamera:
    """Grabs frames continuously; consumers ask for the newest frame they haven't seen yet."""

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend: str = "auto",
        loop_video: bool = False,
    ) -> None:
        self.source = parse_source(source)
        self.is_file = is_file_source(self.source)  # stream URLs are live, like webcams
        self.loop_video = loop_video
        self._cap = open_capture(self.source, width, height, fps, backend)
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._cond = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None
        file_fps = self._cap.get(cv2.CAP_PROP_FPS) if self.is_file else 0
        self._frame_interval = 1.0 / file_fps if file_fps and file_fps > 0 else 0.0

    @property
    def resolution(self) -> tuple[int, int]:
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> "ThreadedCamera":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        failures = 0
        next_time = time.perf_counter()
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                if self.is_file and self.loop_video:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                failures += 1
                if self.is_file or failures > 30:
                    log.info("Video source ended or stopped responding.")
                    break
                time.sleep(0.01)
                continue
            failures = 0
            with self._cond:
                self._frame = frame
                self._frame_id += 1
                self._cond.notify_all()
            if self._frame_interval:  # play files at their native speed
                next_time += self._frame_interval
                delay = next_time - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_time = time.perf_counter()
        with self._cond:
            self._running = False
            self._cond.notify_all()

    def read(self, after_id: int = 0, timeout: float = 1.0) -> tuple[int, np.ndarray | None]:
        """Block until a frame newer than ``after_id`` exists; returns (frame_id, frame).

        Returns ``(after_id, None)`` on timeout or when the source has ended.
        The returned array is shared: copy it before drawing on it.
        """
        with self._cond:
            self._cond.wait_for(
                lambda: self._frame_id > after_id or not self._running, timeout=timeout
            )
            if self._frame_id > after_id and self._frame is not None:
                return self._frame_id, self._frame
            return after_id, None

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cap.release()

    def __enter__(self) -> "ThreadedCamera":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
