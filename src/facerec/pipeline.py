"""Recognition pipeline: detect -> track -> embed (only when needed) -> match -> attendance.

``RecognitionEngine`` is synchronous and deterministic (used by tests and the benchmark).
``AsyncPipeline`` runs the engine on a worker thread against a ``ThreadedCamera`` so the
display loop never waits for inference.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from facerec.attendance import AttendanceLogger
from facerec.capture import ThreadedCamera
from facerec.config import Config
from facerec.detectors.base import Detector
from facerec.gallery import Gallery
from facerec.recognizers.base import Recognizer
from facerec.structures import TrackView
from facerec.tracker import FaceTracker

log = logging.getLogger(__name__)

STAGES = ("detect", "track", "embed", "match", "total")


@dataclass(frozen=True)
class FrameResult:
    frame_id: int
    tracks: tuple[TrackView, ...]
    timings_ms: dict[str, float] = field(default_factory=dict)
    embeddings: int = 0


class RecognitionEngine:
    def __init__(
        self,
        detector: Detector,
        recognizer: Recognizer,
        gallery: Gallery,
        cfg: Config,
        attendance: AttendanceLogger | None = None,
        always_embed: bool = False,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.gallery = gallery  # may be swapped at runtime (attribute assignment is atomic)
        self.threshold = cfg.recognizer.threshold
        self.tracker = FaceTracker(cfg.tracker)
        self.attendance = attendance
        self.always_embed = always_embed  # benchmark mode emulating "recognise every frame"
        self._frame_idx = 0

    def process(self, frame: np.ndarray, frame_id: int | None = None) -> FrameResult:
        self._frame_idx += 1
        idx = self._frame_idx
        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        t1 = time.perf_counter()
        tracks = self.tracker.update(detections)
        t2 = time.perf_counter()

        todo = [
            t for t in tracks
            if (self.always_embed and t.detection.min_side >= self.tracker.cfg.min_face_size)
            or self.tracker.needs_embedding(t, idx)
        ]
        vectors = [self.recognizer.embed(frame, t.detection) for t in todo]
        t3 = time.perf_counter()
        if vectors:
            matches = self.gallery.match(np.stack(vectors), self.threshold)
            for track, match in zip(todo, matches):
                self.tracker.add_vote(track, match, idx)
        t4 = time.perf_counter()

        if self.attendance is not None:
            for t in tracks:
                if not (t.confirmed and t.name):
                    continue
                if t.announced != t.name:
                    self.attendance.arrived(t.name, t.score, t.track_id)
                    t.announced = t.name
                else:
                    self.attendance.seen(t.name, t.score)

        timings = {
            "detect": (t1 - t0) * 1000,
            "track": (t2 - t1) * 1000,
            "embed": (t3 - t2) * 1000,
            "match": (t4 - t3) * 1000,
            "total": (time.perf_counter() - t0) * 1000,
        }
        return FrameResult(
            frame_id if frame_id is not None else idx,
            tuple(t.view() for t in tracks),
            timings,
            len(vectors),
        )


class AsyncPipeline:
    """Runs the engine on the newest camera frame in a background thread."""

    def __init__(self, camera: ThreadedCamera, engine: RecognitionEngine) -> None:
        self.camera = camera
        self.engine = engine
        self._lock = threading.Lock()
        self._result: FrameResult | None = None
        self._fps = 0.0
        self._timings: dict[str, float] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def start(self) -> "AsyncPipeline":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="inference", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        last_id = 0
        last_t = time.perf_counter()
        try:
            while self._running:
                frame_id, frame = self.camera.read(last_id, timeout=0.5)
                if frame is None:
                    if not self.camera.running:
                        break
                    continue
                last_id = frame_id
                result = self.engine.process(frame, frame_id)
                now = time.perf_counter()
                inst = 1.0 / max(now - last_t, 1e-6)
                last_t = now
                with self._lock:
                    self._result = result
                    self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
                    for k, v in result.timings_ms.items():
                        old = self._timings.get(k)
                        self._timings[k] = v if old is None else 0.9 * old + 0.1 * v
        except BaseException as exc:  # report to the UI thread instead of dying silently
            log.exception("Inference thread crashed")
            self.error = exc
        finally:
            self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def snapshot(self) -> tuple[FrameResult | None, float, dict[str, float]]:
        """(latest result, smoothed inference FPS, smoothed per-stage ms)."""
        with self._lock:
            return self._result, self._fps, dict(self._timings)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
