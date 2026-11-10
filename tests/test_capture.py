import time

import numpy as np
import pytest

import facerec.capture as capture
from facerec.capture import is_file_source, parse_source


class FakeCap:
    """A capture whose read() fails once (``fail_at``), then works again."""

    def __init__(self, fail_at=5, fps=25.0):
        self.reads, self.fail_at, self.fps = 0, fail_at, fps

    def isOpened(self):
        return True

    def get(self, prop):
        return self.fps

    def set(self, *args):
        return True

    def release(self):
        pass

    def read(self):
        self.reads += 1
        if self.reads == self.fail_at:
            return False, None
        time.sleep(0.002)
        return True, np.zeros((4, 4, 3), np.uint8)


@pytest.mark.parametrize(
    "source, expected",
    [(0, False), ("1", False), ("rtsp://cam/stream", False), ("http://cam/mjpg", False),
     ("clip.mp4", True), ("D:/videos/a.avi", True)],
)
def test_is_file_source(source, expected):
    assert is_file_source(source) is expected


def test_parse_source_turns_digit_strings_into_webcam_indexes():
    assert parse_source(" 2 ") == 2 and parse_source("clip.mp4") == "clip.mp4"


def _camera(monkeypatch, source, cap):
    monkeypatch.setattr(capture, "open_capture", lambda *a, **k: cap)
    return capture.ThreadedCamera(source)


def test_stream_survives_a_dropped_read_and_is_not_paced(monkeypatch):
    cap = FakeCap(fail_at=5, fps=25.0)
    cam = _camera(monkeypatch, "rtsp://192.168.1.10/stream", cap)
    assert cam._frame_interval == 0.0  # live: read as fast as the stream delivers
    cam.start()
    try:
        deadline = time.perf_counter() + 2
        while cap.reads < 20 and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert cam.running and cap.reads >= 20
    finally:
        cam.stop()


def test_file_ends_on_failed_read_and_is_paced(monkeypatch):
    cap = FakeCap(fail_at=3, fps=100.0)
    cam = _camera(monkeypatch, "clip.mp4", cap)
    assert cam._frame_interval == pytest.approx(0.01)
    cam.start()
    deadline = time.perf_counter() + 2
    while cam.running and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert not cam.running and cap.reads == 3
    cam.stop()
