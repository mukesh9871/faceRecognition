import numpy as np

from facerec.config import Config
from facerec.gallery import Gallery
from facerec.pipeline import RecognitionEngine
from facerec.structures import Detection


class ScriptedDetector:
    """Returns one 100x100 face at x = frame[0,0,0] (so tests control motion via pixels)."""

    name = "scripted"

    def detect(self, frame):
        row = np.zeros(15, np.float32)
        row[:4] = (float(frame[0, 0, 0]), 0, 100, 100)
        return [Detection.from_yunet_row(row)]


class CountingRecognizer:
    name = "counting"
    dim = 2

    def __init__(self):
        self.calls = 0

    def embed(self, frame, det):
        self.calls += 1
        return np.array([1.0, 0.0], np.float32)


class RecordingAttendance:
    def __init__(self):
        self.arrivals, self.seen_calls = [], 0

    def arrived(self, person, score, track_id=None):
        self.arrivals.append((person, track_id))

    def seen(self, person, score):
        self.seen_calls += 1


def frame_at(x):
    f = np.zeros((200, 400, 3), np.uint8)
    f[0, 0, 0] = x
    return f


def test_engine_embeds_only_until_confirmed_then_periodically():
    cfg = Config()
    cfg.tracker.vote_frames = 5
    cfg.tracker.reidentify_interval = 30
    rec = CountingRecognizer()
    att = RecordingAttendance()
    gallery = Gallery(np.array([[1.0, 0.0]], np.float32), ["alice"])
    engine = RecognitionEngine(ScriptedDetector(), rec, gallery, cfg, att)

    results = [engine.process(frame_at(i % 50)) for i in range(60)]
    # 5 votes to start, then one re-check at frame 35 (30 frames after the 5th vote).
    assert rec.calls == 6
    assert results[0].tracks[0].status == "pending"
    assert results[-1].tracks[0].status == "known"
    assert results[-1].tracks[0].name == "alice"
    assert att.arrivals == [("alice", 1)]  # one arrival per confirmed track
    assert att.seen_calls > 0
    assert set(results[-1].timings_ms) == {"detect", "track", "embed", "match", "total"}


def test_every_frame_mode_embeds_each_frame():
    rec = CountingRecognizer()
    engine = RecognitionEngine(
        ScriptedDetector(), rec, Gallery.empty(2), Config(), always_embed=True
    )
    for i in range(10):
        engine.process(frame_at(i))
    assert rec.calls == 10
