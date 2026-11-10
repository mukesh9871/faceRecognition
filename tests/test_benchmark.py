import numpy as np
import pytest

from facerec import app
from facerec.config import Config
from facerec.gallery import EnrollStats, Gallery
from facerec.pipeline import RecognitionEngine
from test_pipeline import CountingRecognizer, ScriptedDetector


class ScriptedCap:
    """read() results: True = a frame, False = a failed read; repeats ``tail`` forever."""

    def __init__(self, script, tail):
        self.script, self.tail, self.rewinds = list(script), tail, 0

    def read(self):
        ok = self.script.pop(0) if self.script else self.tail
        return (True, np.zeros((200, 400, 3), np.uint8)) if ok else (False, None)

    def set(self, prop, value):
        self.rewinds += 1
        return True

    def release(self):
        pass


@pytest.fixture
def bench(monkeypatch):
    def run(source, cap, frames=20):
        cfg = Config()
        cfg.camera.source = source
        engine = RecognitionEngine(
            ScriptedDetector(), CountingRecognizer(), Gallery(np.eye(2)[:1], ["alice"]), cfg
        )
        monkeypatch.setattr(app, "build_engine", lambda *a, **k: (engine, engine.gallery, EnrollStats()))
        monkeypatch.setattr(app, "open_capture", lambda *a, **k: cap)
        monkeypatch.setattr(app.time, "sleep", lambda s: None)
        return app.run_benchmark(cfg, frames=frames, warmup=2)

    return run


def test_webcam_dropped_frames_do_not_end_or_crash_the_benchmark(bench):
    cap = ScriptedCap([True] * 5 + [False] * 3, tail=True)  # a few failed reads mid-run
    report = bench(0, cap)
    assert report["summary"]["faces_per_frame"] == 1.0


def test_webcam_that_stops_reports_frames_measured_so_far(bench, capsys):
    cap = ScriptedCap([True] * 8, tail=False)  # unplugged after 8 frames
    report = bench(0, cap)
    assert "Benchmark: 6 frames, 400x200" in capsys.readouterr().out
    assert report["summary"]["faces_per_frame"] == 1.0


def test_short_clip_loops_but_unreadable_file_does_not_spin(bench):
    looping = ScriptedCap([True] * 5 + [False] + [True] * 30, tail=False)
    bench("clip.avi", looping)
    assert looping.rewinds == 1
    broken = ScriptedCap([True] * 5, tail=False)  # rewind doesn't help: must stop, not loop
    bench("clip.avi", broken)
    assert broken.rewinds == 1
