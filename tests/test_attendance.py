import csv
from datetime import datetime

from facerec.attendance import AttendanceLogger, AttendanceStore, export_csv


class FakeClock:
    def __init__(self, start: datetime):
        self.t = start.timestamp()

    def __call__(self):
        return self.t

    def advance(self, seconds: float):
        self.t += seconds


def test_cooldown_limits_writes_and_tracks_first_last_seen(tmp_path):
    db = tmp_path / "att.db"
    clock = FakeClock(datetime(2026, 11, 8, 9, 0, 0))
    logger = AttendanceLogger(db, cooldown_s=10, clock=clock)
    logger.arrived("alice", 0.8, track_id=1)
    for _ in range(8):  # 8 updates over 2 s -> no writes (cooldown)
        clock.advance(0.25)
        logger.seen("alice", 0.85)
    clock.advance(10)
    logger.seen("alice", 0.9)  # 09:00:12 -> written
    logger.close()

    store = AttendanceStore(db)
    rows = store.rows("2026-11-08")
    (events,) = store._conn.execute("SELECT COUNT(*) FROM events").fetchone()
    store.close()
    assert len(rows) == 1
    r = rows[0]
    assert (r.person, r.first_seen, r.last_seen) == ("alice", "2026-11-08 09:00:00", "2026-11-08 09:00:12")
    assert r.best_score == 0.9
    assert events == 1


def test_day_rollover_creates_new_row(tmp_path):
    db = tmp_path / "att.db"
    clock = FakeClock(datetime(2026, 11, 8, 23, 59, 50))
    logger = AttendanceLogger(db, cooldown_s=5, clock=clock)
    logger.arrived("bob", 0.7)
    clock.advance(20)  # now 2026-11-09 00:00:10
    logger.seen("bob", 0.7)
    logger.close()

    store = AttendanceStore(db)
    days = [r.day for r in store.rows()]
    store.close()
    assert days == ["2026-11-08", "2026-11-09"]


def test_export_csv(tmp_path):
    db = tmp_path / "att.db"
    store = AttendanceStore(db)
    ts = datetime(2026, 11, 8, 8, 30).timestamp()
    store.mark_seen("carol", ts, 0.6)
    store.mark_seen("dave", ts + 60, 0.7)
    store.commit()
    out = tmp_path / "out.csv"
    assert export_csv(store.rows("2026-11-08"), out) == 2
    store.close()
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["person"] for r in rows] == ["carol", "dave"]


def test_writer_survives_a_failed_commit(tmp_path, monkeypatch):
    import sqlite3
    import threading

    real_commit = AttendanceStore.commit
    calls = []
    first_commit_done = threading.Event()

    def flaky_commit(self):
        calls.append(1)
        try:
            if len(calls) == 1:
                raise sqlite3.OperationalError("database is locked")
            real_commit(self)
        finally:
            first_commit_done.set()

    monkeypatch.setattr(AttendanceStore, "commit", flaky_commit)
    db = tmp_path / "att.db"
    clock = FakeClock(datetime(2026, 11, 8, 9, 0, 0))
    logger = AttendanceLogger(db, cooldown_s=10, clock=clock)
    logger.arrived("alice", 0.8, track_id=1)  # this batch's commit fails
    assert first_commit_done.wait(5)
    logger.arrived("bob", 0.9, track_id=2)  # must still be written
    logger.close()
    monkeypatch.setattr(AttendanceStore, "commit", real_commit)

    store = AttendanceStore(db)
    people = [r.person for r in store.rows("2026-11-08")]
    store.close()
    assert people == ["bob"]
