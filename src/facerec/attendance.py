"""Attendance stored in SQLite: one row per person per day (first/last seen) plus an
arrival event log. Writes go through a queue to one writer thread so recognition never
waits on disk."""

from __future__ import annotations

import csv
import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS attendance (
    person      TEXT NOT NULL,
    day         TEXT NOT NULL,          -- YYYY-MM-DD, local time
    first_seen  TEXT NOT NULL,          -- ISO timestamp, local time
    last_seen   TEXT NOT NULL,
    best_score  REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (person, day)
);
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    person    TEXT NOT NULL,
    ts        TEXT NOT NULL,
    score     REAL NOT NULL,
    track_id  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_person_ts ON events(person, ts);
"""


@dataclass(frozen=True)
class AttendanceRow:
    person: str
    day: str
    first_seen: str
    last_seen: str
    best_score: float


def _iso(ts: float) -> tuple[str, str]:
    dt = datetime.fromtimestamp(ts)
    return dt.date().isoformat(), dt.isoformat(sep=" ", timespec="seconds")


class AttendanceStore:
    """Thin synchronous SQLite repository. One instance per thread."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def mark_seen(self, person: str, ts: float, score: float) -> None:
        day, stamp = _iso(ts)
        self._conn.execute(
            """
            INSERT INTO attendance (person, day, first_seen, last_seen, best_score)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(person, day) DO UPDATE SET
                last_seen = MAX(last_seen, excluded.last_seen),
                first_seen = MIN(first_seen, excluded.first_seen),
                best_score = MAX(best_score, excluded.best_score)
            """,
            (person, day, stamp, stamp, score),
        )

    def add_event(self, person: str, ts: float, score: float, track_id: int | None) -> None:
        _, stamp = _iso(ts)
        self._conn.execute(
            "INSERT INTO events (person, ts, score, track_id) VALUES (?, ?, ?, ?)",
            (person, stamp, score, track_id),
        )

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def rows(self, day: str | None = None) -> list[AttendanceRow]:
        sql = "SELECT person, day, first_seen, last_seen, best_score FROM attendance"
        args: tuple = ()
        if day:
            sql += " WHERE day = ?"
            args = (day,)
        sql += " ORDER BY day, first_seen"
        return [AttendanceRow(*r) for r in self._conn.execute(sql, args)]

    def close(self) -> None:
        self._conn.close()


def export_csv(rows: Iterable[AttendanceRow], path: str | Path) -> int:
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["person", "day", "first_seen", "last_seen", "best_score"])
        for r in rows:
            writer.writerow([r.person, r.day, r.first_seen, r.last_seen, f"{r.best_score:.3f}"])
            count += 1
    return count


class AttendanceLogger:
    """Rate-limited, non-blocking front end for :class:`AttendanceStore`.

    * ``arrived()`` - a track was just confirmed as this person: logs an event and marks seen,
      unless the person was already reported within ``cooldown_s`` (then it acts as ``seen``).
    * ``seen()``    - person still visible: updates ``last_seen`` at most every ``cooldown_s``.
    """

    _STOP = object()

    def __init__(
        self,
        db_path: str | Path,
        cooldown_s: float = 10.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.db_path = Path(db_path)
        self.cooldown_s = cooldown_s
        self.clock = clock
        self._last_written: dict[str, float] = {}  # last DB write per person
        self._last_present: dict[str, float] = {}  # last time each person was reported at all
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._writer, name="attendance", daemon=True)
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread.start()
        self._ready.wait(timeout=5.0)
        if self._error:
            raise self._error

    def arrived(self, person: str, score: float, track_id: int | None = None) -> None:
        now = self.clock()
        last_present = self._last_present.get(person)
        if last_present is not None and now - last_present < self.cooldown_s:
            # Same person picked up by a new track after a short gap (turned away, occluded):
            # not a new arrival, so only refresh last_seen under the usual cooldown.
            self.seen(person, score)
            return
        self._last_present[person] = now
        self._queue.put(("event", person, now, score, track_id))
        self._last_written[person] = now

    def seen(self, person: str, score: float) -> None:
        now = self.clock()
        self._last_present[person] = now
        last = self._last_written.get(person)
        if last is None or now - last >= self.cooldown_s:
            self._last_written[person] = now
            self._queue.put(("seen", person, now, score, None))

    def close(self) -> None:
        self._queue.put(self._STOP)
        self._thread.join(timeout=5.0)

    def _writer(self) -> None:
        try:
            store = AttendanceStore(self.db_path)
        except BaseException as exc:  # surface DB errors to the constructor
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        try:
            while True:
                item = self._queue.get()
                batch = [item]
                # Drain whatever else is queued so one commit covers many updates.
                while True:
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
                stop = False
                for entry in batch:
                    if entry is self._STOP:
                        stop = True
                        continue
                    kind, person, ts, score, track_id = entry
                    try:
                        if kind == "event":
                            store.add_event(person, ts, score, track_id)
                        store.mark_seen(person, ts, score)
                    except sqlite3.Error:
                        log.exception("Failed to write attendance for %s", person)
                try:
                    store.commit()
                except sqlite3.Error:
                    # e.g. "database is locked" by another program: drop this batch but keep
                    # the writer alive, otherwise every later update would be silently lost.
                    log.exception("Failed to commit %d attendance update(s)", len(batch))
                    try:
                        store.rollback()
                    except sqlite3.Error:
                        pass
                if stop:
                    break
        finally:
            store.close()
