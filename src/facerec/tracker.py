"""IoU tracker that keeps an identity per face across frames, so faces are not re-embedded
on every frame. Identity is decided by voting over several embeddings to avoid flicker."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from facerec.config import TrackerConfig
from facerec.structures import Detection, MatchResult, TrackView


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4) in x, y, w, h format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1 = a[:, 0:1], a[:, 1:2]
    ax2, ay2 = ax1 + a[:, 2:3], ay1 + a[:, 3:4]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = bx1 + b[:, 2], by1 + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = a[:, 2:3] * a[:, 3:4] + b[:, 2] * b[:, 3] - inter
    return (inter / np.maximum(union, 1e-6)).astype(np.float32)


@dataclass
class Track:
    track_id: int
    detection: Detection
    votes: deque  # of (name | None, score)
    misses: int = 0
    hits: int = 1
    last_embed_frame: int = -(10**9)
    last_embed_area: float = 0.0
    name: str | None = None
    score: float = 0.0
    confirmed: bool = False
    announced: str | None = None  # identity last reported to attendance for this track

    @property
    def bbox(self) -> np.ndarray:
        return self.detection.bbox

    def view(self) -> TrackView:
        x, y, w, h = (int(round(v)) for v in self.bbox)
        if not self.confirmed:
            status = "pending"
        else:
            status = "known" if self.name else "unknown"
        return TrackView(self.track_id, (x, y, w, h), self.name, self.score, status)


class FaceTracker:
    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        """Associate detections with tracks; returns tracks seen in this frame."""
        boxes = np.array([d.bbox for d in detections], dtype=np.float32).reshape(-1, 4)
        prev = np.array([t.bbox for t in self.tracks], dtype=np.float32).reshape(-1, 4)
        ious = iou_matrix(prev, boxes)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        # Greedy assignment by descending IoU (plenty for the handful of faces in a frame).
        for ti, di in sorted(np.argwhere(ious >= self.cfg.iou_threshold), key=lambda p: -ious[p[0], p[1]]):
            if ti in matched_tracks or di in matched_dets:
                continue
            track = self.tracks[ti]
            track.detection = detections[di]
            track.misses = 0
            track.hits += 1
            matched_tracks.add(ti)
            matched_dets.add(di)

        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.cfg.max_misses]

        for di, det in enumerate(detections):
            if di not in matched_dets:
                self.tracks.append(
                    Track(self._next_id, det, deque(maxlen=max(1, self.cfg.vote_frames)))
                )
                self._next_id += 1

        return [t for t in self.tracks if t.misses == 0]

    # ---------------------------------------------------------------- recognition policy

    def needs_embedding(self, track: Track, frame_idx: int) -> bool:
        cfg = self.cfg
        if track.detection.min_side < cfg.min_face_size:
            return False
        if len(track.votes) < cfg.vote_frames:
            return True  # still collecting votes for a new track
        since = frame_idx - track.last_embed_frame
        interval = cfg.reidentify_interval if track.name else cfg.unknown_retry_interval
        if since >= interval:
            return True
        # Face got much bigger (walked closer) -> a better view is worth an extra look.
        return track.last_embed_area > 0 and track.detection.area > 2.0 * track.last_embed_area

    def add_vote(self, track: Track, match: MatchResult, frame_idx: int) -> None:
        track.votes.append((match.name, match.score))
        track.last_embed_frame = frame_idx
        track.last_embed_area = track.detection.area

        counts = Counter(name for name, _ in track.votes)
        name, count = counts.most_common(1)[0]
        scores = [s for n, s in track.votes if n == name]
        needed = min(self.cfg.confirm_votes, self.cfg.vote_frames)
        if count >= needed:
            track.name, track.confirmed = name, True
        elif not track.confirmed:
            track.name = None
        if name == track.name:
            track.score = float(np.mean(scores))
