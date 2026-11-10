import numpy as np
import pytest

from facerec.config import TrackerConfig
from facerec.structures import Detection, MatchResult
from facerec.tracker import FaceTracker, iou_matrix


def det(x, y, w=100, h=100):
    row = np.zeros(15, np.float32)
    row[:4] = (x, y, w, h)
    row[14] = 0.9
    return Detection.from_yunet_row(row)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 10, 10], [20, 20, 5, 5]], np.float32)
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 50 / 150, 0.0], rtol=1e-5)
    assert iou_matrix(a, np.zeros((0, 4), np.float32)).shape == (1, 0)


def test_ids_persist_while_faces_move():
    tr = FaceTracker(TrackerConfig())
    first = tr.update([det(0, 0), det(500, 0)])
    ids = {round(t.bbox[0]): t.track_id for t in first}
    for step in range(1, 10):
        tracks = tr.update([det(500 + 5 * step, 0), det(5 * step, 0)])  # order swapped too
        got = {t.track_id for t in tracks}
        assert got == set(ids.values())
    left = min(tr.tracks, key=lambda t: t.bbox[0])
    assert left.track_id == ids[0]


def test_track_survives_short_dropout_then_expires():
    tr = FaceTracker(TrackerConfig(max_misses=2))
    (t,) = tr.update([det(0, 0)])
    assert tr.update([]) == []
    (again,) = tr.update([det(2, 0)])
    assert again.track_id == t.track_id
    for _ in range(3):
        tr.update([])
    (new,) = tr.update([det(0, 0)])
    assert new.track_id != t.track_id


def test_voting_confirms_and_resists_flicker():
    cfg = TrackerConfig(vote_frames=5, confirm_votes=3, reidentify_interval=30)
    tr = FaceTracker(cfg)
    (t,) = tr.update([det(0, 0)])
    frame = 0
    for name in ["alice", None, "alice"]:
        frame += 1
        assert tr.needs_embedding(t, frame)
        tr.add_vote(t, MatchResult(name, 0.8 if name else 0.1), frame)
    assert t.view().status == "pending"
    frame += 1
    tr.add_vote(t, MatchResult("alice", 0.9), frame)
    assert t.confirmed and t.name == "alice"
    assert t.score == pytest.approx((0.8 + 0.8 + 0.9) / 3)
    # One stray "unknown" vote does not flip a confirmed identity.
    frame += 1
    tr.add_vote(t, MatchResult(None, 0.1), frame)
    assert t.name == "alice" and t.view().status == "known"
    # Votes are full -> no more embeddings until the re-identify interval passes.
    assert not tr.needs_embedding(t, frame + 1)
    assert tr.needs_embedding(t, frame + cfg.reidentify_interval)


def test_unknown_confirmed_and_retried_sooner():
    cfg = TrackerConfig(vote_frames=3, confirm_votes=2, unknown_retry_interval=5)
    tr = FaceTracker(cfg)
    (t,) = tr.update([det(0, 0)])
    for f in (1, 2, 3):
        tr.add_vote(t, MatchResult(None, 0.1), f)
    assert t.view().status == "unknown"
    assert not tr.needs_embedding(t, 7)
    assert tr.needs_embedding(t, 8)


def test_small_faces_are_not_embedded():
    tr = FaceTracker(TrackerConfig(min_face_size=40))
    (t,) = tr.update([det(0, 0, 20, 20)])
    assert not tr.needs_embedding(t, 1)
