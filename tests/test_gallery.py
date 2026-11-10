import numpy as np
import pytest

from facerec.gallery import Gallery, build_gallery, scan_faces_dir
from facerec.structures import Detection


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_match_picks_best_person_and_applies_threshold():
    g = Gallery(
        np.stack([unit([1, 0, 0]), unit([0, 1, 0]), unit([0.9, 0.1, 0])]),
        ["alice", "bob", "alice"],
    )
    assert g.num_people == 2 and len(g) == 3
    res = g.match(np.stack([unit([1, 0.05, 0]), unit([0, 1, 0.1]), unit([0, 0, 1])]), 0.5)
    assert [r.name for r in res] == ["alice", "bob", None]
    assert res[0].score > 0.99
    assert res[2].score == pytest.approx(0.0, abs=1e-6)


def test_match_uses_best_of_several_images_per_person():
    g = Gallery(np.stack([unit([1, 0]), unit([0, 1])]), ["alice", "alice"])
    (r,) = g.match(unit([0, 1]), 0.9)
    assert r.name == "alice" and r.score == pytest.approx(1.0)


def test_empty_gallery_returns_unknown():
    g = Gallery.empty(3)
    res = g.match(np.stack([unit([1, 0, 0])] * 2), 0.3)
    assert [r.name for r in res] == [None, None]


def test_unnormalised_inputs_are_normalised():
    g = Gallery(np.array([[10.0, 0.0]]), ["a"])
    (r,) = g.match(np.array([3.0, 0.0]), 0.5)
    assert r.score == pytest.approx(1.0)


def test_scan_supports_folders_and_loose_files(tmp_path):
    (tmp_path / "alice").mkdir()
    (tmp_path / "alice" / "1.jpg").write_bytes(b"x")
    (tmp_path / "alice" / "2.PNG").write_bytes(b"x")
    (tmp_path / "alice" / "notes.txt").write_bytes(b"x")
    (tmp_path / "bob.jpg").write_bytes(b"x")
    found = [(p, f.name) for p, f in scan_faces_dir(tmp_path)]
    assert found == [("alice", "1.jpg"), ("alice", "2.PNG"), ("bob", "bob.jpg")]


class _FakeDetector:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if frame.mean() < 1:  # an all-black image has "no face"
            return []
        row = np.zeros(15, np.float32)
        row[2:4] = 50
        return [Detection.from_yunet_row(row)]


class _FakeRecognizer:
    name = "fake"
    dim = 3

    def embed(self, frame, det):
        return unit([float(frame.mean()), 1.0, 0.0])


def _write_img(path, value):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((64, 64, 3), value, np.uint8))


def test_build_gallery_caches_and_skips_faceless_images(tmp_path):
    faces = tmp_path / "faces"
    _write_img(faces / "alice" / "a.png", 200)
    _write_img(faces / "bob" / "b.png", 100)
    _write_img(faces / "carol" / "c.png", 0)  # no face
    cache = tmp_path / "gallery.npz"
    det, rec = _FakeDetector(), _FakeRecognizer()

    g, stats = build_gallery(faces, det, rec, cache, "fake:v1")
    assert (stats.images, stats.embedded, stats.reused) == (3, 2, 0)
    assert stats.skipped == [("carol/c.png", "no face found")]
    assert list(g.names) == ["alice", "bob"]
    assert cache.exists()

    det.calls = 0
    g2, stats2 = build_gallery(faces, det, rec, cache, "fake:v1")
    assert (stats2.embedded, stats2.reused) == (0, 2)
    assert det.calls == 1  # only the faceless image is retried
    np.testing.assert_allclose(g2.embeddings, g.embeddings)

    # A different backend invalidates the cache.
    _, stats3 = build_gallery(faces, det, rec, cache, "fake:v2")
    assert stats3.embedded == 2


def test_corrupt_cache_is_rebuilt_not_fatal(tmp_path):
    faces = tmp_path / "faces"
    _write_img(faces / "alice" / "a.png", 200)
    cache = tmp_path / "gallery.npz"
    det, rec = _FakeDetector(), _FakeRecognizer()
    build_gallery(faces, det, rec, cache, "fake:v1")
    data = cache.read_bytes()
    cache.write_bytes(data[: len(data) // 2])  # e.g. killed while the cache was being written

    g, stats = build_gallery(faces, det, rec, cache, "fake:v1")
    assert (stats.embedded, stats.reused) == (1, 0)
    assert list(g.names) == ["alice"]
    _, stats2 = build_gallery(faces, det, rec, cache, "fake:v1")  # the cache was rewritten
    assert stats2.reused == 1


def test_cache_is_written_at_the_configured_name(tmp_path):
    faces = tmp_path / "faces"
    _write_img(faces / "alice" / "a.png", 200)
    cache = tmp_path / "gallery.cache"  # not *.npz: numpy must not rename it
    det, rec = _FakeDetector(), _FakeRecognizer()
    build_gallery(faces, det, rec, cache, "fake:v1")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["faces", "gallery.cache"]
    _, stats = build_gallery(faces, det, rec, cache, "fake:v1")
    assert stats.reused == 1
