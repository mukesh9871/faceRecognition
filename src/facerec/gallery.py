"""Known-face gallery: enrolment from disk, embedding cache, and vectorised matching."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from facerec.detectors.base import Detector
from facerec.recognizers.base import Recognizer
from facerec.structures import MatchResult

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Gallery:
    """L2-normalised embeddings grouped by person, matched with one matrix multiply.

    A person may have several embeddings (several photos); the score for a person is the
    best similarity over their embeddings.
    """

    def __init__(self, embeddings: np.ndarray, labels: list[str] | np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        labels = np.asarray(labels, dtype=str)
        if embeddings.ndim != 2 or embeddings.shape[0] != labels.shape[0]:
            raise ValueError("embeddings must be (N, D) with one label per row")
        # Sort by label so each person's rows are contiguous -> np.maximum.reduceat works.
        order = np.argsort(labels, kind="stable")
        self.embeddings = _l2_normalise(embeddings[order])
        self.labels = labels[order]
        self.names, self._starts = np.unique(self.labels, return_index=True)

    @classmethod
    def empty(cls, dim: int = 128) -> "Gallery":
        return cls(np.zeros((0, dim), np.float32), [])

    def __len__(self) -> int:
        return len(self.embeddings)

    @property
    def num_people(self) -> int:
        return len(self.names)

    def match(self, queries: np.ndarray, threshold: float) -> list[MatchResult]:
        """Match (K, D) or (D,) query embeddings; returns one MatchResult per query."""
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if len(q) == 0:
            return []
        if len(self.embeddings) == 0:
            return [MatchResult(None, 0.0) for _ in range(len(q))]
        sims = _l2_normalise(q) @ self.embeddings.T  # (K, N)
        per_person = np.maximum.reduceat(sims, self._starts, axis=1)  # (K, P)
        best = per_person.argmax(axis=1)
        scores = per_person[np.arange(len(q)), best]
        return [
            MatchResult(str(self.names[b]) if s >= threshold else None, float(s))
            for b, s in zip(best, scores)
        ]


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


# --------------------------------------------------------------------------- enrolment


@dataclass
class EnrollStats:
    images: int = 0
    reused: int = 0
    embedded: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)


def scan_faces_dir(faces_dir: Path) -> list[tuple[str, Path]]:
    """Return (person, image_path) pairs.

    Layout ``faces_dir/<Person>/*.jpg`` is preferred; loose files ``faces_dir/<Person>.jpg``
    are also accepted (the old project's layout).
    """
    if not faces_dir.exists():
        return []
    items: list[tuple[str, Path]] = []
    for entry in sorted(faces_dir.iterdir()):
        if entry.is_dir():
            for img in sorted(entry.rglob("*")):
                if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
                    items.append((entry.name, img))
        elif entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            items.append((entry.stem, entry))
    return items


def read_image(path: Path) -> np.ndarray | None:
    """cv2.imread that also works with non-ASCII paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def build_gallery(
    faces_dir: Path,
    detector: Detector,
    recognizer: Recognizer,
    cache_path: Path | None,
    backend: str,
) -> tuple[Gallery, EnrollStats]:
    """Embed every enrolment image, reusing cached embeddings for unchanged files."""
    faces_dir = Path(faces_dir)
    cached = _load_cache(cache_path, backend) if cache_path else {}
    stats = EnrollStats()

    rel_keys: list[str] = []
    labels: list[str] = []
    vectors: list[np.ndarray] = []
    mtimes: list[int] = []
    sizes: list[int] = []

    for person, img_path in scan_faces_dir(faces_dir):
        stats.images += 1
        st = img_path.stat()
        key = img_path.relative_to(faces_dir).as_posix()
        hit = cached.get(key)
        if hit is not None and hit[1] == st.st_mtime_ns and hit[2] == st.st_size and hit[3] == person:
            vec = hit[0]
            stats.reused += 1
        else:
            vec, reason = _embed_image(img_path, detector, recognizer)
            if vec is None:
                stats.skipped.append((key, reason))
                log.warning("Skipping %s: %s", key, reason)
                continue
            stats.embedded += 1
        rel_keys.append(key)
        labels.append(person)
        vectors.append(vec)
        mtimes.append(st.st_mtime_ns)
        sizes.append(st.st_size)

    dim = getattr(recognizer, "dim", 128)
    emb = np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, dim), np.float32)
    if cache_path is not None and (stats.embedded or len(cached) != len(rel_keys)):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            backend=np.array(backend),
            keys=np.array(rel_keys, dtype=str),
            labels=np.array(labels, dtype=str),
            embeddings=emb,
            mtimes=np.array(mtimes, dtype=np.int64),
            sizes=np.array(sizes, dtype=np.int64),
        )
    return Gallery(emb, labels), stats


def _embed_image(
    path: Path, detector: Detector, recognizer: Recognizer
) -> tuple[np.ndarray | None, str]:
    img = read_image(path)
    if img is None:
        return None, "could not read image"
    faces = detector.detect(img)
    if not faces:
        return None, "no face found"
    if len(faces) > 1:
        log.warning("%s: %d faces found, using the largest", path.name, len(faces))
    face = max(faces, key=lambda d: d.area)
    return recognizer.embed(img, face), ""


def _load_cache(cache_path: Path, backend: str) -> dict[str, tuple[np.ndarray, int, int, str]]:
    if not cache_path.exists():
        return {}
    try:
        with np.load(cache_path, allow_pickle=False) as z:
            if str(z["backend"]) != backend:
                log.info("Gallery cache was built with %s; re-embedding.", z["backend"])
                return {}
            return {
                str(k): (e, int(m), int(s), str(lbl))
                for k, e, m, s, lbl in zip(
                    z["keys"], z["embeddings"], z["mtimes"], z["sizes"], z["labels"]
                )
            }
    except Exception as exc:  # it's only a cache: any damage (e.g. BadZipFile) -> rebuild
        log.warning("Ignoring unreadable gallery cache %s (%s)", cache_path, exc)
        return {}
