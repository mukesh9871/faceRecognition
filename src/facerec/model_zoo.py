"""Download and verify the ONNX models used by the default backends."""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"


@dataclass(frozen=True)
class ModelFile:
    filename: str
    url: str
    sha256: str


MODELS = (
    ModelFile(
        "face_detection_yunet_2023mar.onnx",
        f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    ModelFile(
        "face_recognition_sface_2021dec.onnx",
        f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_models(models_dir: Path, force: bool = False) -> list[Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for m in MODELS:
        dest = models_dir / m.filename
        if dest.exists() and not force and sha256_of(dest) == m.sha256:
            print(f"  ok        {m.filename}")
            paths.append(dest)
            continue
        print(f"  download  {m.filename} ...", flush=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=models_dir, suffix=".part") as tmp:
            tmp_path = Path(tmp.name)
        try:
            with open(tmp_path, "wb") as out, urllib.request.urlopen(m.url, timeout=60) as resp:
                shutil.copyfileobj(resp, out)
            digest = sha256_of(tmp_path)
            if digest != m.sha256:
                raise RuntimeError(
                    f"Checksum mismatch for {m.filename}: expected {m.sha256}, got {digest}"
                )
            tmp_path.replace(dest)
        except BaseException:
            # Network error, bad checksum or Ctrl+C: never leave a partial .part file behind.
            tmp_path.unlink(missing_ok=True)
            raise
        paths.append(dest)
    return paths
