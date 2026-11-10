import io

import pytest

from facerec import model_zoo


class BrokenResponse(io.RawIOBase):
    """Delivers a few bytes, then the connection drops."""

    def __init__(self):
        self.sent = False

    def readinto(self, b):
        if self.sent:
            raise ConnectionResetError("connection reset by peer")
        self.sent = True
        b[:4] = b"ONNX"
        return 4

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_failed_download_leaves_no_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(model_zoo.urllib.request, "urlopen", lambda url, timeout: BrokenResponse())
    with pytest.raises(ConnectionResetError):
        model_zoo.download_models(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_checksum_mismatch_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        model_zoo.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"not the model")
    )
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        model_zoo.download_models(tmp_path)
    assert list(tmp_path.iterdir()) == []
