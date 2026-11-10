from pathlib import Path

import pytest

from facerec.config import Config, load_config


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_relative_paths_resolve_next_to_the_config_file(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "site"
    cfg_dir.mkdir()
    cfg_file = write(cfg_dir / "config.yaml", "paths:\n  models_dir: my_models\n")
    monkeypatch.chdir(tmp_path)  # a different working directory must not matter
    cfg = load_config(cfg_file)
    assert cfg.base_dir == cfg_dir.resolve()
    assert cfg.resolve(cfg.paths.models_dir) == cfg_dir.resolve() / "my_models"
    assert cfg.resolve(tmp_path / "abs.onnx") == tmp_path / "abs.onnx"


def test_sections_override_only_the_given_keys(tmp_path):
    cfg = load_config(write(tmp_path / "c.yaml", "tracker:\n  vote_frames: 7\nnum_threads: 2\n"))
    assert cfg.tracker.vote_frames == 7
    assert cfg.tracker.confirm_votes == Config().tracker.confirm_votes
    assert cfg.num_threads == 2


def test_unknown_section_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="detector"):
        load_config(write(tmp_path / "c.yaml", "detector:\n  max_sid: 320\n"))


def test_unknown_top_level_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="camra"):
        load_config(write(tmp_path / "c.yaml", "camra:\n  source: 1\n"))


def test_missing_explicit_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_defaults_when_no_config_in_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg == Config(base_dir=cfg.base_dir)
    assert cfg.base_dir == tmp_path


def test_repo_config_yaml_matches_the_dataclasses():
    # Every key in the shipped config.yaml must exist in the dataclasses (load would raise).
    repo_cfg = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(repo_cfg)
    assert cfg.detector.name == "yunet" and cfg.recognizer.name == "sface"
