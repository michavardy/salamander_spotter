from __future__ import annotations

from pathlib import Path

from app import config as app_config
from app.config import load_settings


def test_explicit_data_dir_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("SPOTTER_DATA_DIR", str(tmp_path / "from_env"))
    s = load_settings(data_dir=tmp_path / "explicit")
    assert s.data_dir == (tmp_path / "explicit").resolve()


def test_env_data_dir_used_when_no_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("SPOTTER_DATA_DIR", str(tmp_path / "from_env"))
    s = load_settings()
    assert s.data_dir == (tmp_path / "from_env").resolve()


def test_derived_paths(tmp_path):
    s = load_settings(data_dir=tmp_path)
    assert s.app_db_path == tmp_path.resolve() / "app.duckdb"
    assert s.contours_db_path == tmp_path.resolve() / "contours.db"
    assert s.raw_images_dir == tmp_path.resolve() / "images" / "raw"
    assert s.thumb_images_dir == tmp_path.resolve() / "images" / "thumb"


def test_ensure_dirs_is_idempotent(tmp_path):
    s = load_settings(data_dir=tmp_path / "d")
    s.ensure_dirs()
    s.ensure_dirs()
    assert s.raw_images_dir.is_dir()
    assert s.models_dir.is_dir()
    assert s.logs_dir.is_dir()


def test_bootstrap_data_dir_used_when_no_env(monkeypatch, tmp_path):
    fixed = tmp_path / "fixed_bootstrap_home"
    monkeypatch.setattr(app_config, "_default_data_dir", lambda: fixed)
    monkeypatch.delenv("SPOTTER_DATA_DIR", raising=False)
    app_config.write_bootstrap_data_dir(tmp_path / "chosen")
    s = load_settings()
    assert s.data_dir == (tmp_path / "chosen").resolve()


def test_env_data_dir_beats_bootstrap(monkeypatch, tmp_path):
    fixed = tmp_path / "fixed_bootstrap_home"
    monkeypatch.setattr(app_config, "_default_data_dir", lambda: fixed)
    app_config.write_bootstrap_data_dir(tmp_path / "chosen")
    monkeypatch.setenv("SPOTTER_DATA_DIR", str(tmp_path / "from_env"))
    s = load_settings()
    assert s.data_dir == (tmp_path / "from_env").resolve()


def test_read_bootstrap_data_dir_missing_file_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(app_config, "_default_data_dir", lambda: tmp_path / "nowhere")
    assert app_config.read_bootstrap_data_dir() is None


def test_env_bind_and_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("SPOTTER_HOST", "0.0.0.0")
    monkeypatch.setenv("SPOTTER_PORT", "9000")
    monkeypatch.setenv("SPOTTER_EXPOSE_API_DOCS", "true")
    monkeypatch.setenv("SPOTTER_EXTERNAL_BASE_URL", "https://spotter.example.org")
    s = load_settings(data_dir=tmp_path)
    assert (s.host, s.port) == ("0.0.0.0", 9000)
    assert s.expose_api_docs is True
    assert s.external_base_url == "https://spotter.example.org"
