from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from app import backup


def _seed(settings):
    settings.app_db_path.write_text("db")
    settings.contours_db_path.write_text("contours")
    (settings.raw_images_dir / "aa_1_1.jpg").write_bytes(b"img")
    (settings.data_dir / "config.toml").write_text("x=1")


def test_backup_creates_archive(settings):
    _seed(settings)
    dest = settings.data_dir.parent / "backups"
    res = backup.backup(settings, dest)
    assert Path(res.archive).exists()
    with tarfile.open(res.archive) as t:
        names = t.getnames()
    assert "app.duckdb" in names
    assert "images/raw/aa_1_1.jpg" in names


def test_restore_roundtrip(settings, tmp_path):
    _seed(settings)
    dest = tmp_path / "backups"
    res = backup.backup(settings, dest)
    target = tmp_path / "restored"
    backup.restore(Path(res.archive), target)
    assert (target / "app.duckdb").read_text() == "db"
    assert (target / "images" / "raw" / "aa_1_1.jpg").exists()


def test_export_import_full(settings, tmp_path):
    _seed(settings)
    archive = tmp_path / "full.tar.gz"
    backup.export_full(settings, archive)
    target = tmp_path / "host2"
    backup.import_full(archive, target)
    assert (target / "contours.db").read_text() == "contours"


def test_locked_db_raises_clear_error(settings, monkeypatch):
    """A DB file held open by a live server (esp. on Windows, DuckDB's exclusive
    lock) must surface as a clear error, not a raw PermissionError traceback."""
    _seed(settings)

    def _blocked_add(self, name, arcname=None, **kw):
        raise PermissionError(13, "used by another process")

    monkeypatch.setattr(tarfile.TarFile, "add", _blocked_add)
    with pytest.raises(backup.DatabaseLockedError, match="server appears to still be running"):
        backup.backup(settings, settings.data_dir.parent / "backups")


def test_prune_keeps_recent(settings):
    dest = settings.data_dir.parent / "backups"
    dest.mkdir(parents=True)
    for i in range(40):
        (dest / f"spotter_202601{i:02d}T000000Z.tar.gz").write_bytes(b"x")
    _seed(settings)
    backup.backup(settings, dest, keep_daily=5, keep_monthly=1)
    remaining = list(dest.glob("spotter_*.tar.gz"))
    assert len(remaining) <= 7  # 5 daily + up to 1 monthly + the new one
