"""Spec §2.2/§2.3 — backup/export from a *running* server (no stop required)."""

from __future__ import annotations

import tarfile

import duckdb
import pytest

from app import backup
from app.services import ingest


def test_live_backup_works_while_db_connection_stays_open(db, settings, dataset_dir):
    """The whole point: `db` here is exactly like a live server's connection —
    never closed during the call."""
    ingest.transfer_dataset(db, settings, dataset_dir)
    dest = settings.data_dir.parent / "backups"

    res = backup.live_backup(settings, db, dest)

    assert db.scalar("SELECT count(*) FROM images") > 0  # source connection still fine
    with tarfile.open(res.archive) as tar:
        names = tar.getnames()
    assert "app.duckdb" in names
    assert "contours.db" in names
    assert any(n.startswith("images/raw/") for n in names)


def test_live_backup_snapshot_is_a_consistent_independent_copy(db, settings, dataset_dir, tmp_path):
    ingest.transfer_dataset(db, settings, dataset_dir)
    res = backup.live_backup(settings, db, settings.data_dir.parent / "backups")

    extracted = tmp_path / "check"
    with tarfile.open(res.archive) as tar:
        tar.extractall(extracted)

    con = duckdb.connect(str(extracted / "app.duckdb"), read_only=True)
    try:
        assert con.execute("SELECT count(*) FROM images").fetchone()[0] == db.scalar(
            "SELECT count(*) FROM images"
        )
    finally:
        con.close()

    cdb = duckdb.connect(str(extracted / "contours.db"), read_only=True)
    try:
        assert cdb.execute("SELECT count(*) FROM images").fetchone()[0] > 0
    finally:
        cdb.close()


def test_live_export_full_then_import_full_roundtrip(db, settings, dataset_dir, tmp_path):
    ingest.transfer_dataset(db, settings, dataset_dir)
    archive = tmp_path / "full.tar.gz"

    backup.live_export_full(settings, db, archive)
    assert db.scalar("SELECT count(*) FROM individuals") > 0  # untouched, still open

    target = tmp_path / "restored_host"
    backup.import_full(archive, target)

    restored = duckdb.connect(str(target / "app.duckdb"), read_only=True)
    try:
        assert restored.execute("SELECT count(*) FROM individuals").fetchone()[0] == db.scalar(
            "SELECT count(*) FROM individuals"
        )
    finally:
        restored.close()
    assert (target / "images" / "raw").is_dir()
    assert list((target / "images" / "raw").glob("*"))


def test_live_export_excludes_wal_files(db, settings, dataset_dir, tmp_path):
    ingest.transfer_dataset(db, settings, dataset_dir)
    # a stray WAL sidecar (DuckDB itself already holds `app.duckdb.wal` open for
    # the life of the connection — proof enough that a raw copy of *that* file
    # would fail too; this checks a differently-named one is still excluded).
    (settings.data_dir / "contours.db.wal").write_bytes(b"stray-wal")
    archive = tmp_path / "full.tar.gz"
    backup.live_export_full(settings, db, archive)
    with tarfile.open(archive) as tar:
        assert not any(n.endswith(".wal") for n in tar.getnames())
