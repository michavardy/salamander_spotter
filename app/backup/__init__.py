"""Backup / restore / full export / full import (spec §2.2, §2.3, §5.4).

Two ways to produce an archive, same layout either way (``restore``/``import_full``
don't care which made it):

* **Offline** (``backup`` / ``export_full``) — raw file copies. Requires the
  server to be **stopped**: DuckDB locks its file exclusively for the life of a
  connection (notably on Windows, where even a read-only external process is
  refused), so a live ``app.duckdb`` can't be read by a second process.
* **Live / hot** (``live_backup`` / ``live_export_full``) — takes the running
  server's own :class:`~app.db.Database` and uses ``COPY FROM DATABASE`` on its
  *existing* connection to write independent, already-unlocked copies of
  ``app.duckdb`` and ``contours.db`` before taring them up. Safe to run while
  the app is serving traffic; this is what the Settings page button and
  ``POST /api/backup`` / ``POST /api/exports/full`` use.
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings

_ARCHIVE_MEMBERS = ("app.duckdb", "contours.db", "config.toml", "secrets.json")
_SKIP_SUFFIXES = (".wal",)  # a stray DuckDB WAL must never ride along in an archive


class DatabaseLockedError(RuntimeError):
    """Raised when a DB file can't be read because a running server holds it open.

    Fix: stop the server and retry the offline command, or use the live variant
    (``live_backup`` / ``live_export_full`` / the Settings-page button / the
    ``/api/backup`` and ``/api/exports/full`` endpoints) instead.
    """


@dataclass
class BackupResult:
    archive: str
    bytes: int
    files: int


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _add(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    try:
        tar.add(path, arcname=arcname)
    except PermissionError as exc:
        raise DatabaseLockedError(
            f"Cannot read {path} — the server appears to still be running against this "
            f"data dir. Stop it and re-run, or use the live backup/export instead."
        ) from exc


# --------------------------------------------------------------------------- #
#  offline (server stopped)                                                    #
# --------------------------------------------------------------------------- #
def backup(settings: Settings, dest_dir: Path, *, keep_daily: int = 30, keep_monthly: int = 6) -> BackupResult:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / f"spotter_{_stamp()}.tar.gz"

    files = 0
    with tarfile.open(archive, "w:gz") as tar:
        for name in _ARCHIVE_MEMBERS:
            p = settings.data_dir / name
            if p.exists():
                _add(tar, p, name)
                files += 1
        for sub in ("images", "models", "exports"):
            d = settings.data_dir / sub
            if d.is_dir():
                for f in d.rglob("*"):
                    if f.is_file():
                        _add(tar, f, str(f.relative_to(settings.data_dir)))
                        files += 1
    _prune(dest_dir, keep_daily, keep_monthly)
    return BackupResult(archive=str(archive), bytes=archive.stat().st_size, files=files)


def export_full(settings: Settings, archive_path: Path) -> Path:
    archive_path = Path(archive_path)
    with tarfile.open(archive_path, "w:gz") as tar:
        for f in settings.data_dir.rglob("*"):
            if f.is_file() and ".spotter.lock" not in f.name and not f.name.endswith(_SKIP_SUFFIXES):
                _add(tar, f, str(f.relative_to(settings.data_dir)))
    return archive_path


# --------------------------------------------------------------------------- #
#  live / hot (server running) — spec §2.2's "consistent copy"                 #
# --------------------------------------------------------------------------- #
def _snapshot_working_copies(settings: Settings, db) -> tuple[Path, Path | None, Path | None]:
    """Use the live connection to write independent, already-unlocked copies of
    app.duckdb (+ contours.db, if present) into a temp dir. Caller removes it."""
    tmpdir = Path(tempfile.mkdtemp(prefix="spotter_snapshot_"))
    app_copy = tmpdir / "app.duckdb"
    db.snapshot_to(app_copy)

    contours_copy = None
    if settings.contours_db_path.exists():
        contours_copy = tmpdir / "contours.db"
        db.snapshot_attached_to(settings.contours_db_path, contours_copy)
    return tmpdir, app_copy, contours_copy


def live_backup(
    settings: Settings, db, dest_dir: Path, *, keep_daily: int = 30, keep_monthly: int = 6
) -> BackupResult:
    tmpdir, app_copy, contours_copy = _snapshot_working_copies(settings, db)
    try:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        archive = dest_dir / f"spotter_{_stamp()}.tar.gz"
        files = 0
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(app_copy, arcname="app.duckdb")
            files += 1
            if contours_copy is not None:
                tar.add(contours_copy, arcname="contours.db")
                files += 1
            for name in ("config.toml", "secrets.json"):
                p = settings.data_dir / name
                if p.exists():
                    tar.add(p, arcname=name)
                    files += 1
            for sub in ("images", "models", "exports"):
                d = settings.data_dir / sub
                if d.is_dir():
                    for f in d.rglob("*"):
                        if f.is_file():
                            tar.add(f, arcname=str(f.relative_to(settings.data_dir)))
                            files += 1
        _prune(dest_dir, keep_daily, keep_monthly)
        return BackupResult(archive=str(archive), bytes=archive.stat().st_size, files=files)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def live_export_full(settings: Settings, db, archive_path: Path) -> Path:
    tmpdir, app_copy, contours_copy = _snapshot_working_copies(settings, db)
    exclude = {settings.app_db_path.resolve(), settings.contours_db_path.resolve()}
    try:
        archive_path = Path(archive_path)
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(app_copy, arcname="app.duckdb")
            if contours_copy is not None:
                tar.add(contours_copy, arcname="contours.db")
            for f in settings.data_dir.rglob("*"):
                if (
                    f.is_file()
                    and f.resolve() not in exclude
                    and ".spotter.lock" not in f.name
                    and not f.name.endswith(_SKIP_SUFFIXES)
                ):
                    tar.add(f, arcname=str(f.relative_to(settings.data_dir)))
        return archive_path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
def _prune(dest_dir: Path, keep_daily: int, keep_monthly: int) -> None:
    archives = sorted(dest_dir.glob("spotter_*.tar.gz"), reverse=True)
    keep: set[Path] = set(archives[:keep_daily])
    seen_months: set[str] = set()
    for a in archives:
        month = a.name[8:15]  # spotter_YYYYMM..
        if month not in seen_months and len(seen_months) < keep_monthly:
            seen_months.add(month)
            keep.add(a)
    for a in archives:
        if a not in keep:
            a.unlink(missing_ok=True)


def restore(archive: Path, target_data_dir: Path) -> Path:
    target = Path(target_data_dir)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        _safe_extractall(tar, target)
    return target


def import_full(archive_path: Path, target_data_dir: Path) -> Path:
    return restore(Path(archive_path), Path(target_data_dir))


def last_backup_age_hours(dest_dir: Path) -> float | None:
    archives = sorted(Path(dest_dir).glob("spotter_*.tar.gz"), reverse=True)
    if not archives:
        return None
    newest = archives[0].stat().st_mtime
    return (datetime.now(timezone.utc).timestamp() - newest) / 3600.0


def _safe_extractall(tar: tarfile.TarFile, target: Path) -> None:
    target = target.resolve()
    for member in tar.getmembers():
        dest = (target / member.name).resolve()
        if not str(dest).startswith(str(target)):
            raise ValueError(f"unsafe path in archive: {member.name}")
    tar.extractall(target)
