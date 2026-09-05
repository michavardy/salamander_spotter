"""On-demand backup / full export from a *running* server (spec §2.2, §2.3).

Both use the live connection's ``COPY FROM DATABASE`` snapshot (app/backup.py),
never a raw file copy — so, unlike the offline ``app backup`` / ``app export
--full`` CLI, these work while the app keeps serving traffic. Both are slow for
a large data dir, so they return a job id to poll via ``GET /api/jobs/{id}``
rather than blocking the request.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel

from . import _deps

router = APIRouter()


class BackupBody(BaseModel):
    dest: str | None = None


@router.post("/backup")
def start_backup(request: Request, body: BackupBody = BackupBody()) -> dict:
    settings = _deps.settings(request)
    dest = body.dest or str(settings.data_dir.parent / "spotter-backups")
    job_id = _deps.worker(request).submit("backup", {"dest": dest})
    return {"job_id": job_id, "dest": dest}


@router.post("/exports/full")
def start_export_full(request: Request) -> dict:
    """Zip the whole data dir for moving to another host (spec §2.3). The
    archive lands in exports/ so it's downloadable via GET /api/exports/file/*."""
    settings = _deps.settings(request)
    from datetime import datetime, timezone

    name = f"spotter_full_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.tar.gz"
    archive_path = settings.exports_dir / name
    job_id = _deps.worker(request).submit("export_full", {"archive_path": str(archive_path)})
    return {"job_id": job_id, "name": name, "download": f"/api/exports/file/{name}"}
