from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...services import ingest
from . import _deps

router = APIRouter()


class ImportRequest(BaseModel):
    dataset_path: str


@router.post("/imports")
def import_dataset(request: Request, body: ImportRequest) -> dict:
    """One-time / delta dataset transfer (spec §7.9 A). Runs as a worker job."""
    path = Path(body.dataset_path)
    if not path.is_dir():
        raise HTTPException(400, f"not a directory: {path}")
    w = _deps.worker(request)
    result = w.run_sync("import_dataset", {"dataset_path": str(path), "actor": _deps.actor(request)})
    if result["status"] == "failed":
        raise HTTPException(400, result["error"].splitlines()[0])
    return result["result"]


@router.get("/imports/history")
def import_history(request: Request) -> dict:
    rows = _deps.db(request).query_dicts(
        "SELECT * FROM audit WHERE action = 'import_dataset' ORDER BY at_ts DESC LIMIT 50"
    )
    return {"imports": rows}
