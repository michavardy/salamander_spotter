from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from . import _deps

router = APIRouter()


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict:
    """Poll a background job (spec §5.3) — used for anything too slow for a
    single request/response cycle: dataset import, backup, full export, training."""
    status = _deps.worker(request).status(job_id)
    if not status:
        raise HTTPException(404, "job not found")
    return status


@router.get("/jobs/{job_id}/log")
def get_job_log(request: Request, job_id: str) -> dict:
    """Tail a job's on-disk log (currently written by the training job only).
    Tolerates the file not existing yet — the subprocess may not have started
    writing, or this job type never writes one."""
    path = _deps.settings(request).logs_dir / f"{job_id}.log"
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    return {"text": text}
