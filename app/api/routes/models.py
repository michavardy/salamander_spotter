from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...services import models as model_svc
from ...services import training as training_svc
from . import _deps

router = APIRouter()


@router.get("/models")
def list_models(request: Request) -> dict:
    db = _deps.db(request)
    return {
        "models": model_svc.list_models(db),
        "active": model_svc.active_model(db),
        "runs": db.query_dicts("SELECT * FROM training_runs ORDER BY started_at DESC LIMIT 20"),
        "singleton_hint": model_svc.singleton_hint(db),
        "training_due": training_svc.training_due(db, _deps.store(request)),
    }


class ImportModelBody(BaseModel):
    name: str
    kind: str = "custom"
    source_weights_path: str
    source_calibration_path: str | None = None
    metrics: dict = {}
    notes: str | None = None
    make_active: bool = False


@router.post("/models/import")
def import_model(request: Request, body: ImportModelBody) -> dict:
    """Bring an already-trained checkpoint from your research onto the volume
    and register it (spec §7.6) — copies the file, zero GPU/LLM cost."""
    db = _deps.db(request)
    settings = _deps.settings(request)
    store = _deps.store(request)
    try:
        result = model_svc.import_model(
            db, settings, name=body.name, kind=body.kind,
            source_weights_path=body.source_weights_path,
            source_calibration_path=body.source_calibration_path,
            metrics=body.metrics, coefficients=store.get("score_coefficients"),
            notes=body.notes, actor=_deps.actor(request),
        )
    except model_svc.ModelImportError as exc:
        raise HTTPException(400, str(exc)) from exc
    if body.make_active:
        model_svc.promote(db, body.name, actor=_deps.actor(request))
    return result


class PromoteBody(BaseModel):
    name: str


@router.post("/models/promote")
def promote(request: Request, body: PromoteBody) -> dict:
    try:
        return model_svc.promote(_deps.db(request), body.name, actor=_deps.actor(request))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/models/rollback")
def rollback(request: Request) -> dict:
    try:
        return model_svc.rollback(_deps.db(request), actor=_deps.actor(request))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/models/retrain")
def retrain(request: Request) -> dict:
    """Kick off a training run (spec §7.5) — the real (13-model, 5-fold) sweep can take
    hours, so this submits and returns immediately; poll GET /jobs/{job_id} for progress
    and GET /jobs/{job_id}/log for the live log."""
    w = _deps.worker(request)
    job_id = w.submit("training_run", {"trigger": "manual", "actor": _deps.actor(request)})
    return {"job_id": job_id}


@router.get("/models/{name}/log")
def model_log(request: Request, name: str) -> dict:
    db = _deps.db(request)
    rows = db.query_dicts(
        "SELECT * FROM training_run_models WHERE model_name = ? ORDER BY run_id DESC", [name]
    )
    return {"model": name, "history": rows}
