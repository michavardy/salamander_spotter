from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...services import ingest
from ...settings_store import SECRET_KEYS
from . import _deps

router = APIRouter()


@router.get("/setup/status")
def setup_status(request: Request) -> dict:
    """First-run flow state (spec §14.1)."""
    db = _deps.db(request)
    settings_obj = _deps.settings(request)
    secrets = _deps.secrets_store(request).status()
    models_present = [p.name for p in settings_obj.models_dir.iterdir() if p.is_dir()] if settings_obj.models_dir.is_dir() else []
    return {
        "has_data": bool(db.scalar("SELECT count(*) FROM images")),
        "has_api_keys": secrets["llm_api_key"]["set"],
        "contributors": db.scalar("SELECT count(*) FROM contributors"),
        "models_on_volume": models_present,
        "active_model": _deps.store(request).get("active_model"),
        "site": db.query_one("SELECT * FROM sites LIMIT 1"),
    }


class ContributorBody(BaseModel):
    name: str
    short_name: str | None = None
    contact: str | None = None


@router.post("/setup/contributors")
def add_contributor(request: Request, body: ContributorBody) -> dict:
    from ...db import new_id

    db = _deps.db(request)
    cid = new_id("ctb")
    db.insert("contributors", {"id": cid, "name": body.name, "short_name": body.short_name,
                               "contact": body.contact})
    return {"id": cid}


class ActiveModelBody(BaseModel):
    name: str


@router.post("/setup/active-model")
def set_active_model(request: Request, body: ActiveModelBody) -> dict:
    from ...services import models as model_svc

    db = _deps.db(request)
    if not db.query_one("SELECT 1 FROM models WHERE name = ?", [body.name]):
        raise HTTPException(404, "model not registered on the volume")
    return model_svc.promote(db, body.name, actor=_deps.actor(request))
