from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ... import config as app_config
from ...services import models as model_svc
from ...settings_store import DEFAULTS, SECRET_KEYS
from . import _deps

router = APIRouter()


@router.get("/settings")
def get_settings(request: Request) -> dict:
    store = _deps.store(request)
    return {
        "settings": store.all(),
        "defaults": DEFAULTS,
        "secrets": _deps.secrets_store(request).status(),
    }


def _entry(path: Path) -> dict:
    if path.is_file():
        st = path.stat()
        return {"path": str(path), "exists": True, "bytes": st.st_size, "n_files": 1}
    if path.is_dir():
        n, total = 0, 0
        for f in path.rglob("*"):
            if f.is_file():
                n += 1
                total += f.stat().st_size
        return {"path": str(path), "exists": True, "bytes": total, "n_files": n}
    return {"path": str(path), "exists": False, "bytes": 0, "n_files": 0}


@router.get("/settings/paths")
def get_paths(request: Request) -> dict:
    """Every filesystem location this instance reads/writes (spec §4.3) — so you
    can point a local run at your own dataset, or know exactly what a full
    export/import will carry to another host."""
    s = _deps.settings(request)
    return {
        "data_dir": _entry(s.data_dir),
        "app_db": _entry(s.app_db_path),
        "contours_db": _entry(s.contours_db_path),
        "images_raw": _entry(s.raw_images_dir),
        "images_purple": _entry(s.purple_images_dir),
        "images_thumb": _entry(s.thumb_images_dir),
        "models_dir": _entry(s.models_dir),
        "exports_dir": _entry(s.exports_dir),
        "logs_dir": _entry(s.logs_dir),
        "secrets_file": _entry(s.data_dir / "secrets.json"),
        "env_var": "SPOTTER_DATA_DIR",
        "env_override": bool(os.environ.get("SPOTTER_DATA_DIR")),
        "note": "Editable below; takes effect on the next server restart, not live — "
                "DuckDB allows one read-write connection per file (spec §2.4).",
    }


class DataDirBody(BaseModel):
    path: str


@router.post("/settings/data-dir")
def set_data_dir(request: Request, body: DataDirBody) -> dict:
    """Persist a new data directory for the *next* server start (spec §4.3). Does not
    touch the live DB — the running server keeps using its current directory until
    restarted, since DuckDB allows only one read-write connection per file."""
    raw = body.path.strip()
    if not raw:
        raise HTTPException(400, "path must not be empty")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise HTTPException(400, "path must be absolute")
    app_config.write_bootstrap_data_dir(path)
    env_override = bool(os.environ.get("SPOTTER_DATA_DIR"))
    _deps.db(request).log_activity(
        kind="maintenance", summary=f"Data directory set to {path} (takes effect on restart)",
        actor=_deps.actor(request),
    )
    return {
        "ok": True,
        "saved_path": str(path),
        "restart_required": True,
        "env_override": env_override,
        "warning": (
            f"SPOTTER_DATA_DIR is currently set in the environment and will override this "
            f"saved path until that env var is unset." if env_override else None
        ),
    }


class SettingsPatch(BaseModel):
    values: dict


@router.patch("/settings")
def patch_settings(request: Request, body: SettingsPatch) -> dict:
    store = _deps.store(request)
    unknown = set(body.values) - set(DEFAULTS)
    if unknown:
        raise HTTPException(400, f"unknown settings: {sorted(unknown)}")
    # score coefficients: reject anything but a-f -> numeric (spec §7.4, §15)
    if "score_coefficients" in body.values:
        coeffs = body.values["score_coefficients"]
        if not isinstance(coeffs, dict) or any(
            k not in "abcdef" or not isinstance(v, (int, float)) for k, v in coeffs.items()
        ):
            raise HTTPException(400, "score_coefficients must map letters a-f to numbers")
    store.update(body.values, actor=_deps.actor(request))
    return {"settings": store.all()}


class SecretBody(BaseModel):
    key: str
    value: str


@router.put("/settings/secrets")
def set_secret(request: Request, body: SecretBody) -> dict:
    if body.key not in SECRET_KEYS:
        raise HTTPException(400, f"unknown secret {body.key}")
    _deps.secrets_store(request).set(body.key, body.value)
    return {"secrets": _deps.secrets_store(request).status()}


@router.post("/settings/test-llm")
def test_llm(request: Request) -> dict:
    """Spec §10.2 'Test' button — is a key present and does the bridge load?"""
    secret = _deps.secrets_store(request).get("llm_api_key")
    return {"api_key_present": bool(secret), "bridge": type(_deps.bridges(request).extraction).__name__}


class ScoreBody(BaseModel):
    metrics: dict
    coefficients: dict | None = None


@router.post("/settings/score-preview")
def score_preview(request: Request, body: ScoreBody) -> dict:
    coeffs = body.coefficients or _deps.store(request).get("score_coefficients")
    return {"score": model_svc.compute_score(body.metrics, coeffs), "coefficients": coeffs}
