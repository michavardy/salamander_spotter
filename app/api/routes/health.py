from __future__ import annotations

from fastapi import APIRouter, Request

from ... import __version__
from . import _deps

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    d = _deps.db(request)
    ok = True
    try:
        d.scalar("SELECT 1")
    except Exception:  # pragma: no cover
        ok = False
    return {"status": "ok" if ok else "degraded", "version": __version__}
