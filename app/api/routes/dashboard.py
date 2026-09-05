from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ...services import dashboard as dash
from ...services import notify
from . import _deps

router = APIRouter()


@router.get("/dashboard")
def dashboard(request: Request) -> dict:
    return dash.overview(_deps.db(request))


@router.get("/map")
def map_data(request: Request) -> dict:
    return dash.map_data(_deps.db(request))


@router.get("/activity")
def activity(request: Request, limit: int = 100) -> dict:
    rows = _deps.db(request).query_dicts(
        "SELECT * FROM activity ORDER BY created_at DESC LIMIT ?", [min(limit, 500)]
    )
    return {"activity": rows}


@router.get("/notifications")
def notifications(request: Request, include_dismissed: bool = False) -> dict:
    d = _deps.db(request)
    return {
        "notifications": notify.list_notifications(d, include_dismissed=include_dismissed),
        "unread": notify.unread_count(d),
    }


@router.post("/notifications/{nid}/read")
def read_notification(request: Request, nid: str) -> dict:
    notify.mark_read(_deps.db(request), nid)
    return {"ok": True}


@router.post("/notifications/{nid}/dismiss")
def dismiss_notification(request: Request, nid: str) -> dict:
    notify.dismiss(_deps.db(request), nid)
    return {"ok": True}
