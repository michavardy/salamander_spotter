from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...services import batches as batch_svc
from ...services import decision as decision_svc
from ...services import matching
from . import _deps

router = APIRouter()


@router.get("/review")
def review_queue(request: Request, batch_id: str | None = None) -> dict:
    db = _deps.db(request)
    where = "WHERE status IN ('in_review','matched','extracted')"
    params: list = []
    if batch_id:
        where += " AND review_batch_id = ?"
        params.append(batch_id)
    rows = db.query_dicts(
        f"SELECT image_id, individual_id, status, ladder_tier, q_overall, n_spots, review_batch_id "
        f"FROM images {where} ORDER BY ladder_tier, image_id",
        params,
    )
    uncertain = db.query_dicts(
        "SELECT image_id, individual_id FROM images WHERE status = 'flagged_uncertain'"
    )
    return {"queue": rows, "uncertain": uncertain, "count": len(rows)}


@router.post("/review/{image_id}/match")
def run_match(request: Request, image_id: str) -> dict:
    db = _deps.db(request)
    settings_obj = _deps.settings(request)
    store = _deps.store(request)
    if not db.query_one("SELECT 1 FROM images WHERE image_id = ?", [image_id]):
        raise HTTPException(404, "image not found")
    bridge = _deps.bridges(request).matching
    try:
        match_id = matching.run_match(
            db, store, query_image_id=image_id, bridge=bridge,
            contours_db_path=settings_obj.contours_db_path,
        )
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
    triage = decision_svc.triage(db, store, image_id=image_id, match_id=match_id)
    return {
        "match_id": match_id,
        "candidates": matching.load_candidates(db, match_id),
        "triage": triage.__dict__,
    }


@router.get("/review/{image_id}")
def review_detail(request: Request, image_id: str) -> dict:
    db = _deps.db(request)
    img = db.query_one("SELECT * FROM images WHERE image_id = ?", [image_id])
    if not img:
        raise HTTPException(404, "image not found")
    match = matching.latest_match_for(db, image_id)
    candidates = matching.load_candidates(db, match["id"]) if match else []
    decisions = db.query_dicts(
        "SELECT * FROM review_decisions WHERE image_id = ? ORDER BY decided_at DESC", [image_id]
    )
    return {"image": img, "match": match, "candidates": candidates, "decisions": decisions,
            "reason_chips": _deps.store(request).get("reason_chips")}


@router.get("/review/{image_id}/match-lines/{candidate_id}")
def match_lines(request: Request, image_id: str, candidate_id: str) -> dict:
    """Spot correspondences for the Match-lines view (geometric matcher, viz only)."""
    settings_obj = _deps.settings(request)
    bridge = _deps.bridges(request).correspondence
    cand_image = f"{candidate_id}_1"
    try:
        links = bridge.correspond(image_id, cand_image, settings_obj.contours_db_path)
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
    return {"links": [link.__dict__ for link in links]}


class DecisionBody(BaseModel):
    verdict: str
    chosen_individual_id: str | None = None
    reason_chips: list[str] | None = None
    note: str | None = None
    confirm_override: bool = False


@router.post("/review/{image_id}/decision")
def decide(request: Request, image_id: str, body: DecisionBody) -> dict:
    db = _deps.db(request)
    store = _deps.store(request)
    try:
        outcome = decision_svc.apply_decision(
            db, store,
            decision_svc.DecisionInput(
                image_id=image_id, verdict=body.verdict,
                chosen_individual_id=body.chosen_individual_id,
                reason_chips=body.reason_chips, note=body.note,
                reviewer_id=_deps.actor(request), confirm_override=body.confirm_override,
            ),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if outcome.override_warning:
        return {"needs_confirmation": True, "warning": outcome.override_warning}
    return {
        "decision_id": outcome.decision_id,
        "verdict": outcome.verdict,
        "individual_id": outcome.individual_id,
        "new_individual_id": outcome.new_individual_id,
        "was_override": outcome.was_override,
    }


# --- batches -------------------------------------------------------------
class BatchBody(BaseModel):
    name: str


@router.get("/batches")
def list_batches(request: Request) -> dict:
    db = _deps.db(request)
    rows = batch_svc.open_batches(db)
    for r in rows:
        r["counts"] = batch_svc.batch_counts(db, r["id"])
    return {"batches": rows}


@router.post("/batches")
def open_batch(request: Request, body: BatchBody) -> dict:
    bid = batch_svc.create_batch(_deps.db(request), body.name, actor=_deps.actor(request))
    return {"batch_id": bid}


@router.post("/batches/{batch_id}/publish")
def publish(request: Request, batch_id: str) -> dict:
    try:
        return batch_svc.publish_batch(_deps.db(request), batch_id, actor=_deps.actor(request))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/batches/{batch_id}/unpublish")
def unpublish(request: Request, batch_id: str) -> dict:
    batch_svc.unpublish_batch(_deps.db(request), batch_id, actor=_deps.actor(request))
    return {"ok": True}


@router.get("/census")
def census(request: Request) -> dict:
    return batch_svc.census(_deps.db(request))
