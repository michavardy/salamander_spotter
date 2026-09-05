from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...services import extraction_editor
from . import _deps

router = APIRouter()


@router.get("/individuals")
def list_individuals(request: Request, limit: int = 200, offset: int = 0, q: str | None = None) -> dict:
    db = _deps.db(request)
    where = (
        "WHERE ind.merged_into IS NULL AND EXISTS ("
        "SELECT 1 FROM images WHERE individual_id = ind.individual_id AND status != 'disqualified')"
    )
    params: list = []
    if q:
        where += " AND (lower(ind.individual_id) LIKE ? OR lower(ind.display_id) LIKE ? OR lower(coalesce(ind.nickname,'')) LIKE ?)"
        params += [f"%{q.lower()}%"] * 3
    rows = db.query_dicts(
        f"""
        SELECT ind.individual_id, ind.display_id, ind.nickname, ind.status,
               ind.first_seen, ind.last_seen, ind.reference_image_id,
               count(img.image_id) FILTER (WHERE img.status != 'disqualified') AS n_images,
               count(img.image_id) FILTER (WHERE NOT img.is_synthetic AND img.status != 'disqualified') AS n_real,
               min(img.q_overall) AS worst_quality
        FROM individuals ind
        LEFT JOIN images img ON img.individual_id = ind.individual_id
        {where}
        GROUP BY 1,2,3,4,5,6,7
        ORDER BY ind.individual_id
        LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    )
    for r in rows:
        r["health"] = _health(r)
    return {"total": db.scalar(
                f"SELECT count(*) FROM individuals ind {where}", params
            ),
            "individuals": rows}


def _health(row: dict) -> str:
    if (row["n_real"] or 0) >= 2 and (row["worst_quality"] or 0) >= 0.65:
        return "strong"
    if (row["n_real"] or 0) <= 1 or (row["worst_quality"] or 0) < 0.40:
        return "weak" if (row["worst_quality"] or 1) < 0.40 else "thin"
    return "thin"


@router.get("/individuals/{individual_id}")
def get_individual(request: Request, individual_id: str) -> dict:
    db = _deps.db(request)
    ind = db.query_one("SELECT * FROM individuals WHERE individual_id = ?", [individual_id])
    if not ind:
        raise HTTPException(404, "individual not found")
    images = db.query_dicts(
        "SELECT image_id, is_synthetic, status, ladder_tier, q_overall, n_spots, origin, "
        "contributor_id, photographed_at FROM images WHERE individual_id = ? "
        "ORDER BY is_synthetic, image_id",
        [individual_id],
    )
    contributors = db.query_dicts(
        "SELECT c.* FROM contributors c JOIN individual_contributors ic ON ic.contributor_id = c.id "
        "WHERE ic.individual_id = ?",
        [individual_id],
    )
    return {"individual": ind, "images": images, "contributors": contributors}


class RenameBody(BaseModel):
    nickname: str | None = None
    rev: int | None = None


@router.patch("/individuals/{individual_id}")
def update_individual(request: Request, individual_id: str, body: RenameBody) -> dict:
    db = _deps.db(request)
    ind = db.query_one("SELECT * FROM individuals WHERE individual_id = ?", [individual_id])
    if not ind:
        raise HTTPException(404, "individual not found")
    if body.rev is not None and body.rev != ind["rev"]:
        raise HTTPException(409, "stale rev — refetch")
    db.execute(
        "UPDATE individuals SET nickname = ?, rev = rev + 1 WHERE individual_id = ?",
        [body.nickname, individual_id],
    )
    db.audit(action="rename_individual", entity="individual", entity_id=individual_id,
             actor=_deps.actor(request), before={"nickname": ind["nickname"]},
             after={"nickname": body.nickname})
    return {"ok": True, "rev": ind["rev"] + 1}


@router.get("/images/{image_id}")
def get_image(request: Request, image_id: str) -> dict:
    db = _deps.db(request)
    img = db.query_one("SELECT * FROM images WHERE image_id = ?", [image_id])
    if not img:
        raise HTTPException(404, "image not found")
    corrections = db.query_dicts(
        "SELECT id, editor_id, edited_at, diff_json FROM extraction_corrections "
        "WHERE image_id = ? ORDER BY edited_at DESC",
        [image_id],
    )
    return {"image": img, "corrections": corrections}


class EditBody(BaseModel):
    ops: list[dict]


@router.post("/images/{image_id}/extraction/edit")
def edit_extraction(request: Request, image_id: str, body: EditBody) -> dict:
    db = _deps.db(request)
    settings_obj = _deps.settings(request)
    bridge = _deps.bridges(request).editor
    if not db.query_one("SELECT 1 FROM images WHERE image_id = ?", [image_id]):
        raise HTTPException(404, "image not found")
    try:
        result = extraction_editor.apply_edit(
            db, settings_obj.contours_db_path, image_id=image_id, ops=body.ops,
            bridge=bridge, editor_id=_deps.actor(request), ladder=_deps.store(request).ladder_config(),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "correction_id": result.correction_id,
        "n_spots": result.n_spots,
        "ladder_tier": result.ladder_tier,
        "overall_quality": result.overall_quality,
    }


class RevertBody(BaseModel):
    correction_id: str


@router.post("/images/{image_id}/extraction/revert")
def revert_extraction(request: Request, image_id: str, body: RevertBody) -> dict:
    db = _deps.db(request)
    settings_obj = _deps.settings(request)
    bridge = _deps.bridges(request).editor
    try:
        result = extraction_editor.revert_extraction(
            db, settings_obj.contours_db_path, image_id=image_id,
            correction_id=body.correction_id, bridge=bridge, editor_id=_deps.actor(request),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"n_spots": result.n_spots, "ladder_tier": result.ladder_tier}


class CorrespondenceBody(BaseModel):
    query_image_id: str
    query_spot_id: int
    other_image_id: str
    other_spot_id: int
    verdict: str


@router.post("/correspondence-labels")
def add_correspondence_label(request: Request, body: CorrespondenceBody) -> dict:
    try:
        cid = extraction_editor.label_correspondence(
            _deps.db(request), query_image_id=body.query_image_id, query_spot_id=body.query_spot_id,
            other_image_id=body.other_image_id, other_spot_id=body.other_spot_id,
            verdict=body.verdict, editor=_deps.actor(request),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"id": cid}
