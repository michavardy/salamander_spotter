from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from pydantic import BaseModel

from ...services import batches as batch_svc
from ...services import extraction, ingest
from . import _deps

router = APIRouter()


class EstimateRequest(BaseModel):
    count: int


@router.post("/uploads/estimate")
def estimate(request: Request, body: EstimateRequest) -> dict:
    est = extraction.estimate_cost(_deps.db(request), _deps.store(request), photos=body.count)
    return {
        "photos": est.photos,
        "billed_calls_estimate": est.billed_calls_estimate,
        "budget_remaining": est.budget_remaining,
        "within_budget": est.within_budget,
    }


@router.post("/uploads")
async def upload(request: Request, file: UploadFile) -> dict:
    """Incremental ingest of one photo (spec §7.9 B)."""
    settings_obj = _deps.settings(request)
    db = _deps.db(request)
    store = _deps.store(request)
    bridge = _deps.bridges(request).extraction

    suffix = Path(file.filename or "photo.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        result = ingest.ingest_photo(
            db, settings_obj, tmp_path, bridge=bridge,
            source_filename=file.filename, actor=_deps.actor(request),
            ladder=store.ladder_config(),
        )
        extraction.record_usage(db, "extraction", extraction.CALLS_PER_PHOTO)
        extraction.check_budget_and_warn(db, store, reviewer=_deps.actor(request))
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "image_id": result.image_id,
        "status": result.status,
        "ladder_tier": result.ladder_tier,
        "n_spots": result.n_spots,
        "reason": result.reason,
    }


@router.post("/images/{image_id}/re-extract")
def re_extract(request: Request, image_id: str) -> dict:
    """Re-extract a single photo (spec §10.2) — overwrites only that photo's rows."""
    db = _deps.db(request)
    settings_obj = _deps.settings(request)
    img = db.query_one("SELECT * FROM images WHERE image_id = ?", [image_id])
    if not img:
        raise HTTPException(404, "image not found")
    raw = next(settings_obj.raw_images_dir.glob(f"{image_id}.*"), None)
    if raw is None:
        raise HTTPException(409, "no raw file on disk for this image")
    bridge = _deps.bridges(request).extraction
    try:
        res = bridge.extract(image_id, raw, settings_obj.contours_db_path)
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
    store = _deps.store(request)
    from ...services.quality_ladder import ladder_tier

    tier = ladder_tier(
        overall_quality=res.overall_quality, has_axis=res.has_axis, n_spots=res.n_spots,
        extraction_failed=res.failed, config=store.ladder_config(),
    )
    db.execute(
        "UPDATE images SET n_spots = ?, ladder_tier = ?, q_overall = ?, status = 'extracted', "
        "rev = rev + 1 WHERE image_id = ?",
        [res.n_spots, tier, res.overall_quality, image_id],
    )
    extraction.record_usage(db, "extraction", extraction.CALLS_PER_PHOTO)
    return {"image_id": image_id, "ladder_tier": tier, "n_spots": res.n_spots}


@router.get("/uploads/batch/{batch_id}")
def batch_view(request: Request, batch_id: str) -> dict:
    db = _deps.db(request)
    counts = batch_svc.batch_counts(db, batch_id)
    cards = db.query_dicts(
        "SELECT image_id, ladder_tier, status, q_overall, q_blur, q_lighting, q_spot, q_body, "
        "n_spots, use_anyway FROM images WHERE review_batch_id = ? ORDER BY created_at DESC",
        [batch_id],
    )
    return {"batch_id": batch_id, "counts": counts, "cards": cards}
