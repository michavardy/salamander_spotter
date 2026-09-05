"""Training-run orchestration + scheduler (spec §7.5, §7.8)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..db import Database, new_id
from ..pipeline_bridge.training import TrainingBridge
from ..settings_store import SettingsStore
from . import models as model_svc
from .notify import raise_notification


def snapshot_dataset(db: Database, *, note: str | None = None) -> str:
    """Freeze the current reviewed dataset — all confirmed images + kept synthetic
    views (spec §7.5 step 1). New individuals default to train (§7.8)."""
    rows = db.query(
        "SELECT image_id, individual_id, is_synthetic FROM images "
        "WHERE status IN ('confirmed','enrolled_new') OR is_synthetic "
        "ORDER BY image_id"
    )
    n_images = len(rows)
    n_individuals = len({r[1] for r in rows if r[1]})
    checksum = hashlib.sha256(
        json.dumps(rows, default=str, sort_keys=True).encode()
    ).hexdigest()[:16]
    snap_id = new_id("snap")
    db.insert(
        "dataset_snapshots",
        {"id": snap_id, "n_images": n_images, "n_individuals": n_individuals,
         "checksum": checksum, "note": note},
    )
    return snap_id


@dataclass
class TrainingRunResult:
    run_id: str
    snapshot_id: str
    models: list[dict]
    promotion: dict | None


def run_training(
    db: Database,
    settings: SettingsStore,
    *,
    bridge: TrainingBridge,
    models_dir: Path,
    contours_db_path: Path,
    trigger: str = "manual",
    actor: str | None = None,
    progress=None,
    job_id: str | None = None,
    log_path: Path | None = None,
) -> TrainingRunResult:
    run_id = new_id("run")
    db.insert("training_runs", {
        "id": run_id, "trigger": trigger, "status": "running",
        "job_id": job_id, "log_path": str(log_path) if log_path else None,
    })
    snapshot_id = snapshot_dataset(db, note=f"training run {run_id}")
    if progress:
        progress(0.0, "dataset snapshot taken")

    extra_env = {}
    if settings.get("log_dir"):
        extra_env["SPOTTER_LOG_DIR"] = str(settings.get("log_dir"))
    if settings.get("log_level"):
        extra_env["SPOTTER_LOG_LEVEL"] = str(settings.get("log_level"))

    relay_path = None
    relay_interval_s = None
    if log_path is not None and settings.get("remote_logger_enabled"):
        relay_dir = Path(log_path).parent / "remote_relay"
        relay_path = relay_dir / f"{job_id or run_id}.jsonl"
        relay_interval_s = max(1, int(settings.get("remote_logger_interval_minutes"))) * 60
        # "latest" pointer so a phone/Claude-Code watcher doesn't need to know the job id up
        # front — it just follows whichever run is currently active.
        relay_dir.mkdir(parents=True, exist_ok=True)
        (relay_dir / "latest.json").write_text(json.dumps({
            "run_id": run_id, "job_id": job_id, "relay_path": str(relay_path),
            "log_path": str(log_path), "started_at": datetime.now(timezone.utc).isoformat(),
        }), encoding="utf-8")

    trained = bridge.train_and_eval(
        snapshot_id, Path(models_dir), Path(contours_db_path),
        log_path=log_path, progress=progress, extra_env=extra_env,
        relay_path=relay_path, relay_interval_s=relay_interval_s,
    )

    coeffs = settings.get("score_coefficients")
    best_before = model_svc.active_model(db)
    rows_out = []
    with db.transaction():
        for tm in trained:
            score = model_svc.register_model(
                db, name=tm.name, kind=tm.kind, metrics=tm.metrics, coefficients=coeffs,
                dataset_snapshot_id=snapshot_id, weights_path=tm.weights_path,
                calibration_path=tm.calibration_path, status="candidate",
            )
            db.insert(
                "training_run_models",
                {"run_id": run_id, "model_name": tm.name, "score": score, **{
                    k: tm.metrics.get(k) for k in
                    ("r1", "r5", "r10", "bal_acc", "novelty_auroc", "review_at_90")
                }},
            )
            rows_out.append({"name": tm.name, "kind": tm.kind, "score": score, **tm.metrics})
        model_svc.recompute_best(db)

    promotion = None
    active = model_svc.active_model(db)
    best_candidate = db.query_one(
        "SELECT name, score FROM models WHERE status IN ('candidate','best') "
        "ORDER BY score DESC NULLS LAST LIMIT 1"
    )
    beats = (
        best_candidate
        and (active is None or (best_candidate["score"] or 0) > (active["score"] or 0))
    )
    if beats and settings.get("auto_promote"):
        promotion = model_svc.promote(db, best_candidate["name"], actor=actor, auto=True)
        if settings.get("notify_model_auto_promoted"):
            raise_notification(
                db, type="model_auto_promoted", severity="info",
                title=f"Model auto-promoted to {best_candidate['name']}",
                body=f"Score {best_candidate['score']:.3f} beat the previous active model.",
                action_label="Review in Models", action_href="/models",
            )

    with db.transaction():
        db.execute(
            "UPDATE training_runs SET finished_at = now(), status = 'done', "
            "promotion_from = ?, promotion_to = ?, promotion_auto = ? WHERE id = ?",
            [
                best_before["name"] if best_before else None,
                promotion["active"] if promotion else None,
                bool(promotion), run_id,
            ],
        )
        db.log_activity(kind="training", summary=f"Training run finished ({len(trained)} models)",
                        actor=actor, ref_type="training_run", ref_id=run_id)
    if settings.get("notify_training_complete"):
        raise_notification(
            db, type="training_complete", severity="info",
            title="Training complete",
            body=f"{len(trained)} models evaluated"
                 + (f"; promoted {promotion['active']}" if promotion else "; production unchanged"),
            action_label="Review in Models", action_href="/models",
        )
    if progress:
        progress(1.0, "done")
    return TrainingRunResult(run_id=run_id, snapshot_id=snapshot_id, models=rows_out, promotion=promotion)


def training_due(db: Database, settings: SettingsStore, *, now: datetime | None = None) -> dict:
    """Whether the scheduler should kick off a run (spec §7.5 step, §10.11)."""
    now = now or datetime.now(timezone.utc)
    last = db.query_one("SELECT started_at FROM training_runs ORDER BY started_at DESC LIMIT 1")
    every_days = int(settings.get("retrain_every_days"))
    after_images = int(settings.get("retrain_after_images"))

    due_by_time = False
    if every_days > 0:
        if last is None:
            due_by_time = True
        else:
            started = last["started_at"]
            if isinstance(started, str):
                started = datetime.fromisoformat(started)
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            due_by_time = now - started >= timedelta(days=every_days)

    since = db.scalar(
        "SELECT count(*) FROM images WHERE status IN ('confirmed','enrolled_new') "
        "AND created_at > coalesce((SELECT max(started_at) FROM training_runs), '1970-01-01')"
    )
    due_by_images = after_images > 0 and (since or 0) >= after_images
    return {"due": bool(due_by_time or due_by_images), "by_time": due_by_time,
            "by_images": due_by_images, "new_images_since_last": since or 0}
