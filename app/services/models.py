"""Model registry, the Score formula, promotion / rollback (spec §7.4, §7.5)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from ..db import Database
from ..settings_store import SettingsStore

# spec §7.4 — each metric has a FIXED letter; Score is a plain linear combination
METRIC_LETTERS = {
    "a": "r1",
    "b": "r5",
    "c": "r10",
    "d": "bal_acc",
    "e": "novelty_auroc",
    "f": "review_at_90",
}

STATUS_ACTIVE = "active"
STATUS_CANDIDATE = "candidate"
STATUS_BEST = "best"
STATUS_ARCHIVED = "archived"


def compute_score(metrics: dict, coefficients: dict[str, float]) -> float:
    """``Score = Σ coef·metric`` over the lettered metrics. No expression parser —
    a malformed coefficient is simply ignored (spec §7.4, §15)."""
    total = 0.0
    for letter, coef in coefficients.items():
        col = METRIC_LETTERS.get(letter)
        if col is None:
            continue
        try:
            value = float(metrics.get(col) or 0.0)
            total += float(coef) * value
        except (TypeError, ValueError):
            continue
    return round(total, 6)


def register_model(
    db: Database,
    *,
    name: str,
    kind: str,
    metrics: dict,
    coefficients: dict[str, float],
    dataset_snapshot_id: str | None = None,
    weights_path: str | None = None,
    calibration_path: str | None = None,
    status: str = STATUS_CANDIDATE,
    notes: str | None = None,
) -> float:
    score = compute_score(metrics, coefficients)
    db.execute(
        """
        INSERT INTO models (name, kind, trained_at, dataset_snapshot_id, r1, r5, r10,
                            bal_acc, novelty_auroc, review_at_90, score, status,
                            weights_path, calibration_path, notes)
        VALUES (?, ?, now(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (name) DO UPDATE SET
            kind = excluded.kind, trained_at = excluded.trained_at,
            dataset_snapshot_id = excluded.dataset_snapshot_id,
            r1 = excluded.r1, r5 = excluded.r5, r10 = excluded.r10,
            bal_acc = excluded.bal_acc, novelty_auroc = excluded.novelty_auroc,
            review_at_90 = excluded.review_at_90, score = excluded.score,
            weights_path = excluded.weights_path, calibration_path = excluded.calibration_path,
            notes = excluded.notes
        """,
        [
            name, kind, dataset_snapshot_id,
            metrics.get("r1"), metrics.get("r5"), metrics.get("r10"),
            metrics.get("bal_acc"), metrics.get("novelty_auroc"), metrics.get("review_at_90"),
            score, status, weights_path, calibration_path, notes,
        ],
    )
    return score


class ModelImportError(ValueError):
    pass


def import_model(
    db: Database,
    settings: Settings,
    *,
    name: str,
    kind: str,
    source_weights_path: str | Path,
    source_calibration_path: str | Path | None = None,
    metrics: dict | None = None,
    coefficients: dict[str, float],
    notes: str | None = None,
    status: str = STATUS_CANDIDATE,
    actor: str | None = None,
) -> dict:
    """Bring an already-trained checkpoint onto the volume and register it (spec
    §7.6: "placed there directly by the maintainer" — copied in, not downloaded,
    not retrained). Zero GPU/LLM cost; just a file copy + a DB row.

    Metrics you already have (from a bakeoff / results.md run) are optional —
    without them the model still catalogues and can be made active, it just
    scores 0 for auto-promotion purposes until you fill them in.
    """
    src = Path(source_weights_path)
    if not src.is_file():
        raise ModelImportError(f"weights file not found: {src}")
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        raise ModelImportError(f"invalid model name: {name!r} (letters, digits, - and _ only)")

    model_dir = settings.models_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_dest = model_dir / f"weights{src.suffix}"
    shutil.copy2(src, weights_dest)

    calibration_dest = None
    if source_calibration_path:
        cal_src = Path(source_calibration_path)
        if not cal_src.is_file():
            raise ModelImportError(f"calibration file not found: {cal_src}")
        calibration_dest = model_dir / "calibration.json"
        shutil.copy2(cal_src, calibration_dest)

    score = register_model(
        db,
        name=name,
        kind=kind,
        metrics=metrics or {},
        coefficients=coefficients,
        weights_path=str(weights_dest),
        calibration_path=str(calibration_dest) if calibration_dest else None,
        status=status,
        notes=notes,
    )
    db.audit(
        action="import_model", entity="model", entity_id=name, actor=actor,
        after={"source_weights_path": str(src), "score": score, "metrics": metrics},
    )
    db.log_activity(
        kind="model", summary=f"Imported model {name} from {src.name}", actor=actor,
        ref_type="model", ref_id=name,
    )
    return {"name": name, "score": score, "weights_path": str(weights_dest),
            "calibration_path": str(calibration_dest) if calibration_dest else None}


def recompute_best(db: Database) -> None:
    """Tag the highest-scoring model *of each kind* as ``best`` (spec §7.5)."""
    db.execute("UPDATE models SET status = 'archived' WHERE status = 'best'")
    for (kind,) in db.query("SELECT DISTINCT kind FROM models"):
        row = db.query_one(
            "SELECT name FROM models WHERE kind = ? AND status != 'active' "
            "ORDER BY score DESC NULLS LAST LIMIT 1",
            [kind],
        )
        if row:
            db.execute("UPDATE models SET status = 'best' WHERE name = ?", [row["name"]])


def active_model(db: Database) -> dict | None:
    return db.query_one("SELECT * FROM models WHERE status = 'active' LIMIT 1")


def promote(db: Database, name: str, *, actor: str | None = None, auto: bool = False) -> dict:
    target = db.query_one("SELECT * FROM models WHERE name = ?", [name])
    if not target:
        raise ValueError(f"unknown model {name}")
    prev = active_model(db)
    with db.transaction():
        if prev:
            db.execute("UPDATE models SET status = 'archived' WHERE name = ?", [prev["name"]])
        db.execute("UPDATE models SET status = 'active' WHERE name = ?", [name])
        db.execute(
            "INSERT INTO settings (key, value_json) VALUES ('active_model', ?) "
            "ON CONFLICT (key) DO UPDATE SET value_json = excluded.value_json",
            [f'"{name}"'],
        )
        db.audit(action="promote_model", entity="model", entity_id=name, actor=actor,
                 before={"from": prev["name"] if prev else None}, after={"to": name, "auto": auto})
        db.log_activity(
            kind="model",
            summary=f"{'Auto-promoted' if auto else 'Promoted'} {name}"
                    + (f" (was {prev['name']})" if prev else ""),
            actor=actor, ref_type="model", ref_id=name,
        )
    return {"active": name, "previous": prev["name"] if prev else None, "auto": auto}


def rollback(db: Database, *, actor: str | None = None) -> dict:
    """One-click restore of the previously-active model (spec §5.3, §7.5)."""
    current = active_model(db)
    last = db.query_one(
        "SELECT entity_id, before_json FROM audit WHERE action = 'promote_model' "
        "ORDER BY at_ts DESC LIMIT 1"
    )
    import json as _json

    prev_name = None
    if last and last["before_json"]:
        prev_name = _json.loads(last["before_json"]).get("from")
    if not prev_name or not db.query_one("SELECT 1 FROM models WHERE name = ?", [prev_name]):
        raise ValueError("no previous model to roll back to")
    return promote(db, prev_name, actor=actor)


def list_models(db: Database) -> list[dict]:
    return db.query_dicts("SELECT * FROM models ORDER BY score DESC NULLS LAST")


def singleton_hint(db: Database) -> dict:
    """How many individuals would benefit from synthetic views (spec §7.7, §7.8)."""
    singletons = db.scalar(
        """
        SELECT count(*) FROM (
            SELECT individual_id FROM images
            WHERE NOT is_synthetic AND status != 'disqualified' AND individual_id IS NOT NULL
            GROUP BY individual_id HAVING count(*) = 1
        )
        """
    )
    return {"singletons": singletons or 0}
