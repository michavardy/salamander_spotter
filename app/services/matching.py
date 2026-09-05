"""Run the active matcher for a sighting and build the ranked candidate list (spec §7.3).

* similarity -> calibrated confidence (per-model calibrator; identity fallback)
* Top-N = smallest N whose confidence mass covers ``coverage_target``, capped at
  ``max_candidates``
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from ..db import Database, new_id
from ..pipeline_bridge.matching import MatchingBridge, PairScore
from ..settings_store import SettingsStore


@dataclass(frozen=True)
class Calibrator:
    """Platt-style ``sigmoid(a * s + b)``; identity when ``a is None``."""

    a: float | None = None
    b: float = 0.0

    @classmethod
    def from_json(cls, blob: str | dict | None) -> "Calibrator":
        if not blob:
            return cls()
        data = json.loads(blob) if isinstance(blob, str) else blob
        return cls(a=data.get("a"), b=data.get("b", 0.0))

    def __call__(self, similarity: float) -> float:
        if self.a is None:
            return max(0.0, min(1.0, similarity))
        return 1.0 / (1.0 + math.exp(-(self.a * similarity + self.b)))


@dataclass(frozen=True)
class Candidate:
    rank: int
    individual_id: str
    best_photo_id: str
    similarity: float
    calibrated_confidence: float


def top_n_by_coverage(
    confidences: list[float], *, coverage_target: float, max_candidates: int
) -> int:
    """Smallest N whose normalised confidence mass covers the target (spec §7.3)."""
    if not confidences:
        return 0
    total = sum(confidences) or 1.0
    acc = 0.0
    for i, c in enumerate(sorted(confidences, reverse=True), start=1):
        acc += c / total
        if acc + 1e-9 >= coverage_target or i >= max_candidates:
            return min(i, max_candidates)
    return min(len(confidences), max_candidates)


def rank_candidates(
    pairs: list[PairScore], calibrator: Calibrator, settings: SettingsStore
) -> list[Candidate]:
    thr = settings.decision_thresholds()
    scored = sorted(
        ((p, calibrator(p.similarity)) for p in pairs),
        key=lambda t: t[1],
        reverse=True,
    )
    n = top_n_by_coverage(
        [c for _, c in scored],
        coverage_target=thr.coverage_target,
        max_candidates=thr.max_candidates,
    )
    return [
        Candidate(
            rank=i + 1,
            individual_id=p.individual_id,
            best_photo_id=p.best_photo_id,
            similarity=p.similarity,
            calibrated_confidence=round(conf, 4),
        )
        for i, (p, conf) in enumerate(scored[:n])
    ]


def gallery_individuals(db: Database) -> list[str]:
    """Enrolled individuals eligible to match against (confirmed/published, not merged)."""
    return [
        r[0]
        for r in db.query(
            "SELECT individual_id FROM individuals "
            "WHERE merged_into IS NULL AND status IN ('provisional','confirmed','published')"
        )
    ]


def run_match(
    db: Database,
    settings: SettingsStore,
    *,
    query_image_id: str,
    bridge: MatchingBridge,
    contours_db_path,
    calibration: str | dict | None = None,
) -> str:
    """Score ``query_image_id`` against the gallery, persist a ``match_results`` row
    with its ranked candidates, and return the match_result id."""
    gallery = [g for g in gallery_individuals(db) if g != _individual_of(db, query_image_id)]
    pairs = bridge.score(query_image_id, gallery, contours_db_path)
    calibrator = Calibrator.from_json(calibration)
    candidates = rank_candidates(pairs, calibrator, settings)

    model_name = getattr(bridge, "model_name", settings.get("active_model") or "active")
    thr = settings.decision_thresholds()
    match_id = new_id("mr")
    with db.transaction():
        db.insert(
            "match_results",
            {
                "id": match_id,
                "query_image_id": query_image_id,
                "model_name": model_name,
                "coverage_target": thr.coverage_target,
                "params_json": json.dumps({"max_candidates": thr.max_candidates}),
            },
        )
        for c in candidates:
            db.insert(
                "match_candidates",
                {
                    "match_result_id": match_id,
                    "rank": c.rank,
                    "individual_id": c.individual_id,
                    "best_photo_id": c.best_photo_id,
                    "similarity": c.similarity,
                    "calibrated_confidence": c.calibrated_confidence,
                },
            )
        db.execute(
            "UPDATE images SET status = 'matched', rev = rev + 1 "
            "WHERE image_id = ? AND status NOT IN ('confirmed','disqualified','enrolled_new')",
            [query_image_id],
        )
    return match_id


def load_candidates(db: Database, match_id: str) -> list[dict]:
    return db.query_dicts(
        "SELECT rank, individual_id, best_photo_id, similarity, calibrated_confidence "
        "FROM match_candidates WHERE match_result_id = ? ORDER BY rank",
        [match_id],
    )


def latest_match_for(db: Database, image_id: str) -> dict | None:
    return db.query_one(
        "SELECT * FROM match_results WHERE query_image_id = ? ORDER BY created_at DESC LIMIT 1",
        [image_id],
    )


def _individual_of(db: Database, image_id: str) -> str | None:
    row = db.query_one("SELECT individual_id FROM images WHERE image_id = ?", [image_id])
    return row["individual_id"] if row else None
