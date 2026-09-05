"""Extraction batches, cost estimate and the daily LLM budget (spec §7.1, §10.2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ..config import Settings
from ..db import Database
from ..pipeline_bridge import ExtractionBridge
from ..settings_store import SettingsStore
from .ingest import ingest_photo
from .notify import low_token_block, raise_notification

# billed LLM calls per photo through the full pipeline (spec §7.1: screen +
# magenta repaint + body fill + anatomy + judge). Tunable estimate.
CALLS_PER_PHOTO = 4


@dataclass
class CostEstimate:
    photos: int
    billed_calls_estimate: int
    budget_remaining: int | None    # None = unlimited
    within_budget: bool


def estimate_cost(db: Database, settings: SettingsStore, *, photos: int) -> CostEstimate:
    est = photos * CALLS_PER_PHOTO
    budget = int(settings.get("daily_llm_budget"))
    if budget <= 0:
        return CostEstimate(photos, est, None, True)
    used = used_today(db, "extraction")
    remaining = max(0, budget - used)
    return CostEstimate(photos, est, remaining, est <= remaining)


def used_today(db: Database, kind: str, *, day: date | None = None) -> int:
    day = day or date.today()
    return db.scalar(
        "SELECT coalesce(sum(calls), 0) FROM llm_usage WHERE day = ? AND kind = ?", [day, kind]
    ) or 0


def record_usage(db: Database, kind: str, calls: int, *, day: date | None = None) -> None:
    day = day or date.today()
    db.execute(
        "INSERT INTO llm_usage (day, kind, calls) VALUES (?, ?, ?) "
        "ON CONFLICT (day, kind) DO UPDATE SET calls = llm_usage.calls + excluded.calls",
        [day, kind, calls],
    )


def check_budget_and_warn(db: Database, settings: SettingsStore, *, reviewer: str | None = None) -> None:
    budget = int(settings.get("daily_llm_budget"))
    if budget <= 0:
        return
    remaining = max(0, budget - used_today(db, "extraction"))
    warn_at = int(settings.get("llm_low_warn_at"))
    if remaining <= warn_at and settings.get("notify_extraction_llm_low"):
        model = settings.get("active_model") or "extraction model"
        raise_notification(
            db, type="extraction_llm_low", severity="warning",
            title="Extraction LLM low on tokens",
            body=f"~{remaining} calls left today.",
            copy_text=low_token_block(str(model), remaining, "midnight UTC", reviewer),
        )


@dataclass
class BatchExtractionResult:
    processed: int
    tiers: dict
    failures: int


def extract_batch(
    db: Database,
    settings_obj: Settings,
    store: SettingsStore,
    *,
    photo_paths: list,
    bridge: ExtractionBridge,
    batch_id: str | None = None,
    reviewer: str | None = None,
    progress=None,
) -> BatchExtractionResult:
    ladder = store.ladder_config()
    tiers: dict[str, int] = {}
    failures = 0
    total = len(photo_paths)
    for i, path in enumerate(photo_paths):
        res = ingest_photo(
            db, settings_obj, path, bridge=bridge, batch_id=batch_id,
            actor=reviewer, ladder=ladder,
        )
        record_usage(db, "extraction", CALLS_PER_PHOTO)
        key = res.ladder_tier or res.reason or "failed"
        tiers[key] = tiers.get(key, 0) + 1
        if res.extraction_failed:
            failures += 1
        if progress:
            progress((i + 1) / total, f"{i + 1}/{total}")
    check_budget_and_warn(db, store, reviewer=reviewer)
    return BatchExtractionResult(processed=total, tiers=tiers, failures=failures)
