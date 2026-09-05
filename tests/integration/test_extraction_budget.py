"""Spec §7.1 / §10.2 — cost estimate, daily LLM budget, low-token notification."""

from __future__ import annotations

from datetime import date

from app.services import extraction
from app.settings_store import SettingsStore


def test_estimate_unlimited_when_no_budget(db):
    store = SettingsStore(db)
    store.set("daily_llm_budget", 0)
    est = extraction.estimate_cost(db, store, photos=10)
    assert est.billed_calls_estimate == 10 * extraction.CALLS_PER_PHOTO
    assert est.budget_remaining is None and est.within_budget


def test_estimate_respects_budget_and_usage(db):
    store = SettingsStore(db)
    store.set("daily_llm_budget", 100)
    extraction.record_usage(db, "extraction", 80)
    est = extraction.estimate_cost(db, store, photos=10)  # 40 calls, only 20 left
    assert est.budget_remaining == 20
    assert est.within_budget is False


def test_low_token_notification_raised(db):
    store = SettingsStore(db)
    store.update({"daily_llm_budget": 100, "llm_low_warn_at": 30})
    extraction.record_usage(db, "extraction", 80)  # 20 left, below 30
    extraction.check_budget_and_warn(db, store, reviewer="dana")
    n = db.query_one("SELECT * FROM notifications WHERE type = 'extraction_llm_low'")
    assert n is not None
    assert "calls left" in n["copy_text"] and "dana" in n["copy_text"]


def test_usage_accumulates_per_day(db):
    extraction.record_usage(db, "extraction", 5)
    extraction.record_usage(db, "extraction", 7)
    assert extraction.used_today(db, "extraction") == 12
