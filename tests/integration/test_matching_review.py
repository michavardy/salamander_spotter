"""Spec §7.3 / §9.1 — matching, triage, the decision engine."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.pipeline_bridge import FakeExtractionBridge
from app.pipeline_bridge.matching import FakeMatchingBridge
from app.services import decision as decision_svc
from app.services import ingest, matching
from app.settings_store import SettingsStore


@pytest.fixture
def photo(tmp_path):
    p = tmp_path / "q.jpg"
    Image.new("RGB", (200, 260), (30, 80, 40)).save(p)
    return p


@pytest.fixture
def enrolled(db, settings, dataset_dir):
    ingest.transfer_dataset(db, settings, dataset_dir)
    return db


def _new_sighting(db, settings, photo):
    return ingest.ingest_photo(db, settings, photo, bridge=FakeExtractionBridge()).image_id


def test_match_ranks_gallery_and_persists(enrolled, settings, photo):
    store = SettingsStore(enrolled)
    image_id = _new_sighting(enrolled, settings, photo)
    bridge = FakeMatchingBridge(scores={(image_id, "aa_1"): 0.88, (image_id, "bb_1"): 0.30})
    match_id = matching.run_match(
        enrolled, store, query_image_id=image_id, bridge=bridge,
        contours_db_path=settings.contours_db_path,
    )
    cands = matching.load_candidates(enrolled, match_id)
    assert cands[0]["individual_id"] == "aa_1"
    assert enrolled.query_one("SELECT status FROM images WHERE image_id = ?", [image_id])["status"] == "matched"


def test_auto_approve_above_threshold(enrolled, settings, photo):
    store = SettingsStore(enrolled)
    store.set("auto_approve_threshold", 0.8)
    image_id = _new_sighting(enrolled, settings, photo)
    bridge = FakeMatchingBridge(scores={(image_id, "aa_1"): 0.99})
    match_id = matching.run_match(enrolled, store, query_image_id=image_id, bridge=bridge,
                                  contours_db_path=settings.contours_db_path)
    triage = decision_svc.triage(enrolled, store, image_id=image_id, match_id=match_id)
    assert triage.outcome == "auto_confirmed"
    row = enrolled.query_one("SELECT * FROM images WHERE image_id = ?", [image_id])
    assert row["status"] == "confirmed" and row["individual_id"] == "aa_1"
    # a review_decision was logged even for the auto path
    assert enrolled.scalar("SELECT count(*) FROM review_decisions WHERE image_id = ?", [image_id]) == 1


def test_triage_queues_below_threshold(enrolled, settings, photo):
    store = SettingsStore(enrolled)
    image_id = _new_sighting(enrolled, settings, photo)
    bridge = FakeMatchingBridge(scores={(image_id, "aa_1"): 0.60, (image_id, "bb_1"): 0.20})
    match_id = matching.run_match(enrolled, store, query_image_id=image_id, bridge=bridge,
                                  contours_db_path=settings.contours_db_path)
    triage = decision_svc.triage(enrolled, store, image_id=image_id, match_id=match_id)
    assert triage.outcome == "queued"
    assert triage.suggestion_individual_id == "aa_1"
    assert enrolled.query_one("SELECT status FROM images WHERE image_id=?", [image_id])["status"] == "in_review"


def test_confirm_decision_relabels_and_updates_individual(enrolled, settings, photo):
    store = SettingsStore(enrolled)
    image_id = _new_sighting(enrolled, settings, photo)
    bridge = FakeMatchingBridge(scores={(image_id, "aa_1"): 0.62})
    matching.run_match(enrolled, store, query_image_id=image_id, bridge=bridge,
                       contours_db_path=settings.contours_db_path)
    out = decision_svc.apply_decision(
        enrolled, store,
        decision_svc.DecisionInput(image_id, "confirm", chosen_individual_id="aa_1", reviewer_id="dana"),
    )
    assert out.individual_id == "aa_1"
    assert enrolled.query_one("SELECT status,individual_id FROM images WHERE image_id=?", [image_id])["status"] == "confirmed"


def test_override_guard_then_confirm(enrolled, settings, photo):
    store = SettingsStore(enrolled)
    image_id = _new_sighting(enrolled, settings, photo)
    bridge = FakeMatchingBridge(scores={(image_id, "aa_1"): 0.70, (image_id, "bb_1"): 0.30})
    matching.run_match(enrolled, store, query_image_id=image_id, bridge=bridge,
                       contours_db_path=settings.contours_db_path)
    # picking bb_1 over the suggested aa_1 triggers the guard
    first = decision_svc.apply_decision(
        enrolled, store, decision_svc.DecisionInput(image_id, "confirm", chosen_individual_id="bb_1"),
    )
    assert first.override_warning and first.decision_id == ""
    # acknowledging goes through and records the override
    second = decision_svc.apply_decision(
        enrolled, store,
        decision_svc.DecisionInput(image_id, "confirm", chosen_individual_id="bb_1", confirm_override=True),
    )
    assert second.was_override is True
    d = enrolled.query_one("SELECT was_override FROM review_decisions WHERE image_id=?", [image_id])
    assert d["was_override"] is True


def test_new_individual_enrollment_allocates_code(enrolled, settings, photo):
    store = SettingsStore(enrolled)
    image_id = _new_sighting(enrolled, settings, photo)
    bridge = FakeMatchingBridge(scores={(image_id, "aa_1"): 0.10})
    matching.run_match(enrolled, store, query_image_id=image_id, bridge=bridge,
                       contours_db_path=settings.contours_db_path)
    out = decision_svc.apply_decision(
        enrolled, store, decision_svc.DecisionInput(image_id, "new", reviewer_id="dana"),
    )
    assert out.new_individual_id is not None
    ind = enrolled.query_one("SELECT * FROM individuals WHERE individual_id = ?", [out.new_individual_id])
    assert ind["status"] == "provisional"
    assert enrolled.query_one("SELECT status FROM images WHERE image_id=?", [image_id])["status"] == "enrolled_new"


def test_disqualify_requires_reason(enrolled, settings, photo):
    store = SettingsStore(enrolled)
    image_id = _new_sighting(enrolled, settings, photo)
    with pytest.raises(ValueError):
        decision_svc.apply_decision(enrolled, store, decision_svc.DecisionInput(image_id, "disqualify"))
    decision_svc.apply_decision(
        enrolled, store,
        decision_svc.DecisionInput(image_id, "disqualify", reason_chips=["possible duplicate frame"]),
    )
    assert enrolled.query_one("SELECT status FROM images WHERE image_id=?", [image_id])["status"] == "disqualified"


def test_uncertain_flag(enrolled, settings, photo):
    store = SettingsStore(enrolled)
    image_id = _new_sighting(enrolled, settings, photo)
    decision_svc.apply_decision(enrolled, store, decision_svc.DecisionInput(image_id, "uncertain"))
    assert enrolled.query_one("SELECT status FROM images WHERE image_id=?", [image_id])["status"] == "flagged_uncertain"
