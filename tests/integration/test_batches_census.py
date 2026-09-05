"""Spec §9.2 — batches, publish/unpublish, the published census."""

from __future__ import annotations

import pytest
from PIL import Image

from app.pipeline_bridge import FakeExtractionBridge
from app.pipeline_bridge.matching import FakeMatchingBridge
from app.services import batches, decision as decision_svc, ingest, matching
from app.settings_store import SettingsStore


@pytest.fixture
def photo(tmp_path):
    p = tmp_path / "q.jpg"
    Image.new("RGB", (200, 260), (30, 80, 40)).save(p)
    return p


def test_one_open_batch_at_a_time(db):
    b1 = batches.create_batch(db, "Spring")
    b2 = batches.create_batch(db, "Autumn")
    statuses = {r["id"]: r["status"] for r in batches.open_batches(db)}
    assert statuses[b1] == "closed"
    assert statuses[b2] == "open"


def test_publish_promotes_provisional_and_counts_census(db, settings, dataset_dir, photo):
    ingest.transfer_dataset(db, settings, dataset_dir)
    store = SettingsStore(db)
    bid = batches.create_batch(db, "Autumn 2026")

    # a new sighting, enrolled as a new provisional individual
    image_id = ingest.ingest_photo(db, settings, photo, bridge=FakeExtractionBridge(), batch_id=bid).image_id
    matching.run_match(db, store, query_image_id=image_id,
                       bridge=FakeMatchingBridge(scores={(image_id, "aa_1"): 0.1}),
                       contours_db_path=settings.contours_db_path)
    out = decision_svc.apply_decision(db, store, decision_svc.DecisionInput(image_id, "new"))
    new_id = out.new_individual_id
    assert db.query_one("SELECT status FROM individuals WHERE individual_id=?", [new_id])["status"] == "provisional"

    result = batches.publish_batch(db, bid)
    assert db.query_one("SELECT status FROM individuals WHERE individual_id=?", [new_id])["status"] == "published"
    assert db.query_one("SELECT status FROM review_batches WHERE id=?", [bid])["status"] == "published"

    c = batches.census(db)
    # imported batch (7 imgs, dd excluded) + this one new enrolled sighting
    assert c["confirmed_sightings"] >= 1
    assert c["published_individuals"] >= 1


def test_publish_does_not_require_all_decided(db, settings, dataset_dir, photo):
    ingest.transfer_dataset(db, settings, dataset_dir)
    bid = batches.create_batch(db, "b")
    ingest.ingest_photo(db, settings, photo, bridge=FakeExtractionBridge(), batch_id=bid)
    counts = batches.batch_counts(db, bid)
    assert counts["undecided"] == 1
    res = batches.publish_batch(db, bid)  # succeeds despite the undecided sighting
    assert res["batch_id"] == bid


def test_unpublish_reopens(db):
    bid = batches.create_batch(db, "b")
    batches.publish_batch(db, bid)
    batches.unpublish_batch(db, bid)
    assert db.query_one("SELECT status FROM review_batches WHERE id=?", [bid])["status"] == "open"


def test_uncertain_sighting_holds_individual_back(db, settings, dataset_dir, photo, tmp_path):
    ingest.transfer_dataset(db, settings, dataset_dir)
    store = SettingsStore(db)
    bid = batches.create_batch(db, "b")

    p2 = tmp_path / "q2.jpg"
    Image.new("RGB", (200, 260), (1, 2, 3)).save(p2)
    a = ingest.ingest_photo(db, settings, photo, bridge=FakeExtractionBridge(), batch_id=bid).image_id
    b = ingest.ingest_photo(db, settings, p2, bridge=FakeExtractionBridge(), batch_id=bid).image_id
    for iid in (a, b):
        matching.run_match(db, store, query_image_id=iid,
                           bridge=FakeMatchingBridge(scores={(iid, "aa_1"): 0.1}),
                           contours_db_path=settings.contours_db_path)
    out = decision_svc.apply_decision(db, store, decision_svc.DecisionInput(a, "new"))
    new_id = out.new_individual_id
    # second photo of that same new individual, but flagged uncertain
    db.execute("UPDATE images SET individual_id = ? WHERE image_id = ?", [new_id, b])
    decision_svc.apply_decision(db, store, decision_svc.DecisionInput(b, "uncertain"))

    batches.publish_batch(db, bid)
    assert db.query_one("SELECT status FROM individuals WHERE individual_id=?", [new_id])["status"] == "provisional"
