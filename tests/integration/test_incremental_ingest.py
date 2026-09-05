"""Spec §7.9 B — a new photo is extracted individually and appended; nothing
already in either database is touched."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from PIL import Image

from app.pipeline_bridge import ExtractionResult, FakeExtractionBridge
from app.services import ingest


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    p = tmp_path / "field_photo.jpg"
    Image.new("RGB", (300, 400), (40, 90, 50)).save(p)
    return p


def test_ingest_photo_appends_one_row(db, settings, dataset_dir, photo):
    ingest.transfer_dataset(db, settings, dataset_dir)
    before_images = db.scalar("SELECT count(*) FROM images")
    before_individuals = db.scalar("SELECT count(*) FROM individuals")
    before_contours = _contour_ids(settings.contours_db_path)

    bridge = FakeExtractionBridge()
    result = ingest.ingest_photo(db, settings, photo, bridge=bridge)

    assert result.image_id.startswith("up_")
    assert result.status == "extracted"
    assert bridge.calls == [result.image_id]

    # exactly one new images row, no new individual (undecided sighting)
    assert db.scalar("SELECT count(*) FROM images") == before_images + 1
    assert db.scalar("SELECT count(*) FROM individuals") == before_individuals
    row = db.query_one("SELECT * FROM images WHERE image_id = ?", [result.image_id])
    assert row["individual_id"] is None
    assert row["origin"] == "upload"
    assert row["review_batch_id"] is not None

    # existing contours rows untouched; the new photo's rows were appended
    after_contours = _contour_ids(settings.contours_db_path)
    assert before_contours < after_contours
    assert after_contours - before_contours == {result.image_id}

    # raw + thumb written
    assert list(settings.raw_images_dir.glob(f"{result.image_id}.*"))
    assert (settings.thumb_images_dir / f"{result.image_id}.webp").exists()


def test_ingest_photo_derives_tier(db, settings, photo):
    class LowBridge(FakeExtractionBridge):
        def result_for(self, image_id):
            return ExtractionResult(
                image_id=image_id, n_spots=9, has_axis=True, overall_quality=0.20,
                quality={"blur": 0.1},
            )

    result = ingest.ingest_photo(db, settings, photo, bridge=LowBridge())
    assert result.ladder_tier == "hand_correction"


def test_multi_animal_frame_is_disqualified(db, settings, photo):
    class RejectBridge(FakeExtractionBridge):
        def result_for(self, image_id):
            return ExtractionResult(
                image_id=image_id, n_spots=0, has_axis=False, overall_quality=None,
                failed=True, reason="multi-animal frame",
            )

    result = ingest.ingest_photo(db, settings, photo, bridge=RejectBridge())
    assert result.status == "disqualified"
    assert result.reason == "multi-animal frame"
    row = db.query_one("SELECT * FROM images WHERE image_id = ?", [result.image_id])
    assert row["status"] == "disqualified"
    assert row["ladder_tier"] is None


def test_ingest_photo_does_not_retrigger_pipeline_for_existing(db, settings, dataset_dir, photo):
    """Adding a datapoint must not re-extract anything already present."""
    ingest.transfer_dataset(db, settings, dataset_dir)
    bridge = FakeExtractionBridge()
    ingest.ingest_photo(db, settings, photo, bridge=bridge)
    ingest.ingest_photo(db, settings, photo, bridge=bridge)
    # extraction ran once per *new* photo only — never for the 7 imported ones
    assert len(bridge.calls) == 2
    assert all(c.startswith("up_") for c in bridge.calls)


def test_ingest_photo_attaches_to_open_batch(db, settings, photo):
    b1 = ingest.open_batch(db, "Autumn 2099")
    r = ingest.ingest_photo(db, settings, photo, bridge=FakeExtractionBridge())
    assert db.query_one("SELECT review_batch_id FROM images WHERE image_id = ?", [r.image_id])[
        "review_batch_id"
    ] == b1


def _contour_ids(path: Path) -> set[str]:
    con = duckdb.connect(str(path), read_only=True)
    try:
        return {r[0] for r in con.execute("SELECT salamander_id FROM images").fetchall()}
    finally:
        con.close()
