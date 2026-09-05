"""Spec §7.9 A — one-time / delta dataset transfer, no pipeline re-run."""

from __future__ import annotations

import duckdb
import pytest

from app.services import ingest
from app.services.ingest import IMPORTED_BATCH_ID
from tests.fixtures import FixtureImage, build_fixture_dataset


def test_transfer_copies_contours_verbatim_and_derives_rows(db, settings, dataset_dir):
    report = ingest.transfer_dataset(db, settings, dataset_dir)

    # 7 fixture images; dd_1_1 is still *imported* then excluded (a row exists, disqualified)
    assert report.images_added == 7
    assert report.contours_db_copied_verbatim is True
    assert report.llm_calls == 0

    # contours.db is present in the data dir and byte-identical to the source
    src = (dataset_dir / "db" / "contours.db").read_bytes()
    assert settings.contours_db_path.read_bytes() == src

    # app.duckdb rows derived from filenames + joins
    ids = {r[0] for r in db.query("SELECT image_id FROM images")}
    assert {"aa_1_1", "aa_1_2", "aa_1_g0", "bb_1_1", "cc_1_1", "dd_1_1"} <= ids

    individuals = {r[0] for r in db.query("SELECT individual_id FROM individuals")}
    assert {"aa_1", "bb_1", "cc_1"} <= individuals

    # raw images copied, thumbnails generated (the only thing computed)
    assert (settings.raw_images_dir / "aa_1_1.jpg").exists()
    assert (settings.thumb_images_dir / "aa_1_1.webp").exists()
    assert report.thumbnails_written == 7

    # no purple renders were fabricated
    assert not any(settings.purple_images_dir.iterdir())


def test_transfer_derives_ladder_tier_and_quality(db, settings, dataset_dir):
    ingest.transfer_dataset(db, settings, dataset_dir)
    rows = {r["image_id"]: r for r in db.query_dicts("SELECT * FROM images")}
    assert rows["aa_1_1"]["ladder_tier"] == "auto_accept"
    assert rows["aa_1_2"]["ladder_tier"] == "needs_a_look"
    assert rows["bb_1_1"]["ladder_tier"] == "hand_correction"
    assert rows["aa_1_1"]["q_overall"] == pytest.approx(0.80)
    assert rows["aa_1_1"]["origin"] == "import"


def test_synthetic_views_flagged_training_only(db, settings, dataset_dir):
    ingest.transfer_dataset(db, settings, dataset_dir)
    g0 = db.query_one("SELECT * FROM images WHERE image_id = 'aa_1_g0'")
    assert g0["is_synthetic"] is True
    assert g0["individual_id"] == "aa_1"
    # reference image for the individual is a real photo, never the synthetic view
    ind = db.query_one("SELECT * FROM individuals WHERE individual_id = 'aa_1'")
    assert ind["reference_image_id"] == "aa_1_1"


def test_corrections_json_replayed(db, settings, dataset_dir):
    report = ingest.transfer_dataset(db, settings, dataset_dir)
    assert report.merges_applied == 1
    assert report.exclusions_applied == 1

    # merge: cc_2 folded into cc_1
    cc2 = db.query_one("SELECT * FROM individuals WHERE individual_id = 'cc_2'")
    assert cc2["status"] == "merged_into" and cc2["merged_into"] == "cc_1"
    assert db.query_one("SELECT individual_id FROM images WHERE image_id = 'cc_2_1'")[
        "individual_id"
    ] == "cc_1"

    # exclude: dd_1_1 disqualified + an extraction_corrections row kept the history
    assert db.query_one("SELECT status FROM images WHERE image_id = 'dd_1_1'")["status"] == "disqualified"
    assert db.scalar("SELECT count(*) FROM extraction_corrections WHERE image_id = 'dd_1_1'") == 1


def test_imported_batch_is_published(db, settings, dataset_dir):
    ingest.transfer_dataset(db, settings, dataset_dir)
    batch = db.query_one("SELECT * FROM review_batches WHERE id = ?", [IMPORTED_BATCH_ID])
    assert batch["status"] == "published"
    assert db.scalar(
        "SELECT count(*) FROM images WHERE review_batch_id = ?", [IMPORTED_BATCH_ID]
    ) == 7


def test_transfer_is_idempotent(db, settings, dataset_dir):
    first = ingest.transfer_dataset(db, settings, dataset_dir)
    n_images = db.scalar("SELECT count(*) FROM images")
    n_ind = db.scalar("SELECT count(*) FROM individuals")

    second = ingest.transfer_dataset(db, settings, dataset_dir)
    assert second.images_added == 0
    assert second.images_skipped == first.images_added
    assert db.scalar("SELECT count(*) FROM images") == n_images
    assert db.scalar("SELECT count(*) FROM individuals") == n_ind


def test_delta_import_adds_only_new_images(db, settings, tmp_path):
    ds1 = build_fixture_dataset(
        tmp_path / "d1",
        name="all_sasa_norm_2099_01_01",
        images=[FixtureImage("aa_1_1", overall_quality=0.8)],
        corrections=None,
    )
    ingest.transfer_dataset(db, settings, ds1)
    assert db.scalar("SELECT count(*) FROM images") == 1

    ds2 = build_fixture_dataset(
        tmp_path / "d2",
        name="all_sasa_norm_2099_02_01",
        images=[
            FixtureImage("aa_1_1", overall_quality=0.8),   # already present
            FixtureImage("aa_1_2", overall_quality=0.7),   # new photo of a known individual
            FixtureImage("ee_1_1", overall_quality=0.7),   # brand-new individual
        ],
        corrections=None,
    )
    report = ingest.transfer_dataset(db, settings, ds2)
    assert report.images_added == 2
    assert report.images_skipped == 1
    assert report.contours_db_copied_verbatim is False  # delta-merged, not recopied
    assert db.scalar("SELECT count(*) FROM images") == 3
    assert db.query_one("SELECT status FROM individuals WHERE individual_id = 'ee_1'") is not None

    # delta rows landed in the working contours.db too
    con = duckdb.connect(str(settings.contours_db_path), read_only=True)
    try:
        merged = {r[0] for r in con.execute("SELECT salamander_id FROM images").fetchall()}
    finally:
        con.close()
    assert {"aa_1_1", "aa_1_2", "ee_1_1"} <= merged


def test_conflicting_checksum_is_left_untouched(db, settings, tmp_path):
    ds1 = build_fixture_dataset(
        tmp_path / "d1", name="all_sasa_norm_2099_01_01",
        images=[FixtureImage("aa_1_1", overall_quality=0.8)], corrections=None,
    )
    ingest.transfer_dataset(db, settings, ds1)

    # same id, different pixels
    (ds1 / "raw" / "aa_1_1.jpg").write_bytes(b"\xff\xd8\xff\xd9 different bytes")
    report = ingest.transfer_dataset(db, settings, ds1)
    assert report.images_conflicted == 1
    assert report.images_added == 0
    assert any("different checksum" in w for w in report.warnings)


def test_missing_raw_dir_raises(db, settings, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        ingest.transfer_dataset(db, settings, tmp_path / "empty")
