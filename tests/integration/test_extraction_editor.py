"""Spec §10.6 — in-place extraction correction, re-bin/re-score, revert."""

from __future__ import annotations

import duckdb
import pytest

from app.pipeline_bridge.editor import FakeEditorBridge
from app.services import extraction_editor, ingest


@pytest.fixture
def imported(db, settings, dataset_dir):
    ingest.transfer_dataset(db, settings, dataset_dir)
    return db


def _spot_count(path, image_id):
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute("SELECT count(*) FROM spots WHERE salamander_id = ?", [image_id]).fetchone()[0]
    finally:
        con.close()


def test_join_spots_reduces_count_and_rescores(imported, settings):
    before = _spot_count(settings.contours_db_path, "aa_1_1")  # 8
    result = extraction_editor.apply_edit(
        imported, settings.contours_db_path, image_id="aa_1_1",
        ops=[{"op": "join_spots", "spot_ids": [1, 2, 3]}], bridge=FakeEditorBridge(),
        editor_id="dana",
    )
    assert result.n_spots == before - 2
    assert _spot_count(settings.contours_db_path, "aa_1_1") == before - 2
    img = imported.query_one("SELECT n_spots, ladder_tier FROM images WHERE image_id = 'aa_1_1'")
    assert img["n_spots"] == before - 2
    assert imported.scalar("SELECT count(*) FROM extraction_corrections WHERE image_id='aa_1_1'") == 1


def test_revert_restores_snapshot(imported, settings):
    before = _spot_count(settings.contours_db_path, "aa_1_1")
    r = extraction_editor.apply_edit(
        imported, settings.contours_db_path, image_id="aa_1_1",
        ops=[{"op": "delete_spot", "spot_id": 1}], bridge=FakeEditorBridge(),
    )
    assert _spot_count(settings.contours_db_path, "aa_1_1") == before - 1
    extraction_editor.revert_extraction(
        imported, settings.contours_db_path, image_id="aa_1_1",
        correction_id=r.correction_id, bridge=FakeEditorBridge(),
    )
    assert _spot_count(settings.contours_db_path, "aa_1_1") == before


def test_split_spot_adds_one(imported, settings):
    before = _spot_count(settings.contours_db_path, "bb_1_1")
    extraction_editor.apply_edit(
        imported, settings.contours_db_path, image_id="bb_1_1",
        ops=[{"op": "split_spot", "spot_id": 1}], bridge=FakeEditorBridge(),
    )
    assert _spot_count(settings.contours_db_path, "bb_1_1") == before + 1


def test_set_head_tail_marks_corrected(imported, settings):
    extraction_editor.apply_edit(
        imported, settings.contours_db_path, image_id="aa_1_1",
        ops=[{"op": "set_head_tail", "head": [1.0, 2.0], "tail": [9.0, 9.0]}],
        bridge=FakeEditorBridge(),
    )
    con = duckdb.connect(str(settings.contours_db_path), read_only=True)
    try:
        src = con.execute("SELECT source, head_x FROM body_axis WHERE salamander_id='aa_1_1'").fetchone()
    finally:
        con.close()
    assert src[0] == "mask_corrected" and src[1] == 1.0


def test_unknown_op_rejected(imported, settings):
    with pytest.raises(ValueError):
        extraction_editor.apply_edit(
            imported, settings.contours_db_path, image_id="aa_1_1",
            ops=[{"op": "nonsense"}], bridge=FakeEditorBridge(),
        )


def test_correspondence_label_recorded(imported):
    cid = extraction_editor.label_correspondence(
        imported, query_image_id="aa_1_1", query_spot_id=3,
        other_image_id="aa_1_2", other_spot_id=5, verdict="same", editor="dana",
    )
    row = imported.query_one("SELECT * FROM correspondence_labels WHERE id = ?", [cid])
    assert row["verdict"] == "same" and row["editor"] == "dana"
    with pytest.raises(ValueError):
        extraction_editor.label_correspondence(
            imported, query_image_id="a", query_spot_id=1, other_image_id="b",
            other_spot_id=1, verdict="maybe",
        )
