"""The real matcher (spec M3, §7.3): soft-chamfer over precomputed per-spot
embeddings already in ``contours.db`` — no GPU, no LLM call at inference time.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from app.pipeline_bridge.matching import PipelineCorrespondenceBridge, PipelineMatchingBridge

# two unit 31-dim halves concatenated -> ||v|| == sqrt(2), matching the real
# pipeline/spot_transformer/core/embeddings.py "concat" layout.
_A = np.zeros(62)
_A[0] = 1.0
_A[31] = 1.0  # shape-half unit vector "A", position-half unit vector "A"

_B = np.zeros(62)
_B[1] = 1.0
_B[32] = 1.0  # orthogonal to _A in both halves -> additive cosine 0

_A_NOISY = np.zeros(62)
_A_NOISY[0] = 0.9
_A_NOISY[3] = np.sqrt(1 - 0.9**2)  # still mostly "A"-like in the shape half
_A_NOISY[31] = 1.0


def _build_contours_db(path, images: dict[str, dict]) -> None:
    con = duckdb.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE images (salamander_id VARCHAR PRIMARY KEY, is_synthetic BOOLEAN)"
        )
        con.execute("CREATE TABLE spot_embeddings (salamander_id VARCHAR, spot_id INTEGER, embedding DOUBLE[])")
        for image_id, spec in images.items():
            con.execute("INSERT INTO images VALUES (?, ?)", [image_id, spec.get("is_synthetic", False)])
            for spot_id, vec in enumerate(spec["spots"], start=1):
                con.execute(
                    "INSERT INTO spot_embeddings VALUES (?, ?, ?)", [image_id, spot_id, vec.tolist()]
                )
    finally:
        con.close()


@pytest.fixture
def contours_db(tmp_path):
    path = tmp_path / "contours.db"
    _build_contours_db(
        path,
        {
            "aa_1_1": {"spots": [_A, _B]},          # query
            "aa_1_2": {"spots": [_A_NOISY, _B]},     # same individual, close match
            "bb_1_1": {"spots": [_B, _B]},           # different individual, poor match
            "bb_1_g0": {"spots": [_A, _B], "is_synthetic": True},  # would look perfect — must be excluded
        },
    )
    return path


class TestScore:
    def test_ranks_the_true_match_first(self, contours_db):
        bridge = PipelineMatchingBridge()
        pairs = {p.individual_id: p for p in bridge.score("aa_1_1", ["aa_1", "bb_1"], contours_db)}
        assert pairs["aa_1"].similarity > pairs["bb_1"].similarity
        assert pairs["aa_1"].best_photo_id == "aa_1_2"

    def test_synthetic_photos_never_chosen_as_best(self, contours_db):
        bridge = PipelineMatchingBridge()
        pairs = {p.individual_id: p for p in bridge.score("aa_1_1", ["bb_1"], contours_db)}
        # bb_1's only real photo is bb_1_1 (poor match); bb_1_g0 is a perfect-looking
        # synthetic decoy and must never be picked (spec §7.7, D11)
        assert pairs["bb_1"].best_photo_id == "bb_1_1"

    def test_empty_query_returns_no_candidates(self, contours_db, tmp_path):
        path = tmp_path / "c2.db"
        _build_contours_db(path, {"zz_1_1": {"spots": []}, "aa_1_1": {"spots": [_A]}})
        bridge = PipelineMatchingBridge()
        assert bridge.score("zz_1_1", ["aa_1"], path) == []

    def test_missing_embeddings_table_raises_clear_error(self, tmp_path):
        path = tmp_path / "no_embeddings.db"
        con = duckdb.connect(str(path))
        con.execute("CREATE TABLE images (salamander_id VARCHAR, is_synthetic BOOLEAN)")
        con.close()
        with pytest.raises(RuntimeError, match="build-embeddings"):
            PipelineMatchingBridge().score("aa_1_1", ["bb_1"], path)

    def test_gallery_individual_with_no_real_photos_is_skipped(self, contours_db):
        bridge = PipelineMatchingBridge()
        pairs = bridge.score("aa_1_1", ["cc_1"], contours_db)  # cc_1 doesn't exist at all
        assert pairs == []


class TestCorrespond:
    def test_mutual_nearest_neighbours(self, contours_db):
        bridge = PipelineCorrespondenceBridge()
        links = bridge.correspond("aa_1_1", "aa_1_2", contours_db)
        assert {(l.query_spot_id, l.cand_spot_id) for l in links} == {(1, 1), (2, 2)}
        assert all(l.cand_image_id == "aa_1_2" for l in links)

    def test_empty_when_one_side_has_no_spots(self, contours_db, tmp_path):
        path = tmp_path / "c3.db"
        _build_contours_db(path, {"zz_1_1": {"spots": []}, "aa_1_1": {"spots": [_A]}})
        assert PipelineCorrespondenceBridge().correspond("zz_1_1", "aa_1_1", path) == []
