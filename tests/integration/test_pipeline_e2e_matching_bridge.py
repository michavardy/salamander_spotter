"""The trained-checkpoint matcher (spec M3, successor to soft-chamfer): a real
``E2EVoter`` forward pass over precomputed per-spot embeddings already in ``contours.db``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pytest
import torch

from app.pipeline_bridge.matching import PipelineE2EMatchingBridge

_ST = Path(__file__).resolve().parents[2] / "pipeline" / "spot_transformer"
for _sub in (".", "core", "models", "eval"):
    _p = str((_ST / _sub).resolve())
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aggregator_e2e import E2EVoter  # noqa: E402

_A = np.zeros(62)
_A[0] = 1.0
_A[31] = 1.0

_B = np.zeros(62)
_B[1] = 1.0
_B[32] = 1.0


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
            "aa_1_1": {"spots": [_A, _B]},
            "aa_1_2": {"spots": [_A, _B]},
            "bb_1_1": {"spots": [_B, _B]},
            "bb_1_g0": {"spots": [_A, _B], "is_synthetic": True},
        },
    )
    return path


@pytest.fixture
def checkpoint_path(tmp_path):
    """A tiny, randomly-initialized (but seeded, so deterministic) checkpoint — enough to
    exercise the bridge's load/batch/score plumbing without needing a real trained model."""
    torch.manual_seed(0)
    model_config = dict(in_dim=62, out_dim=62, arch="transformer", depth=1, dropout=0.0,
                        n_heads=2, residual=True, tau=0.05, sharp=20.0, hidden=8)
    model = E2EVoter(**model_config)
    path = tmp_path / "test_checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "model_config": model_config}, path)
    return path


class TestScore:
    def test_scores_every_reachable_gallery_individual(self, contours_db, checkpoint_path):
        bridge = PipelineE2EMatchingBridge(checkpoint_path)
        pairs = {p.individual_id: p for p in bridge.score("aa_1_1", ["aa_1", "bb_1"], contours_db)}
        assert set(pairs) == {"aa_1", "bb_1"}
        assert pairs["aa_1"].best_photo_id == "aa_1_2"  # leave-one-out excludes the query itself
        assert pairs["bb_1"].best_photo_id == "bb_1_1"
        assert all(0.0 <= p.similarity <= 1.0 for p in pairs.values())  # sigmoid output range

    def test_synthetic_photos_never_chosen_as_candidate(self, contours_db, checkpoint_path):
        bridge = PipelineE2EMatchingBridge(checkpoint_path)
        pairs = {p.individual_id: p for p in bridge.score("aa_1_1", ["bb_1"], contours_db)}
        assert pairs["bb_1"].best_photo_id == "bb_1_1"  # never the synthetic bb_1_g0

    def test_individual_whose_only_real_photo_is_the_query_is_skipped(self, contours_db, checkpoint_path):
        # aa_1's only OTHER real photo is aa_1_2, so scoring aa_1_2 as query leaves aa_1 with
        # nothing but itself excluded -> aa_1 has aa_1_1 left, so it should still appear; test the
        # genuine zero-remaining-photos case directly instead.
        path = contours_db.parent / "c2.db"
        _build_contours_db(path, {"aa_1_1": {"spots": [_A]}, "bb_1_1": {"spots": [_B]}})
        bridge = PipelineE2EMatchingBridge(checkpoint_path)
        assert bridge.score("aa_1_1", ["aa_1"], path) == []

    def test_empty_query_returns_no_candidates(self, checkpoint_path, tmp_path):
        path = tmp_path / "c3.db"
        _build_contours_db(path, {"zz_1_1": {"spots": []}, "aa_1_1": {"spots": [_A]}})
        bridge = PipelineE2EMatchingBridge(checkpoint_path)
        assert bridge.score("zz_1_1", ["aa_1"], path) == []

    def test_missing_embeddings_table_raises_clear_error(self, checkpoint_path, tmp_path):
        path = tmp_path / "no_embeddings.db"
        con = duckdb.connect(str(path))
        con.execute("CREATE TABLE images (salamander_id VARCHAR, is_synthetic BOOLEAN)")
        con.close()
        with pytest.raises(RuntimeError, match="build-embeddings"):
            PipelineE2EMatchingBridge(checkpoint_path).score("aa_1_1", ["bb_1"], path)

    def test_gallery_individual_with_no_real_photos_is_skipped(self, contours_db, checkpoint_path):
        bridge = PipelineE2EMatchingBridge(checkpoint_path)
        assert bridge.score("aa_1_1", ["cc_1"], contours_db) == []
