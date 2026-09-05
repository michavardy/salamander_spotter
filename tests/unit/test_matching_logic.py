from __future__ import annotations

import pytest

from app.services.matching import Calibrator, rank_candidates, top_n_by_coverage
from app.pipeline_bridge.matching import PairScore
from app.settings_store import SettingsStore


class TestTopNByCoverage:
    def test_empty(self):
        assert top_n_by_coverage([], coverage_target=0.9, max_candidates=6) == 0

    def test_one_dominant_candidate(self):
        assert top_n_by_coverage([0.9, 0.05, 0.03], coverage_target=0.9, max_candidates=6) == 1

    def test_spreads_until_covered(self):
        n = top_n_by_coverage([0.3, 0.3, 0.3, 0.1], coverage_target=0.9, max_candidates=6)
        assert n == 3

    def test_capped_at_max(self):
        n = top_n_by_coverage([0.1] * 20, coverage_target=0.99, max_candidates=6)
        assert n == 6


class TestCalibrator:
    def test_identity_default(self):
        c = Calibrator()
        assert c(0.7) == 0.7
        assert c(1.5) == 1.0
        assert c(-0.2) == 0.0

    def test_platt_is_monotonic(self):
        c = Calibrator.from_json({"a": 6.0, "b": -3.0})
        assert c(0.9) > c(0.5) > c(0.1)
        assert 0.0 <= c(0.5) <= 1.0


def test_rank_candidates_orders_and_limits(db):
    store = SettingsStore(db)
    store.update({"coverage_target": 0.9, "max_candidates": 3})
    pairs = [
        PairScore("a_1", "a_1_1", 0.9),
        PairScore("b_1", "b_1_1", 0.4),
        PairScore("c_1", "c_1_1", 0.2),
        PairScore("d_1", "d_1_1", 0.1),
    ]
    cands = rank_candidates(pairs, Calibrator(), store)
    assert [c.individual_id for c in cands][:1] == ["a_1"]
    assert cands[0].rank == 1
    assert len(cands) <= 3
