from __future__ import annotations

import pytest

from app.services.models import compute_score

METRICS = {"r1": 0.6, "r5": 0.8, "r10": 0.9, "bal_acc": 0.7, "novelty_auroc": 0.65, "review_at_90": 0.5}


def test_default_formula_half_r1_half_novelty():
    assert compute_score(METRICS, {"a": 0.5, "e": 0.5}) == pytest.approx(0.5 * 0.6 + 0.5 * 0.65)


def test_arbitrary_linear_combo():
    assert compute_score(METRICS, {"a": 0.3, "c": 0.6, "e": 0.1}) == pytest.approx(
        0.3 * 0.6 + 0.6 * 0.9 + 0.1 * 0.65
    )


def test_coefficients_need_not_sum_to_one():
    assert compute_score(METRICS, {"a": 1.0, "b": 1.0}) == pytest.approx(0.6 + 0.8)


def test_malformed_letter_is_ignored_not_executed():
    # "z" is not a metric letter; "__import__" style junk is just ignored
    assert compute_score(METRICS, {"z": 5.0, "a": 1.0}) == pytest.approx(0.6)
    assert compute_score(METRICS, {"a": "not-a-number"}) == 0.0


def test_missing_metric_treated_as_zero():
    assert compute_score({"r1": 0.6}, {"a": 1.0, "e": 1.0}) == pytest.approx(0.6)
