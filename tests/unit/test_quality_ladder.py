from __future__ import annotations

import pytest

from app.services.quality_ladder import (
    LadderConfig,
    TIER_AUTO_ACCEPT,
    TIER_FAILED,
    TIER_HAND_CORRECTION,
    TIER_NEEDS_A_LOOK,
    ladder_tier,
)


def test_auto_accept_needs_quality_axis_and_spots():
    assert ladder_tier(overall_quality=0.70, has_axis=True, n_spots=5) == TIER_AUTO_ACCEPT


def test_high_quality_but_too_few_spots_is_not_auto_accept():
    assert ladder_tier(overall_quality=0.90, has_axis=True, n_spots=2) == TIER_NEEDS_A_LOOK


def test_high_quality_but_no_axis_is_not_auto_accept():
    assert ladder_tier(overall_quality=0.90, has_axis=False, n_spots=9) == TIER_NEEDS_A_LOOK


def test_mid_band_needs_a_look():
    assert ladder_tier(overall_quality=0.40, has_axis=True, n_spots=9) == TIER_NEEDS_A_LOOK
    assert ladder_tier(overall_quality=0.64, has_axis=True, n_spots=9) == TIER_NEEDS_A_LOOK


def test_low_quality_with_geometry_is_hand_correction():
    assert ladder_tier(overall_quality=0.39, has_axis=True, n_spots=9) == TIER_HAND_CORRECTION


def test_failed_extraction():
    assert ladder_tier(overall_quality=None, has_axis=False, n_spots=None) == TIER_FAILED
    assert (
        ladder_tier(overall_quality=0.9, has_axis=True, n_spots=9, extraction_failed=True)
        == TIER_FAILED
    )


def test_min_spots_is_configurable():
    cfg = LadderConfig(min_spots_auto_accept=8)
    assert ladder_tier(overall_quality=0.8, has_axis=True, n_spots=5, config=cfg) == TIER_NEEDS_A_LOOK
    assert ladder_tier(overall_quality=0.8, has_axis=True, n_spots=8, config=cfg) == TIER_AUTO_ACCEPT


@pytest.mark.parametrize("q", [0.0, 0.399999])
def test_boundary_just_below_floor(q):
    assert ladder_tier(overall_quality=q, has_axis=True, n_spots=9) == TIER_HAND_CORRECTION
