"""spec §7 / M3 — production defaults to the safe stub matcher, not the
validated-but-uninformative soft-chamfer baseline (see app/bridges.py)."""

from __future__ import annotations

import pytest

from app.bridges import Bridges
from app.pipeline_bridge.matching import (
    PipelineCorrespondenceBridge,
    PipelineMatchingBridge,
    StubCorrespondenceBridge,
    StubMatchingBridge,
)


def test_production_defaults_to_the_safe_stub_matcher():
    b = Bridges.production()
    assert isinstance(b.matching, StubMatchingBridge)
    assert isinstance(b.correspondence, StubCorrespondenceBridge)
    with pytest.raises(NotImplementedError, match="chance"):
        b.matching.score("q", ["g"], "irrelevant.db")


def test_experimental_matcher_opts_into_the_real_baseline():
    b = Bridges.experimental_matcher()
    assert isinstance(b.matching, PipelineMatchingBridge)
    assert isinstance(b.correspondence, PipelineCorrespondenceBridge)


def test_fakes_are_unaffected():
    b = Bridges.fakes()
    assert b.matching.score("q", ["g1"], "irrelevant.db")  # FakeMatchingBridge just works
