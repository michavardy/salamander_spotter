"""The bundle of pipeline-bridge adapters an app instance uses (spec §7).

Real deployments get the ``Pipeline*`` adapters; tests inject ``Fake*`` ones by
replacing ``app.state.bridges``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .pipeline_bridge import ExtractionBridge, PipelineExtractionBridge
from .pipeline_bridge.editor import EditorBridge, PipelineEditorBridge
from .pipeline_bridge.matching import (
    CorrespondenceBridge,
    MatchingBridge,
    PipelineCorrespondenceBridge,
    PipelineMatchingBridge,
    StubCorrespondenceBridge,
    StubMatchingBridge,
)
from .pipeline_bridge.training import PipelineTrainingBridge, TrainingBridge


@dataclass
class Bridges:
    extraction: ExtractionBridge
    matching: MatchingBridge
    correspondence: CorrespondenceBridge
    editor: EditorBridge
    training: TrainingBridge

    @classmethod
    def production(cls, *, active_model: str | None = None, weights_path: Path | None = None) -> "Bridges":
        # NOTE: PipelineMatchingBridge/PipelineCorrespondenceBridge now have a real,
        # tested implementation (soft-chamfer over contours.db's precomputed
        # `spot_embeddings`) — but validation against the live dataset showed its
        # rank-1 is at population chance (median rank ~= gallery_size/2 across
        # gallery sizes 5/20/60), i.e. no real discriminating signal, consistent
        # with the pipeline's own docs calling soft-chamfer a weak baseline "to
        # improve upon". Shipping it as the default here would produce confident-
        # looking but essentially random match suggestions, which is worse than
        # not matching at all. Left wired for `Bridges.experimental_matcher()`
        # (opt-in) and further work — see docs/salamander_spotter_spec.md M3.
        return cls(
            extraction=PipelineExtractionBridge(),
            matching=StubMatchingBridge(),
            correspondence=StubCorrespondenceBridge(),
            editor=PipelineEditorBridge(),
            training=PipelineTrainingBridge(),
        )

    @classmethod
    def experimental_matcher(cls) -> "Bridges":
        """Like :meth:`production`, but with the soft-chamfer baseline matcher
        turned on — validated to rank near population chance (see the note in
        :meth:`production`). Useful for exercising the review workflow end to
        end; do not trust its confidences for real identification decisions."""
        base = cls.production()
        base.matching = PipelineMatchingBridge()
        base.correspondence = PipelineCorrespondenceBridge()
        return base

    @classmethod
    def fakes(cls) -> "Bridges":
        from .pipeline_bridge import FakeExtractionBridge
        from .pipeline_bridge.editor import FakeEditorBridge
        from .pipeline_bridge.matching import FakeCorrespondenceBridge, FakeMatchingBridge
        from .pipeline_bridge.training import FakeTrainingBridge

        return cls(
            extraction=FakeExtractionBridge(),
            matching=FakeMatchingBridge(),
            correspondence=FakeCorrespondenceBridge(),
            editor=FakeEditorBridge(),
            training=FakeTrainingBridge(),
        )
