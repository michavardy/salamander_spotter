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
    PipelineE2EMatchingBridge,
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
        # NOTE: PipelineMatchingBridge (soft-chamfer over contours.db's precomputed
        # `spot_embeddings`) validated at population-chance rank-1 on the live dataset — no real
        # discriminating signal, so it stays opt-in only via `Bridges.experimental_matcher()`.
        #
        # PipelineE2EMatchingBridge (a trained e2e_transformer checkpoint) is the real matcher:
        # activated when the caller resolves an active model's weights_path and passes it in (see
        # app/api/__init__.py, which looks up `models.active_model()`). No active model yet, or
        # its weights file missing -> falls back to the safe stub rather than guessing.
        if weights_path is not None and Path(weights_path).is_file():
            matching: MatchingBridge = PipelineE2EMatchingBridge(Path(weights_path))
            correspondence: CorrespondenceBridge = PipelineCorrespondenceBridge()
        else:
            matching = StubMatchingBridge()
            correspondence = StubCorrespondenceBridge()
        return cls(
            extraction=PipelineExtractionBridge(),
            matching=matching,
            correspondence=correspondence,
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
