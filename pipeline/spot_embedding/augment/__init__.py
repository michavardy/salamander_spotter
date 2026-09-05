"""augment — the two-tier augmentation pipeline.

Constellation / mask transforms (Phase 2.3):
* ``geometry``   — spot-set transforms (similarity, affine, elastic, jitter, dropout, spurious)
* ``appearance`` — spot-mask transforms (blur, rotate, stretch, occlude)
* ``pipeline``   — two-view sampler (positive pairs), no-reflection guard, visual preview

Learned-encoder / generative tiers (Phase 3+):
* ``spot_crops`` — **OpenCV** per-spot-crop augmentation feeding the learned SpotCNN
* ``gemini_view`` — **Gemini** whole-image augmentation (opt-in, billed, offline synthetic views)

All transforms are orientation-preserving (no mirror flips).
"""
from . import appearance, geometry, pipeline, spot_crops
from .pipeline import (
    AugConfig,
    SpotSample,
    assert_no_reflection,
    dump_preview,
    geometric_view,
    one_view,
    two_views,
)

__all__ = [
    "geometry", "appearance", "pipeline",
    "SpotSample", "AugConfig", "one_view", "two_views", "geometric_view",
    "assert_no_reflection", "dump_preview",
]
