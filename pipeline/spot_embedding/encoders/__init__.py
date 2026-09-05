"""encoders — per-spot shape + relative-geometry encoding (Phase 2).

* ``handfeatures`` — rotation-invariant shape core (Hu, solidity, eccentricity, skeleton
  topology) + orientation-dependent radial signature; the orientation A/B knob lives here.
* ``spot_encoder`` — crop/normalise each spot mask and build cached per-spot tokens per mode.

The learned-CNN spot embedding is Phase 3; Phase 2's "shape embedding" is the hand-feature
descriptor, which needs no training.
"""
from . import handfeatures
from .spot_encoder import SpotTokens, build_spot_tokens, descriptor_standardizer

__all__ = ["handfeatures", "SpotTokens", "build_spot_tokens", "descriptor_standardizer"]
