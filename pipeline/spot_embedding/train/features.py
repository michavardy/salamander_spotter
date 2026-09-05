"""Turn a SpotSet into the tensors the learned aggregators consume (Phase 3).

Per spot the node carries a **rotation-invariant** shape vector (the invariant core of the
Phase-2 descriptor + log-area) and a **relative position** (centroid, centred and scaled by the
median inter-spot distance). Using the *invariant* shape core — not the orientation-dependent
radial signature — keeps shape consistent when augmentation globally rotates the positions, so
rotation invariance is learned from the position augmentation without desyncing the shape.

``build_set_features`` reads the cached mode-A descriptors ([spot_encoder](../encoders/spot_encoder.py));
``augment`` applies the Phase-2 spot-set augmentation to a sample (positions warp/dropout/spurious,
shape carried along) to manufacture the two views a contrastive loss trains on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..augment import geometry as G
from ..encoders import build_spot_tokens
from ..encoders.handfeatures import INVARIANT_DIM

FEAT_DIM = INVARIANT_DIM + 1   # invariant shape core + log-area


@dataclass
class SetFeatures:
    salamander_id: str
    label: str
    feat: np.ndarray   # (N, FEAT_DIM) raw (pre-standardisation)
    pos: np.ndarray    # (N, 2) image-pixel centroids


def build_set_features(spotsets, dataset: str) -> dict[str, SetFeatures]:
    """id -> SetFeatures for every non-empty spotset (mode-A descriptors)."""
    tokens = build_spot_tokens(dataset, mode="A")
    out: dict[str, SetFeatures] = {}
    for ss in spotsets:
        if ss.is_empty or ss.salamander_id not in tokens:
            continue
        t = tokens[ss.salamander_id]
        inv = t.descriptors[:, :INVARIANT_DIM]
        logarea = np.log1p(t.areas)[:, None]
        out[ss.salamander_id] = SetFeatures(
            salamander_id=ss.salamander_id,
            label=ss.label,
            feat=np.hstack([inv, logarea]).astype(np.float32),
            pos=t.centroids.astype(np.float32),
        )
    return out


def feature_standardizer(sets: list[SetFeatures]) -> tuple[np.ndarray, np.ndarray]:
    allf = np.vstack([s.feat for s in sets])
    mean = allf.mean(0)
    std = allf.std(0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_pos(pos: np.ndarray) -> np.ndarray:
    """Centre on the centroid and scale by the median inter-spot distance (translation+scale norm)."""
    if len(pos) == 0:
        return pos
    c = pos.mean(0, keepdims=True)
    p = pos - c
    if len(pos) >= 2:
        d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        scale = np.median(d[d > 0]) if (d > 0).any() else 1.0
    else:
        scale = 1.0
    return (p / (scale + 1e-6)).astype(np.float32)


def augment(sf: SetFeatures, rng: np.random.Generator, *, p_dropout=0.2, spurious_frac=0.1,
            max_rot=np.pi, scale=(0.85, 1.15), aniso=0.12, shear=0.10,
            elastic=0.06, jitter=0.015) -> tuple[np.ndarray, np.ndarray]:
    """One augmented view → (feat, pos). Shape carried along; positions warped; rows dropped/added."""
    feat, pos, areas = sf.feat.copy(), sf.pos.copy(), np.ones(len(sf.pos), np.float32)
    # geometry on positions (shape features are rotation-invariant, so they ride along unchanged)
    pos, _ = G.similarity(pos, areas, rng, max_rot=max_rot, scale=scale)
    pos, _ = G.affine_jitter(pos, areas, rng, aniso=aniso, shear=shear)
    pos, _ = G.elastic_warp(pos, areas, rng, strength=elastic)
    pos, _ = G.jitter(pos, areas, rng, sigma_frac=jitter)

    n = len(pos)
    if n > 3:                                            # dropout (keep ≥3)
        keep = rng.random(n) >= p_dropout
        if keep.sum() < 3:
            keep[rng.choice(n, 3, replace=False)] = True
        feat, pos = feat[keep], pos[keep]

    k = rng.binomial(len(pos), spurious_frac)            # spurious spots (false detections)
    if k > 0:
        lo, hi = pos.min(0), pos.max(0)
        extra_pos = rng.uniform(lo, hi, size=(k, 2)).astype(np.float32)
        idx = rng.choice(len(feat), size=k)
        extra_feat = feat[idx] + rng.normal(0, 0.1, size=(k, feat.shape[1])).astype(np.float32)
        feat = np.vstack([feat, extra_feat])
        pos = np.vstack([pos, extra_pos])
    return feat, pos
