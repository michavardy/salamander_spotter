"""Two-view augmentation sampler (Phase 2.3).

Composes the geometry-space transforms into a single "view" of a constellation, and produces
**two** independent views of the same spot-set — the positive pair a contrastive/triplet loss
trains on (Phase 3). Guarantees:

* **identity-preserving** — every transform is orientation-preserving, so a view is never a
  mirror image (a reflection would be a different animal, §4c); :func:`assert_no_reflection`
  checks this;
* **variable cardinality** — dropout + spurious injection change the spot count, forcing the
  downstream matcher to tolerate missing/extra spots (§4).

Phase 2 has no trainer yet, so this ships with tests + a visual preview; Phase 3 consumes it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry as G


@dataclass
class SpotSample:
    centroids: np.ndarray   # (N, 2)
    areas: np.ndarray       # (N,)


@dataclass
class AugConfig:
    p_dropout: float = 0.2
    spurious_frac: float = 0.1
    max_rot: float = np.pi
    scale: tuple = (0.85, 1.15)
    aniso: float = 0.12
    shear: float = 0.10
    elastic_strength: float = 0.06
    jitter_frac: float = 0.015


def geometric_view(sample: SpotSample, rng: np.random.Generator,
                   cfg: AugConfig = AugConfig()) -> SpotSample:
    """The order- and count-preserving warp: similarity → affine → elastic → jitter.

    Separated from the set operations so orientation can be checked point-for-point
    (:func:`assert_no_reflection`).
    """
    c, a = sample.centroids.copy(), sample.areas.copy()
    c, a = G.similarity(c, a, rng, max_rot=cfg.max_rot, scale=cfg.scale)
    c, a = G.affine_jitter(c, a, rng, aniso=cfg.aniso, shear=cfg.shear)
    c, a = G.elastic_warp(c, a, rng, strength=cfg.elastic_strength)
    c, a = G.jitter(c, a, rng, sigma_frac=cfg.jitter_frac)
    return SpotSample(c, a)


def one_view(sample: SpotSample, rng: np.random.Generator, cfg: AugConfig = AugConfig()) -> SpotSample:
    """One augmented view: geometric warp, then dropout + spurious (which change the count)."""
    g = geometric_view(sample, rng, cfg)
    c, a = G.dropout(g.centroids, g.areas, rng, p=cfg.p_dropout)
    c, a = G.add_spurious(c, a, rng, frac=cfg.spurious_frac)
    return SpotSample(c, a)


def two_views(sample: SpotSample, rng: np.random.Generator, cfg: AugConfig = AugConfig()):
    """A positive pair: two independent augmented views of the same spot-set."""
    return one_view(sample, rng, cfg), one_view(sample, rng, cfg)


def _best_fit_linear_det(before: np.ndarray, after: np.ndarray) -> float:
    """Determinant of the least-squares linear map before→after (correspondence by index).

    A mirror flip forces this determinant negative; proper rotation/scale/shear + small
    deformation keep it positive. This is the well-conditioned way to detect reflection (the
    signed area of spots in arbitrary order is not).
    """
    X = np.hstack([before, np.ones((len(before), 1))])   # (N, 3): [x, y, 1]
    P, *_ = np.linalg.lstsq(X, after, rcond=None)        # after ≈ X @ P, P is (3, 2)
    M = P[:2, :2]
    return float(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])


def assert_no_reflection(before: SpotSample, after: SpotSample) -> bool:
    """True if the warp preserved orientation (no mirror flip).

    Requires equal count / index correspondence, so pass a :func:`geometric_view` output (not a
    dropped/spurious one). Checks the best-fit linear map has positive determinant.
    """
    if len(before.centroids) != len(after.centroids):
        raise ValueError("assert_no_reflection needs order-preserving views (use geometric_view)")
    if len(before.centroids) < 3:
        return True
    return _best_fit_linear_det(before.centroids, after.centroids) > 0


def dump_preview(sample: SpotSample, va: SpotSample, vb: SpotSample, path) -> None:
    """Render original + two views as a side-by-side scatter PNG (cv2, no matplotlib)."""
    import cv2

    panels = []
    for s, title in [(sample, "orig"), (va, "view A"), (vb, "view B")]:
        img = np.full((256, 256, 3), 30, np.uint8)
        if len(s.centroids):
            pts = s.centroids - s.centroids.min(0)
            span = pts.max(0)
            span[span == 0] = 1
            pts = (pts / span) * 210 + 22
            for (x, y) in pts.astype(int):
                cv2.circle(img, (int(x), int(y)), 4, (80, 200, 255), -1)
        cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        panels.append(img)
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.hstack(panels))
