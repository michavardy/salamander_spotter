"""Shape-aware constellation matcher (Phase 2) — the training-free vehicle for the A/B test.

Phase 1's ``classical`` matcher used only spot **positions** (centroids). This one adds spot
**shape**: tentative correspondences are formed by nearest-neighbour in the per-spot *shape
descriptor* space ([spot_encoder](../encoders/spot_encoder.py)), then verified geometrically by
the same RANSAC similarity-transform as ``classical``. So it answers two things at once:

1. **does shape help?** — compare to ``classical`` (position-only) on the same harness;
2. **orientation A vs B** — run with ``mode='A'`` (as-is descriptor) vs ``mode='B'``
   (principal-axis canonicalised) to settle §2a in the training-free regime.

No training — the "shape embedding" is the hand-feature descriptor. The definitive A/B on a
*learned* CNN encoder is Phase 3; this is the first, cheap data point.
"""
from __future__ import annotations

import numpy as np

from ..encoders import build_spot_tokens, descriptor_standardizer
from .base import Matcher
from .classical import _MIN_SPOTS, _similarity_from_two

_INLIER_FRAC = 0.08
_MAX_ITERS = 200


class SpotDescMatcher(Matcher):
    name = "spotdesc"
    produces_embedding = False

    def __init__(self, dataset: str, mode: str = "A", inlier_frac: float = _INLIER_FRAC,
                 max_iters: int = _MAX_ITERS):
        if mode not in ("A", "B"):
            raise ValueError("mode must be 'A' or 'B'")
        self.mode = mode
        self.inlier_frac = inlier_frac
        self.max_iters = max_iters
        self._tokens = build_spot_tokens(dataset, mode=mode)
        self._mean, self._std = descriptor_standardizer(self._tokens)

    def _z(self, sid: str) -> np.ndarray:
        d = self._tokens[sid].descriptors
        return (d - self._mean) / self._std

    def _global_scale(self, centroids: np.ndarray) -> float:
        if len(centroids) < 2:
            return 1.0
        diff = centroids[:, None, :] - centroids[None, :, :]
        d = np.sqrt((diff * diff).sum(-1))
        finite = d[d > 0]
        return float(np.median(finite)) if finite.size else 1.0

    def _score(self, a, b) -> float:
        ca, cb = a.centroids, b.centroids
        na, nb = len(ca), len(cb)
        if na < _MIN_SPOTS or nb < _MIN_SPOTS:
            return 0.0

        # tentative correspondences: mutual nearest neighbour in shape-descriptor space
        za, zb = self._z(a.salamander_id), self._z(b.salamander_id)
        dd = np.sqrt(((za[:, None, :] - zb[None, :, :]) ** 2).sum(-1))   # (na, nb)
        a2b = dd.argmin(axis=1)
        b2a = dd.argmin(axis=0)
        corr = [(i, j) for i, j in enumerate(a2b) if b2a[j] == i]
        if len(corr) < 2:
            return len(corr) / min(na, nb)

        src = np.array([ca[i] for i, _ in corr])
        dst = np.array([cb[j] for _, j in corr])
        tol2 = (self.inlier_frac * self._global_scale(cb)) ** 2

        from itertools import combinations

        pairs = list(combinations(range(len(corr)), 2))
        if len(pairs) > self.max_iters:
            rng = np.random.default_rng(0)
            pairs = [pairs[t] for t in rng.choice(len(pairs), self.max_iters, replace=False)]

        best = 0
        for u, v in pairs:
            T = _similarity_from_two(src[u], src[v], dst[u], dst[v])
            if T is None:
                continue
            s, R, t = T
            proj = (s * (R @ src.T)).T + t
            best = max(best, int((((proj - dst) ** 2).sum(1) < tol2).sum()))
        return best / min(na, nb)

    def distance_matrix(self, queries: list, gallery: list) -> np.ndarray:
        out = np.empty((len(queries), len(gallery)), dtype=np.float64)
        for i, q in enumerate(queries):
            for j, g in enumerate(gallery):
                out[i, j] = 1.0 - self._score(q, g)
        return out
