"""Classical constellation matcher (4.5) — the non-DNN floor. numpy + cv2 only, no training.

The domain-standard approach for spot-patterned wildlife (HotSpotter / I3S / star-matching):
treat each salamander as a point set of spot centroids and score two photos by how well their
constellations align under a **similarity transform** (translation + rotation + uniform scale),
found by RANSAC. This is invariant to translation, rotation and scale by construction, and it
degrades gracefully when spots are missing (it aligns a consistent *subset*).

Pipeline per pair (a, b):
  1. per-spot local descriptor — sorted, locally-scale-normalized distances to the k nearest
     neighbours (rotation- + scale-invariant) → tentative mutual-nearest correspondences;
  2. RANSAC over 2-correspondence similarity transforms → the transform with the most inliers;
  3. score = inliers / min(|a|, |b|)  ∈ [0, 1];  distance = 1 − score.

Deterministic: correspondence pairs are enumerated exhaustively when few, else sampled with a
fixed-seed RNG, so repeated runs give identical numbers.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .base import Matcher

_MIN_SPOTS = 3          # fewer than this → not enough to align
_K = 5                  # neighbours in the local descriptor
_MAX_ITERS = 200        # RANSAC transform hypotheses when sampling
_INLIER_FRAC = 0.08     # inlier tolerance as a fraction of the target constellation's scale


def _pairwise(centroids: np.ndarray) -> np.ndarray:
    diff = centroids[:, None, :] - centroids[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def _descriptors(centroids: np.ndarray, k: int = _K):
    """Per-spot descriptor: sorted distances to k NN, normalised by the spot's own local scale.

    Tiny constellations (n < 2) get a zero descriptor and unit scale — they are scored 0 by the
    ``_MIN_SPOTS`` gate anyway, so this just avoids empty-slice warnings.
    """
    n = len(centroids)
    if n < 2:
        return np.zeros((n, k), dtype=np.float64), 1.0
    D = _pairwise(centroids)
    np.fill_diagonal(D, np.inf)
    kk = min(k, n - 1)
    nn = np.sort(D, axis=1)[:, :kk]                       # (n, kk) ascending NN distances
    local_scale = np.median(nn, axis=1, keepdims=True)    # per-spot scale → scale invariance
    local_scale[local_scale == 0] = 1.0
    desc = nn / local_scale
    if kk < k:                                            # pad tiny constellations
        pad_val = float(desc[:, -1:].mean()) if desc.size else 0.0
        desc = np.pad(desc, ((0, 0), (0, k - kk)), constant_values=pad_val)
    finite = D[np.isfinite(D)]
    global_scale = float(np.median(finite)) if finite.size else 1.0
    return desc, global_scale


class _Prepared:
    __slots__ = ("centroids", "desc", "scale")

    def __init__(self, centroids: np.ndarray):
        self.centroids = np.asarray(centroids, dtype=np.float64)
        self.desc, self.scale = _descriptors(self.centroids)


def _mutual_correspondences(pa: _Prepared, pb: _Prepared):
    """Tentative matches: spots that are each other's nearest in descriptor space."""
    da, db = pa.desc, pb.desc
    dd = np.sqrt(((da[:, None, :] - db[None, :, :]) ** 2).sum(-1))  # (Na, Nb)
    a2b = dd.argmin(axis=1)
    b2a = dd.argmin(axis=0)
    return [(i, j) for i, j in enumerate(a2b) if b2a[j] == i]


def _similarity_from_two(a1, a2, b1, b2):
    """Closed-form similarity transform (s, R, t) mapping a→b from two correspondences."""
    da, db = a2 - a1, b2 - b1
    na = np.hypot(*da)
    if na < 1e-9:
        return None
    s = np.hypot(*db) / na
    ang = np.arctan2(db[1], db[0]) - np.arctan2(da[1], da[0])
    c, sn = np.cos(ang), np.sin(ang)
    R = np.array([[c, -sn], [sn, c]])
    t = b1 - s * (R @ a1)
    return s, R, t


class ConstellationMatcher(Matcher):
    name = "classical"
    produces_embedding = False

    def __init__(self, k: int = _K, max_iters: int = _MAX_ITERS, inlier_frac: float = _INLIER_FRAC):
        self.k = k
        self.max_iters = max_iters
        self.inlier_frac = inlier_frac
        self._cache: dict[int, _Prepared] = {}

    def _prep(self, ss) -> _Prepared:
        key = id(ss)
        p = self._cache.get(key)
        if p is None:
            p = _Prepared(ss.centroids)
            self._cache[key] = p
        return p

    def _score(self, pa: _Prepared, pb: _Prepared) -> float:
        na, nb = len(pa.centroids), len(pb.centroids)
        if na < _MIN_SPOTS or nb < _MIN_SPOTS:
            return 0.0
        corr = _mutual_correspondences(pa, pb)
        if len(corr) < 2:
            return len(corr) / min(na, nb)

        src = np.array([pa.centroids[i] for i, _ in corr])
        dst = np.array([pb.centroids[j] for _, j in corr])
        tol = self.inlier_frac * pb.scale
        tol2 = tol * tol

        pairs = list(combinations(range(len(corr)), 2))
        if len(pairs) > self.max_iters:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(pairs), size=self.max_iters, replace=False)
            pairs = [pairs[t] for t in idx]

        best = 0
        for u, v in pairs:
            T = _similarity_from_two(src[u], src[v], dst[u], dst[v])
            if T is None:
                continue
            s, R, t = T
            proj = (s * (R @ src.T)).T + t          # (M, 2)
            d2 = ((proj - dst) ** 2).sum(1)
            best = max(best, int((d2 < tol2).sum()))
        return best / min(na, nb)

    def distance_matrix(self, queries: list, gallery: list) -> np.ndarray:
        gp = [self._prep(g) for g in gallery]
        out = np.empty((len(queries), len(gallery)), dtype=np.float64)
        for i, q in enumerate(queries):
            qp = self._prep(q)
            for j, g in enumerate(gp):
                out[i, j] = 1.0 - self._score(qp, g)
        return out
