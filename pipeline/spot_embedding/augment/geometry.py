"""Geometry-space augmentation (Phase 2.3) — transforms on the spot *set*.

These manufacture plausible alternate views of the *same* individual from one photo — the
mechanism that supplies positive pairs for the 202 single-photo individuals (§7). Every
transform is **orientation-preserving** (proper rotation + positive scales), so it never
mirror-flips the pattern (a reflection would be a *different* identity — §4c).

All functions take/return ``(centroids, areas)`` and an ``np.random.Generator`` for
reproducibility.
"""
from __future__ import annotations

import numpy as np


def _center(centroids: np.ndarray) -> np.ndarray:
    return centroids.mean(axis=0, keepdims=True) if len(centroids) else centroids


def similarity(centroids, areas, rng, *, max_rot=np.pi, scale=(0.85, 1.15), shift_frac=0.05):
    """Proper rotation + uniform scale + small translation (no reflection)."""
    c = _center(centroids)
    theta = rng.uniform(-max_rot, max_rot)
    s = rng.uniform(*scale)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    span = np.ptp(centroids, axis=0).mean() if len(centroids) else 1.0
    t = rng.uniform(-shift_frac, shift_frac, size=2) * span
    out = (s * (R @ (centroids - c).T)).T + c + t
    return out, areas


def affine_jitter(centroids, areas, rng, *, aniso=0.12, shear=0.10):
    """Mild anisotropic scale + shear (foreshortening proxy). Determinant kept > 0."""
    c = _center(centroids)
    sx, sy = 1 + rng.uniform(-aniso, aniso), 1 + rng.uniform(-aniso, aniso)
    sh = rng.uniform(-shear, shear)
    A = np.array([[sx, sh], [0.0, sy]])          # upper-triangular → det = sx*sy > 0, no flip
    out = (A @ (centroids - c).T).T + c
    return out, areas


def elastic_warp(centroids, areas, rng, *, n_control=4, strength=0.06):
    """Smooth low-frequency displacement (a lightweight thin-plate-spline stand-in for pose)."""
    if len(centroids) == 0:
        return centroids, areas
    span = np.ptp(centroids, axis=0).mean() or 1.0
    ctrl = rng.uniform(centroids.min(0), centroids.max(0), size=(n_control, 2))
    disp = rng.normal(0, strength * span, size=(n_control, 2))
    sigma = span / 2.0
    out = centroids.copy()
    for p, d in zip(ctrl, disp):
        w = np.exp(-((centroids - p) ** 2).sum(1) / (2 * sigma ** 2))
        out += w[:, None] * d
    return out, areas


def jitter(centroids, areas, rng, *, sigma_frac=0.015):
    """Per-spot Gaussian centroid jitter (annotation noise)."""
    if len(centroids) == 0:
        return centroids, areas
    span = np.ptp(centroids, axis=0).mean() or 1.0
    return centroids + rng.normal(0, sigma_frac * span, size=centroids.shape), areas


def dropout(centroids, areas, rng, *, p=0.2, keep_min=3):
    """Remove a random fraction of spots (missing-spot robustness — the key augmentation, §4)."""
    n = len(centroids)
    if n <= keep_min:
        return centroids, areas
    keep = rng.random(n) >= p
    if keep.sum() < keep_min:                    # never drop below the floor
        keep[rng.choice(n, keep_min, replace=False)] = True
    return centroids[keep], areas[keep]


def add_spurious(centroids, areas, rng, *, frac=0.1):
    """Inject a few false-positive spots inside the bounding box (detector noise)."""
    n = len(centroids)
    if n == 0:
        return centroids, areas
    k = rng.binomial(n, frac)
    if k == 0:
        return centroids, areas
    lo, hi = centroids.min(0), centroids.max(0)
    extra = rng.uniform(lo, hi, size=(k, 2))
    extra_a = rng.choice(areas, size=k) if len(areas) else np.ones(k)
    return np.vstack([centroids, extra]), np.concatenate([areas, extra_a])


def signed_area_sign(pts: np.ndarray) -> float:
    """Sign of the first triangle's signed area — used to assert orientation is preserved."""
    if len(pts) < 3:
        return 0.0
    a, b, c = pts[0], pts[1], pts[2]
    return float(np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])))
