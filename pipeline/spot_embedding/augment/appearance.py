"""Appearance-space augmentation (Phase 2.3) — transforms on a spot's mask crop.

Used when encoding spots (Phase 3): they simulate blur, viewing angle and foreshortening on a
single spot without changing its identity. Kept orientation-preserving (proper rotation, no
mirror flip). All take a mask crop + an ``np.random.Generator``.
"""
from __future__ import annotations

import cv2
import numpy as np


def blur(mask: np.ndarray, rng, *, max_ksize=5) -> np.ndarray:
    """Odd-kernel Gaussian blur (then re-binarise) — resolution / focus robustness."""
    k = int(rng.integers(0, max_ksize // 2 + 1)) * 2 + 1
    if k <= 1:
        return mask
    b = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
    return (b > 127).astype(np.uint8) * 255


def rotate(mask: np.ndarray, rng, *, max_deg=180.0) -> np.ndarray:
    """Proper rotation of the spot (viewing-angle proxy). No reflection."""
    h, w = mask.shape
    deg = rng.uniform(-max_deg, max_deg)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)


def stretch(mask: np.ndarray, rng, *, aniso=0.2) -> np.ndarray:
    """Anisotropic scale (foreshortening on a curved body). Positive scales → no flip."""
    h, w = mask.shape
    sx, sy = 1 + rng.uniform(-aniso, aniso), 1 + rng.uniform(-aniso, aniso)
    M = np.array([[sx, 0, (1 - sx) * w / 2.0], [0, sy, (1 - sy) * h / 2.0]], np.float32)
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)


def occlude(mask: np.ndarray, rng, *, max_frac=0.3) -> np.ndarray:
    """Erase a random slab of the spot (grass / mud / water sheen partial occlusion)."""
    h, w = mask.shape
    out = mask.copy()
    fh = rng.uniform(0, max_frac)
    y0 = int(rng.uniform(0, h * (1 - fh)))
    out[y0 : y0 + int(h * fh), :] = 0
    return out
