"""OpenCV spot-crop augmentation (Phase 3+).

Augments a single spot's normalised mask crop so the learned per-spot CNN encoder
([SpotCNN](../models/nn.py)) sees each spot under many appearances and learns rotation /
stretch / blur / occlusion invariance directly from pixels — the "OpenCV spot augmentation"
tier of the pipeline. Every transform is orientation-preserving (no mirror flip).

Composed by :func:`augment_crop` with per-op probabilities; cheap (a few 32² warpAffines).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import appearance as A


@dataclass
class SpotCropAug:
    p_rotate: float = 0.9
    max_deg: float = 180.0
    p_stretch: float = 0.5
    aniso: float = 0.2
    p_blur: float = 0.4
    max_ksize: int = 5
    p_occlude: float = 0.25
    max_occ_frac: float = 0.3
    p_noise: float = 0.2
    noise_frac: float = 0.05


def _speckle(mask: np.ndarray, rng, frac: float) -> np.ndarray:
    """Flip a small fraction of pixels (extraction/edge noise)."""
    out = mask.copy()
    flip = rng.random(mask.shape) < frac
    out[flip] = 255 - out[flip]
    return out


def augment_crop(crop: np.ndarray, rng: np.random.Generator, cfg: SpotCropAug = SpotCropAug()) -> np.ndarray:
    """Return an augmented copy of a binary spot crop (uint8 0/255)."""
    out = crop
    if rng.random() < cfg.p_rotate:
        out = A.rotate(out, rng, max_deg=cfg.max_deg)
    if rng.random() < cfg.p_stretch:
        out = A.stretch(out, rng, aniso=cfg.aniso)
    if rng.random() < cfg.p_occlude:
        out = A.occlude(out, rng, max_frac=cfg.max_occ_frac)
    if rng.random() < cfg.p_blur:
        out = A.blur(out, rng, max_ksize=cfg.max_ksize)
    if rng.random() < cfg.p_noise:
        out = _speckle(out, rng, cfg.noise_frac)
    return out
