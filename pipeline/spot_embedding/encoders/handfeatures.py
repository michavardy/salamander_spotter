"""Per-spot shape descriptors (Phase 2.1).

Two families, computed from a spot's binary mask crop:

* **rotation-invariant core** — 7 log-Hu moments, solidity, extent, eccentricity, and the
  skeleton **topology** (endpoint + branch-point counts). These make a distinctive shape
  explicit: an ``L`` has 2 endpoints / 0 branches, a forked ``M`` has branch points, a round
  blob has ~0 of each — so the encoder cannot average an L into a dot ([§3](../../docs/per_spot_embedding_aggregation.md)).
* **orientation-dependent signature** — a radial max-radius profile in K angular sectors. This
  part *does* depend on how the spot is turned, so it is what the orientation A/B experiment
  (§2a) toggles: mode A leaves it as-is, mode B canonicalises the crop to its principal axis
  first.

``spot_descriptor(mask, mode)`` returns ``[invariant | radial(mode)]``. The invariant block is
identical for A and B; only the radial block differs, which is exactly the knob under test.
"""
from __future__ import annotations

import cv2
import numpy as np

INVARIANT_DIM = 12          # 7 Hu + solidity + extent + eccentricity + endpoints + branches
RADIAL_BINS = 16
DESCRIPTOR_DIM = INVARIANT_DIM + RADIAL_BINS


# --- orientation ------------------------------------------------------------
def principal_angle(mask: np.ndarray) -> float:
    """Orientation (radians) of the mask's major axis, from second-order central moments."""
    m = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        return 0.0
    mu20 = m["mu20"] / m["m00"]
    mu02 = m["mu02"] / m["m00"]
    mu11 = m["mu11"] / m["m00"]
    return 0.5 * np.arctan2(2.0 * mu11, (mu20 - mu02))


def canonicalize(mask: np.ndarray) -> np.ndarray:
    """Rotate a mask so its major axis is horizontal (principal-axis frame).

    Note the inherent **180° ambiguity** (this rotation cannot tell head from tail) and that a
    near-circular spot has no stable axis — the two weaknesses of canonicalisation that the A/B
    experiment is meant to expose.
    """
    h, w = mask.shape
    angle_deg = np.degrees(principal_angle(mask))
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)


# --- feature blocks ---------------------------------------------------------
def _largest_contour(mask: np.ndarray):
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def _skeleton_topology(mask: np.ndarray) -> tuple[float, float]:
    """(endpoints, branch_points) of the mask's skeleton, normalised to [0, ~1] by log."""
    from skimage.morphology import skeletonize

    skel = skeletonize(mask > 0).astype(np.uint8)
    if skel.sum() == 0:
        return 0.0, 0.0
    # neighbour count per skeleton pixel (8-connectivity)
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    neigh = cv2.filter2D(skel, -1, k, borderType=cv2.BORDER_CONSTANT) * skel
    endpoints = int(((neigh == 1) & (skel == 1)).sum())
    branches = int(((neigh >= 3) & (skel == 1)).sum())
    return np.log1p(endpoints), np.log1p(branches)


def invariant_features(mask: np.ndarray) -> np.ndarray:
    """Rotation/scale/translation-invariant shape descriptor (INVARIANT_DIM,)."""
    m = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    hu = cv2.HuMoments(m).flatten()
    # log-scale the Hu moments (signed) so their huge dynamic range is usable
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    area = float(m["m00"])
    cnt = _largest_contour(mask)
    if cnt is not None and area > 0:
        hull_area = cv2.contourArea(cv2.convexHull(cnt)) or area
        x, y, w, h = cv2.boundingRect(cnt)
        solidity = area / hull_area
        extent = area / float(max(w * h, 1))
    else:
        solidity = extent = 0.0

    if m["m00"] > 0:
        mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
        common = np.sqrt(max((mu20 - mu02) ** 2 + 4 * mu11 ** 2, 0.0))
        lam1 = (mu20 + mu02 + common) / 2.0
        lam2 = (mu20 + mu02 - common) / 2.0
        eccentricity = float(np.sqrt(1 - lam2 / lam1)) if lam1 > 0 else 0.0
    else:
        eccentricity = 0.0

    endpoints, branches = _skeleton_topology(mask)
    return np.concatenate([hu_log, [solidity, extent, eccentricity, endpoints, branches]]).astype(np.float64)


def radial_signature(mask: np.ndarray, bins: int = RADIAL_BINS) -> np.ndarray:
    """Orientation-dependent shape signature: max radius per angular sector, scale-normalised."""
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return np.zeros(bins, dtype=np.float64)
    cx, cy = xs.mean(), ys.mean()
    ang = np.arctan2(ys - cy, xs - cx)
    rad = np.hypot(xs - cx, ys - cy)
    idx = ((ang + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    sig = np.zeros(bins, dtype=np.float64)
    for b in range(bins):
        sel = rad[idx == b]
        if sel.size:
            sig[b] = sel.max()
    peak = sig.max()
    return sig / peak if peak > 0 else sig


def spot_descriptor(mask: np.ndarray, mode: str = "A") -> np.ndarray:
    """Full per-spot shape descriptor. ``mode='A'`` as-is, ``mode='B'`` principal-axis canonical.

    Only the orientation-dependent radial block differs between modes; the invariant block is
    computed on the as-is mask either way (it is rotation-invariant, so canonicalising it is a
    no-op).
    """
    inv = invariant_features(mask)
    crop = canonicalize(mask) if mode == "B" else mask
    return np.concatenate([inv, radial_signature(crop)])
