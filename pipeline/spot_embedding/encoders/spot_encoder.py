"""Per-spot token front-end (Phase 2.2).

Turns every spot into a fixed-length token by cropping its mask from the DB, normalising it
(aspect-preserving pad → resize to a canonical size), and computing a shape descriptor
([handfeatures](handfeatures.py)) under the chosen orientation mode:

* **mode A** — descriptor on the as-is crop (invariance must come from augmentation/training).
* **mode B** — crop is rotated to its principal axis first (geometric canonicalisation).

This is the orientation A/B switch of §2a. Tokens (+ their centroids and areas) are cached per
mode to ``prepared/<name>/spot_tokens_<mode>.pkl``. The learned-CNN shape embedding is Phase 3;
here the "shape embedding" is the hand-feature descriptor, which needs no training.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

from .._common import contours_db_path, dataset_name, prepared_dir, resolve_dataset
from . import handfeatures as H

_CANON_SIZE = 64


@dataclass
class SpotTokens:
    """Per-image spot tokens aligned to the SpotSet's spot_id order."""

    salamander_id: str
    descriptors: np.ndarray   # (n_spots, H.DESCRIPTOR_DIM)
    centroids: np.ndarray     # (n_spots, 2)
    areas: np.ndarray         # (n_spots,)


def _normalize_crop(mask: np.ndarray, size: int = _CANON_SIZE) -> np.ndarray:
    """Aspect-preserving crop of the spot to a square canvas, resized to ``size`` (scale-norm)."""
    import cv2

    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return np.zeros((size, size), np.uint8)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = mask[y0:y1, x0:x1]
    h, w = crop.shape
    side = max(h, w)
    canvas = np.zeros((side, side), np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy : oy + h, ox : ox + w] = crop
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST)


def build_spot_tokens(dataset: str, mode: str = "A", size: int = _CANON_SIZE) -> dict[str, SpotTokens]:
    """id -> SpotTokens for the whole dataset, cached per orientation mode."""
    import cv2
    import duckdb

    dataset_dir = resolve_dataset(dataset)
    name = dataset_name(dataset_dir)
    cache = prepared_dir(name) / f"spot_tokens_{mode}.pkl"
    if cache.is_file():
        blob = pickle.loads(cache.read_bytes())
        if blob.get("size") == size and blob.get("mode") == mode:
            return blob["tokens"]

    con = duckdb.connect(database=str(contours_db_path(dataset_dir)), read_only=True)
    try:
        rows = con.execute(
            "SELECT salamander_id, spot_id, global_centroid_x, global_centroid_y, area_pixels, mask_png "
            "FROM spots ORDER BY salamander_id, spot_id"
        ).fetchall()
    finally:
        con.close()

    grouped: dict[str, list] = {}
    for sid, spot_id, cx, cy, area, png in rows:
        grouped.setdefault(sid, []).append((cx, cy, area, png))

    tokens: dict[str, SpotTokens] = {}
    for sid, items in grouped.items():
        descs, cents, areas = [], [], []
        for cx, cy, area, png in items:
            m = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
            crop = _normalize_crop(m, size) if m is not None else np.zeros((size, size), np.uint8)
            descs.append(H.spot_descriptor(crop, mode=mode))
            cents.append((cx, cy))
            areas.append(area)
        tokens[sid] = SpotTokens(
            salamander_id=sid,
            descriptors=np.asarray(descs, dtype=np.float64),
            centroids=np.asarray(cents, dtype=np.float64),
            areas=np.asarray(areas, dtype=np.float64),
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps({"size": size, "mode": mode, "tokens": tokens}))
    return tokens


def descriptor_standardizer(tokens: dict[str, SpotTokens]) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std over all spot descriptors, for z-scoring so no feature dominates the distance."""
    alld = np.vstack([t.descriptors for t in tokens.values() if len(t.descriptors)])
    mean = alld.mean(axis=0)
    std = alld.std(axis=0)
    std[std == 0] = 1.0
    return mean, std
