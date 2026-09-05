"""Cached per-spot mask crops for the learned CNN encoder (Phase 3+).

Decodes each spot's mask, normalises it (aspect-preserving → ``size²``), and keeps the
**largest ``cap`` spots** per photo (by area) so compute/memory stay bounded on the long-tail
images (up to 359 spots). Cached to ``prepared/<name>/spot_crops_<size>_<cap>.pkl``.

Each photo → ``SpotCrops(crops (n, size, size) uint8, logarea (n,), pos (n, 2))``, spot order
= descending area (so the informative spots are always kept when capping).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

from .._common import contours_db_path, dataset_name, prepared_dir, resolve_dataset
from ..encoders.spot_encoder import _normalize_crop

DEFAULT_SIZE = 32
DEFAULT_CAP = 40


@dataclass
class SpotCrops:
    salamander_id: str
    label: str
    crops: np.ndarray   # (n, size, size) uint8
    logarea: np.ndarray # (n,)
    pos: np.ndarray     # (n, 2)


def build_spot_crops(dataset: str, spotsets, *, size: int = DEFAULT_SIZE,
                     cap: int = DEFAULT_CAP) -> dict[str, SpotCrops]:
    """id -> SpotCrops for every non-empty photo, cached per (size, cap)."""
    import cv2
    import duckdb

    dataset_dir = resolve_dataset(dataset)
    name = dataset_name(dataset_dir)
    cache = prepared_dir(name) / f"spot_crops_{size}_{cap}.pkl"
    if cache.is_file():
        blob = pickle.loads(cache.read_bytes())
        if blob.get("size") == size and blob.get("cap") == cap:
            return blob["crops"]

    label_of = {ss.salamander_id: ss.label for ss in spotsets}
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

    out: dict[str, SpotCrops] = {}
    for sid, items in grouped.items():
        items.sort(key=lambda t: t[2], reverse=True)      # largest area first
        items = items[:cap]
        crops, cents, areas = [], [], []
        for cx, cy, area, png in items:
            m = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
            crops.append(_normalize_crop(m, size) if m is not None else np.zeros((size, size), np.uint8))
            cents.append((cx, cy))
            areas.append(area)
        out[sid] = SpotCrops(
            salamander_id=sid,
            label=label_of.get(sid, sid.rsplit("_", 1)[0]),
            crops=np.asarray(crops, dtype=np.uint8),
            logarea=np.log1p(np.asarray(areas, np.float64)),
            pos=np.asarray(cents, np.float64),
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps({"size": size, "cap": cap, "crops": out}))
    return out
