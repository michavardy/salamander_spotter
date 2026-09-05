"""Read the packaged ``contours.db`` into per-image ``SpotSet`` objects.

One ``SpotSet`` per photo: the spot centroids + areas + identity label, plus lazy access
to the full-frame per-spot masks. This is the single representation every matcher reads.

Policy for the two degenerate cases (verified against the DB on 2026-07-11):

* **Zero-spot images** (13 of them, e.g. ``ca_1_1``, ``jd_1_7``) are loaded but flagged
  ``is_empty``; the splits drop them from gallery/query roles — a photo with no spots
  cannot be matched.
* **Raw/DB mismatch**: the loader keys off the DB (444 rows), so a raw file absent from the
  DB (``jt_1_1``) is simply not loaded. :func:`reconcile_raw` reports it; we never invent a
  row for it.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .._common import (
    contours_db_path,
    dataset_name,
    derive_code,
    derive_label,
    prepared_dir,
    raw_dir,
    resolve_dataset,
)

_CACHE_VERSION = 1
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class SpotSet:
    """The spots of one photo, plus its identity label.

    ``centroids`` is ``(n_spots, 2)`` float (image px, x/y); ``areas`` is ``(n_spots,)``
    float (px²). Both are empty arrays for a zero-spot image.
    """

    salamander_id: str          # image id / raw stem, e.g. "aj_1_2"
    label: str                  # identity label, e.g. "aj_1" (id minus the instance)
    code: str                   # source/session code, e.g. "aj"
    width: int
    height: int
    n_spots: int
    centroids: np.ndarray       # (n_spots, 2) float
    areas: np.ndarray           # (n_spots,) float

    @property
    def is_empty(self) -> bool:
        return self.n_spots == 0


def _read_from_db(db_path: Path) -> list[SpotSet]:
    import duckdb

    con = duckdb.connect(database=str(db_path), read_only=True)
    try:
        imgs = con.execute(
            "SELECT salamander_id, width, height, n_spots FROM images ORDER BY salamander_id"
        ).fetchall()
        spots = con.execute(
            "SELECT salamander_id, spot_id, global_centroid_x, global_centroid_y, area_pixels "
            "FROM spots ORDER BY salamander_id, spot_id"
        ).fetchall()
    finally:
        con.close()

    # group spot rows by image id
    by_id: dict[str, list[tuple]] = {}
    for sid, spot_id, cx, cy, area in spots:
        by_id.setdefault(sid, []).append((cx, cy, area))

    out: list[SpotSet] = []
    for sid, width, height, n_spots in imgs:
        rows = by_id.get(sid, [])
        if rows:
            arr = np.asarray(rows, dtype=np.float64)
            centroids = arr[:, :2].copy()
            areas = arr[:, 2].copy()
        else:
            centroids = np.empty((0, 2), dtype=np.float64)
            areas = np.empty((0,), dtype=np.float64)
        out.append(
            SpotSet(
                salamander_id=sid,
                label=derive_label(sid),
                code=derive_code(sid),
                width=int(width),
                height=int(height),
                n_spots=int(n_spots),
                centroids=centroids,
                areas=areas,
            )
        )
    return out


def load_spotsets(dataset: str, *, use_cache: bool = True, rebuild: bool = False) -> list[SpotSet]:
    """Load every photo's ``SpotSet`` from the dataset (DB-backed, with a pickle cache).

    The cache lives at ``artifacts/spot_embedding/prepared/<name>/spotsets.pkl``; pass
    ``rebuild=True`` (or ``use_cache=False``) to bypass it.
    """
    dataset_dir = resolve_dataset(dataset)
    name = dataset_name(dataset_dir)
    cache = prepared_dir(name) / "spotsets.pkl"

    if use_cache and not rebuild and cache.is_file():
        blob = pickle.loads(cache.read_bytes())
        if blob.get("version") == _CACHE_VERSION:
            return blob["spotsets"]

    spotsets = _read_from_db(contours_db_path(dataset_dir))

    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps({"version": _CACHE_VERSION, "spotsets": spotsets}))
    return spotsets


def load_mask(dataset: str, salamander_id: str, spot_id: int) -> np.ndarray:
    """Decode one spot's full-frame binary mask (``HxW`` uint8, 0/255) from the DB.

    Lazy: only touched by tests / visualisation, so cv2 + duckdb stay off the hot path.
    """
    import cv2  # noqa: PLC0415 — lazy, keeps the loader light

    import duckdb

    dataset_dir = resolve_dataset(dataset)
    con = duckdb.connect(database=str(contours_db_path(dataset_dir)), read_only=True)
    try:
        row = con.execute(
            "SELECT mask_png FROM spots WHERE salamander_id = ? AND spot_id = ?",
            [salamander_id, spot_id],
        ).fetchone()
    finally:
        con.close()
    if row is None:
        raise KeyError(f"no spot ({salamander_id!r}, spot_id={spot_id})")
    mask = cv2.imdecode(np.frombuffer(row[0], np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"failed to decode mask_png for ({salamander_id!r}, {spot_id})")
    return mask


def dataset_summary(spotsets: list[SpotSet]) -> dict:
    """Counts used to verify the loader against the DB ground truth."""
    from collections import Counter

    per_label = Counter(ss.label for ss in spotsets)
    empty = [ss.salamander_id for ss in spotsets if ss.is_empty]
    return {
        "images": len(spotsets),
        "labels": len(per_label),
        "singletons": sum(1 for v in per_label.values() if v == 1),
        "multi": sum(1 for v in per_label.values() if v >= 2),
        "empty_images": len(empty),
        "empty_ids": sorted(empty),
        "total_spots": int(sum(ss.n_spots for ss in spotsets)),
    }


def reconcile_raw(dataset: str, spotsets: list[SpotSet]) -> dict:
    """Compare the DB ids against the ``raw/`` files, reporting any mismatch.

    ``raw_only`` (files present in raw/ but missing from the DB, e.g. ``jt_1_1``) is a
    warning, not an error — the loader intentionally keys off the DB.
    """
    dataset_dir = resolve_dataset(dataset)
    rd = raw_dir(dataset_dir)
    raw_stems = (
        {p.stem for p in rd.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS}
        if rd.is_dir()
        else set()
    )
    db_ids = {ss.salamander_id for ss in spotsets}
    return {
        "raw_files": len(raw_stems),
        "db_images": len(db_ids),
        "raw_only": sorted(raw_stems - db_ids),   # in raw/, absent from DB
        "db_only": sorted(db_ids - raw_stems),    # in DB, absent from raw/
    }
