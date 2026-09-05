"""Per-image quality scoring + filtering (drop bad images from eval *and* training).

A photo is "bad" for spot re-identification when there is too little usable pattern or the
extraction clearly failed. We score three cheap, interpretable signals:

* **n_spots** — too few detected spots → nothing to match on (the dominant signal).
* **blur** — variance of the Laplacian of the raw grayscale image (low = out-of-focus /
  motion-blurred). Cached per dataset since it reads the raw files.
* **largest_frac** — area of the biggest spot / total spot area. Near 1.0 means one giant blob
  swallowed the pattern (a segmentation failure, e.g. the whole body keyed magenta).

:func:`filter_ids` turns a :class:`QualityConfig` into a kept-id set + a ranked drop list; the
runner applies it to the fold gallery/query **and** the learned-model training pool, so a
dropped image disappears from both.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

from .._common import contours_db_path, dataset_name, prepared_dir, raw_dir, resolve_dataset
from .spot_store import SpotSet

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


@dataclass
class QualityConfig:
    min_spots: int = 3            # drop images with fewer real spots than this
    min_blur: float = 0.0         # drop images blurrier than this (0 = blur filter off)
    max_largest_frac: float = 0.95  # drop images where one spot is ≥ this fraction of all spot area
    use_blur: bool = True         # compute the (cached) blur score

    @property
    def blur_on(self) -> bool:
        return self.use_blur and self.min_blur > 0


def _blur_scores(dataset: str, ids: list[str]) -> dict[str, float]:
    """Variance-of-Laplacian per image id, cached to prepared/<name>/blur.pkl."""
    import cv2

    dataset_dir = resolve_dataset(dataset)
    name = dataset_name(dataset_dir)
    cache_path = prepared_dir(name) / "blur.pkl"
    cache: dict[str, float] = {}
    if cache_path.is_file():
        cache = pickle.loads(cache_path.read_bytes())

    rd = raw_dir(dataset_dir)
    changed = False
    for sid in ids:
        if sid in cache:
            continue
        hits = [p for ext in _IMG_EXTS for p in rd.glob(f"{sid}{ext}")]
        if not hits:
            cache[sid] = float("nan")
            continue
        img = cv2.imread(str(hits[0]), cv2.IMREAD_GRAYSCALE)
        cache[sid] = float(cv2.Laplacian(img, cv2.CV_64F).var()) if img is not None else float("nan")
        changed = True
    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps(cache))
    return cache


def _largest_frac(ss: SpotSet) -> float:
    if ss.n_spots == 0 or ss.areas.sum() <= 0:
        return 1.0
    return float(ss.areas.max() / ss.areas.sum())


def assess(spotsets: list[SpotSet], dataset: str, cfg: QualityConfig) -> list[dict]:
    """Score every non-empty image. Returns dicts with signals + flags, worst-first."""
    live = [ss for ss in spotsets if not ss.is_empty]
    blur = _blur_scores(dataset, [ss.salamander_id for ss in live]) if cfg.blur_on else {}
    rows = []
    for ss in live:
        b = blur.get(ss.salamander_id, float("nan"))
        lf = _largest_frac(ss)
        reasons = []
        if ss.n_spots < cfg.min_spots:
            reasons.append(f"n_spots<{cfg.min_spots}")
        if cfg.blur_on and b == b and b < cfg.min_blur:      # b==b filters NaN
            reasons.append(f"blur<{cfg.min_blur:g}")
        if lf > cfg.max_largest_frac:
            reasons.append(f"largest_frac>{cfg.max_largest_frac:g}")
        rows.append({"id": ss.salamander_id, "label": ss.label, "n_spots": ss.n_spots,
                     "blur": b, "largest_frac": lf, "reasons": reasons, "drop": bool(reasons)})
    rows.sort(key=lambda r: (not r["drop"], r["n_spots"], r["blur"] if r["blur"] == r["blur"] else 0))
    return rows


def filter_ids(spotsets: list[SpotSet], dataset: str, cfg: QualityConfig):
    """Return (kept_ids set, dropped rows). Empties are already excluded by ``assess``."""
    rows = assess(spotsets, dataset, cfg)
    dropped = [r for r in rows if r["drop"]]
    kept = {r["id"] for r in rows if not r["drop"]}
    return kept, dropped
