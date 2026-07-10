#!/usr/bin/env python3
"""Stage 2 — isolate the magenta spots and store their contours + masks in DuckDB.

Reads the purple images produced by stage 1 (``<input_dir>/purple/*.png``), keys
out the flat magenta (#FF00FF) spots in HSV, traces each spot's outline with
``cv2.findContours`` and simplifies it with the Douglas-Peucker algorithm
(``cv2.approxPolyDP`` at ``epsilon = eps_frac * perimeter``) so simple shapes stay
cheap and jagged ones keep their detail.

Each spot row carries two representations: a centroid-local contour (compact, ordered
2-D points **normalized to the spot's own centroid** — rotation/translation friendly),
and ``mask_png`` — a full-frame binary mask (image dimensions, black everywhere except
that one spot, 255) rasterised from the spot's full-resolution contour and PNG-encoded.
PNG collapses the all-black background to almost nothing, so per-spot full-frame masks
stay small. Everything lands in a DuckDB database at ``<input_dir>/contours/contours.db``
(no separate masks/ directory).

Per-spot record (the ``spots`` table mirrors this shape; ``mask_png`` is a BLOB)::

    {
      "salamander_id": "aa_1",
      "image_dimensions": {"width": 1024, "height": 2048},
      "spots": [
        {
          "spot_id": 0,
          "global_centroid": [452.3, 1102.8],
          "area_pixels": 1432.5,
          "local_contour": [[-12, -45], [15, -42], [30, -10], [5, 35], [-20, 22]],
          "mask_png": b"<full-frame PNG: black except this spot>"
        }
      ]
    }

Pure logic: no argument parsing here. Call :func:`extract_dir` for a whole folder, or
:func:`process_purple_image` / :func:`extract_spots` per image. The CLI that configures
and calls it is ``scripts/extract_spot_labels.py contours`` (``pixi run extract-spot-labels
contours``).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ._common import (
    contours_db_for,
    purple_dir_for,
    resolve_input_dir,
)

# Magenta key in HSV (OpenCV hue 0..179; #FF00FF -> H=150, S=255, V=255).
# A band around 150 tolerates the anti-aliasing Gemini leaves at spot edges.
KEY_HSV_LOW = (140, 80, 80)
KEY_HSV_HIGH = (170, 255, 255)

DEFAULT_EPS_FRAC = 0.001  # Douglas-Peucker epsilon as a fraction of the perimeter
DEFAULT_MIN_AREA = 15.0   # drop specks smaller than this (px^2)
DEFAULT_MORPH = 3         # morphological kernel size to clean the mask (0 = off)


def magenta_mask(bgr: np.ndarray,
                 morph: int = DEFAULT_MORPH) -> np.ndarray:
    """Binary (0/255) mask of the flat-magenta spots in a BGR image."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(KEY_HSV_LOW, np.uint8),
                       np.array(KEY_HSV_HIGH, np.uint8))
    if morph and morph >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # kill single-pixel noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # seal thin gaps
    return mask


def spot_mask_png(cnt: np.ndarray, height: int, width: int) -> bytes:
    """PNG bytes of a full-frame binary mask: black everywhere except this one spot (255).

    Filled from the spot's full-resolution external contour, so the edge stays as crisp as
    the magenta key — no Douglas-Peucker smoothing. The all-black background makes the PNG
    tiny, so storing a per-spot full-frame mask per row is cheap.
    """
    canvas = np.zeros((height, width), np.uint8)
    cv2.drawContours(canvas, [cnt], -1, 255, thickness=cv2.FILLED)
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("failed to PNG-encode spot mask")
    return buf.tobytes()


def extract_spots(bgr: np.ndarray,
                  eps_frac: float = DEFAULT_EPS_FRAC,
                  min_area: float = DEFAULT_MIN_AREA,
                  morph: int = DEFAULT_MORPH) -> list[dict]:
    """Return spot records (centroid + area + centroid-local contour + full-frame mask)."""
    height, width = bgr.shape[:2]
    mask = magenta_mask(bgr, morph=morph)
    # RETR_EXTERNAL: outer outline only; CHAIN_APPROX_NONE: full-res before we
    # down-sample deliberately with approxPolyDP.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    spots: list[dict] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:  # degenerate (zero-area) contour
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]

        perimeter = cv2.arcLength(cnt, True)
        eps = max(eps_frac * perimeter, 0.5)  # never below sub-pixel
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float64)

        local = np.round(approx - (cx, cy), 2)
        spots.append({
            "global_centroid": [round(cx, 2), round(cy, 2)],
            "area_pixels": round(area, 2),
            "local_contour": local.tolist(),
            # full-frame mask from the *unsimplified* contour (sharp edges)
            "mask_png": spot_mask_png(cnt, height, width),
        })

    # Largest first, then assign stable spot_ids.
    spots.sort(key=lambda s: s["area_pixels"], reverse=True)
    for i, s in enumerate(spots):
        s["spot_id"] = i
    return spots


def build_record(purple_path: Path, salamander_id: str, spots: list[dict],
                 width: int, height: int, source_image: str | None = None) -> dict:
    return {
        "salamander_id": salamander_id,
        "image_dimensions": {"width": width, "height": height},
        "source_image": source_image,
        "purple_image": str(purple_path.name),
        "spots": [
            {
                "spot_id": s["spot_id"],
                "global_centroid": s["global_centroid"],
                "area_pixels": s["area_pixels"],
                "local_contour": s["local_contour"],
                "mask_png": s["mask_png"],
            }
            for s in spots
        ],
    }


class ContourStore:
    """DuckDB store for spot contours (one images row + N spots rows per image)."""

    def __init__(self, db_path: Path):
        import duckdb
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self.con = duckdb.connect(str(db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS images (
                salamander_id VARCHAR PRIMARY KEY,
                width         INTEGER,
                height        INTEGER,
                n_spots       INTEGER,
                source_image  VARCHAR,
                purple_image  VARCHAR,
                created_at    TIMESTAMP
            );
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS spots (
                salamander_id     VARCHAR,
                spot_id           INTEGER,
                global_centroid_x DOUBLE,
                global_centroid_y DOUBLE,
                area_pixels       DOUBLE,
                local_contour     DOUBLE[][],  -- ordered [[x,y],...] relative to centroid
                mask_png          BLOB,        -- full-frame PNG, black except this spot
                PRIMARY KEY (salamander_id, spot_id)
            );
        """)
        # Migrate DBs created before per-spot masks: add the column (as the last one, so
        # the positional INSERT still lines up). Existing rows get masks on their next run.
        cols = {r[0] for r in self.con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'spots'").fetchall()}
        if "mask_png" not in cols:
            self.con.execute("ALTER TABLE spots ADD COLUMN mask_png BLOB")

    def write(self, record: dict) -> None:
        """Idempotent upsert: replace all rows for this salamander_id."""
        sid = record["salamander_id"]
        dims = record["image_dimensions"]
        spots = record["spots"]
        self.con.execute("BEGIN")
        try:
            self.con.execute("DELETE FROM spots WHERE salamander_id = ?", [sid])
            self.con.execute("DELETE FROM images WHERE salamander_id = ?", [sid])
            self.con.execute(
                "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?)",
                [sid, dims["width"], dims["height"], len(spots),
                 record.get("source_image"), record.get("purple_image"),
                 datetime.now()],
            )
            for s in spots:
                gx, gy = s["global_centroid"]
                self.con.execute(
                    "INSERT INTO spots VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [sid, s["spot_id"], gx, gy, s["area_pixels"],
                     s["local_contour"], s["mask_png"]],
                )
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self.con.close()


def process_purple_image(purple_path: Path, store: ContourStore,
                         eps_frac: float = DEFAULT_EPS_FRAC,
                         min_area: float = DEFAULT_MIN_AREA,
                         morph: int = DEFAULT_MORPH,
                         source_image: str | None = None) -> dict:
    """Full stage 2 for one purple image: extract spots and persist to DuckDB."""
    bgr = cv2.imread(str(purple_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"could not read image: {purple_path}")
    height, width = bgr.shape[:2]
    spots = extract_spots(bgr, eps_frac=eps_frac, min_area=min_area, morph=morph)
    record = build_record(purple_path, purple_path.stem, spots, width, height,
                          source_image=source_image)
    store.write(record)
    return record


def extract_dir(
    input: str,
    *,
    eps_frac: float = DEFAULT_EPS_FRAC,
    min_area: float = DEFAULT_MIN_AREA,
    morph: int = DEFAULT_MORPH,
    limit: int | None = None,
) -> int:
    """Stage 2 over one input dir: extract spot contours from ``purple/*.png`` into DuckDB.

    Returns a process exit code (non-zero if the input dir has no purple images yet).
    """
    input_dir = resolve_input_dir(input)
    purple_dir = purple_dir_for(input_dir)
    if not purple_dir.is_dir():
        print(f"error: no purple/ dir in {input_dir} (run stage 1 first)", file=sys.stderr)
        return 1

    purples = sorted(p for p in purple_dir.iterdir()
                     if p.is_file() and p.suffix.lower() == ".png")
    if limit is not None:
        purples = purples[:limit]
    if not purples:
        print(f"error: no purple PNGs in {purple_dir}", file=sys.stderr)
        return 1

    db_path = contours_db_for(input_dir)
    store = ContourStore(db_path)
    print(f"processing purple images in {purple_dir}")
    total = len(purples)
    total_spots = 0
    try:
        for i, pp in enumerate(purples, start=1):
            print(f"processing image {i} / {total}: {pp.name}")
            rec = process_purple_image(pp, store, eps_frac=eps_frac,
                                       min_area=min_area, morph=morph)
            n = len(rec["spots"])
            total_spots += n
            print(f"  {n} spots -> db")
    finally:
        store.close()

    print(f"\ndone: {total_spots} spots from {total} images -> {db_path}")
    return 0
