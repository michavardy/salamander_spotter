#!/usr/bin/env python3
"""Stage 2 — isolate the magenta spots and store their contours + masks in DuckDB.

Reads the purple images produced by stage 1 (``<input_dir>/purple/*.png``) and keys out the
spots, confined to the body mask. The default ``adaptive`` key scores each pixel's redness
(``r-g``, high for MAGENTA #FF00FF and RED/crimson alike, which Gemini uses across images) and
picks a PER-IMAGE Otsu threshold inside the body, so spots Gemini shaded or washed out stay
whole instead of shattering into fragments; the original fixed-floor ``hsv`` key is kept as a
fallback (see :func:`spot_mask`). Each spot's outline is then traced with ``cv2.findContours``
and simplified with the Douglas-Peucker algorithm (``cv2.approxPolyDP`` at
``epsilon = eps_frac * perimeter``) so simple shapes stay cheap and jagged ones keep detail.

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
and calls it is ``scripts/dataset/extract_spot_labels.py contours`` (``pixi run extract-spot-labels
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
from .binning import bin_polygons, bin_spots

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

# Spot key in HSV (OpenCV hue 0..179). Gemini paints the flat spots either MAGENTA
# (#FF00FF -> H~150) or, on a large fraction of images, RED/CRIMSON (H~0-10). Keying only
# magenta missed the red-spot images (spots came out as tiny fragments or nothing), so we
# key BOTH bands. The red band also matches red leaf-litter in the background, so it is used
# ONLY when a body mask confines the key to the animal (see ``magenta_mask``).
KEY_S_MIN = 90            # spots are flat + saturated; this rejects dull background
KEY_V_MIN = 80
MAGENTA_BAND = (140, 179)  # magenta / pink — always keyed
RED_BAND = (0, 13)         # red / crimson — only keyed inside the body mask

DEFAULT_EPS_FRAC = 0.001  # Douglas-Peucker epsilon as a fraction of the perimeter
DEFAULT_MIN_AREA = 15.0   # drop specks smaller than this (px^2)
DEFAULT_MORPH = 3         # morphological OPEN kernel to kill single-pixel noise (0 = off)
DEFAULT_CLOSE = 5         # morphological CLOSE kernel (adaptive key) — seals shading pin-holes

# Two keys. 'adaptive' (default) is lighting-tolerant: it scores each pixel's redness (r-g,
# high for both magenta AND red spots, ~0 for green/yellow) and picks a PER-IMAGE threshold
# with Otsu *inside the body mask*, so a spot Gemini painted dark, washed-out or shaded stays
# one blob instead of shattering into a swarm of tiny fragments. 'hsv' is the original fixed
# S/V-floor key, kept as a fallback (and used automatically when no body mask is available,
# since the adaptive threshold needs the animal region to place itself).
DEFAULT_KEY_MODE = "adaptive"
KEY_MODES = ("adaptive", "hsv")
ADAPTIVE_MIN_SCORE = 8    # r-g at/below this is un-painted body, never spot (floors the Otsu cut)

# QC — a heuristic that flags likely FRAGMENTATION (a spot broken into many pieces), so a
# follow-up pass can re-key or re-send just those images. A genuinely spotty animal has many
# spots too, so count alone is not enough: the tell is many spots that are almost all tiny.
QC_MIN_SPOTS = 40         # fewer than this and a small median is just a small animal
QC_MEDIAN_AREA = 150.0    # px^2: this many spots, nearly all near the floor == shattering


def _hue_mask(hsv: np.ndarray, bands, s_min: int, v_min: int) -> np.ndarray:
    """Union of several HSV hue bands (each ``(h_lo, h_hi)``) into one 0/255 mask."""
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for h_lo, h_hi in bands:
        mask |= cv2.inRange(hsv, np.array([h_lo, s_min, v_min], np.uint8),
                            np.array([h_hi, 255, 255], np.uint8))
    return mask


def magenta_mask(bgr: np.ndarray,
                 morph: int = DEFAULT_MORPH,
                 body_mask: np.ndarray | None = None) -> np.ndarray:
    """Binary (0/255) mask of the flat spots in a BGR (purple) image.

    Always keys magenta. When ``body_mask`` (a 0/255 array the size of ``bgr``) is given, the
    RED band is added too and the whole key is AND-ed with the mask, so red spots are recovered
    without also grabbing red leaf-litter outside the animal. Without a body mask, only magenta
    is keyed (the safe, original behaviour).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if body_mask is not None:
        mask = _hue_mask(hsv, (MAGENTA_BAND, RED_BAND), KEY_S_MIN, KEY_V_MIN)
        mask = cv2.bitwise_and(mask, body_mask)
    else:
        mask = _hue_mask(hsv, (MAGENTA_BAND,), KEY_S_MIN, KEY_V_MIN)
    if morph and morph >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # kill single-pixel noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # seal thin gaps
    return mask


def _redness(bgr: np.ndarray) -> np.ndarray:
    """Per-pixel spot score r-g (0..255). High for magenta AND red spots, ~0 for green/yellow.

    Unlike HSV saturation/value this does not collapse when Gemini shades or washes out a spot:
    a dark magenta patch still has r well above g, so it keeps a strong score.
    """
    b, g, r = (bgr[..., 0].astype(np.int16), bgr[..., 1].astype(np.int16),
               bgr[..., 2].astype(np.int16))
    return np.clip(r - g, 0, 255).astype(np.uint8)


def adaptive_spot_mask(bgr: np.ndarray, body_mask: np.ndarray,
                       open_k: int = DEFAULT_MORPH, close_k: int = DEFAULT_CLOSE) -> np.ndarray:
    """Binary (0/255) spot mask from a PER-IMAGE Otsu cut on the redness score, inside the body.

    Otsu is computed only over painted body pixels (score above :data:`ADAPTIVE_MIN_SCORE`), so
    the split it finds is spot-vs-body for *this* image's lighting rather than a global guess.
    The result is AND-ed with ``body_mask`` (keeps red leaf-litter out), OPEN-ed to drop specks
    and CLOSE-d to seal the pin-holes that shading would otherwise punch through a spot.
    """
    score = _redness(bgr)
    inside = score[body_mask > 127]
    inside = inside[inside > ADAPTIVE_MIN_SCORE]
    if inside.size < 50:                     # no real spot signal on this animal
        return np.zeros(bgr.shape[:2], np.uint8)
    thr, _ = cv2.threshold(inside, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cut = max(int(thr), ADAPTIVE_MIN_SCORE)
    mask = np.where(score >= cut, np.uint8(255), np.uint8(0))
    mask = cv2.bitwise_and(mask, body_mask)
    if open_k and open_k >= 3:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    if close_k and close_k >= 3:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    return mask


def spot_mask(bgr: np.ndarray, key_mode: str = DEFAULT_KEY_MODE, morph: int = DEFAULT_MORPH,
              close: int = DEFAULT_CLOSE, body_mask: np.ndarray | None = None) -> np.ndarray:
    """Dispatch to the chosen spot key. ``adaptive`` needs a body mask; without one it falls
    back to the ``hsv`` key so a mask-less image still extracts (just less lighting-tolerant)."""
    if key_mode == "adaptive" and body_mask is not None:
        return adaptive_spot_mask(bgr, body_mask, open_k=morph, close_k=close)
    return magenta_mask(bgr, morph=morph, body_mask=body_mask)


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
                  morph: int = DEFAULT_MORPH,
                  body_mask: np.ndarray | None = None,
                  key_mode: str = DEFAULT_KEY_MODE,
                  close: int = DEFAULT_CLOSE) -> list[dict]:
    """Return spot records (centroid + area + centroid-local contour + full-frame mask).

    ``body_mask`` (0/255, image-sized) confines the colour key to the animal and enables the
    red-spot band — see :func:`magenta_mask`. ``key_mode`` picks the key (:data:`KEY_MODES`);
    ``close`` is the CLOSE kernel for the adaptive key. See :func:`spot_mask`.
    """
    height, width = bgr.shape[:2]
    mask = spot_mask(bgr, key_mode=key_mode, morph=morph, close=close, body_mask=body_mask)
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
                 width: int, height: int, source_image: str | None = None,
                 anatomy=None, body_mask_png: bytes | None = None) -> dict:
    """Assemble the DB record. With an ``Anatomy`` (stage 1b), every spot is also binned.

    Binning happens here rather than in the store so the caller gets the bins back in the
    returned record (the runner prints the histogram from it).
    """
    body_axis = body_bins = None
    if anatomy is not None and anatomy.ok:
        bin_spots(spots, anatomy.midline)
        body_axis = {
            "head": anatomy.head, "tail_tip": anatomy.tail_tip, "midline": anatomy.midline,
            "left": anatomy.left, "right": anatomy.right,
            "length_px": round(anatomy.length_px, 2), "source": anatomy.source,
            "judged_ok": anatomy.judged_ok, "judge_feedback": anatomy.judge_feedback,
        }
        body_bins = bin_polygons(anatomy.midline, width, height)
    else:
        bin_spots(spots, None)              # explicit NULLs — never a stale bin

    return {
        "salamander_id": salamander_id,
        "image_dimensions": {"width": width, "height": height},
        "source_image": source_image,
        "purple_image": str(purple_path.name),
        "body_mask_png": body_mask_png,
        "body_axis": body_axis,
        "body_bins": body_bins,
        "spots": [
            {
                "spot_id": s["spot_id"],
                "global_centroid": s["global_centroid"],
                "area_pixels": s["area_pixels"],
                "local_contour": s["local_contour"],
                "mask_png": s["mask_png"],
                "axial_bin": s.get("axial_bin"),
                "lateral_bin": s.get("lateral_bin"),
                "bin": s.get("bin"),
                "axis_t": s.get("axis_t"),
                "axis_side": s.get("axis_side"),
                "axis_offset": s.get("axis_offset"),
            }
            for s in spots
        ],
    }


class ContourStore:
    """DuckDB store for spot contours + the body grid.

    Three tables, all keyed by ``salamander_id``:
      ``images``     one row per photo.
      ``spots``      one row per spot: geometry, mask, and its body-grid ``bin`` (1..8).
      ``body_axis``  one row per photo: head, tail tip, and the midline polyline between
                     them (stage 1b) — the axis every bin is measured against.
      ``body_bins``  one row per box (8 per photo): the grid geometry as a clipped polygon.
    """

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
                body_mask_png BLOB,        -- the WHOLE-body mask (anatomy/<stem>_mask.png)
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
                axial_bin         INTEGER,     -- 1..4: quarter of the body, 1 = head end
                lateral_bin       VARCHAR,     -- 'left'|'right'|'overlap' (straddles the line)
                bin               INTEGER,     -- 1..8 = axial_bin x lateral_bin; NULL if overlap
                axis_t            DOUBLE,      -- 0 at the head, 1 at the tail tip
                axis_side         VARCHAR,     -- the CENTROID's side, always left/right
                axis_offset       DOUBLE,      -- signed px from the midline (+ = left)
                PRIMARY KEY (salamander_id, spot_id)
            );
        """)
        # One head->tail axis per photo: the geometry every bin is measured against.
        # The midline is DERIVED as the mean of the two body outlines Gemini drew (left/right),
        # so it bisects the animal by construction. `judged_ok` is the LLM judge's verdict on
        # those outlines. A rejected axis is still binned — but it is marked, not hidden.
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS body_axis (
                salamander_id  VARCHAR PRIMARY KEY,
                head_x         DOUBLE,
                head_y         DOUBLE,
                tail_tip_x     DOUBLE,
                tail_tip_y     DOUBLE,
                length_px      DOUBLE,
                midline_x      DOUBLE[],   -- derived centre line, head -> tail tip
                midline_y      DOUBLE[],
                left_x         DOUBLE[],   -- the cyan body outline, as drawn
                left_y         DOUBLE[],
                right_x        DOUBLE[],   -- the magenta body outline, as drawn
                right_y        DOUBLE[],
                source         VARCHAR,    -- 'outlines' | 'none'
                judged_ok      BOOLEAN,    -- the judge accepted the outlines (NULL = not judged)
                judge_feedback VARCHAR     -- why they were rejected, if they were
            );
        """)
        # Migrate DBs created before the outlines were stored.
        axis_cols = {r[0] for r in self.con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'body_axis'").fetchall()}
        for name in ("left_x", "left_y", "right_x", "right_y"):
            if name not in axis_cols:
                self.con.execute(f'ALTER TABLE body_axis ADD COLUMN "{name}" DOUBLE[]')
        # The 8 boxes per photo, as drawn.
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS body_bins (
                bin_id        VARCHAR PRIMARY KEY,   -- '<salamander_id>_b<bin>'
                salamander_id VARCHAR,
                bin           INTEGER,
                quartile      INTEGER,
                side          VARCHAR,
                t_lo          DOUBLE,
                t_hi          DOUBLE,
                polygon_x     DOUBLE[],
                polygon_y     DOUBLE[]
            );
        """)
        # Migrate DBs created before the whole-body mask was stored.
        img_cols = {r[0] for r in self.con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'images'").fetchall()}
        if "body_mask_png" not in img_cols:
            self.con.execute('ALTER TABLE images ADD COLUMN body_mask_png BLOB')
        # Migrate DBs created before per-spot masks / the body grid. Columns are appended,
        # so every INSERT below names its columns explicitly rather than relying on order.
        cols = {r[0] for r in self.con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'spots'").fetchall()}
        for name, sql_type in (("mask_png", "BLOB"), ("bin", "INTEGER"),
                               ("axial_bin", "INTEGER"), ("lateral_bin", "VARCHAR"),
                               ("axis_t", "DOUBLE"), ("axis_side", "VARCHAR"),
                               ("axis_offset", "DOUBLE")):
            if name not in cols:
                self.con.execute(f'ALTER TABLE spots ADD COLUMN "{name}" {sql_type}')

    def write(self, record: dict) -> None:
        """Idempotent upsert: replace every row for this salamander_id, across all 4 tables."""
        sid = record["salamander_id"]
        dims = record["image_dimensions"]
        spots = record["spots"]
        axis = record.get("body_axis")          # None when no usable anatomy was recovered
        bins = record.get("body_bins") or []
        self.con.execute("BEGIN")
        try:
            for table in ("spots", "images", "body_axis", "body_bins"):
                self.con.execute(f"DELETE FROM {table} WHERE salamander_id = ?", [sid])
            self.con.execute(
                "INSERT INTO images (salamander_id, width, height, n_spots, source_image, "
                "purple_image, body_mask_png, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [sid, dims["width"], dims["height"], len(spots),
                 record.get("source_image"), record.get("purple_image"),
                 record.get("body_mask_png"), datetime.now()],
            )
            for s in spots:
                gx, gy = s["global_centroid"]
                self.con.execute(
                    'INSERT INTO spots (salamander_id, spot_id, global_centroid_x, '
                    'global_centroid_y, area_pixels, local_contour, mask_png, axial_bin, '
                    'lateral_bin, "bin", axis_t, axis_side, axis_offset) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    [sid, s["spot_id"], gx, gy, s["area_pixels"], s["local_contour"],
                     s["mask_png"], s.get("axial_bin"), s.get("lateral_bin"), s.get("bin"),
                     s.get("axis_t"), s.get("axis_side"), s.get("axis_offset")],
                )
            if axis is not None:
                self._insert_axis(sid, axis)
            self._insert_bins(sid, bins)
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    def _insert_axis(self, sid: str, axis: dict) -> None:
        """One body_axis row. Caller must have deleted any existing row for ``sid``."""
        xs = lambda key: [float(p[0]) for p in axis.get(key) or []]
        ys = lambda key: [float(p[1]) for p in axis.get(key) or []]
        self.con.execute(
            "INSERT INTO body_axis (salamander_id, head_x, head_y, tail_tip_x, "
            "tail_tip_y, length_px, midline_x, midline_y, left_x, left_y, right_x, "
            "right_y, source, judged_ok, judge_feedback) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [sid, axis["head"][0], axis["head"][1],
             axis["tail_tip"][0], axis["tail_tip"][1], axis["length_px"],
             xs("midline"), ys("midline"), xs("left"), ys("left"),
             xs("right"), ys("right"), axis["source"],
             axis.get("judged_ok"), axis.get("judge_feedback") or ""],
        )

    def _insert_bins(self, sid: str, bins: list[dict]) -> None:
        """The 8 body_bins rows. Caller must have deleted any existing rows for ``sid``."""
        for b in bins:
            self.con.execute(
                'INSERT INTO body_bins (bin_id, salamander_id, "bin", quartile, side, '
                't_lo, t_hi, polygon_x, polygon_y) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                [f"{sid}_b{b['bin']}", sid, b["bin"], b["quartile"], b["side"],
                 b["t_lo"], b["t_hi"],
                 [float(p[0]) for p in b["polygon"]],
                 [float(p[1]) for p in b["polygon"]]],
            )

    def rebin_axis(self, salamander_id: str, axis: dict | None, midline) -> int:
        """Re-apply a CORRECTED axis to an image already in the DB, without re-extracting spots.

        The axis-correction pass (``correct-axis``) moves only the centre line, never the spots
        themselves — so re-running the whole of stage 2 (decode the purple, re-trace every spot,
        re-encode every per-spot mask) to change a few bins would be gratuitous. This updates the
        one image in place: replace its ``body_axis`` and ``body_bins`` rows, and recompute each
        spot's bin from the new ``midline`` using the contour already stored. Returns spots
        rebinned. No-op-safe if the image is not in the DB.
        """
        dims = self.con.execute(
            "SELECT width, height FROM images WHERE salamander_id = ?", [salamander_id]).fetchone()
        if dims is None:
            return 0
        width, height = dims
        rows = self.con.execute(
            "SELECT spot_id, global_centroid_x, global_centroid_y, local_contour "
            "FROM spots WHERE salamander_id = ?", [salamander_id]).fetchall()
        spots = [{"spot_id": r[0], "global_centroid": (r[1], r[2]), "local_contour": r[3]}
                 for r in rows]
        bin_spots(spots, midline)                       # None midline -> all NULL, never stale
        bins = bin_polygons(midline, width, height) if midline else []

        self.con.execute("BEGIN")
        try:
            for s in spots:
                self.con.execute(
                    'UPDATE spots SET axial_bin = ?, lateral_bin = ?, "bin" = ?, axis_t = ?, '
                    'axis_side = ?, axis_offset = ? WHERE salamander_id = ? AND spot_id = ?',
                    [s.get("axial_bin"), s.get("lateral_bin"), s.get("bin"), s.get("axis_t"),
                     s.get("axis_side"), s.get("axis_offset"), salamander_id, s["spot_id"]])
            self.con.execute("DELETE FROM body_axis WHERE salamander_id = ?", [salamander_id])
            self.con.execute("DELETE FROM body_bins WHERE salamander_id = ?", [salamander_id])
            if axis is not None:
                self._insert_axis(salamander_id, axis)
            self._insert_bins(salamander_id, bins)
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        return len(spots)

    def close(self) -> None:
        self.con.close()


def spot_qc(spots: list[dict]) -> dict:
    """Heuristic fragmentation check: ``{n_spots, median_area, flagged}``.

    ``flagged`` is True when an image has many spots (:data:`QC_MIN_SPOTS`) that are almost all
    tiny (median area below :data:`QC_MEDIAN_AREA`) — the signature of one real spot shattered
    into fragments, as opposed to a genuinely spotty animal (many spots, but a healthy median).
    """
    n = len(spots)
    med = float(np.median([s["area_pixels"] for s in spots])) if n else 0.0
    return {"n_spots": n, "median_area": round(med, 1),
            "flagged": n >= QC_MIN_SPOTS and med < QC_MEDIAN_AREA}


def process_purple_image(purple_path: Path, store: ContourStore,
                         eps_frac: float = DEFAULT_EPS_FRAC,
                         min_area: float = DEFAULT_MIN_AREA,
                         morph: int = DEFAULT_MORPH,
                         source_image: str | None = None,
                         anatomy=None, body_mask_png: bytes | None = None,
                         key_mode: str = DEFAULT_KEY_MODE,
                         close: int = DEFAULT_CLOSE) -> dict:
    """Full stage 2 for one purple image: extract spots, bin them, persist to DuckDB.

    ``anatomy`` is the stage-1b :class:`~.llm_anatomy.Anatomy` for this image (or None). Its
    coordinates are in the ORIGINAL photo's frame, which is also the purple image's frame —
    stage 1 conforms Gemini's output back onto the original's pixel grid — so the midline and
    the spot centroids are directly comparable with no transform.

    ``body_mask_png`` is the whole-body mask (``anatomy/<stem>_mask.png``) as raw PNG bytes,
    stored verbatim on the ``images`` row so the mask travels inside the packaged DB rather than
    only living loose on disk. None if it was never produced.
    """
    bgr = cv2.imread(str(purple_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"could not read image: {purple_path}")
    height, width = bgr.shape[:2]
    # Decode the body mask so extract_spots can confine the colour key to the animal (and so
    # add the red-spot band). NEAREST resize keeps it binary if dimensions ever differ.
    body_mask = None
    if body_mask_png is not None:
        bm = cv2.imdecode(np.frombuffer(body_mask_png, np.uint8), cv2.IMREAD_GRAYSCALE)
        if bm is not None:
            if bm.shape != (height, width):
                bm = cv2.resize(bm, (width, height), interpolation=cv2.INTER_NEAREST)
            body_mask = np.where(bm > 127, np.uint8(255), np.uint8(0))
    spots = extract_spots(bgr, eps_frac=eps_frac, min_area=min_area, morph=morph,
                          body_mask=body_mask, key_mode=key_mode, close=close)
    record = build_record(purple_path, purple_path.stem, spots, width, height,
                          source_image=source_image, anatomy=anatomy,
                          body_mask_png=body_mask_png)
    record["qc"] = spot_qc(spots)          # not persisted — the runner reads it to flag fragments
    store.write(record)
    return record


def bin_histogram(record: dict) -> dict:
    """{bin: count} over the 8 boxes — overlapping spots have no box, so they are absent."""
    hist: dict = {}
    for s in record["spots"]:
        if s.get("bin") is not None:
            hist[s["bin"]] = hist.get(s["bin"], 0) + 1
    return dict(sorted(hist.items()))


def position_summary(record: dict) -> str:
    """'axial{1:6 2:9 …} lateral{left:20 right:15 overlap:4}' — the per-image log line.

    An overlapping spot IS positioned (it has an axial_bin and lateral='overlap'); it just has
    no 8-box `bin`. So it is counted here, unlike in `bin_histogram`.
    """
    axial: dict = {}
    lateral: dict = {}
    for s in record["spots"]:
        if s.get("axial_bin") is not None:
            axial[s["axial_bin"]] = axial.get(s["axial_bin"], 0) + 1
        if s.get("lateral_bin") is not None:
            lateral[s["lateral_bin"]] = lateral.get(s["lateral_bin"], 0) + 1
    a = " ".join(f"{k}:{v}" for k, v in sorted(axial.items()))
    order = {"left": 0, "right": 1, "overlap": 2}
    lat = " ".join(f"{k}:{v}" for k, v in sorted(lateral.items(), key=lambda kv: order.get(kv[0], 9)))
    return f"axial{{{a}}} lateral{{{lat}}}"


def n_positioned(record: dict) -> int:
    """How many spots got positional labels (overlapping spots included)."""
    return sum(1 for s in record["spots"] if s.get("axial_bin") is not None)


FLAGGED_SPOTS_CSV = "flagged_spots.csv"


def write_flagged_spots_csv(purple_dir: Path, names: list[str]) -> Path:
    """Write the fragmentation flag list next to the purple images. Feeds ``--rewrite`` /
    a re-key run so a follow-up redoes only these. Returns the CSV path."""
    path = purple_dir / FLAGGED_SPOTS_CSV
    path.write_text(
        "# images whose spots look FRAGMENTED (many tiny spots — likely shaded/washed-out\n"
        "# magenta the key shattered); re-key or regenerate just these with --rewrite\n"
        + "\n".join(names) + "\n",
        encoding="utf-8")
    return path


def extract_dir(
    input: str,
    *,
    eps_frac: float = DEFAULT_EPS_FRAC,
    min_area: float = DEFAULT_MIN_AREA,
    morph: int = DEFAULT_MORPH,
    key_mode: str = DEFAULT_KEY_MODE,
    close: int = DEFAULT_CLOSE,
    limit: int | None = None,
) -> int:
    """Stage 2 over one input dir: extract spot contours from ``purple/*.png`` into DuckDB.

    Picks up each image's stage-1b anatomy from ``anatomy/<stem>.json`` when it is there, so
    the spots get binned; images without one keep NULL bins rather than a guessed axis.
    Returns a process exit code (non-zero if the input dir has no purple images yet).
    """
    from .llm_anatomy import anatomy_dir_for, load_anatomy

    input_dir = resolve_input_dir(input)
    purple_dir = purple_dir_for(input_dir)
    if not purple_dir.is_dir():
        logger.error(f"error: no purple/ dir in {input_dir} (run stage 1 first)")
        return 1

    purples = sorted(p for p in purple_dir.iterdir()
                     if p.is_file() and p.suffix.lower() == ".png")
    if limit is not None:
        purples = purples[:limit]
    if not purples:
        logger.error(f"error: no purple PNGs in {purple_dir}")
        return 1

    anatomy_dir = anatomy_dir_for(input_dir)
    db_path = contours_db_for(input_dir)
    store = ContourStore(db_path)
    logger.info(f"processing purple images in {purple_dir}")
    total = len(purples)
    total_spots = 0
    total_binned = 0
    no_axis = 0
    flagged: list[str] = []      # images whose spots look fragmented — for a re-key/rewrite pass
    try:
        for i, pp in enumerate(purples, start=1):
            logger.info(f"processing image {i} / {total}: {pp.name}")
            anatomy = load_anatomy(anatomy_dir, pp.stem)
            mask_path = anatomy_dir / f"{pp.stem}_mask.png"
            rec = process_purple_image(
                pp, store, eps_frac=eps_frac, min_area=min_area, morph=morph,
                anatomy=anatomy, key_mode=key_mode, close=close,
                body_mask_png=mask_path.read_bytes() if mask_path.is_file() else None)
            n = len(rec["spots"])
            total_spots += n
            qc = rec["qc"]
            frag = "  [FRAGMENTED?]" if qc["flagged"] else ""
            if qc["flagged"]:
                flagged.append(pp.name)
            if rec["body_axis"] is None:
                no_axis += 1
                logger.info(f"  {n} spots (median {qc['median_area']:.0f}px) -> db "
                      f"(no anatomy — spots unpositioned){frag}")
            else:
                binned = n_positioned(rec)
                total_binned += binned
                logger.info(f"  {n} spots (median {qc['median_area']:.0f}px) -> db, "
                      f"{binned} positioned  {position_summary(rec)}{frag}")
    finally:
        store.close()

    if flagged:
        csv = write_flagged_spots_csv(purple_dir, flagged)
        logger.info(f"flagged {len(flagged)} image(s) for fragmentation -> {csv}")
    logger.info(f"done: {total_spots} spots ({total_binned} positioned) from {total} images "
          f"-> {db_path}"
          + (f", {len(flagged)} flagged for fragmentation" if flagged else "")
          + (f", {no_axis} image(s) had no anatomy" if no_axis else ""))
    return 0
