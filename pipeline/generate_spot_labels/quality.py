#!/usr/bin/env python3
"""Cheap, model-free quality markers for each photo — for filtering a dataset after the fact.

Every metric here is a few lines of OpenCV/numpy over things already computed: the original
photo, the body mask, and the extracted spots. No model, no network, milliseconds per image. The
point is to be able to say later "drop the blurriest 10 %", "drop anything where the paint bled
off the animal", "keep only well-lit, well-segmented photos" — without re-running anything
expensive.

Two layers are stored, because they answer different questions:

* RAW measurements — the honest numbers (a Laplacian variance, a clipped-pixel fraction, an
  aspect ratio). Filter on these when you want an absolute, explainable threshold.
* COMPOSITE 0..1 scores — the four ``*_quality`` fields, each a documented heuristic blend of the
  raw numbers, higher = better. Filter on these when you just want "good enough". They are
  deliberately simple and the weights live in named constants at the top so you can retune them.

The composites are heuristics, not ground truth. ``blur_quality`` in particular is normalised
against the DATASET's own median sharpness (:func:`composite_scores`), because the absolute
Laplacian variance of a sharp photo depends on its resolution and content — a per-dataset
median makes "typical photo -> 0.5" hold regardless.

What each RAW metric means and why it is a useful filter:

  center_line_length_px  body length along the midline.
  avg_width_px           body_area / length — mean width of the animal as a ribbon.
  aspect_ratio           length / avg_width. A fire salamander sits in a band; a wild outlier is
                         usually a bad mask (too fat = leaked, too thin = only part painted).
  body_area_frac         mask / frame. Tiny = distant/small animal (few pixels of pattern);
                         huge = the fill escaped onto the background.
  solidity               mask / convex-hull area. Low = ragged outline, holes, or legs painted.
  border_frac            fraction of the frame edge the mask touches. >0 = animal cropped by the
                         frame, so its true extent (and some spots) are missing.
  min_dim_px             min(W, H). A crude resolution floor — pattern matching needs pixels.
  line_inside_frac       fraction of the midline that lies on the mask. <1 = the axis leaves the
                         body (mask is two blobs, or a bad tip) — the strongest body-mask alarm.
  curl_deg               mean turning angle along the midline. High = a tightly curled pose,
                         which is harder to match and more likely to carry axis error.
  blur_score             variance of the Laplacian WITHIN the body. Low = out of focus. Measured
                         inside the mask so a blurred background does not mask a blurred animal.
  mean_brightness        mean V (0..255) within the body.
  underexposed_frac      body pixels with V < 25 — detail lost in shadow.
  overexposed_frac       body pixels with V > 240 — detail blown out.
  glare_frac             body pixels that are bright AND desaturated (specular highlight). These
                         wet animals are shiny; glare erases the very spots we match on.
  pattern_contrast       Otsu split of body brightness: mean(bright) - mean(dark). Low = a flat,
                         washed-out pattern; high = crisp yellow-on-black.
  n_spots                spots extracted.
  spot_area_frac         spot area / body area — how much of the animal is yellow pattern. ~0 =
                         nothing extracted; absurdly high = over-segmentation or bleed.
  spots_outside_frac     fraction of spot centroids that fall OFF the body mask. >0 = the magenta
                         paint bled onto the background — a direct spot-extraction alarm.
  median_spot_area_px    typical spot size, for sanity.

Pure CV: cv2 + numpy, no network. The Gemini stages are long finished by the time this runs.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .binning import as_polyline, polyline_length

# --- composite weights ----------------------------------------------------------------------
# These say HOW MUCH each defect matters — design choices, not per-dataset knobs. The SCALE each
# defect is measured against (how much bleed / clipping / glare counts as "bad") is LEARNED from
# the dataset in :func:`calibrate`, so there is nothing here you have to retune when the dataset
# changes: a penalty term reaches ~one full unit at the dataset's own p90-worst image.
BLUR_MEDIAN_TO_HALF = 2.0    # blur: dataset-median sharpness -> 0.5, 2x median -> 1.0
BRIGHT_OK = (0.28, 0.72)     # mean-brightness band (fraction of 255) with no exposure penalty
W_OVER, W_UNDER, W_GLARE, W_BRIGHT = 0.6, 0.6, 0.5, 0.5   # lighting_quality penalty weights
W_BLEED = 0.8                # spot_extraction_quality: weight on paint bled off the body
W_BORDER = 0.5               # body_extraction_quality: weight on a frame-cropped animal
MIN_SPOTS = 2                # fewer than this and spot extraction is treated as failed
MIN_SPOT_COVERAGE = 0.01     # spot_area_frac below this = essentially nothing extracted
CALIB_PCTL = 0.90            # penalties scale so the dataset's p90-worst ~= one full unit
# absolute floors, so a clean dataset (p90 ~ 0 bleed) does not divide by almost-zero
FLOORS = {"bleed": 0.05, "over": 0.02, "under": 0.02, "glare": 0.02, "border": 0.01}


@dataclass
class Calibration:
    """The dataset-derived scales the composites are measured against. This IS the "tuning".

    Built by :func:`calibrate` from one pass over the dataset's raw metrics, so the scores adapt
    to how blurry / bled / clipped this particular set of photos actually is, with no hand-set
    magnitude constants. Printed by the tool and stored alongside the scores so a run is
    reproducible.
    """
    blur_median: float = 1.0
    bleed_scale: float = FLOORS["bleed"]
    over_scale: float = FLOORS["over"]
    under_scale: float = FLOORS["under"]
    glare_scale: float = FLOORS["glare"]
    border_scale: float = FLOORS["border"]

    def as_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def calibrate(raws: list[dict], pctl: float = CALIB_PCTL) -> Calibration:
    """Learn the per-metric scales from the dataset itself — the auto-tune pass.

    Sharpness is scaled by its median (so "typical -> 0.5"); each defect metric is scaled by its
    p90 across the dataset (so the worst ~10 % gets a full penalty unit), floored so a clean
    dataset does not produce a near-zero divisor.
    """
    if not raws:
        return Calibration()

    def p90(metric: str, floor: float) -> float:
        vals = sorted(r[metric] for r in raws)
        v = vals[min(int(pctl * (len(vals) - 1)), len(vals) - 1)]
        return max(v, floor)

    return Calibration(
        blur_median=max(statistics.median(r["blur_score"] for r in raws), 1e-6),
        bleed_scale=p90("spots_outside_frac", FLOORS["bleed"]),
        over_scale=p90("overexposed_frac", FLOORS["over"]),
        under_scale=p90("underexposed_frac", FLOORS["under"]),
        glare_scale=p90("glare_frac", FLOORS["glare"]),
        border_scale=p90("border_frac", FLOORS["border"]),
    )


def _otsu_contrast(vals: np.ndarray) -> float:
    """Otsu-split a 1-D intensity sample into dark/bright, return mean(bright) - mean(dark)."""
    if len(vals) < 16:
        return 0.0
    hist = cv2.calcHist([vals.astype(np.uint8)], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total <= 0:
        return 0.0
    idx = np.arange(256)
    w = np.cumsum(hist)
    mu = np.cumsum(hist * idx)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w - mu) ** 2 / (w * (total - w))
    t = int(np.nanargmax(between))
    dark, bright = vals[vals <= t], vals[vals > t]
    if len(dark) == 0 or len(bright) == 0:
        return 0.0
    return float(bright.mean() - dark.mean())


def _curl_deg(midline) -> float:
    """Mean absolute turning angle (deg) along the midline — how curled the pose is."""
    p = as_polyline(midline)
    if len(p) < 3:
        return 0.0
    d = p[1:] - p[:-1]
    n = np.hypot(d[:, 0], d[:, 1])
    d, n = d[n > 1e-6], n[n > 1e-6]
    if len(d) < 2:
        return 0.0
    u = d / n[:, None]
    return float(np.degrees(np.arccos(np.clip((u[:-1] * u[1:]).sum(1), -1, 1))).mean())


def raw_metrics(orig_bgr: np.ndarray, mask: np.ndarray, midline,
                spots: list[dict], length_px: float | None = None) -> dict:
    """All the RAW, dataset-independent numbers for one photo. Never raises.

    ``spots`` is a list of dicts with ``global_centroid`` (x, y) and ``area_pixels`` — exactly
    the shape stored in the DB. ``length_px`` overrides the midline length if you already have
    the stored value; otherwise it is measured from ``midline``.
    """
    H, W = orig_bgr.shape[:2]
    m = mask > 0
    body_area = int(m.sum())
    q: dict = {"body_area_px": body_area, "min_dim_px": int(min(W, H)),
               "body_area_frac": round(body_area / float(max(W * H, 1)), 5)}

    # --- shape ---
    length = float(length_px) if length_px else polyline_length(midline)
    avg_w = body_area / length if length > 1 else 0.0
    q["center_line_length_px"] = round(length, 1)
    q["avg_width_px"] = round(avg_w, 1)
    q["aspect_ratio"] = round(length / avg_w, 2) if avg_w > 1 else 0.0
    q["curl_deg"] = round(_curl_deg(midline), 1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        big = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(big))
        q["solidity"] = round(body_area / hull, 3) if hull > 1 else 0.0
    else:
        q["solidity"] = 0.0
    border = int(m[0, :].sum() + m[-1, :].sum() + m[:, 0].sum() + m[:, -1].sum())
    q["border_frac"] = round(border / float(2 * (W + H)), 4)

    # midline on the mask?
    poly = as_polyline(midline)
    if len(poly) >= 2:
        v = poly.astype(int)
        v[:, 0] = np.clip(v[:, 0], 0, W - 1)
        v[:, 1] = np.clip(v[:, 1], 0, H - 1)
        q["line_inside_frac"] = round(float((mask[v[:, 1], v[:, 0]] > 0).mean()), 3)
    else:
        q["line_inside_frac"] = 0.0

    # --- photo (blur / lighting), measured inside the body ---
    gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2HSV)
    if body_area > 16:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        q["blur_score"] = round(float(lap[m].var()), 1)
        V, S = hsv[:, :, 2], hsv[:, :, 1]
        vb = V[m]
        q["mean_brightness"] = round(float(vb.mean()), 1)
        q["underexposed_frac"] = round(float((vb < 25).mean()), 4)
        q["overexposed_frac"] = round(float((vb > 240).mean()), 4)
        q["glare_frac"] = round(float(((V > 230) & (S < 40))[m].mean()), 4)
        q["pattern_contrast"] = round(_otsu_contrast(vb), 1)
    else:
        q.update(blur_score=0.0, mean_brightness=0.0, underexposed_frac=1.0,
                 overexposed_frac=0.0, glare_frac=0.0, pattern_contrast=0.0)

    # --- spots ---
    areas = [float(s.get("area_pixels") or 0) for s in spots]
    q["n_spots"] = len(spots)
    q["spot_area_frac"] = round(sum(areas) / float(max(body_area, 1)), 4)
    q["median_spot_area_px"] = round(float(np.median(areas)), 1) if areas else 0.0
    outside = 0
    for s in spots:
        cx, cy = s["global_centroid"]
        ix, iy = min(max(int(cx), 0), W - 1), min(max(int(cy), 0), H - 1)
        if not m[iy, ix]:
            outside += 1
    q["spots_outside_frac"] = round(outside / float(max(len(spots), 1)), 3)
    return q


def composite_scores(raw: dict, calib: Calibration) -> dict:
    """The four named 0..1 quality scores + an overall, from one photo's RAW metrics.

    ``calib`` is the dataset-derived scaling from :func:`calibrate` — that is where the tuning
    lives, so this function has no per-dataset magic left, only the importance weights at the top
    of the module. Higher = better for every score.
    """
    clip01 = lambda x: float(min(max(x, 0.0), 1.0))

    # sharpness: dataset median -> 0.5, BLUR_MEDIAN_TO_HALF x median -> 1.0.
    blur_q = clip01((raw["blur_score"] / calib.blur_median) / BLUR_MEDIAN_TO_HALF)

    # lighting: each exposure defect measured against its dataset p90, then weighted.
    b = raw["mean_brightness"] / 255.0
    bright_pen = max(0.0, BRIGHT_OK[0] - b, b - BRIGHT_OK[1]) / BRIGHT_OK[0]
    lighting_q = clip01(1.0
                        - W_OVER * (raw["overexposed_frac"] / calib.over_scale)
                        - W_UNDER * (raw["underexposed_frac"] / calib.under_scale)
                        - W_GLARE * (raw["glare_frac"] / calib.glare_scale)
                        - W_BRIGHT * bright_pen)

    spot_q = clip01(1.0 - W_BLEED * (raw["spots_outside_frac"] / calib.bleed_scale))
    if raw["n_spots"] < MIN_SPOTS:
        spot_q *= 0.3
    if raw["spot_area_frac"] < MIN_SPOT_COVERAGE:
        spot_q *= 0.5

    judged = raw.get("judged_ok")
    judged_term = 1.0 if judged else (0.5 if judged is None else 0.4)
    body_q = clip01(0.35 * judged_term + 0.35 * raw["solidity"] + 0.30 * raw["line_inside_frac"]
                    - W_BORDER * (raw["border_frac"] / calib.border_scale))

    scores = {"blur_quality": round(blur_q, 3), "lighting_quality": round(lighting_q, 3),
              "spot_extraction_quality": round(spot_q, 3),
              "body_extraction_quality": round(body_q, 3)}
    # overall = geometric mean, so any single bad axis drags it down (good for filtering)
    prod = 1.0
    for v in scores.values():
        prod *= max(v, 1e-4)
    scores["overall_quality"] = round(prod ** 0.25, 3)
    return scores


# The DB columns, in order — shared by the store's CREATE and INSERT so they cannot drift.
RAW_FIELDS = [
    ("center_line_length_px", "DOUBLE"), ("avg_width_px", "DOUBLE"), ("aspect_ratio", "DOUBLE"),
    ("body_area_px", "INTEGER"), ("body_area_frac", "DOUBLE"), ("solidity", "DOUBLE"),
    ("border_frac", "DOUBLE"), ("min_dim_px", "INTEGER"), ("line_inside_frac", "DOUBLE"),
    ("curl_deg", "DOUBLE"), ("blur_score", "DOUBLE"), ("mean_brightness", "DOUBLE"),
    ("underexposed_frac", "DOUBLE"), ("overexposed_frac", "DOUBLE"), ("glare_frac", "DOUBLE"),
    ("pattern_contrast", "DOUBLE"), ("n_spots", "INTEGER"), ("spot_area_frac", "DOUBLE"),
    ("spots_outside_frac", "DOUBLE"), ("median_spot_area_px", "DOUBLE"),
    ("axis_source", "VARCHAR"), ("judged_ok", "BOOLEAN"),
]
SCORE_FIELDS = [
    ("blur_quality", "DOUBLE"), ("lighting_quality", "DOUBLE"),
    ("spot_extraction_quality", "DOUBLE"), ("body_extraction_quality", "DOUBLE"),
    ("overall_quality", "DOUBLE"),
]
