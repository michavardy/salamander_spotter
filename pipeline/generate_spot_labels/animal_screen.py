#!/usr/bin/env python3
"""Stage 0, layer 1 — the FREE screen: can we settle "one animal?" without a billed call?

This is the cheap half of the two-layer cascade. It reads artefacts that already exist on disk
(the body mask, the derived midline, the spot table) and tries to clear each frame as a single
animal. A frame it clears costs nothing; a frame it cannot clear is handed to
:mod:`llm_animal_count`, which pays for an answer.

The screen is deliberately built to be WRONG IN ONE DIRECTION ONLY. Its job is not to find
multi-animal frames — the measurements below say it cannot do that reliably — but to certify the
easy majority as single so the model is only paid for the hard minority. Every uncertainty
resolves to "defer", including missing inputs.

Two gates, because there are two failure modes with opposite signatures
----------------------------------------------------------------------
**Overlapping animals -> gate 1, off-axis mass.** The fill spans both animals, so the mask comes
back as a Y: the midline tracks one animal and a whole second body hangs off it. Measured as the
depth-weighted mask area lying further than ``K`` half-widths from the midline, split into
connected blobs, keeping the largest. The three factors are multiplied because a second animal
must be all three things at once —

    blob_frac    how much of the body's mass is out there      (a leg is a few percent)
    blob_depth   how THICK it is, in half-widths               (a leg is thin)
    blob_len     how LONG it is, in half-widths                (a leg is short)

— and no single factor separates them: legs are off-axis too, which is exactly why plain
``solidity`` fails (see below).

**Separate animals -> gate 2, spots outside the body.** ``solidify()`` keeps only the largest
blob, so a second animal standing clear of the first is *deleted from the mask* and gate 1 is
structurally blind to it. But stage 1 paints every yellow spot in the frame, not just the
subject's, so the other animal's spots are still extracted and land off the mask. That is
``image_quality.spots_outside_frac``, already computed.

What was measured, and why it is a screen rather than a detector
----------------------------------------------------------------
Ranked over 1265 real photos, off-axis mass puts the confirmed two-animal frame ``ca_14_5``
first and separates it from the pack — but ranks eleven single animals in a tight C-curl above
everything else, because the arms of a curl also sit far from the midline. The legacy
``solidity`` x spot-count heuristic is worse: same #1, then eleven single animals with splayed
legs.

So neither number is trustworthy as a verdict, and this module never treats them as one. What
they ARE good for is the other end of the distribution: a frame with essentially no off-axis
mass and no spots off the body has no room to hide a second animal, and that is most of the
corpus. Cost falls by whatever fraction clears; accuracy on the deferred remainder is the
model's problem, not this module's.

``ca_14_5`` is wired in as :data:`KNOWN_MULTI` and its verdict is printed on every run. If the
screen ever clears it, the screen is broken and the run says so.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from ._common import contours_db_for, resolve_input_dir

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

# --- tuning -----------------------------------------------------------------
RIBBON_K = 1.35          # ribbon half-width, in units of the body's OWN half-width
MAX_DIM = 480            # masks are shrunk to this; the metric is scale-free and this is 20x faster
LONG_BLOB = 6.0          # a residual blob this many half-widths long counts as fully "long"

DEFAULT_OFFAXIS_CUTOFF = 0.010    # defer at or above this off-axis score
# Gate 2 measures the DISTANCE of the furthest stray spot from the body, in body lengths -- not
# how many strays there are. Measured over all 1265 real photos, the furthest outside spot in the
# entire corpus sits 0.011 body lengths from its mask, and the counting version of this gate
# deferred 44 frames of which zero held a second animal: every one was a spot straddling the mask
# boundary. A count cannot tell "the mask clipped this spot's edge" from "there is another animal
# over there"; a distance can, because a second salamander is a whole body length away, not 1 %.
DEFAULT_GAP_CUTOFF = 0.020        # defer if a spot sits this far (in body lengths) off the body

KNOWN_MULTI = "ca_14_5"           # confirmed by eye: two overlapping animals. Must always defer.

SINGLE = "single"                 # cleared by the algorithm — no billed call
DEFER = "defer"                   # hand to the model


class Screen(NamedTuple):
    """One image's free verdict, with the numbers that produced it."""
    stem: str
    verdict: str                  # SINGLE | DEFER
    offaxis: float                # gate 1 score, or -1 when it could not be computed
    gap: float                    # gate 2: furthest stray spot, in body lengths; -1 = unknown
    reason: str
    outside: float = 0.0          # spots_outside_frac, recorded for context, NOT a gate

    @property
    def single(self) -> bool:
        return self.verdict == SINGLE

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "offaxis": round(self.offaxis, 5),
                "stray_spot_gap": round(self.gap, 5),
                "spots_outside_frac": round(self.outside, 4), "reason": self.reason}


# --- gate 1: off-axis mass --------------------------------------------------
def _dist_to_polyline(shape: tuple[int, int], poly: np.ndarray) -> np.ndarray:
    """Distance from every pixel to the midline, by rasterising it and running an EDT.

    Cheaper and simpler than projecting each pixel onto each segment, and the answer is the same
    quantity — perpendicular distance to the curve — because the EDT measures to the nearest set
    pixel and the curve is rasterised unbroken.
    """
    line = np.zeros(shape, np.uint8)
    cv2.polylines(line, [np.asarray(poly, np.int32).reshape(-1, 1, 2)], False, 255, 1, cv2.LINE_8)
    return cv2.distanceTransform(255 - line, cv2.DIST_L2, 5)


def offaxis_score(mask: np.ndarray, midline: np.ndarray, half_width: float) -> float:
    """-> 0 for a clean single body; grows with a thick, long structure off the midline.

    ``mask`` is binary 0/255, ``midline`` an (N, 2) polyline and ``half_width`` a length, all in
    the SAME pixel grid. Weighting by depth inside the body is what makes limbs harmless — a leg
    is thin, so its pixels sit shallow and contribute almost nothing next to the meaty core, the
    same trick :func:`~.body_mask.centre_line` uses to keep limbs from dragging the median.
    """
    depth = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    total = float(depth.sum())
    if total <= 0 or half_width <= 0:
        return -1.0

    r = _dist_to_polyline(mask.shape, midline)
    residual = ((mask > 0) & (r > RIBBON_K * half_width)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(residual, 8)

    best, best_mass = 0, 0.0
    for i in range(1, n):
        m = float(depth[lab == i].sum())
        if m > best_mass:
            best, best_mass = i, m
    if not best:
        return 0.0

    frac = best_mass / total
    thick = float(depth[lab == best].max()) / half_width
    length = float(max(stats[best, cv2.CC_STAT_WIDTH],
                       stats[best, cv2.CC_STAT_HEIGHT])) / half_width
    return frac * min(thick, 1.0) * min(length / LONG_BLOB, 1.0)


# --- gate 2: how far away is the furthest stray spot? -----------------------
def stray_spot_gap(mask: np.ndarray, spots: list[tuple[float, float]],
                   body_len: float) -> float:
    """-> distance from the body to the furthest spot lying off it, in BODY LENGTHS.

    0.0 when every spot is on the body. Normalising by body length is what makes the number
    comparable across photos taken at wildly different distances, and it is what gives the
    threshold a physical meaning: a spot 1 % of a body length off the mask is the mask's edge
    being a pixel tight, while a spot half a body length away is somewhere else entirely.

    Stage 1 paints every yellow spot in the FRAME, not just the subject's — verified on the
    confirmed two-animal photo, where both animals came back purple — so a separate second
    salamander does leave its spots in the table for this to find.
    """
    if body_len <= 0 or not spots:
        return 0.0
    outside = cv2.distanceTransform((mask < 128).astype(np.uint8) * 255, cv2.DIST_L2, 5)
    h, w = mask.shape[:2]
    worst = 0.0
    for x, y in spots:
        ix, iy = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
        if mask[iy, ix] < 128:
            worst = max(worst, float(outside[iy, ix]) / body_len)
    return worst


# --- the screen over a directory --------------------------------------------
SCREEN_FIELDS = ("stem", "verdict", "offaxis", "stray_spot_gap", "spots_outside_frac", "reason")


def _load_cache(path: Path) -> dict[str, Screen]:
    if not path.is_file():
        return {}
    out: dict[str, Screen] = {}
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            # An older cache has no gap column; refuse it rather than silently reading a
            # different quantity into the field the gate now keys on.
            if rd.fieldnames is None or "stray_spot_gap" not in rd.fieldnames:
                return {}
            for row in rd:
                out[row["stem"]] = Screen(row["stem"], row["verdict"],
                                          float(row["offaxis"]),
                                          float(row["stray_spot_gap"]),
                                          row.get("reason", ""),
                                          float(row.get("spots_outside_frac", 0.0)))
    except (OSError, KeyError, ValueError):
        return {}
    return out


def _save_cache(path: Path, screens: dict[str, Screen]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(SCREEN_FIELDS)
        for stem in sorted(screens):
            s = screens[stem]
            w.writerow([s.stem, s.verdict, f"{s.offaxis:.5f}", f"{s.gap:.5f}",
                        f"{s.outside:.4f}", s.reason])


def screen_dir(input: str | Path, stems: list[str] | None = None, *,
               db_path: Path | None = None, cache: Path | None = None,
               offaxis_cutoff: float = DEFAULT_OFFAXIS_CUTOFF,
               gap_cutoff: float = DEFAULT_GAP_CUTOFF,
               refresh: bool = False, verbose: bool = True) -> dict[str, Screen]:
    """Run the free screen over one input dir. -> ``{stem: Screen}``.

    Reads the dir's own ``contours/contours.db``. **Anything missing defers**: no DB, no mask, no
    midline and no quality row all mean the algorithm has nothing to judge on, which is precisely
    the case the paid layer exists for.

    Results are cached to ``cache`` (default ``animals/screen.csv``) because the distance
    transforms cost a couple of minutes over a full corpus and nothing about them changes between
    runs. ``refresh=True`` recomputes. Note the cache stores the METRICS, and the verdict is
    re-derived from the current cutoffs on load, so re-running with a different
    ``--screen-cutoff`` does not need a recompute.
    """
    input_dir = resolve_input_dir(input)
    db = Path(db_path) if db_path else contours_db_for(input_dir)
    cache = Path(cache) if cache else (input_dir / "animals" / "screen.csv")

    def verdict_for(offaxis: float, gap: float, note: str = "") -> tuple[str, str]:
        if offaxis < 0 or gap < 0:
            return DEFER, note or "the algorithm could not measure this frame"
        if offaxis >= offaxis_cutoff:
            return DEFER, (f"off-axis mass {offaxis:.3f} >= {offaxis_cutoff} — a thick, long "
                           f"structure sits off the midline (fused second animal, or a curl)")
        if gap >= gap_cutoff:
            return DEFER, (f"a spot sits {gap:.3f} body lengths off the mask >= {gap_cutoff} "
                           f"— too far to be the mask's edge; possible separate second animal")
        return SINGLE, f"off-axis {offaxis:.3f}, furthest stray spot {gap:.3f} body lengths — clear"

    cached = {} if refresh else _load_cache(cache)
    if cached and (stems is None or all(s in cached for s in stems)):
        # Re-derive verdicts so a changed cutoff takes effect without recomputing the metrics.
        out = {}
        for stem, s in cached.items():
            if stems is not None and stem not in stems:
                continue
            v, why = verdict_for(s.offaxis, s.gap)
            out[stem] = Screen(stem, v, s.offaxis, s.gap, why, s.outside)
        if verbose:
            logger.info(f"  screen: reusing cached metrics for {len(out)} image(s) ({cache.name})")
        return out

    if not db.is_file():
        if verbose:
            logger.warning(f"  screen: no contours DB at {db} — every frame defers to the model")
        return {s: Screen(s, DEFER, -1.0, -1.0, "no contours DB; run `contours` first")
                for s in (stems or [])}

    import duckdb
    con = duckdb.connect(str(db), read_only=True)
    try:
        rows = con.execute("""
            SELECT i.salamander_id, i.body_mask_png, a.midline_x, a.midline_y,
                   q.avg_width_px, q.spots_outside_frac, q.center_line_length_px
            FROM images i
            LEFT JOIN body_axis a USING(salamander_id)
            LEFT JOIN image_quality q USING(salamander_id)
        """).fetchall()
        spots_by: dict[str, list[tuple[float, float]]] = {}
        for sid, x, y in con.execute(
                "SELECT salamander_id, global_centroid_x, global_centroid_y FROM spots").fetchall():
            spots_by.setdefault(sid, []).append((float(x), float(y)))
    finally:
        con.close()

    wanted = set(stems) if stems is not None else None
    screens: dict[str, Screen] = {}
    todo = [r for r in rows if wanted is None or r[0] in wanted]
    if verbose:
        logger.info(f"  screen: measuring {len(todo)} image(s) from {db.name} ...")

    for k, (sid, blob, mx, my, avgw, outside, body_len) in enumerate(todo):
        if verbose and k and k % 250 == 0:
            logger.info(f"    {k}/{len(todo)}")
        out_frac = float(outside) if outside is not None and np.isfinite(outside) else 0.0
        offaxis, gap, note = -1.0, -1.0, ""

        if blob is None:
            note = "no body mask stored for this image"
        elif not mx or len(mx) < 2:
            note = "no body axis — stage 1b never produced a usable midline"
        elif not avgw or not np.isfinite(avgw) or avgw <= 0:
            note = "no body width in image_quality — run `compute-quality`"
        else:
            m = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
            if m is None:
                note = "the stored body mask could not be decoded"
            else:
                s = min(1.0, MAX_DIM / float(max(m.shape)))
                m = cv2.resize(m, (max(int(m.shape[1] * s), 1), max(int(m.shape[0] * s), 1)),
                               interpolation=cv2.INTER_NEAREST)
                m = (m > 127).astype(np.uint8) * 255
                poly = np.column_stack([np.asarray(mx, float) * s, np.asarray(my, float) * s])
                poly[:, 0] = np.clip(poly[:, 0], 0, m.shape[1] - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, m.shape[0] - 1)
                offaxis = offaxis_score(m, poly, (float(avgw) / 2.0) * s)
                if offaxis < 0:
                    note = "degenerate mask (no area)"
                elif body_len and np.isfinite(body_len) and body_len > 0:
                    gap = stray_spot_gap(
                        m, [(x * s, y * s) for x, y in spots_by.get(sid, [])],
                        float(body_len) * s)
                else:
                    note = "no centre-line length in image_quality — run `compute-quality`"

        v, why = verdict_for(offaxis, gap, note)
        screens[sid] = Screen(sid, v, offaxis, gap, why, out_frac)

    # Anything asked for that the DB has never heard of also defers.
    for stem in (stems or []):
        if stem not in screens:
            screens[stem] = Screen(stem, DEFER, -1.0, -1.0, "not in the contours DB")

    _save_cache(cache, screens)
    if verbose:
        logger.info(f"  screen: wrote {cache}")
    return screens


def report(screens: dict[str, Screen], *, offaxis_cutoff: float = DEFAULT_OFFAXIS_CUTOFF,
           gap_cutoff: float = DEFAULT_GAP_CUTOFF) -> None:
    """Print what the free layer settled and what it is handing to the model."""
    n = len(screens)
    if not n:
        return
    single = [s for s in screens.values() if s.single]
    defer = [s for s in screens.values() if not s.single]
    unmeasured = [s for s in defer if s.offaxis < 0 or s.gap < 0]
    by_axis = [s for s in defer if s.offaxis >= offaxis_cutoff]
    by_gap = [s for s in defer if 0 <= s.offaxis < offaxis_cutoff and s.gap >= gap_cutoff]

    logger.info(f"  LAYER 1 (free algorithm) — cutoffs: off-axis {offaxis_cutoff}, "
          f"stray-spot gap {gap_cutoff} body lengths")
    logger.info(f"    cleared as single : {len(single):>5}  ({len(single) / n:.1%})  — no billed call")
    logger.info(f"    deferred to Gemini: {len(defer):>5}  ({len(defer) / n:.1%})")
    if by_axis:
        logger.info(f"        {len(by_axis):>5} by off-axis mass (fused second body, or a tight curl)")
    if by_gap:
        logger.info(f"        {len(by_gap):>5} by a far-off stray spot (separate second animal)")
    if unmeasured:
        logger.info(f"        {len(unmeasured):>5} unmeasurable (no mask / axis / quality row)")

    anchor = screens.get(KNOWN_MULTI)
    if anchor is not None:
        ok = not anchor.single
        logger.info(f"    anchor {KNOWN_MULTI} (confirmed two animals): "
              f"{'DEFERRED — the screen works' if ok else 'CLEARED AS SINGLE — THE SCREEN IS BROKEN'}"
              f"  [off-axis {anchor.offaxis:.3f}]")
        if not ok:
            logger.warning("    ^ lower --screen-cutoff until the anchor defers before trusting this run.")
