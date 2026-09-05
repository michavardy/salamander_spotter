#!/usr/bin/env python3
"""Where on the salamander a spot sits — its positional labels.

Each spot gets TWO independent labels off the body's centre line (derived in stage 1b):

    axial_bin  1..4   which quarter of the body, cutting the centre line into 4 equal
                      ARC-length segments. 1 is the head end, 4 the tail end.
    lateral_bin  left | right | overlap
                      which side of the centre line the spot is on — or "overlap" if the
                      spot straddles the line, in which case it is on NEITHER side.

``left`` is the image-left of a salamander whose head is at the top of the frame. It is
computed as the sign of the perpendicular offset against the BODY AXIS, not the image axes,
so it names the same flank however the animal happens to be rotated in the photo.

Overlap is why the spot's OUTLINE is needed and not just its centroid: a spot lying across
the spine has contour points on both sides, and calling it "left" because its centroid landed
a pixel that way would be a lie.

``bin`` (1..8) is the two labels multiplied together, kept for convenience::

        quarter        left  right
        0-25   %  ->     1     2
        25-50  %  ->     3     4
        50-75  %  ->     5     6
        75-100 %  ->     7     8

It is **NULL for an overlapping spot**, which is in neither box. Prefer ``axial_bin`` +
``lateral``, which can express all three lateral states.

Percent is measured along the ARC, not along the straight chord, so a curled tail still bins
by distance *along the body*.

Everything works in the ORIGINAL photo's pixel frame — the same coordinates the spot
centroids and contours live in — so no alignment step is needed. ``left``/``right`` are
therefore relative to the head->tail direction *as it runs in the photo*, not to the image
axes: the sign of the 2-D cross product, which is invariant to how the animal happens to be
rotated in frame. That is what makes the bin a usable positional key for matching two photos
of the same animal taken at different orientations.

The stored ``bin_polygons`` are the same partition drawn as quads (perpendicular cuts at
25/50/75 %, the midline as the median divider, clipped to the image). They are what the grid
*looks* like; a spot's own bin is always computed analytically from (axis_t, side), never by
point-in-polygon, so a spot lying outside the drawn quad still lands in the right box.

Pure geometry: numpy only, no cv2, no network.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "N_BINS", "QUARTILES", "SIDES", "OVERLAP", "FIELDS",
    "bin_of", "bin_bounds", "axial_bin_of", "lateral_of", "bin_spots",
    "as_polyline", "polyline_length", "project", "point_at", "bin_polygons",
]

N_BINS = 8
QUARTILES = 4
SIDES = ("left", "right")
OVERLAP = "overlap"          # the spot straddles the centre line: neither left nor right
Point = tuple[float, float]


def bin_of(t: float, side: str) -> int:
    """(axis_t in 0..1, "left"|"right") -> box 1..8. See the table above."""
    q = min(int(float(t) * QUARTILES), QUARTILES - 1)   # t == 1.0 belongs to the last quarter
    return 2 * max(q, 0) + (1 if side == "left" else 2)


def bin_bounds(b: int) -> tuple[float, float, str]:
    """Box 1..8 -> (t_lo, t_hi, side) — the inverse of `bin_of`."""
    q = (int(b) - 1) // 2
    return q / QUARTILES, (q + 1) / QUARTILES, SIDES[(int(b) - 1) % 2]


def axial_bin_of(t: float) -> int:
    """axis_t in 0..1 -> which QUARTER of the body the point is in: 1, 2, 3 or 4.

    The centre line is cut into 4 equal-ARC-length segments. **1 is the head end** (0-25 %) and
    4 is the tail end (75-100 %), matching axis_t's 0-at-the-head convention.
    """
    return min(int(float(t) * QUARTILES), QUARTILES - 1) + 1


def lateral_of(contour: Sequence | None, centroid: Point,
               midline: Sequence) -> str:
    """Which side of the centre line a spot is on: "left", "right", or "overlap".

    A spot whose OUTLINE has points on both sides of the centre line straddles it — it is
    neither left nor right, and saying "left" because its centroid happens to fall a pixel that
    way would be a lie. Those come back as ``"overlap"``.

    Otherwise the side is the sign of the centroid's perpendicular offset: positive is the
    image-LEFT of a head-at-the-top salamander (see `project`). Because that sign is taken
    against the body axis rather than the image axes, it names the same flank however the animal
    is rotated in the photo.

    With no contour (centroid only), the centroid's own side is returned — overlap cannot be
    detected from a single point.
    """
    side = SIDES[0] if project(centroid, midline)[1] >= 0 else SIDES[1]
    if contour is None or len(contour) < 3:
        return side
    offsets = [project(p, midline)[1] for p in contour]
    if min(offsets) < 0.0 < max(offsets):
        return OVERLAP
    return side


# --- polyline geometry ------------------------------------------------------
def as_polyline(points: Sequence) -> np.ndarray:
    """(N, 2) float array with consecutive duplicate vertices dropped."""
    p = np.asarray(points, float).reshape(-1, 2)
    if len(p) < 2:
        return p
    keep = np.concatenate([[True], np.hypot(*(p[1:] - p[:-1]).T) > 1e-9])
    return p[keep]


def _segments(poly: np.ndarray):
    """(a, d, seg_len, cum_len_at_a, total_len) for the polyline."""
    a = poly[:-1]
    d = poly[1:] - poly[:-1]
    seg = np.hypot(d[:, 0], d[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return a, d, seg, cum[:-1], float(cum[-1])


def polyline_length(points: Sequence) -> float:
    poly = as_polyline(points)
    return 0.0 if len(poly) < 2 else _segments(poly)[4]


def project(point: Point, points: Sequence) -> tuple[float, float]:
    """Project a point onto the midline -> (axis_t in 0..1, signed offset in px).

    ``axis_t`` is the arc length of the closest foot point over the total length: 0 at the
    head, 1 at the tail tip. The offset is the perpendicular distance to that segment,
    POSITIVE on the left of the head->tail direction (the 2-D cross product's sign).
    """
    poly = as_polyline(points)
    if len(poly) < 2:
        return 0.0, 0.0
    a, d, seg, cum, total = _segments(poly)
    if total < 1e-9:
        return 0.0, 0.0

    p = np.asarray(point, float)
    v = p - a                                       # (S, 2) offset from each segment start
    L2 = np.maximum(seg ** 2, 1e-12)
    u = np.clip((v * d).sum(1) / L2, 0.0, 1.0)      # position along each segment
    foot = a + u[:, None] * d
    i = int(np.argmin(np.hypot(*(p - foot).T)))     # the closest segment wins

    t = float((cum[i] + u[i] * seg[i]) / total)
    cross = float(d[i, 0] * v[i, 1] - d[i, 1] * v[i, 0])
    offset = cross / float(max(seg[i], 1e-9))
    return min(max(t, 0.0), 1.0), offset


def point_at(points: Sequence, t: float) -> tuple[np.ndarray, np.ndarray]:
    """The midline point at arc-length fraction `t` and the unit LEFT normal there."""
    poly = as_polyline(points)
    a, d, seg, cum, total = _segments(poly)
    s = float(np.clip(t, 0.0, 1.0)) * total
    i = int(np.clip(np.searchsorted(cum + seg, s, side="left"), 0, len(seg) - 1))
    u = float((s - cum[i]) / max(seg[i], 1e-9))
    tangent = d[i] / max(seg[i], 1e-9)
    # Left normal: the n with cross(d, n) > 0, matching `project`'s sign convention.
    return a[i] + u * d[i], np.array([-tangent[1], tangent[0]])


# --- the 8 boxes as polygons (storage + QA overlay) -------------------------
def _clip_to_rect(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex polygon against the image rectangle."""
    out = poly
    for axis, limit, keep_ge in ((0, 0.0, True), (0, float(w), False),
                                 (1, 0.0, True), (1, float(h), False)):
        if len(out) == 0:
            return out
        clipped = []
        for j, cur in enumerate(out):
            prev = out[j - 1]
            cur_in = (cur[axis] >= limit) if keep_ge else (cur[axis] <= limit)
            prev_in = (prev[axis] >= limit) if keep_ge else (prev[axis] <= limit)
            if cur_in != prev_in:                   # the edge crosses the boundary
                denom = cur[axis] - prev[axis]
                if abs(denom) > 1e-9:
                    clipped.append(prev + (limit - prev[axis]) / denom * (cur - prev))
            if cur_in:
                clipped.append(cur)
        out = np.asarray(clipped, float).reshape(-1, 2)
    return out


def bin_polygons(points: Sequence, width: int, height: int) -> list[dict]:
    """The 8 boxes as image-clipped polygons: [{bin, quartile, side, t_lo, t_hi, polygon}].

    Each box is the region between two cuts perpendicular to the midline (at t_lo and t_hi),
    on one side of the midline, widened past the frame and then clipped back to it — so the
    8 boxes tile the whole image. Returns [] for a degenerate midline.
    """
    poly = as_polyline(points)
    if len(poly) < 2 or polyline_length(poly) < 1e-6:
        return []
    reach = float(np.hypot(width, height))          # far enough to leave the frame

    out: list[dict] = []
    for q in range(QUARTILES):
        t_lo, t_hi = q / QUARTILES, (q + 1) / QUARTILES
        p_lo, n_lo = point_at(poly, t_lo)
        p_hi, n_hi = point_at(poly, t_hi)
        for side in SIDES:
            sign = 1.0 if side == "left" else -1.0
            quad = np.array([p_lo, p_lo + sign * reach * n_lo,
                             p_hi + sign * reach * n_hi, p_hi])
            clipped = _clip_to_rect(quad, width, height)
            out.append({
                "bin": bin_of((t_lo + t_hi) / 2, side), "quartile": q, "side": side,
                "t_lo": t_lo, "t_hi": t_hi,
                "polygon": [(float(x), float(y)) for x, y in clipped],
            })
    return sorted(out, key=lambda d: d["bin"])


# --- spot binning -----------------------------------------------------------
# NB: the column is `lateral_bin`, not `lateral` — LATERAL is a reserved word in DuckDB and a
# bare `lateral` column would have to be double-quoted in every query anyone ever writes.
FIELDS = ("bin", "axial_bin", "lateral_bin", "axis_t", "axis_side", "axis_offset")


def bin_spots(spots: list[dict], midline: Sequence | None) -> int:
    """Fill each spot dict's positional labels in place. Returns the count binned.

    Per spot:
      ``axial_bin``   1..4 — which quarter of the body the CENTROID is in (1 = head end).
      ``lateral_bin`` "left" | "right" | "overlap" — which side of the centre line the SPOT is
                      on, where "overlap" means its outline crosses the line.
      ``bin``         1..8 — the (quarter x side) box, for convenience. **NULL when the spot
                      overlaps the line**, because a straddling spot is not in one box or the
                      other; use ``axial_bin`` + ``lateral_bin`` as the primary labels.
      ``axis_t``      0..1 along the centre line, 0 at the head.
      ``axis_side``   the CENTROID's side, always left/right (even when the spot overlaps).
      ``axis_offset`` signed perpendicular px from the line (+ = left).

    Reads each spot's ``global_centroid`` and, when present, its ``local_contour`` (which is
    relative to the centroid) — the contour is what makes overlap detectable at all. A midline
    shorter than two distinct points leaves every spot unlabelled (all fields None) rather than
    guessing.
    """
    poly = as_polyline(midline or [])
    binned = len(poly) >= 2 and polyline_length(poly) >= 1e-6
    for s in spots:
        if not binned:
            for f in FIELDS:
                s[f] = None
            continue
        centroid = s["global_centroid"]
        t, offset = project(centroid, poly)
        contour = s.get("local_contour")
        absolute = ([(px + centroid[0], py + centroid[1]) for px, py in contour]
                    if contour else None)
        lateral = lateral_of(absolute, centroid, poly)

        s["axis_t"] = round(t, 4)
        s["axis_offset"] = round(offset, 2)
        s["axis_side"] = SIDES[0] if offset >= 0 else SIDES[1]
        s["axial_bin"] = axial_bin_of(t)
        s["lateral_bin"] = lateral
        s["bin"] = None if lateral == OVERLAP else bin_of(t, lateral)
    return len(spots) if binned else 0
