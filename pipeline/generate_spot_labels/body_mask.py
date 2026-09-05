#!/usr/bin/env python3
"""Stage 1b, mask mode — the body as a FILLED REGION, and the geometry derived from it.

The model is asked for one thing it is actually good at: **paint the animal's body flat
magenta** (and drop a green dot on the snout, a red dot on the tail tip). Everything else —
the two body outlines, the centre line, the true tips — is then computed here, from the mask,
with no model in the loop.

Why this replaces the two-outline method
----------------------------------------
The old stage 1b asked Gemini for two separate half-outlines (cyan down one flank, magenta
down the other) and took their mean as the centre line. Measured over a 56-image run, the
judge rejected **73 %** of drafts, and the ladder (2.5-flash -> 3.1-flash -> 3-pro) did not
help: 93 % / 87 % / 89 % rejected. Reading the rejected drafts says why — the model reliably
finds the body BOUNDARY but will not decompose it into two side-curves. It kept drawing one
closed loop in both colours (`follows_edges` failed 74 % of the time, `opposite_sides` 18 %).

Filling a region is a masking task, which is the trick stage 1 already relies on for the spots.
So we ask for the mask, and derive the rest:

    outline      the mask's contour                     (cv2.findContours)
    left/right   that contour, cut at the two tips      (:func:`outline_halves`)
    centre line  the equal-area line through the mask   (:func:`centre_line`)
    head/tail    the mask's true extreme tips           (:func:`snap_tip`)

Three of the old judge's four failure modes stop existing, because nothing is being *drawn*
that could be wrong: there are no edges to follow, no two lines to keep on opposite sides, and
the tips come from the mask rather than from the model's aim. What remains ("is this a good
salamander mask?") is checkable in pure OpenCV — see :func:`check`. **No LLM judge.**

The centre line: "the same amount of pink on both sides"
--------------------------------------------------------
Literally that, enforced per slice rather than globally. A single straight head->tail line with
equal area either side is useless on these photos — a C-curled salamander's snout-to-tail chord
runs outside the animal, through the gravel. So :func:`centre_line` slices the mask into bands
along the body and, in each band, finds the offset with half the pink on each side.

That "half the pink on each side" offset is the **median** of the band's signed offsets, not the
mean — the mean is the centroid (an area-weighted balance point) and a filled leg drags it
sideways, while the median is the equal-area point. Each pixel is additionally weighted by its
DEPTH INSIDE THE BODY (its distance to the boundary), which is what finally makes limbs
harmless: a leg is thin, so its pixels are shallow and count for almost nothing next to the
meaty core of the trunk. That matters more than it sounds — a thick leg fills about half of its
own narrow band, so the unweighted median really does move. No morphological de-legging is
used, because an opening large enough to remove a leg also amputates the tail, which is thinner.

Everything here was checked against a synthetic salamander with a known spine before a single
API call was made (curvature sweep, one-sided legs, deliberately mis-aimed dots); the surprises
that fell out of that — the dots severing the tail, iteration making curvature *worse*, and
straight-line distance snapping to the wrong tip on a curled body — are recorded at the
functions they shaped.

Pure geometry: cv2 + numpy + skimage, no network, no model. The Gemini call lives in
:mod:`llm_body_fill`; this module is offline-testable and is where the maths is checked.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .binning import as_polyline, point_at, polyline_length

Point = tuple[float, float]

# The three marks. The body fill reuses stage 1's magenta — the key is already tuned for it,
# and there is no collision because the anatomy render is a SEPARATE image from purple/.
FILL_HEX = "#FF00FF"     # the body, painted flat
HEAD_HEX = "#00FF00"     # green dot — the snout
TAIL_HEX = "#FF0000"     # red dot — the tip of the tail

# OpenCV HSV (H 0..179). High S/V floors keep natural colour out of the keys: no moss is this
# saturated, and the salamander's own yellow is nowhere near these hues.
FILL_HSV_LOW = (135, 100, 90)       # magenta ~ H 150
FILL_HSV_HIGH = (165, 255, 255)
# The model's "pure green" lands anywhere in H 33..80 in practice (measured: a dot whose hue came
# back at 36-42, i.e. straddling a 40 floor and half-missed). The band is wide because it can
# afford to be: the dot keys are applied ONLY inside the painted body (see `dots_on`), so a
# generous green cannot swallow moss, and the animal's own yellow is painted over before we look.
HEAD_HSV_LOW = (33, 100, 90)        # green ~ H 60, but the models drift yellow-ward
HEAD_HSV_HIGH = (85, 255, 255)
TAIL_HSV_LOW = (0, 110, 90)         # red straddles the hue origin -> two bands
TAIL_HSV_HIGH = (8, 255, 255)
TAIL_HSV_LOW2 = (172, 110, 90)
TAIL_HSV_HIGH2 = (179, 255, 255)

DEFAULT_MORPH = 5            # cleanup kernel for the keyed fill
DEFAULT_DOT_MIN_AREA = 25    # ignore keyed dot blobs smaller than this (px^2)
DEFAULT_DOT_PAD = 15         # px of slack around the body when hunting the dots (`dots_on`)
DEFAULT_SAMPLES = 32         # bands along the body
DEFAULT_ITERS = 2            # re-slice passes: best worst-case over the curl sweep (`centre_line`)
DEFAULT_SMOOTH = 7           # Savitzky-Golay window over the centre line's vertices (`smooth`)
DEFAULT_SMOOTH_ORDER = 2     # ...and its polynomial order. 2 keeps curvature; 0 would be a mean.
DEFAULT_SNAP_FRAC = 0.10     # search this fraction of the diagonal around a dot for the true tip
MAX_PTS = 30000              # mask pixels sampled for the centre line (median needs no more)

# --- acceptance gates (all free — this is what replaces the LLM judge) ---
DEFAULT_MIN_AREA_FRAC = 0.005   # a body smaller than 0.5 % of the frame is not a salamander
DEFAULT_MAX_AREA_FRAC = 0.80    # ...and one bigger than 80 % means it flood-filled the photo
DEFAULT_MIN_AXIS_FRAC = 0.10    # reject an axis shorter than this fraction of the diagonal
DEFAULT_MIN_INSIDE = 0.95       # this fraction of centre-line vertices must lie ON the mask


@dataclass
class Body:
    """One image's body geometry, in ORIGINAL-photo pixels.

    Field-for-field compatible with the old :class:`~.llm_anatomy.Anatomy`, so stage 2 and the
    ``body_axis`` table need no changes — but ``left``/``right`` are now the two halves of the
    mask's own contour rather than two curves the model drew, and ``midline`` is the equal-area
    line rather than the mean of two guesses.
    """
    head: Point | None = None
    tail_tip: Point | None = None
    midline: list[Point] = field(default_factory=list)
    left: list[Point] = field(default_factory=list)      # outline, one side of head->tail
    right: list[Point] = field(default_factory=list)     # outline, the other side
    source: str = "none"                 # "mask" | "none"
    area_px: int = 0
    checks: dict = field(default_factory=dict)   # gate name -> passed?
    reason: str = ""                     # why it failed, when it did

    @property
    def ok(self) -> bool:
        return (self.head is not None and self.tail_tip is not None
                and len(self.midline) >= 2)

    @property
    def passed(self) -> bool:
        """Cleared every gate — the mask-mode equivalent of "the judge accepted it"."""
        return self.ok and bool(self.checks) and all(self.checks.values())

    @property
    def length_px(self) -> float:
        return polyline_length(self.midline)

    @property
    def failures(self) -> list[str]:
        return [k for k, v in self.checks.items() if not v]

    def as_dict(self) -> dict:
        return {"head": list(self.head) if self.head else None,
                "tail_tip": list(self.tail_tip) if self.tail_tip else None,
                "midline": [list(p) for p in self.midline],
                "left": [list(p) for p in self.left],
                "right": [list(p) for p in self.right],
                "length_px": round(self.length_px, 2),
                "area_px": self.area_px,
                "source": self.source,
                "judged_ok": self.passed if self.checks else None,
                "judge_feedback": self.reason,
                "checks": self.checks}

    @classmethod
    def from_dict(cls, d: dict) -> "Body":
        head, tail = d.get("head"), d.get("tail_tip")
        return cls(head=tuple(head) if head else None,
                   tail_tip=tuple(tail) if tail else None,
                   midline=[tuple(p) for p in d.get("midline") or []],
                   left=[tuple(p) for p in d.get("left") or []],
                   right=[tuple(p) for p in d.get("right") or []],
                   source=d.get("source", "none"),
                   area_px=int(d.get("area_px") or 0),
                   checks=d.get("checks") or {},
                   reason=d.get("judge_feedback", ""))


# --- keying -----------------------------------------------------------------
def _clean(mask: np.ndarray, morph: int) -> np.ndarray:
    if morph and morph >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)   # close first: seal antialiased gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def fill_mask(bgr: np.ndarray, morph: int = DEFAULT_MORPH) -> np.ndarray:
    """The painted body -> a binary mask."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return _clean(cv2.inRange(hsv, np.array(FILL_HSV_LOW, np.uint8),
                              np.array(FILL_HSV_HIGH, np.uint8)), morph)


def head_mask(bgr: np.ndarray, morph: int = 3) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return _clean(cv2.inRange(hsv, np.array(HEAD_HSV_LOW, np.uint8),
                              np.array(HEAD_HSV_HIGH, np.uint8)), morph)


def tail_mask(bgr: np.ndarray, morph: int = 3) -> np.ndarray:
    """Red mask (the tail dot). Red straddles hue 0, so it takes two bands."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = cv2.inRange(hsv, np.array(TAIL_HSV_LOW, np.uint8), np.array(TAIL_HSV_HIGH, np.uint8))
    hi = cv2.inRange(hsv, np.array(TAIL_HSV_LOW2, np.uint8), np.array(TAIL_HSV_HIGH2, np.uint8))
    return _clean(cv2.bitwise_or(lo, hi), morph)


def largest_centroid(mask: np.ndarray, min_area: int = DEFAULT_DOT_MIN_AREA) -> Point | None:
    """The centre of the biggest blob in a dot mask, or None if nothing big enough was drawn."""
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    best, best_area = None, min_area - 1
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best, best_area = i, area
    return (float(cents[best][0]), float(cents[best][1])) if best is not None else None


def dots_on(bgr: np.ndarray, region: np.ndarray, morph: int = 3,
            min_area: int = DEFAULT_DOT_MIN_AREA,
            pad: int = DEFAULT_DOT_PAD) -> tuple[Point | None, Point | None, np.ndarray]:
    """Find the two dots ON THE BODY. -> (head, tail_tip, the dots' mask).

    The restriction to ``region`` is the whole point, and it is not a micro-optimisation: hunting
    the green dot across the WHOLE FRAME finds moss. Real photos here have vivid, saturated
    vegetation sitting in the same hue band as the marker, so the largest green blob in the image
    is often a patch of undergrowth rather than the 4 000 px dot on the animal's head — and the
    pipeline then reports "no dots" while staring straight at two perfectly good ones. (The old
    two-outline module asserted that "foliage is never this saturated". It is. That belief cost
    two of the first eight prototype images.)

    The animal, meanwhile, is the one thing we can key with total confidence: it is a solid slab
    of magenta that nothing in nature resembles. So find the body first, then look for the dots
    only where a dot could possibly be — on it, plus ``pad`` px of slack for one drawn over the
    edge.
    """
    near = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad, pad)))
    green = cv2.bitwise_and(head_mask(bgr, morph), near)
    red = cv2.bitwise_and(tail_mask(bgr, morph), near)
    return (largest_centroid(green, min_area), largest_centroid(red, min_area),
            cv2.bitwise_or(green, red))


def body_of(bgr: np.ndarray, dots: np.ndarray, morph: int = DEFAULT_MORPH) -> np.ndarray:
    """The keyed body: the magenta fill UNION the two dots.

    The dots must be folded in, not merely tolerated. They are painted ON the animal, so they
    ARE body — but they are painted over the fill, so they subtract from the magenta key. Where
    the body is thicker than the dot that only punches a hole (which :func:`solidify` closes
    anyway); where it is THINNER, the dot severs the animal in two. Not hypothetical: the tail
    tapers to a few px, so a red dot on the tip snips the last of the tail off as its own blob,
    `solidify` discards it as "not the largest", and the recovered tip lands short of the true
    one. The synthetic test caught exactly this — 42 px of error against 8 px of real taper.
    """
    return _clean(cv2.bitwise_or(fill_mask(bgr, 0), dots), morph)


def solidify(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Keep only the largest blob, with its holes filled. -> (mask, its contour)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(mask), None
    biggest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [biggest], -1, 255, cv2.FILLED)
    return out, biggest


# --- the true tips ----------------------------------------------------------
GEODESIC_MAX_DIM = 700      # the mask is shrunk to this for the (only) geodesic pass


def _geodesic_from(mask: np.ndarray, seed: Point) -> tuple[np.ndarray, float]:
    """Distance from ``seed`` to every mask pixel, measured ALONG THE BODY. -> (dists, scale).

    Straight-line distance is the wrong ruler on a curled animal — see :func:`snap_tip`. This
    walks through the mask instead, so a tail that curls back beside the head is still *far*
    from it, as it should be. Computed on a shrunk mask (a tip does not move under a 3x
    downscale) and returned with the scale used, since one geodesic pass on a 12 MP photo would
    cost more than the rest of the pipeline put together.

    Background pixels are unreachable (infinite cost), which is exactly the point.
    """
    from skimage.graph import MCP_Geometric

    h, w = mask.shape[:2]
    scale = min(1.0, GEODESIC_MAX_DIM / float(max(h, w)))
    small = (cv2.resize(mask, (max(int(w * scale), 1), max(int(h * scale), 1)),
                        interpolation=cv2.INTER_NEAREST) if scale < 1.0 else mask)

    ys, xs = np.nonzero(small)
    if len(xs) == 0:
        return np.full(small.shape, np.inf), scale
    # The seed must sit ON the mask, so pull it to the nearest body pixel first.
    sx, sy = float(seed[0]) * scale, float(seed[1]) * scale
    i = int(np.argmin(np.hypot(xs - sx, ys - sy)))

    costs = np.where(small > 0, 1.0, np.inf)
    dists, _ = MCP_Geometric(costs).find_costs([(int(ys[i]), int(xs[i]))])
    return dists, scale


def snap_tip(mask: np.ndarray, dot: Point, other: Point, radius: float,
             geodesic: np.ndarray | None = None, scale: float = 1.0) -> Point | None:
    """A dot -> the mask's actual extreme tip near it.

    The model's aim is the old `dots_at_tips` failure (30 % of drafts): it puts the green dot on
    the middle of the head rather than on the snout. It does not have to any more. The MASK
    knows where the animal ends, so the dot only has to say WHICH END this is — we take, among
    the mask pixels within ``radius`` of the dot, the one FURTHEST FROM THE OTHER TIP.

    "Furthest" must be measured **along the body**, not through the air. Straight-line distance
    silently picks the wrong point on a curled animal: when the snout curls back towards the
    tail, the true snout is nearer the tail in a straight line than the outside of the bend is,
    so a Euclidean argmax walks off to the side of the curl instead of to the tip. The synthetic
    horseshoe caught this. ``geodesic`` (from :func:`_geodesic_from`, seeded at the other dot)
    is a distance map that only travels through the mask, which is the right ruler.

    The geodesic map is computed on a shrunk mask, so the winner is refined back at full
    resolution among its immediate neighbours — over a few px the body is locally straight, so
    Euclidean and geodesic agree there and the cheap test is exact.

    Returns None if the mask has no pixels near the dot (the dot is nowhere near the body, so
    the annotation is not trustworthy).
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    near_dot = np.hypot(*(pts - np.asarray(dot, float)).T) <= radius
    near = pts[near_dot]
    if len(near) == 0:
        return None

    if geodesic is None:                            # no map: straight-line fallback
        return tuple(near[int(np.argmax(np.hypot(*(near - np.asarray(other, float)).T)))])

    gy = np.clip((near[:, 1] * scale).astype(int), 0, geodesic.shape[0] - 1)
    gx = np.clip((near[:, 0] * scale).astype(int), 0, geodesic.shape[1] - 1)
    d = geodesic[gy, gx]
    if not np.any(np.isfinite(d)):                  # the dot's neighbourhood is unreachable
        return None
    d = np.where(np.isfinite(d), d, -np.inf)
    coarse = near[int(np.argmax(d))]

    # Refine: the geodesic map is coarse, so take the true extremity among the full-res pixels
    # around the winner. Locally the body is straight, so plain distance is the right test here.
    window = 2.0 / max(scale, 1e-6) + 4.0
    local = near[np.hypot(*(near - coarse).T) <= window]
    if len(local) == 0:
        return tuple(coarse)
    return tuple(local[int(np.argmax(np.hypot(*(local - np.asarray(other, float)).T)))])


# --- the centre line --------------------------------------------------------
def _project_many(pts: np.ndarray, poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project many points onto a polyline at once -> (t in 0..1, signed offset px).

    The vectorised twin of :func:`~.binning.project` (same sign convention: offset is POSITIVE
    on the left of the head->tail direction). Per-point Python would be far too slow here — this
    runs over every sampled mask pixel, several times per image.
    """
    a = poly[:-1]                                   # (S, 2) segment starts
    d = poly[1:] - poly[:-1]                        # (S, 2) segment vectors
    seg = np.hypot(d[:, 0], d[:, 1])                # (S,)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-9:
        return np.zeros(len(pts)), np.zeros(len(pts))

    v = pts[:, None, :] - a[None, :, :]             # (P, S, 2)
    u = np.clip((v * d[None, :, :]).sum(-1) / np.maximum(seg ** 2, 1e-12), 0.0, 1.0)
    foot = a[None, :, :] + u[..., None] * d[None, :, :]
    dist = np.hypot(*(pts[:, None, :] - foot).transpose(2, 0, 1))    # (P, S)
    i = np.argmin(dist, axis=1)                     # the closest segment wins
    rows = np.arange(len(pts))

    t = (cum[i] + u[rows, i] * seg[i]) / total
    cross = d[i, 0] * v[rows, i, 1] - d[i, 1] * v[rows, i, 0]
    return np.clip(t, 0.0, 1.0), cross / np.maximum(seg[i], 1e-9)


def _sample(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The mask's pixels -> ((P, 2) float xy, (P,) weights), strided down to at most MAX_PTS.

    The weight is the pixel's distance to the nearest background pixel — how deep inside the
    body it is. See :func:`centre_line` for why that is what makes limbs harmless. A median over
    30 000 pixels is the same median as over 500 000, and the projection is O(points x segments),
    so the striding is pure speed at no cost in the answer.
    """
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(float)
    if len(pts) == 0:
        return pts, np.zeros(0)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    w = dist[ys, xs].astype(float)
    if len(pts) > MAX_PTS:
        step = int(np.ceil(len(pts) / MAX_PTS))
        pts, w = pts[::step], w[::step]
    return pts, w


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The v with half the WEIGHT below it — the equal-area point when weights are areas."""
    if len(values) == 0:
        return 0.0
    total = float(weights.sum())
    if total <= 1e-9:
        return float(np.median(values))
    order = np.argsort(values)
    v, cum = values[order], np.cumsum(weights[order])
    i = int(np.searchsorted(cum, 0.5 * total, side="left"))
    return float(v[min(max(i, 0), len(v) - 1)])


def smooth(points: list[Point], window: int = DEFAULT_SMOOTH,
           polyorder: int = DEFAULT_SMOOTH_ORDER) -> list[Point]:
    """Round off a polyline's corners, keeping its two endpoints EXACTLY where they were.

    Each vertex of the raw centre line is an independent band median, so the line arrives with a
    px-scale zigzag on it — the staircase visible down the animal's back in the QA renders. The
    wobble is not cosmetic: a spot's ``lateral_bin`` is the side of the line its outline falls
    on, so a spot lying near the spine can be flipped from left to right, or wrongly called
    "overlap", by a kink of a few px in the line beside it. A smooth line makes that call stable.

    **Savitzky-Golay, not a moving average.** The obvious choice is to average each vertex with
    its neighbours, and it is actively wrong here: averaging CUTS CORNERS, dragging a curved line
    toward its chord. On a curled salamander that is a large systematic error, not a rounding —
    measured against a known spine, a 5-wide moving average took the error at high curl from
    9.7 px to 44 px, and on a tight enough curl it walks the line out of the body altogether.
    Savitzky-Golay fits a local polynomial instead, so a smooth arc passes through essentially
    untouched (a quadratic is reproduced exactly) while the high-frequency jitter still goes.

    The head and the tail tip are pinned afterwards: they are the axis's 0 % and 100 %, they were
    derived exactly by :func:`snap_tip`, and they must not be blurred toward the body's interior.
    """
    from scipy.signal import savgol_filter

    p = np.asarray(points, float)
    k = max(3, int(window) | 1)                             # odd window
    if len(p) < 5 or k > len(p) or polyorder >= k:
        return [(float(x), float(y)) for x, y in p]

    head, tail = p[0].copy(), p[-1].copy()
    q = np.column_stack([savgol_filter(p[:, 0], k, polyorder, mode="interp"),
                         savgol_filter(p[:, 1], k, polyorder, mode="interp")])
    q[0], q[-1] = head, tail                                # the tips are exact; keep them exact
    return [(float(x), float(y)) for x, y in q]


def centre_line(mask: np.ndarray, head: Point, tail_tip: Point,
                samples: int = DEFAULT_SAMPLES, iters: int = DEFAULT_ITERS,
                window: int = DEFAULT_SMOOTH,
                polyorder: int = DEFAULT_SMOOTH_ORDER) -> list[Point]:
    """The body mask -> the line with the SAME AMOUNT OF PINK ON EITHER SIDE of it.

    Slice the mask into ``samples`` bands along the body; in each band, take the signed
    perpendicular offsets of the pink pixels and move to the offset with **half the pink on
    each side**. Stitch the band midpoints head-to-tail.

    Half the pink, not the average pink: the mean is the centroid (the area-weighted balance
    point), which a filled leg drags sideways; the median is the equal-area point. Stating the
    goal as "the same amount on both sides" is what buys the robustness.

    But a plain median is not robust ENOUGH, and the synthetic test says so: a thick leg fills
    roughly HALF of its own narrow band, so it does move the median — and then the next pass
    re-slices around the already-dragged line and swallows even more of the limb. Iterating made
    it worse (9.4 px -> 14.9 px mean error). So each pixel is weighted by its DISTANCE TO THE
    BOUNDARY: a leg is thin, so its pixels sit shallow and count for almost nothing, while the
    meaty core of the trunk dominates. Same definition, measured with the thin bits discounted —
    and limbs stop mattering without a single morphological hack (an opening that could remove a
    leg would also amputate the tapering tail, which is thinner than the legs are).

    The first pass cuts bands perpendicular to the straight head->tail CHORD, which on a curled
    animal need not lie inside the body at all (on the synthetic C it is 92 % outside). Each
    further pass re-slices perpendicular to the line found by the last one, so the bands come to
    follow the animal rather than the chord. Measured against a known spine over a curvature
    sweep, mean px error::

        curl   chord outside   iters=1   iters=2   iters=3
        0.5             53 %       8.1      10.6      12.9
        1.0             92 %       5.5       5.4       6.8
        1.4             93 %       5.8       4.9       5.9
        1.8             90 %      11.2       4.9       4.5
        2.2             69 %      15.5      11.1       9.7

    Re-slicing earns its keep exactly where it should: it is a wash on a nearly straight animal
    (where the chord is already the answer, and the extra passes only add wobble) and it halves
    the error on a strongly curled one. Two passes has both the best worst case and the best
    average, so that is the default.

    A warning to anyone re-tuning this. The first version of the sweep said iters=1 won at
    *every* curl, which is the opposite of the table above, and the difference was not the
    banding at all — it was :func:`snap_tip` anchoring the head in the wrong place on curled
    bodies. A polyline pinned to a bad endpoint gets worse, not better, when you re-slice around
    it, so the tip bug masqueraded convincingly as an iteration bug. **Re-measure the tips
    before trusting any conclusion about the bands.**

    Returns [] if the mask is empty or the two tips coincide.
    """
    pts, weight = _sample(mask)
    if len(pts) == 0:
        return []
    poly = np.asarray([head, tail_tip], float)
    if polyline_length(poly) < 1e-6:
        return []

    edges = np.linspace(0.0, 1.0, samples + 1)
    for _ in range(max(1, iters)):
        t, offset = _project_many(pts, poly)
        band = np.clip(np.digitize(t, edges) - 1, 0, samples - 1)

        verts: list[Point] = []
        for b in range(samples):
            sel = band == b
            if not np.any(sel):                     # nothing here: the line spans the gap
                continue
            t_mid = float((edges[b] + edges[b + 1]) / 2.0)
            base, normal = point_at(poly, t_mid)    # normal points LEFT (matches offset's sign)
            shift = _weighted_median(offset[sel], weight[sel])
            verts.append(tuple(base + shift * normal))

        if len(verts) < 2:
            return []
        poly = as_polyline([tuple(head)] + verts + [tuple(tail_tip)])
        if len(poly) < 2:
            return []

    return smooth([(float(x), float(y)) for x, y in poly], window, polyorder)


# --- the outline, cut in two ------------------------------------------------
def outline_halves(contour: np.ndarray, head: Point, tail_tip: Point,
                   midline: list[Point]) -> tuple[list[Point], list[Point]]:
    """The mask's contour, cut at the two tips -> (left, right), head-first.

    This is the old cyan/magenta pair, except nobody drew it: the closed contour is split at the
    points nearest the head and the tail tip, giving two arcs, and each arc is named by which
    side of the centre line it falls on. The model's chronic `opposite_sides` failure cannot
    happen — the two arcs are opposite halves of one closed curve by construction.
    """
    if contour is None or len(contour) < 4 or len(midline) < 2:
        return [], []
    pts = contour.reshape(-1, 2).astype(float)
    i_head = int(np.argmin(np.hypot(*(pts - np.asarray(head, float)).T)))
    i_tail = int(np.argmin(np.hypot(*(pts - np.asarray(tail_tip, float)).T)))
    if i_head == i_tail:
        return [], []

    rolled = np.roll(pts, -i_head, axis=0)          # start the loop at the head
    cut = (i_tail - i_head) % len(pts)
    arc_a = rolled[: cut + 1]                       # head -> tail, one way round
    arc_b = np.concatenate([rolled[cut:], rolled[:1]])[::-1]    # head -> tail, the other way

    poly = as_polyline(midline)
    def side(arc: np.ndarray) -> float:
        """Mean signed offset of an arc from the centre line: > 0 is left."""
        if len(arc) < 3:
            return 0.0
        _, offset = _project_many(arc[1:-1], poly)  # drop the shared tips, which sit ON the line
        return float(np.median(offset))

    a_left = side(arc_a) >= side(arc_b)
    left, right = (arc_a, arc_b) if a_left else (arc_b, arc_a)
    return ([(float(x), float(y)) for x, y in left],
            [(float(x), float(y)) for x, y in right])


# --- the gates (this is what replaces the judge) ----------------------------
def check(body: Body, mask: np.ndarray, width: int, height: int,
          min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
          max_area_frac: float = DEFAULT_MAX_AREA_FRAC,
          min_axis_frac: float = DEFAULT_MIN_AXIS_FRAC,
          min_inside: float = DEFAULT_MIN_INSIDE) -> None:
    """Grade the mask WITHOUT a model, filling ``body.checks`` and ``body.reason`` in place.

    The old judge existed because, as the two-outline module put it, no pixel metric can see
    whether a *drawn* line bisects the body. Once the body is a region, the bisection is
    COMPUTED — it is right by construction — and the only open question is whether the mask is a
    plausible salamander. That question is arithmetic:

      ``mask_found``   anything keyed at all
      ``area_sane``    the body covers a believable slice of the frame (not a speck, not the
                       whole photo — a flood-fill that escaped onto the background)
      ``dots_found``   both dots keyed, and both near enough to the mask to snap to a tip
      ``axis_long``    the centre line spans a real animal, not a collapsed blob
      ``line_inside``  the centre line lies ON the mask. This is the strong one: a line through
                       the body cannot leave it, so a vertex on the background means the mask is
                       two blobs, or C-curled tightly enough that a band spans the gap.

    A body that fails any gate keeps its geometry (the spots still bin) but is marked
    ``judged_ok = false``, exactly as a judge-rejected draft was.
    """
    diag = float(np.hypot(width, height))
    area = int(np.count_nonzero(mask))
    frac = area / float(max(width * height, 1))
    body.area_px = area

    inside = 0.0
    if body.midline:
        verts = np.asarray(body.midline, int)
        verts[:, 0] = np.clip(verts[:, 0], 0, width - 1)
        verts[:, 1] = np.clip(verts[:, 1], 0, height - 1)
        on = mask[verts[:, 1], verts[:, 0]] > 0
        inside = float(np.count_nonzero(on)) / len(on)

    body.checks = {
        "mask_found": area > 0,
        "area_sane": min_area_frac <= frac <= max_area_frac,
        "dots_found": body.head is not None and body.tail_tip is not None,
        "axis_long": body.length_px >= min_axis_frac * diag,
        "line_inside": inside >= min_inside,
    }
    notes = {
        "mask_found": "nothing was painted magenta",
        "area_sane": f"the painted body covers {frac:.1%} of the frame",
        "dots_found": "the green and/or red dot is missing or not on the body",
        "axis_long": f"the centre line is only {body.length_px / max(diag, 1):.1%} of the diagonal",
        "line_inside": f"only {inside:.0%} of the centre line lies on the painted body",
    }
    body.reason = "; ".join(notes[k] for k, v in body.checks.items() if not v)


# --- geometry from a known mask + tips (the offline half) --------------------
def build_body(mask: np.ndarray, head: Point, tail_tip: Point,
               samples: int = DEFAULT_SAMPLES, iters: int = DEFAULT_ITERS,
               min_axis_frac: float = DEFAULT_MIN_AXIS_FRAC) -> Body:
    """A body MASK plus its two tips -> a fully derived :class:`Body`. No image, no model.

    This is everything :func:`extract_body` does *after* the tips are known: the centre line,
    the two outline halves, and the gates. Split out so a SAVED mask can be re-processed for
    nothing — a different smoothing window, more bands, or corrected tips
    (:func:`correct_tips_from_shape`) — without repainting a single image. The masks on disk are
    the expensive artefact; this is how you get value out of keeping them.
    """
    h, w = mask.shape[:2]
    solid, contour = solidify(mask)
    body = Body(head=(float(head[0]), float(head[1])),
                tail_tip=(float(tail_tip[0]), float(tail_tip[1])))
    body.midline = centre_line(solid, body.head, body.tail_tip, samples, iters)
    if body.midline:
        body.left, body.right = outline_halves(contour, body.head, body.tail_tip, body.midline)
        body.source = "mask"
    check(body, solid, w, h, min_axis_frac=min_axis_frac)
    return body


# --- the whole thing --------------------------------------------------------
def extract_body(bgr: np.ndarray, morph: int = DEFAULT_MORPH,
                 samples: int = DEFAULT_SAMPLES, iters: int = DEFAULT_ITERS,
                 snap_frac: float = DEFAULT_SNAP_FRAC,
                 min_axis_frac: float = DEFAULT_MIN_AXIS_FRAC) -> tuple[Body, np.ndarray]:
    """The painted image -> (:class:`Body`, the cleaned mask). Never raises.

    Every gate is applied, so the returned Body already knows whether it passed. A Body that
    could not be recovered at all comes back with ``source="none"`` and an empty midline; its
    spots are simply left unbinned, as before.
    """
    h, w = bgr.shape[:2]
    diag = float(np.hypot(w, h))
    body = Body()

    # The magenta slab is the one mark nothing in nature imitates, so trust it FIRST and use it
    # to say where a dot is even allowed to be (see `dots_on` — otherwise moss wins).
    region, _ = solidify(fill_mask(bgr, morph))
    if not np.any(region):
        check(body, region, w, h, min_axis_frac=min_axis_frac)
        return body, region

    head_dot, tail_dot, dots = dots_on(bgr, region)
    mask, contour = solidify(body_of(bgr, dots, morph))
    if head_dot is None or tail_dot is None or not np.any(mask):
        check(body, mask, w, h, min_axis_frac=min_axis_frac)
        return body, mask

    # The dots only say WHICH END; the mask says where the end actually is. Each tip is the
    # point furthest from the OTHER tip measured along the body, so one geodesic map per end.
    radius = snap_frac * diag
    g_from_tail, scale = _geodesic_from(mask, tail_dot)
    g_from_head, _ = _geodesic_from(mask, head_dot)
    body.head = snap_tip(mask, head_dot, tail_dot, radius, g_from_tail, scale)
    body.tail_tip = snap_tip(mask, tail_dot, head_dot, radius, g_from_head, scale)
    if body.head is None or body.tail_tip is None:
        check(body, mask, w, h, min_axis_frac=min_axis_frac)
        return body, mask

    body.midline = centre_line(mask, body.head, body.tail_tip, samples, iters)
    if body.midline:
        body.left, body.right = outline_halves(contour, body.head, body.tail_tip, body.midline)
        body.source = "mask"

    check(body, mask, w, h, min_axis_frac=min_axis_frac)
    return body, mask


# --- geometric axis correction (no model, no image — just the saved mask) ----
# When the model paints a good body but drops the head/tail dots in the MIDDLE of it, the axis
# collapses: both tips end up mid-body and the centre line spans only part of the animal. That
# is not rare enough to ignore, and it is fixable from the mask ALONE — the body's two true ends
# are a property of its shape, not of where the model aimed.
#
# The wrong way to find them is a bounding box or a straight long axis: on a U-curled salamander
# the straight principal axis runs ACROSS the U and its extremes are the outer flanks of the two
# arms, not the snout and tail. Everything here is therefore measured ALONG THE BODY (geodesic),
# which is curvature-proof — for a bent rod the two ends are simply its farthest-apart pair.

DEFAULT_COVERAGE = 0.75     # correct only if the dots span LESS than this fraction of the body
DEFAULT_THICK_R = 0.18      # taper probe radius, as a fraction of body length


def _geodesic_endpoints(mask: np.ndarray):
    """The mask's two geodesically farthest-apart points + the body length between them.

    Robust to curvature by construction: two ends of a bent rod are its farthest-apart pair
    however it curls. Standard two-pass trick — farthest point from an arbitrary seed is one
    end; farthest from THAT is the other. Runs on the shrunk mask like the rest of the geodesic
    code. Returns (A, B in full-res xy, length in small-px, small mask, scale) or None.
    """
    from skimage.graph import MCP_Geometric

    h, w = mask.shape[:2]
    scale = min(1.0, GEODESIC_MAX_DIM / float(max(h, w)))
    small = (cv2.resize(mask, (max(int(w * scale), 1), max(int(h * scale), 1)),
                        interpolation=cv2.INTER_NEAREST) if scale < 1.0 else mask.copy())
    ys, xs = np.nonzero(small)
    if len(xs) < 10:
        return None
    costs = np.where(small > 0, 1.0, np.inf)

    def farthest(rc):
        d, _ = MCP_Geometric(costs).find_costs([rc])
        flat = np.where(np.isfinite(d[ys, xs]), d[ys, xs], -1.0)
        i = int(np.argmax(flat))
        return (int(ys[i]), int(xs[i])), d

    a_rc, _ = farthest((int(ys[0]), int(xs[0])))
    b_rc, d_a = farthest(a_rc)
    length = float(d_a[b_rc])
    if not np.isfinite(length) or length < 1.0:
        return None
    to_full = lambda rc: (rc[1] / scale, rc[0] / scale)     # (row, col) -> (x, y) full-res
    return to_full(a_rc), to_full(b_rc), length, small, scale


def _geo_between(small: np.ndarray, scale: float, p: Point, q: Point) -> float:
    """Geodesic distance (in small-px) between two full-res points, through the mask."""
    from skimage.graph import MCP_Geometric

    ys, xs = np.nonzero(small)
    costs = np.where(small > 0, 1.0, np.inf)

    def snap(pt):
        i = int(np.argmin(np.hypot(xs - pt[0] * scale, ys - pt[1] * scale)))
        return (int(ys[i]), int(xs[i]))

    d, _ = MCP_Geometric(costs).find_costs([snap(p)])
    val = d[snap(q)]
    return float(val) if np.isfinite(val) else float("inf")


def _thicker_end(small: np.ndarray, scale: float, a: Point, b: Point,
                 length: float, r_frac: float = DEFAULT_THICK_R) -> bool:
    """True if end ``a`` is the THICK end (more body within a probe radius) — i.e. the head.

    The taper prior: a salamander's head/trunk end carries more mass than its tapering tail. Used
    to orient the axis only when the dots cannot (both landed mid-body); when a dot does reach an
    end, that is trusted over this.
    """
    ys, xs = np.nonzero(small)
    pts = np.column_stack([xs, ys]).astype(float)
    r = r_frac * length
    ca = np.array([a[0] * scale, a[1] * scale])
    cb = np.array([b[0] * scale, b[1] * scale])
    area_a = int(np.count_nonzero(np.hypot(*(pts - ca).T) <= r))
    area_b = int(np.count_nonzero(np.hypot(*(pts - cb).T) <= r))
    return area_a >= area_b


def correct_tips_from_shape(mask: np.ndarray, head: Point | None, tail_tip: Point | None,
                            coverage: float = DEFAULT_COVERAGE):
    """Replace mid-body head/tail dots with the mask's own true ends. -> (head, tail, did, info).

    Fires ONLY when the current dots span less than ``coverage`` of the body length measured
    ALONG the body — i.e. the labelled axis misses a real chunk of animal (a dot stranded
    mid-body). A correct annotation whose dots already reach both ends spans ~100 % and is left
    untouched, U-curled or not. This is the conservative trigger you asked for, stated as body
    coverage rather than distance-to-a-box-edge so it survives curvature.

    Orientation: if the two dots point at DIFFERENT ends, that assignment is trusted (one good
    dot is enough to say which way round the animal is). Only when the dots cannot decide — both
    nearest the same end, or a dot missing — does it fall back to the taper prior (head = the
    thicker end). ``info['orient_by']`` records which, so the dry-run can flag the taper-only
    ones for a human glance.
    """
    info: dict = {"corrected": False}
    solid, _ = solidify(mask)
    ep = _geodesic_endpoints(solid)
    if ep is None:
        info["reason"] = "degenerate mask"
        return head, tail_tip, False, info
    a, b, length, small, scale = ep

    span = (_geo_between(small, scale, head, tail_tip)
            if head is not None and tail_tip is not None else 0.0)
    cov = span / length if np.isfinite(span) else 0.0
    info.update(diam_px=round(length / scale, 1),
                span_px=round((span if np.isfinite(span) else 0.0) / scale, 1),
                coverage=round(cov, 3))
    if cov >= coverage:
        info["reason"] = f"dots already span {cov:.0%} of the body — left alone"
        return head, tail_tip, False, info

    thick_is_a = _thicker_end(small, scale, a, b, length)
    vote = None
    if head is not None and tail_tip is not None:
        d = lambda p, e: float(np.hypot(p[0] - e[0], p[1] - e[1]))
        if d(head, a) < d(head, b) and d(tail_tip, b) < d(tail_tip, a):
            vote = True                                     # head nearer A, tail nearer B
        elif d(head, b) < d(head, a) and d(tail_tip, a) < d(tail_tip, b):
            vote = False
    head_is_a = vote if vote is not None else thick_is_a
    info["orient_by"] = "dots" if vote is not None else "taper"
    info["orient_agrees_taper"] = (vote == thick_is_a) if vote is not None else None

    new_head = a if head_is_a else b
    new_tail = b if head_is_a else a
    moved = max((float(np.hypot(head[0] - new_head[0], head[1] - new_head[1]))
                 if head is not None else 0.0),
                (float(np.hypot(tail_tip[0] - new_tail[0], tail_tip[1] - new_tail[1]))
                 if tail_tip is not None else 0.0))
    info.update(corrected=True, moved_px=round(moved, 1),
                reason=f"labelled axis covers only {cov:.0%} of the body; "
                       f"retipped to the geodesic ends (orient by {info['orient_by']})")
    return ((float(new_head[0]), float(new_head[1])),
            (float(new_tail[0]), float(new_tail[1])), True, info)


# --- QA render --------------------------------------------------------------
def overlay(bgr: np.ndarray, body: Body, mask: np.ndarray | None = None) -> np.ndarray:
    """The ORIGINAL photo with the derived geometry drawn on it, for a human to eyeball.

    The mask is tinted, the two outline halves are drawn cyan/magenta and the centre line white
    — so the QA image looks like the old one, except every mark on it was computed rather than
    drawn. If the white line does not run down the middle of the animal, the MASK was wrong.
    """
    out = bgr.copy()
    if mask is not None and np.any(mask):
        tint = np.zeros_like(out)
        tint[mask > 0] = (255, 0, 255)
        out = cv2.addWeighted(out, 0.75, tint, 0.25, 0)
    for poly, colour in ((body.left, (255, 255, 0)), (body.right, (255, 0, 255))):
        if len(poly) >= 2:
            cv2.polylines(out, [np.asarray(poly, np.int32).reshape(-1, 1, 2)], False,
                          colour, 3, cv2.LINE_AA)
    if len(body.midline) >= 2:
        pts = np.asarray(body.midline, np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, (0, 0, 0), 6, cv2.LINE_AA)          # halo
        cv2.polylines(out, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)
    if body.head:
        cv2.circle(out, tuple(np.asarray(body.head, int)), 10, (0, 255, 0), -1)
    if body.tail_tip:
        cv2.circle(out, tuple(np.asarray(body.tail_tip, int)), 10, (0, 0, 255), -1)
    return out
