#!/usr/bin/env python3
"""Stage 1b — body anatomy via Gemini: the head, the tail tip, and the two body outlines.

Gemini annotates a COPY of the photo with four marks:

    green dot    the FRONT-MOST tip of the snout
    red dot      the BACK-MOST tip of the tail
    cyan line    one edge of the body, head -> tail
    magenta line the other edge of the body, head -> tail

and **we compute the centre line ourselves**, as the mean of the two outlines (see
:func:`centre_line`).

The two dots are the axis's 0 % and 100 %, so their placement is not cosmetic: a green dot on the
middle of the head rather than on the snout tip shortens the axis and shifts every bin boundary
backwards along the body. The judge checks this explicitly (``dots_at_tips``).

Why not just ask for the centre line? Because we did, and it never worked. Asking a model to
bisect a body is asking it to compute a medial axis in its head: across every attempt, on every
model in the ladder, the judge's `bisects` criterion failed — the line was always drawn hugging
one flank. Tracing the two visible EDGES is a *perceptual* task the model is good at, and the
mean of two edges bisects BY CONSTRUCTION. The failure mode simply stops existing.

Two consequences worth knowing:

* It does not matter which outline the model puts on which side. The midpoint of two edges is
  symmetric, so swapping cyan and magenta yields the identical centre line.
* The legs are deliberately ignored (the prompt tells the model to pass straight across the base
  of a limb). We want the trunk-and-tail axis; toes would only pull the outlines — and therefore
  the midline — sideways.

The centre line being a true bisector is load-bearing, not cosmetic: a bin is (quarter along the
body) x (side of the line), so a line that hugs one flank pushes spots that belong on one side
over to the other, and bins 1/3/5/7 stop being comparable with 2/4/6/8.

The model is asked for an image ONLY — no JSON. Asking these image models for text alongside the
picture measurably increases how often they answer with prose and draw nothing at all; the dots
key reliably enough that the JSON was never needed. (:func:`parse_anatomy_json` is still used if
a reply happens to carry text, purely as an anchor fallback.)

**The marks are drawn on a copy of the ORIGINAL photo, never on the purple image.** Stage 2's
entire method is keying flat #FF00FF, and a red line running the length of the body would slice
every magenta spot it crosses into two. The copy is read for geometry and then kept only as QA;
stage 2 still keys the untouched purple image. Output::

    <input_dir>/anatomy/<stem>.png    the annotated copy (QA — never used downstream)
    <input_dir>/anatomy/<stem>.json   {head, tail_tip, midline, length_px, source}

Coordinates are pixels in the ORIGINAL photo's frame — the same frame the spot centroids live
in — because the model's output is conformed back onto the original's pixel grid first (see
:func:`llm_spot_segmentation.conform_to_original`).

Nothing here raises on a bad annotation: an image whose axis cannot be recovered gets
``source="none"`` and its spots are simply left unbinned.

Pure logic: no argument parsing. The CLI is ``scripts/dataset/extract_spot_labels.py anatomy``
(``pixi run extract-spot-labels anatomy``). The network is only touched inside this module.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from ._common import getenv, list_images, load_dotenv, resolve_input_dir
from .llm_spot_segmentation import conform_to_original, mime_for, parse_model_list
from .runlog import ACCEPTED, REGENERATE, RunLog

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

# The four marks. Gemini draws the two body OUTLINES; we derive the centre line ourselves as
# their mean. Asking a model to bisect a body is asking it to compute a medial axis in its head,
# which it cannot do — every draft failed `bisects`. Asking it to trace the two visible EDGES is
# a perceptual task it is good at, and the mean of two edges bisects BY CONSTRUCTION.
HEAD_HEX = "#00FF00"     # green dot — the head
TAIL_HEX = "#FF0000"     # red dot — the tip of the tail
LEFT_HEX = "#00FFFF"     # cyan line — one side of the body
RIGHT_HEX = "#FF00FF"    # magenta line — the other side
AXIS_HEX = TAIL_HEX      # back-compat alias

# Vision+text model that grades the drawn line; override with GEMINI_JUDGE_MODEL.
# NB: a model appearing in `client.models.list()` does NOT mean your key may call it —
# gemini-2.5-flash lists but 404s ("no longer available to new users"). Verify with a real
# generate_content call before changing this (see `GeminiAnatomyJudge.preflight`).
DEFAULT_JUDGE_MODEL = "gemini-3.5-flash"

PROMPT = f"""\
System Instruction / Role: Act as a precise image annotation engine. You are NOT re-imagining
the photo — you copy every pixel through unchanged and draw four marks on top of it.

Input: The attached photo of a fire salamander is the absolute baseline template. Keep the
same framing, the same animal, the same spots and the same background.

Task — draw exactly these four marks, and nothing else:

1. A small filled circle in pure green ({HEAD_HEX}) on the FRONT-MOST TIP OF THE SNOUT: the very
   foremost point of the animal, right at the front edge of its head where the head meets the
   background. NOT in the middle of the head, NOT between the eyes, NOT on the neck — on the
   nose tip itself, the last bit of salamander before the background begins.
2. A small filled circle in pure red ({TAIL_HEX}) on the BACK-MOST TIP OF THE TAIL: the very
   end of the tail, the furthest point from the head, right where the tapering tail finally
   stops and the background begins. NOT partway along the tail — on its final point.

   Marks 1 and 2 are the two EXTREME ENDS of the animal. The straight distance between them
   should span the animal's entire nose-to-tail extent; if you could trim the salamander to
   just what lies between your two dots, you would lose nothing of the snout and nothing of
   the tail.

3. A cyan ({LEFT_HEX}) line tracing the LEFT-HAND EDGE of the body, as you look at the image.
4. A magenta ({RIGHT_HEX}) line tracing the RIGHT-HAND EDGE of the body, as you look at the image.

Rules for the two edge lines (3 and 4):
- Each one runs the full length of the animal: it STARTS at the green head circle and ENDS at
  the red tail-tip circle. Unbroken, no gaps, no separate segments.
- Each one sits ON the outline of the body — where the animal meets the background — following
  every curve of that edge, all the way down the trunk and around the tapering tail.
- IGNORE THE LEGS, FEET AND TOES. Do not trace out around them. Where a leg sticks out, the
  line passes straight across its base, staying on the outline of the main body. You are
  outlining the trunk and tail only, as if the animal had no limbs.
- The two lines are on OPPOSITE sides of the animal. They must never cross each other, and
  neither may cut through the middle of the body.
- Do not draw a line down the centre of the animal. Neither line is a centre line — one hugs
  one edge, the other hugs the other edge.

The marks must be flat, fully saturated, unshaded colours. Use each colour ONLY for its own
mark. Do not recolour the spots. Do not add text, arrows, labels or shading.

Output the annotated IMAGE. Do not reply with a text description instead of an image."""

JUDGE_PROMPT = """\
You are a strict quality inspector. The attached image is a photo of a fire salamander that an
annotator has drawn four marks on: a GREEN dot on the head, a RED dot on the tip of the tail, a
CYAN line that should trace one edge of the body, and a MAGENTA line that should trace the other
edge. The legs and feet are deliberately ignored — the lines are meant to outline the trunk and
tail only.

Judge the marks against exactly four criteria. Be harsh — a mark that is "close" fails. Judge
only what is drawn; do not comment on the spots or the background.

1. dots_at_tips — Is the GREEN dot on the FRONT-MOST TIP OF THE SNOUT (the animal's foremost
   point, at the front edge of the head), and the RED dot on the BACK-MOST TIP OF THE TAIL (its
   rearmost point, where the tail finally ends)? It fails if the green dot sits in the middle of
   the head, behind the snout, or on the neck; it fails if the red dot sits partway along the
   tail rather than at its very end. The two dots must mark the animal's two extreme ends —
   there should be no salamander left in front of the green dot, and none behind the red dot.
2. goes_head_to_tail — Do BOTH lines run the full length of the animal, each STARTING at the
   green dot and ENDING at the red dot? It fails if either line stops short, starts partway down
   the body, or is broken into pieces.
3. follows_edges — Does each line sit ON the outline of the body, where the animal meets the
   background, following that edge's curves? It fails if a line drifts out onto the background,
   sinks inside the body away from its edge, or cuts across the animal. Ignoring the legs is
   CORRECT and must not be penalised — a line passing straight across the base of a leg is
   exactly right.
4. opposite_sides — Are the two lines on OPPOSITE edges of the animal — one down each side —
   without crossing each other? It fails if both lines hug the same edge, if they cross or
   overlap, or if either runs down the middle of the body instead of along an edge.

Return ONLY this JSON object (no prose, no code fence):
{"dots_at_tips": true|false, "goes_head_to_tail": true|false, "follows_edges": true|false,
 "opposite_sides": true|false, "feedback": "..."}

"feedback" is for the annotator who will REDRAW the marks. Write one to three short sentences
that say concretely what was wrong and where, in plain spatial language (e.g. "the green dot sits
on top of the head instead of on the snout tip, and the magenta line drifted off the tail onto
the ground"). If all four criteria pass, set feedback to ""."""

RETRY_PREFIX = """\
Your PREVIOUS attempt at this annotation was REJECTED by a reviewer. The second image attached is
your rejected attempt; the first image is the clean original you must annotate again.

The reviewer said:
{feedback}

Draw the marks again on the clean original, fixing exactly those problems. Do not repeat the same
mistake. The instructions are unchanged:

"""

# Marker keys (OpenCV HSV, H 0..179). The four hues are deliberately far apart, but green (60)
# and cyan (90) are close enough to need a gap between their bands, as are magenta (150) and red
# (0/179). The high S/V floors keep natural greens (moss, grass) out of the head key — foliage is
# never this saturated.
HEAD_HSV_LOW = (40, 100, 90)        # green ~ H 60
HEAD_HSV_HIGH = (80, 255, 255)
LEFT_HSV_LOW = (85, 100, 90)        # cyan ~ H 90
LEFT_HSV_HIGH = (100, 255, 255)
RIGHT_HSV_LOW = (135, 100, 90)      # magenta ~ H 150
RIGHT_HSV_HIGH = (165, 255, 255)
AXIS_HSV_LOW = (0, 110, 90)         # red straddles the hue origin -> two bands
AXIS_HSV_HIGH = (8, 255, 255)
AXIS_HSV_LOW2 = (172, 110, 90)
AXIS_HSV_HIGH2 = (179, 255, 255)

DEFAULT_MIN_AREA = 25       # ignore keyed blobs smaller than this (px^2)
DEFAULT_MORPH = 3           # morphological cleanup kernel
DEFAULT_SAMPLES = 24        # polyline vertices sampled along the keyed midline
DEFAULT_MAX_ATTEMPTS = 1    # judged drafts on the PRIMARY model before escalating
DEFAULT_MIN_AXIS_FRAC = 0.10  # reject an axis shorter than this fraction of the image diagonal

# Escalation ladder for the drawing model, cheapest first — same idea as stage 1's
# --escalate-models. One draft per rung, so a rejected line is retried by a BETTER model rather
# than by the same one that just failed:
#     attempt 1/3  gemini-2.5-flash-image
#     attempt 2/3  gemini-3.1-flash-image
#     attempt 3/3  gemini-3-pro-image
# The pricier rungs are only paid for on the images the cheap one could not get right.
DEFAULT_ESCALATE_MODELS = "gemini-3.1-flash-image,gemini-3-pro-image"
DEFAULT_ESCALATE_ATTEMPTS = 1   # judged drafts per escalate rung

# These image models are unreliable about actually RETURNING an image: given the same prompt and
# config they will sometimes reply with a text description instead ("no image part"). Measured
# across all three rungs and every response_modalities setting, so it cannot be configured away
# — it has to be retried through. Retries here are cheap relative to losing the image entirely.
DEFAULT_IMAGE_RETRIES = 3       # tries to coax an image out of ONE rung before escalating

_NORMALIZED_MAX = 1.5       # a 0..1 payload may overshoot slightly; 1.5+ means it is in pixels
_CANVAS_SLACK = 0.25        # tolerate this much overshoot (clamped); beyond it, drop the point

Point = tuple[float, float]


@dataclass
class Anatomy:
    """Head, tail tip, the two body outlines, and the centre line, in ORIGINAL-photo pixels.

    ``midline`` is NOT drawn by the model — it is computed as the mean of ``left`` and ``right``
    (see :func:`centre_line`), which is what makes it bisect the animal.
    """
    head: Point | None = None
    tail_tip: Point | None = None
    midline: list[Point] = field(default_factory=list)   # derived: the mean of the two edges
    left: list[Point] = field(default_factory=list)      # the cyan outline, as drawn
    right: list[Point] = field(default_factory=list)     # the magenta outline, as drawn
    source: str = "none"                # "outlines" | "none"
    judged_ok: bool | None = None       # the LLM judge's verdict (None = not judged)
    judge_feedback: str = ""            # why it was rejected, if it was

    @property
    def ok(self) -> bool:
        return self.head is not None and self.tail_tip is not None and len(self.midline) >= 2

    @property
    def length_px(self) -> float:
        if len(self.midline) < 2:
            return 0.0
        p = np.asarray(self.midline, float)
        return float(np.hypot(*(p[1:] - p[:-1]).T).sum())

    def as_dict(self) -> dict:
        return {"head": list(self.head) if self.head else None,
                "tail_tip": list(self.tail_tip) if self.tail_tip else None,
                "midline": [list(p) for p in self.midline],
                "left": [list(p) for p in self.left],
                "right": [list(p) for p in self.right],
                "length_px": round(self.length_px, 2),
                "source": self.source,
                "judged_ok": self.judged_ok,
                "judge_feedback": self.judge_feedback}

    @classmethod
    def from_dict(cls, d: dict) -> "Anatomy":
        head, tail = d.get("head"), d.get("tail_tip")
        return cls(head=tuple(head) if head else None,
                   tail_tip=tuple(tail) if tail else None,
                   midline=[tuple(p) for p in d.get("midline") or []],
                   left=[tuple(p) for p in d.get("left") or []],
                   right=[tuple(p) for p in d.get("right") or []],
                   source=d.get("source", "none"),
                   judged_ok=d.get("judged_ok"),
                   judge_feedback=d.get("judge_feedback", ""))


class JudgeVerdict(NamedTuple):
    """The judge's answer to the four questions, plus the note it wrote for the redraw.

    ``dots_at_tips`` comes first because the two dots anchor everything else: they are the 0 %
    and 100 % of the body axis, so a green dot placed on the middle of the head (rather than on
    the snout tip) shortens the axis and shifts every bin boundary backwards.
    """
    dots_at_tips: bool
    goes_head_to_tail: bool
    follows_edges: bool
    opposite_sides: bool
    feedback: str
    raw: str = ""              # the model's untouched reply, for debugging

    @property
    def passed(self) -> bool:
        return (self.dots_at_tips and self.goes_head_to_tail and self.follows_edges
                and self.opposite_sides)

    @property
    def score(self) -> int:
        """How many of the four criteria held — used to keep the least-bad draft."""
        return sum((self.dots_at_tips, self.goes_head_to_tail, self.follows_edges,
                    self.opposite_sides))

    @property
    def failures(self) -> list[str]:
        return [n for n, v in (("dots-at-tips", self.dots_at_tips),
                               ("head-to-tail", self.goes_head_to_tail),
                               ("follows-edges", self.follows_edges),
                               ("opposite-sides", self.opposite_sides)) if not v]

    @classmethod
    def unusable(cls, why: str) -> "JudgeVerdict":
        """The draft failed before it could even be judged (no marks found)."""
        return cls(False, False, False, False, why)


JUDGE_LOG_NAME = "pipeline_log.jsonl"

# The draw/judge audit trail now lives in the shared run log, so stage 1 (purple), stage 1b
# (anatomy) and stage 2 (contours) all append to ONE file for a whole dataset build.
AnatomyLog = RunLog


def parse_verdict(text: str) -> JudgeVerdict | None:
    """The judge's JSON reply -> a `JudgeVerdict`; None if it cannot be parsed."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        raw = json.loads(m.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None

    def flag(key: str) -> bool:
        v = raw.get(key)
        if isinstance(v, str):                 # models sometimes answer "true"/"yes"
            return v.strip().lower() in ("true", "yes", "1", "pass")
        return bool(v)

    return JudgeVerdict(flag("dots_at_tips"), flag("goes_head_to_tail"),
                        flag("follows_edges"), flag("opposite_sides"),
                        str(raw.get("feedback") or "").strip(), text)


# --- the JSON half (authoritative for the anchors) --------------------------
def _pair(pt) -> Point | None:
    try:
        x, y = float(pt[0]), float(pt[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (x, y) if np.isfinite(x) and np.isfinite(y) else None


def parse_anatomy_json(text: str, w: int, h: int) -> dict:
    """The model's JSON reply -> {head, tail_tip, midline} in pixels; {} if unusable.

    Tolerates a code fence or surrounding prose, and accepts coordinates either normalized
    (0..1, as asked) or already in pixels — models drift between the two. That call is made
    ONCE, from the largest coordinate in the whole payload, never per point: a reply that
    overshoots to y=1.06 on one vertex of an otherwise normalized payload must not have that
    vertex alone read as "6 pixels down", which would fold the midline in half.
    """
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.DOTALL)   # first {...} block, fence or prose be damned
    if not m:
        return {}
    try:
        raw = json.loads(m.group(0))
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(raw, dict):
        return {}

    head, tail = _pair(raw.get("head")), _pair(raw.get("tail_tip"))
    mid = [p for p in (_pair(q) for q in raw.get("midline") or []) if p is not None]
    every = [p for p in (head, tail) if p is not None] + mid
    if not every:
        return {}

    biggest = max(max(abs(x), abs(y)) for x, y in every)
    sx, sy = (w, h) if biggest <= _NORMALIZED_MAX else (1.0, 1.0)

    def to_px(pt: Point | None) -> Point | None:
        if pt is None:
            return None
        x, y = pt[0] * sx, pt[1] * sy
        if not (-_CANVAS_SLACK * w <= x <= w * (1 + _CANVAS_SLACK)
                and -_CANVAS_SLACK * h <= y <= h * (1 + _CANVAS_SLACK)):
            return None                     # wildly off-canvas -> a hallucination
        return (min(max(x, 0.0), float(w)), min(max(y, 0.0), float(h)))

    out: dict = {}
    for key, pt in (("head", head), ("tail_tip", tail)):
        px = to_px(pt)
        if px is not None:
            out[key] = px
    poly = [p for p in (to_px(q) for q in mid) if p is not None]
    if len(poly) >= 2:
        out["midline"] = poly
    return out


# --- the keyed half (authoritative for the curve) ---------------------------
def _clean(mask: np.ndarray, morph: int) -> np.ndarray:
    if morph and morph >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def head_mask(bgr: np.ndarray, morph: int = DEFAULT_MORPH) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return _clean(cv2.inRange(hsv, np.array(HEAD_HSV_LOW, np.uint8),
                              np.array(HEAD_HSV_HIGH, np.uint8)), morph)


def axis_mask(bgr: np.ndarray, morph: int = DEFAULT_MORPH) -> np.ndarray:
    """Red mask (the tail-tip dot). Red straddles hue 0, so it takes two bands."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = cv2.inRange(hsv, np.array(AXIS_HSV_LOW, np.uint8), np.array(AXIS_HSV_HIGH, np.uint8))
    hi = cv2.inRange(hsv, np.array(AXIS_HSV_LOW2, np.uint8), np.array(AXIS_HSV_HIGH2, np.uint8))
    return _clean(cv2.bitwise_or(lo, hi), morph)


def left_mask(bgr: np.ndarray, morph: int = DEFAULT_MORPH) -> np.ndarray:
    """Cyan mask — one of the two body outlines."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return _clean(cv2.inRange(hsv, np.array(LEFT_HSV_LOW, np.uint8),
                              np.array(LEFT_HSV_HIGH, np.uint8)), morph)


def right_mask(bgr: np.ndarray, morph: int = DEFAULT_MORPH) -> np.ndarray:
    """Magenta mask — the other body outline."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return _clean(cv2.inRange(hsv, np.array(RIGHT_HSV_LOW, np.uint8),
                              np.array(RIGHT_HSV_HIGH, np.uint8)), morph)


def _banded(mask: np.ndarray, head: Point, tail_tip: Point, samples: int):
    """Slice a mask's pixels into `samples` bands along the head->tail chord.

    Returns (list of per-band mean pixels or None, n_pixels). A band with no pixels is None, so
    callers can tell "this side is missing here" from "this side is at (x, y) here".
    """
    h, t = np.asarray(head, float), np.asarray(tail_tip, float)
    span = t - h
    L = float(np.hypot(*span))
    ys, xs = np.nonzero(mask)
    if L < 1e-6 or len(xs) == 0:
        return [None] * samples, 0

    pts = np.column_stack([xs, ys]).astype(float)
    proj = (pts - h) @ (span / L) / L                # 0 at the head, 1 at the tail tip
    inside = (proj >= 0.0) & (proj <= 1.0)
    pts, proj = pts[inside], proj[inside]
    if len(pts) == 0:
        return [None] * samples, 0

    edges = np.linspace(0.0, 1.0, samples + 1)
    band = np.clip(np.digitize(proj, edges) - 1, 0, samples - 1)
    out = []
    for b in range(samples):
        sel = pts[band == b]
        out.append(tuple(sel.mean(0)) if len(sel) else None)
    return out, len(pts)


def centre_line(left: np.ndarray, right: np.ndarray, head: Point, tail_tip: Point,
                samples: int = DEFAULT_SAMPLES) -> list[Point]:
    """The two outlines -> the body's centre line, as the MEAN of the two edges.

    Both masks are sliced into the same bands along the head->tail chord; in each band the mean
    cyan pixel and the mean magenta pixel are averaged, giving a point exactly halfway between
    the two edges *at that point along the body*. Stitching those midpoints head-to-tail yields
    a line that bisects the animal BY CONSTRUCTION — which is the whole reason we ask for edges
    instead of asking the model for a centre line it cannot compute.

    Because the midpoint of two edges is symmetric, it does not matter which outline the model
    put on which side: swapping cyan and magenta yields the identical centre line.

    Bands where only one edge was drawn are skipped rather than guessed (the polyline simply
    spans the gap). Returns [] if the two edges never overlap in any band.
    """
    lb, ln = _banded(left, head, tail_tip, samples)
    rb, rn = _banded(right, head, tail_tip, samples)
    if ln == 0 or rn == 0:
        return []

    mid = [tuple((np.asarray(l) + np.asarray(r)) / 2.0)
           for l, r in zip(lb, rb) if l is not None and r is not None]
    if len(mid) < 2:
        return []
    return [tuple(head)] + mid + [tuple(tail_tip)]


def _largest_centroid(mask: np.ndarray, min_area: int) -> Point | None:
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    best, best_area = None, min_area - 1
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best, best_area = i, area
    return (float(cents[best][0]), float(cents[best][1])) if best is not None else None


def trace_polyline(mask: np.ndarray, head: Point, tail_tip: Point,
                   samples: int = DEFAULT_SAMPLES) -> list[Point]:
    """Order one drawn line's pixels into a head->tail polyline.

    The pixels are sliced into ``samples`` bands by their projection onto the head->tail chord
    and each band collapses to its mean pixel, so the vertices come out ordered head-first and
    tracking the line's actual path rather than the chord. Used to store each body outline;
    the centre line itself comes from :func:`centre_line`.
    """
    bands, n = _banded(mask, head, tail_tip, samples)
    if n < samples:
        return []
    verts = [b for b in bands if b is not None]
    return [tuple(head)] + verts + [tuple(tail_tip)] if len(verts) >= 2 else []


def key_markers(bgr: np.ndarray, min_area: int = DEFAULT_MIN_AREA,
                morph: int = DEFAULT_MORPH) -> dict:
    """The drawn annotation -> {head, tail_tip, left, right} (masks for the two outlines)."""
    out: dict = {}
    head = _largest_centroid(head_mask(bgr, morph), min_area)
    if head is not None:
        out["head"] = head
    tail = _largest_centroid(axis_mask(bgr, morph), min_area)
    if tail is not None:
        out["tail_tip"] = tail
    out["left"] = left_mask(bgr, morph)
    out["right"] = right_mask(bgr, morph)
    return out


def extract_anatomy(bgr: np.ndarray, text: str = "", min_area: int = DEFAULT_MIN_AREA,
                    morph: int = DEFAULT_MORPH, samples: int = DEFAULT_SAMPLES) -> Anatomy:
    """The annotated image -> one `Anatomy` (never raises).

    The head and tail tip are the green and red dots. The centre line is computed HERE, from the
    two drawn outlines — the model is never asked for it. ``text`` is an optional JSON reply
    (only ``gemini-2.5-flash-image`` ever returns one); it is used solely as a fallback for the
    two anchors if a dot could not be keyed.
    """
    h, w = bgr.shape[:2]
    js = parse_anatomy_json(text, w, h) if text else {}
    keyed = key_markers(bgr, min_area, morph)
    left, right = keyed["left"], keyed["right"]

    head = keyed.get("head") or js.get("head")
    tail_tip = keyed.get("tail_tip") or js.get("tail_tip")
    if head is None or tail_tip is None:
        return Anatomy(head=head, tail_tip=tail_tip, source="none")

    midline = centre_line(left, right, head, tail_tip, samples)
    if not midline:
        # Both dots landed, but the two edges never overlapped — one side is missing or they
        # were drawn on top of each other. Do not fake a centre line from the chord: say so, so
        # the judge/ladder redraws it.
        return Anatomy(head=head, tail_tip=tail_tip, source="none")

    return Anatomy(head=head, tail_tip=tail_tip, midline=midline,
                   left=trace_polyline(left, head, tail_tip, samples),
                   right=trace_polyline(right, head, tail_tip, samples),
                   source="outlines")


# --- paths ------------------------------------------------------------------
def anatomy_dir_for(input_dir: Path) -> Path:
    return input_dir / "anatomy"


def anatomy_json_for(anatomy_dir: Path, src: Path) -> Path:
    return anatomy_dir / f"{src.stem}.json"


def anatomy_png_for(anatomy_dir: Path, src: Path) -> Path:
    return anatomy_dir / f"{src.stem}.png"


def load_anatomy(anatomy_dir: Path, stem: str) -> Anatomy | None:
    """One image's stored axis, or None if it was never produced / is unreadable."""
    path = anatomy_dir / f"{stem}.json"
    if not path.is_file():
        return None
    try:
        return Anatomy.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


# --- the Gemini call --------------------------------------------------------
class AnatomyResult(NamedTuple):
    png: bytes                          # the annotated copy (QA)
    anatomy: Anatomy
    passed: bool                        # cleared both gates (local check + the judge)
    attempts: int
    model: str
    verdict: "JudgeVerdict | None" = None   # the judge's answer, when it ran


class NoImageError(RuntimeError):
    """The model answered with prose instead of drawing. Re-ask; do not abandon the image."""


class FatalJudgeError(RuntimeError):
    """The judge is misconfigured (bad model id, bad key) — retrying cannot help.

    Raised instead of degrading to "accept unjudged", because an unreachable judge means the
    ONLY thing checking that the line is on the body and centred is gone. Silently accepting
    every draft would look exactly like a clean run while producing untrustworthy bins.
    """


# Substrings of a permanent, configuration-level failure: no amount of retrying fixes these.
_PERMANENT = ("not_found", "404", "permission_denied", "403", "unauthenticated", "401",
              "api key not valid", "invalid_argument", "400")


def _is_permanent(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in _PERMANENT)


class GeminiAnatomyJudge:
    """LLM-as-judge over a drawn annotation: three questions, one verdict, one note.

    Deliberately a *different* model from the annotator — a vision+text model, which is both
    cheaper and better at holding three criteria in mind than the image model is at grading its
    own work. The three failures it exists to catch (line off the body, line off-centre, line
    not spanning head to tail) are all semantic: no pixel metric sees them, because the marks
    themselves are perfectly well-formed flat colour in every case.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 max_retries: int = 4, base_delay: float = 2.0):
        dotenv = load_dotenv()
        self.api_key = api_key or getenv("GEMINI_API_KEY", "", dotenv)
        self.model = model or getenv("GEMINI_JUDGE_MODEL", DEFAULT_JUDGE_MODEL, dotenv)
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is empty (set it in salamander_spotter/.env)")
        self.max_retries = max_retries
        self.base_delay = base_delay
        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    def preflight(self) -> None:
        """Prove the judge model actually answers, BEFORE any billed image generation.

        Costs one tiny call. Worth it: without it a bad model id is only discovered after the
        first (expensive) draw, and across a 600-image run that is 600 wasted image calls with
        nothing grading them. Listing a model is not proof you can call it.
        """
        from google.genai import types
        probe = _probe_png()
        try:
            self._client.models.generate_content(
                model=self.model,
                contents=[types.Part.from_bytes(data=probe, mime_type="image/png"),
                          'Reply with only this JSON: {"ok": true}'])
        except Exception as exc:
            if _is_permanent(exc):
                raise FatalJudgeError(
                    f"the judge model {self.model!r} is not usable with this API key:\n"
                    f"    {exc}\n"
                    f"  Pick one your key can actually call, e.g.:\n"
                    f"    --judge-model gemini-3.5-flash        (recommended)\n"
                    f"    --judge-model gemini-flash-latest\n"
                    f"    --judge-model gemini-3-flash-preview\n"
                    f"  or set GEMINI_JUDGE_MODEL in .env. To run without any grading of the\n"
                    f"  drawn line (NOT recommended — nothing then catches a line off the body\n"
                    f"  or off-centre), pass --no-judge."
                ) from exc
            raise                                   # transient: let the caller see it

    def judge(self, annotated_png: bytes) -> JudgeVerdict | None:
        """Grade one annotated image.

        Returns None only for a *transient* failure that survived every retry — the draft is
        then accepted unjudged rather than stalling the run. A permanent failure (bad model,
        bad key) raises :class:`FatalJudgeError` and stops the run instead.
        """
        from google.genai import types

        contents = [types.Part.from_bytes(data=annotated_png, mime_type="image/png"),
                    JUDGE_PROMPT]
        delay = self.base_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=contents)
                return parse_verdict(getattr(resp, "text", "") or "")
            except Exception as exc:
                if _is_permanent(exc):
                    raise FatalJudgeError(
                        f"the judge model {self.model!r} stopped being usable mid-run:\n"
                        f"    {exc}\n  Re-run with --judge-model <a model your key can call>."
                    ) from exc
                if attempt == self.max_retries:
                    logger.warning(f"  warning: judge unreachable after {attempt} tries ({exc}) — "
                          "accepting this draft UNJUDGED")
                    return None
                msg = str(exc).lower()
                cool = delay * 4 if ("429" in msg or "quota" in msg or "rate" in msg) else delay
                time.sleep(cool + random.uniform(0, delay))
                delay = min(delay * 2, 60.0)
        return None


class GeminiAnatomyAnnotator:
    """Thin wrapper over google-genai for the anatomy-annotation call.

    Carries an escalation ladder like :class:`~.llm_spot_segmentation.GeminiSpotSegmenter`:
    the primary (cheap) model gets ``max_attempts`` draws, then each pricier rung in
    ``escalate_models`` gets ``escalate_attempts`` more — but only for images the judge has
    not already accepted, so the expensive model is paid for only where it is needed.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 max_retries: int = 5, base_delay: float = 2.0,
                 temperature: float | None = None,
                 escalate_models: str | list[str] | None = None,
                 escalate_attempts: int = DEFAULT_ESCALATE_ATTEMPTS):
        dotenv = load_dotenv()
        self.api_key = api_key or getenv("GEMINI_API_KEY", "", dotenv)
        self.model = model or getenv("GEMINI_MODEL", "gemini-2.5-flash-image", dotenv)
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is empty (set it in salamander_spotter/.env)")
        raw = (escalate_models if escalate_models is not None
               else getenv("GEMINI_ANATOMY_ESCALATE_MODELS", DEFAULT_ESCALATE_MODELS, dotenv))
        self.escalate_models = [m for m in parse_model_list(raw) if m != self.model]
        self.escalate_attempts = escalate_attempts
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.temperature = temperature
        from google import genai  # lazy import — only needed for the real call
        self._client = genai.Client(api_key=self.api_key)

    def tiers(self, max_attempts: int) -> list[tuple[str, int]]:
        """The ladder as (model, draws), cheapest first."""
        rungs = [(self.model, max(1, max_attempts))]
        if self.escalate_attempts > 0:
            rungs.extend((m, self.escalate_attempts) for m in self.escalate_models)
        return rungs

    def ladder_str(self, max_attempts: int) -> str:
        return " -> ".join(f"{m} x{n}" for m, n in self.tiers(max_attempts))

    @staticmethod
    def _parts(resp) -> tuple[bytes | None, str]:
        """(first inline image, all text joined) out of a generate_content response."""
        image, text = None, []
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if image is None and inline and getattr(inline, "data", None):
                    image = inline.data
                if getattr(part, "text", None):
                    text.append(part.text)
        return image, "\n".join(text)

    def annotate(self, image_bytes: bytes, mime_type: str = "image/jpeg",
                 model: str | None = None, prompt: str | None = None,
                 prior_png: bytes | None = None) -> tuple[bytes, str]:
        """One call, two products: the annotated image and the coordinates as JSON.

        On a re-draw, ``prior_png`` (the rejected attempt) is attached as a SECOND image after
        the clean original, and ``prompt`` carries the judge's complaint — so the model can see
        what it got wrong rather than being asked to try again blind.
        """
        from google.genai import types

        contents: list = [types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
        if prior_png:
            contents.append(types.Part.from_bytes(data=prior_png, mime_type="image/png"))
        contents.append(prompt or PROMPT)
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            **({"temperature": self.temperature} if self.temperature is not None else {}),
        )
        model = model or self.model
        delay = self.base_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=model, contents=contents, config=config)
            except Exception as exc:  # broad: SDK error taxonomy varies by version
                if _is_permanent(exc) or attempt == self.max_retries:
                    raise
                msg = str(exc).lower()
                cool = delay * 4 if ("429" in msg or "quota" in msg or "rate" in msg) else delay
                time.sleep(cool + random.uniform(0, delay))   # backoff + jitter
                delay = min(delay * 2, 60.0)
                continue

            image, text = self._parts(resp)
            if image is None:
                # The CALL succeeded — the model simply chose to answer in prose rather than
                # draw. Backing off and repeating the same transport would not help, so this is
                # raised for the caller to re-ask (and, if the rung keeps refusing, escalate).
                raise NoImageError(
                    f"{model} returned no image part (it replied with text instead)")
            return image, text
        raise RuntimeError("unreachable")

    def annotate_file(self, src: Path, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                      min_axis_frac: float = DEFAULT_MIN_AXIS_FRAC,
                      keep_gemini_size: bool = False,
                      judge: "GeminiAnatomyJudge | None" = None,
                      log: "AnatomyLog | None" = None,
                      image_retries: int = DEFAULT_IMAGE_RETRIES) -> AnatomyResult:
        """Annotate ``src``, re-drawing with the judge's feedback until the line is right.

        Two gates, cheapest first:

        1. A local geometric check — were the marks found at all, and is the midline at least
           ``min_axis_frac`` of the image diagonal? This catches "drew nothing" and "collapsed
           the axis to a dot" for free, without spending a judge call.
        2. The LLM judge (:class:`GeminiAnatomyJudge`) — does the line run head to tail, stay
           inside the body, and bisect it? These are the failures no pixel metric can see.

        A rejected draft is not simply retried: the judge's written complaint AND the rejected
        image itself are fed back into the next draw, so the model corrects a specific mistake
        rather than resampling blindly. If no draft ever passes, the best one (most criteria
        met, then longest axis) is returned with ``passed=False`` and the verdict attached, so
        the caller can flag it — its spots are still binned, just marked untrustworthy.

        The model re-renders at its own resolution, so each draft is conformed back onto the
        original's pixel grid FIRST — that is what puts the recovered coordinates in the same
        frame as the spot centroids, and it is also the image the judge grades.
        """
        src_bytes, mime = src.read_bytes(), mime_for(src)
        original = cv2.imread(str(src), cv2.IMREAD_COLOR)
        diag = float(np.hypot(*original.shape[:2])) if original is not None else 0.0
        floor = min_axis_frac * diag

        rungs = self.tiers(max_attempts)
        total = sum(n for _, n in rungs)
        judge_model = judge.model if judge is not None else None
        best: AnatomyResult | None = None
        best_score = -1
        feedback = ""
        prior_png: bytes | None = None
        draws = 0

        def record(model: str, verdict: JudgeVerdict | None, result: str,
                   anat: Anatomy, note: str = "") -> None:
            if log is not None:
                log.attempt(image=src.name, attempt=draws, max_attempts=total,
                            draw_model=model, judge_model=judge_model,
                            verdict=verdict, result=result, anatomy=anat, note=note)

        for model, n_draws in rungs:
            for _ in range(n_draws):
                # Coax an image out of THIS rung. These models intermittently answer in prose
                # instead of drawing; a rung that never draws is escalated past, not fatal.
                png = text = None
                last: Exception | None = None
                for tri in range(1, max(1, image_retries) + 1):
                    prompt = ((RETRY_PREFIX.format(feedback=feedback) + PROMPT) if feedback
                              else PROMPT)
                    try:
                        raw, text = self.annotate(src_bytes, mime, model=model, prompt=prompt,
                                                  prior_png=prior_png)
                        png = raw if keep_gemini_size else conform_to_original(raw, src)
                        break
                    except NoImageError as exc:
                        logger.warning(f"  {model}: no image returned "
                              f"(try {tri}/{image_retries}) — re-asking")
                        last = exc
                    except Exception as exc:
                        if _is_permanent(exc):
                            logger.warning(f"  warning: draw model {model!r} is not usable with this "
                                  f"key ({str(exc)[:70]}) — skipping this rung")
                            last = exc
                            break
                        raise
                if png is None:
                    # This rung would not draw. Log it and climb the ladder.
                    draws += 1
                    record(model, None, REGENERATE, Anatomy(),
                           f"{model} never returned an image ({last}) — escalating")
                    break

                draws += 1
                bgr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    record(model, None, REGENERATE, Anatomy(),
                           "the model returned an undecodable image")
                    continue
                anat = extract_anatomy(bgr, text or "")

                # Gate 1 — local, free. Catches "drew nothing" without spending a judge call.
                if not anat.ok or anat.length_px < floor:
                    verdict = JudgeVerdict.unusable(
                        "The marks could not be read. Draw all four: a green dot on the "
                        "front-most tip of the snout, a red dot on the back-most tip of the "
                        "tail, and TWO separate lines — one cyan, one magenta — each running "
                        "the whole length of the animal along OPPOSITE edges of its body, from "
                        "the green dot to the red dot.")
                    record(model, None, REGENERATE, anat,
                           "no usable marks (a dot or one of the two outlines is missing)")
                elif judge is None:                 # judging disabled — gate 1 is the whole bar
                    record(model, None, ACCEPTED, anat, "judging disabled")
                    return AnatomyResult(png, anat, True, draws, model)
                else:
                    # Gate 2 — the LLM judge.
                    verdict = judge.judge(png)
                    if verdict is None:             # transient judge outage: accept, don't stall
                        record(model, None, ACCEPTED, anat,
                               "judge unreachable — accepted unjudged")
                        return AnatomyResult(png, anat, True, draws, model)
                    anat.judged_ok = verdict.passed
                    anat.judge_feedback = verdict.feedback
                    record(model, verdict, ACCEPTED if verdict.passed else REGENERATE, anat)
                    if verdict.passed:
                        return AnatomyResult(png, anat, True, draws, model, verdict)

                result = AnatomyResult(png, anat, False, draws, model, verdict)
                if verdict.score > best_score or (
                        verdict.score == best_score and best is not None
                        and anat.length_px > best.anatomy.length_px):
                    best, best_score = result, verdict.score
                # The next rung inherits the complaint AND the rejected image, so escalating
                # is a correction, not a fresh start.
                feedback = verdict.feedback or "The red line was not a correct centre line."
                prior_png = png

        if best is None:                            # every draft failed to even decode
            return AnatomyResult(b"", Anatomy(), False, max(draws, 1), self.model)
        return best._replace(attempts=draws)


def _probe_png() -> bytes:
    """A tiny throwaway image for the judge preflight."""
    img = np.zeros((16, 16, 3), np.uint8)
    cv2.circle(img, (8, 8), 4, (0, 0, 255), -1)
    return cv2.imencode(".png", img)[1].tobytes()


def make_judge(enabled: bool, model: str | None = None) -> GeminiAnatomyJudge | None:
    """The judge, or None when judging is turned off.

    Preflights the model so a bad id fails here — before a single billed image is drawn —
    rather than after the first draw of every image in the directory.
    """
    if not enabled:
        return None
    j = GeminiAnatomyJudge(model=model)
    j.preflight()
    return j


def annotate_dir(
    input: str,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    model: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_axis_frac: float = DEFAULT_MIN_AXIS_FRAC,
    temperature: float | None = None,
    judge: bool = True,
    judge_model: str | None = None,
    escalate_models: str | list[str] | None = None,
    escalate_attempts: int = DEFAULT_ESCALATE_ATTEMPTS,
    image_retries: int = DEFAULT_IMAGE_RETRIES,
    run_log: RunLog | None = None,
) -> int:
    """Stage 1b over one input dir: write ``anatomy/<stem>.{png,json}`` for each image.

    Existing anatomy is reused unless ``overwrite``, so a re-run only pays for what is
    missing. Returns a process exit code (non-zero if any image errored).
    """
    input_dir = resolve_input_dir(input)
    out_dir = anatomy_dir_for(input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if limit is not None:
        images = images[:limit]
    if not images:
        logger.error(f"error: no images found in {input_dir}")
        return 1

    annotator = GeminiAnatomyAnnotator(model=model, temperature=temperature,
                                       escalate_models=escalate_models,
                                       escalate_attempts=escalate_attempts)
    the_judge = make_judge(judge, judge_model)
    the_log = (run_log.child(task="anatomy", input=input_dir.name) if run_log
               else make_log(out_dir, input=input_dir.name))
    logger.info(f"annotating anatomy in dir {input_dir}")
    logger.info(f"  draw ladder: {annotator.ladder_str(max_attempts)}")
    logger.info(f"  judge model: {the_judge.model if the_judge else 'OFF (--no-judge)'}")
    logger.info(f"  log        : {the_log.path}")
    total, failures, flagged = len(images), 0, 0
    rejected: list[str] = []
    for i, src in enumerate(images, start=1):
        logger.info(f"processing image {i} / {total}: {src.name}")
        if anatomy_json_for(out_dir, src).is_file() and not overwrite:
            logger.info("  anatomy exists, skipping (use --overwrite to redo)")
            continue
        try:
            res = annotate_file_to_disk(annotator, src, out_dir,
                                        max_attempts=max_attempts,
                                        min_axis_frac=min_axis_frac, judge=the_judge,
                                        log=the_log, image_retries=image_retries)
            if not res.passed:
                flagged += 1
                rejected.append(src.name)
        except FatalJudgeError:
            raise            # misconfigured judge: stop, do not burn the rest of the dir
        except Exception as exc:
            failures += 1
            logger.error(f"  ERROR: {exc}")

    if rejected:
        # A single-column CSV that feeds straight back into `all --rewrite`, so a follow-up
        # run redoes only the images whose axis the judge never accepted.
        csv_path = out_dir / "flagged_axis.csv"
        csv_path.write_text(
            "# images whose line the judge never accepted; redo with --rewrite\n"
            + "\n".join(rejected) + "\n", encoding="utf-8")
        logger.info(f"flagged {len(rejected)} image(s) -> {csv_path}")

    done = total - failures
    logger.info(f"done: {done}/{total} anatomy labels in {out_dir}"
          + (f", {flagged} rejected by the judge" if flagged else "")
          + (f" ({failures} failed)" if failures else ""))
    logger.info(f"draw/judge log -> {the_log.path}")
    return 1 if failures else 0


def make_log(out_dir: Path, echo: bool = True, path: Path | None = None,
             run_id: str | None = None, step: int | None = None,
             input: str = "") -> RunLog:
    """The stage-1b view on the run log. `path` points it at a whole-build log instead."""
    return RunLog(path or (out_dir / JUDGE_LOG_NAME), run_id=run_id, echo=echo,
                  step=step, task="anatomy", input=input)


def with_midline(png: bytes, anatomy: Anatomy) -> bytes:
    """The annotated PNG with the DERIVED centre line drawn on it in white, for QA."""
    if not anatomy.ok:
        return png
    bgr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return png
    pts = np.asarray(anatomy.midline, np.int32).reshape(-1, 1, 2)
    cv2.polylines(bgr, [pts], False, (0, 0, 0), 5, cv2.LINE_AA)      # halo
    cv2.polylines(bgr, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes() if ok else png


def annotate_file_to_disk(annotator: GeminiAnatomyAnnotator, src: Path, out_dir: Path,
                          max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                          min_axis_frac: float = DEFAULT_MIN_AXIS_FRAC,
                          judge: GeminiAnatomyJudge | None = None,
                          log: AnatomyLog | None = None,
                          image_retries: int = DEFAULT_IMAGE_RETRIES) -> AnatomyResult:
    """One image: annotate (judged + re-drawn), persist the QA png + the geometry json, report.

    Shared by :func:`annotate_dir` and the ``all`` runner so both behave identically.
    """
    res = annotator.annotate_file(src, max_attempts=max_attempts,
                                  min_axis_frac=min_axis_frac, judge=judge, log=log,
                                  image_retries=image_retries)
    if res.png:
        # Save the drawing with OUR derived centre line drawn on it in white. The judge graded
        # the raw bytes; this copy is for a human to eyeball — if the white line does not run
        # down the middle, the two outlines were wrong even if the judge let them through.
        anatomy_png_for(out_dir, src).write_bytes(with_midline(res.png, res.anatomy))
    anatomy_json_for(out_dir, src).write_text(
        json.dumps(res.anatomy.as_dict(), indent=2), encoding="utf-8")
    if log is not None:
        log.summary(image=src.name, attempts=res.attempts, accepted=res.passed,
                    anatomy=res.anatomy)

    a = res.anatomy
    if res.passed:
        pass                                    # the log already printed "IMAGE ACCEPTED"
    elif a.ok:
        fails = ", ".join(res.verdict.failures) if res.verdict else "unusable"
        logger.warning(f"  GIVING UP after {res.attempts} draw(s), ladder exhausted: the judge never "
              f"accepted the line [{fails}]. Keeping the best draft ({res.model}) — its spots "
              f"WILL be binned but marked judged_ok=false")
    else:
        logger.warning(f"  GIVING UP after {res.attempts} draw(s): no usable axis "
              f"[{a.source}] — spots will be unbinned")
    return res
