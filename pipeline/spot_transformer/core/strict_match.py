"""Stricter spot-match scoring.

Manual pair review (``scripts/tools/pair_review.py``) surfaced six recurring reasons a
match *looks* wrong that the permissive soft-chamfer score (mean best-cosine — see
``visualize_match_errors.rank_embedding_rows``) is blind to:

    1. unmatched characteristic spots   a striking spot on one animal, absent on the other
    2. round / non-distinctive spots    a small round blob is weak evidence
    3. low match count                  1-3 corresponding spots is low confidence
    4. mismatches                       a spot whose nearest neighbour clearly disagrees
    5. positional differences           similar shape, very different body location
    6. shape differences                aligned location, substantially different shape

The common primitive underneath 1, 2 and 4 is a per-spot **distinctiveness** weight: how
much this spot should count. The reviewer described it as several things at once — "shape",
"rarity", "very long or very curved", "not many other spots around", "some spots just draw
the eye" — so it is built here as a transparent, tunable composite of five factors, each a
column you can inspect, rather than a single opaque number.

This module is deliberately self-contained (numpy + scipy only): it takes a ``spots`` frame
and normalized embedding arrays, so both the raw ranker (viz) and the learned aggregator
(models) can call it without an import cycle through ``embeddings``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree

# ------------------------------------------------------- the correspondence gate (ablatable)
# WHICH query spot is paired with WHICH candidate spot, decided before any weighting happens.
# Three lines of :func:`strict_pair_score` settle it and every score in this repo is computed
# downstream of them, so they are exposed here instead of left as literals — see
# ``scripts/experiments/run_gate.sh``. Defaults reproduce the shipped behaviour exactly.
#
#   GATE_ASSIGN=mutual     (default, as shipped) i and j must pick each other. Strictest: a spot
#                          whose partner prefers someone else contributes nothing at all.
#   GATE_ASSIGN=best       each query spot takes its argmax candidate, reciprocated or not. One
#                          candidate spot may then explain several query spots — the failure mode
#                          this module's residual docstring names — so coverage is clamped below.
#   GATE_ASSIGN=hungarian  optimal one-to-one assignment: "explaining a spot costs a spot". The fix
#                          that docstring proposes for the residual path and that the shipped
#                          score has never used.
#   GATE_MATCH_THR         cosine floor for a correspondence to count at all.
GATE_ASSIGN = os.environ.get("GATE_ASSIGN", "mutual")
GATE_MATCH_THR = float(os.environ.get("GATE_MATCH_THR", "0.4"))


def correspondences(S: np.ndarray, assign: str | None = None) -> list[tuple[int, int]]:
    """``[(query_spot, candidate_spot), ...]`` for a (nq, nc) cosine matrix under ``assign``.

    Pulled out of :func:`strict_pair_score` so the assignment rule is one testable function
    rather than three lines buried in a scorer.
    """
    assign = GATE_ASSIGN if assign is None else assign
    nq, nc = S.shape
    if assign == "hungarian":
        from scipy.optimize import linear_sum_assignment                 # noqa: PLC0415
        r, c = linear_sum_assignment(-S)
        return [(int(i), int(j)) for i, j in zip(r, c)]
    qbest = S.argmax(1)
    if assign == "best":
        return [(i, int(qbest[i])) for i in range(nq)]
    if assign != "mutual":
        raise ValueError(f"GATE_ASSIGN must be mutual|best|hungarian, got {assign!r}")
    cbest = S.argmax(0)
    return [(i, int(qbest[i])) for i in range(nq) if cbest[qbest[i]] == i]


# ----------------------------------------------------------------------------- shape

def _polygon_area(pts: np.ndarray) -> float:
    """Unsigned shoelace area of a closed polygon given its (k, 2) boundary points."""
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _perimeter(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    d = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def shape_descriptors(contour: np.ndarray) -> tuple[float, float, float]:
    """Three scale-invariant shape numbers for one spot's centroid-relative contour.

    Returns ``(elongation, noncircularity, irregularity)`` — each 0 for an ideal round blob,
    growing as the spot becomes more "shapy":

    elongation      PCA aspect ratio of the boundary point cloud minus 1 (a stick or a banana
                    is long; a circle is 0). Captures "very long".
    noncircularity  ``perimeter² / (4π·area) − 1`` — isoperimetric excess. 0 for a circle,
                    positive for anything elongated *or* wiggly. Captures "very curved".
    irregularity    ``1 − area / convex_hull_area`` — concavity. 0 for any convex shape,
                    positive for notched / lobed / star-like outlines. Captures "special".
    """
    if contour is None or len(contour) < 3:
        return 0.0, 0.0, 0.0
    pts = np.asarray(contour, float)

    # elongation: eigenvalues of the boundary-point covariance
    c = pts - pts.mean(0)
    cov = (c.T @ c) / len(c)
    ev = np.linalg.eigvalsh(cov)
    lo, hi = float(ev[0]), float(ev[-1])
    elongation = float(np.sqrt(hi / lo) - 1.0) if lo > 1e-9 else 0.0

    area = _polygon_area(pts)
    perim = _perimeter(pts)
    noncirc = float(perim * perim / (4.0 * np.pi * area) - 1.0) if area > 1e-6 else 0.0

    try:
        hull_area = float(ConvexHull(pts).volume)   # 'volume' is area in 2-D
        irregularity = float(1.0 - area / hull_area) if hull_area > 1e-6 else 0.0
    except Exception:
        irregularity = 0.0

    return elongation, max(noncirc, 0.0), max(min(irregularity, 1.0), 0.0)


# ----------------------------------------------------------------------------- combine

def _rank01(x: np.ndarray) -> np.ndarray:
    """Map values to their percentile in [0, 1] — robust to the heavy tails these factors have."""
    x = np.asarray(x, float)
    order = x.argsort()
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x))
    return ranks / max(len(x) - 1, 1)


# The blend below was tuned by reading the manual pair review. Three later measurements agreed that
# ``rarity`` — its second-heaviest term — is not merely useless but actively harmful, so it is now 0:
#
#   * fitting the human interesting-spot clicks gives rarity a standardized weight of **-0.070**,
#     i.e. nothing, while irregularity (+0.824) and size (+0.707) carry the whole score
#     (``models/distinctiveness.py``);
#   * a spot's chance of SURVIVING to another photo of the same animal falls with the hand
#     distinctiveness score (**-0.32** controlling for size, ``eval/feasibility.py``). A spot scores
#     as "rare" largely when the segmenter merged or split it, and such artefacts do not reappear —
#     so the weight was steering evidence toward the extractor's own mistakes;
#   * ``isolation`` fits at **-0.069**, equally nothing, and ``noncircularity`` fits **negative**
#     (-0.337): once a spot is known to be big and lobed, "curved" argues AGAINST it being special.
#
# rarity is set to 0 (which also skips its O(n log n) neighbour search). isolation and
# noncircularity are left at 1.0 deliberately — the evidence says they are worthless rather than
# harmful, and changing three things at once would make the next measurement unreadable. Use
# :data:`LEGACY_WEIGHTS` to reproduce any result recorded before this change.
DEFAULT_WEIGHTS = {
    "size": 2.0,            # big spots are the identifying ones (the "96/100" spots in review)
    "elongation": 1.0,      # "very long"
    "noncircularity": 1.0,  # "very curved" -- fits negative; a candidate for removal next
    "irregularity": 1.0,    # weird / lobed outline -- the strongest supported factor
    "rarity": 0.0,          # WAS 1.5. See above: ~zero predictive value, anti-correlated with
                            # spot survival. Kept as a key so the factor stays inspectable.
    "isolation": 1.0,       # "not many other spots around" -- fits ~zero; candidate for removal
}

# The pre-2026-08 blend, for reproducing earlier numbers and for the A/B that justifies the change.
LEGACY_WEIGHTS = {**DEFAULT_WEIGHTS, "rarity": 1.5}


def _size_pct(spots: pd.DataFrame) -> np.ndarray:
    """Scale-invariant size percentile (0..1) — the SAME metric the review size-maps label spots by
    (equivalent-circle diameter / body length, ranked dataset-wide), so 'big spot' means here what
    the '96'/'100' labels meant to the reviewer. Reuses ``embeddings._size_percentile`` when the
    body-length column is present; otherwise ranks ``area_pixels`` directly."""
    try:
        from embeddings import _size_percentile          # bare-name import (viz path) or...
    except ImportError:
        try:
            from pipeline.spot_transformer.core.embeddings import _size_percentile
        except ImportError:
            _size_percentile = None
    if _size_percentile is not None and "length_px" in spots.columns:
        p = np.asarray(_size_percentile(spots), float)
        return _rank01(np.nan_to_num(p, nan=np.nanmin(p) if np.isfinite(p).any() else 0.0))
    return _rank01(spots.get("area_pixels", pd.Series(np.zeros(len(spots)))).to_numpy(float))


def distinctiveness(
    spots: pd.DataFrame,
    emb: np.ndarray,
    *,
    weights: dict[str, float] | None = None,
    k_rarity: int = 10,
    k_isolation: int = 3,
    attach: bool = False,
) -> np.ndarray:
    """Per-spot distinctiveness in [0, 1] — a tunable blend of five interpretable factors.

    ``spots`` must carry ``local_contour``, ``salamander_id``, ``axis_t`` and ``axis_offset`` /
    ``avg_width_px`` (or ``length_px``) for the body-normalized isolation coordinate. ``emb`` is
    the (n, d) L2-normalized per-spot embedding, row-aligned to ``spots`` (the population that
    ``rarity`` is measured against).

    Each factor is percentile-normalized to [0, 1], combined by ``weights`` (see
    ``DEFAULT_WEIGHTS``), and the weighted mean is returned — so the output is already in [0, 1]
    and every factor contributes on the same scale regardless of its raw units. With
    ``attach=True`` the five factor columns and ``distinctiveness`` are written back onto a copy
    of ``spots`` and that frame is returned instead (for inspection / debugging).
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    n = len(spots)
    contours = spots["local_contour"].to_numpy()
    desc = np.array([shape_descriptors(c) for c in contours])       # (n, 3)
    elong, noncirc, irreg = desc[:, 0], desc[:, 1], desc[:, 2]

    # rarity: mean cosine distance to the k nearest OTHER spots in embedding space (population
    # isolation). Rare shape+position -> far from everything -> high. Excludes self (col 0).
    # Skipped entirely at weight 0 (the default since the factor measured worthless): this is the
    # only term needing a neighbour search over the whole population, so skipping it is also the
    # difference between seconds and minutes on a 36k-spot dataset.
    if w.get("rarity", 0.0) == 0.0:
        rarity = np.zeros(n)
    elif n > k_rarity + 1:
        tree = cKDTree(emb)
        dist, _ = tree.query(emb, k=k_rarity + 1)                   # euclidean on unit vectors
        rarity = dist[:, 1:].mean(1)
    else:
        rarity = np.zeros(n)

    # spatial isolation: distance to the k nearest spots on the SAME animal, in body-normalized
    # (axis_t, lateral-offset) coords. A spot with empty surroundings "draws the eye".
    t = np.nan_to_num(spots["axis_t"].to_numpy(float), nan=0.5)
    half_w = spots["avg_width_px"].to_numpy(float) / 2.0 if "avg_width_px" in spots else None
    if half_w is None:
        half_w = np.full(n, np.nanmedian(spots.get("length_px", pd.Series(np.ones(n)))) / 6.0)
    off = spots["axis_offset"].to_numpy(float) if "axis_offset" in spots else np.zeros(n)
    u = np.divide(off, half_w, out=np.zeros(n), where=half_w > 0)
    u = np.clip(np.nan_to_num(u), -3.0, 3.0)
    coords = np.column_stack([t, u])
    isolation = np.zeros(n)
    sal = spots["salamander_id"].to_numpy()
    for s in np.unique(sal):
        idx = np.flatnonzero(sal == s)
        if len(idx) <= 1:
            isolation[idx] = 1.0                                     # a lone spot is maximally isolated
            continue
        kk = min(k_isolation, len(idx) - 1)
        d, _ = cKDTree(coords[idx]).query(coords[idx], k=kk + 1)
        isolation[idx] = d[:, 1:].mean(1)

    factors = {
        "size": _size_pct(spots),
        "elongation": _rank01(elong),
        "noncircularity": _rank01(noncirc),
        "irregularity": _rank01(irreg),
        # A skipped rarity is CONSTANT, never rank-normalized: `_rank01` of an all-zero array
        # returns an arbitrary 0..1 ramp in row order, which would look like a real feature to any
        # model fitted on the attached columns while carrying nothing but the sort order.
        "rarity": (np.full(n, 0.5) if w.get("rarity", 0.0) == 0.0 else _rank01(rarity)),
        "isolation": _rank01(isolation),
    }
    wsum = sum(w[k] for k in factors) or 1.0
    score = sum(w[k] * factors[k] for k in factors) / wsum

    if attach:
        out = spots.copy()
        for k, v in factors.items():
            out[k] = v
        out["distinctiveness"] = score
        return out
    return score


# ----------------------------------------------------------------------------- strict score

def strict_pair_score(
    eq: np.ndarray,
    ec: np.ndarray,
    wq: np.ndarray,
    wc: np.ndarray,
    *,
    xyq: np.ndarray | None = None,
    xyc: np.ndarray | None = None,
    match_thr: float | None = None,
    sigma_pos: float = 0.5,
    tau_support: float = 2.5,
    return_parts: bool = False,
    assign: str | None = None,
):
    """A stricter replacement for the soft-chamfer mean, in ``[0, 1]``.

    ``strict = coverage * support``:

    coverage  the fraction of *distinctiveness mass* that good matches explain. Each spot owns a
              mass ``w`` (its distinctiveness); a spot is explained in proportion to the quality of
              its mutual-nearest-neighbour match (clipped cosine × a position-agreement gate). An
              unmatched **characteristic** spot (high ``w``, quality 0) drags coverage down hard; an
              unmatched round spot (``w≈0``) barely moves it; a spot matched to the wrong body
              location gets a low position gate, so it is explained little though it is not
              unmatched — i.e. mismatches *lower* the score rather than merely not helping.

    support   ``1 − exp(−n_good / tau_support)`` — a confidence factor on the *number* of quality
              matches, so 1-3 corresponding spots score far below a densely corroborated match.

    ``eq``/``ec`` are the (nq,d)/(nc,d) L2-normalized spot embeddings; ``wq``/``wc`` their
    distinctiveness weights (see :func:`distinctiveness`). ``xyq``/``xyc`` are body-normalized
    (axis_t, lateral-offset) coords; when given, a Gaussian position gate (width ``sigma_pos`` in
    body-length units) down-weights appearance matches that sit in different body locations. With
    ``return_parts`` a diagnostics dict is returned alongside the score.
    """
    nq, nc = len(eq), len(ec)
    if nq == 0 or nc == 0:
        return (0.0, {}) if return_parts else 0.0
    match_thr = GATE_MATCH_THR if match_thr is None else match_thr

    S = eq @ ec.T                                                   # (nq, nc) cosine in [-1, 1]
    mutual = correspondences(S, assign)         # the gate: mutual-NN (shipped) / best / hungarian

    total_mass = float(wq.sum() + wc.sum()) + 1e-9
    explained, n_good = 0.0, 0
    matched_q, matched_c = [], []
    for i, j in mutual:
        s = float(S[i, j])
        if s < match_thr:
            continue
        gate = 1.0
        if xyq is not None and xyc is not None:
            d2 = float(np.sum((xyq[i] - xyc[j]) ** 2))
            gate = float(np.exp(-d2 / (2.0 * sigma_pos * sigma_pos)))
        quality = max(s, 0.0) * gate                              # match quality in [0, 1]
        explained += quality * (float(wq[i]) + float(wc[j]))
        matched_q.append(i); matched_c.append(j)
        if quality >= 0.3:
            n_good += 1

    # Clamped because `assign="best"` lets one candidate spot explain several query spots, so the
    # same wc[j] can be banked more than once and explained may exceed the mass that exists. Under
    # the shipped mutual/hungarian rules each index appears at most once and the clamp never fires.
    coverage = min(explained / total_mass, 1.0)                   # in [0, 1]
    support = 1.0 - np.exp(-n_good / tau_support)
    score = float(coverage * support)

    if return_parts:
        unexplained_q = float(sum(wq[i] for i in range(nq) if i not in set(matched_q)))
        unexplained_c = float(sum(wc[j] for j in range(nc) if j not in set(matched_c)))
        parts = {
            "score": score, "coverage": coverage, "support": float(support),
            "n_matched": len(matched_q), "n_good": n_good, "nq": nq, "nc": nc,
            "explained_mass": explained, "total_mass": total_mass,
            "unexplained_q_mass": unexplained_q, "unexplained_c_mass": unexplained_c,
        }
        return score, parts
    return score


# ------------------------------------------------------------------- residual (contradiction)
"""Why a second, stricter score.

``strict_pair_score`` above only ever *withholds credit*: an unmatched spot fails to add to
``explained``, so one striking blotch with no counterpart moves a 25-spot pair's coverage by ~4%.
Manual review says that is not how a person decides — a big spot on a body region the other photo
clearly shows, with nothing resembling it there, is **evidence against**, and on its own settles
the pair. Three things are needed to say that arithmetically:

1. **negative, not merely missing, evidence** — an explicit contradiction mass ``C`` that trades
   off against the explained mass ``E`` (``E / (E + lambda*C)``), so unmatched pattern actively
   pushes the score down instead of diluting slowly.
2. **observability** — occlusion must not be punished. A spot only contradicts if the other photo
   actually *shows* that stretch of body (:func:`observability`); a tail spot on a photo cropped at
   the hips is unexplained but not contradictory.
3. **one-to-one correspondence** — with best-match (``argmax``) scoring one candidate spot can
   "explain" five query spots at once, which is exactly how a non-match keeps a high score. A
   Hungarian assignment makes explaining a spot cost a spot, so leftovers are real leftovers.

The parts are returned raw (:func:`residual_parts`) and combined separately
(:func:`combine_residual`), because ``lambda`` (penalty strength) and ``gamma`` (single-worst-spot
veto) can then be swept over a whole dataset without re-running any matching.
"""

RESIDUAL_PART_NAMES = [
    "expl_q", "expl_c",            # Σ w·quality — salience mass with a counterpart, each side
    "res_q", "res_c",              # Σ w·(1-quality)·observability — contradiction mass, each side
    "mass_q", "mass_c",            # Σ w — total salience mass, each side
    "worst_res_q", "worst_res_c",  # the single most conspicuous contradicted spot, each side
    "wmax",                        # max spot weight in the pair (the veto's reference scale)
    "n_good", "n_assigned", "nq", "nc",
    "max_sim", "geom", "ransac",
    "obs_q", "obs_c",              # mean observability — how much of each pattern was co-observed
]


def axial_span(t: np.ndarray, *, lo_pct: float = 2.0, hi_pct: float = 98.0
               ) -> tuple[float, float]:
    """The head..tail stretch a photo actually shows, as ``(t_lo, t_hi)`` in axis units.

    Estimated from the percentiles of its own spots' ``axis_t`` rather than the min/max, so one
    stray extraction does not claim the whole body was visible. Fewer than three usable spots ->
    ``(0, 1)``, i.e. "assume everything was visible", which is the conservative choice: it never
    invents an excuse for an unmatched spot.
    """
    t = np.asarray(t, float)
    finite = t[np.isfinite(t)]
    if len(finite) < 3:
        return 0.0, 1.0
    return float(np.percentile(finite, lo_pct)), float(np.percentile(finite, hi_pct))


def observability(t: np.ndarray, span: tuple[float, float], *, margin: float = 0.08) -> np.ndarray:
    """In [0, 1] per spot: how much the OTHER photo can be trusted to have shown this body position.

    1 inside the other photo's ``span``, ramping to 0 over ``margin`` outside it. This is the gate
    that separates "absent" from "not photographed": only observable spots are allowed to
    contradict, so a cropped tail costs nothing while a bare mid-back costs everything. Spots with
    no axial coordinate get 0 — unknown position, no accusation.
    """
    t = np.asarray(t, float)
    lo, hi = span
    m = max(margin, 1e-6)
    o = np.minimum(np.clip((t - lo) / m + 1.0, 0.0, 1.0),
                   np.clip((hi - t) / m + 1.0, 0.0, 1.0))
    return np.where(np.isfinite(t), o, 0.0)


def assign_one_to_one(G: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian max-weight matching on the gated similarity ``G``, keeping pairs ``>= thr``.

    One-to-one is the point: under best-match scoring a single candidate spot can be the answer for
    every query spot, so a pair of unrelated animals whose blobs are all "roundish mid-body" scores
    as well as a true pair. Here explaining a spot consumes a spot, and whatever is left over is
    genuinely unaccounted for. ``linear_sum_assignment`` on the (nq x nc) matrices these images
    produce (tens of spots) costs microseconds.
    """
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(-G)
    keep = G[ri, ci] >= thr
    return ri[keep], ci[keep]


def residual_parts(
    eq: np.ndarray,
    ec: np.ndarray,
    wq: np.ndarray,
    wc: np.ndarray,
    *,
    xyq: np.ndarray | None = None,
    xyc: np.ndarray | None = None,
    match_thr: float = 0.4,
    good_thr: float = 0.5,
    sigma_pos: float = 0.12,
    margin: float = 0.08,
    geom_xyq: np.ndarray | None = None,
    geom_xyc: np.ndarray | None = None,
) -> np.ndarray:
    """The raw evidence bookkeeping for one photo-vs-photo pair, as ``RESIDUAL_PART_NAMES``.

    Correspondences are a one-to-one assignment on ``quality = relu(cosine) * position-gate``; each
    spot's ``quality`` is its assigned partner's (0 if unassigned). From there every spot on BOTH
    animals lands in exactly one of three buckets, weighted by its distinctiveness ``w``:

        explained     w·quality                      — pattern the two animals share
        contradicted  w·(1-quality)·observability    — pattern one has and the other visibly lacks
        unobserved    the remainder                  — outside the other photo's field, ignored

    ``xyq``/``xyc`` are body-frame ``(axis_t, lateral/length)`` coords: they drive both the position
    gate and observability, and without them every unmatched spot is treated as fully observed.
    ``geom_xy*`` are image-px centroids for the constellation checks (optional).
    """
    det = residual_detail(eq, ec, wq, wc, xyq=xyq, xyc=xyc, match_thr=match_thr,
                          good_thr=good_thr, sigma_pos=sigma_pos, margin=margin,
                          geom_xyq=geom_xyq, geom_xyc=geom_xyc)
    if det is None:
        return np.zeros(len(RESIDUAL_PART_NAMES))
    return np.array([
        float((det["wq"] * det["qual_q"]).sum()), float((det["wc"] * det["qual_c"]).sum()),
        float(det["res_q"].sum()), float(det["res_c"].sum()),
        float(det["wq"].sum()), float(det["wc"].sum()),
        float(det["res_q"].max()), float(det["res_c"].max()),
        float(max(det["wq"].max(), det["wc"].max())),
        float(det["n_good"]), float(len(det["assign_q"])), float(len(eq)), float(len(ec)),
        float(det["max_sim"]), det["geom"], det["ransac"],
        float(det["obs_q"].mean()), float(det["obs_c"].mean()),
    ], float)


def residual_detail(
    eq: np.ndarray,
    ec: np.ndarray,
    wq: np.ndarray,
    wc: np.ndarray,
    *,
    xyq: np.ndarray | None = None,
    xyc: np.ndarray | None = None,
    match_thr: float = 0.4,
    good_thr: float = 0.5,
    sigma_pos: float = 0.12,
    margin: float = 0.08,
    geom_xyq: np.ndarray | None = None,
    geom_xyc: np.ndarray | None = None,
) -> dict | None:
    """Per-SPOT version of :func:`residual_parts` — the same arithmetic, un-summed.

    Returns ``None`` for an empty side, else a dict of row-aligned arrays: ``qual_*`` (each spot's
    assigned match quality, 0 = nothing), ``obs_*`` (was this position visible on the other animal),
    ``res_*`` (its contradiction mass), ``assign_q``/``assign_c`` (the surviving assignment, as
    index arrays into each side), plus the scalars the aggregate needs. This is what a reviewer
    wants drawn: ``res_q`` names, spot by spot, exactly which pattern the score charged for.
    """
    nq, nc = len(eq), len(ec)
    if nq == 0 or nc == 0:
        return None

    S = eq @ ec.T
    G = np.clip(S, 0.0, None)
    if xyq is not None and xyc is not None:
        d2 = ((np.asarray(xyq, float)[:, None, :] - np.asarray(xyc, float)[None, :, :]) ** 2).sum(-1)
        G = G * np.exp(-d2 / (2.0 * sigma_pos * sigma_pos))

    ri, ci = assign_one_to_one(G, match_thr)
    qual_q = np.zeros(nq); qual_q[ri] = G[ri, ci]
    qual_c = np.zeros(nc); qual_c[ci] = G[ri, ci]

    if xyq is not None and xyc is not None:
        tq, tc = np.asarray(xyq, float)[:, 0], np.asarray(xyc, float)[:, 0]
        obs_q = observability(tq, axial_span(tc), margin=margin)
        obs_c = observability(tc, axial_span(tq), margin=margin)
    else:
        obs_q, obs_c = np.ones(nq), np.ones(nc)

    wq = np.asarray(wq, float); wc = np.asarray(wc, float)

    geom = ransac = 0.0
    if geom_xyq is not None and geom_xyc is not None and len(ri) >= 3:
        from scipy.spatial.distance import pdist
        from scipy.stats import spearmanr
        qp, cp = np.asarray(geom_xyq, float)[ri], np.asarray(geom_xyc, float)[ci]
        dq, dc = pdist(qp), pdist(cp)
        if dq.std() > 1e-9 and dc.std() > 1e-9:
            r = spearmanr(dq, dc).correlation
            geom = 0.0 if np.isnan(r) else float(r)
        try:
            from aggregator import _similarity_inliers                        # noqa: PLC0415
        except ImportError:
            from pipeline.spot_transformer.models.aggregator import _similarity_inliers  # noqa: PLC0415
        ransac = float(_similarity_inliers(qp, cp))

    return {
        "wq": wq, "wc": wc,
        "qual_q": qual_q, "qual_c": qual_c,
        "obs_q": obs_q, "obs_c": obs_c,
        "res_q": wq * (1.0 - qual_q) * obs_q,
        "res_c": wc * (1.0 - qual_c) * obs_c,
        "assign_q": ri, "assign_c": ci, "assign_quality": G[ri, ci],
        "n_good": int((G[ri, ci] >= good_thr).sum()),
        "max_sim": float(S.max()), "geom": geom, "ransac": ransac,
    }


def combine_residual(P: np.ndarray, *, lam: float = 1.0, gamma: float = 0.5,
                     tau_support: float = 2.5, w_ref: float = 1.0) -> np.ndarray:
    """``RESIDUAL_PART_NAMES`` rows -> a score in [0, 1]. Vectorized over rows.

        evidence = E / (E + lambda*C)      explained mass vs contradiction mass
        support  = 1 - exp(-n_good/tau)    few corroborating matches -> low
        veto     = 1 - gamma * (worst single contradicted spot / w_ref)

    ``lam`` is how much a unit of contradicted pattern costs relative to a unit of shared pattern:
    0 reproduces a penalty-free coverage score, 1 makes them equal, >1 makes disagreement dominate.
    ``gamma`` is the separate "one look is enough" term — a single conspicuous spot that the other
    animal visibly lacks scales the whole score down, however much else lines up, which is the part
    a mass average can never express.

    ``w_ref`` is what "conspicuous" is measured against, and it must be a POPULATION scale (e.g. the
    95th percentile weight over the dataset), not anything derived from the pair: divide by the
    pair's own heaviest spot and every pair contains a full-strength veto candidate by construction,
    which fires the veto on animals whose spots are all unremarkable. 1.0 reads the weights as the
    absolute [0, 1] scores they already are.
    """
    P = np.atleast_2d(np.asarray(P, float))
    idx = {n: i for i, n in enumerate(RESIDUAL_PART_NAMES)}
    E = P[:, idx["expl_q"]] + P[:, idx["expl_c"]]
    C = P[:, idx["res_q"]] + P[:, idx["res_c"]]
    evidence = E / (E + lam * C + 1e-9)
    support = 1.0 - np.exp(-P[:, idx["n_good"]] / tau_support)
    veto = np.maximum(P[:, idx["worst_res_q"]], P[:, idx["worst_res_c"]]) / max(w_ref, 1e-9)
    return evidence * support * (1.0 - gamma * np.clip(veto, 0.0, 1.0))


def residual_pair_score(eq, ec, wq, wc, *, lam: float = 1.0, gamma: float = 0.5, **kw) -> float:
    """One-call convenience: :func:`residual_parts` -> :func:`combine_residual`."""
    tau = kw.pop("tau_support", 2.5)
    w_ref = kw.pop("w_ref", 1.0)
    return float(combine_residual(residual_parts(eq, ec, wq, wc, **kw),
                                  lam=lam, gamma=gamma, tau_support=tau, w_ref=w_ref)[0])
