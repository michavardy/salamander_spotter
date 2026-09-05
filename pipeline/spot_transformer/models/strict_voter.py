"""Distinctiveness-weighted, **strict** matching + a novelty signal.

The champion aggregator (``aggregator.py``) weighs every spot equally: its features are means and
counts over ``qmax`` (each query spot's best cosine to the candidate). The manual review says that
is wrong — a match on a big, oddly-shaped spot is strong evidence; a match on a small round blob is
almost none; and a *distinctive* spot present on one animal but absent on the other is evidence
*against* the match. This module rebuilds the match summary so each spot contributes in proportion
to its learned **distinctiveness** (``distinctiveness.py``, trained on the human interesting-spot
clicks), and adds the two things equal-weight voting cannot express:

* **strictness** — ``support = 1 - exp(-n_good / tau)`` collapses the score when only 1-3 spots
  corroborate (the review's "only 2/3 lines should be penalized").
* **novelty** — ``unexplained_q`` = the distinctiveness mass of query spots that matched *nothing*.
  A query whose striking spots go unexplained by the best candidate is probably a NEW animal; this
  is the per-query signal the census threshold needs and the equal-weight features never had.

Two consumers:

* :func:`strict_features` -> a fixed-order vector for a logistic regression (learned combination),
  the direct analogue of ``aggregator.match_features`` but distinctiveness-weighted.
* :func:`strict_hand_score` -> the training-free ``coverage x support`` thesis
  (``strict_match.strict_pair_score``), so we can see how far the intuition gets with no fitting.

Weights come in as a ``(sid, spot_id) -> distinctiveness`` lookup so a candidate individual, whose
spots are concatenated from several gallery photos, keeps a correct per-spot weight (the plain
aggregator's ``concatenate`` throws the identity of each spot away).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval"):
    p = str(_ST / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import strict_match as sm                                    # noqa: E402
from aggregator import _norm, _similarity_inliers            # noqa: E402

# --- geometry x quality interaction -------------------------------------------------------------
# ``eval/feasibility.py`` Part C measured how much the CONSTELLATION evidence can be trusted as a
# function of how good the two photos are. Splitting photo pairs at the median of
# ``min(overall_quality)``, the geometric-consistency feature separates same-from-different at
# **0.603** on the good half and **0.500 -- exactly chance -- on the bad half** (gap +0.103). The
# same test on curl (+0.026) and camera angle (+0.005) found nothing, so those are deliberately NOT
# interacted: a product term there would be a parameter fitted to noise.
#
# This has to be a PRODUCT. Adding quality as another column teaches a linear model "low-quality
# pairs match less often" -- a main effect on the score. What the measurement actually says is
# "geometry means less when quality is low", which is a statement about how much to weight OTHER
# evidence, and no additive combination of the two columns can express it. The main effects are
# included alongside so the interaction's coefficient is interpretable rather than absorbing them.
#
# **OFF by default, because it was measured and it does nothing yet.** The A/B
# (``scripts/experiments/run_quality_interaction.sh``, 2 folds) moved strict_logreg census F0.5 by
# **-0.005** against a fold spread of +/-0.07, with strict_hand_pos identical in both arms (so the
# test was clean). The model DID learn the predicted pattern -- ``geom_consistency`` went -0.112 ->
# -0.222 while ``geom_x_quality`` came in at **+0.163**, i.e. "geometry counts against, less so when
# the photos are good". It changes nothing because geometry carries a NEGATIVE weight in both arms:
# the aggregator has already decided the current geometric feature is not evidence, and scaling how
# much to trust something already distrusted is a no-op.
#
# The dependency runs the other way round from how it was planned: this interaction is worth
# switching on only once the constellation features are strong enough to earn a positive weight
# (triplet/angle invariants, spine-relative coordinates). The plumbing is kept ready for that.
#   QUALITY_INTERACTION=1  re-enables it.
USE_QUALITY_INTERACTION = os.environ.get("QUALITY_INTERACTION", "0") == "1"
_QI_FEATURES = ["quality", "geom_x_quality", "ransac_x_quality"]

# thresholds mirror aggregator.FEATURE_NAMES so the two feature sets are comparable
STRICT_FEATURE_NAMES = [
    "w_softchamfer_mean",   # Σ w·best / Σ w         (distinctiveness-weighted soft-chamfer)
    "w_softchamfer_sum",    # Σ w·best               (absolute, size-sensitive)
    "max_sim",              # single best match
    "wcov_q>=.9", "wcov_q>=.8", "wcov_q>=.7",   # weighted fraction of query distinctiveness explained
    "wcov_c>=.8",           # symmetric: weighted fraction of candidate distinctiveness explained
    "unexplained_q",        # Σ w·[best<.6] / Σ w    (distinctive query spots matching nothing → novelty)
    "unexplained_c",        # symmetric
    "worst_unmatched_q",    # the single most-distinctive unmatched query spot ("spot 96 absent")
    "support",              # 1 - exp(-n_good / tau) (few corroborating matches → low)
    "mutual_count", "mutual_frac",
    "log_nq", "log_nc",
    "geom_consistency", "ransac_frac",
] + (_QI_FEATURES if USE_QUALITY_INTERACTION else [])

_TAU_GOOD = 0.75          # a "good" match
_TAU_LOW = 0.6            # below this a distinctive spot counts as unexplained
_SUPPORT_TAU = 2.5


def strict_features(qe, ce, wq, wc, q_xy=None, c_xy=None, quality: float = 1.0) -> np.ndarray:
    """Distinctiveness-weighted match summary between a query image and a candidate individual.

    ``qe`` (nq, d) / ``ce`` (nc, d) are L2-normalized spot embeddings; ``wq`` (nq) / ``wc`` (nc)
    their distinctiveness in [0, 1]; ``q_xy`` / ``c_xy`` image-px centroids for the constellation
    geometry. Returns the ``STRICT_FEATURE_NAMES`` vector.

    ``quality`` in [0, 1] is how trustworthy this comparison's geometry is — see
    :data:`USE_QUALITY_INTERACTION`. It enters as a main effect plus its products with the two
    geometric features, so the model can learn to discount the constellation evidence on poor
    photos instead of trusting it equally everywhere.
    """
    S = qe @ ce.T                                            # (nq, nc) cosine
    nq, nc = S.shape
    qmax = S.max(1); qbest = S.argmax(1)
    cmax = S.max(0); cbest = S.argmax(0)
    mutual = np.array([i for i in range(nq) if cbest[qbest[i]] == i], dtype=int)
    n_mut = len(mutual)

    Wq = float(wq.sum()) + 1e-9
    Wc = float(wc.sum()) + 1e-9
    n_good = int((qmax[mutual] >= _TAU_GOOD).sum()) if n_mut else 0
    relu = np.clip(qmax, 0.0, None)

    geom = ransac = 0.0
    if q_xy is not None and n_mut >= 3:
        qp = np.asarray(q_xy)[mutual]; cp = np.asarray(c_xy)[qbest[mutual]]
        from scipy.spatial.distance import pdist
        dq, dc = pdist(qp), pdist(cp)
        if dq.std() > 1e-9 and dc.std() > 1e-9:
            r = spearmanr(dq, dc).correlation
            geom = 0.0 if np.isnan(r) else float(r)
        ransac = _similarity_inliers(qp, cp)

    feats = [
        float((wq * relu).sum() / Wq),
        float((wq * relu).sum()),
        float(qmax.max()),
        float((wq * (qmax >= 0.9)).sum() / Wq),
        float((wq * (qmax >= 0.8)).sum() / Wq),
        float((wq * (qmax >= 0.7)).sum() / Wq),
        float((wc * (cmax >= 0.8)).sum() / Wc),
        float((wq * (qmax < _TAU_LOW)).sum() / Wq),
        float((wc * (cmax < _TAU_LOW)).sum() / Wc),
        float((wq * (1.0 - relu)).max() / (wq.max() + 1e-9)),
        float(1.0 - np.exp(-n_good / _SUPPORT_TAU)),
        float(n_mut), float(n_mut / nq),
        float(np.log1p(nq)), float(np.log1p(nc)),
        geom, ransac,
    ]
    if USE_QUALITY_INTERACTION:
        q = float(np.clip(quality, 0.0, 1.0))
        feats += [q, geom * q, ransac * q]
    return np.array(feats, float)


def strict_hand_score(qe, ce, wq, wc, xyq=None, xyc=None, sigma_pos=0.12) -> float:
    """Training-free ``coverage x support`` from :func:`strict_match.strict_pair_score` — the
    thesis with no fitting.

    With ``xyq`` / ``xyc`` (body-frame ``(axis_t, axis_offset/length)`` coords) a Gaussian position
    gate of width ``sigma_pos`` down-weights appearance matches that sit in *different* body
    locations — the review's "penalize because of the difference in location". ``None`` disables it.
    """
    return float(sm.strict_pair_score(qe, ce, wq, wc, xyq=xyq, xyc=xyc, sigma_pos=sigma_pos))


# --------------------------------------------------------------------- residual (per photo)
"""Photo-vs-photo residual scoring.

Everything above compares a query photo with a candidate INDIVIDUAL, whose spots are the
concatenation of all its gallery photos. That is harmless while the score only counts matches, but
it breaks the moment unmatched pattern is penalized: a candidate with three photos carries three
copies of its spots, so most of its mass can never be matched by one query and the animals we know
best would score worst. So the residual score is computed **per photo pair** and the candidate's
photos are combined by ``max`` — "does any single photo of this animal agree with the query",
which is also the question a person answers when they flip through a folder.

``residual_pairs`` therefore returns one row per (query photo, candidate photo) with the raw parts;
:func:`fold_to_candidates` collapses those to one score per (query, individual) afterwards, so a
whole (lambda, gamma) grid can be swept over the cached parts without rematching anything.
"""

RESIDUAL_FEATURE_NAMES = [
    "evidence",                 # E / (E + C)              — the penalty score at lambda=1
    "cov_q", "cov_c",           # explained / total mass, each side
    "res_frac_q", "res_frac_c", # contradicted / total mass, each side
    "veto",                     # worst single contradicted spot, against the population scale
                                # `w_ref` ("that big one just isn't there")
    "support",                  # 1 - exp(-n_good / tau)
    "n_good", "assigned_frac",
    "log_nq", "log_nc",
    "max_sim", "geom", "ransac",
    "obs_q", "obs_c",           # how much of each pattern the other photo even showed
]


def residual_features(P: np.ndarray, *, w_ref: float = 1.0) -> np.ndarray:
    """``strict_match.RESIDUAL_PART_NAMES`` rows -> ``RESIDUAL_FEATURE_NAMES`` rows.

    Same bookkeeping as :func:`strict_match.combine_residual`, but left as separate normalized
    columns so a logistic regression can fit the trade-off (how much a contradiction costs, whether
    the veto is worth its weight) instead of us fixing ``lambda`` and ``gamma`` by hand. ``w_ref``
    is the population weight scale the veto column is measured against — see ``combine_residual``.
    """
    P = np.atleast_2d(np.asarray(P, float))
    i = {n: k for k, n in enumerate(sm.RESIDUAL_PART_NAMES)}
    E = P[:, i["expl_q"]] + P[:, i["expl_c"]]
    C = P[:, i["res_q"]] + P[:, i["res_c"]]
    mq = P[:, i["mass_q"]] + 1e-9
    mc = P[:, i["mass_c"]] + 1e-9
    return np.column_stack([
        E / (E + C + 1e-9),
        P[:, i["expl_q"]] / mq, P[:, i["expl_c"]] / mc,
        P[:, i["res_q"]] / mq, P[:, i["res_c"]] / mc,
        np.clip(np.maximum(P[:, i["worst_res_q"]], P[:, i["worst_res_c"]]) / max(w_ref, 1e-9),
                0.0, 1.0),
        1.0 - np.exp(-P[:, i["n_good"]] / _SUPPORT_TAU),
        P[:, i["n_good"]], P[:, i["n_assigned"]] / (P[:, i["nq"]] + 1e-9),
        np.log1p(P[:, i["nq"]]), np.log1p(P[:, i["nc"]]),
        P[:, i["max_sim"]], P[:, i["geom"]], P[:, i["ransac"]],
        P[:, i["obs_q"]], P[:, i["obs_c"]],
    ])


def residual_pairs(sets, images, weight_lookup, pos_lookup, *, gallery_images=None,
                   neg_per_query=None, seed=0, match_thr=0.4, sigma_pos=0.12, margin=0.08):
    """One row per (query photo, candidate photo): the raw residual parts + who they belong to.

    Returns ``(P, y, qids, clab, gids)`` where ``P`` is ``(n, len(RESIDUAL_PART_NAMES))``, ``y`` is
    same/different, ``qids``/``gids`` index ``sets`` and ``clab`` is the candidate's individual.
    Candidates come from ``images + gallery_images`` (distractors), queries only from ``images`` —
    matching :func:`aggregator.iter_pairs`, so the two feature sets are measured on the same pairs.
    """
    rng = np.random.default_rng(seed)
    pool = list(dict.fromkeys(list(images) + list(gallery_images or [])))
    arr = precompute(sets, weight_lookup, pool)
    # no positions -> no position gate and no observability gate: every unmatched spot then counts
    # as fully contradicting, which is the strict-but-blind version of the score.
    xy = body_coords(sets, pool, pos_lookup) if pos_lookup is not None else {i: None for i in pool}
    by_label: dict[str, list[int]] = {}
    for i in pool:
        by_label.setdefault(sets[i].label, []).append(i)
    all_labels = list(by_label)

    P, y, qids, clab, gids = [], [], [], [], []
    for q in images:
        yq = sets[q].label
        if not any(g != q for g in by_label[yq]):
            continue
        cand = [c for c in all_labels if any(g != q for g in by_label[c])]
        if neg_per_query is not None:
            negs = [c for c in cand if c != yq]
            rng.shuffle(negs)
            cand = [yq] + negs[:neg_per_query]
        qe, wq, q_cent = arr[q]
        for c in cand:
            for g in by_label[c]:
                if g == q:
                    continue
                ce, wc, c_cent = arr[g]
                P.append(sm.residual_parts(qe, ce, wq, wc, xyq=xy[q], xyc=xy[g],
                                           match_thr=match_thr, sigma_pos=sigma_pos, margin=margin,
                                           geom_xyq=q_cent, geom_xyc=c_cent))
                y.append(int(c == yq)); qids.append(q); clab.append(c); gids.append(g)
    return (np.array(P), np.array(y, float), np.array(qids),
            np.array(clab, dtype=object), np.array(gids))


def fold_to_candidates(scores, qids, clab, gids=None):
    """Photo-pair scores -> one row per (query, candidate individual), keeping the best photo.

    Returns ``(score, qids, clab, gids)`` deduplicated on (query, candidate); ``gids`` names the
    winning gallery photo, which is what the pair-review app renders.
    """
    scores = np.asarray(scores, float)
    best: dict[tuple, int] = {}
    for k, (q, c) in enumerate(zip(qids, clab)):
        prev = best.get((q, c))
        if prev is None or scores[k] > scores[prev]:
            best[(q, c)] = k
    keep = np.array(sorted(best.values()))
    return (scores[keep], np.asarray(qids)[keep], np.asarray(clab, dtype=object)[keep],
            (np.asarray(gids)[keep] if gids is not None else None))


def body_coords(sets, images, pos_lookup):
    """``i -> (N, 2)`` body-frame coords ``(axis_t, axis_offset/length)`` per spot; NaN -> (0.5, 0)
    so a spot with no fitted axis neither matches nor blocks on position."""
    out = {}
    for i in images:
        s = sets[i]
        xy = np.array([pos_lookup.get((s.sid, int(sid)), (np.nan, np.nan)) for sid in s.spot_ids],
                      float)
        xy[~np.isfinite(xy[:, 0]), 0] = 0.5
        xy[~np.isfinite(xy[:, 1]), 1] = 0.0
        out[i] = xy
    return out


# ----------------------------------------------------------------------------- quality
def load_quality(db_path=None) -> dict[str, float]:
    """``salamander_id -> overall_quality`` in [0, 1]. Missing rows are absent, not defaulted."""
    import duckdb
    import data as _d                                                    # noqa: PLC0415
    con = duckdb.connect(str(db_path or _d.DB_PATH), read_only=True)
    try:
        rows = con.execute("SELECT salamander_id, overall_quality FROM image_quality").fetchall()
    finally:
        con.close()
    return {sid: float(q) for sid, q in rows if q is not None and np.isfinite(q)}


def pair_quality(sets, q_idx: int, gal_idx, quality: dict[str, float] | None) -> float:
    """How much this comparison's geometry can be trusted, in [0, 1].

    The query is one photo, but a candidate INDIVIDUAL is several photos concatenated, so it has no
    single quality. The candidate side is represented by its **best** photo — the question being
    "how good is the best evidence available for this animal" — and the pair is then the **worse**
    of the two sides, because a comparison is only as trustworthy as its weaker half.

    Returns 1.0 when quality is unknown, which makes the interaction collapse to the plain geometric
    feature: an unmeasured photo is not assumed bad.
    """
    if not quality:
        return 1.0
    qq = quality.get(sets[q_idx].sid)
    gq = [quality.get(sets[g].sid) for g in gal_idx]
    gq = [x for x in gq if x is not None]
    if qq is None or not gq:
        return 1.0
    return float(min(qq, max(gq)))


# ----------------------------------------------------------------------------- image arrays
def precompute(sets, weight_lookup, images):
    """``i -> (emb, w, xy)`` for the given image indices: L2-normalized spots, per-spot
    distinctiveness (default 0.5 if a spot has no learned weight), and centroids."""
    out = {}
    for i in images:
        s = sets[i]
        w = np.array([weight_lookup.get((s.sid, int(sid)), 0.5) for sid in s.spot_ids], float)
        out[i] = (_norm(s.spots), w, np.asarray(s.centroids, float))
    return out


# ----------------------------------------------------------------------------- pair builders
def build_pairs(sets, images, weight_lookup, *, neg_per_query=None, seed=0, quality=None):
    """Training rows: strict feature vector + same/different label, one per (query, candidate)."""
    rng = np.random.default_rng(seed)
    arr = precompute(sets, weight_lookup, images)
    by_label: dict[str, list[int]] = {}
    for i in images:
        by_label.setdefault(sets[i].label, []).append(i)
    all_labels = list(by_label)

    X, y, qids, clab = [], [], [], []
    for q in images:
        yq = sets[q].label
        if not any(g != q for g in by_label[yq]):
            continue
        cand = [c for c in all_labels if any(g != q for g in by_label[c])]
        if neg_per_query is not None:
            negs = [c for c in cand if c != yq]
            rng.shuffle(negs)
            cand = [yq] + negs[:neg_per_query]
        qe, wq, qxy = arr[q]
        for c in cand:
            gal = [g for g in by_label[c] if g != q]
            ce = np.concatenate([arr[g][0] for g in gal])
            wc = np.concatenate([arr[g][1] for g in gal])
            cxy = np.concatenate([arr[g][2] for g in gal])
            X.append(strict_features(qe, ce, wq, wc, qxy, cxy,
                                     quality=pair_quality(sets, q, gal, quality)))
            y.append(int(c == yq)); qids.append(q); clab.append(c)
    return np.array(X), np.array(y, float), np.array(qids), np.array(clab, dtype=object)


def build_openset(sets, gallery_imgs, query_imgs, weight_lookup, quality=None):
    """Census rows: strict feature vector per (query image, gallery individual), leave-query-out.

    Also returns ``unexplained`` — the ``unexplained_q`` feature per pair — so the caller can use
    the best candidate's unexplained-distinctive mass directly as a novelty score.
    """
    arr = precompute(sets, weight_lookup, list(set(gallery_imgs) | set(query_imgs)))
    by_label: dict[str, list[int]] = {}
    for i in gallery_imgs:
        by_label.setdefault(sets[i].label, []).append(i)
    uidx = STRICT_FEATURE_NAMES.index("unexplained_q")

    X, qids, clab = [], [], []
    for q in query_imgs:
        qe, wq, qxy = arr[q]
        for c, gal_all in by_label.items():
            gal = [g for g in gal_all if g != q]
            if not gal:
                continue
            ce = np.concatenate([arr[g][0] for g in gal])
            wc = np.concatenate([arr[g][1] for g in gal])
            cxy = np.concatenate([arr[g][2] for g in gal])
            X.append(strict_features(qe, ce, wq, wc, qxy, cxy,
                                     quality=pair_quality(sets, q, gal, quality)))
            qids.append(q); clab.append(c)
    X = np.array(X)
    return X, np.array(qids), np.array(clab, dtype=object), (X[:, uidx] if len(X) else np.zeros(0))


def build_openset_hand(sets, gallery_imgs, query_imgs, weight_lookup,
                       pos_lookup=None, sigma_pos=0.12):
    """Same census pairs but scored by the training-free :func:`strict_hand_score`.

    Pass ``pos_lookup`` ``(sid, spot_id) -> (axis_t, axis_offset/length)`` to enable the position
    gate; leave it ``None`` for appearance-only (the original behaviour)."""
    pool = list(set(gallery_imgs) | set(query_imgs))
    arr = precompute(sets, weight_lookup, pool)
    xy = body_coords(sets, pool, pos_lookup) if pos_lookup is not None else None
    by_label: dict[str, list[int]] = {}
    for i in gallery_imgs:
        by_label.setdefault(sets[i].label, []).append(i)
    sc, qids, clab = [], [], []
    for q in query_imgs:
        qe, wq, _ = arr[q]
        xyq = xy[q] if xy is not None else None
        for c, gal_all in by_label.items():
            gal = [g for g in gal_all if g != q]
            if not gal:
                continue
            ce = np.concatenate([arr[g][0] for g in gal])
            wc = np.concatenate([arr[g][1] for g in gal])
            xyc = np.concatenate([xy[g] for g in gal]) if xy is not None else None
            sc.append(strict_hand_score(qe, ce, wq, wc, xyq=xyq, xyc=xyc, sigma_pos=sigma_pos))
            qids.append(q); clab.append(c)
    return np.array(sc, float), np.array(qids), np.array(clab, dtype=object)
