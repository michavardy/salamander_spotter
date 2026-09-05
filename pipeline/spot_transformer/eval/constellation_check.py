"""Phase 0 of the constellation plan — is the relational evidence simply in the wrong frame?

The matcher already computes constellation evidence. ``aggregator.match_features`` ends with
``geom_consistency`` (Spearman of pairwise distances among matched spots) and ``ransac_frac`` (the
inlier fraction of a fitted **similarity** transform), and the learned rule leans on the second at
+0.51. But both are computed on ``global_centroid_x/y`` — **image pixels** — and a similarity
transform is rigid by construction. Euclidean pixel distances contract and expand when a salamander
curls, which is exactly the deformation the geometry is supposed to survive.

Meanwhile the body-intrinsic frame the dataset already carries — ``axis_t`` (arc length head->tail)
and ``axis_offset/length_px`` (signed lateral offset) — is consulted **one spot at a time**, as
``strict_pair_score``'s Gaussian position gate. Nothing ever measures a *relation between two spots*
in the frame where relations are stable.

So before any constellation machinery gets written, three measurements say whether it would pay:

**M1. The coordinate-frame swap.** Take the two deployed features plus a third parameter-free one,
   compute each in image pixels and in body coordinates on the same photo pairs, and compare
   true/false AUROC. This is the whole hypothesis in one table: if the body frame wins, the frame
   was the problem and the features were fine. If they tie, the constellation signal is weak on its
   own terms and the rest of the plan is not worth building.

   A note on conditioning, which is the mechanism the swap is betting on. In image pixels the model
   is a 4-DOF similarity fitted from 2 sampled points, so at the n=3-5 correspondences a typical
   true pair actually has, almost every degree of freedom is spent on the fit and one or two points
   remain as evidence. In body coordinates both photos are *already* normalised to the same frame,
   so the expected transform is the identity: a 0-DOF or (robustly) 1-DOF model, where every
   correspondence counts as evidence. The frame does not merely remove curl — it buys back degrees
   of freedom exactly where the data is thinnest.

**M2. "No evidence" is currently scored as "contradictory evidence".** In ``match_features`` the
   two geometry features initialise to ``0.0`` and stay there when fewer than 3 mutual matches
   exist. A pair with two matches is therefore handed to the classifier with the same geometry
   value as a pair whose constellation actively disagrees. With a median of 21.5 spots and 24%
   survival, that is not a corner case. This part counts how often it fires on TRUE pairs — and,
   because the honest possibility is that the conflation is accidentally load-bearing, also asks
   whether "undefined" is itself predictive of a false pair.

**M3. The edge-level ceiling.** ``representation_check`` established the null this plan rests on:
   on 417 human-judged correspondences the 62-dim cosine scores **0.484** (chance) at separating
   accepted from rejected, while separating accepted from random cross-animal pairs at **0.997**.
   The stated conclusion is that the reviewer is judging *configuration*. That is a hypothesis, and
   it is testable on the labels already collected: give each judged edge a **relational** score —
   does this correspondence agree with the other correspondences in its photo pair — and see
   whether it recovers the verdict the appearance cosine cannot.

Every input is already in the database. No training, no new labels, ~2-4 minutes.

    pixi run constellation-check
    pixi run constellation-check --include-synth      # Gemini views too (generator artifacts)
    pixi run constellation-check --neg-per-true 3     # negatives sampled per true pair

**Anchors.** Two controls run alongside, because a "constellation" feature that scores well without
a real correspondence is measuring spot spacing rather than constellation: ``shuffled`` recomputes
every feature with the candidate side permuted, and must sit at chance. Read it first — if the
control moves, nothing below it means anything (results.md #18).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval"):
    p = str(_ST / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import data as d                                              # noqa: E402
import review_labels as rl                                    # noqa: E402
from aggregator import _norm, _similarity_inliers, attach_centroids   # noqa: E402
from census import auroc                                      # noqa: E402

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

MATCH_THR = 0.4        # a correspondence counts once its mutual-NN cosine clears this
MIN_GEOM = 3           # fewer matched spots than this and no geometry is defined at all
AXIAL_TOL = 0.05       # inlier band for the axial-shift model, in body lengths (5% of head->tail)

# Features computable in EITHER frame -- these are the frame swap, and the M1 gate reads them.
SHARED = ["geom_spearman", "ransac_frac", "nbr_agree"]
# Features that only exist once the coordinates mean something anatomically. Preview of Phase 1,
# not part of the gate: they are hand-designed and reported on the same pairs that motivated them,
# so they are a reason to run the census sweep, never a result on their own.
BODY_ONLY = ["axial_inliers", "kendall_axial", "side_agree"]


# ----------------------------------------------------------------------------- inputs
def body_lookup(db_path=None) -> dict[tuple[str, int], tuple[float, float]]:
    """``(sid, spot_id) -> (axis_t, axis_offset/length_px)`` — the scale-invariant body frame.

    Same construction as ``compare_strict.build_pos_lookup``: dividing the signed lateral offset by
    the axis's own arc length puts both coordinates in body-length units, so a distance in this
    frame is a distance *along the animal* and the two axes are directly comparable. Spots with no
    fitted axis come back NaN and are dropped by the callers rather than defaulted to the midline,
    which would fabricate agreement.
    """
    import duckdb
    con = duckdb.connect(str(db_path or d.DB_PATH), read_only=True)
    try:
        length = {r[0]: r[1] for r in
                  con.execute("SELECT salamander_id, length_px FROM body_axis").fetchall()}
        sp = con.execute("SELECT salamander_id AS sid, spot_id, axis_t, axis_offset "
                         "FROM spots").df()
    finally:
        con.close()
    out = {}
    for r in sp.itertuples(index=False):
        L = length.get(r.sid)
        t = float(r.axis_t) if r.axis_t is not None and np.isfinite(r.axis_t) else np.nan
        off = float(r.axis_offset) if r.axis_offset is not None and np.isfinite(r.axis_offset) \
            else np.nan
        u = off / L if (L and np.isfinite(off) and L > 0) else np.nan
        out[(r.sid, int(r.spot_id))] = (t, u)
    return out


def build_context(include_synth: bool = False):
    """``(sets with centroids, body-frame lookup)`` — everything the three parts share."""
    sets = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))
    if not include_synth:
        sets = [s for s in sets if not s.is_synth]
    return sets, body_lookup()


def frames_for(s, blook) -> dict[str, np.ndarray]:
    """Both coordinate frames for one photo's spots, aligned to ``s.spots`` row order."""
    body = np.array([blook.get((s.sid, int(i)), (np.nan, np.nan)) for i in s.spot_ids], float)
    return {"image": np.asarray(s.centroids, float), "body": body}


# ----------------------------------------------------------------------------- correspondences
def mutual_matches(A, B, thr: float = MATCH_THR):
    """Indices ``(ia, ib)`` of mutual nearest neighbours clearing ``thr``.

    The matcher's own rule, unchanged, so everything measured here describes the correspondences
    the system actually forms rather than an idealisation of them.
    """
    S = _norm(A.spots) @ _norm(B.spots).T
    ab, ba = S.argmax(1), S.argmax(0)
    ia = [i for i in range(len(S)) if ba[ab[i]] == i and float(S[i, ab[i]]) >= thr]
    return np.array(ia, int), np.array([ab[i] for i in ia], int)


# ----------------------------------------------------------------------------- features
def geom_spearman(Q, C) -> float:
    """Spearman correlation of the two pairwise-distance vectors — the deployed feature."""
    from scipy.spatial.distance import pdist                             # noqa: PLC0415
    from scipy.stats import spearmanr                                    # noqa: PLC0415
    dq, dc = pdist(Q), pdist(C)
    if dq.std() < 1e-9 or dc.std() < 1e-9:
        return np.nan
    r = spearmanr(dq, dc).correlation
    return float(r) if np.isfinite(r) else np.nan


def nbr_agree(Q, C, k: int = 3) -> float:
    """Fraction of a matched spot's k nearest matched neighbours that stay its neighbours.

    Parameter-free consensus: no transform is fitted, so nothing is spent on estimation and the
    measure degrades gracefully at small n — which is the regime that matters here. Rows of ``Q``
    and ``C`` are already aligned by the correspondence, so neighbour index sets compare directly.
    """
    from scipy.spatial.distance import pdist, squareform                 # noqa: PLC0415
    n = len(Q)
    if n < 3:
        return np.nan
    k = min(k, n - 1)
    nq = np.argsort(squareform(pdist(Q)), axis=1)[:, 1:k + 1]
    nc = np.argsort(squareform(pdist(C)), axis=1)[:, 1:k + 1]
    return float(np.mean([len(set(nq[i]) & set(nc[i])) / k for i in range(n)]))


def axial_inliers(tq, tc, tol: float = AXIAL_TOL) -> float:
    """Inlier fraction of a 1-DOF model: ``t_c = t_q + b``, with ``b`` the median shift.

    The body frame's payoff in one line. A head/tail anchor that lands slightly early or late
    produces a *constant* axial shift, so one robust parameter absorbs it and every correspondence
    remains evidence — against 4 parameters fitted from 2 points in the image frame.
    """
    if len(tq) < MIN_GEOM:
        return np.nan
    dt = tc - tq
    return float(np.mean(np.abs(dt - np.median(dt)) <= tol))


def kendall_axial(tq, tc) -> float:
    """Kendall tau of head->tail ORDER. Survives any monotone re-parametrisation of the spine,
    so it is invariant to curling, foreshortening and a mis-scaled axis alike — the weakest
    assumption on this list, and the only one that holds when the animal is photographed at an
    angle that compresses one end."""
    from scipy.stats import kendalltau                                   # noqa: PLC0415
    if len(tq) < MIN_GEOM or np.std(tq) < 1e-12 or np.std(tc) < 1e-12:
        return np.nan
    t = kendalltau(tq, tc).correlation
    return float(t) if np.isfinite(t) else np.nan


def side_agree(uq, uc) -> float:
    """Fraction of correspondences that keep the same side of the midline. Doubles as a
    reversed-axis detector: a flipped body axis drives this toward 0 while appearance is unchanged
    (the ``sj_1_1`` fault in ``data_hygiene``)."""
    if len(uq) < MIN_GEOM:
        return np.nan
    ok = np.isfinite(uq) & np.isfinite(uc)
    if ok.sum() < MIN_GEOM:
        return np.nan
    return float(np.mean(np.sign(uq[ok]) == np.sign(uc[ok])))


def _features_once(fa, fb, ia, ib, perm) -> dict:
    """Every feature for one photo pair under one assignment of the correspondence."""
    out = {}
    for frame in ("image", "body"):
        Q, C = fa[frame][ia], fb[frame][ib[perm]]
        ok = np.isfinite(Q).all(1) & np.isfinite(C).all(1)
        if ok.sum() < MIN_GEOM:
            continue
        Q, C = Q[ok], C[ok]
        out[f"{frame}.geom_spearman"] = geom_spearman(Q, C)
        out[f"{frame}.ransac_frac"] = _similarity_inliers(Q, C)
        out[f"{frame}.nbr_agree"] = nbr_agree(Q, C)
        if frame == "body":
            out["body.axial_inliers"] = axial_inliers(Q[:, 0], C[:, 0])
            out["body.kendall_axial"] = kendall_axial(Q[:, 0], C[:, 0])
            out["body.side_agree"] = side_agree(Q[:, 1], C[:, 1])
    return out


def pair_features(A, B, fa, fb, ia, ib, *, n_perm: int = 3, rng=None) -> dict:
    """Features for one pair, plus a **per-pair permutation null** and the excess over it.

    The correction matters more than it sounds. Every one of these statistics is confounded with
    the number of correspondences: with 3 matched points a random similarity transform fits a large
    fraction of them, and a 2-neighbour neighbourhood agrees by luck. True pairs have more matches
    than false ones (median 4 vs 3), so the confound points the wrong way — the raw feature can be
    *anti*-correlated with truth purely through n. A single global shuffled arm can detect that but
    cannot remove it.

    So each pair gets its own null: recompute every feature with the candidate side permuted,
    ``n_perm`` times, and subtract the mean. The excess is "how much more consistent than chance
    *for a pair with exactly these spots and exactly this many matches*", which is the quantity the
    hypothesis is actually about. ``excess.*`` is what the M1 gate reads; the raw values are kept
    alongside because they are what the deployed feature actually computes today.
    """
    if len(ia) < MIN_GEOM:
        return {}
    out = _features_once(fa, fb, ia, ib, np.arange(len(ib)))
    if not n_perm or rng is None or not out:
        return out
    nulls = [_features_once(fa, fb, ia, ib, rng.permutation(len(ib))) for _ in range(n_perm)]
    for k, v in list(out.items()):
        vals = [n[k] for n in nulls if np.isfinite(n.get(k, np.nan))]
        out[f"null.{k}"] = float(np.mean(vals)) if vals else np.nan
        out[f"excess.{k}"] = (v - np.mean(vals)) if (vals and np.isfinite(v)) else np.nan
    return out


# ----------------------------------------------------------------------------- pair table
def build_pairs(sets, blook, *, neg_per_true: int = 3, seed: int = 0, n_perm: int = 3,
                max_neg: int = 6000) -> pd.DataFrame:
    """One row per photo pair: every feature in both frames, plus the shuffled control.

    Pair construction mirrors ``feasibility.pair_quality`` (all same-label pairs, negatives sampled
    at ``neg_per_true``x) so the two screens describe the same population and their numbers can be
    read side by side.
    """
    rng = np.random.default_rng(seed)
    idxs = [i for i, s in enumerate(sets) if len(s.spots) >= MIN_GEOM]
    by_label: dict[str, list[int]] = {}
    for i in idxs:
        by_label.setdefault(sets[i].label, []).append(i)

    pos = [(a, b) for v in by_label.values() for k, a in enumerate(v) for b in v[k + 1:]]
    want = min(len(pos) * neg_per_true, max_neg)
    neg: set[tuple[int, int]] = set()
    guard = 0
    while len(neg) < want and guard < want * 50:
        guard += 1
        a, b = int(rng.choice(idxs)), int(rng.choice(idxs))
        if sets[a].label == sets[b].label:
            continue
        neg.add((min(a, b), max(a, b)))

    frames = {}

    def F(i):
        if i not in frames:
            frames[i] = frames_for(sets[i], blook)
        return frames[i]

    ctrl_rng = np.random.default_rng(seed + 1)
    rows = []
    for a, b, same in [(a, b, 1) for a, b in pos] + [(a, b, 0) for a, b in neg]:
        A, B = sets[a], sets[b]
        ia, ib = mutual_matches(A, B)
        row = dict(sid_a=A.sid, sid_b=B.sid, label_a=A.label, label_b=B.label, same=same,
                   n_mutual=len(ia), n_min=min(len(A.spots), len(B.spots)),
                   geom_defined=int(len(ia) >= MIN_GEOM))
        row.update(pair_features(A, B, F(a), F(b), ia, ib, n_perm=n_perm, rng=ctrl_rng))
        rows.append(row)
    df = pd.DataFrame(rows)
    # Guarantee every feature column exists even if no pair could produce it, so a degenerate run
    # reports "too few computable pairs" per feature instead of dying on a KeyError.
    want = ([f"{fr}.{f}" for fr in ("image", "body") for f in SHARED]
            + [f"body.{f}" for f in BODY_ONLY])
    for c in want + [f"{p}.{c}" for p in ("null", "excess") for c in want]:
        if c not in df:
            df[c] = np.nan
    return df


# ----------------------------------------------------------------------------- statistics
def cluster_boot_delta(x, y, labels, groups, *, n_boot=2000, seed=0):
    """95% CI for ``AUROC(x) - AUROC(y)`` on the SAME rows, resampling individuals.

    Paired because the two frames score identical pairs: the interesting quantity is the
    difference, and a paired resample cancels the pair-sampling noise both share. Clustered on the
    individual because photo pairs from one animal share its extraction quality and its axis fit,
    and an unclustered interval would be two or three times too narrow (results.md #15).
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        yy = labels[idx]
        if yy.sum() == 0 or (yy == 0).sum() == 0:
            continue
        a, b = auroc(x[idx], yy), auroc(y[idx], yy)
        if np.isfinite(a) and np.isfinite(b):
            vals.append(a - b)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def frame_table(pairs: pd.DataFrame, *, kind: str = "excess", n_boot: int = 2000) -> pd.DataFrame:
    """Per feature: AUROC in each frame, the paired delta with a CI, and the permutation null.

    ``kind="raw"`` scores the feature as the deployed code computes it; ``kind="excess"`` scores it
    net of its own per-pair permutation null, which is the confound-free version and the one the
    gate reads (see :func:`pair_features`).

    Rows are restricted to pairs where BOTH frames are computable, so the comparison is on
    identical data and a difference cannot come from one frame quietly scoring a different subset.
    """
    pre = "" if kind == "raw" else f"{kind}."
    out = []
    for f in SHARED:
        ci, cb = f"{pre}image.{f}", f"{pre}body.{f}"
        ok = pairs[ci].notna() & pairs[cb].notna()
        sub = pairs[ok]
        if len(sub) < 50:
            out.append(dict(feature=f, note="too few computable pairs"))
            continue
        y = sub["same"].to_numpy().astype(int)
        xi, xb = sub[ci].to_numpy(float), sub[cb].to_numpy(float)
        lo, hi = cluster_boot_delta(xb, xi, y, sub["label_a"].to_numpy(), n_boot=n_boot)
        ctrl = sub[f"null.body.{f}"].to_numpy(float)
        cok = np.isfinite(ctrl)
        out.append(dict(
            feature=f, n=int(len(sub)), kind=kind,
            auroc_image=float(auroc(xi, y)), auroc_body=float(auroc(xb, y)),
            delta=float(auroc(xb, y) - auroc(xi, y)), lo=lo, hi=hi,
            auroc_null=float(auroc(ctrl[cok], y[cok])) if cok.sum() > 50 else float("nan"),
        ))
    return pd.DataFrame(out)


def body_only_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """The features that only exist in the anatomical frame, raw and net of the permutation null."""
    out = []
    for f in BODY_ONLY:
        col, ex, nl = f"body.{f}", f"excess.body.{f}", f"null.body.{f}"
        ok = pairs[col].notna()
        if ok.sum() < 50:
            out.append(dict(feature=f, note="too few computable pairs"))
            continue
        sub = pairs[ok]
        y = sub["same"].to_numpy().astype(int)
        eok = sub[ex].notna().to_numpy()
        nok = sub[nl].notna().to_numpy()
        out.append(dict(feature=f, n=int(ok.sum()),
                        auroc=float(auroc(sub[col].to_numpy(float), y)),
                        auroc_excess=(float(auroc(sub[ex].to_numpy(float)[eok], y[eok]))
                                      if eok.sum() > 50 else float("nan")),
                        auroc_null=(float(auroc(sub[nl].to_numpy(float)[nok], y[nok]))
                                    if nok.sum() > 50 else float("nan")),
                        mean_true=float(sub[sub.same == 1][col].mean()),
                        mean_false=float(sub[sub.same == 0][col].mean())))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- M2
def undefined_analysis(pairs: pd.DataFrame) -> dict:
    """How often geometry is undefined, and whether the ``0.0`` fallback is accidentally a feature.

    The second question is the honest one. If undefined fires mostly on FALSE pairs then collapsing
    it onto "inconsistent" is approximately right by luck, and replacing it with a missing-indicator
    is a refactor rather than a fix. If it fires on true pairs too, the classifier is being told
    that a thinly-matched genuine pair is geometrically contradictory.
    """
    t, f = pairs[pairs.same == 1], pairs[pairs.same == 0]
    y = pairs["same"].to_numpy().astype(int)
    # The indicator ALONE as a classifier: does "geometry was computable" predict a true pair?
    ind = pairs["geom_defined"].to_numpy(float)
    defined = pairs[pairs.geom_defined == 1]
    yd = defined["same"].to_numpy().astype(int)
    res = dict(
        n_true=int(len(t)), n_false=int(len(f)),
        undef_true=float((t.geom_defined == 0).mean()),
        undef_false=float((f.geom_defined == 0).mean()),
        auroc_indicator=float(auroc(ind, y)),
        median_mutual_true=float(t.n_mutual.median()),
        median_mutual_false=float(f.n_mutual.median()),
    )
    for f_ in SHARED:
        col = f"body.{f_}"
        # as deployed: undefined -> 0.0, indistinguishable from "inconsistent"
        filled = pairs[col].fillna(0.0).to_numpy(float)
        res[f"auroc_{f_}_filled"] = float(auroc(filled, y))
        ok = defined[col].notna()
        res[f"auroc_{f_}_defined_only"] = (float(auroc(defined[col][ok].to_numpy(float),
                                                       yd[ok.to_numpy()]))
                                           if ok.sum() > 50 else float("nan"))
    return res


# ----------------------------------------------------------------------------- M3
def edge_relational(sets, blook, edges: pd.DataFrame) -> pd.DataFrame:
    """Per human-judged correspondence: appearance cosine vs RELATIONAL agreement.

    ``representation_check`` asks what an edge looks like; this asks whether the edge agrees with
    its neighbours. Three scores, each the local version of a pair-level feature:

    ``consensus_resid``  how far this edge's axial shift sits from the median shift of the OTHER
                         correspondences in the same photo pair (negated, so higher is better) —
                         an edge that disagrees with the pair's own consensus is the arithmetic
                         form of "that one is in the wrong place".
    ``nbr_support``      of this spot's nearest matched neighbours, how many keep their rank around
                         the partner — the reviewer's "everything around it lines up".
    ``rank_agree``       agreement of head->tail rank position within each photo's own spots, which
                         needs no other correspondence at all and so survives sparse pairs.

    The comparison is deliberately on the same rows as the cosine baseline, so the two numbers are
    answerable against each other rather than against different subsets.
    """
    by_sid = {s.sid: s for s in sets}
    rows = []
    cache: dict[tuple[str, str], tuple] = {}
    for r in edges.itertuples(index=False):
        A, B = by_sid.get(r.sid_a), by_sid.get(r.sid_b)
        if A is None or B is None:
            continue
        key = (r.sid_a, r.sid_b)
        if key not in cache:
            ia, ib = mutual_matches(A, B)
            cache[key] = (ia, ib, frames_for(A, blook), frames_for(B, blook))
        ia, ib, fa, fb = cache[key]

        pa = np.flatnonzero(A.spot_ids == r.spot_a)
        pb = np.flatnonzero(B.spot_ids == r.spot_b)
        if not len(pa) or not len(pb):
            continue
        pa, pb = int(pa[0]), int(pb[0])
        ta, tb = fa["body"][pa, 0], fb["body"][pb, 0]
        if not (np.isfinite(ta) and np.isfinite(tb)):
            continue

        va = A.spots[pa] / (np.linalg.norm(A.spots[pa]) + 1e-12)
        vb = B.spots[pb] / (np.linalg.norm(B.spots[pb]) + 1e-12)

        # consensus over the OTHER correspondences in this pair — never this edge itself, which
        # would let an edge vote for its own plausibility.
        others = [(i, j) for i, j in zip(ia, ib) if i != pa and j != pb]
        oq = np.array([fa["body"][i, 0] for i, _ in others], float)
        oc = np.array([fb["body"][j, 0] for _, j in others], float)
        m = np.isfinite(oq) & np.isfinite(oc)
        shift = np.median(oc[m] - oq[m]) if m.sum() >= 2 else np.nan
        resid = abs((tb - ta) - shift) if np.isfinite(shift) else np.nan

        # rank position of each spot along its own photo's head->tail order
        ra = np.array([fa["body"][i, 0] for i in range(len(A.spot_ids))], float)
        rb = np.array([fb["body"][j, 0] for j in range(len(B.spot_ids))], float)
        fra = np.mean(ra[np.isfinite(ra)] <= ta) if np.isfinite(ra).any() else np.nan
        frb = np.mean(rb[np.isfinite(rb)] <= tb) if np.isfinite(rb).any() else np.nan

        # neighbourhood support: do the nearest OTHER correspondences agree on their shift too
        if m.sum() >= 2:
            near = np.argsort(np.abs(oq[m] - ta))[:3]
            local = (oc[m] - oq[m])[near]
            nbr = float(np.mean(np.abs(local - (tb - ta)) <= AXIAL_TOL))
        else:
            nbr = np.nan

        rows.append(dict(
            individual=r.individual, accepted=bool(r.accepted),
            sid_a=r.sid_a, sid_b=r.sid_b, n_mutual=len(ia),
            cosine=float(va @ vb),
            consensus_resid=(-resid if np.isfinite(resid) else np.nan),
            nbr_support=nbr,
            rank_agree=(-abs(fra - frb) if np.isfinite(fra) and np.isfinite(frb) else np.nan),
        ))
    return pd.DataFrame(rows)


def edge_table(df: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """AUROC + clustered CI for each edge-level score against the human verdict."""
    from representation_check import cluster_bootstrap_auroc               # noqa: PLC0415
    out = []
    for f in ("cosine", "consensus_resid", "nbr_support", "rank_agree"):
        ok = df[f].notna()
        if ok.sum() < 30 or df.loc[ok, "accepted"].nunique() < 2:
            out.append(dict(score=f, note="too few judged edges"))
            continue
        s = df.loc[ok, f].to_numpy(float)
        y = df.loc[ok, "accepted"].to_numpy().astype(int)
        g = df.loc[ok, "individual"].to_numpy()
        lo, hi = cluster_bootstrap_auroc(s, y, g, n_boot=n_boot, seed=seed)
        out.append(dict(score=f, n=int(ok.sum()), n_individuals=int(pd.unique(g).size),
                        auroc=float(auroc(s, y)), lo=lo, hi=hi))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- report
def main():
    import argparse
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-synth", action="store_true",
                    help="include Gemini views (their geometry describes the generator)")
    ap.add_argument("--neg-per-true", type=int, default=3, help="negative pairs per true pair")
    ap.add_argument("--perm", type=int, default=3,
                    help="permutations per pair for the null (0 disables the correction)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (0 to skip)")
    ap.add_argument("--pair-kind", default=None, choices=["real-real", "real-synth", "synth-synth"],
                    help="M3 only: restrict the judged edges (default: all kinds, n=417)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logger.info(f"dataset {d.dataset_name} · embedding table {d.EMB_TABLE}")
    sets, blook = build_context(args.include_synth)
    n_axis = sum(1 for v in blook.values() if np.isfinite(v[0]))
    logger.info(f" {len(sets)} photos ({'incl.' if args.include_synth else 'excl.'} synthetic) · "
          f"{n_axis:,}/{len(blook):,} spots carry a body-frame position")

    pairs = build_pairs(sets, blook, neg_per_true=args.neg_per_true, seed=args.seed,
                        n_perm=args.perm)
    logger.info(f" {int((pairs.same == 1).sum())} true pairs · {int((pairs.same == 0).sum())} false pairs")

    # ------------------------------------------------------------------ M1
    logger.info("=" * 78)
    logger.info(" M1 — the coordinate-frame swap: same feature, image pixels vs body coordinates")
    logger.info("=" * 78)
    raw = frame_table(pairs, kind="raw", n_boot=args.boot)
    ft = frame_table(pairs, kind="excess", n_boot=args.boot)

    def show(tbl, title):
        logger.info(f" {title}")
        logger.info(f"{'feature':<18}{'image':>9}{'body':>9}{'delta':>9}{'95% CI':>16}{'null':>9}")
        logger.info("-" * 70)
        for r in tbl.itertuples(index=False):
            if not np.isfinite(getattr(r, "auroc_body", np.nan)):
                logger.info(f"{r.feature:<18}{getattr(r, 'note', 'not computable')}")
                continue
            ci = f"[{r.lo:+.3f},{r.hi:+.3f}]"
            logger.info(f"{r.feature:<18}{r.auroc_image:>9.3f}{r.auroc_body:>9.3f}{r.delta:>+9.3f}"
                  f"{ci:>16}{r.auroc_null:>9.3f}")

    show(raw, "raw — the feature as the deployed code computes it")
    show(ft, "excess over each pair's own permutation null — CONFOUND-FREE, this is the gate")
    logger.info(" null = the permuted feature scored against truth. Far from 0.50 means the raw feature")
    logger.info(" is confounded with the NUMBER of matches (fewer points fit anything), which is exactly")
    logger.info(" what the excess rows correct for. A large gap between the two tables is a finding about")
    logger.info(" the deployed feature, not a defect in this screen.")

    ctrl_bad = [r.feature for r in ft.itertuples(index=False)
                if np.isfinite(getattr(r, "auroc_null", np.nan))
                and abs(r.auroc_null - 0.5) > 0.10]
    if ctrl_bad:
        logger.warning(f" NOTE: the null arm is off chance for {', '.join(ctrl_bad)} — the RAW feature")
        logger.warning("    carries match-count signal rather than constellation signal. Read the excess")
        logger.warning("    table for the frame question, and treat this as evidence about `ransac_frac`'s")
        logger.warning("    +0.51 weight in the deployed aggregator: part of it may be counting, not geometry.")

    won = [r.feature for r in ft.itertuples(index=False)
           if np.isfinite(getattr(r, "delta", np.nan)) and r.delta >= 0.05 and r.lo > 0]
    lost = [r.feature for r in ft.itertuples(index=False)
            if np.isfinite(getattr(r, "delta", np.nan)) and r.delta <= -0.05 and r.hi < 0]
    flat = [r.feature for r in ft.itertuples(index=False)
            if np.isfinite(getattr(r, "delta", np.nan)) and abs(r.delta) < 0.05]
    if won:
        logger.info(f" => FRAME SWAP SUPPORTED for: {', '.join(won)}.")
        logger.info("    The features were fine; they were computed in the frame that curling breaks.")
        logger.info("    Recompute these on (axis_t, axis_offset/length) and re-run sweep_all9.")
    if flat:
        logger.info(f" => No frame effect for: {', '.join(flat)}. The pixel frame was not what held")
        logger.info("    these back, so swapping it will not pay — look at the correspondences instead.")
    if lost:
        logger.info(f" => BACKWARDS for: {', '.join(lost)} — the image frame is genuinely better. Do not")
        logger.info("    build on this until it is explained; suspect the body axis, not the idea.")

    bt = body_only_table(pairs)
    logger.info(" body-native features (no image-frame counterpart; PREVIEW, not part of the gate)")
    logger.info(f"{'feature':<18}{'raw':>9}{'excess':>9}{'null':>9}{'mean true':>11}{'mean false':>12}")
    logger.info("-" * 68)
    for r in bt.itertuples(index=False):
        if not np.isfinite(getattr(r, "auroc", np.nan)):
            logger.info(f"{r.feature:<18}{getattr(r, 'note', 'not computable')}")
            continue
        logger.info(f"{r.feature:<18}{r.auroc:>9.3f}{r.auroc_excess:>9.3f}{r.auroc_null:>9.3f}"
              f"{r.mean_true:>11.3f}{r.mean_false:>12.3f}")
    logger.info(" These are hand-designed and scored on the pairs that motivated them — a reason to run")
    logger.info(" the census sweep, never a result. Anything here must be confirmed by sweep_all9.")

    # ------------------------------------------------------------------ M2
    logger.info("=" * 78)
    logger.info(" M2 — how often is geometry undefined, and is the 0.0 fallback carrying signal?")
    logger.info("=" * 78)
    ua = undefined_analysis(pairs)
    logger.info(f" median mutual matches: true {ua['median_mutual_true']:.0f} · "
          f"false {ua['median_mutual_false']:.0f}   (geometry needs >= {MIN_GEOM})")
    logger.info(f" undefined on TRUE pairs   {ua['undef_true']:.1%}   <- these are handed to the "
          f"classifier as geom=0.0")
    logger.info(f" undefined on FALSE pairs  {ua['undef_false']:.1%}")
    logger.info(f" the indicator alone as a classifier: AUROC {ua['auroc_indicator']:.3f}")
    logger.info(f"{'feature':<18}{'as deployed':>13}{'defined only':>14}{'difference':>12}")
    logger.info("-" * 57)
    for f in SHARED:
        a, b = ua[f"auroc_{f}_filled"], ua[f"auroc_{f}_defined_only"]
        logger.info(f"{f:<18}{a:>13.3f}{b:>14.3f}{b - a:>+12.3f}")
    logger.info(" The two columns score DIFFERENT row sets — 'defined only' drops the undefined pairs,")
    logger.info(" which are disproportionately false, so it is a harder problem and a lower number there")
    logger.info(" is not evidence that the 0.0 fill helps. Read the columns as two descriptions, not as")
    logger.info(" an A/B test; the A/B test is `geom_valid` in sweep_all9.")

    if ua["undef_true"] > 0.20:
        logger.warning(f" => FIX REQUIRED: {ua['undef_true']:.0%} of genuine pairs carry a geometry value")
        logger.warning("    that means 'inconsistent' but was produced by 'not computable'. Add an explicit")
        logger.warning("    `geom_valid` column and let the model separate the two before anything else.")
    else:
        logger.info(f" => Minor: only {ua['undef_true']:.0%} of true pairs are affected. Worth the")
        logger.info("    indicator for cleanliness, but it is not what is capping recall.")
    if ua["auroc_indicator"] >= 0.60:
        logger.info(f"    NOTE: the indicator alone scores {ua['auroc_indicator']:.3f}, so the current")
        logger.info("    conflation is partly load-bearing — keep the count as a feature when you split")
        logger.info("    it out, or the refactor will cost signal it was smuggling in.")

    # ------------------------------------------------------------------ M3
    logger.info("=" * 78)
    logger.info(" M3 — do RELATIONAL scores recover the human verdict that appearance cannot?")
    logger.info("=" * 78)
    edges = rl.verdict_edges(d.dataset_name, proposed_by="algorithm", pair_kind=args.pair_kind)
    et = pd.DataFrame()
    if not len(edges):
        logger.warning(" no adjudicated machine-proposed edges — skipping (run the preprocessing review)")
    else:
        # The reviewer judged whatever pairs the app served, and most of those involve a Gemini
        # view. Scoring them against a real-only context silently drops ~86% of the labels (417 ->
        # 59) and quietly turns this into the `real-real` subset, so M3 always builds its own
        # context. `--pair-kind real-real` asks for that subset deliberately; it is the only fully
        # honest one, and at n=59 it is a sanity check rather than a gate.
        sets_edges = sets if args.include_synth else build_context(True)[0]
        er = edge_relational(sets_edges, blook, edges)
        logger.info(f" {len(er)}/{len(edges)} judged correspondences scored, over "
              f"{er['individual'].nunique()} individuals "
              f"({int(er['accepted'].sum())} accepted / {int((~er['accepted']).sum())} rejected)"
              + (f"  [pair_kind={args.pair_kind}]" if args.pair_kind else ""))
        et = edge_table(er, n_boot=args.boot, seed=args.seed)
        logger.info(f"{'score':<20}{'AUROC':>9}{'95% CI':>16}{'n':>7}")
        logger.info("-" * 52)
        for r in et.itertuples(index=False):
            if getattr(r, "note", None) and not np.isfinite(getattr(r, "auroc", np.nan)):
                logger.info(f"{r.score:<20}{r.note}")
                continue
            logger.info(f"{r.score:<20}{r.auroc:>9.3f}{f'[{r.lo:.2f},{r.hi:.2f}]':>16}{r.n:>7}")
        logger.info(" `cosine` is the baseline: representation_check measures it at 0.484 on this set.")
        rel = [r for r in et.itertuples(index=False)
               if r.score != "cosine" and np.isfinite(getattr(r, "auroc", np.nan))]
        best = max(rel, key=lambda r: r.auroc) if rel else None
        if best is not None and best.auroc >= 0.60:
            logger.info(f" => CONFIRMED at the edge level: `{best.score}` reaches {best.auroc:.3f} where")
            logger.info("    appearance is at chance. The reviewer IS judging configuration, and it is")
            logger.info("    recoverable from coordinates already in the database.")
        elif best is not None:
            logger.info(f" => NOT CONFIRMED: the best relational score is {best.auroc:.3f}. Either the")
            logger.info("    reviewer's rejections are not configurational after all, or these three")
            logger.info("    statistics are the wrong local read of configuration. Do not spend the")
            logger.info("    census run on the strength of M1 alone until this is understood.")

    # ------------------------------------------------------------------ write
    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "constellation"
    outdir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(outdir / "pair_frames.csv", index=False)
    (outdir / "constellation_check.json").write_text(
        json.dumps(dict(frame_raw=raw.to_dict("records"), frame_excess=ft.to_dict("records"),
                        body_only=bt.to_dict("records"),
                        undefined=ua, edges=et.to_dict("records") if len(et) else []),
                   indent=2, default=float), encoding="utf-8")

    md = ["# Phase 0 — is the constellation evidence in the wrong coordinate frame?", "",
          f"- dataset `{d.dataset_name}` · synthetic views "
          f"{'included' if args.include_synth else 'excluded'} · no training, no new labels",
          f"- {int((pairs.same == 1).sum())} true / {int((pairs.same == 0).sum())} false photo "
          f"pairs · correspondences are the matcher's own mutual-NN rule at cosine >= {MATCH_THR}",
          "- **Screens, not results.** Headline numbers still come from `sweep_all9` on the census",
          "  protocol. CIs are cluster bootstraps over individuals.", "",
          "## M1. The coordinate-frame swap", ""]
    for tbl, title, note in (
        (raw, "Raw — the feature as the deployed code computes it",
         "`null` is the permuted feature scored against truth. Far from 0.50 means the raw "
         "feature is confounded with the *number* of matches, not with constellation agreement."),
        (ft, "Excess over each pair's own permutation null — confound-free, **this is the gate**",
         "Gate: delta >= +0.05 with the CI clear of zero."),
    ):
        md += [f"### {title}", "",
               "| feature | image frame | body frame | delta | 95% CI | null |",
               "|---|---|---|---|---|---|"]
        for r in tbl.itertuples(index=False):
            if not np.isfinite(getattr(r, "auroc_body", np.nan)):
                continue
            md.append(f"| `{r.feature}` | {r.auroc_image:.3f} | **{r.auroc_body:.3f}** | "
                      f"{r.delta:+.3f} | [{r.lo:+.3f}, {r.hi:+.3f}] | {r.auroc_null:.3f} |")
        md += ["", note, ""]
    md += ["### Body-native features (preview — not part of the gate)", "",
           "| feature | raw | excess | null | mean true | mean false |",
           "|---|---|---|---|---|---|"]
    for r in bt.itertuples(index=False):
        if not np.isfinite(getattr(r, "auroc", np.nan)):
            continue
        md.append(f"| `{r.feature}` | {r.auroc:.3f} | **{r.auroc_excess:.3f}** | "
                  f"{r.auroc_null:.3f} | {r.mean_true:.3f} | {r.mean_false:.3f} |")
    md += ["", "## M2. Undefined geometry", "",
           f"- median mutual matches: true **{ua['median_mutual_true']:.0f}**, "
           f"false {ua['median_mutual_false']:.0f} (geometry needs >= {MIN_GEOM})",
           f"- undefined on **true** pairs: **{ua['undef_true']:.1%}** — scored as `geom = 0.0`, "
           f"i.e. as geometric contradiction",
           f"- undefined on false pairs: {ua['undef_false']:.1%}",
           f"- the indicator alone: AUROC **{ua['auroc_indicator']:.3f}**", "",
           "| feature | as deployed (undefined -> 0.0) | defined pairs only | difference |",
           "|---|---|---|---|"]
    for f in SHARED:
        a, b = ua[f"auroc_{f}_filled"], ua[f"auroc_{f}_defined_only"]
        md.append(f"| `{f}` | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    md += ["", "The two columns score **different row sets** — 'defined pairs only' drops the "
           "undefined pairs, which are disproportionately false, so it is a harder problem and a "
           "lower number there is not evidence that the `0.0` fill helps. These are two "
           "descriptions, not an A/B test; the A/B test is `geom_valid` in `sweep_all9`.",
           "", "## M3. Relational scores vs the human verdict", ""]
    if len(et):
        md += ["| score | AUROC | 95% CI | n |", "|---|---|---|---|"]
        for r in et.itertuples(index=False):
            if not np.isfinite(getattr(r, "auroc", np.nan)):
                continue
            md.append(f"| `{r.score}` | **{r.auroc:.3f}** | [{r.lo:.2f}, {r.hi:.2f}] | {r.n} |")
        md += ["", "`cosine` is the appearance baseline `representation_check` measures at 0.484.",
               "Gate: a relational score >= 0.60 confirms the rejections are configurational."]
    else:
        md.append("_No adjudicated correspondences available._")
    (outdir / "RESULTS_constellation_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info(f"wrote {outdir / 'RESULTS_constellation_check.md'}")
    logger.info(f"      {outdir / 'pair_frames.csv'}  ({len(pairs):,} pairs, for your own digging)")


if __name__ == "__main__":
    main()
