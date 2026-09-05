"""Learned aggregation (re-ranking) over RAW spot matches.

The raw 62-dim spot embeddings stay fixed (they are already good). What we learn is the small
function that turns a bag of query->candidate spot matches into a per-candidate score +
confidence -- i.e. the *voting strategy*: how to weigh "few strong matches" vs "many weak
matches", how to use geometric consistency, etc.

Why this is well-posed where learning the representation was not: every (query image,
candidate individual) pair is a labeled example (same / different), so supervision is
abundant (~hundreds of thousands of pairs) for a *tiny* model over match-statistics -- and
those statistics are individual-agnostic, so the learned rule transfers to unseen animals.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))   # repo root, so `pipeline.*` resolves when run as a script

import sys
from pathlib import Path

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

from pipeline.spot_transformer.core import data as d   # noqa: E402


# ----------------------------------------------------------------------------- centroids
def attach_centroids(sets, db_path=None):
    """Attach per-spot geometry from the ``spots`` table.

    ``ImageSet.centroids`` (x, y image px) drives the geometric-consistency features; ``axis_t``
    (0=head..1=tail) is the body-intrinsic axial coordinate the axis-ordered models order by.
    Both come from the same scan, so this stays one query.
    """
    import duckdb
    con = duckdb.connect(str(db_path or d.DB_PATH), read_only=True)
    try:
        df = con.execute(
            "SELECT salamander_id, spot_id, global_centroid_x, global_centroid_y, axis_t FROM spots"
        ).df()
    finally:
        con.close()
    xy = {(r[0], int(r[1])): (r[2], r[3], r[4]) for r in df.itertuples(index=False)}
    for s in sets:
        got = [xy.get((s.sid, int(i)), (np.nan, np.nan, np.nan)) for i in s.spot_ids]
        s.centroids = np.array([(g[0], g[1]) for g in got], float)
        s.axis_t = np.array([g[2] for g in got], float)
    return sets


def spot_geometry(s):
    """``(N, 3)`` = image-px ``x, y`` + body-frame ``axis_t``, in ``s.spots`` row order.

    One array rather than two so the per-candidate stacking in ``iter_pairs`` stays a single
    ``concatenate``. ``match_features`` accepts the legacy 2-column form as well, in which case the
    axial features report "not computable" rather than guessing.
    """
    return np.column_stack([np.asarray(s.centroids, float),
                            np.asarray(s.axis_t, float)])


# ----------------------------------------------------------------------------- features
# Permutations used to estimate each pair's geometry null (env `GEOM_PERM`; 0 ablates the
# correction, leaving the `*_excess` columns equal to their raw counterparts). See `_geom_block`.
GEOM_PERM = int(os.environ.get("GEOM_PERM", "3"))
AXIAL_TOL = 0.05          # inlier band for the axial-shift model, in body lengths

FEATURE_NAMES = [
    "softchamfer_sum", "softchamfer_mean", "max_sim", "top3_mean",
    "frac>=.9", "frac>=.8", "frac>=.7", "frac>=.6", "count>=.8", "count>=.7",
    "qmax_std", "log_nq", "log_nc", "mutual_count", "mutual_frac", "geom_consistency",
    "ransac_frac",
    # --- added 2026-08-15, from the constellation Phase 0 screen (docs/tricks.md A11) ---
    "geom_valid",        # was geometry computable at all? Without it, "not computable" reaches the
                         # model as 0.0 -- the same value as "geometrically contradictory". MEASURED
                         # AFTERWARDS: that fires on 33% of true IMAGE-to-IMAGE pairs but only
                         # ~1.2% here, because a candidate is an INDIVIDUAL (spots pooled over its
                         # photos, median 10 mutual matches) and this function thresholds nothing.
                         # Kept because it is free and makes the distinction explicit; expect ~0.
    "ransac_excess",     # ransac_frac net of its own permutation null. The raw feature is
    "nbr_agree",         # confounded with the NUMBER of matches (a random assignment scores 0.69
    "nbr_agree_excess",  # at n=3 and 0.47 at n=6), and true pairs match more spots than false ones
                         # (6.27 vs 4.30), so the confound points the wrong way: screened AUROC
                         # 0.544 -> 0.621 for ransac, 0.424 -> 0.619 for neighbourhood agreement.
    "axial_inliers",     # fraction of matches agreeing on ONE head->tail shift. The pair-level form
                         # of the only relational statistic that beat appearance on the human
                         # correspondence verdicts (0.597 vs 0.484).
]


def _norm(x):
    x = np.asarray(x, np.float64)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _similarity_inliers(Q, C, iters=40, thr_frac=0.12, seed=0):
    """RANSAC inlier FRACTION under a similarity transform (scale+rot+translation).

    Fit sR+t from 2 sampled matched pairs (Q_i->C_i), apply to all query points, count how
    many land within ``thr_frac * median(candidate spacing)`` of their candidate match.
    Best over ``iters`` samples / len(Q). This scores whether the matched spots form the same
    *constellation* under a pose change -- the thing per-spot position alone can't see."""
    n = len(Q)
    if n < 3:
        return 0.0
    ref = float(np.median(pdist(C)))
    if ref < 1e-9:
        return 0.0
    thr = thr_frac * ref
    rng = np.random.default_rng(seed)
    best = 0
    for _ in range(iters):
        a, b = rng.choice(n, 2, replace=False)
        dq, dc = Q[b] - Q[a], C[b] - C[a]
        nq = np.hypot(*dq)
        if nq < 1e-9:
            continue
        s = np.hypot(*dc) / nq
        ang = np.arctan2(dc[1], dc[0]) - np.arctan2(dq[1], dq[0])
        R = s * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        pred = Q @ R.T + (C[a] - R @ Q[a])
        best = max(best, int((np.hypot(*(pred - C).T) < thr).sum()))
    return best / n


def _neighbour_agreement(Q, C, k=3) -> float:
    """Fraction of a matched spot's k nearest matched neighbours that stay its neighbours.

    Parameter-free: nothing is fitted, so no degrees of freedom are spent before the evidence is
    counted -- which matters because the median true pair here has three correspondences. Rows of
    ``Q`` and ``C`` are aligned by the correspondence, so the neighbour index sets compare directly.
    """
    from scipy.spatial.distance import squareform                        # noqa: PLC0415
    n = len(Q)
    if n < 3:
        return 0.0
    k = min(k, n - 1)
    nq = np.argsort(squareform(pdist(Q)), axis=1)[:, 1:k + 1]
    nc = np.argsort(squareform(pdist(C)), axis=1)[:, 1:k + 1]
    return float(np.mean([len(set(nq[i]) & set(nc[i])) / k for i in range(n)]))


def _axial_inliers(tq, tc, tol=AXIAL_TOL) -> float:
    """Inlier fraction of a 1-DOF model: ``t_c = t_q + b``, ``b`` the median shift.

    Both photos are already normalised head->tail, so the expected transform is near-identity and
    one robust parameter absorbs an anchor that landed early or late. Compare the image-frame
    RANSAC, which spends 4 degrees of freedom fitted from 2 sampled points.
    """
    ok = np.isfinite(tq) & np.isfinite(tc)
    if ok.sum() < 3:
        return 0.0
    dt = tc[ok] - tq[ok]
    return float(np.mean(np.abs(dt - np.median(dt)) <= tol))


def _geom_block(q_xy, c_xy, mutual, qbest, *, n_perm=None) -> tuple[float, ...]:
    """``(geom_valid, geom_consistency, ransac_frac, ransac_excess, nbr, nbr_excess, axial)``.

    **The excess columns are the point.** ``ransac_frac`` and neighbour agreement both ask "do the
    matched points agree?", and both questions get *easier the fewer points there are* -- any two
    points lie on some line, five rarely do. Measured over the dataset, a random assignment scores
    0.69 at three matches and 0.47 at six. True pairs match more spots than false ones (6.27 vs
    4.30), so the raw feature quietly rewards the sparse comparisons, which are mostly the wrong
    animals: permuting the correspondence leaves an AUROC of 0.377, i.e. *backwards*.

    The fix is a null built from the same amount of evidence: recompute the feature with this
    pair's own correspondence permuted, average, subtract. What remains is "better than luck, for a
    comparison with exactly this many matches". Screened AUROC 0.544 -> 0.621 (ransac) and
    0.424 -> 0.619 (neighbourhood). ``geom_consistency`` is a rank correlation and is nearly clean
    already (null 0.523), so it is left uncorrected. See docs/tricks.md A11.

    ``geom_valid`` is emitted separately so the 0.0 that an uncomputable pair receives is
    distinguishable from a genuine disagreement -- 33% of true pairs land there.
    """
    n_perm = GEOM_PERM if n_perm is None else n_perm
    if q_xy is None or len(mutual) < 3:
        return (0.0,) * 7
    q_xy, c_xy = np.asarray(q_xy, float), np.asarray(c_xy, float)
    qp, cp = q_xy[mutual], c_xy[qbest[mutual]]
    Q, C = qp[:, :2], cp[:, :2]

    geom = 0.0
    dq, dc = pdist(Q), pdist(C)
    if dq.std() > 1e-9 and dc.std() > 1e-9:
        r = spearmanr(dq, dc).correlation                    # scale/rot/translation-invariant
        geom = 0.0 if np.isnan(r) else float(r)

    ransac, nbr = _similarity_inliers(Q, C), _neighbour_agreement(Q, C)
    axial = (_axial_inliers(qp[:, 2], cp[:, 2]) if qp.shape[1] > 2 else 0.0)

    r_ex, n_ex = ransac, nbr
    if n_perm > 0:
        rng = np.random.default_rng(len(mutual))             # deterministic given the pair's size
        nulls = [(lambda p: (_similarity_inliers(Q, C[p]), _neighbour_agreement(Q, C[p])))(
                 rng.permutation(len(C))) for _ in range(n_perm)]
        r_ex = ransac - float(np.mean([a for a, _ in nulls]))
        n_ex = nbr - float(np.mean([b for _, b in nulls]))
    return (1.0, geom, ransac, r_ex, nbr, n_ex, axial)


def match_features(q_emb, c_emb, q_xy=None, c_xy=None) -> np.ndarray:
    """Summarize the match between a query image's spots and a candidate individual's spots.

    ``q_emb`` (nq, d), ``c_emb`` (nc, d) are L2-normalized. Returns the FEATURE_NAMES vector.
    The current hand-coded voting score is just ``softchamfer_sum`` (feature 0) -- everything
    else is extra signal the aggregator can learn to weigh.

    ``q_xy`` / ``c_xy`` are ``(N, 3)`` from :func:`spot_geometry` -- image ``x, y`` plus body-frame
    ``axis_t``. A legacy ``(N, 2)`` array still works; only ``axial_inliers`` goes dark.
    """
    S = q_emb @ c_emb.T                                      # (nq, nc) cosine
    nq, nc = S.shape
    qmax = S.max(1)                                          # best match per query spot
    top3 = np.sort(qmax)[::-1][:3]
    qbest = S.argmax(1)                                      # each query spot's best candidate spot
    cbest = S.argmax(0)                                      # each candidate spot's best query spot
    mutual = np.array([i for i in range(nq) if cbest[qbest[i]] == i], dtype=int)

    valid, geom, ransac, r_ex, nbr, n_ex, axial = _geom_block(q_xy, c_xy, mutual, qbest)

    return np.array([
        qmax.sum(), qmax.mean(), qmax.max(), top3.mean(),
        (qmax >= 0.9).mean(), (qmax >= 0.8).mean(), (qmax >= 0.7).mean(), (qmax >= 0.6).mean(),
        (qmax >= 0.8).sum(), (qmax >= 0.7).sum(),
        qmax.std(), np.log1p(nq), np.log1p(nc),
        len(mutual), len(mutual) / nq, geom, ransac,
        valid, r_ex, nbr, n_ex, axial,
    ], float)


def iter_pairs(sets, images, *, gallery_images=None, neg_per_query=None, seed=0, use_geom=True):
    """Yield ``(q_emb, c_emb, q_xy, c_xy, label, q_img, c_label)`` for each (query image,
    candidate individual) pair. Gallery = OTHER images (leave-query-out); ``neg_per_query``
    subsamples negatives for training, ``None`` keeps all candidates (eval / full ranking).
    Shared by the summary-feature and set-record aggregators.

    ``gallery_images`` adds DISTRACTOR individuals to the candidate list without making them
    queries. Queries always come from ``images``; candidates come from ``images +
    gallery_images``. Passing the train fold here is what makes the eval gallery the size of the
    real population instead of the size of the eval fold (~17 individuals) -- see
    ``crossval_aggregator(full_gallery=True)``. There is no leakage: ``get_cv_folds`` holds eval
    individuals out of train entirely, so a distractor can never be a query's correct answer.
    """
    rng = np.random.default_rng(seed)
    pool = list(dict.fromkeys(list(images) + list(gallery_images or [])))
    by_label: dict[str, list[int]] = {}
    for i in pool:
        by_label.setdefault(sets[i].label, []).append(i)
    emb = {i: _norm(sets[i].spots) for i in pool}
    xy = {i: (spot_geometry(sets[i]) if use_geom else None) for i in pool}
    all_labels = list(by_label)

    # Stack each candidate's spots ONCE. With a full distractor gallery every candidate is scored
    # against every query, so rebuilding these arrays inside the loop dominates the runtime.
    cache = {c: (np.concatenate([emb[g] for g in gs]),
                 np.concatenate([xy[g] for g in gs]) if use_geom else None)
             for c, gs in by_label.items()}

    for q in images:
        yq = sets[q].label
        if not any(g != q for g in by_label[yq]):            # no gallery positive -> unusable query
            continue
        cand = [c for c in all_labels if any(g != q for g in by_label[c])]
        if neg_per_query is not None:
            negs = [c for c in cand if c != yq]
            rng.shuffle(negs)
            cand = [yq] + negs[:neg_per_query]
        for c in cand:
            if c == yq:                                      # own label: drop the query itself
                gal = [g for g in by_label[c] if g != q]
                c_emb = np.concatenate([emb[g] for g in gal])
                c_xy = np.concatenate([xy[g] for g in gal]) if use_geom else None
            else:                                            # q is never in another label's pool
                c_emb, c_xy = cache[c]
            yield emb[q], c_emb, xy[q], c_xy, int(c == yq), q, c


def build_pairs(sets, images, *, gallery_images=None, neg_per_query=None, seed=0, use_geom=True):
    """One row per (query, candidate): the summary FEATURE vector + label. See ``iter_pairs``."""
    X, y, qids, clabels = [], [], [], []
    for qe, ce, qx, cx, lab, q, c in iter_pairs(sets, images, gallery_images=gallery_images,
                                                neg_per_query=neg_per_query,
                                                seed=seed, use_geom=use_geom):
        X.append(match_features(qe, ce, qx, cx))
        y.append(lab); qids.append(q); clabels.append(c)
    return np.array(X), np.array(y), np.array(qids), np.array(clabels, dtype=object)


# ----------------------------------------------------------------------------- model
class Aggregator(nn.Module):
    """``hidden=0`` -> plain logistic regression (the champion); ``hidden>0`` -> an MLP.

    ``n_layers`` stacks that many hidden blocks, so ``hidden=64, n_layers=3`` is the deep
    fully-connected variant. Depth is the thing under test: the summary features are only 17-dim
    and the champion is linear, so more capacity here is expected to overfit -- that is the
    hypothesis this knob exists to falsify.
    """

    def __init__(self, n_feat, hidden=0, dropout=0.1, n_layers=1):
        super().__init__()
        if not hidden:
            self.net = nn.Linear(n_feat, 1)
            return
        blocks, prev = [], n_feat
        for _ in range(max(1, n_layers)):
            blocks += [nn.Linear(prev, hidden), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden
        self.net = nn.Sequential(*blocks, nn.Linear(prev, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_aggregator(X, y, *, hidden=0, n_layers=1, dropout=0.1, epochs=400, lr=0.05,
                     weight_decay=1e-4, seed=0, on_epoch=None):
    """Fit an aggregator (hidden=0 -> logistic regression). Standardizes features, balances
    the loss by pos_weight. Returns (model, (mean, std)).

    ``on_epoch(ep, model, scaler, train_loss)`` is called after each step, mirroring
    ``train_strict_e2e``, so a sweep can log a metric curve instead of one final number.
    """
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    model = Aggregator(X.shape[1], hidden=hidden, dropout=dropout, n_layers=n_layers)
    n_pos = max(int(y.sum()), 1)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y) - n_pos) / n_pos], dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(epochs):
        opt.zero_grad(); loss = lossf(model(Xt), yt); loss.backward(); opt.step()
        if on_epoch is not None:
            on_epoch(ep, model, (mu, sd), float(loss.detach()))
    return model, (mu, sd)


def _logit(model, scaler, X):
    mu, sd = scaler
    with torch.no_grad():
        return model(torch.tensor((X - mu) / sd, dtype=torch.float32)).numpy()


def _prob(model, scaler, X):
    return 1.0 / (1.0 + np.exp(-_logit(model, scaler, X)))


def platt_fit(logit, y, epochs=300, lr=0.05):
    """Platt scaling: fit sigmoid(a*logit + b) so probabilities are calibrated. -> (a, b)."""
    z = torch.tensor(logit, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    a = torch.ones(1, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr); lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad(); lossf(a * z + b, yt).backward(); opt.step()
    return a.item(), b.item()


def rank_eval(score_fn, sets, eval_images, ks=(1, 5, 10), use_geom=True, gallery_images=None):
    """Rank candidate individuals per query by ``score_fn(X)``. Returns recall@k plus a details
    dict incl. per-query top1-top2 ``margins`` and ``correct`` (for confidence analysis).

    ``gallery_images`` widens the candidate list with distractor individuals (see ``iter_pairs``);
    ``details["n_candidates"]`` reports the median gallery size the numbers were measured at,
    because recall@k is only interpretable next to it."""
    X, y, qids, clab = build_pairs(sets, eval_images, gallery_images=gallery_images,
                                   neg_per_query=None, use_geom=use_geom)
    scores = np.asarray(score_fn(X))
    hits = {k: 0 for k in ks}; n = 0; margins = []; correct = []
    for q in np.unique(qids):
        m = qids == q
        s_sorted = np.sort(scores[m])[::-1]
        ranked = clab[m][np.argsort(-scores[m])]
        rank = int(np.where(ranked == sets[q].label)[0][0]) + 1
        for k in ks:
            hits[k] += int(rank <= k)
        margins.append(float(s_sorted[0] - s_sorted[1]) if len(s_sorted) > 1 else float(s_sorted[0]))
        correct.append(int(rank == 1))
        n += 1
    out = {f"recall@{k}": (hits[k] / n if n else float("nan")) for k in ks}
    out["n_queries"] = n
    n_cand = float(np.median([int((qids == q).sum()) for q in np.unique(qids)])) if n else float("nan")
    out["n_candidates"] = n_cand
    return out, dict(X=X, y=y, qids=qids, clab=clab, scores=scores, n_candidates=n_cand,
                     margins=np.array(margins), correct=np.array(correct))


def crossval_aggregator(sets, *, k=5, seed=0, hidden=0, n_layers=1, dropout=0.1,
                        weight_decay=1e-4, neg_per_query=30, use_geom=True, full_gallery=False):
    """5-fold CV of the learned aggregator vs raw soft-chamfer voting. Reports per-fold and
    mean+-std R@1, plus a margin->accuracy confidence check pooled over folds.

    ``full_gallery`` ranks each eval query against EVERY individual in the dataset (the train
    fold joins as distractors) instead of only the ~n/k individuals in its own eval fold. The
    fold-only number answers "can it tell these 17 apart"; the full-gallery number answers "can
    it find one animal in the whole population", which is the deployment task. Training is
    identical either way -- only what the query is ranked against changes.
    """
    folds = d.get_cv_folds(sets, k=k, seed=seed)
    raws, learns, r5 = [], [], []
    margin_correct = []
    n_cands = []
    logger.info(f" gallery: {'FULL population (train fold as distractors)' if full_gallery else 'eval fold only'}")
    logger.info(f" {'fold':>4} | {'cands':>5} | {'raw R@1':>7} | {'learned R@1':>11} | {'R@5':>5} | {'dR@1':>6}")
    logger.info(" " + "-" * 56)
    for fi, (tr, ev) in enumerate(folds):
        train_imgs = [i for i in tr if not sets[i].is_synth]
        Xtr, ytr, _, _ = build_pairs(sets, train_imgs, neg_per_query=neg_per_query, seed=seed, use_geom=use_geom)
        model, scaler = train_aggregator(Xtr, ytr, hidden=hidden, n_layers=n_layers,
                                         dropout=dropout, weight_decay=weight_decay, seed=seed)
        gal = train_imgs if full_gallery else None
        raw, _ = rank_eval(lambda X: X[:, 0], sets, ev, use_geom=use_geom, gallery_images=gal)
        learned, det = rank_eval(lambda X: _prob(model, scaler, X), sets, ev, use_geom=use_geom,
                                 gallery_images=gal)
        raws.append(raw["recall@1"]); learns.append(learned["recall@1"]); r5.append(learned["recall@5"])
        n_cands.append(learned["n_candidates"])
        margin_correct += list(zip(det["margins"], det["correct"]))
        logger.info(f" {fi:>4} | {learned['n_candidates']:>5.0f} | {raw['recall@1']:>7.3f} | "
              f"{learned['recall@1']:>11.3f} | "
              f"{learned['recall@5']:>5.3f} | {learned['recall@1']-raw['recall@1']:>+6.3f}")
    logger.info(" " + "-" * 56)
    logger.info(f" median candidates per query: {np.median(n_cands):.0f}  "
          f"(recall@k means nothing without this)")
    logger.info(f" MEAN | {np.mean(raws):>7.3f} | {np.mean(learns):>11.3f} | {np.mean(r5):>5.3f} | "
          f"{np.mean(learns)-np.mean(raws):>+6.3f}")
    logger.info(f"      raw  {np.mean(raws):.3f}+-{np.std(raws):.3f}   learned {np.mean(learns):.3f}+-{np.std(learns):.3f}")

    mc = np.array(margin_correct); o = np.argsort(mc[:, 0]); t = len(o) // 3
    logger.info(" confidence (top1-top2 margin -> identification accuracy):")
    for lab, sl in [("low   ", o[:t]), ("mid   ", o[t:2*t]), ("high  ", o[2*t:])]:
        logger.info(f"   {lab} margin: accuracy {mc[sl, 1].mean():.3f}   (n={len(sl)})")
    return raws, learns


def calibration(y, p, bins=5):
    """Reliability: predicted prob vs actual same-rate, in ``bins`` quantile buckets."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        if m.any():
            rows.append((p[m].mean(), y[m].mean(), int(m.sum())))
    return rows


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-gallery", action="store_true",
                    help="rank each eval query against EVERY individual (deployment-sized "
                         "gallery) instead of only its own eval fold (~n/k individuals)")
    ap.add_argument("--k", type=int, default=5, help="CV folds (default 5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interactive", action="store_true",
                    help="drop into breakpoint() at the end with sets/model/scaler/det live")
    args = ap.parse_args()

    sets = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))

    logger.info("=" * 50)
    logger.info(" 5-FOLD CV: learned aggregator vs raw voting")
    logger.info("=" * 50)
    crossval_aggregator(sets, k=args.k, seed=args.seed, hidden=0,
                        full_gallery=args.full_gallery)

    # ---- fold-0 detail: learned weights + Platt-calibrated confidence ----
    logger.info("=" * 50)
    logger.info(" fold-0 detail")
    logger.info("=" * 50)
    tr, ev = d.get_cv_folds(sets, k=args.k, seed=args.seed)[0]
    train_imgs = [i for i in tr if not sets[i].is_synth]
    Xtr, ytr, qtr, _ = build_pairs(sets, train_imgs, neg_per_query=30, seed=0)
    # carve a calibration slice OUT of train (by query) so Platt isn't fit on eval
    uq = np.unique(qtr); rng = np.random.default_rng(0); rng.shuffle(uq)
    cal_q = set(uq[: len(uq) // 5])
    fit_m = np.array([q not in cal_q for q in qtr])
    model, scaler = train_aggregator(Xtr[fit_m], ytr[fit_m], hidden=0, seed=0)
    a, b = platt_fit(_logit(model, scaler, Xtr[~fit_m]), ytr[~fit_m])   # calibrate on held-out slice

    w = model.net.weight.detach().numpy().ravel()
    logger.info(" learned weights (standardized; how each feature votes):")
    for nm, wt in sorted(zip(FEATURE_NAMES, w), key=lambda t: -abs(t[1])):
        logger.info(f"   {nm:16s} {wt:+.3f}")

    _, det = rank_eval(lambda X: _prob(model, scaler, X), sets, ev,
                       gallery_images=train_imgs if args.full_gallery else None)
    p_raw = _prob(model, scaler, det["X"])
    p_cal = 1.0 / (1.0 + np.exp(-(a * _logit(model, scaler, det["X"]) + b)))
    logger.info(" calibration on eval (pred prob -> actual same-rate):")
    logger.info(f"   before Platt: {[f'{pm:.2f}->{ym:.2f}' for pm, ym, _ in calibration(det['y'], p_raw)]}")
    logger.info(f"   after  Platt: {[f'{pm:.2f}->{ym:.2f}' for pm, ym, _ in calibration(det['y'], p_cal)]}")
    if args.interactive:
        breakpoint()   # live: sets, model, scaler, a, b, det
