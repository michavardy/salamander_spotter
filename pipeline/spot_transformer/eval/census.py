"""Open-set / census evaluation for the learned spot aggregator.

The re-ID track (``aggregator*.py``) ranks candidate individuals for a query and reports R@k --
a *human-in-the-loop* metric (right animal in the top-k shortlist). A **fully automated
population census** is a different question: for each captured animal decide *match an existing
individual* vs *a brand-new individual*, via a score threshold. There, **precision is king** --
a false MATCH collapses two animals into one profile and **deflates** the population estimate,
which mark-recapture math cannot undo; a false NEW only inflates (a re-sight counted as new),
which capture-probability models tolerate. So we optimize **F_beta with beta=0.5** (precision
weighted 2x recall) and sweep the decision threshold.

This module is pure metrics + the open-set data plumbing; the model/scoring lives in
``aggregator_set.py`` and the driver in ``sweep_set.py``.
"""
from __future__ import annotations

import numpy as np

try:  # run as script (path[0] = this dir) OR imported as a package
    import data as d
    from aggregator import _norm, spot_geometry as _spot_geometry
    from aggregator_set import match_records
except ModuleNotFoundError:
    from pipeline.spot_transformer import data as d
    from pipeline.spot_transformer.aggregator import _norm, spot_geometry as _spot_geometry
    from pipeline.spot_transformer.aggregator_set import match_records


# ============================================================ pairwise precision / F_beta
def fbeta(precision, recall, beta=0.5):
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def pairwise_curve(scores, same):
    """Precision/recall of the match decision as the threshold sweeps every score.

    ``scores`` = P(same individual) for each (query, candidate) pair; ``same`` = 0/1 truth.
    Returns thresholds (each = 'accept pairs with score >= t') with precision, recall, and the
    count accepted. This is 'pairwise precision': of the pairs the model calls a match, how many
    truly are -- the learned analogue of the raw ``naive_match_all`` cutoff table."""
    scores = np.asarray(scores, float); same = np.asarray(same).astype(bool)
    P = int(same.sum())
    o = np.argsort(-scores)
    s, sm = scores[o], same[o]
    tp = np.cumsum(sm); fp = np.cumsum(~sm)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(P, 1)
    return dict(thr=s, precision=prec, recall=rec, n_accept=np.arange(1, len(s) + 1), n_pos=P)


def average_precision(scores, same):
    """Threshold-free area under the precision-recall curve (== sklearn average_precision)."""
    c = pairwise_curve(scores, same)
    r = np.concatenate([[0.0], c["recall"]]); p = np.concatenate([[1.0], c["precision"]])
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


def best_fbeta(scores, same, beta=0.5):
    """Best F_beta over all thresholds + the threshold and (precision, recall) that achieve it."""
    c = pairwise_curve(scores, same)
    f = np.array([fbeta(p, r, beta) for p, r in zip(c["precision"], c["recall"])])
    if not len(f):
        return dict(f=float("nan"), thr=float("nan"), precision=float("nan"), recall=float("nan"))
    i = int(np.argmax(f))
    return dict(f=float(f[i]), thr=float(c["thr"][i]),
                precision=float(c["precision"][i]), recall=float(c["recall"][i]))


def fbeta_at(scores, same, thr, beta=0.5):
    """F_beta / precision / recall at a FIXED threshold (chosen on train, applied to test)."""
    scores = np.asarray(scores, float); same = np.asarray(same).astype(bool)
    acc = scores >= thr
    tp = int((acc & same).sum()); fp = int((acc & ~same).sum()); P = int(same.sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / P if P else float("nan")
    return dict(f=fbeta(prec, rec, beta) if (tp + fp) and P else float("nan"),
                precision=prec, recall=rec, thr=float(thr), n_accept=int(acc.sum()))


# ============================================================ open-set census (population count)
def census_confusion(top1, top1_correct, is_known, thr):
    """Per-query census decision at threshold ``thr`` on the top-1 score.

    Decision: top1 >= thr -> 'MATCH to top-1 individual', else -> 'NEW individual'. Ground
    truth per query: a *known* re-sight (its individual is in the gallery) or a *novel* animal
    (not in the gallery -> the only correct action is NEW).

        known + match + correct id   -> TP  (a caught, correctly-linked re-sight)
        known + match + wrong id     -> FP  (false match: linked to the WRONG animal)
        known + new                  -> FN  (missed re-sight -> phantom individual -> INFLATE)
        novel + new                  -> TN  (correctly a new animal)
        novel + match                -> FP  (false match: a new animal absorbed -> DEFLATE)

    ``precision`` = of all MATCH calls, the fraction that are a correct re-sight link.
    ``population count bias`` decomposes as ``FN`` (phantom new individuals, inflation) minus
    ``fp_novel`` (real new animals absorbed into existing profiles, deflation)."""
    top1 = np.asarray(top1, float); tc = np.asarray(top1_correct).astype(bool)
    known = np.asarray(is_known).astype(bool)
    match = top1 >= thr
    tp = int((known & match & tc).sum())
    fp_known = int((known & match & ~tc).sum())
    fn = int((known & ~match).sum())
    tn = int((~known & ~match).sum())
    fp_novel = int((~known & match).sum())
    fp = fp_known + fp_novel
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fp_known + fn) if (tp + fp_known + fn) else float("nan")   # recall over known
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, fp_known=fp_known, fp_novel=fp_novel,
                precision=prec, recall=rec, thr=float(thr),
                n_known=int(known.sum()), n_novel=int((~known).sum()),
                inflation=fn, deflation=fp_novel, count_bias=fn - fp_novel)


def census_sweep(top1, top1_correct, is_known, beta=0.5, n_thr=200):
    """Sweep the census threshold; return the per-threshold curve and the F_beta-optimal point."""
    top1 = np.asarray(top1, float)
    grid = np.unique(np.quantile(top1, np.linspace(0, 1, n_thr)))
    rows = [census_confusion(top1, top1_correct, is_known, t) for t in grid]
    fs = np.array([fbeta(r["precision"], r["recall"], beta)
                   if np.isfinite(r["precision"]) and np.isfinite(r["recall"]) else np.nan
                   for r in rows])
    best = rows[int(np.nanargmax(fs))] if np.isfinite(fs).any() else rows[0]
    best = dict(best, f=float(np.nanmax(fs)) if np.isfinite(fs).any() else float("nan"))
    return dict(thr=grid, precision=np.array([r["precision"] for r in rows]),
                recall=np.array([r["recall"] for r in rows]), fbeta=fs, rows=rows), best


# ============================================================ decomposed open-set metrics
#
# WHY these exist. ``census_confusion`` scores the fully-automated census decision, and its
# precision/recall deliberately ignore TN (``novel + new`` -- correctly recognising an animal we
# have never seen). So a model that is PERFECT at spotting new individuals gets no credit for it,
# and census F0.5 alone cannot answer "how good are we at the new-vs-known call?".
#
# That matters because the task is really two decisions of very different difficulty:
#   verification -- "is photo X the same animal as photo Y?"   (comparative, tractable)
#   novelty      -- "is this animal in the database AT ALL?"   (absolute, needs a calibrated
#                                                               threshold; much harder)
# Reporting one blended number hides which of the two is actually failing. These separate them.

def auroc(scores, positive):
    """Threshold-free AUROC via the rank statistic (no sklearn dependency).

    Threshold-free is the point: it measures whether the score *orders* the two groups at all,
    independently of where a threshold happens to sit. A low F-beta with a high AUROC means the
    signal is there and the threshold is miscalibrated -- a completely different fix from a low
    AUROC, which means the score carries no information.
    """
    s = np.asarray(scores, float)
    p = np.asarray(positive).astype(bool)
    n_pos, n_neg = int(p.sum()), int((~p).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties, so tied scores cannot fake separation
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[p].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def novelty_metrics(top1, is_known, thr):
    """The known-vs-novel decision ALONE, with TN counted -- what census F0.5 is blind to.

    ``novel_recall`` (specificity) is the number the census metric never shows: of the animals
    that genuinely are new, how many did we correctly enrol as new. ``balanced_acc`` averages it
    with known-recall so neither class can be ignored by predicting one label everywhere.
    """
    t = np.asarray(top1, float)
    known = np.asarray(is_known).astype(bool)
    match = t >= thr
    known_recall = float((match & known).sum() / max(int(known.sum()), 1))
    novel_recall = float((~match & ~known).sum() / max(int((~known).sum()), 1))
    return dict(known_recall=known_recall, novel_recall=novel_recall,
                balanced_acc=0.5 * (known_recall + novel_recall),
                openset_auroc=auroc(t, known),
                n_known=int(known.sum()), n_novel=int((~known).sum()))


# ============================================================ selective / risk-coverage
def risk_coverage(confidence, correct):
    """Answer only the most-confident queries: accuracy vs coverage (fraction answered).

    Sort by ``confidence`` descending; at each cut, coverage = fraction answered, accuracy =
    correctness on the answered subset. ``aurc`` (area under risk=1-accuracy vs coverage) is a
    threshold-free summary -- lower is better. Directly answers 'if I drop the least-confident
    X%, how accurate is the rest?'."""
    confidence = np.asarray(confidence, float); correct = np.asarray(correct).astype(float)
    o = np.argsort(-confidence)
    c = correct[o]
    cov = np.arange(1, len(c) + 1) / len(c)
    acc = np.cumsum(c) / np.arange(1, len(c) + 1)
    risk = 1 - acc
    aurc = float(np.sum(np.diff(cov) * (risk[1:] + risk[:-1]) / 2))       # trapezoid (numpy 2 safe)
    return dict(coverage=cov, accuracy=acc, risk=risk, aurc=aurc)


# ============================================================ image-quality markers
QUALITY_FIELDS = ["overall_quality", "blur_quality", "lighting_quality",
                  "spot_extraction_quality", "body_extraction_quality"]


def load_image_quality(db_path=None) -> dict[str, dict]:
    """``salamander_id -> {quality field: value}`` from the dataset's ``image_quality`` table
    (blur / glare / lighting / spot-extraction markers). Empty dict if the table is absent."""
    import duckdb
    con = duckdb.connect(str(db_path or d.DB_PATH), read_only=True)
    try:
        if "image_quality" not in [r[0] for r in con.execute("SHOW TABLES").fetchall()]:
            return {}
        cols = [c for c in QUALITY_FIELDS + ["glare_frac", "blur_score", "n_spots"]
                if c in [r[1] for r in con.execute("PRAGMA table_info('image_quality')").fetchall()]]
        df = con.execute(f"SELECT salamander_id, {', '.join(cols)} FROM image_quality").df()
    finally:
        con.close()
    return {r["salamander_id"]: {c: r[c] for c in cols} for _, r in df.iterrows()}


def quality_strata(sids, correct, quality, field="overall_quality", bins=3):
    """R@1 stratified into quality terciles (does bad blur/glare/extraction hurt matching?)."""
    vals = np.array([quality.get(s, {}).get(field, np.nan) for s in sids], float)
    correct = np.asarray(correct, float)
    ok = np.isfinite(vals)
    vals, correct, sids = vals[ok], correct[ok], np.asarray(sids)[ok]
    if len(vals) < bins:
        return []
    edges = np.quantile(vals, np.linspace(0, 1, bins + 1))
    out = []
    for lo, hi, name in zip(edges[:-1], edges[1:], ["low", "mid", "high"][:bins]):
        m = (vals >= lo) & (vals <= hi)
        if m.any():
            out.append(dict(bin=name, lo=float(lo), hi=float(hi),
                            accuracy=float(correct[m].mean()), n=int(m.sum())))
    return out


# ============================================================ open-set fold plumbing
def make_openset_split(sets, eval_idx, novel_frac=0.35, seed=0):
    """Split a fold's eval individuals into KNOWN (seed the gallery) and NOVEL (query-only,
    never in the gallery -> the system must reject them). Returns ``(gallery_imgs, query_imgs)``
    over REAL images only. Every eval query is scored against the KNOWN gallery; a known query's
    true individual is present (a findable re-sight), a novel query's is not."""
    real = [i for i in eval_idx if not sets[i].is_synth]
    by_label: dict[str, list[int]] = {}
    for i in real:
        by_label.setdefault(sets[i].label, []).append(i)
    labels = sorted(by_label)
    rng = np.random.default_rng(seed)
    labels = [labels[i] for i in rng.permutation(len(labels))]
    n_novel = int(round(novel_frac * len(labels)))
    novel = set(labels[:n_novel]); known = set(labels[n_novel:])
    gallery = [i for l in known for i in by_label[l]]
    queries = [i for l in (known | novel) for i in by_label[l]]
    return gallery, queries


def iter_pairs_openset(sets, gallery_imgs, query_imgs, *, use_geom=True):
    """Yield ``(q_emb, c_emb, q_xy, c_xy, same, q, c_label)`` for every (query image, gallery
    individual) pair. Gallery candidates = individuals present in ``gallery_imgs`` only; a
    query's own image is left out of its candidate (leave-query-out), so a known re-sight can
    still match its OTHER gallery images while a novel query matches none."""
    by_label: dict[str, list[int]] = {}
    for i in gallery_imgs:
        by_label.setdefault(sets[i].label, []).append(i)
    pool = set(gallery_imgs) | set(query_imgs)
    emb = {i: _norm(sets[i].spots) for i in pool}
    # spot_geometry, not `.centroids`: the (N, 3) form carries body-frame axis_t alongside the
    # image-px centroid, which `match_features` needs for `axial_inliers`.
    xy = {i: (_spot_geometry(sets[i]) if use_geom else None) for i in pool}
    cand_labels = list(by_label)
    for q in query_imgs:
        yq = sets[q].label
        for c in cand_labels:
            gal = [g for g in by_label[c] if g != q]
            if not gal:
                continue
            c_emb = np.concatenate([emb[g] for g in gal])
            c_xy = np.concatenate([xy[g] for g in gal]) if use_geom else None
            yield emb[q], c_emb, xy[q], c_xy, int(c == yq), q, c


def build_openset_records(sets, gallery_imgs, query_imgs, *, use_geom=True, n_bands=0):
    """Open-set match records + ``true_by_q`` (query -> its true label) for the census eval.

    ``n_bands>0`` returns axis-ordered band grids instead of raw record sets, so the axial CNN is
    evaluated under exactly this protocol rather than a parallel one."""
    if n_bands:
        try:
            from aggregator_set import axial_bands
        except ModuleNotFoundError:
            from pipeline.spot_transformer.aggregator_set import axial_bands
    recs, qids, clab = [], [], []
    for qe, ce, qx, cx, _same, q, c in iter_pairs_openset(sets, gallery_imgs, query_imgs, use_geom=use_geom):
        r = match_records(qe, ce, qx, cx)
        if n_bands:
            r = axial_bands(r, sets[q].axis_t, n_bands)
        recs.append(r); qids.append(q); clab.append(c)
    true_by_q = {q: sets[q].label for q in query_imgs}
    return recs, np.array(qids), np.array(clab, dtype=object), true_by_q


def build_openset_features(sets, gallery_imgs, query_imgs, *, use_geom=True):
    """Open-set summary features (``aggregator.FEATURE_NAMES``) for the logreg baseline under the
    census protocol."""
    try:
        from aggregator import match_features
    except ModuleNotFoundError:
        from pipeline.spot_transformer.aggregator import match_features
    X, qids, clab = [], [], []
    for qe, ce, qx, cx, _same, q, c in iter_pairs_openset(sets, gallery_imgs, query_imgs, use_geom=use_geom):
        X.append(match_features(qe, ce, qx, cx)); qids.append(q); clab.append(c)
    return np.array(X), np.array(qids), np.array(clab, dtype=object)
