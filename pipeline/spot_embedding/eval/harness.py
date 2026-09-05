"""The shared evaluation harness — every matcher is scored through this one path.

For each CV fold it computes, from the matcher's ``(Q, G)`` distance matrices:

* **closed-set retrieval** — rank-1 / rank-5 / mAP over the closed queries vs. the gallery;
* **verification** — ROC-AUC and TPR@1%FPR over all query×gallery pairs (positive = same label);
* **open-set** — AUROC separating closed (known) from open (novel) queries by best gallery sim;
* **selective prediction** — a risk–coverage curve using top-1 similarity as confidence.

``evaluate`` scores a fixed matcher across all folds. ``evaluate_per_fold`` takes a factory that
builds a (freshly trained) matcher per fold — the path the learned Phase-3 models use, since they
must train on each fold's train split before embedding that fold's gallery/query. Both share
:func:`score_fold` and :func:`aggregate`, so learned and non-learned models are scored identically.
"""
from __future__ import annotations

import numpy as np

from ..models.base import Matcher
from . import metrics as M


def score_fold(matcher: Matcher, fold, id2ss, *, fpr_target: float = 0.01):
    """One fold → (row dict, closed-query rank-1 correctness, closed-query top-1 similarity)."""
    gallery = [id2ss[i] for i in fold.gallery_ids]
    q_closed = [id2ss[i] for i in fold.query_closed_ids]
    q_open = [id2ss[i] for i in fold.query_open_ids]
    g_labels = [g.label for g in gallery]

    row: dict = {"fold": fold.fold, "n_gallery": len(gallery),
                 "n_closed": len(q_closed), "n_open": len(q_open)}
    correct = conf = None
    known_best = np.empty(0)

    if q_closed and gallery:
        dist = matcher.distance_matrix(q_closed, gallery)
        sim = -dist
        qc_labels = [q.label for q in q_closed]
        row.update(M.closed_set_metrics(dist, qc_labels, g_labels, ks=(1, 5)))
        same = (np.asarray(qc_labels)[:, None] == np.asarray(g_labels)[None, :]).ravel()
        row["verify_auc"] = M.roc_auc(sim.ravel(), same)
        row["verify_tpr@fpr"] = M.tpr_at_fpr(sim.ravel(), same, fpr_target)
        known_best = sim.max(axis=1) if sim.size else np.empty(0)
        nn = np.argmin(dist, axis=1)
        correct = np.asarray(g_labels)[nn] == np.asarray(qc_labels)
        conf = known_best

    if q_open and gallery:
        novel_best = (-matcher.distance_matrix(q_open, gallery)).max(axis=1)
        row["openset_auroc"] = M.openset_auroc(known_best, novel_best)
    else:
        row["openset_auroc"] = float("nan")

    return row, correct, conf


def aggregate(matcher_name: str, per_fold: list, correct_parts: list, conf_parts: list) -> dict:
    keys = ["rank1", "rank5", "mAP", "verify_auc", "verify_tpr@fpr", "openset_auroc"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in per_fold if k in r and not _isnan(r[k])]
        agg[k] = float(np.mean(vals)) if vals else float("nan")

    parts = [c for c in correct_parts if c is not None and len(c)]
    if parts:
        correct = np.concatenate(parts)
        conf = np.concatenate([c for c in conf_parts if c is not None and len(c)])
        cov, risk, aurc = M.risk_coverage(correct, conf)
        agg["aurc"] = aurc
        rc = {"coverage": cov.tolist(), "risk": risk.tolist()}
    else:
        agg["aurc"] = float("nan")
        rc = {"coverage": [], "risk": []}

    return {"matcher": matcher_name, "n_folds": len(per_fold),
            "aggregate": agg, "folds": per_fold, "risk_coverage": rc}


def evaluate(matcher: Matcher, spotsets: list, folds: list, *, fpr_target: float = 0.01) -> dict:
    """Score a fixed matcher across all folds."""
    id2ss = {ss.salamander_id: ss for ss in spotsets}
    rows, corrects, confs = [], [], []
    for fd in folds:
        row, c, cf = score_fold(matcher, fd, id2ss, fpr_target=fpr_target)
        rows.append(row); corrects.append(c); confs.append(cf)
    return aggregate(getattr(matcher, "name", type(matcher).__name__), rows, corrects, confs)


def evaluate_per_fold(make_matcher, spotsets: list, folds: list, *, name: str,
                      fpr_target: float = 0.01) -> dict:
    """Score a matcher that is (re)built per fold. ``make_matcher(fold) -> Matcher``."""
    id2ss = {ss.salamander_id: ss for ss in spotsets}
    rows, corrects, confs = [], [], []
    for fd in folds:
        matcher = make_matcher(fd)
        row, c, cf = score_fold(matcher, fd, id2ss, fpr_target=fpr_target)
        rows.append(row); corrects.append(c); confs.append(cf)
    return aggregate(name, rows, corrects, confs)


def _isnan(x) -> bool:
    return isinstance(x, float) and x != x
