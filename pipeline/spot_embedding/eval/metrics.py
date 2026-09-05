"""Retrieval / verification / open-set metrics — pure numpy, no sklearn.

Conventions:

* ``dist`` is a ``(Q, G)`` distance matrix (smaller = more similar).
* verification / open-set take **similarity** scores (larger = more similar); pass
  ``-dist`` if you only have distances.
* ROC-AUC is the tie-aware Mann-Whitney statistic, so a perfect separator → 1.0 and random
  scores → ~0.5 exactly in expectation.
"""
from __future__ import annotations

import numpy as np


# --- retrieval --------------------------------------------------------------
def closed_set_metrics(
    dist: np.ndarray,
    query_labels,
    gallery_labels,
    ks=(1, 5),
) -> dict:
    """rank-k and mAP for closed-set retrieval (every query has a gallery match)."""
    q = np.asarray(query_labels)
    g = np.asarray(gallery_labels)
    if dist.shape != (len(q), len(g)):
        raise ValueError(f"dist {dist.shape} != (Q={len(q)}, G={len(g)})")
    if len(q) == 0 or len(g) == 0:
        return {**{f"rank{k}": float("nan") for k in ks}, "mAP": float("nan"), "n_query": 0}

    rank_hits = {k: 0 for k in ks}
    aps = []
    for i in range(len(q)):
        order = np.argsort(dist[i], kind="stable")
        matches = g[order] == q[i]
        for k in ks:
            if matches[:k].any():
                rank_hits[k] += 1
        if matches.any():
            positions = np.flatnonzero(matches)               # 0-based ranks of hits
            precisions = (np.arange(len(positions)) + 1) / (positions + 1)
            aps.append(float(precisions.mean()))
        else:
            aps.append(0.0)

    out = {f"rank{k}": rank_hits[k] / len(q) for k in ks}
    out["mAP"] = float(np.mean(aps))
    out["n_query"] = int(len(q))
    return out


# --- verification -----------------------------------------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), tie-aware — a tiny substitute for scipy.stats.rankdata."""
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort(kind="stable")
    ranks = np.empty(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie block
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def roc_auc(scores, y_true) -> float:
    """Tie-aware ROC-AUC (Mann-Whitney). ``scores``: larger = more likely positive."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y_true).astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def tpr_at_fpr(scores, y_true, fpr_target: float = 0.01) -> float:
    """True-positive rate at the largest threshold whose FPR ≤ ``fpr_target``."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y_true).astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")   # high score first
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg
    ok = np.flatnonzero(fpr <= fpr_target)
    return float(tpr[ok[-1]]) if len(ok) else 0.0


# --- open set ---------------------------------------------------------------
def openset_auroc(known_scores, novel_scores) -> float:
    """AUROC that a *known* query's best gallery similarity exceeds a *novel* query's.

    ``*_scores`` are per-query max similarities to the gallery. 1.0 = perfectly separable.
    """
    known = np.asarray(known_scores, dtype=np.float64)
    novel = np.asarray(novel_scores, dtype=np.float64)
    if len(known) == 0 or len(novel) == 0:
        return float("nan")
    scores = np.concatenate([known, novel])
    y = np.concatenate([np.ones(len(known), bool), np.zeros(len(novel), bool)])
    return roc_auc(scores, y)


# --- selective prediction ---------------------------------------------------
def risk_coverage(correct, confidence):
    """Risk–coverage curve: sort by descending confidence, report cumulative (coverage, risk).

    ``risk`` = error rate over the most-confident ``coverage`` fraction. Returns
    ``(coverage, risk)`` arrays plus the area under the risk–coverage curve (lower better).
    """
    correct = np.asarray(correct).astype(bool)
    confidence = np.asarray(confidence, dtype=np.float64)
    n = len(correct)
    if n == 0:
        return np.array([]), np.array([]), float("nan")
    order = np.argsort(-confidence, kind="stable")
    c = correct[order]
    cum_correct = np.cumsum(c)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = 1.0 - cum_correct / k
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 renamed trapz -> trapezoid
    aurc = float(_trapz(risk, coverage))
    return coverage, risk, aurc
