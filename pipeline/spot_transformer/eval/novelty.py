"""Novelty detection — "is this animal in the database AT ALL?", the hard half of the task.

The matcher answers a COMPARATIVE question well ("is A more like B or C?") and an ABSOLUTE one
badly ("is this a re-sight?"). Measured on this data the absolute call is close to chance
(open-set AUROC ~0.64, where 0.5 is a coin flip), and the F0.5-optimal threshold copes by
labelling almost everything "new" -- which scores ~0.95 on novel animals for the trivial reason
that it says "new" nearly always, while missing 65-86% of genuine re-sights.

The structural cause is visible in ``aggregator.FEATURE_NAMES``: every one of the 17 features is
an absolute property of a SINGLE (query, candidate) pair. Each candidate is scored in isolation,
so the model never sees how the best match compares to the rest of the field. "This match scored
0.7" is uninterpretable on its own -- 0.7 may be excellent for a blurry photo and poor for a
sharp one.

This module supplies what was missing: **features that describe the top match relative to its
competition**, which transfer across photos in a way absolute magnitudes do not. Two ways to use
them, deliberately kept separate so they can be compared:

* **b'** -- a single relative statistic used directly as the novelty score (no training).
* **c'** -- a small classifier trained on the KNOWN/NOVEL label (not the same/different label the
  matcher already learns), consuming all of them.

Both are scored by threshold-free open-set AUROC, so the comparison measures signal rather than
where a threshold happens to sit.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:  # run as script (path[0] = this dir) OR imported as a package
    from census import auroc
except ModuleNotFoundError:
    from pipeline.spot_transformer.census import auroc

# Ordered; ``margin`` and ``z_top1`` are the two the analysis expects to matter most.
NOVELTY_FEATURES = [
    "top1",           # the absolute score — kept ONLY as the reference/ablation baseline
    "margin",         # top1 - top2: how far clear the winner is of the runner-up
    "margin_ratio",   # the same, scale-free
    "z_top1",         # top1 in standard deviations above this query's own candidate field
    "top1_minus_med", # robust version of the same idea
    "top1_minus_top5",# is the winner distinct, or is the whole leaderboard bunched?
    "score_std",      # spread of the field: a flat field means nothing stood out
    "score_iqr",
    "frac_near_top",  # fraction of candidates within 10% of the winner — many = ambiguous
    "log_ncand",      # gallery size: more candidates = more chances at a lucky high score
]


def relative_features(scores: np.ndarray) -> np.ndarray:
    """All candidate scores for ONE query -> the relative feature vector (NOVELTY_FEATURES).

    Everything except ``top1`` is defined against the query's own field of candidates, which is
    what makes it comparable across photos of differing quality and gallery size.
    """
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return np.zeros(len(NOVELTY_FEATURES))
    o = np.sort(s)[::-1]
    top1 = float(o[0])
    top2 = float(o[1]) if len(o) > 1 else 0.0
    top5 = float(o[:5].mean())
    mu, sd = float(s.mean()), float(s.std())
    med = float(np.median(s))
    iqr = float(np.percentile(s, 75) - np.percentile(s, 25))
    denom = abs(top1) + 1e-9
    return np.array([
        top1,
        top1 - top2,
        (top1 - top2) / denom,
        (top1 - mu) / (sd + 1e-9),
        top1 - med,
        top1 - top5,
        sd,
        iqr,
        float((s >= 0.9 * top1).mean()),
        float(np.log1p(len(s))),
    ], float)


def build_novelty_table(scores, qids, clab, true_by_q):
    """Per-query relative features + labels.

    Returns ``(X, is_known, top1_correct, queries)`` -- one row per QUERY (not per pair), because
    novelty is a per-query decision made after the candidates have competed.
    """
    scores = np.asarray(scores, float)
    qids = np.asarray(qids)
    clab = np.asarray(clab, dtype=object)
    X, known, correct, qs = [], [], [], []
    for q in np.unique(qids):
        m = qids == q
        sc, cl = scores[m], clab[m]
        if not len(sc):
            continue
        true = true_by_q.get(q)
        X.append(relative_features(sc))
        known.append(int(true in set(cl)))
        correct.append(int(cl[int(np.argmax(sc))] == true))
        qs.append(q)
    return (np.array(X), np.array(known, float), np.array(correct, float),
            np.array(qs, dtype=object))


# ============================================================ b' — single-statistic scorers
def single_feature_auroc(X, is_known) -> dict[str, float]:
    """Open-set AUROC of EACH relative feature used alone as the novelty score (no training).

    This is ``b'``, and it doubles as the diagnostic that says which relative signal is worth
    having: a feature at 0.5 is noise, and one well above the ``top1`` row is buying something
    the current absolute score does not have.
    """
    return {nm: auroc(X[:, i], is_known) for i, nm in enumerate(NOVELTY_FEATURES)}


# ============================================================ c' — trained novelty classifier
class NoveltyNet(nn.Module):
    """``hidden=0`` -> logistic regression. Small on purpose: one row per query means only a few
    hundred training examples per fold, far less data than the matcher's pair-level supervision."""

    def __init__(self, n_feat, hidden=0, dropout=0.1):
        super().__init__()
        self.net = (nn.Linear(n_feat, 1) if not hidden else
                    nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1)))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_novelty(X, y, *, hidden=0, epochs=300, lr=0.01, weight_decay=1e-3, seed=0):
    """Fit the known-vs-novel classifier. Returns ``(model, (mu, sd))``."""
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    model = NoveltyNet(X.shape[1], hidden=hidden)
    npos = max(int(y.sum()), 1)
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(len(y) - npos) / npos], dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(model(Xt), yt); loss.backward(); opt.step()
    return model, (mu, sd)


@torch.no_grad()
def score_novelty(model, scaler, X, *, as_prob=False):
    """Score for "this query is a KNOWN re-sight".

    Returns the raw LOGIT by default. Sigmoid is monotonic so it cannot change AUROC in theory,
    but a saturated logit maps to the identical float for every row, and the resulting mass of
    ties collapses AUROC to exactly 0.5 -- a real failure observed here before this default
    changed. Rank on logits; ask for ``as_prob`` only when a calibrated probability is wanted.
    """
    mu, sd = scaler
    z = model(torch.tensor((X - mu) / sd, dtype=torch.float32)).numpy()
    return 1.0 / (1.0 + np.exp(-z)) if as_prob else z


# ============================================================ fix 1 — threshold selection
def select_threshold(score, is_known, criterion="balanced"):
    """Pick the accept/reject cut. ``criterion``:

    ``balanced``  -- maximize balanced accuracy: mean of (re-sights caught) and (new animals
                     correctly enrolled). Both classes count, so the degenerate "call everything
                     new" strategy scores 0.5 instead of looking excellent.
    ``f1``        -- maximize F1 over the known class (precision/recall balanced).
    ``youden``    -- maximize sensitivity + specificity - 1.

    F0.5 is deliberately NOT offered here: weighting precision 2x is what produced the
    all-but-degenerate operating point this module exists to fix.
    """
    s = np.asarray(score, float)
    k = np.asarray(is_known).astype(bool)
    if k.all() or (~k).all():
        return dict(thr=float(np.median(s)), balanced_acc=float("nan"),
                    known_recall=float("nan"), novel_recall=float("nan"))
    grid = np.unique(np.quantile(s, np.linspace(0, 1, 200)))
    best = None
    for t in grid:
        acc = s >= t
        kr = float((acc & k).sum() / max(int(k.sum()), 1))         # re-sights caught
        nr = float((~acc & ~k).sum() / max(int((~k).sum()), 1))    # new animals enrolled
        if criterion == "balanced":
            v = 0.5 * (kr + nr)
        elif criterion == "youden":
            v = kr + nr - 1.0
        else:                                                       # f1 over the known class
            tp = float((acc & k).sum()); fp = float((acc & ~k).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            v = 2 * prec * kr / (prec + kr) if (prec + kr) else 0.0
        if best is None or v > best[0]:
            best = (v, t, kr, nr)
    _, thr, kr, nr = best
    return dict(thr=float(thr), balanced_acc=0.5 * (kr + nr),
                known_recall=kr, novel_recall=nr)
