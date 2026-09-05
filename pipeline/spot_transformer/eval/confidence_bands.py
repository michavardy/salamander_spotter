"""Accuracy@k broken down by confidence band — "how good is the answer when the model says 90%?"

R@1 = 0.603 is a single number over every query, and it is the wrong number for a workflow where
a person reviews the shortlist. What a reviewer needs is a **triage rule**: which queries can be
accepted unseen, which need a look at the top-5, and which are worth no one's time. That requires
a per-query confidence whose stated value means what it says.

**The confidence reported here is per QUERY, not per candidate.** ``accuracy@k`` is a property of
one query's whole ranked list ("is the right animal in the top k"), so the confidence bucketing it
has to live at the same level. A per-candidate probability cannot be bucketed against accuracy@5
without a category error.

Two confidences are computed and tabulated side by side, because they are calibrated against
different base rates and only one of them means what the band label says:

``conf_rel``   **the one to use.** A logistic regression over the query's own score field
               (``novelty.NOVELTY_FEATURES``: margin, z_top1, frac_near_top, ...) trained directly
               on the target "was top-1 correct". Its base rate IS top-1 accuracy (~0.6), so band
               "90-95" should contain queries that are right ~90-95% of the time — the diagonal
               this table exists to check. Relative features are used rather than the raw score
               because 0.7 means something different on a blurry photo than a sharp one
               (``novelty.py`` docstring).

``conf_pair``  the Platt-calibrated ``P(same individual)`` of the winning candidate — the number a
               naive UI would print next to the top match. Tabulated as the CONTRAST: it is
               calibrated on training pairs at a ~1:30 negative ratio and then applied against a
               ~750-individual gallery, so its stated probability is against the wrong prior and
               its bands do not line up with accuracy. Keep it in the table so the gap is visible
               rather than argued about.

Both are strictly OUT-OF-FOLD: the aggregator is trained on the fold's train individuals, and the
confidence model for fold *f* is fitted only on the query rows of the other folds. No query ever
contributes to the model that scores it.

**Closed-set.** Every eval query's individual is guaranteed present in the gallery (``get_cv_folds``
only evaluates individuals with >=2 real photos). So this measures "can it rank the right animal to
the top", not "is this animal in the database at all" — the latter is near chance here (open-set
AUROC ~0.64, see ``novelty.py``) and no band in this table should be read as answering it.

    pixi run confidence-bands                  # full 750-individual gallery (the deployment number)
    pixi run confidence-bands --fold-gallery   # ~65 candidates, minutes instead of ~25 min
    pixi run confidence-bands --workers 14     # the pair features parallelize cleanly
    pixi run confidence-bands --from-csv <p>   # re-bucket a finished run, no recompute

The per-query CSV is the real output: re-banding is free, so the band edges below are a default
rather than a decision.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval"):
    _p = str(_ST / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data as d                                             # noqa: E402
from aggregator import (attach_centroids, build_pairs,       # noqa: E402
                        _logit, _prob, platt_fit, train_aggregator)
from novelty import NOVELTY_FEATURES, relative_features      # noqa: E402

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

# Descending, half-open ``[lo, hi)`` in PERCENT, exactly as asked for. The CSV keeps every query's
# raw confidence, so changing these costs a --from-csv rerun and no compute.
BANDS = [(95.0, 100.01, ">95"), (90.0, 95.0, "90-95"), (85.0, 90.0, "85-90"),
         (80.0, 85.0, "80-85"), (75.0, 80.0, "75-80"), (70.0, 75.0, "70-75"),
         (65.0, 70.0, "65-70"), (50.0, 65.0, "50-65"), (-0.01, 50.0, "<50")]

# The query-level confidence model's inputs: the relative score-field features, plus how much
# pattern the query photo actually offered (a 6-spot animal is a weaker question than a 30-spot one).
CONF_FEATURES = NOVELTY_FEATURES + ["log_nq"]

KS = (1, 5, 10)


# ----------------------------------------------------------------------------- worker
_W: dict = {}


def _init_worker(gallery: list[int]) -> None:
    """Rebuild the image sets inside each worker (0.3s) instead of pickling them per chunk."""
    _W["sets"] = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))
    _W["gallery"] = gallery


def _eval_chunk(chunk: list[int]):
    """Pair features for a slice of the eval queries."""
    return build_pairs(_W["sets"], chunk, gallery_images=_W["gallery"],
                       neg_per_query=None, use_geom=True)


def _eval_features(sets, queries, gallery, workers: int):
    """``build_pairs`` for ``queries`` against ``gallery``, optionally across processes.

    ``gallery`` must ALREADY contain every eval image of the fold, not just the distractors:
    ``iter_pairs`` derives its candidate pool from ``images + gallery_images``, so a query whose
    own individual is represented only by images outside ``queries`` would otherwise find no
    gallery positive and be silently dropped. Holding that invariant here is also what makes
    chunking exact — each chunk sees the identical candidate set, only the query list differs —
    and what lets ``--smoke`` subsample queries without shrinking the gallery.
    """
    if workers <= 1:
        return build_pairs(sets, queries, gallery_images=gallery,
                           neg_per_query=None, use_geom=True)
    n_chunks = min(workers * 4, len(queries))
    chunks = [list(c) for c in np.array_split(np.array(queries), n_chunks) if len(c)]
    parts = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(list(gallery),)) as pool:
        for i, out in enumerate(pool.map(_eval_chunk, chunks)):
            parts.append(out)
            logger.info(f"    chunk {i + 1}/{len(chunks)}")
    return tuple(np.concatenate([p[j] for p in parts]) for j in range(4))


# ----------------------------------------------------------------------------- pass 1
def _crossfit_platt(Xtr, ytr, qtr, *, inner=5, seed=0) -> tuple[float, float]:
    """Platt ``(a, b)`` fitted on CROSS-FITTED train logits, so the ranking model keeps every pair.

    The obvious alternative — reserve 20% of train as a calibration holdout — measurably costs
    ranking accuracy (fold 0: R@1 0.186 -> 0.163), because this dataset is small enough that a
    fifth of the pairs matters. Cross-fitting buys the same never-fitted-on-its-own-prediction
    guarantee for the price of ``inner`` extra fits of a 22-feature logistic regression, which is
    seconds: the expensive part (the pair features) is already built and is reused by every split.

    Splits are by QUERY IMAGE, not by row, so a query's positive and its negatives never straddle
    the boundary.
    """
    uq = np.unique(qtr)
    rng = np.random.default_rng(seed)
    rng.shuffle(uq)
    oof = np.zeros(len(Xtr))
    for chunk in np.array_split(uq, inner):
        held = np.isin(qtr, chunk)
        if held.all() or not held.any():
            continue
        m_in, s_in = train_aggregator(Xtr[~held], ytr[~held], hidden=0, seed=seed)
        oof[held] = _logit(m_in, s_in, Xtr[held])
    return platt_fit(oof, ytr)


def collect_queries(sets, *, k=5, seed=0, full_gallery=True, neg_per_query=30, workers=1,
                    max_queries=None, source="all", train_on_all=True) -> pd.DataFrame:
    """One row per eval query: its rank, its relative-score features, and both confidences.

    Mirrors ``aggregator.crossval_aggregator`` fold for fold — same folds, same ``train_aggregator``
    call on the same pairs, same ``_prob`` ranking — so the ``ALL`` row of the band table is that
    function's R@1 by construction, and the two were checked equal to machine precision on fold 0.

    Do NOT expect ``results.md``'s 0.603: that table was written when 83 individuals were
    eval-eligible and 17 features existed. This dataset now has 319 eligible of 742 (the Haifa
    merge) and 22 features, and fold 0 measures R@1 0.186 at a 64-candidate gallery — the raw
    soft-chamfer baseline fell with it (0.556 -> 0.151), so this is the task getting harder, not
    the aggregator regressing.

    ``max_queries`` subsamples the eval side (``--smoke`` only). It does NOT shrink the gallery, so
    a smoke run still exercises the real candidate volume per query.

    ``source`` restricts BOTH the queries and the gallery to one collection (see
    ``data.source_keep_mask``) — gallery included, because gating the queries alone would still
    rank each sasa animal against 461 Haifa distractors and measure the harder problem anyway.
    ``train_on_all`` keeps the other collection's photos as training data, which is a separate
    question from what gets scored; set it False for a literal "that data is gone" run.
    """
    keep = d.source_keep_mask(sets, source)
    folds = d.get_cv_folds(sets, k=k, seed=seed, eval_mask=keep)
    d.assert_evaluable(folds, what=f"the source gate (SOURCE={source})")
    rows = []
    for fi, (tr, ev) in enumerate(folds):
        t0 = time.time()
        train_imgs = [i for i in tr if not sets[i].is_synth and (train_on_all or keep[i])]
        queries = ev
        if max_queries is not None and len(ev) > max_queries:
            queries = sorted(np.random.default_rng(seed).choice(ev, max_queries, replace=False))

        # --- aggregator, trained exactly as crossval_aggregator does (serial: identical negatives)
        Xtr, ytr, qtr, _ = build_pairs(sets, train_imgs, neg_per_query=neg_per_query,
                                       seed=seed, use_geom=True)
        model, scaler = train_aggregator(Xtr, ytr, hidden=0, seed=seed)
        a, b = _crossfit_platt(Xtr, ytr, qtr, seed=seed)

        # --- score the eval fold against the deployment-sized gallery. The eval images are always
        # part of the gallery (that is where a query's own individual lives); --full-gallery adds
        # the train fold's ~660 further individuals as distractors.
        # Distractors are gated by SOURCE even when training is not: the gallery IS the population
        # the metric is about, so a sasa run must not be ranked against Haifa individuals.
        distract = [i for i in train_imgs if keep[i]] if full_gallery else []
        gal = list(dict.fromkeys(distract + list(ev)))
        X, _, qids, clab = _eval_features(sets, queries, gal, workers)
        p = _prob(model, scaler, X)
        p_cal = 1.0 / (1.0 + np.exp(-(a * _logit(model, scaler, X) + b)))

        for q in np.unique(qids):
            m = qids == q
            s, cl = p[m], clab[m]
            order = np.argsort(-s)
            ranked = cl[order]
            true = sets[q].label
            hit = np.where(ranked == true)[0]
            if not len(hit):                      # unrankable query (no gallery positive)
                continue
            feats = relative_features(s)
            rows.append({
                "fold": fi, "sid": sets[q].sid, "label": true,
                "rank": int(hit[0]) + 1, "n_candidates": int(m.sum()),
                "log_nq": float(np.log1p(len(sets[q].spots))),
                "conf_pair": float(p_cal[m][order[0]]),      # calibrated P(same) of the WINNER
                "top1_label": str(ranked[0]),
                **{nm: float(v) for nm, v in zip(NOVELTY_FEATURES, feats)},
            })
        logger.info(f"  fold {fi}: {int(len(np.unique(qids)))} queries x "
              f"{int(np.median([int((qids == q).sum()) for q in np.unique(qids)]))} candidates "
              f"({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df["top1_correct"] = (df["rank"] == 1).astype(int)
    return df


# ----------------------------------------------------------------------------- pass 2
def _fit_logreg(X, y, *, epochs=600, lr=0.05, weight_decay=1e-3, seed=0):
    """Plain logistic regression, **no** ``pos_weight``.

    Class balancing is deliberately omitted: it improves ranking metrics and destroys calibration by
    shifting every predicted probability off the true base rate — which is the one property this
    whole table depends on. ``novelty.train_novelty`` weights (it optimizes AUROC); this does not.
    """
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    net = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(net(Xt).squeeze(-1), yt).backward()
        opt.step()
    return net, (mu, sd)


@torch.no_grad()
def _predict(net, scaler, X):
    mu, sd = scaler
    z = net(torch.tensor((X - mu) / sd, dtype=torch.float32)).squeeze(-1).numpy()
    return 1.0 / (1.0 + np.exp(-z))


def add_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold ``conf_rel``: fold *f*'s confidences come from a model fitted on the others.

    Folds are disjoint by INDIVIDUAL (``get_cv_folds``), so this is leakage-free at the identity
    level, not merely at the photo level.
    """
    X = df[CONF_FEATURES].to_numpy(float)
    y = df["top1_correct"].to_numpy(float)
    conf = np.zeros(len(df))
    for f in sorted(df["fold"].unique()):
        m = (df["fold"] == f).to_numpy()
        net, scaler = _fit_logreg(X[~m], y[~m])
        conf[m] = _predict(net, scaler, X[m])
    out = df.copy()
    out["conf_rel"] = conf
    return out


# ----------------------------------------------------------------------------- reporting
def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval. Printed because several bands hold only tens of queries, where a bare
    point estimate invites a decision the sample cannot support."""
    if n == 0:
        return float("nan"), float("nan")
    p = hits / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / denom, (centre + half) / denom


def _acc(sub: pd.DataFrame, k: int) -> float:
    return float((sub["rank"] <= k).mean()) if len(sub) else float("nan")


def band_table(df: pd.DataFrame, conf_col: str, title: str) -> pd.DataFrame:
    """Per-band n / accuracy@1,5,10, plus the cumulative 'accept everything above here' view."""
    c = df[conf_col].to_numpy(float) * 100.0
    logger.info(f"{'=' * 96}\n {title}   (confidence = {conf_col})\n{'=' * 96}")
    logger.info(f" {'band':>7} | {'photos':>6} | {'indiv':>5} | {'acc@1':>6} {'[95% CI]':>14} | "
          f"{'acc@5':>6} | {'acc@10':>6} | {'mean conf':>9}")
    logger.info(" " + "-" * 94)

    recs = []
    for lo, hi, lab in BANDS:
        sub = df[(c >= lo) & (c < hi)]
        n = len(sub)
        if n == 0:
            logger.info(f" {lab:>7} | {0:>6} | {'-':>5} | {'-':>6} {'':>14} | {'-':>6} | {'-':>6} | {'-':>9}")
            recs.append(dict(band=lab, n=0))
            continue
        a1, a5, a10 = (_acc(sub, k) for k in KS)
        lo_ci, hi_ci = wilson(int((sub["rank"] == 1).sum()), n)
        logger.info(f" {lab:>7} | {n:>6} | {sub['label'].nunique():>5} | {a1:>6.3f} "
              f"{f'[{lo_ci:.2f}-{hi_ci:.2f}]':>14} | {a5:>6.3f} | {a10:>6.3f} | "
              f"{sub[conf_col].mean() * 100:>8.1f}%")
        recs.append(dict(band=lab, n=n, n_individuals=int(sub["label"].nunique()),
                         **{f"acc@{k}": a for k, a in zip(KS, (a1, a5, a10))},
                         ci_lo=lo_ci, ci_hi=hi_ci, mean_conf=float(sub[conf_col].mean())))

    logger.info(" " + "-" * 94)
    a1, a5, a10 = (_acc(df, k) for k in KS)
    logger.info(f" {'ALL':>7} | {len(df):>6} | {df['label'].nunique():>5} | {a1:>6.3f} "
          f"{'':>14} | {a5:>6.3f} | {a10:>6.3f} | {df[conf_col].mean() * 100:>8.1f}%")

    # Cumulative: the number a triage policy is actually set from.
    logger.info(" cumulative - accept every query at or above the band ('coverage' = share answered):")
    logger.info(f" {'>= band':>7} | {'coverage':>8} | {'photos':>6} | {'acc@1':>6} | {'acc@5':>6} | {'acc@10':>6}")
    logger.info(" " + "-" * 60)
    for lo, _hi, lab in BANDS:
        sub = df[c >= lo]
        if not len(sub):
            continue
        logger.info(f" {lab:>7} | {len(sub) / len(df):>8.3f} | {len(sub):>6} | "
              f"{_acc(sub, 1):>6.3f} | {_acc(sub, 5):>6.3f} | {_acc(sub, 10):>6.3f}")
    return pd.DataFrame(recs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5, help="CV folds (default 5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold-gallery", action="store_true",
                    help="rank against the eval fold only (~65 candidates) instead of the full "
                         "~750-individual population. Much faster; acc@10 means much less.")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes for the eval pair features (the ~57min single-core part)")
    ap.add_argument("--from-csv", type=Path, default=None,
                    help="re-band a finished run's per-query CSV without recomputing anything")
    ap.add_argument("--source", choices=("all", "sasa", "kf"), default=None,
                    help="restrict queries AND gallery to one collection (default: env SOURCE, "
                         "else 'all'). 'sasa' drops the 461 merged Haifa/KF individuals, leaving "
                         "the 87-eligible population results.md was measured on")
    ap.add_argument("--sasa-train-only", action="store_true",
                    help="with --source sasa, also drop the Haifa photos from TRAINING (the "
                         "literal 'that data is gone' run). Default keeps them as training data: "
                         "822 photos the individual-agnostic aggregator may still learn from")
    ap.add_argument("--smoke", action="store_true",
                    help="2 folds, 25 queries each, fold gallery, 8 negatives — exercises every "
                         "code path in ~2min. The NUMBERS ARE MEANINGLESS; this only proves it runs")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.k, args.fold_gallery = 2, True
    source = args.source or d.SOURCE
    train_tag = "_trainsame" if args.sasa_train_only else ""

    out_dir = args.out or (d.REPO_ROOT / "artifacts" / "spot_transformer" /
                           f"confidence_bands{d.emb_tag()}{d.quality_tag()}"
                           f"{'' if source == 'all' else '_' + source}{train_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        df = pd.read_csv(args.from_csv)
    else:
        logger.info("=" * 96)
        logger.info(f" confidence-banded accuracy   source={source}"
              f"{' (train: sasa only)' if args.sasa_train_only else ''}  gallery="
              f"{'eval fold only' if args.fold_gallery else 'FULL population'}  "
              f"k={args.k}  workers={args.workers}")
        logger.info("=" * 96)
        sets = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))
        df = collect_queries(sets, k=args.k, seed=args.seed,
                             full_gallery=not args.fold_gallery, workers=args.workers,
                             neg_per_query=8 if args.smoke else 30,
                             max_queries=25 if args.smoke else None,
                             source=source, train_on_all=not args.sasa_train_only)
        df = add_confidence(df)
        csv = out_dir / ("per_query_smoke.csv" if args.smoke else "per_query.csv")
        df.to_csv(csv, index=False)
        logger.info(f" per-query rows -> {csv}")

    logger.info(f" {len(df)} queries, {df['label'].nunique()} individuals, "
          f"median gallery {df['n_candidates'].median():.0f} candidates")

    tbl = band_table(df, "conf_rel",
                     "PRIMARY - query-level confidence (calibrated on 'is top-1 correct')")
    band_table(df, "conf_pair",
               "CONTRAST - Platt-calibrated P(same) of the top candidate (wrong prior; expect "
               "the bands NOT to line up)")

    tbl.to_csv(out_dir / "bands_conf_rel.csv", index=False)
    logger.info(f" band summary -> {out_dir / 'bands_conf_rel.csv'}")


if __name__ == "__main__":
    main()
