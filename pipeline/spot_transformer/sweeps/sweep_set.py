"""Open-set / census sweep for the learned spot aggregator.

Answers the "fully automated population census" ask: the model must decide *match an existing
individual* vs *a brand-new one* at a score threshold, where a false MATCH is the costly error
(it collapses two animals and DEFLATES the count). So the headline is **F0.5** (precision 2x
recall), reported two ways: pairwise (over all query->candidate pairs) and a census simulation
with novel-individual holdout that also reports the population-count bias. R@1 is kept only as a
ranking DIAGNOSTIC. Baselines: raw soft-chamfer voting and the summary-feature logistic
regression (the standing champion) under the *same* metrics.

What it sweeps
  regularization : dropout / L1 / L2 / feature-jitter / width (Deep-Sets only)
  ensembling     : average scores over N seeds (the fold-variance fix)
and, per config, saves train/test learning curves (F0.5 + R@1, logged every epoch), an overlay,
diagnostic panels (pairwise & census PR, risk-coverage, image-quality strata, population bias),
a cross-seed variance study, and RESULTS_census.md.

    pixi run python pipeline/spot_transformer/sweep_set.py            # full run
    QUICK=1 pixi run python pipeline/spot_transformer/sweep_set.py    # fast smoke (few epochs/configs)
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*nested tensor.*")   # torch transformer prototype notice

try:  # run as script (path[0] = this dir) OR imported as a package
    import data as d
    import census as cen
    from aggregator import attach_centroids, build_pairs, train_aggregator, _prob, rank_eval
    from aggregator_set import (build_record_pairs, train_set, score_pairs_set, per_query_top1,
                                _score_set)
except ModuleNotFoundError:
    from pipeline.spot_transformer import data as d
    from pipeline.spot_transformer import census as cen
    from pipeline.spot_transformer.aggregator import (attach_centroids, build_pairs,
                                                      train_aggregator, _prob, rank_eval)
    from pipeline.spot_transformer.aggregator_set import (build_record_pairs, train_set,
                                                          score_pairs_set, per_query_top1, _score_set)

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

OUT_DIR = d.REPO_ROOT / "artifacts" / "spot_transformer" / "sweeps" / f"set{d.quality_tag()}"
BETA = 0.5                                                   # census priority: precision 2x recall

# Okabe-Ito (colorblind-safe)
C_TRAIN, C_TEST, C_BASE = "#E69F00", "#0072B2", "#999999"
C_LOGREG, C_RAW = "#009E73", "#CC79A7"
OVERLAY = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
           "#000000", "#F0E442", "#8C564B", "#7F7F7F"]

# --- config grid ---
# LEAN re-run on the 2x-bigger all_sasa_norm_2026_19_07, to test whether MORE DATA improves the
# learned matcher. The prior 10-config x 3-seed-variance grid was ~5 h on the old (733-img) set and
# would be ~10-14 h here, so this keeps just three models: the DeepSets reference, its old winner
# (feature jitter), and ONE transformer re-added on purpose. xf_1L2h was the top census-F0.5 config
# last time AND is the data-hungry model that was dropped for "diverging at this data size" — so
# doubling the data is exactly its test. Same recipe as before (h=32, dropout .3, wd 1e-2) so the
# ONLY variable vs the old run is the amount of data.
#
# Previously in the grid (dropped for this lean run, not deleted from history):
#   deepsets_drop / deepsets_l1 / deepsets_small  — regularization variants, never beat jit.
#   ensemble5                                     — 5x cost, didn't beat jit.
#   xf_2L4h / xf_wide                             — deeper/wider transformers that diverged.
#   deepsets_estop                                — early-stop halted at near-init epochs.
CONFIGS = [
    dict(name="deepsets_ref", arch="deepsets",    h=32, dropout=0.3, weight_decay=1e-2, l1=0.0, jitter=0.0),
    dict(name="deepsets_jit", arch="deepsets",    h=32, dropout=0.3, weight_decay=1e-2, l1=0.0, jitter=0.02),
    dict(name="xf_1L2h",      arch="transformer", h=32, dropout=0.3, weight_decay=1e-2, l1=0.0, jitter=0.0,
         n_layers=1, n_heads=2),
]

DATA = dict(neg_per_query=60, use_synth=True, use_geom=True, novel_frac=0.35, val_frac=0.12)


# ============================================================ fold data plumbing
def carve_val(sets, train_labels_pool, frac, seed):
    """Hold ~frac of the multi-image TRAIN individuals fully out of training as a val fold
    (all their images removed), for early stopping. Returns (val_labels set)."""
    by = {}
    for lbl in train_labels_pool:
        by.setdefault(lbl, True)
    multi = [l for l in train_labels_pool
             if sum(not sets[i].is_synth for i in _by_label(sets, l)) >= 2]
    rng = np.random.default_rng(seed + 7)
    multi = [multi[i] for i in rng.permutation(len(multi))]
    return set(multi[: max(1, int(round(frac * len(multi))))])


_BY_LABEL_CACHE: dict[int, dict] = {}
def _by_label(sets, lbl):
    key = id(sets)
    if key not in _BY_LABEL_CACHE:
        m: dict[str, list[int]] = {}
        for i, s in enumerate(sets):
            m.setdefault(s.label, []).append(i)
        _BY_LABEL_CACHE[key] = m
    return _BY_LABEL_CACHE[key][lbl]


def build_fold_bundle(sets, tr, ev, *, seed, use_synth, use_geom, novel_frac, val_frac):
    """Everything jitter-free and config-independent for one fold (built once, reused by all
    model configs): the train/val label split, monitor record-sets for the learning curves, the
    closed-set eval pairs (pairwise + R@1), and the open-set census substrate."""
    train_labels = {sets[i].label for i in tr}
    val_labels = carve_val(sets, train_labels, val_frac, seed)
    tr_pool = [i for i in tr if sets[i].label not in val_labels]              # training images
    tr_real = [i for i in tr_pool if not sets[i].is_synth]
    val_imgs = [i for l in val_labels for i in _by_label(sets, l) if i in tr and not sets[i].is_synth]

    # train-diagnostic monitor: a gallery-matched subset of TRAIN (mimics eval), for the curve
    rbl: dict[str, list[int]] = {}
    for i in tr_real:
        rbl.setdefault(sets[i].label, []).append(i)
    multi = [l for l, ii in rbl.items() if len(ii) >= 2]
    n_ev = len({sets[i].label for i in ev})
    chosen = np.random.default_rng(seed).choice(multi, size=min(n_ev, len(multi)), replace=False)
    td = [i for l in chosen for i in rbl[l]]
    r_td, _, q_td, c_td = build_record_pairs(sets, td, neg_per_query=None, seed=seed, use_geom=use_geom)

    # closed-set eval pairs (== test monitor): pairwise precision + R@1 diagnostic
    r_ev, y_ev, q_ev, c_ev = build_record_pairs(sets, ev, neg_per_query=None, seed=seed, use_geom=use_geom)
    true_ev = {q: sets[q].label for q in np.unique(q_ev)}

    # val monitor (early stopping) — held-out train individuals, leave-one-out
    monitors = {"train": (r_td, q_td, c_td, {q: sets[q].label for q in np.unique(q_td)}),
                "test":  (r_ev, q_ev, c_ev, true_ev)}
    if len(val_imgs) >= 2:
        r_va, _, q_va, c_va = build_record_pairs(sets, val_imgs, neg_per_query=None, seed=seed, use_geom=use_geom)
        monitors["val"] = (r_va, q_va, c_va, {q: sets[q].label for q in np.unique(q_va)})

    # closed-set 17-dim summary features for the logreg baseline (pairwise)
    X_ev, yX_ev, qX_ev, cX_ev = build_pairs(sets, ev, neg_per_query=None, seed=seed, use_geom=use_geom)

    # open-set census substrate: novel-individual holdout gallery/query split
    gal, qry = cen.make_openset_split(sets, ev, novel_frac, seed)
    os_recs, os_q, os_c, os_true = cen.build_openset_records(sets, gal, qry, use_geom=use_geom)
    os_feat, osf_q, osf_c = cen.build_openset_features(sets, gal, qry, use_geom=use_geom)

    return dict(tr_pool=tr_pool, tr_real=tr_real, monitors=monitors,
                closed=(r_ev, y_ev, q_ev, c_ev, true_ev),
                closed_feat=(X_ev, yX_ev, qX_ev, cX_ev),
                openset=dict(recs=os_recs, q=os_q, c=os_c, true=os_true,
                             feat=os_feat, feat_q=osf_q, feat_c=osf_c,
                             raw=np.array([r[:, 0].sum() for r in os_recs])),  # soft-chamfer sum
                sids_ev=np.array([sets[q].sid for q in np.unique(q_ev)]))


# ============================================================ metric helpers
def metrics_from_scores(scores, y, os_scores, os_q, os_c, os_true, *, train_thr=None):
    """Common metric block from per-pair scores. ``scores/y`` = closed-set eval pairs (pairwise
    F0.5, R@1 diagnostic); ``os_*`` = open-set pairs (census F0.5 + population bias)."""
    ap = cen.average_precision(scores, y)
    bf = cen.best_fbeta(scores, y, BETA)                      # oracle threshold on test
    honest = cen.fbeta_at(scores, y, train_thr, BETA) if train_thr is not None else None

    top = per_query_top1(os_scores, os_q, os_c, os_true)      # per-query top-1 census decision
    sweep, cbest = cen.census_sweep(top["top1"], top["top1_correct"], top["is_known"], BETA)
    return dict(pair_ap=ap, pair_bestf=bf["f"], pair_bestf_thr=bf["thr"],
                pair_honestf=(honest["f"] if honest else None),
                census_bestf=cbest["f"], census_prec=cbest["precision"], census_rec=cbest["recall"],
                census_bias=cbest["count_bias"], census_infl=cbest["inflation"],
                census_defl=cbest["deflation"], census_thr=cbest["thr"],
                census_sweep=sweep, census_top=top)


def r1_diagnostic(scores, q, c, true):
    top = per_query_top1(scores, q, c, true)
    return float(np.mean(top["top1_correct"]))


# ============================================================ baselines (raw voting, logreg)
def baseline_fold(sets, ev, bundle, *, seed, use_geom):
    """Per-fold metrics for the two references, under the SAME pairwise + census metrics.

    raw   = soft-chamfer voting (no training): pair score = sum of per-query-spot best sim.
    logreg= summary-feature logistic regression (the standing R@1 champion), native recipe
            (real-only training, 30 negatives/query)."""
    r_ev, y_ev, q_ev, c_ev, true_ev = bundle["closed"]
    X_ev, _, _, _ = bundle["closed_feat"]
    os = bundle["openset"]

    raw_pair = np.array([r[:, 0].sum() for r in r_ev])                    # closed-set pairwise
    raw = metrics_from_scores(raw_pair, y_ev, os["raw"], os["q"], os["c"], os["true"])
    raw.update(r1=r1_diagnostic(raw_pair, q_ev, c_ev, true_ev), closed_scores=raw_pair, closed_y=y_ev)

    Xtr, ytr, _, _ = build_pairs(sets, bundle["tr_real"], neg_per_query=30, seed=seed, use_geom=use_geom)
    lr, sc = train_aggregator(Xtr, ytr, hidden=0, seed=seed)
    lr_pair = _prob(lr, sc, X_ev)                                         # closed-set pairwise
    lr_os = _prob(lr, sc, os["feat"])                                     # open-set census
    logreg = metrics_from_scores(lr_pair, y_ev, lr_os, os["feat_q"], os["feat_c"], os["true"])
    logreg.update(r1=r1_diagnostic(lr_pair, q_ev, c_ev, true_ev), closed_scores=lr_pair, closed_y=y_ev)
    return raw, logreg


# ============================================================ one model config, one fold
def run_model_fold(sets, ev, bundle, cfg, *, seed, epochs, use_geom, log_every):
    """Train a config on one fold (single model, or an N-seed ensemble) and score it under the
    pairwise + census metrics. Returns (metrics, history) — history is the learning curve."""
    r_ev, y_ev, q_ev, c_ev, true_ev = bundle["closed"]
    os = bundle["openset"]
    mk = {k: cfg[k] for k in ("arch", "h", "dropout", "weight_decay", "l1", "n_heads", "n_layers")
          if k in cfg}
    patience = 0 if cfg.get("ensemble") else cfg.get("patience", 0)
    n_seed = int(cfg.get("ensemble", 1))

    # training records (jitter is a data-config; built per (fold,jitter) by the caller & passed in)
    tr_recs, tr_y = bundle["_train_recs"]
    models, scalers, hist0 = [], [], None
    for si in range(n_seed):
        model, scaler, hist = train_set(
            tr_recs, tr_y, epochs=epochs, seed=seed + si, patience=patience,
            log_every=(log_every if si == 0 else 0), monitors=bundle["monitors"],
            beta=BETA, verbose=(si == 0), **mk)
        models.append(model); scalers.append(scaler)
        if si == 0:
            hist0 = hist

    M = models if n_seed > 1 else models[0]                  # list -> ensemble (mean prob)
    S = scalers if n_seed > 1 else scalers[0]
    scores = score_pairs_set(M, S, r_ev, q_ev, c_ev)                      # closed-set pairwise
    os_scores = score_pairs_set(M, S, os["recs"], os["q"], os["c"])       # open-set census
    train_thr = cen.best_fbeta(*_train_pair_scores(M, S, bundle), BETA)["thr"]
    met = metrics_from_scores(scores, y_ev, os_scores, os["q"], os["c"], os["true"], train_thr=train_thr)
    closed_top = per_query_top1(scores, q_ev, c_ev, true_ev)
    met["r1"] = float(np.mean(closed_top["top1_correct"]))
    met["closed_top"] = closed_top                                        # per-query (R@1, margin, ...)
    met["closed_scores"] = scores; met["closed_y"] = y_ev                 # pooled pairwise PR
    met["sids"] = bundle["sids_ev"]                                       # per-query image ids (quality join)
    return met, hist0


def _train_pair_scores(M, S, bundle):
    """Scores + same-labels on the train-diagnostic monitor, for picking the honest threshold."""
    r_td, q_td, c_td, true_td = bundle["monitors"]["train"]
    scores = score_pairs_set(M, S, r_td, q_td, c_td)
    same = np.array([c_td[i] == true_td[q_td[i]] for i in range(len(q_td))], float)
    return scores, same


# ============================================================ CV orchestration
def run_cv(sets, folds, configs, *, seed, epochs, data, log_every, quiet=False):
    """Run all configs over all folds, caching per-fold data and per-(fold,jitter) train records.
    Returns (results, histories, raw_results, logreg_results)."""
    results = {c["name"]: [] for c in configs}
    histories = {c["name"]: [] for c in configs}
    raw_res, lr_res = [], []
    for fi, (tr, ev) in enumerate(folds):
        if not quiet:
            logger.info(f"===== fold {fi} " + "=" * 44)
        bundle = build_fold_bundle(sets, tr, ev, seed=seed, use_synth=data["use_synth"],
                                   use_geom=data["use_geom"], novel_frac=data["novel_frac"],
                                   val_frac=data["val_frac"])
        raw, logreg = baseline_fold(sets, ev, bundle, seed=seed, use_geom=data["use_geom"])
        raw_res.append(raw); lr_res.append(logreg)
        if not quiet:
            logger.info(f"  baselines  raw: R@1 {raw['r1']:.3f} pairF{BETA:g} {raw['pair_bestf']:.3f} "
                  f"censusF {raw['census_bestf']:.3f} | logreg: R@1 {logreg['r1']:.3f} "
                  f"pairF {logreg['pair_bestf']:.3f} censusF {logreg['census_bestf']:.3f}")
        rec_cache: dict[float, tuple] = {}
        for cfg in configs:
            jit = cfg.get("jitter", 0.0)
            if jit not in rec_cache:
                recs, y, _, _ = build_record_pairs(sets, bundle["tr_pool"], neg_per_query=data["neg_per_query"],
                                                   seed=seed, use_geom=data["use_geom"], jitter=jit)
                rec_cache[jit] = (recs, y)
            bundle["_train_recs"] = rec_cache[jit]
            met, hist = run_model_fold(sets, ev, bundle, cfg, seed=seed, epochs=epochs,
                                       use_geom=data["use_geom"], log_every=log_every)
            results[cfg["name"]].append(met); histories[cfg["name"]].append(hist)
            if not quiet:
                logger.info(f"    {cfg['name']:<14} R@1 {met['r1']:.3f} | pairF{BETA:g} {met['pair_bestf']:.3f} "
                      f"(AP {met['pair_ap']:.3f}) | censusF {met['census_bestf']:.3f} "
                      f"P {met['census_prec']:.3f} R {met['census_rec']:.3f} bias {met['census_bias']:+d}")
    return results, histories, raw_res, lr_res


def agg(res_list, key):
    v = np.array([r[key] for r in res_list if r.get(key) is not None], float)
    return (float(np.mean(v)), float(np.std(v))) if len(v) else (float("nan"), float("nan"))


# ============================================================ plotting
def _style(ax):
    ax.grid(True, color="0.9", lw=0.6); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def _stack(histories, key):
    """Mean +/- std over folds of a per-epoch series (truncated to the shortest fold)."""
    series = [[h.get(key, np.nan) for h in fold] for fold in histories]
    n = min(len(s) for s in series)
    A = np.array([s[:n] for s in series], float)
    return np.arange(n), np.nanmean(A, 0), np.nanstd(A, 0)


def plot_curve(name, folds_hist, path, ref):
    """Per-config learning curve: LEFT train/test F0.5 (+ test R@1 faint) with fold band and
    raw/logreg reference lines; RIGHT train loss."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, color, lab in [("train_f", C_TRAIN, "train F0.5"), ("test_f", C_TEST, "test F0.5 (held out)")]:
        ep, m, s = _stack(folds_hist, key)
        axL.plot(ep, m, color=color, lw=2, label=lab)
        axL.fill_between(ep, m - s, m + s, color=color, alpha=0.15)
    ep, m, _ = _stack(folds_hist, "test_r1")
    axL.plot(ep, m, color=C_TEST, lw=1, ls=":", alpha=0.7, label="test R@1 (diag)")
    axL.axhline(ref["raw_f"], color=C_RAW, ls="--", lw=1.3, label=f"raw voting ({ref['raw_f']:.2f})")
    axL.axhline(ref["logreg_f"], color=C_LOGREG, ls="--", lw=1.3, label=f"logreg ({ref['logreg_f']:.2f})")
    axL.set_title(f"{name}  -  pairwise F0.5"); axL.set_xlabel("epoch"); axL.set_ylabel("F0.5 / R@1")
    axL.set_ylim(0, 1); axL.legend(frameon=False, fontsize=8); _style(axL)

    ep, m, s = _stack(folds_hist, "loss")
    axR.plot(ep, m, color=C_TRAIN, lw=2); axR.fill_between(ep, m - s, m + s, color=C_TRAIN, alpha=0.15)
    axR.set_title(f"{name}  -  BCE loss"); axR.set_xlabel("epoch"); axR.set_ylabel("loss"); _style(axR)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_overlay(histories, path, ref):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axhline(ref["logreg_f"], color=C_LOGREG, ls="--", lw=1.5, label=f"logreg ({ref['logreg_f']:.2f})")
    ax.axhline(ref["raw_f"], color=C_RAW, ls="--", lw=1.5, label=f"raw voting ({ref['raw_f']:.2f})")
    for (name, fh), color in zip(histories.items(), OVERLAY):
        ep, m, _ = _stack(fh, "test_f")
        ax.plot(ep, m, color=color, lw=2, label=name)
    ax.set_title("test pairwise F0.5 (held out) — all configs vs baselines")
    ax.set_xlabel("epoch"); ax.set_ylabel("F0.5"); ax.set_ylim(0, 1); _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_diagnostics(best_name, best_res, raw_res, lr_res, quality, path):
    """2x2: pairwise PR, census P/R/F0.5 vs threshold, risk-coverage (confidence vs quality),
    image-quality strata. All pooled across folds for the winning config."""
    def pool(res, k):   return np.concatenate([r[k] for r in res])
    bs, by = pool(best_res, "closed_scores"), pool(best_res, "closed_y")
    rs, ry = pool(raw_res, "closed_scores"), pool(raw_res, "closed_y")
    ls, ly = pool(lr_res, "closed_scores"), pool(lr_res, "closed_y")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]                                          # pairwise PR
    for sc, yy, color, lab in [(rs, ry, C_RAW, "raw"), (ls, ly, C_LOGREG, "logreg"),
                               (bs, by, C_TEST, best_name)]:
        c = cen.pairwise_curve(sc, yy)
        ax.plot(c["recall"], c["precision"], color=color, lw=2,
                label=f"{lab} (AP {cen.average_precision(sc, yy):.2f})")
    ax.set_title("pairwise precision-recall (match decision)")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_ylim(0, 1.02); _style(ax); ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]                                          # census P/R/F0.5 vs threshold
    top = {k: np.concatenate([r["census_top"][k] for r in best_res]) for k in ("top1", "top1_correct", "is_known")}
    sweep, cbest = cen.census_sweep(top["top1"], top["top1_correct"], top["is_known"], BETA)
    ax.plot(sweep["thr"], sweep["precision"], color=C_LOGREG, lw=2, label="precision")
    ax.plot(sweep["thr"], sweep["recall"], color=C_TRAIN, lw=2, label="recall")
    ax.plot(sweep["thr"], sweep["fbeta"], color=C_TEST, lw=2.5, label="F0.5")
    ax.axvline(cbest["thr"], color="0.4", ls=":", lw=1.2, label=f"F0.5* @ thr {cbest['thr']:.2f}")
    ax.set_title(f"census decision — {best_name} (novel-holdout)")
    ax.set_xlabel("match threshold"); ax.set_ylabel("score"); ax.set_ylim(0, 1.02); _style(ax); ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]                                          # risk-coverage: confidence vs quality
    margin = np.concatenate([r["closed_top"]["margin"] for r in best_res])
    correct = np.concatenate([r["closed_top"]["top1_correct"] for r in best_res])
    sids = np.concatenate([r["sids"] for r in best_res])
    rc = cen.risk_coverage(margin, correct)
    ax.plot(rc["coverage"], rc["accuracy"], color=C_TEST, lw=2.5, label=f"by confidence (AURC {rc['aurc']:.3f})")
    if quality:
        qv = np.array([quality.get(s, {}).get("overall_quality", np.nan) for s in sids], float)
        ok = np.isfinite(qv)
        if ok.sum() > 5:
            rq = cen.risk_coverage(qv[ok], correct[ok])
            ax.plot(rq["coverage"], rq["accuracy"], color=C_LOGREG, lw=2, ls="--",
                    label=f"by image quality (AURC {rq['aurc']:.3f})")
    ax.axhline(correct.mean(), color=C_BASE, ls=":", lw=1.2, label=f"answer-all ({correct.mean():.2f})")
    ax.set_title("risk-coverage: accuracy vs fraction answered")
    ax.set_xlabel("coverage"); ax.set_ylabel("accuracy on answered"); ax.set_ylim(0, 1.02); _style(ax); ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]                                          # image-quality strata
    if quality:
        fields = ["overall_quality", "blur_quality", "lighting_quality", "spot_extraction_quality"]
        x = np.arange(len(fields)); w = 0.25
        for j, (nm, col) in enumerate(zip(["low", "mid", "high"], [C_RAW, C_TRAIN, C_TEST])):
            vals = []
            for f in fields:
                st = cen.quality_strata(sids, correct, quality, field=f, bins=3)
                vals.append(next((s["accuracy"] for s in st if s["bin"] == nm), np.nan))
            ax.bar(x + (j - 1) * w, vals, w, color=col, label=nm)
        ax.set_xticks(x); ax.set_xticklabels([f.replace("_quality", "").replace("_extraction", "") for f in fields], fontsize=8)
        ax.set_title("R@1 by image-quality tercile"); ax.set_ylabel("R@1"); ax.set_ylim(0, 1.02); _style(ax)
        ax.legend(frameon=False, fontsize=8, title="quality")
    else:
        ax.text(0.5, 0.5, "no image_quality table", ha="center", va="center"); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_population_bias(best_name, best_res, path):
    """Population-count bias vs threshold: inflation (missed re-sights -> phantom individuals)
    vs deflation (new animals absorbed into existing profiles). Net bias = inflation - deflation."""
    top = {k: np.concatenate([r["census_top"][k] for r in best_res]) for k in ("top1", "top1_correct", "is_known")}
    grid = np.unique(np.quantile(top["top1"], np.linspace(0, 1, 120)))
    infl, defl = [], []
    for t in grid:
        cc = cen.census_confusion(top["top1"], top["top1_correct"], top["is_known"], t)
        infl.append(cc["inflation"]); defl.append(-cc["deflation"])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axhline(0, color="0.4", lw=1)
    ax.plot(grid, infl, color=C_TRAIN, lw=2, label="inflation (+ phantom new individuals)")
    ax.plot(grid, defl, color=C_RAW, lw=2, label="deflation (- animals absorbed)")
    ax.plot(grid, np.array(infl) + np.array(defl), color=C_TEST, lw=2.5, label="net count bias")
    ax.set_title(f"population-count bias vs match threshold — {best_name}")
    ax.set_xlabel("match threshold"); ax.set_ylabel("individuals miscounted"); _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_variance(var, path):
    """Cross-seed / cross-fold spread of the ref config vs baselines (does it generalize, or is
    fold-3's collapse a split artifact?)."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in [(axL, "r1", "R@1 (ranking diagnostic)"),
                           (axR, "census_bestf", "census F0.5 (deployment)")]:
        labels = list(var.keys())
        data = [var[l][key] for l in labels]
        colors = [C_TEST if l == "set(ref)" else (C_LOGREG if l == "logreg" else C_RAW) for l in labels]
        parts = ax.boxplot(data, positions=range(1, len(labels) + 1), showmeans=True,
                           patch_artist=True, widths=0.6)
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels)
        for patch, col in zip(parts["boxes"], colors):
            patch.set_facecolor(col); patch.set_alpha(0.35)
        for i, dd in enumerate(data, 1):
            ax.scatter(np.full(len(dd), i) + np.random.uniform(-0.08, 0.08, len(dd)), dd,
                       color="0.2", s=14, zorder=3)
        ax.set_title(title); ax.set_ylim(0, 1); _style(ax)
    fig.suptitle("cross-seed variance (each point = one fold across CV-split seeds)", y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


# ============================================================ variance study
def run_variance(sets, configs_ref, *, seeds, epochs, data):
    """Re-run the reference config + baselines under several CV-split seeds -> per-fold spread."""
    var = {"set(ref)": {"r1": [], "census_bestf": []},
           "logreg":   {"r1": [], "census_bestf": []},
           "raw":      {"r1": [], "census_bestf": []}}
    for s in seeds:
        folds = d.get_cv_folds(sets, k=5, seed=s)
        res, _, raw_res, lr_res = run_cv(sets, folds, configs_ref, seed=s, epochs=epochs,
                                         data=data, log_every=0, quiet=True)
        name = configs_ref[0]["name"]
        for k in ("r1", "census_bestf"):
            var["set(ref)"][k] += [r[k] for r in res[name]]
            var["logreg"][k] += [r[k] for r in lr_res]
            var["raw"][k] += [r[k] for r in raw_res]
        logger.info(f"  seed {s}: set R@1 {np.mean([r['r1'] for r in res[name]]):.3f} "
              f"censusF {np.mean([r['census_bestf'] for r in res[name]]):.3f}")
    return var


# ============================================================ report + RESULTS.md
def summary_rows(results, raw_res, lr_res):
    """One row per config + the two baselines: mean+-std of each metric across folds."""
    rows = []
    def row(name, res):
        return dict(name=name,
                    r1=agg(res, "r1"), pair=agg(res, "pair_bestf"), honest=agg(res, "pair_honestf"),
                    ap=agg(res, "pair_ap"), census=agg(res, "census_bestf"),
                    cprec=agg(res, "census_prec"), crec=agg(res, "census_rec"), bias=agg(res, "census_bias"))
    rows.append(row("raw voting", raw_res))
    rows.append(row("logreg (champ)", lr_res))
    for name, res in results.items():
        rows.append(row(name, res))
    return rows


def print_summary(rows):
    logger.info("=" * 100)
    logger.info(f" OPEN-SET / CENSUS SWEEP  (headline = F{BETA:g}; R@1 is ranking diagnostic only)")
    logger.info("=" * 100)
    logger.info(f" {'config':<15} {'R@1':>12} {'pairF0.5':>12} {'honestF':>9} {'AP':>6} "
          f"{'censusF0.5':>12} {'cPrec':>7} {'cRec':>7} {'bias':>7}")
    logger.info("-" * 100)
    for r in rows:
        m = lambda t: f"{t[0]:.3f}±{t[1]:.3f}"
        logger.info(f" {r['name']:<15} {m(r['r1']):>12} {m(r['pair']):>12} "
              f"{r['honest'][0]:>9.3f} {r['ap'][0]:>6.3f} {m(r['census']):>12} "
              f"{r['cprec'][0]:>7.3f} {r['crec'][0]:>7.3f} {r['bias'][0]:>+7.1f}")
    logger.info("=" * 100)


def write_results_md(rows, best_name, ref, var, out_dir, cfg_by_name):
    def m(t): return f"{t[0]:.3f} ± {t[1]:.3f}"
    lines = [
        "# Open-set / census evaluation — spot aggregator",
        "",
        "Automated population census: each photo forces a **match-vs-new-individual** decision at a",
        "score threshold. A false MATCH collapses two animals and **deflates** the count (mark-recapture",
        f"can't undo it), so the headline is **F{BETA:g}** (precision weighted 2× recall). R@1 is kept only as a",
        "ranking diagnostic. Two protocols: **pairwise** (all query→candidate pairs) and a **census",
        "simulation** with novel-individual holdout that also reports population-count bias.",
        "",
        f"- Data: `{d.dataset_name}` · {DATA['neg_per_query']} neg/query · synth={DATA['use_synth']} ·",
        f"  novel holdout={DATA['novel_frac']:.0%} of eval individuals · 5-fold CV (split by individual).",
        f"- Baselines under the same metrics: raw soft-chamfer voting, summary-feature logistic regression",
        f"  (the standing R@1 champion, 0.603).",
        "",
        "## Results (mean ± std over folds)",
        "",
        "| config | R@1 (diag) | pairwise F0.5 | honest F0.5 | AP | census F0.5 | census P | census R | count bias |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['name']} | {m(r['r1'])} | {m(r['pair'])} | {r['honest'][0]:.3f} | "
                     f"{r['ap'][0]:.3f} | {m(r['census'])} | {r['cprec'][0]:.3f} | {r['crec'][0]:.3f} | "
                     f"{r['bias'][0]:+.1f} |")
    lines += [
        "",
        "- **pairwise F0.5 / AP** — precision-first quality of the match/no-match decision over all pairs.",
        "- **honest F0.5** — threshold picked on TRAIN, applied to test (deployable, not oracle).",
        "- **census F0.5 / P / R** — top-1 + threshold decision with novel individuals held out of the gallery.",
        "- **count bias** — individuals miscounted per fold = inflation (missed re-sights → phantom new)",
        "  − deflation (new animals absorbed). Positive = over-count, negative = under-count.",
        "",
        f"Best config by census F0.5: **{best_name}**.",
        "",
    ]
    if var:
        lines += [
            "## Cross-seed variance (does it generalize, or was fold-3 a split artifact?)",
            "",
            "| method | R@1 mean ± std | census F0.5 mean ± std | min fold R@1 |",
            "|---|---|---|---|",
        ]
        for lab in var:
            r1 = np.array(var[lab]["r1"], float); cf = np.array(var[lab]["census_bestf"], float)
            lines.append(f"| {lab} | {r1.mean():.3f} ± {r1.std():.3f} | {cf.mean():.3f} ± {cf.std():.3f} | {r1.min():.3f} |")
        lines.append("")
    lines += [
        "## Plots",
        "- `curve_<config>.png` — per-config train/test F0.5 learning curve (fold band) + loss.",
        "- `overlay_test_f05.png` — every config's held-out F0.5 vs the baselines.",
        "- `diagnostics_best.png` — pairwise PR, census P/R/F0.5, risk-coverage (confidence vs quality), quality strata.",
        "- `population_bias.png` — inflation vs deflation vs threshold for the winner.",
        "- `variance_seeds.png` — cross-seed/fold spread of the ref config vs baselines.",
        "",
    ]
    (out_dir / "RESULTS_census.md").write_text("\n".join(lines), encoding="utf-8")


def main(quick=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sets = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))
    sets = d.apply_quality_filter(sets)                       # no-op unless MIN_QUALITY/MAX_SPOTS_OUTSIDE set
    quality = cen.load_image_quality()
    logger.info(f" images={len(sets)}  image_quality rows={len(quality)}  beta={BETA}")

    configs = CONFIGS[:4] if quick else CONFIGS
    epochs = 20 if quick else 120
    seeds = [0, 1] if quick else [0, 1, 2]
    folds = d.get_cv_folds(sets, k=5, seed=0)

    results, histories, raw_res, lr_res = run_cv(sets, folds, configs, seed=0, epochs=epochs,
                                                 data=DATA, log_every=(5 if quick else 20))
    ref = dict(raw_f=agg(raw_res, "pair_bestf")[0], logreg_f=agg(lr_res, "pair_bestf")[0])

    rows = summary_rows(results, raw_res, lr_res)
    print_summary(rows)
    best_name = max(results, key=lambda n: agg(results[n], "census_bestf")[0])

    # The cross-seed variance study re-runs the ref config across several CV-split seeds — the
    # single most expensive part of the sweep (~3x the main cost). Off by default for the lean run
    # (5-fold CV already gives a spread); set RUN_VARIANCE=1 to bring it back.
    var = {}
    if os.environ.get("RUN_VARIANCE"):
        logger.info(" variance study (ref config across CV-split seeds)...")
        var = run_variance(sets, [c for c in configs if c["name"] == "deepsets_ref"] or [configs[0]],
                           seeds=seeds, epochs=epochs, data=DATA)

    logger.info(f" writing plots -> {OUT_DIR}")
    for name, fh in histories.items():
        plot_curve(name, fh, OUT_DIR / f"curve_{name}.png", ref)
    plot_overlay(histories, OUT_DIR / "overlay_test_f05.png", ref)
    plot_diagnostics(best_name, results[best_name], raw_res, lr_res, quality, OUT_DIR / "diagnostics_best.png")
    plot_population_bias(best_name, results[best_name], OUT_DIR / "population_bias.png")
    if var:
        plot_variance(var, OUT_DIR / "variance_seeds.png")
    write_results_md(rows, best_name, ref, var, OUT_DIR, {c["name"]: c for c in configs})
    logger.info(f" done. summary + curves + RESULTS_census.md in {OUT_DIR}")


if __name__ == "__main__":
    main(quick=bool(os.environ.get("QUICK")))


