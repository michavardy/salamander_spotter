from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # headless: write PNGs, no window
import matplotlib.pyplot as plt

import numpy as np

try:  # run as script (path[0] = this dir) OR imported as a package
    import data as d
    from train import train_one_fold
    from eval import (encode_baseline, encode_model,
                      match_precision_at_thresholds, print_match_table, match_auc)
except ModuleNotFoundError:
    from pipeline.spot_transformer import data as d
    from pipeline.spot_transformer.train import train_one_fold
    from pipeline.spot_transformer.eval import (encode_baseline, encode_model,
                                                match_precision_at_thresholds, print_match_table, match_auc)

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

OUT_DIR = Path(__file__).resolve().parent / "sweep_results"

# Okabe-Ito: a peer-reviewed colorblind-safe categorical palette. Two fixed roles for the
# per-config curves (train vs eval), then a distinct list for the overlay (one hue per config).
C_TRAIN = "#E69F00"    # orange
C_EVAL  = "#0072B2"    # blue
C_BASE  = "#999999"    # neutral grey = reference line, never a series
OVERLAY = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]

# The regularization sweep. Each dict overrides train_one_fold defaults; every config
# targets the overfitting gap we saw (train R@1 >> eval R@1). Aggressive early stopping
# (patience=3) stops each run as soon as held-out R@1 stalls.
CONFIGS = [
    dict(name="A_ref",          dropout_p=0.2, jitter_std=0.0,  n_layers=2, d_model=128, model_dropout=0.1),
    dict(name="B_heavy_drop",   dropout_p=0.4, jitter_std=0.0,  n_layers=2, d_model=128, model_dropout=0.2),
    dict(name="C_jitter",       dropout_p=0.3, jitter_std=0.03, n_layers=2, d_model=128, model_dropout=0.1),
    dict(name="D_shallow",      dropout_p=0.3, jitter_std=0.0,  n_layers=1, d_model=128, model_dropout=0.1),
    dict(name="E_small",        dropout_p=0.3, jitter_std=0.0,  n_layers=1, d_model=96,  model_dropout=0.2),
    dict(name="F_all_reg",      dropout_p=0.4, jitter_std=0.03, n_layers=1, d_model=96,  model_dropout=0.2),
]


def _r1(m):  # helper: recall@1 out of a metrics dict
    return m["recall@1"]


def plot_curves(name: str, report: dict, path: Path) -> None:
    """Per-config learning curves: LEFT = R@1 train vs eval (+ baseline reference),
    RIGHT = SupCon train loss. Two panels, never a dual y-axis (loss and R@1 don't share
    a scale)."""
    hist = report["history"]
    ep = [h["epoch"] for h in hist]
    tr = [h["train"]["recall@1"] for h in hist]
    ev = [h["eval"]["recall@1"] for h in hist]
    loss = [h["loss"] for h in hist]
    base_r1 = report["baseline"]["recall@1"]
    best_ep = report["best_epoch"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    axL.axhline(base_r1, color=C_BASE, ls="--", lw=1.5, label=f"mean-pool baseline ({base_r1:.2f})")
    axL.plot(ep, tr, color=C_TRAIN, lw=2, marker="o", ms=4, label="train R@1")
    axL.plot(ep, ev, color=C_EVAL, lw=2, marker="o", ms=4, label="eval R@1 (held out)")
    axL.axvline(best_ep, color=C_EVAL, lw=1, alpha=0.35)                 # best epoch marker
    axL.set_title(f"{name}  -  retrieval R@1")
    axL.set_xlabel("epoch"); axL.set_ylabel("recall@1"); axL.set_ylim(0, 1)
    axL.grid(True, color="0.9", lw=0.6); axL.set_axisbelow(True)
    axL.legend(frameon=False, fontsize=9, loc="best")

    axR.plot(ep, loss, color=C_TRAIN, lw=2, marker="o", ms=4, label="train loss")
    axR.set_title(f"{name}  -  SupCon loss")
    axR.set_xlabel("epoch"); axR.set_ylabel("loss")
    axR.grid(True, color="0.9", lw=0.6); axR.set_axisbelow(True)
    axR.legend(frameon=False, fontsize=9, loc="best")

    for ax in (axL, axR):        # recessive spines
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(reports: dict[str, dict], path: Path) -> None:
    """One eval-R@1 curve per config, overlaid, so you can compare which config
    generalizes best without overfitting."""
    fig, ax = plt.subplots(figsize=(8, 5))
    base_r1 = next(iter(reports.values()))["baseline"]["recall@1"]
    ax.axhline(base_r1, color=C_BASE, ls="--", lw=1.5, label=f"baseline ({base_r1:.2f})")
    for (name, rep), color in zip(reports.items(), OVERLAY):
        ep = [h["epoch"] for h in rep["history"]]
        ev = [h["eval"]["recall@1"] for h in rep["history"]]
        ax.plot(ep, ev, color=color, lw=2, marker="o", ms=3, label=name)
    ax.set_title("regularization sweep - eval R@1 (held out)")
    ax.set_xlabel("epoch"); ax.set_ylabel("recall@1"); ax.set_ylim(0, 1)
    ax.grid(True, color="0.9", lw=0.6); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def crossval_match_report(sets, *, cutoffs=(0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9),
                          k=5, seed=0, **train_kwargs):
    """Dataset-wide, HONEST match-precision table (the true analog of the spot-level
    naive_match_all table). Trains all ``k`` folds; for each, embeds its HELD-OUT images with
    that fold's model, and pools the WITHIN-fold match counts across folds (cross-fold pairs
    are dropped -- different models = different, non-comparable cosine spaces). Returns
    ``(before_rows, after_rows)`` for mean-pool vs trained, and prints both."""
    folds = d.get_cv_folds(sets, k=k, seed=seed)
    acc = {"before": {c: [0, 0] for c in cutoffs}, "after": {c: [0, 0] for c in cutoffs}}  # [n_above, n_true]
    aucs = {"before": [], "after": []}                                     # per-fold, threshold-free
    total_true = 0

    for fi, (tr, ev) in enumerate(folds):
        logger.info(f"# fold {fi}: training (held out {len(ev)} imgs)...")
        model, _ = train_one_fold(sets, tr, ev, verbose=False, seed=seed, **train_kwargs)
        y = np.array([sets[i].label for i in ev])
        total_true += int(np.triu(y[:, None] == y[None, :], 1).sum())      # same-individual pairs in this fold
        for tag, (Z, yy) in {"before": encode_baseline(sets, ev),
                             "after": encode_model(model, sets, ev)}.items():
            for r in match_precision_at_thresholds(Z, yy, cutoffs):
                acc[tag][r["cutoff"]][0] += r["n_above"]
                acc[tag][r["cutoff"]][1] += r["n_true"]
            aucs[tag].append(match_auc(Z, yy))                             # same-space, per fold

    def rows_for(tag):
        out = []
        for c in cutoffs:
            n_above, n_true = acc[tag][c]
            out.append(dict(cutoff=c, n_above=n_above, n_true=n_true,
                            precision=(n_true / n_above if n_above else float("nan")),
                            recall=(n_true / total_true if total_true else float("nan"))))
        return out

    before, after = rows_for("before"), rows_for("after")
    auc_b, auc_a = float(np.nanmean(aucs["before"])), float(np.nanmean(aucs["after"]))
    logger.info("=" * 78)
    logger.info(f" DATASET-WIDE cross-validated held-out matching ({total_true} true pairs pooled)")
    logger.info("=" * 78)
    logger.info(f" match AUC (threshold-free, comparable):  BEFORE {auc_b:.3f}   AFTER {auc_a:.3f}   "
          f"{'IMPROVED' if auc_a > auc_b else 'no gain'} ({auc_a-auc_b:+.3f})")
    logger.info("-" * 78)
    print_match_table(before, " BEFORE (mean-pool):  [read at matched RECALL, not fixed cutoff]")
    print_match_table(after, " AFTER  (trained):")
    return before, after


def report_block(name: str, rep: dict) -> None:
    """Per-config before/after report: baseline + untrained (BEFORE) vs best (AFTER)."""
    base, pre, best = rep["baseline"], rep["pretrain"], rep["best"]
    logger.info(f"---- {name} " + "-" * (60 - len(name)))
    logger.info(f"   config      : {rep['config']['n_layers']}L d{rep['config']['d_model']}  "
          f"drop={rep['config']['dropout_p']} jit={rep['config']['jitter_std']} "
          f"mdrop={rep['config']['model_dropout']}")
    logger.info(f"   BEFORE (untrained) : R@1={pre['recall@1']:.3f}  R@5={pre['recall@5']:.3f}  mAP={pre['mAP']:.3f}")
    logger.info(f"   BASELINE (meanpool): R@1={base['recall@1']:.3f}  R@5={base['recall@5']:.3f}  mAP={base['mAP']:.3f}")
    logger.info(f"   AFTER  (best ep {rep['best_epoch']:>2}) : R@1={best['recall@1']:.3f}  "
          f"R@5={best['recall@5']:.3f}  mAP={best['mAP']:.3f}")
    dR1 = best["recall@1"] - base["recall@1"]
    logger.info(f"   improvement vs baseline: R@1 {dR1:+.3f}   "
          f"{'BEATS baseline' if dR1 > 0 else 'below baseline'}")
    ab, aa = rep["match_auc_before"], rep["match_auc_after"]
    logger.info(f"   match AUC (eval): before {ab:.3f} -> after {aa:.3f}  ({aa-ab:+.3f})")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    sets = d.get_image_sets(d.get_spot_embeddings())
    train_idx, eval_idx = d.get_cv_folds(sets, k=5, seed=0)[0]

    reports: dict[str, dict] = {}
    for cfg in CONFIGS:
        name = cfg.pop("name")
        logger.info(f"########## training {name} ##########")
        _, rep = train_one_fold(
            sets, train_idx, eval_idx,
            P=16, K=4, num_batches=50, epochs=30,
            lr=1e-3, temperature=0.1,
            patience=3,                       # aggressive early stopping
            seed=0, verbose=False,            # per-config report + curves instead of per-epoch spam
            **cfg,
        )
        reports[name] = rep
        report_block(name, rep)
        plot_curves(name, rep, OUT_DIR / f"curve_{name}.png")

    # overlay + comparison table
    plot_overlay(reports, OUT_DIR / "overlay_eval_r1.png")

    logger.info("=" * 78)
    logger.info(" SWEEP SUMMARY   (baseline R@1 = %.3f)" % next(iter(reports.values()))["baseline"]["recall@1"])
    logger.info("=" * 78)
    logger.info(" config         best_ep  R@1_before  R@1_after  dR@1   R@5    mAP   mAUC   verdict")
    logger.info("-" * 78)
    for name, rep in sorted(reports.items(), key=lambda kv: -_r1(kv[1]["best"])):
        base, pre, best = rep["baseline"], rep["pretrain"], rep["best"]
        dR1 = best["recall@1"] - base["recall@1"]
        verdict = "BEATS" if dR1 > 0 else "below"
        logger.info(f" {name:<14} {rep['best_epoch']:>5}     {pre['recall@1']:.3f}      {best['recall@1']:.3f}   "
              f"{dR1:+.3f}  {best['recall@5']:.3f}  {best['mAP']:.3f}  {rep['match_auc_after']:.3f}  {verdict}")
    logger.info("=" * 78)
    logger.info(f" curves + overlay written to: {OUT_DIR}")

    breakpoint()   # live: reports, sets
