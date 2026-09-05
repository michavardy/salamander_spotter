from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # run as script (path[0] = this dir) OR imported as a package
    import data as d
    from train_spot import train_one_fold_spot
except ModuleNotFoundError:
    from pipeline.spot_transformer import data as d
    from pipeline.spot_transformer.train_spot import train_one_fold_spot

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

OUT_DIR = Path(__file__).resolve().parent / "sweep_results_spot"

# Okabe-Ito (colorblind-safe): fixed roles for the per-config curves + a list for the overlay.
C_TRAIN = "#E69F00"   # orange
C_EVAL  = "#0072B2"   # blue
C_BASE  = "#999999"   # neutral grey = raw-spot baseline reference line
OVERLAY = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]

# Spot-level sweep. Metric = VOTING R@1; bar = raw-spot voting. The point of this sweep is the
# head-to-head: contextual TRANSFORMER (T_) vs context-free MLP (M_) -- "is it learning, or
# is it context?" If M_ beats raw voting where T_ can't, context was the problem.
CONFIGS = [
    dict(name="T_ref",     model_type="transformer", n_layers=1, d_model=128, dropout_p=0.3, model_dropout=0.1, temperature=0.07),
    dict(name="T_small",   model_type="transformer", n_layers=1, d_model=96,  dropout_p=0.3, model_dropout=0.2, temperature=0.07),
    dict(name="M_ref",     model_type="mlp",         d_model=128, dropout_p=0.3, model_dropout=0.1, temperature=0.07),
    dict(name="M_lowtemp", model_type="mlp",         d_model=128, dropout_p=0.3, model_dropout=0.1, temperature=0.05),
    dict(name="M_small",   model_type="mlp",         d_model=96,  dropout_p=0.3, model_dropout=0.2, temperature=0.07),
    dict(name="M_wide",    model_type="mlp",         d_model=256, dropout_p=0.3, model_dropout=0.2, temperature=0.07),
]


def plot_curves(name, rep, path):
    """Per-config: LEFT = voting R@1 train vs eval (+ raw-spot baseline), RIGHT = SupCon loss."""
    h = rep["history"]
    ep = [x["epoch"] for x in h]
    tr = [x["train"]["recall@1"] for x in h]
    ev = [x["eval"]["recall@1"] for x in h]
    loss = [x["loss"] for x in h]
    base = rep["baseline"]["recall@1"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    axL.axhline(base, color=C_BASE, ls="--", lw=1.5, label=f"raw-spot voting ({base:.2f})")
    axL.plot(ep, tr, color=C_TRAIN, lw=2, marker="o", ms=4, label="train vote R@1")
    axL.plot(ep, ev, color=C_EVAL, lw=2, marker="o", ms=4, label="eval vote R@1 (held out)")
    axL.axvline(rep["best_epoch"], color=C_EVAL, lw=1, alpha=0.35)
    axL.set_title(f"{name}  -  voting R@1"); axL.set_xlabel("epoch"); axL.set_ylabel("recall@1")
    axL.set_ylim(0, 1); axL.grid(True, color="0.9", lw=0.6); axL.set_axisbelow(True)
    axL.legend(frameon=False, fontsize=9)

    axR.plot(ep, loss, color=C_TRAIN, lw=2, marker="o", ms=4, label="train loss")
    axR.set_title(f"{name}  -  spot SupCon loss"); axR.set_xlabel("epoch"); axR.set_ylabel("loss")
    axR.grid(True, color="0.9", lw=0.6); axR.set_axisbelow(True); axR.legend(frameon=False, fontsize=9)
    for ax in (axL, axR):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_overlay(reports, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    base = next(iter(reports.values()))["baseline"]["recall@1"]
    ax.axhline(base, color=C_BASE, ls="--", lw=1.5, label=f"raw-spot voting ({base:.2f})")
    for (name, rep), color in zip(reports.items(), OVERLAY):
        ep = [x["epoch"] for x in rep["history"]]
        ev = [x["eval"]["recall@1"] for x in rep["history"]]
        ax.plot(ep, ev, color=color, lw=2, marker="o", ms=3, label=name)
    ax.set_title("spot-level sweep - eval voting R@1 (held out)")
    ax.set_xlabel("epoch"); ax.set_ylabel("recall@1"); ax.set_ylim(0, 1)
    ax.grid(True, color="0.9", lw=0.6); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def report_block(name, rep):
    base, pre, best = rep["baseline"], rep["pretrain"], rep["best"]
    logger.info(f"---- {name} " + "-" * (60 - len(name)))
    c = rep["config"]
    arch = f"{c['n_layers']}L d{c['d_model']}" if c["model_type"] == "transformer" else f"MLP h{c['d_model']}"
    logger.info(f"   config      : [{c['model_type']}] {arch}  "
          f"drop={c['dropout_p']} temp={c['temperature']}")
    logger.info(f"   BEFORE (untrained model) : vote R@1={pre['recall@1']:.3f}")
    logger.info(f"   BASELINE (raw-spot vote) : R@1={base['recall@1']:.3f}  R@5={base['recall@5']:.3f}")
    logger.info(f"   AFTER  (best ep {rep['best_epoch']:>2})      : R@1={best['recall@1']:.3f}  "
          f"R@5={best['recall@5']:.3f}   match-AUC {rep['baseline_auc']:.3f}->{rep['best_auc']:.3f}")
    dR1 = best["recall@1"] - base["recall@1"]
    logger.info(f"   vs raw-spot baseline: R@1 {dR1:+.3f}   "
          f"{'BEATS raw spots' if dR1 > 0 else 'below raw spots'}")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    sets = d.get_image_sets(d.get_spot_embeddings())
    train_idx, eval_idx = d.get_cv_folds(sets, k=5, seed=0)[0]

    reports = {}
    for cfg in CONFIGS:
        name = cfg.pop("name")
        logger.info(f"########## training {name} ##########")
        _, rep = train_one_fold_spot(sets, train_idx, eval_idx, P=16, K=4, num_batches=50,
                                     epochs=30, lr=1e-3, patience=3, seed=0, verbose=False, **cfg)
        reports[name] = rep
        report_block(name, rep)
        plot_curves(name, rep, OUT_DIR / f"curve_{name}.png")

    plot_overlay(reports, OUT_DIR / "overlay_eval_r1.png")

    base_r1 = next(iter(reports.values()))["baseline"]["recall@1"]
    logger.info("=" * 82)
    logger.info(f" SPOT-LEVEL SWEEP SUMMARY   (raw-spot voting baseline R@1 = {base_r1:.3f})")
    logger.info("=" * 82)
    logger.info(" config        best_ep  untrained  AFTER   dR@1(vs raw)  R@5    match-AUC   verdict")
    logger.info("-" * 82)
    for name, rep in sorted(reports.items(), key=lambda kv: -kv[1]["best"]["recall@1"]):
        base, pre, best = rep["baseline"], rep["pretrain"], rep["best"]
        dR1 = best["recall@1"] - base["recall@1"]
        verdict = "BEATS" if dR1 > 0 else "below"
        logger.info(f" {name:<13} {rep['best_epoch']:>5}     {pre['recall@1']:.3f}     {best['recall@1']:.3f}   "
              f"{dR1:+.3f}       {best['recall@5']:.3f}   {rep['best_auc']:.3f}     {verdict}")
    logger.info("=" * 82)
    logger.info(f" curves + overlay written to: {OUT_DIR}")
    breakpoint()   # live: reports, sets
