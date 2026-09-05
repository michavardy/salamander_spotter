"""Data-scaling curve for the learned spot aggregator — the controlled "does more data help?" test.

The comparison sweep (`sweep_set.py`) re-runs configs on the new, bigger dataset and eyeballs the
numbers against the old ones — but there the *eval* set also changed, so a moving number could be an
easier/harder eval rather than the extra training data. This isolates the data axis: for each of the
5 CV folds we hold the eval fold FIXED and train on **nested** 25 / 50 / 75 / 100 % subsets of that
fold's TRAINING individuals (subset by individual, so 25% ⊂ 50% ⊂ 75% ⊂ 100% — a clean monotone
curve). If census F0.5 / R@1 is still climbing at 100 %, more images will keep paying off.

Trained model: `deepsets_jit` (the DeepSets winner). Baselines under the same eval each point:
logreg (also trained on the subset → its own scaling line) and raw soft-chamfer voting (data-free →
a flat reference). Reuses `sweep_set` / `aggregator_set` verbatim so it's the same recipe as the sweep.

    pixi run python pipeline/spot_transformer/scaling_set.py           # full run (~2-2.5 h)
    QUICK=1 pixi run python pipeline/spot_transformer/scaling_set.py   # fast smoke
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

warnings.filterwarnings("ignore", message=".*nested tensor.*")

try:
    import data as d
    import census as cen
    import sweep_set as S
    from aggregator_set import build_record_pairs
    from aggregator import attach_centroids
except ModuleNotFoundError:
    from pipeline.spot_transformer import data as d
    from pipeline.spot_transformer import census as cen
    from pipeline.spot_transformer import sweep_set as S
    from pipeline.spot_transformer.aggregator_set import build_record_pairs
    from pipeline.spot_transformer.aggregator import attach_centroids

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

OUT_DIR = d.REPO_ROOT / "artifacts" / "spot_transformer" / "sweeps" / f"scaling{d.quality_tag()}"
BETA = S.BETA
CFG = dict(name="deepsets_jit", arch="deepsets", h=32, dropout=0.3, weight_decay=1e-2, l1=0.0, jitter=0.02)
DATA = S.DATA
C_JIT, C_LOGREG, C_RAW = "#0072B2", "#009E73", "#CC79A7"


def subset_train(sets, tr, frac, seed):
    """Nested subsample of the fold's TRAIN individuals to `frac` (all images of the kept ones).
    Nested: a fixed seeded permutation, prefix at the cutoff, so smaller fracs are subsets of larger."""
    if frac >= 1.0:
        return list(tr)
    labels = sorted({sets[i].label for i in tr})
    rng = np.random.default_rng(seed + 991)
    perm = [labels[i] for i in rng.permutation(len(labels))]
    keep = set(perm[: max(2, int(round(frac * len(labels))))])
    return [i for i in tr if sets[i].label in keep]


def one_point(sets, tr_sub, ev, *, seed, epochs, log_every=0):
    """Train deepsets_jit + baselines on tr_sub, eval on the (fixed) fold `ev`. Returns a dict of
    metrics for the learned model, logreg and raw, or None if the subset can't form a valid fold."""
    try:
        bundle = S.build_fold_bundle(sets, tr_sub, ev, seed=seed, use_synth=DATA["use_synth"],
                                     use_geom=DATA["use_geom"], novel_frac=DATA["novel_frac"],
                                     val_frac=DATA["val_frac"])
        recs, y, _, _ = build_record_pairs(sets, bundle["tr_pool"], neg_per_query=DATA["neg_per_query"],
                                           seed=seed, use_geom=DATA["use_geom"], jitter=CFG.get("jitter", 0.0))
        if int(y.sum()) < 2:                       # need same-individual positives to train on
            return None
        bundle["_train_recs"] = (recs, y)
        met, _ = S.run_model_fold(sets, ev, bundle, CFG, seed=seed, epochs=epochs,
                                  use_geom=DATA["use_geom"], log_every=log_every)
        raw, logreg = S.baseline_fold(sets, ev, bundle, seed=seed, use_geom=DATA["use_geom"])
    except Exception as e:                          # a too-thin subset on some fold — skip that point
        logger.warning(f"      (skipped: {type(e).__name__}: {e})")
        return None
    n_imgs = len(bundle["tr_pool"])
    n_ind = len({sets[i].label for i in bundle["tr_pool"]})
    pick = lambda m: dict(census=m["census_bestf"], pair=m["pair_bestf"], r1=m["r1"])
    return dict(n_imgs=n_imgs, n_ind=n_ind, jit=pick(met), logreg=pick(logreg), raw=pick(raw))


def plot_scaling(points, path):
    """Two panels — census F0.5 (deployment headline) and R@1 (ranking diag) vs #train images —
    each with the learned model, logreg and raw, mean ± std over folds."""
    fracs = sorted(points)
    x = np.array([np.mean([p["n_imgs"] for p in points[f]]) for f in fracs])

    def band(ax, method, key, color, label):
        m = np.array([np.mean([p[method][key] for p in points[f]]) for f in fracs])
        s = np.array([np.std([p[method][key] for p in points[f]]) for f in fracs])
        ax.plot(x, m, color=color, lw=2.2, marker="o", ms=5, label=label)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15)
        return m

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key, title in [(axL, "census", "census F0.5 (deployment)"),
                           (axR, "r1", "R@1 (ranking diagnostic)")]:
        mj = band(ax, "jit", key, C_JIT, "deepsets_jit (learned)")
        band(ax, "logreg", key, C_LOGREG, "logreg (champ)")
        band(ax, "raw", key, C_RAW, "raw voting (data-free)")
        for xi, mi, f in zip(x, mj, fracs):
            ax.annotate(f"{f:.0%}", (xi, mi), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=C_JIT)
        ax.set_title(title); ax.set_xlabel("# training images"); ax.set_ylabel(key)
        ax.set_ylim(0, 1); ax.grid(True, color="0.9", lw=0.6); ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Does adding images help? — nested train-size scaling, fixed eval folds", y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def write_md(points, out_dir):
    fracs = sorted(points)
    lines = [
        "# Data-scaling curve — does adding images help the learned matcher?",
        "",
        f"Controlled test on `{d.dataset_name}`: 5-fold CV, each eval fold held FIXED while the model",
        "trains on nested 25/50/75/100% subsets of that fold's training **individuals**. If the curve is",
        "still rising at 100%, more images will keep improving the matcher. `deepsets_jit` is the learned",
        "model; logreg (trained on the same subset) and raw voting (data-free) are references.",
        "",
        "| train frac | # train images | deepsets_jit census F0.5 | logreg census F0.5 | jit R@1 | logreg R@1 | raw census F0.5 |",
        "|---|---|---|---|---|---|---|",
    ]
    ms = lambda vals: f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
    for f in fracs:
        pl = points[f]
        lines.append(
            f"| {f:.0%} | {np.mean([p['n_imgs'] for p in pl]):.0f} | "
            f"{ms([p['jit']['census'] for p in pl])} | {ms([p['logreg']['census'] for p in pl])} | "
            f"{ms([p['jit']['r1'] for p in pl])} | {ms([p['logreg']['r1'] for p in pl])} | "
            f"{ms([p['raw']['census'] for p in pl])} |")
    # slope over the last two points = marginal value of the most recent data
    if len(fracs) >= 2:
        a, b = fracs[-2], fracs[-1]
        dj = np.mean([p["jit"]["census"] for p in points[b]]) - np.mean([p["jit"]["census"] for p in points[a]])
        di = np.mean([p["n_imgs"] for p in points[b]]) - np.mean([p["n_imgs"] for p in points[a]])
        lines += ["",
                  f"Marginal slope over the last step ({a:.0%}→{b:.0%}): "
                  f"**{dj:+.3f} census F0.5 per {di:.0f} extra images** "
                  f"({'still climbing — more data should help' if dj > 0.005 else 'flattening — near saturation' if abs(dj) <= 0.005 else 'declining'})."]
    lines += ["", "## Plot", "- `scaling_curve.png` — census F0.5 and R@1 vs # training images, mean ± std over folds.", ""]
    (out_dir / "RESULTS_scaling.md").write_text("\n".join(lines), encoding="utf-8")


def main(quick=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sets = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))
    sets = d.apply_quality_filter(sets)                       # no-op unless MIN_QUALITY/MAX_SPOTS_OUTSIDE set
    logger.info(f" images={len(sets)}  beta={BETA}  config={CFG['name']}")

    fracs = [0.5, 1.0] if quick else [0.25, 0.5, 0.75, 1.0]
    epochs = 20 if quick else 120
    folds = d.get_cv_folds(sets, k=5, seed=0)

    points: dict[float, list] = {f: [] for f in fracs}
    for fi, (tr, ev) in enumerate(folds):
        logger.info(f"===== fold {fi}  (train imgs={len(tr)}, eval imgs={len(ev)}) " + "=" * 20)
        for f in fracs:
            tr_sub = subset_train(sets, tr, f, seed=fi)
            res = one_point(sets, tr_sub, ev, seed=0, epochs=epochs, log_every=0)
            if res is None:
                logger.warning(f"   frac {f:.0%}: skipped (subset too thin)")
                continue
            points[f].append(res)
            logger.info(f"   frac {f:.0%}  train_imgs={res['n_imgs']:>4} ind={res['n_ind']:>3} | "
                  f"jit: censusF {res['jit']['census']:.3f} R@1 {res['jit']['r1']:.3f} | "
                  f"logreg censusF {res['logreg']['census']:.3f} | raw censusF {res['raw']['census']:.3f}")

    points = {f: pl for f, pl in points.items() if pl}    # drop fracs that never built
    logger.info(f" writing -> {OUT_DIR}")
    plot_scaling(points, OUT_DIR / "scaling_curve.png")
    write_md(points, OUT_DIR)
    logger.info(f" done. scaling_curve.png + RESULTS_scaling.md in {OUT_DIR}")


if __name__ == "__main__":
    main(quick=bool(os.environ.get("QUICK")))
