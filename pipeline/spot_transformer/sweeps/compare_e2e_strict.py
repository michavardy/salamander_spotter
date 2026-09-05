"""Does a retrained e2e transformer (special-spot gate + default-no-match) beat the training-free
strict thesis?  — the second half of "both, in order".

Trains three e2e variants and scores them in the SAME open-set census as ``compare_strict``, with
``strict_hand_pos`` (the current best: distinctiveness-weighted coverage x support + position gate)
as the anchor to beat:

  e2e_strict_xf       transformer encoder + learned gate (supervised by clicks) + strict vote
  e2e_strict_frozen   SAME, encoder FROZEN — isolates the gate+objective from re-embedding
  e2e_strict_nogate   transformer + gate but gate_lambda=0 — ablation: does human supervision help?

Also reports **gate AUROC** — whether the learned per-spot gate recovers the human interesting-spot
labels on held-out animals.

e2e is CPU-heavy, so this runs FEWER folds/epochs than ``compare_strict`` by default (tune with env
vars). It is a directional test of the transformer route, not a final leaderboard.

    QUICK=1 pixi run python pipeline/spot_transformer/sweeps/compare_e2e_strict.py
    EPOCHS=25 E2E_NEG=30 KFOLDS=5 pixi run python .../compare_e2e_strict.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval", "sweeps"):
    p = str(_ST / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

import data as d                                            # noqa: E402
import census as cen                                        # noqa: E402
import distinctiveness as dist                              # noqa: E402
import strict_voter as sv                                   # noqa: E402
import compare_strict as cs                                 # noqa: E402  (shared eval helpers)
import review_labels as rl                                  # noqa: E402
import aggregator_e2e_strict as e2s                         # noqa: E402
from aggregator import attach_centroids, train_aggregator   # noqa: E402
from aggregator_e2e import build_e2e_pairs, build_e2e_openset_pairs  # noqa: E402

QUICK = bool(os.environ.get("QUICK"))
VERBOSE = bool(os.environ.get("VERBOSE"))       # per-epoch train loss + per-variant timing
K_FOLDS = 2 if QUICK else int(os.environ.get("KFOLDS", "3"))
EPOCHS = 3 if QUICK else int(os.environ.get("EPOCHS", "20"))
E2E_NEG = 8 if QUICK else int(os.environ.get("E2E_NEG", "25"))
SEED = 0
NOVEL_FRAC = 0.35
# Which interesting-spot labels supervise the gate: the live review store (default) or the legacy
# union export, which was 92 images / 452 clicks stale. Running both is the label-refresh experiment
# for the ONE place the clicks have a measured causal effect (results.md #31: removing this
# supervision drops census F0.5 0.609 -> 0.493 and novelty AUROC to chance).
LABELS = os.environ.get("LABELS", "review")
EVAL_GATE = os.environ.get("EVAL_GATE", "quality")
UNREVIEWED = os.environ.get("UNREVIEWED", "keep")

E2E_VARIANTS = [
    dict(name="e2e_strict_xf",     arch="transformer", gate_lambda=0.5),
    dict(name="e2e_strict_frozen", arch="frozen",      gate_lambda=0.5),
    dict(name="e2e_strict_nogate", arch="transformer", gate_lambda=0.0),
]
MODELS = ["strict_hand_pos"] + [v["name"] for v in E2E_VARIANTS]


def run_fold(sets, frame, pos_lookup, tr, ev):
    train_imgs = [i for i in tr if not sets[i].is_synth]
    train_labels = {sets[i].label for i in train_imgs}
    gal, qry = cen.make_openset_split(sets, ev, NOVEL_FRAC, SEED)
    true_by_q = {q: sets[q].label for q in qry}
    out = {}

    # --- anchor: strict_hand_pos (distinctiveness weights fit on train individuals only) ---
    ind = np.array(["_".join(s.split("_")[:2]) for s in frame["sid"]])
    fit_mask = frame["labeled"].to_numpy() & np.isin(ind, list(train_labels))
    dm, dsc = train_aggregator(frame.loc[fit_mask, dist.FACTOR_NAMES].to_numpy(float),
                               frame.loc[fit_mask, "y"].to_numpy(float), hidden=0, seed=SEED)
    wlookup = dist.weight_lookup(dm, dsc, frame)
    s_p, pq, pc = sv.build_openset_hand(sets, gal, qry, wlookup,
                                        pos_lookup=pos_lookup, sigma_pos=cs.SIGMA_POS)
    out["strict_hand_pos"] = cs._census_row(s_p, pq, pc, true_by_q)

    # --- e2e variants ---
    pairs_tr, y_tr, _, _ = build_e2e_pairs(sets, train_imgs, neg_per_query=E2E_NEG, seed=SEED)
    pairs_os, oq, oc = build_e2e_openset_pairs(sets, gal, qry)
    if VERBOSE:
        logger.info(f"   anchor strict_hand_pos F{out['strict_hand_pos']['census_f']:.3f}  ·  "
              f"{len(pairs_tr)} train pairs, {len(pairs_os)} open-set pairs")
    for v in E2E_VARIANTS:
        if VERBOSE:
            logger.info(f"   [{v['name']}] {EPOCHS} epochs ...")
        t_v = time.time()
        model = e2s.train_strict_e2e(sets, pairs_tr, y_tr, arch=v["arch"], epochs=EPOCHS,
                                     gate_lambda=v["gate_lambda"], pos_weight=None, seed=SEED,
                                     verbose=VERBOSE)
        sc = e2s.score_strict_e2e(model, sets, pairs_os)
        row = cs._census_row(sc, oq, oc, true_by_q)
        row["gate_auroc"] = e2s.gate_auroc(model, sets, qry)
        out[v["name"]] = row
        if VERBOSE:
            logger.info(f"   [{v['name']}] F{row['census_f']:.3f}  gate {row['gate_auroc']:.3f}"
                  f"  ({time.time()-t_v:.0f}s)")
    return out


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    logger.info("loading data + factors + interesting labels ...")
    logger.info(" " + rl.review_summary(d.dataset_name))
    sets = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))
    labels = dist.load_interesting(path=dist.INTERESTING_JSON if LABELS == "legacy" else None)
    e2s.attach_interesting(sets, labels)
    factors = dist.load_spot_factors(sets)
    frame = dist.attach_labels(factors, labels)
    pos_lookup = cs.build_pos_lookup(frame)
    e2s.attach_positions(sets, pos_lookup)          # the strict vote's position gate needs per-spot xy
    logger.info(f" labels={LABELS} ({frame['sid'][frame['labeled']].nunique()} labeled images, "
          f"{int(frame['y'].sum())} interesting spots)  eval_gate={EVAL_GATE}")

    eval_mask = np.ones(len(sets), dtype=bool)
    if EVAL_GATE in ("quality", "both"):
        eval_mask &= d.quality_keep_mask(sets)
    if EVAL_GATE in ("review", "both"):
        eval_mask &= d.review_keep_mask(sets, unreviewed=UNREVIEWED)
    folds = d.get_cv_folds(sets, k=K_FOLDS, seed=SEED, eval_mask=eval_mask)
    d.assert_evaluable(folds, what=f"EVAL_GATE={EVAL_GATE}")

    logger.info("=" * 74)
    logger.info(f" E2E-STRICT vs strict_hand_pos  ({len(MODELS)} models x {K_FOLDS} folds, "
          f"{EPOCHS} ep, neg {E2E_NEG}{'  [QUICK]' if QUICK else ''})   headline = census F0.5")
    logger.info("=" * 74)

    per_fold = {m: [] for m in MODELS}
    for fi, (tr, ev) in enumerate(folds):
        t0 = time.time()
        res = run_fold(sets, frame, pos_lookup, tr, ev)
        for m, v in res.items():
            per_fold[m].append(v)
        line = "  ".join(f"{m.replace('e2e_strict_','e2e_')}:F{res[m]['census_f']:.3f}"
                         for m in MODELS)
        logger.info(f" fold {fi}  {line}   ({time.time()-t0:.0f}s)")

    def agg(m, k):
        v = [r[k] for r in per_fold[m] if k in r]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), 0.0)

    logger.info("=" * 104)
    logger.info(f" {'model':<18}{'censusF0.5':>12}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'AUROC':>8}"
          f"{'cov@P90':>9}{'gate':>7}")
    logger.info(" (transformer = learns its own embedding on top of the 62-dim; frozen = uses it as-is)")
    logger.info("-" * 104)
    rows = []
    for m in MODELS:
        fm, fs = agg(m, "census_f")
        rows.append((m, fm, fs, agg(m, "ident_r1")[0], agg(m, "r5")[0], agg(m, "r10")[0],
                     agg(m, "auroc_top1")[0], agg(m, "cov_at_p")[0], agg(m, "gate_auroc")[0]))
    for m, fm, fs, ir, r5, r10, at, cov, ga in sorted(rows, key=lambda x: -x[1]):
        ga_s = f"{ga:.3f}" if np.isfinite(ga) else "  -  "
        logger.info(f" {m:<18}{fm:>7.3f}±{fs:<4.3f}{ir:>7.3f}{r5:>7.3f}{r10:>7.3f}{at:>8.3f}"
              f"{cov:>9.3f}{ga_s:>7}")

    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "sweeps" / "strict"
    outdir.mkdir(parents=True, exist_ok=True)
    md = ["# e2e-strict (does the model learning its OWN spot embedding help?) vs strict_hand_pos", "",
          f"- dataset `{d.dataset_name}` · folds {K_FOLDS} · {EPOCHS} epochs · e2e neg {E2E_NEG}"
          f" · quality `{d.quality_tag() or 'none'}`"
          + ("  **[QUICK — not a result]**" if QUICK else ""),
          f"- gate labels `{LABELS}` · eval gate `{EVAL_GATE}` · {rl.review_summary(d.dataset_name)}",
          "- **e2e_strict_xf** = transformer re-embeds the 62-dim (learns its own representation);",
          "  **e2e_strict_frozen** = uses the 62-dim as-is (only gate + vote train);",
          "  **e2e_strict_nogate** = transformer, but no human special-spot supervision (ablation).",
          "- gate = does the learned per-spot gate recover the human interesting-spot labels? "
          "cov@P90 = abstain coverage at >=90% precision.",
          "",
          "| model | census F0.5 | R@1 | R@5 | R@10 | AUROC | cov@P90 | gate |",
          "|---|---|---|---|---|---|---|---|"]
    for m, fm, fs, ir, r5, r10, at, cov, ga in sorted(rows, key=lambda x: -x[1]):
        ga_s = f"{ga:.3f}" if np.isfinite(ga) else "—"
        md.append(f"| {m} | **{fm:.3f} ± {fs:.3f}** | {ir:.3f} | {r5:.3f} | {r10:.3f} | {at:.3f} "
                  f"| {cov:.3f} | {ga_s} |")
    fname = ("RESULTS_e2e_strict" + d.quality_tag()
             + ("" if LABELS == "review" else f"_{LABELS}")
             + ("" if EVAL_GATE == "quality" else f"_gate{EVAL_GATE}")
             + ("_quick" if QUICK else "") + ".md")
    (outdir / fname).write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info(f"wrote {outdir / fname}")


if __name__ == "__main__":
    main()
