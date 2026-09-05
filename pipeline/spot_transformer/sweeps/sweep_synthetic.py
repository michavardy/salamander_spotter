"""Two questions real data is too small to answer, on synthetic populations of any size.

    Q1  DOES SCALE HELP?   -- and it is two questions, which collecting more animals conflates:
          gallery scale   more individuals to be confused BY   (task gets harder)
          train scale     more individuals to LEARN from       (model gets better)
        Swept independently here. Note which models can even respond to the second: `raw_voting`
        and `strict_hand_pos` are TRAINING-FREE, so more training data cannot help them by
        construction -- only `logreg` (and the e2e transformers) can convert train scale into
        accuracy. If logreg's train curve is flat too, "collect more animals" is not the fix.

    Q2  WHAT SCORE MEANS "DON'T EVEN TRY"?  -- for a battery of per-query scores computable at
        deployment time WITHOUT labels, how well does each predict that the match will be wrong,
        and where is the cut below which accuracy falls under a target? Reported as: AUROC for
        predicting failure, accuracy by decile, and the abstain point (reject the bottom X%).
        The oracle `true_quality` is included as the ceiling -- no deployable score can beat it,
        so the gap tells you how much triage headroom is left.

    MODE=scale   pixi run python pipeline/spot_transformer/sweeps/sweep_synthetic.py
    MODE=quality N_POOL=3000 pixi run python .../sweep_synthetic.py
    MODE=both SHAPE_RANK=6 GALLERY_SIZES=25,50,100,200,400 pixi run python .../sweep_synthetic.py

Cost is dominated by (queries x gallery individuals) pair scorings, so MAX_QUERIES caps the
query side: the gallery is the variable of interest, queries only buy precision on the estimate.
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

import census as cen                                          # noqa: E402
import strict_voter as sv                                     # noqa: E402
import compare_strict as cs                                   # noqa: E402  (_census_row)
import data as d                                              # noqa: E402
from synthetic import SynthConfig, generate, compare_to_real  # noqa: E402
from aggregator import build_pairs as build_pairs_plain, train_aggregator, _prob   # noqa: E402
from aggregator_set import per_query_top1                     # noqa: E402

MODE = os.environ.get("MODE", "scale")
PRESET = os.environ.get("PRESET", "realistic")               # easy | realistic | hard
SEED = int(os.environ.get("SEED", "0"))
N_POOL = int(os.environ.get("N_POOL", "2500"))               # individuals generated in total
PHOTOS = int(os.environ.get("PHOTOS", "3"))
SHAPE_RANK = int(os.environ.get("SHAPE_RANK", "0"))          # 0 = take the preset's value
MEAN_SPOTS = float(os.environ.get("MEAN_SPOTS", "25"))
MAX_QUERIES = int(os.environ.get("MAX_QUERIES", "300"))
NOVEL_FRAC = 0.35
NEG_PER_QUERY = int(os.environ.get("NEG_PER_QUERY", "25"))
SIGMA_POS = float(os.environ.get("SIGMA_POS", "0.12"))
TARGET_ACC = float(os.environ.get("TARGET_ACC", "0.5"))      # "cannot handle" bar for Q2
REPEATS = int(os.environ.get("REPEATS", "3"))                # draws of individuals per sweep point

GALLERY_SIZES = [int(x) for x in os.environ.get("GALLERY_SIZES", "25,50,100,200,400,800").split(",")]
TRAIN_SIZES = [int(x) for x in os.environ.get("TRAIN_SIZES", "25,50,100,200,400,800").split(",")]
FIXED_TRAIN = int(os.environ.get("FIXED_TRAIN", "400"))      # train size while sweeping gallery
FIXED_GALLERY = int(os.environ.get("FIXED_GALLERY", "200"))  # gallery size while sweeping train

MODELS = ["raw_voting", "strict_hand_pos", "logreg"]


# ----------------------------------------------------------------------------- splits
def split_labels(labels, n_gallery, n_train, rng):
    """Disjoint EVAL and TRAIN individual sets. Disjoint is the whole point: a learned model must
    never have seen the animals it is scored on, or train scale measures memorisation."""
    perm = [labels[i] for i in rng.permutation(len(labels))]
    if n_gallery + n_train > len(perm):
        raise ValueError(f"pool has {len(perm)} individuals, need {n_gallery}+{n_train}; "
                         f"raise N_POOL")
    return perm[:n_gallery], perm[n_gallery:n_gallery + n_train]


def eval_once(pop, eval_labels, train_labels, models, *, seed=0, verbose=True):
    """Run the census protocol on one (gallery, train) configuration. Returns {model: metrics}."""
    by_label = pop.index_by_label()
    eval_idx = [i for l in eval_labels for i in by_label[l]]
    train_imgs = [i for l in train_labels for i in by_label[l]]

    gal, qry = cen.make_openset_split(pop.sets, eval_idx, NOVEL_FRAC, seed)
    rng = np.random.default_rng(seed)
    if len(qry) > MAX_QUERIES:                     # cap the query side; gallery stays intact
        qry = [qry[i] for i in rng.permutation(len(qry))[:MAX_QUERIES]]
    true_by_q = {q: pop.sets[q].label for q in qry}
    out = {}

    if {"raw_voting", "logreg"} & set(models):
        Xo, oq, oc = cen.build_openset_features(pop.sets, gal, qry)
        if "raw_voting" in models:
            out["raw_voting"] = cs._census_row(Xo[:, 0], oq, oc, true_by_q)
        if "logreg" in models:
            Xtr, ytr, _, _ = build_pairs_plain(pop.sets, train_imgs,
                                               neg_per_query=NEG_PER_QUERY, seed=seed)
            m, sc = train_aggregator(Xtr, ytr, hidden=0, seed=seed)
            out["logreg"] = cs._census_row(_prob(m, sc, Xo), oq, oc, true_by_q)

    if "strict_hand_pos" in models:
        s, sq, sc2 = sv.build_openset_hand(pop.sets, gal, qry, pop.weight_lookup,
                                           pos_lookup=pop.pos_lookup, sigma_pos=SIGMA_POS)
        out["strict_hand_pos"] = cs._census_row(s, sq, sc2, true_by_q)
        out["_per_query"] = (s, sq, sc2, true_by_q, qry)      # kept for the Q2 analysis
    return out


# ----------------------------------------------------------------------------- Q1: scale
_METRICS = ("census_f", "ident_r1", "r5", "auroc_top1", "cov_at_p")


def repeat_eval(pop, n_gal, n_tr, models, rng):
    """``REPEATS`` independent draws of the (gallery, train) individuals -> mean and std.

    Not optional. A single draw of 100 gallery individuals out of the pool gave R@1 0.278 in one
    sweep and 0.214 in another at the SAME configuration — the between-draw spread swamps the
    effect being measured, so one point per size is unreadable.
    """
    rows = []
    for rep in range(REPEATS):
        ev, tr = split_labels(pop.labels, n_gal, n_tr, rng)
        rows.append(eval_once(pop, ev, tr, models, seed=SEED + rep))
    return {m: {k: (float(np.mean([r[m][k] for r in rows])),
                    float(np.std([r[m][k] for r in rows]))) for k in _METRICS}
            for m in models}


def _cell(mu_sd):
    mu, sd = mu_sd
    return f"{mu:.3f} ± {sd:.3f}"


def sweep_scale(pop, md):
    rng = np.random.default_rng(SEED)
    md += ["## Q1a — gallery scale (task difficulty)", "",
           f"Train set fixed at {FIXED_TRAIN} individuals; only the number of candidate "
           f"individuals grows. `raw_voting` and `strict_hand_pos` are training-free, so this is "
           f"the pure difficulty curve for them. {REPEATS} draws per point, mean ± std.", "",
           "| gallery individuals | model | census F0.5 | R@1 | R@5 | AUROC | cov@P90 |",
           "|---|---|---|---|---|---|---|"]
    logger.info("=== Q1a  gallery scale " + "=" * 50)
    for n_gal in GALLERY_SIZES:
        if n_gal + FIXED_TRAIN > len(pop.labels):
            logger.warning(f" skip gallery={n_gal}: pool too small (raise N_POOL)")
            continue
        t0 = time.time()
        agg = repeat_eval(pop, n_gal, FIXED_TRAIN, MODELS, rng)
        for m in MODELS:
            a = agg[m]
            md.append(f"| {n_gal} | {m} | {_cell(a['census_f'])} | {_cell(a['ident_r1'])} | "
                      f"{_cell(a['r5'])} | {_cell(a['auroc_top1'])} | {_cell(a['cov_at_p'])} |")
        line = "  ".join(f"{m.replace('strict_hand_pos','strict')}:R1="
                         f"{agg[m]['ident_r1'][0]:.3f}±{agg[m]['ident_r1'][1]:.3f}"
                         for m in MODELS)
        logger.info(f" gallery {n_gal:>5}  {line}   ({time.time()-t0:.0f}s)")

    md += ["", "## Q1b — train scale (model capability)", "",
           f"Gallery fixed at {FIXED_GALLERY} individuals; only the training pool grows. "
           "Training-free models are omitted — they cannot respond to this axis, which is itself "
           "the point: if `logreg` is flat here, more animals will not fix the matcher. "
           f"{REPEATS} draws per point.", "",
           "| train individuals | census F0.5 | R@1 | R@5 | AUROC | cov@P90 |",
           "|---|---|---|---|---|---|"]
    logger.info("=== Q1b  train scale (logreg) " + "=" * 44)
    for n_tr in TRAIN_SIZES:
        if n_tr + FIXED_GALLERY > len(pop.labels):
            logger.warning(f" skip train={n_tr}: pool too small (raise N_POOL)")
            continue
        t0 = time.time()
        a = repeat_eval(pop, FIXED_GALLERY, n_tr, ["logreg"], rng)["logreg"]
        md.append(f"| {n_tr} | {_cell(a['census_f'])} | {_cell(a['ident_r1'])} | "
                  f"{_cell(a['r5'])} | {_cell(a['auroc_top1'])} | {_cell(a['cov_at_p'])} |")
        logger.info(f" train {n_tr:>5}  logreg:R1={a['ident_r1'][0]:.3f}±{a['ident_r1'][1]:.3f}"
              f"  F={a['census_f'][0]:.3f}   ({time.time()-t0:.0f}s)")
    return md


# ----------------------------------------------------------------------------- Q2: triage
def _abstain_point(score, correct, target):
    """Reject the lowest-scoring queries while their accuracy is below ``target``.

    Scans thresholds low->high and returns the largest rejected prefix whose accuracy stays under
    the bar: "below this score the matcher is worse than ``target``, so don't ask it". Returns
    ``(threshold, frac_rejected, acc_rejected, acc_kept)``.
    """
    o = np.argsort(score)
    s, c = np.asarray(score)[o], np.asarray(correct, float)[o]
    best = (float("nan"), 0.0, float("nan"), float(c.mean()))
    for i in range(1, len(s)):
        if c[:i].mean() < target:
            best = (float(s[i - 1]), i / len(s), float(c[:i].mean()),
                    float(c[i:].mean()) if i < len(s) else float("nan"))
    return best


def sweep_quality(pop, md):
    """Per-query triage: which deployable score tells you the answer will be wrong?"""
    rng = np.random.default_rng(SEED)
    ev, tr = split_labels(pop.labels, FIXED_GALLERY, FIXED_TRAIN, rng)
    logger.info("=== Q2  query triage " + "=" * 53)
    t0 = time.time()
    res = eval_once(pop, ev, tr, ["strict_hand_pos"], seed=SEED)
    s, sq, sc2, true_by_q, qry = res["_per_query"]
    top = per_query_top1(np.asarray(s, float), sq, sc2, true_by_q)
    logger.info(f" scored {len(top['q'])} queries against {FIXED_GALLERY} gallery individuals "
          f"({time.time()-t0:.0f}s)")

    known = top["is_known"].astype(bool)
    correct = top["top1_correct"][known].astype(float)
    qids = [int(q) for q in top["q"][known]]

    truth = pop.truth.set_index("sid")
    scores = {}
    for name in ("n_spots", "evidence_mass", "mean_w", "max_w", "true_quality",
                 "top1_score", "margin"):
        scores[name] = []
    for j, q in enumerate(qids):
        st = pop.sets[q]
        w = np.array([pop.weight_lookup[(st.sid, int(i))] for i in st.spot_ids], float)
        scores["n_spots"].append(len(w))
        scores["evidence_mass"].append(float(w.sum()))
        scores["mean_w"].append(float(w.mean()))
        scores["max_w"].append(float(w.max()))
        scores["true_quality"].append(float(truth.loc[st.sid, "quality"]))
    idx_known = np.flatnonzero(known)
    scores["top1_score"] = list(top["top1"][idx_known])
    scores["margin"] = list(top["margin"][idx_known])

    degenerate = correct.mean() < TARGET_ACC
    md += ["", "## Q2 — which score says \"don't even try\"?", "",
           f"{len(correct)} known queries vs a {FIXED_GALLERY}-individual gallery · overall "
           f"R@1 **{correct.mean():.3f}** · target accuracy {TARGET_ACC}", ""]
    if degenerate:
        md += [f"**Triage is undefined at this operating point.** Overall accuracy "
               f"{correct.mean():.3f} is already below the {TARGET_ACC} target, so every prefix "
               "fails the bar and the abstain point degenerates to rejecting ~everything. Lower "
               "`TARGET_ACC`, shrink `FIXED_GALLERY`, or use an easier `PRESET`: there is no "
               "\"below this score it breaks\" when it is broken everywhere. The AUROC column is "
               "still meaningful — it is threshold-free.", ""]
        logger.warning(f" !! overall accuracy {correct.mean():.3f} < TARGET_ACC {TARGET_ACC}: abstain "
              "points are degenerate, only AUROC is meaningful")
    md += [
           "- **AUROC** = does the score rank correct matches above wrong ones (0.5 = useless).",
           "- **abstain** = reject the lowest-scoring queries while their accuracy is under the "
           "target; `rejected` is how much of the workload that costs, `kept acc` what the rest "
           "is worth.",
           "- `true_quality` is the ORACLE (the generator's own per-photo quality) — the ceiling "
           "no deployable score can beat. `top1_score`/`margin` are only available AFTER matching;",
           "  the rest are known from the photo alone, so only they can pre-filter in the field.",
           "",
           "| score | available | AUROC | thr | rejected | rej acc | kept acc |",
           "|---|---|---|---|---|---|---|"]
    avail = {"n_spots": "pre-match", "evidence_mass": "pre-match", "mean_w": "pre-match",
             "max_w": "pre-match", "true_quality": "oracle", "top1_score": "post-match",
             "margin": "post-match"}
    logger.info(f" {'score':<16}{'avail':<11}{'AUROC':>7}{'reject':>9}{'rej acc':>9}{'kept acc':>10}")
    logger.info(" " + "-" * 62)
    for name, vals in scores.items():
        v = np.asarray(vals, float)
        a = cen.auroc(v, correct.astype(bool))
        thr, frac, arej, akeep = _abstain_point(v, correct, TARGET_ACC)
        md.append(f"| {name} | {avail[name]} | {a:.3f} | {thr:.3f} | {frac:.1%} | "
                  f"{arej:.3f} | {akeep:.3f} |")
        logger.info(f" {name:<16}{avail[name]:<11}{a:>7.3f}{frac:>9.1%}{arej:>9.3f}{akeep:>10.3f}")

    # accuracy by decile of each pre-match score — the shape of the failure, not just a cut
    md += ["", "### Accuracy by decile (pre-match scores)", "",
           "| score | d1 (worst) | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 (best) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("n_spots", "evidence_mass", "mean_w", "true_quality"):
        v = np.asarray(scores[name], float)
        edges = np.quantile(v, np.linspace(0, 1, 11))
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (v >= lo) & (v <= hi)
            cells.append(f"{correct[m].mean():.2f}" if m.sum() >= 3 else "—")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    return md


# ----------------------------------------------------------------------------- calibration
# R@1 measured on the REAL dataset, 5 folds (artifacts/.../strict/RESULTS_strict*.md).
#
# CALIBRATE ON ALL THREE, NOT ONE. The first version of this file matched `strict_hand_pos` alone
# and looked perfect (0.188 vs 0.192) while `raw_voting` sat at EXACTLY chance -- the synthetic
# descriptor noise had made spot-level matching impossible, so only a learned model reading
# summary statistics still worked. That is a different regime from the real data, where all three
# models are within ~2x of each other and all well above chance. One matching scalar proves
# nothing; the RELATIVE ORDERING and the distance from chance are what say the task is the same.
REAL_R1 = {
    "no filter": {"raw_voting": 0.118, "strict_hand_pos": 0.192, "logreg": 0.183},
    "q0.4":      {"raw_voting": 0.319, "strict_hand_pos": 0.365, "logreg": 0.489},
}


def calibrate(cfg):
    """Is the synthetic task as hard as the real one? Run this before trusting any sweep."""
    from dataclasses import replace

    n_gal = int(os.environ.get("CAL_GALLERY", "25"))
    n_tr = int(os.environ.get("CAL_TRAIN", "100"))
    chance = 1.0 / n_gal
    logger.info(f" Real R@1 (5-fold) vs synthetic at a {n_gal}-individual gallery "
          f"(chance = {chance:.3f}):")
    logger.info(f" {'regime':<16}{'raw_voting':>12}{'strict_hand_pos':>17}{'logreg':>9}")
    for k, v in REAL_R1.items():
        logger.info(f" REAL {k:<11}{v['raw_voting']:>12.3f}{v['strict_hand_pos']:>17.3f}"
              f"{v['logreg']:>9.3f}")
    for name in ("easy", "realistic", "hard"):
        c = replace(SynthConfig.preset(name), n_individuals=n_gal + n_tr,
                    photos_per_individual=cfg.photos_per_individual,
                    mean_spots=cfg.mean_spots, seed=cfg.seed)
        pop = generate(c, verbose=False)
        res = eval_once(pop, pop.labels[:n_gal], pop.labels[n_gal:], MODELS, seed=cfg.seed)
        r1 = {m: res[m]["ident_r1"] for m in MODELS}
        # the check that actually matters: is every model clear of chance, and do they stay in
        # the same ballpark as each other (as they do on real data)?
        flag = ""
        if min(r1.values()) < 2 * chance:
            flag = "  <- a model is AT CHANCE: wrong regime"
        elif max(r1.values()) > 0.85:
            flag = "  <- too easy: optimistic"
        logger.info(f" SYN  {name:<11}{r1['raw_voting']:>12.3f}{r1['strict_hand_pos']:>17.3f}"
              f"{r1['logreg']:>9.3f}{flag}")
    logger.info(" Pick the preset whose THREE numbers sit closest to a real row — matching one model"
          "\n while another collapses to chance means the synthetic task is not yours.")


# ----------------------------------------------------------------------------- driver
def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    over = dict(n_individuals=N_POOL, photos_per_individual=PHOTOS, mean_spots=MEAN_SPOTS,
                seed=SEED)
    if SHAPE_RANK:
        over["shape_rank"] = SHAPE_RANK
    cfg = SynthConfig.preset(PRESET, **over)
    logger.info("=" * 74)
    logger.info(f" SYNTHETIC SWEEP  mode={MODE}  preset={PRESET}  pool={N_POOL} individuals  "
          f"shape_rank={cfg.shape_rank}")
    logger.info("=" * 74)
    if MODE == "calibrate":
        return calibrate(cfg)
    t0 = time.time()
    pop = generate(cfg)
    logger.info(f" generated in {time.time()-t0:.0f}s")

    md = [f"# Synthetic scaling & triage — `{cfg.tag()}`", "",
          "Generated by `sweeps/sweep_synthetic.py` on populations from `core/synthetic.py`.",
          "Spots are synthesised directly in the 62-dim `[shape|position]` embedding space, so the",
          "matcher runs unmodified; pixels/segmentation are NOT simulated (see the module docstring",
          "for what that does and does not license you to conclude).", "",
          f"- pool {N_POOL} individuals x {PHOTOS} photos · mean {MEAN_SPOTS:g} spots · "
          f"shape_rank {SHAPE_RANK} · novel_frac {NOVEL_FRAC} · max_queries {MAX_QUERIES}", ""]

    try:
        cal = compare_to_real(pop)
        md += ["### Calibration against the real dataset", "",
               "Statistics the generator was *not* fitted to. Large divergence = treat absolute",
               "numbers below as method-relative, not as claims about real salamanders.", "",
               "| statistic | real | synthetic |", "|---|---|---|"]
        for _, r in cal.iterrows():
            md.append(f"| {r['statistic']} | {r['real']:.3f} | {r['synthetic']:.3f} |")
        md.append("")
        logger.info(" calibration vs real data:")
        logger.info(cal.to_string(index=False))
    except Exception as e:                       # no DB present / different schema — not fatal
        logger.warning(f" (calibration skipped: {e})")
        md += [f"_Calibration against real data unavailable: {e}_", ""]

    if MODE in ("scale", "both"):
        md = sweep_scale(pop, md)
    if MODE in ("quality", "both"):
        md = sweep_quality(pop, md)

    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "sweeps" / "synthetic"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"RESULTS_synthetic_{MODE}_{cfg.tag()}.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info(f"wrote {path}   (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
