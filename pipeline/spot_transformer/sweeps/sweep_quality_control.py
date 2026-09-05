"""Does the image-quality filter make the MODEL better, or the PROBLEM smaller?

The all-9 quality sweep reported identR@1 rising 0.153 -> 0.464 as MIN_QUALITY went 0 -> 0.4, and
read that as the filter helping. It is mostly an artefact. ``data.get_cv_folds`` only admits
individuals with >=2 real images, so filtering does not just drop photos -- it deletes evaluable
INDIVIDUALS, and the gallery a query competes against collapses with them:

    MIN_QUALITY   none   0.2   0.3   0.4   0.5
    gallery ids    105    41    36    27    14
    chance R@1   0.010 0.024 0.028 0.036 0.071

Naming 1-of-14 is not the task that naming 1-of-105 is. This driver removes that confound by
holding the gallery at a FIXED number of individuals (default 14, the q0.5 ceiling) at every
quality level, so the only thing that varies is which photos are in play. It runs each level
BOTH ways -- natural gallery and fixed gallery -- so the size of the artefact is visible in one
table rather than inferred.

Three further fixes to how the numbers are reported:

  * **Repeated splits, pooled queries.** A 14-of-105 gallery draw is itself a coin toss, so the
    split is redrawn ``REPEATS`` times and each fold's model is scored against ``DRAWS`` galleries.
    Every per-query decision goes into one pool instead of being averaged into a per-fold mean.
  * **Cluster bootstrap by individual.** The reported interval is a 95% percentile CI over
    resampled INDIVIDUALS, not a std over 2 folds. An animal's photos are not independent
    evidence, and resampling the animal (all its queries together) is what keeps them from being
    counted as if they were.
  * **Base score only.** ``sweep_all9`` picks the b' feature -- and then the winner among
    base/b'/c' -- by argmax on the eval split itself, which biases b', balAcc and review@90
    upward. Novelty here is scored on the raw top-1 score alone, which is unbiased and is the
    only novelty column of the original that was.

Why only logreg: the set/e2e families need full epochs to be worth scoring (QUICK cuts them 15-40x)
and the full 5-fold run at q0.4 already put the top six models inside one std of each other. This
is a question about the data, so it is answered with the cheapest scorer held fixed.

    pixi run python pipeline/spot_transformer/sweep_quality_control.py
    QUICK=1 pixi run python pipeline/spot_transformer/sweep_quality_control.py   # smoke path
    TRAIN_POOL=full pixi run python pipeline/spot_transformer/sweep_quality_control.py

``TRAIN_POOL=full`` trains on the UNFILTERED images of the train-side individuals while still
evaluating on filtered ones -- that separates "filtering gives a better model" from "filtering
gives an easier eval", which the default (``filtered``, matching the original sweep) blends.

Results -> ``artifacts/spot_transformer/sweeps/quality_control/RESULTS_quality_control.md``.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

import census as cen                                            # noqa: E402
import data as d                                                # noqa: E402
from aggregator import _prob, attach_centroids, build_pairs, train_aggregator  # noqa: E402
from aggregator_set import per_query_top1                       # noqa: E402

QUICK = bool(os.environ.get("QUICK"))
SEED = 0
BETA = 0.5
NOVEL_FRAC = 0.35                                               # as in census.make_openset_split

# k=2, deliberately. A fixed 14-individual gallery needs >=22 individuals in an eval fold
# (14 known + 8 novel at NOVEL_FRAC); q0.5 leaves 42 eligible in total, so k=5 would put ~8 in a
# fold and the fixed gallery could not be built at all. Query volume is recovered by REPEATS
# (fresh individual-level splits), not by more folds.
K_FOLDS = 2
GAL_SIZE = int(os.environ.get("GAL_SIZE", 14))
REPEATS = int(os.environ.get("REPEATS", 2 if QUICK else 5))     # independent 2-fold splits
DRAWS = int(os.environ.get("DRAWS", 3 if QUICK else 5))         # gallery draws per trained fold
# The natural gallery is a REFERENCE row -- it documents the artefact, it is not the experiment.
# It is also by far the most expensive thing here: an unfiltered natural split scores 545 queries
# against 105 candidates, i.e. ~57k RANSAC feature builds per fold, against ~1.3k for a fixed
# 14-gallery. Running it on fewer repeats keeps the reference honest and the run affordable.
NAT_REPEATS = int(os.environ.get("NAT_REPEATS", 1 if QUICK else 2))
NEG_PER_QUERY = int(os.environ.get("NEG_PER_QUERY", 10 if QUICK else 30))
N_BOOT = int(os.environ.get("N_BOOT", 300 if QUICK else 2000))
TRAIN_POOL = os.environ.get("TRAIN_POOL", "filtered")           # 'filtered' | 'full'
# "none" = no quality gate, the baseline every other level is compared against. Kept as a token
# rather than hard-prepended so a smoke run can skip it -- unfiltered is the single most expensive
# level (largest train pool AND largest natural gallery), which makes it a bad smoke test.
LEVELS = [None if x.strip().lower() == "none" else float(x)
          for x in os.environ.get("LEVELS", "none,0.2,0.3,0.4,0.5").split(",") if x.strip()]


# ============================================================ pooled metrics (vectorised)
def pooled_metrics(top1, correct, known):
    """All four headline numbers from one pool of per-query decision rows.

    Vectorised because the cluster bootstrap calls this a few thousand times per cell.
    Sorting by score descending and taking prefixes is the same threshold sweep
    ``census.census_sweep`` does with a Python loop over 200 quantiles, but exact and ~200x
    cheaper: prefix j = "accept the j highest-scoring queries as MATCH".
    """
    top1 = np.asarray(top1, float)
    corr = np.asarray(correct).astype(bool)
    kn = np.asarray(known).astype(bool)
    n_k, n_n = int(kn.sum()), int((~kn).sum())
    nan = float("nan")
    if n_k == 0 or n_n == 0:
        return dict(r1=nan, auroc=nan, census_f=nan, bal_acc=nan)

    r1 = float(corr[kn].mean())
    auroc = cen.auroc(top1, kn)

    o = np.argsort(-top1, kind="mergesort")
    k_s, c_s = kn[o], corr[o]
    tp = np.cumsum(k_s & c_s)                                   # known, matched, right id
    fp_known = np.cumsum(k_s & ~c_s)                            # known, matched, WRONG id
    fp_novel = np.cumsum(~k_s)                                  # novel absorbed -> deflation
    prec = tp / np.maximum(tp + fp_known + fp_novel, 1)
    rec = tp / n_k                                              # tp + fp_known + fn == n_k
    den = BETA * BETA * prec + rec
    f = np.where(den > 0, (1 + BETA * BETA) * prec * rec / np.maximum(den, 1e-12), 0.0)

    # balanced accuracy of the new-vs-known call, at its own best cut (TN finally counts)
    known_rec = np.cumsum(k_s) / n_k
    novel_rec = (n_n - fp_novel) / n_n
    bal = 0.5 * (known_rec + novel_rec)

    return dict(r1=r1, auroc=auroc, census_f=float(f.max()), bal_acc=float(bal.max()))


def cluster_bootstrap(pool, n_boot=2000, seed=0):
    """95% percentile CIs, resampling INDIVIDUALS (with replacement) rather than queries.

    An individual contributes many correlated rows -- several photos, and the same photo re-scored
    under several gallery draws. Resampling the individual moves all of them together, which is
    what stops repeated looks at one animal from being counted as independent evidence. A
    query-level bootstrap would report intervals several times too narrow.
    """
    lbl = np.asarray(pool["label"], dtype=object)
    uniq = np.unique(lbl)
    where = {l: np.flatnonzero(lbl == l) for l in uniq}
    t1, co, kn = pool["top1"], pool["correct"], pool["known"]
    rng = np.random.default_rng(seed)
    keys = ("r1", "auroc", "census_f", "bal_acc")
    acc: dict[str, list] = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        sel = np.concatenate([where[uniq[p]] for p in pick])
        m = pooled_metrics(t1[sel], co[sel], kn[sel])
        for k in keys:
            # NaN rather than drop, so replicate i means the same resample in every metric and
            # across levels -- that alignment is what lets diff_ci difference the arrays directly
            acc[k].append(m[k] if np.isfinite(m[k]) else np.nan)
    reps = {k: np.asarray(v, float) for k, v in acc.items()}
    ci = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
          if np.isfinite(v).sum() > 10 else (float("nan"), float("nan"))
          for k, v in reps.items()}
    return ci, reps


def diff_ci(reps_a, reps_b, key, seed=0):
    """95% CI for (level B - level A) on one metric, from their bootstrap replicates.

    The two levels are INDEPENDENT samples -- different individuals, different photos -- so this
    is a two-sample bootstrap, not a paired one. Differencing independent replicate vectors is
    valid for that, but the replicate ORDER carries no pairing information, so one vector is
    shuffled first to make the arbitrary alignment explicit rather than accidental.

    This is the test the verdict should use. Comparing whether two CIs overlap is far more
    conservative than testing the difference directly: intervals can overlap by a hair while the
    difference still excludes zero comfortably, which is exactly what happened at none vs q0.4.
    """
    a, b = reps_a[key], reps_b[key]
    n = min(len(a), len(b))
    if n < 20:
        return (float("nan"), float("nan"))
    a = np.random.default_rng(seed).permutation(a[:n])
    d = b[:n] - a
    d = d[np.isfinite(d)]
    if len(d) < 20:
        return (float("nan"), float("nan"))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ============================================================ splits
def _by_label(sets, idx):
    out: dict[str, list[int]] = {}
    for i in idx:
        out.setdefault(sets[i].label, []).append(i)
    return out


def novel_count_for(n_known):
    """Novel individuals to pair with ``n_known``, by INVERTING census.make_openset_split.

    That function takes L individuals and keeps ``L - round(NOVEL_FRAC*L)`` as known, so the
    achievable known/novel pairs are set by its rounding, not by the ratio. Computing the novel
    side as ``n_known * f/(1-f)`` instead looks equivalent and is not: at n_known=14 it asks for
    8 novel (total 22) while make_openset_split yields exactly 14 known from a 21-individual fold
    (novel = round(0.35*21) = 7). One individual short is enough to make every q0.5 fold
    unbuildable -- which is exactly what it did.

    So: find the smallest total L whose known side reaches n_known, and take the rest as novel.
    """
    L = n_known
    while L - int(round(NOVEL_FRAC * L)) < n_known:
        L += 1
    return L - n_known


def fixed_gallery_split(sets, eval_idx, n_known, rng):
    """Open-set split with the gallery pinned to exactly ``n_known`` individuals.

    Mirrors ``census.make_openset_split`` except for the size: the novel count comes from
    ``novel_count_for`` so the known/novel PRIOR tracks NOVEL_FRAC under that function's own
    rounding. Letting the prior drift would move AUROC and balanced accuracy for a reason
    unrelated to photo quality, which is the whole thing this driver exists to prevent. Returns
    ``None`` when the fold has too few individuals to build a comparable split -- the cell is
    then reported as missing rather than as a smaller, non-comparable one.
    """
    real = [i for i in eval_idx if not sets[i].is_synth]
    by_label = _by_label(sets, real)
    labels = sorted(by_label)
    n_novel = novel_count_for(n_known)
    if len(labels) < n_known + n_novel:
        return None
    pick = [labels[i] for i in rng.permutation(len(labels))[:n_known + n_novel]]
    known, novel = set(pick[:n_known]), set(pick[n_known:])
    gallery = [i for l in known for i in by_label[l]]
    queries = [i for l in (known | novel) for i in by_label[l]]
    return gallery, queries


def folds_over(sets, keep, k, seed):
    """k individual-level folds, built over the KEPT images only.

    ``keep`` is a boolean mask over ``sets`` rather than a filtered list, so indices stay valid
    against the full set and ``TRAIN_POOL=full`` can reach the dropped images of a train-side
    individual. Eligibility (>=2 real images) is judged on kept images, matching how the
    filtered sweep behaves.
    """
    by_label = _by_label(sets, range(len(sets)))
    n_real_kept = {l: sum(1 for i in ix if keep[i] and not sets[i].is_synth)
                   for l, ix in by_label.items()}
    eligible = sorted(l for l in by_label if n_real_kept[l] >= 2)
    rng = np.random.default_rng(seed)
    eligible = [eligible[i] for i in rng.permutation(len(eligible))]
    all_labels = set(by_label)
    out = []
    for evl in np.array_split(eligible, k):
        evl = set(evl)
        tr_lbl = all_labels - evl
        # synthetic views are training-only and eval never sees them, so they ride along
        train = [i for l in tr_lbl for i in by_label[l]
                 if keep[i] or (TRAIN_POOL == "full" and not sets[i].is_synth)]
        ev = [i for l in evl for i in by_label[l] if keep[i] and not sets[i].is_synth]
        out.append((sorted(train), sorted(ev)))
    return out, len(eligible)


# ============================================================ one quality level, both galleries
def run_level(sets, keep, log):
    """Pool per-query decisions over REPEATS x K_FOLDS x DRAWS, for BOTH gallery modes.

    Both modes are scored from the SAME trained model per fold. That is the cheap way round
    (training and the natural-gallery feature build dominate the cost) and also the correct one:
    the natural/fixed gap is then attributable to the gallery alone, with model fitting held
    literally identical rather than merely re-seeded.
    """
    rows = {m: {k: [] for k in ("top1", "correct", "known", "label")}
            for m in ("natural", "fixed")}
    gal_sizes: dict[str, list] = {"natural": [], "fixed": []}
    n_eligible = 0

    def collect(mode, gal, qry, model, scaler):
        if not gal or not qry:
            return
        Xo, oq, oc = cen.build_openset_features(sets, gal, qry)
        if not len(Xo):
            return
        top = per_query_top1(_prob(model, scaler, Xo), oq, oc,
                             {q: sets[q].label for q in qry})
        gal_sizes[mode].append(len({sets[i].label for i in gal}))
        r = rows[mode]
        r["top1"].append(np.asarray(top["top1"], float))
        r["correct"].append(np.asarray(top["top1_correct"], float))
        r["known"].append(np.asarray(top["is_known"], float))
        r["label"].append(np.array([sets[q].label for q in top["q"]], dtype=object))

    for rep in range(REPEATS):
        folds, n_eligible = folds_over(sets, keep, K_FOLDS, SEED + 1000 * rep)
        for fi, (tr, ev) in enumerate(folds):
            t0 = time.time()
            train_imgs = [i for i in tr if not sets[i].is_synth]
            if len(train_imgs) < 10 or len(ev) < 4:
                continue
            X, y, _, _ = build_pairs(sets, train_imgs, neg_per_query=NEG_PER_QUERY,
                                     seed=SEED + rep)
            if len(np.unique(y)) < 2:
                continue
            model, scaler = train_aggregator(X, y, seed=SEED, hidden=0)

            # natural: the split census.make_openset_split would have produced, one per fold
            if rep < NAT_REPEATS:
                collect("natural", *cen.make_openset_split(sets, ev, NOVEL_FRAC,
                                                           SEED + 1000 * rep + fi), model, scaler)
            # fixed: the model is already trained, so extra gallery draws are nearly free
            for dr in range(DRAWS):
                rng = np.random.default_rng(SEED + 7919 * rep + 131 * fi + dr)
                split = fixed_gallery_split(sets, ev, GAL_SIZE, rng)
                if split is None:
                    log(f"      rep{rep} fold{fi}: only "
                        f"{len({sets[i].label for i in ev})} individuals in eval — too few for "
                        f"a {GAL_SIZE}-gallery, skipped")
                    break
                collect("fixed", *split, model, scaler)
            log(f"      rep{rep} fold{fi}: {len(train_imgs)} train imgs, "
                f"{len({sets[i].label for i in ev})} eval individuals ({time.time() - t0:.0f}s)")

    out = {}
    for mode in ("natural", "fixed"):
        if not rows[mode]["top1"]:
            out[mode] = None
            continue
        pool = {k: np.concatenate(v) for k, v in rows[mode].items()}
        met = pooled_metrics(pool["top1"], pool["correct"], pool["known"])
        met["ci"], met["reps"] = cluster_bootstrap(pool, N_BOOT, SEED)
        met["n_query"] = int(len(pool["top1"]))
        met["n_known"] = int(pool["known"].sum())
        met["n_ind"] = int(len(np.unique(pool["label"])))
        met["gal"] = float(np.mean(gal_sizes[mode])) if gal_sizes[mode] else float("nan")
        met["eligible"] = n_eligible
        met["chance"] = 1.0 / max(met["gal"], 1)
        out[mode] = met
    return out


# ============================================================ driver
def _fmt(v, ci):
    if not np.isfinite(v):
        return f"{'—':>22}"
    return f"{v:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"


def main():
    for stream in (sys.stdout, sys.stderr):                     # cp1255 console; see sweep_all9
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    t_start = time.time()
    base = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))

    logger.info("=" * 96)
    logger.info(f" QUALITY-FILTER CONTROL  ·  gallery pinned to {GAL_SIZE} individuals"
          f"{'   [QUICK]' if QUICK else ''}")
    logger.info(f" {REPEATS} repeats x {K_FOLDS} folds x {DRAWS} gallery draws · logreg only · "
          f"train pool: {TRAIN_POOL}  (natural reference: {NAT_REPEATS} repeats)")
    logger.info(f" novelty = raw top-1 score (no eval-side feature selection) · "
          f"CI = {N_BOOT} cluster bootstraps over individuals")
    logger.info("=" * 96)

    results: dict = {}
    for lvl in LEVELS:
        tag = "none" if lvl is None else f"{lvl}"
        if lvl is None:
            os.environ.pop("MIN_QUALITY", None)
        else:
            os.environ["MIN_QUALITY"] = str(lvl)
        # reuse the shipped gate rather than reimplementing it, but express the result as a MASK
        # over `base` so indices stay valid and TRAIN_POOL=full can still reach dropped images
        kept_sids = {s.sid for s in d.apply_quality_filter(base)}
        keep = np.array([s.sid in kept_sids for s in base])

        logger.info("=== MIN_QUALITY " + tag + " " + "=" * 40)
        t0 = time.time()
        cell = run_level(base, keep, log=lambda s: logger.info(s))
        for mode in ("natural", "fixed"):
            met = cell[mode]
            results[(tag, mode)] = met
            if met is None:
                logger.warning(f"    {mode:<8} — no usable split")
                continue
            logger.info(f"    {mode:<8} gal {met['gal']:5.1f}  n_q {met['n_query']:5d}  "
                  f"R@1 {_fmt(met['r1'], met['ci']['r1'])}  "
                  f"AUROC {_fmt(met['auroc'], met['ci']['auroc'])}")
        logger.info(f"    ({time.time() - t0:.0f}s)")
    os.environ.pop("MIN_QUALITY", None)

    # ---- tables ----
    def table(mode, title):
        lines = [f"\n{title}",
                 f" {'MIN_Q':<7}{'ind':>5}{'gal':>6}{'nq':>6}{'chance':>8}"
                 f"{'identR@1 [95% CI]':>24}{'AUROC [95% CI]':>24}"
                 f"{'censusF0.5 [95% CI]':>24}{'balAcc [95% CI]':>24}",
                 "-" * 128]
        for lvl in LEVELS:
            tag = "none" if lvl is None else f"{lvl}"
            m = results.get((tag, mode))
            if m is None:
                lines.append(f" {tag:<7}{'—':>5}")
                continue
            lines.append(
                f" {tag:<7}{m['n_ind']:>5}{m['gal']:>6.1f}{m['n_query']:>6d}"
                f"{m['chance']:>8.3f}"
                f"{_fmt(m['r1'], m['ci']['r1']):>24}"
                f"{_fmt(m['auroc'], m['ci']['auroc']):>24}"
                f"{_fmt(m['census_f'], m['ci']['census_f']):>24}"
                f"{_fmt(m['bal_acc'], m['ci']['bal_acc']):>24}")
        return lines

    out = []
    out += table("natural", "NATURAL gallery — reproduces the original sweep's confound")
    out += table("fixed", f"FIXED {GAL_SIZE}-individual gallery — the controlled comparison")

    # ---- level-vs-unfiltered differences, on all four metrics ----
    # Tested on the bootstrap CI of the DIFFERENCE, not on whether two CIs overlap. The overlap
    # rule is far too conservative and it already misled this analysis once: at none vs q0.4 the
    # intervals overlapped by 0.009 and the rule said "no separation" while the difference was
    # large and consistent on every metric.
    tags = ["none" if l is None else str(l) for l in LEVELS]
    fx = dict((t, results[(t, "fixed")]) for t in tags if results.get((t, "fixed")))
    metrics = [("r1", "identR@1"), ("auroc", "AUROC"),
               ("census_f", "censusF0.5"), ("bal_acc", "balAcc")]
    if "none" not in fx or len(fx) < 2:
        verdict = ("inconclusive — needs the unfiltered baseline plus at least one filtered "
                   "level in the same run (LEVELS must include `none`)")
        out += ["", "(no baseline in this run — difference table skipped)"]
    else:
        base = fx["none"]
        out += ["", f"DIFFERENCE vs unfiltered, fixed {GAL_SIZE}-gallery "
                    f"(95% bootstrap CI of the difference; * = excludes zero)",
                f" {'MIN_Q':<7}" + "".join(f"{n:>26}" for _, n in metrics),
                "-" * 116]
        wins: dict[str, list] = {}
        for t in tags:
            if t == "none" or t not in fx:
                continue
            cells = []
            for key, _ in metrics:
                lo, hi = diff_ci(base["reps"], fx[t]["reps"], key, seed=SEED)
                star = "*" if np.isfinite(lo) and (lo > 0 or hi < 0) else " "
                if np.isfinite(lo) and lo > 0:
                    wins.setdefault(t, []).append(key)
                delta = fx[t][key] - base[key]
                cells.append(f"{delta:+.3f} [{lo:+.3f},{hi:+.3f}]{star}")
            out.append(f" {t:<7}" + "".join(f"{c:>26}" for c in cells))
        if wins:
            best = sorted(wins, key=lambda t: -len(wins[t]))
            verdict = ("at a fixed gallery, filtering beats unfiltered on "
                       + "; ".join(f"{t} ({len(wins[t])}/4 metrics: "
                                   f"{', '.join(wins[t])})" for t in best))
        else:
            verdict = ("at a fixed gallery NO quality level separates from unfiltered on any "
                       "metric — the original R@1 climb was gallery size")
    logger.info("\n".join(out))
    logger.info(f" VERDICT: {verdict}")
    logger.info(f" total {time.time() - t_start:.0f}s")

    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "sweeps" / "quality_control"
    outdir.mkdir(parents=True, exist_ok=True)
    # Every knob that changes what the numbers MEAN goes in the filename. A single fixed name let
    # a two-level TRAIN_POOL=full run silently destroy a 56-minute five-level sweep; the console
    # output was the only surviving copy.
    lv = "-".join("none" if l is None else str(l) for l in LEVELS)
    fname = (f"RESULTS_gal{GAL_SIZE}_{TRAIN_POOL}_r{REPEATS}f{K_FOLDS}d{DRAWS}"
             f"_lv{lv}{'_quick' if QUICK else ''}.md")
    md = [f"# Quality filter: better model, or smaller problem?", "",
          ("> **QUICK run — smoke path, not a result.**" if QUICK else ""),
          f"- {REPEATS} repeats × {K_FOLDS} folds × {DRAWS} gallery draws · logreg only "
          f"· neg/query {NEG_PER_QUERY} · train pool `{TRAIN_POOL}`",
          f"- the `natural` reference row uses {NAT_REPEATS} repeats (it documents the artefact; "
          f"the `fixed` row is the experiment and gets all {REPEATS})",
          f"- gallery pinned to **{GAL_SIZE} individuals**; novel count scaled to hold the "
          f"known/novel prior at {NOVEL_FRAC}",
          f"- CI = 95% percentile over {N_BOOT} bootstraps resampling **individuals**, not queries",
          f"- novelty scored on the **raw top-1 score only** — no eval-side b′/c′ selection",
          f"- level-vs-unfiltered is tested on the **CI of the difference**, not on CI overlap "
          f"(overlap is far more conservative and already produced one wrong call here)",
          "", "```"] + out + ["```", "",
          f"**Verdict.** {verdict}", "",
          "`natural` reproduces the original protocol: the gallery shrinks with the filter, so",
          "R@1 rises partly because there are fewer wrong answers available. `fixed` holds the",
          "gallery constant, so a gap there is attributable to photo quality.",
          "", "Run `TRAIN_POOL=full` to separate 'filtering gives a better model' from",
          "'filtering gives an easier eval' — it trains on unfiltered images of the train-side",
          "individuals while still evaluating on filtered ones."]
    (outdir / fname).write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info(f"wrote {outdir / fname}")


if __name__ == "__main__":
    main()
