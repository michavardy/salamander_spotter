"""Does distinctiveness-weighted STRICT matching beat equal-weight voting? (census F0.5)

Compares four matchers under one open-set census protocol, split by individual, on the current
dataset:

  raw_voting     soft-chamfer sum, no training                 (free baseline)
  logreg         equal-weight 17 summary features + logreg     (the current champion)
  strict_hand    coverage x support, distinctiveness-weighted, NO training (the thesis alone)
  strict_logreg  distinctiveness-weighted strict features + logreg          (the thesis, learned)

The distinctiveness weight is the supervised head from ``distinctiveness.py``, retrained **per fold
on the training individuals only** so a spot's animal is never seen when its weight is used at eval.

Headline = **census F0.5** (a false MATCH deflates the count, so precision is weighted 2x). Also
reported: ident R@1 (of true re-sights, top-1 correct) and open-set AUROC for the known-vs-novel
call — both from the top-1 score and from ``unexplained_q`` (the distinctive mass the best match
left unexplained), the novelty signal equal-weight voting cannot form.

    QUICK=1 pixi run python pipeline/spot_transformer/sweeps/compare_strict.py   # 2 folds
    ONLY=raw_voting,strict_logreg pixi run python .../compare_strict.py
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
import likelihood as llr                                    # noqa: E402
import review_labels as rl                                  # noqa: E402
import strict_match as sm                                   # noqa: E402
import strict_voter as sv                                   # noqa: E402
from aggregator import (attach_centroids, build_pairs as build_pairs_plain,   # noqa: E402
                        train_aggregator, _prob)
from aggregator_set import per_query_top1                   # noqa: E402

BETA = 0.5
QUICK = bool(os.environ.get("QUICK"))
ONLY = {s.strip() for s in os.environ.get("ONLY", "").split(",") if s.strip()}
K_FOLDS = 2 if QUICK else 5
SEED = 0
NEG_PER_QUERY = 10 if QUICK else 40
NOVEL_FRAC = 0.35
SIGMA_POS = float(os.environ.get("SIGMA_POS", "0.12"))       # body-frame position-gate width

# Where the per-spot weight comes from — the ablation that says what the interesting-spot clicks
# are worth INSIDE this matcher. `learned` fits the head per fold on training individuals only
# (the default, and what every previous run used); `hand` uses strict_match.DEFAULT_WEIGHTS with no
# fitting; `uniform` sets every spot to 0.5, collapsing the strict score to un-weighted
# coverage x support. Ablating to `uniform` is the control for "is distinctiveness load-bearing".
WEIGHT_MODE = os.environ.get("WEIGHT_MODE", "learned")
# Which labels feed the learned head: the live review store, or the legacy union export that was
# 92 images stale. Only meaningful with WEIGHT_MODE=learned.
LABELS = os.environ.get("LABELS", "review")
# Which photos may be SCORED. `quality` = MIN_QUALITY/MAX_SPOTS_OUTSIDE (the existing gate),
# `review` = photos a human accepted, `both` = the intersection. See data.review_keep_mask for why
# UNREVIEWED (default keep) decides whether this is a subtraction or a resample.
EVAL_GATE = os.environ.get("EVAL_GATE", "quality")
UNREVIEWED = os.environ.get("UNREVIEWED", "keep")
# Which COLLECTION to score. `all_sasa_norm` is the sasa study group (290 individuals) merged with
# the Haifa-KF field export (461). results.md predates the merge, so `SOURCE=sasa` is the
# like-for-like comparison. Read in data.py; d.source_keep_mask / d.source_tag() follow it. The
# gate restricts the scored side; SASA_TRAIN_ONLY=1 also drops the other collection from training.
SASA_TRAIN_ONLY = bool(os.environ.get("SASA_TRAIN_ONLY"))
# strict_hand_pos = the thesis WITH the location penalty; strict_logreg dropped from the default
# (it diluted the thesis in the first run) but still selectable via ONLY=.
ALL_MODELS = ["raw_voting", "logreg", "strict_hand", "strict_hand_pos", "strict_logreg", "llr"]
DEFAULT_MODELS = ["raw_voting", "logreg", "strict_hand", "strict_hand_pos"]


def build_pos_lookup(frame, db_path=None):
    """``(sid, spot_id) -> (axis_t, axis_offset/length_px)`` — the scale-invariant body-frame
    position each spot sits at. ``length_px`` (per image) comes from the ``body_axis`` table;
    ``axis_offset`` is already signed (left negative / right positive), so mirror positions differ.
    """
    import duckdb
    con = duckdb.connect(str(db_path or d.DB_PATH), read_only=True)
    try:
        length = {r[0]: r[1] for r in
                  con.execute("SELECT salamander_id, length_px FROM body_axis").fetchall()}
    finally:
        con.close()
    out = {}
    for sid, spid, t, off in zip(frame["sid"], frame["spot_id"], frame["axis_t"],
                                 frame["axis_offset"]):
        L = length.get(sid)
        lat = (off / L) if (L and np.isfinite(off)) else np.nan
        out[(sid, int(spid))] = (float(t) if np.isfinite(t) else np.nan, float(lat))
    return out


def _recall_at_k(scores, qids, clab, true_by_q, ks=(1, 5, 10)):
    """Recall@k over KNOWN queries: is the true individual in the top-k ranked candidates. This is
    the human-in-the-loop shortlist metric ('is the answer in the top-10 I review'). Novel queries
    (no gallery match) are excluded — they have no correct answer to rank."""
    scores = np.asarray(scores, float); qids = np.asarray(qids); clab = np.asarray(clab, dtype=object)
    hits = {k: 0 for k in ks}; n = 0
    for q in np.unique(qids):
        m = qids == q; true = true_by_q.get(q); cl = clab[m]
        if true not in set(cl):
            continue
        n += 1
        ranked = cl[np.argsort(-scores[m])]
        rank = int(np.where(ranked == true)[0][0]) + 1
        for k in ks:
            hits[k] += int(rank <= k)
    return {k: (hits[k] / n if n else float("nan")) for k in ks}


def _coverage_at_precision(sweep, target=0.90):
    """The ABSTAIN trade-off: raise the confidence bar until emitted matches are >= ``target``
    precise, and report how many queries you can still answer (coverage). Returns
    ``(coverage, achieved_precision, threshold)`` — coverage = fraction of queries where the model
    commits to a match (the rest it abstains on = 'new / not sure'). NaN if the bar is unreachable.
    """
    best = (0.0, float("nan"), float("nan"))
    for r in sweep["rows"]:
        nq = r["n_known"] + r["n_novel"]
        emitted = r["tp"] + r["fp"]
        cov = emitted / nq if nq else 0.0
        p = r["precision"]
        if np.isfinite(p) and p >= target and cov > best[0]:
            best = (cov, float(p), r["thr"])
    return best


def _census_row(scores, qids, clab, true_by_q, unexpl=None, target_p=0.90):
    """Census F0.5 + ident R@1/5/10 + open-set AUROC + the ABSTAIN operating point.

    ``cov_at_p`` answers the user's ask directly: if the matcher only commits when it is
    ``target_p`` (default 0.90) likely correct and abstains otherwise, on what fraction of photos
    can it still commit? High precision + decent coverage = "trust the confident calls."
    """
    top = per_query_top1(np.asarray(scores, float), qids, clab, true_by_q)
    sweep, best = cen.census_sweep(top["top1"], top["top1_correct"], top["is_known"], BETA)
    known = top["is_known"].astype(bool)
    ident_r1 = float(top["top1_correct"][known].mean()) if known.any() else float("nan")
    auroc_top1 = cen.auroc(top["top1"], top["is_known"])
    rk = _recall_at_k(scores, qids, clab, true_by_q)
    cov_p, ach_p, _ = _coverage_at_precision(sweep, target_p)

    auroc_unexpl = float("nan")
    if unexpl is not None:
        qids = np.asarray(qids); scores = np.asarray(scores, float); unexpl = np.asarray(unexpl, float)
        kscore, isk = [], []
        for q in np.unique(qids):
            m = qids == q
            j = np.argmax(scores[m])
            kscore.append(-float(unexpl[m][j]))
            isk.append(int(true_by_q.get(q) in set(np.asarray(clab)[m])))
        auroc_unexpl = cen.auroc(np.array(kscore), np.array(isk))

    return dict(census_f=best["f"], census_p=best["precision"], census_r=best["recall"],
                ident_r1=ident_r1, r5=rk[5], r10=rk[10], auroc_top1=auroc_top1,
                auroc_unexpl=auroc_unexpl, cov_at_p=cov_p, prec_at_cov=ach_p)


def build_size_tpos(frame, pos_lookup):
    """``(sid, spot_id) -> size percentile`` and ``-> axis_t``, the two per-spot inputs the
    likelihood model conditions on (a big spot in a visible region is expected to survive; its
    absence is therefore informative, and a small one's is not)."""
    size = {(sid, int(spid)): float(v) if np.isfinite(v) else 0.5
            for sid, spid, v in zip(frame["sid"], frame["spot_id"], frame["size"])}
    tpos = {k: v[0] for k, v in pos_lookup.items()}
    return size, tpos


def run_fold(sets, factors_frame, pos_lookup, tr, ev, models, quality=None,
             size_lookup=None, tpos_lookup=None):
    train_imgs = [i for i in tr if not sets[i].is_synth]
    train_labels = {sets[i].label for i in train_imgs}

    # --- per-spot weight: learned head (fit on TRAIN individuals only), hand blend, or uniform ---
    fr = factors_frame
    if WEIGHT_MODE == "learned":
        ind = np.array(["_".join(s.split("_")[:2]) for s in fr["sid"]])
        fit_mask = fr["labeled"].to_numpy() & np.isin(ind, list(train_labels))
        Xd = fr.loc[fit_mask, dist.FACTOR_NAMES].to_numpy(float)
        yd = fr.loc[fit_mask, "y"].to_numpy(float)
        dmodel, dscaler = train_aggregator(Xd, yd, hidden=0, seed=SEED)
        wlookup = dist.weight_lookup(dmodel, dscaler, fr)
    elif WEIGHT_MODE == "hand":
        wlookup = dist.hand_weight_lookup(fr)
    elif WEIGHT_MODE == "uniform":
        wlookup = dist.uniform_weight_lookup(fr)
    else:
        raise SystemExit(f"WEIGHT_MODE must be learned|hand|uniform, got {WEIGHT_MODE!r}")

    gal, qry = cen.make_openset_split(sets, ev, NOVEL_FRAC, SEED)
    true_by_q = {q: sets[q].label for q in qry}
    out = {}

    # plain summary features (shared by raw_voting + logreg)
    need_plain = {"raw_voting", "logreg"} & set(models)
    if need_plain:
        Xo, oq, oc = cen.build_openset_features(sets, gal, qry)
        if "raw_voting" in models:
            out["raw_voting"] = _census_row(Xo[:, 0], oq, oc, true_by_q)
        if "logreg" in models:
            Xtr, ytr, _, _ = build_pairs_plain(sets, train_imgs, neg_per_query=NEG_PER_QUERY, seed=SEED)
            m, sc = train_aggregator(Xtr, ytr, hidden=0, seed=SEED)
            out["logreg"] = _census_row(_prob(m, sc, Xo), oq, oc, true_by_q)

    if "strict_hand" in models:
        s_h, hq, hc = sv.build_openset_hand(sets, gal, qry, wlookup)
        out["strict_hand"] = _census_row(s_h, hq, hc, true_by_q)

    if "strict_hand_pos" in models:
        s_p, pq, pc = sv.build_openset_hand(sets, gal, qry, wlookup,
                                            pos_lookup=pos_lookup, sigma_pos=SIGMA_POS)
        out["strict_hand_pos"] = _census_row(s_p, pq, pc, true_by_q)

    if "llr" in models:
        # Partial-match scoring: every spot contributes what it is measured to be WORTH -- a match
        # ~+0.44 log-odds, a miss only ~-0.11 -- instead of the pair being charged for the 75% of
        # pattern that a true pair is expected to lose anyway. Fitted on training individuals only.
        lm = llr.LLRModel.fit(sets, train_imgs, size_lookup, tpos_lookup, seed=SEED)
        s_l, lq, lc = llr.build_openset_llr(lm, sets, gal, qry, size_lookup, tpos_lookup)
        out["llr"] = _census_row(s_l, lq, lc, true_by_q)
        out["_llr_model"] = lm

    if "strict_logreg" in models:
        Xs_tr, ys_tr, _, _ = sv.build_pairs(sets, train_imgs, wlookup,
                                            neg_per_query=NEG_PER_QUERY, seed=SEED,
                                            quality=quality)
        ms, scs = train_aggregator(Xs_tr, ys_tr, hidden=0, seed=SEED)
        Xs, sq, sc2, unexpl = sv.build_openset(sets, gal, qry, wlookup, quality=quality)
        out["strict_logreg"] = _census_row(_prob(ms, scs, Xs), sq, sc2, true_by_q, unexpl=unexpl)
        out["_strict_weights"] = (ms.net.weight.detach().numpy().ravel(), scs)
    return out


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    pool = ALL_MODELS if ONLY else DEFAULT_MODELS
    models = [m for m in pool if not ONLY or m in ONLY]
    logger.info("loading data + distinctiveness factors ...")
    logger.info(" " + rl.review_summary(d.dataset_name))
    sets = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))
    factors = dist.load_spot_factors(sets)
    labels = dist.load_interesting(path=dist.INTERESTING_JSON if LABELS == "legacy" else None)
    frame = dist.attach_labels(factors, labels)
    pos_lookup = build_pos_lookup(frame)
    quality = sv.load_quality() if sv.USE_QUALITY_INTERACTION else None
    logger.info(f" weights={WEIGHT_MODE}  labels={LABELS} "
          f"({frame['sid'][frame['labeled']].nunique()} labeled images, "
          f"{int(frame['y'].sum())} interesting spots)  eval_gate={EVAL_GATE}")
    logger.info(f" quality-interaction={'ON' if sv.USE_QUALITY_INTERACTION else 'OFF'}"
          + (f" ({len(quality)} photos with a quality score)" if quality else ""))
    logger.info(f" correspondence gate  assign={sm.GATE_ASSIGN}  match_thr={sm.GATE_MATCH_THR:g}"
          f"   (strict_* only; raw_voting/logreg are the untouched control)")

    # The gate applies to the SCORED (eval) side only, keeping the dropped photos as training data —
    # the setting that measured better in sweep_quality_control.
    eval_mask = np.ones(len(sets), dtype=bool)
    if EVAL_GATE in ("quality", "both"):
        eval_mask &= d.quality_keep_mask(sets)
    if EVAL_GATE in ("review", "both"):
        eval_mask &= d.review_keep_mask(sets, unreviewed=UNREVIEWED)
    source_keep = d.source_keep_mask(sets)          # all-True unless SOURCE=sasa|kf
    eval_mask &= source_keep
    folds = d.get_cv_folds(sets, k=K_FOLDS, seed=SEED, eval_mask=eval_mask)
    if SASA_TRAIN_ONLY and not source_keep.all():
        folds = [([i for i in tr if source_keep[i]], ev) for tr, ev in folds]
    d.assert_evaluable(folds, what=f"EVAL_GATE={EVAL_GATE} SOURCE={d.SOURCE}")

    qtag = d.quality_tag()
    # The correspondence gate goes in the tag for the same reason EMB_TABLE does: without it every
    # arm of run_gate.sh writes one filename and only the last one survives (results.md #46
    # Correction — a change does not ship until every reader of it is checked).
    gate_default = sm.GATE_ASSIGN == "mutual" and sm.GATE_MATCH_THR == 0.4
    tag = qtag + ("" if WEIGHT_MODE == "learned" else f"_w{WEIGHT_MODE}") \
              + ("" if LABELS == "review" else f"_{LABELS}") \
              + ("" if EVAL_GATE == "quality" else f"_gate{EVAL_GATE}") \
              + ("" if gate_default else f"_{sm.GATE_ASSIGN}{sm.GATE_MATCH_THR:g}") \
              + ("_qi" if sv.USE_QUALITY_INTERACTION else "") \
              + d.source_tag() + ("_traingone" if (SASA_TRAIN_ONLY and d.SOURCE != "all") else "")
    logger.info("=" * 74)
    logger.info(f" STRICT vs EQUAL-WEIGHT  ({len(models)} models x {K_FOLDS} folds"
          f"{'  [QUICK]' if QUICK else ''})   headline = census F0.5   quality={qtag or 'none'}")
    logger.info("=" * 74)

    size_lookup, tpos_lookup = build_size_tpos(frame, pos_lookup)
    per_fold = {m: [] for m in models}
    last_weights = None
    last_llr = None
    for fi, (tr, ev) in enumerate(folds):
        t0 = time.time()
        res = run_fold(sets, frame, pos_lookup, tr, ev, models, quality=quality,
                       size_lookup=size_lookup, tpos_lookup=tpos_lookup)
        if "_strict_weights" in res:
            last_weights = res.pop("_strict_weights")
        if "_llr_model" in res:
            last_llr = res.pop("_llr_model")
        for m, v in res.items():
            per_fold[m].append(v)
        line = "  ".join(f"{m}:F{res[m]['census_f']:.3f}/R1:{res[m]['ident_r1']:.3f}"
                         for m in models if m in res)
        logger.info(f" fold {fi}  {line}   ({time.time()-t0:.0f}s)")

    def agg(m, k):
        v = [r[k] for r in per_fold[m]]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), 0.0)

    logger.info("=" * 104)
    logger.info(f" {'model':<15}{'censusF0.5':>13}{'R@1':>7}{'R@5':>7}{'R@10':>7}"
          f"{'AUROC':>8}{'cov@P90':>9}   (abstain: %queries answered at >=90% precision)")
    logger.info("-" * 104)
    rows = []
    for m in models:
        f_m, f_s = agg(m, "census_f")
        rows.append((m, f_m, f_s, agg(m, "ident_r1")[0], agg(m, "r5")[0], agg(m, "r10")[0],
                     agg(m, "auroc_top1")[0], agg(m, "cov_at_p")[0]))
    for m, fm, fs, ir, r5, r10, at, cov in sorted(rows, key=lambda x: -x[1]):
        logger.info(f" {m:<15}{fm:>7.3f}±{fs:<5.3f}{ir:>7.3f}{r5:>7.3f}{r10:>7.3f}{at:>8.3f}{cov:>9.3f}")

    if last_llr is not None:
        logger.info(" llr — what a match and a miss are WORTH (last fold, fitted on train only):")
        logger.info(last_llr.describe())

    if last_weights is not None:
        w, _ = last_weights
        logger.info(" strict_logreg learned weights (standardized, last fold):")
        for nm, wt in sorted(zip(sv.STRICT_FEATURE_NAMES, w), key=lambda t: -abs(t[1])):
            logger.info(f"   {nm:20s} {wt:+.3f}")

    # --- write results ---
    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "sweeps" / "strict"
    outdir.mkdir(parents=True, exist_ok=True)
    md = ["# Strict distinctiveness-weighted matching vs equal-weight voting", "",
          f"- dataset `{d.dataset_name}` · folds {K_FOLDS} · seed {SEED} · neg/query {NEG_PER_QUERY}"
          f" · quality `{qtag or 'none'}` · sigma_pos {SIGMA_POS}"
          + ("  **[QUICK — not a result]**" if QUICK else ""),
          f"- weights `{WEIGHT_MODE}` · labels `{LABELS}` · eval gate `{EVAL_GATE}`"
          + (f" (unreviewed={UNREVIEWED})" if EVAL_GATE in ("review", "both") else "")
          + f" · source `{d.SOURCE}`"
          + ("  (Haifa-KF removed from training too)"
             if (SASA_TRAIN_ONLY and d.SOURCE != "all") else ""),
          (f"- Scored on the **{d.SOURCE}** population only; results.md's figures predate the "
           "Haifa-KF merge, so this is the like-for-like comparison."
           ) if d.SOURCE != "all" else "",
          f"- {rl.review_summary(d.dataset_name)}",
          "- **R@1/5/10** = correct animal in the top-k shortlist (known queries). **cov@P90** = the",
          "  ABSTAIN operating point: fraction of photos the matcher can commit to while keeping",
          "  emitted matches >=90% precise (the rest it abstains on = 'new / not sure').",
          "",
          "| model | census F0.5 | R@1 | R@5 | R@10 | AUROC | cov@P90 |",
          "|---|---|---|---|---|---|---|"]
    for m, fm, fs, ir, r5, r10, at, cov in sorted(rows, key=lambda x: -x[1]):
        md.append(f"| {m} | **{fm:.3f} ± {fs:.3f}** | {ir:.3f} | {r5:.3f} | {r10:.3f} | {at:.3f} "
                  f"| {cov:.3f} |")
    if last_weights is not None:
        w, _ = last_weights
        md += ["", "**strict_logreg weights** (standardized, last fold):", ""]
        for nm, wt in sorted(zip(sv.STRICT_FEATURE_NAMES, w), key=lambda t: -abs(t[1])):
            md.append(f"- `{nm}` {wt:+.3f}")
    # emb_tag() names the active EMB_TABLE, as sweep_all9 does: without it a descriptor comparison
    # (baseline vs morph) silently overwrites its own first arm and only the last one survives.
    fname = "RESULTS_strict" + tag + ("_quick" if QUICK else "") + d.emb_tag() + ".md"
    (outdir / fname).write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info(f"wrote {outdir / fname}")


if __name__ == "__main__":
    main()
