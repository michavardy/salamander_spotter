"""Fit the strict matcher's edge gate on human accept/reject verdicts.

``strict_match`` decides whether two spots correspond with four hand-set constants::

    quality = relu(cosine) * exp(-d2 / (2*sigma_pos^2))     # appearance x body-position agreement
    keep the edge if quality >= match_thr                    # 0.4
    it counts toward n_good  if quality >= good_thr          # 0.5 in residual_parts, 0.3 in
                                                             # strict_pair_score — they disagree
and ``sigma_pos`` itself is 0.5 in :func:`strict_match.strict_pair_score` but 0.12 in
:func:`strict_match.residual_parts` and ``strict_voter``. Nobody picked these against data; there
was no data to pick them against.

There is now. The preprocessing review holds **427 machine-proposed correspondences with a human
verdict** (342 accepted, 85 rejected) — the 85 are hard negatives the current gate scored as
matches. Four scalars fitted on 427 labels is a comfortable ratio, and it attacks the measured
80.1% edge precision directly.

**What this can and cannot conclude.** The verdict edges were themselves produced by the matcher
(mutual nearest neighbours above its own cutoff), so this is a *conditional* calibration: it fits
"given the matcher proposed this edge, should it be kept", which is exactly the re-ranking problem,
and says nothing about correspondences the matcher never proposed. Those exist in quantity — 415
of them, drawn by hand — and no threshold on a proposal set can recover them. Recall is an
extraction problem, not a threshold problem, and this module deliberately does not pretend
otherwise.

Second caveat, equally load-bearing: only 79 of these edges join two real photographs. The rest
involve a Gemini-generated view, where "the same spot" holds by construction. Per-``pair_kind``
numbers are printed for that reason; trust the ``real-real`` column and treat the rest as a much
larger, much softer prior.

    pixi run gate-calibration                       # fit + report + write fitted_gate.json
    pixi run gate-calibration --pair-kind real-real # the honest subset (n=79)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval"):
    p = str(_ST / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

import data as d                                  # noqa: E402
import review_labels as rl                        # noqa: E402
from census import auroc                          # noqa: E402

# the constants as they stand today, for the before/after
CURRENT = {"sigma_pos": 0.12, "match_thr": 0.40, "good_thr": 0.50}
SIGMA_GRID = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.30, 0.50, 1e9])
THR_GRID = np.round(np.arange(0.02, 0.96, 0.02), 3)


# ----------------------------------------------------------------------------- inputs
def load_spot_geometry(dataset: str | None = None, db_path=None) -> pd.DataFrame:
    """``sid, spot_id, axis_t, lateral`` — body-frame position per spot.

    ``lateral`` is ``axis_offset / length_px``: the signed left/right offset in body-length units,
    which is what makes the position gate scale-invariant (and mirror-sensitive, deliberately —
    a spot on the left flank is not the same spot as its mirror on the right).
    """
    import duckdb
    con = duckdb.connect(str(db_path or d.DB_PATH), read_only=True)
    try:
        spots = con.execute(
            "SELECT salamander_id AS sid, spot_id, axis_t, axis_offset FROM spots"
        ).df()
        length = {r[0]: r[1] for r in
                  con.execute("SELECT salamander_id, length_px FROM body_axis").fetchall()}
    finally:
        con.close()
    L = spots["sid"].map(length)
    spots["lateral"] = np.where(L.notna() & (L > 0), spots["axis_offset"] / L, np.nan)
    return spots[["sid", "spot_id", "axis_t", "lateral"]]


def load_edge_features(dataset: str | None = None) -> pd.DataFrame:
    """The verdict edges joined to the two things the gate looks at: appearance cosine and
    body-frame separation. One row per adjudicated correspondence.

    Returns the verdict columns plus ``cos`` (embedding cosine, the same 62-dim space the matcher
    scores in) and ``d2`` (squared body-frame distance). Edges whose spots are missing from the
    embedding table — the review can outlive a re-extraction — are dropped and counted.
    """
    dataset = dataset or d.dataset_name
    df = rl.verdict_edges(dataset, proposed_by="algorithm")
    if not len(df):
        return df

    emb = d.get_spot_embeddings()
    E = {(r.salamander_id, int(r.spot_id)): r.embedding for r in emb.itertuples(index=False)}
    geo = load_spot_geometry(dataset)
    G = {(r.sid, int(r.spot_id)): (r.axis_t, r.lateral) for r in geo.itertuples(index=False)}

    cos, d2, ok = [], [], []
    for r in df.itertuples(index=False):
        ka, kb = (r.sid_a, r.spot_a), (r.sid_b, r.spot_b)
        ea, eb = E.get(ka), E.get(kb)
        if ea is None or eb is None:
            cos.append(np.nan); d2.append(np.nan); ok.append(False); continue
        ea = np.asarray(ea, float); eb = np.asarray(eb, float)
        c = float(ea @ eb / ((np.linalg.norm(ea) * np.linalg.norm(eb)) + 1e-12))
        pa, pb = G.get(ka), G.get(kb)
        if pa is None or pb is None or not np.isfinite([*pa, *pb]).all():
            # no fitted axis -> the position gate cannot speak. NaN, not 0: treating an unknown
            # position as perfect agreement is how a gate silently stops gating.
            dd = np.nan
        else:
            dd = float((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        cos.append(c); d2.append(dd); ok.append(True)

    df = df.assign(cos=cos, d2=d2, resolved=ok)
    return df


def quality(cos: np.ndarray, d2: np.ndarray, sigma: float) -> np.ndarray:
    """The gate's own score: ``relu(cosine) * exp(-d2 / 2sigma^2)``.

    A NaN ``d2`` (no fitted body axis on one side) leaves the position gate open at 1.0, matching
    ``strict_voter.body_coords``, which defaults a missing axis to mid-body rather than dropping
    the spot. Recorded here so the fallback is visible in the fit rather than buried.
    """
    cos = np.asarray(cos, float); d2 = np.asarray(d2, float)
    if sigma >= 1e8:                     # position gate disabled -> appearance only
        gate = np.ones_like(cos)
    else:
        gate = np.exp(-np.nan_to_num(d2, nan=0.0) / (2.0 * sigma * sigma))
        gate = np.where(np.isfinite(d2), gate, 1.0)
    return np.clip(cos, 0.0, None) * gate


# ----------------------------------------------------------------------------- fitting
def fbeta_at(q: np.ndarray, y: np.ndarray, thr: float, beta: float = 0.5) -> float:
    """F-beta of "keep this edge" at ``thr``. beta=0.5 weights precision 2x, matching the repo's
    census headline — a wrong correspondence is worse than a missed one here too, because a wrong
    one actively contributes explained mass to a false pair."""
    pred = q >= thr
    tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum())
    fn = float((~pred & (y == 1)).sum())
    if tp == 0:
        return 0.0
    p = tp / (tp + fp); r = tp / (tp + fn)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def precision_at(q: np.ndarray, y: np.ndarray, thr: float) -> tuple[float, float]:
    """``(precision, coverage)`` of keeping edges at ``thr``."""
    pred = q >= thr
    if not pred.any():
        return float("nan"), 0.0
    return float(y[pred].mean()), float(pred.mean())


def fit_gate(df: pd.DataFrame, *, beta: float = 0.5, k: int = 5, seed: int = 0) -> dict:
    """Individual-split CV over ``(sigma_pos, match_thr)``, plus a ``good_thr`` at 90% precision.

    Splitting on individual (not on edge) is the whole point: two edges from the same animal share
    its extraction quality and its body-axis fit, so an edge-level split would report a threshold
    that memorised 55 animals rather than one that transfers to a new one.
    """
    use = df[df["resolved"]].reset_index(drop=True)
    y = use["accepted"].to_numpy().astype(int)
    inds = use["individual"].to_numpy()
    uniq = np.array(sorted(set(inds)))
    rng = np.random.default_rng(seed)
    folds = np.array_split(uniq[rng.permutation(len(uniq))], min(k, len(uniq)))

    # --- sigma: which position-gate width best separates accepted from rejected (AUROC, no
    #     threshold involved, so it is not confounded with the cutoff choice) ---
    sigma_auroc = {}
    for s in SIGMA_GRID:
        sigma_auroc[float(s)] = float(auroc(quality(use["cos"], use["d2"], s), y))
    best_sigma = max(sigma_auroc, key=sigma_auroc.get)

    # --- threshold: held-out F-beta, sigma fixed per fold by the same rule on the training part ---
    rows, oof = [], np.full(len(use), np.nan)
    for f in folds:
        te = np.isin(inds, f); tr = ~te
        if y[tr].sum() == 0 or y[te].sum() == 0 or (y[tr] == 0).sum() == 0:
            continue
        s_tr = max(SIGMA_GRID, key=lambda s: auroc(quality(use["cos"][tr], use["d2"][tr], s), y[tr]))
        q_tr = quality(use["cos"][tr], use["d2"][tr], s_tr)
        q_te = quality(use["cos"][te], use["d2"][te], s_tr)
        t_tr = max(THR_GRID, key=lambda t: fbeta_at(q_tr, y[tr], t, beta))
        oof[te] = q_te
        rows.append(dict(sigma=float(s_tr), thr=float(t_tr),
                         f_te=fbeta_at(q_te, y[te], t_tr, beta),
                         f_cur=fbeta_at(quality(use["cos"][te], use["d2"][te],
                                                CURRENT["sigma_pos"]),
                                        y[te], CURRENT["match_thr"], beta),
                         n_te=int(te.sum())))
    cv = pd.DataFrame(rows)

    q_all = quality(use["cos"], use["d2"], best_sigma)
    best_thr = float(max(THR_GRID, key=lambda t: fbeta_at(q_all, y, t, beta)))

    # good_thr: the quality at which kept edges are >=90% correct — the "counts as corroboration"
    # bar, set to the same precision target cov@P90 uses elsewhere.
    good = float("nan")
    for t in THR_GRID:
        p, cov = precision_at(q_all, y, t)
        if np.isfinite(p) and p >= 0.90 and cov > 0:
            good = float(t); break

    p_new, cov_new = precision_at(q_all, y, best_thr)
    p_cur, cov_cur = precision_at(quality(use["cos"], use["d2"], CURRENT["sigma_pos"]), y,
                                  CURRENT["match_thr"])

    # The separability check that decides whether any of the above means anything. If the inputs the
    # gate sees do not differ between accepted and rejected edges, no setting of its constants can
    # separate them, and a "fitted" threshold is noise dressed as a result.
    acc = use["accepted"].to_numpy().astype(bool)
    sep = dict(
        cos_pos=float(use["cos"][acc].mean()), cos_neg=float(use["cos"][~acc].mean()),
        cos_pos_sd=float(use["cos"][acc].std()), cos_neg_sd=float(use["cos"][~acc].std()),
        d2_pos=float(np.nanmean(use["d2"][acc])), d2_neg=float(np.nanmean(use["d2"][~acc])),
        auroc_cos=float(auroc(use["cos"].to_numpy(float), y)),
        auroc_d2=float(auroc(-np.nan_to_num(use["d2"].to_numpy(float), nan=0.0), y)),
    )
    return dict(separability=sep,
        n=len(use), n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
        n_individuals=len(uniq), n_unresolved=int((~df["resolved"]).sum()),
        sigma_auroc=sigma_auroc, best_sigma=float(best_sigma), best_thr=best_thr,
        good_thr=good,
        auroc_best=float(auroc(q_all, y)),
        auroc_current=float(auroc(quality(use["cos"], use["d2"], CURRENT["sigma_pos"]), y)),
        auroc_oof=float(auroc(oof[~np.isnan(oof)], y[~np.isnan(oof)])) if np.isfinite(oof).any()
        else float("nan"),
        cv=cv, fitted_precision=p_new, fitted_coverage=cov_new,
        current_precision=p_cur, current_coverage=cov_cur,
    )


# ----------------------------------------------------------------------------- report
def main():
    import argparse
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair-kind", default=None,
                    choices=["real-real", "real-synth", "synth-synth"],
                    help="restrict to one pair kind; real-real is the only fully honest subset")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = d.dataset_name
    logger.info(f"dataset {ds}")
    logger.info(" " + rl.review_summary(ds))
    logger.info("loading verdict edges + embeddings ...")
    df = load_edge_features(ds)
    if not len(df):
        raise SystemExit("no algorithm-proposed verdict edges — nothing to calibrate")
    if args.pair_kind:
        df = df[df["pair_kind"] == args.pair_kind].reset_index(drop=True)
        logger.info(f" restricted to pair_kind={args.pair_kind}: {len(df)} edges")

    logger.info(f" {len(df)} edges  ({int(df['accepted'].sum())} accepted / "
          f"{int((~df['accepted']).sum())} rejected)  "
          f"{int((~df['resolved']).sum())} unresolvable (spot not in the embedding table)")
    by_kind = df.groupby("pair_kind")["accepted"].agg(["size", "sum"])
    logger.info(" by pair kind:")
    for kind, r in by_kind.iterrows():
        logger.info(f"   {kind:<12} n={int(r['size']):>4}  accepted {int(r['sum']):>4} "
              f"({r['sum'] / r['size']:.3f})")

    res = fit_gate(df, beta=args.beta, k=args.k, seed=args.seed)

    logger.info("=== Which position-gate width separates accepted from rejected? (AUROC) ===")
    for s, a in sorted(res["sigma_auroc"].items()):
        lab = "appearance only" if s >= 1e8 else f"sigma={s:.2f}"
        star = "  <- best" if s == res["best_sigma"] else ("  (current)" if s == CURRENT["sigma_pos"] else "")
        logger.info(f"   {lab:<18} {a:.3f}{star}")

    logger.info("=== Fitted vs current constants ===")
    logger.info(f"   {'':22}{'sigma_pos':>11}{'match_thr':>11}{'AUROC':>8}{'precision':>11}{'coverage':>10}")
    logger.info(f"   {'current (hand-set)':22}{CURRENT['sigma_pos']:>11.2f}{CURRENT['match_thr']:>11.2f}"
          f"{res['auroc_current']:>8.3f}{res['current_precision']:>11.3f}{res['current_coverage']:>10.3f}")
    sig = "appearance" if res["best_sigma"] >= 1e8 else f"{res['best_sigma']:.2f}"
    logger.info(f"   {'fitted (in-sample)':22}{sig:>11}{res['best_thr']:>11.2f}"
          f"{res['auroc_best']:>8.3f}{res['fitted_precision']:>11.3f}{res['fitted_coverage']:>10.3f}")
    base = res["n_pos"] / max(res["n"], 1)
    good_s = (f"{res['good_thr']:.2f}" if np.isfinite(res["good_thr"])
              else f"UNREACHABLE (base rate {base:.2f}; no threshold reaches 90%)")
    logger.info(f"   good_thr at 90% edge precision: {good_s}   (currently {CURRENT['good_thr']:.2f})")

    sep = res["separability"]
    logger.info("=== Do the gate's INPUTS separate accepted from rejected at all? ===")
    logger.info(f"   cosine     accepted {sep['cos_pos']:.3f} ± {sep['cos_pos_sd']:.3f}   "
          f"rejected {sep['cos_neg']:.3f} ± {sep['cos_neg_sd']:.3f}   AUROC {sep['auroc_cos']:.3f}")
    logger.info(f"   body dist  accepted {sep['d2_pos']:.4f}          rejected {sep['d2_neg']:.4f}"
          f"           AUROC {sep['auroc_d2']:.3f}")
    # At OR BELOW chance both mean the same thing here: the inputs do not carry the verdict.
    # Below-chance is the more dangerous case, because a fitted threshold can still post a high
    # F-beta by collapsing to "keep everything" and harvesting the base rate.
    at_chance = sep["auroc_cos"] <= 0.55 and res["auroc_best"] <= 0.55
    if at_chance:
        logger.info("   VERDICT: the gate's inputs are at chance on the human verdict. The reviewer is")
        logger.info("   judging these edges on something the 62-dim embedding does not encode, so NO")
        logger.info("   setting of sigma_pos/match_thr/good_thr can reproduce their decision. Retuning the")
        logger.info("   constants is not the fix; this is results.md #36 (the representation is the")
        logger.info("   ceiling) measured directly at the edge level.")
        if sep["auroc_cos"] < 0.45:
            logger.info(f"   (Cosine is BELOW chance at {sep['auroc_cos']:.3f} — on this subset the "
                  f"matcher's own\n    similarity is mildly ANTI-correlated with human agreement.)")

    degenerate = res["fitted_coverage"] > 0.95 and res["best_thr"] <= 0.10
    if degenerate:
        logger.warning(f"   WARNING: the fitted threshold ({res['best_thr']:.2f}) keeps "
              f"{res['fitted_coverage']:.0%} of edges — it has collapsed to\n   'keep everything'. "
              f"Any F-beta gain it shows is the {base:.0%} base rate, not discrimination.\n"
              f"   Read the AUROC row instead; F-beta is not interpretable at this operating point.")

    cv = res["cv"]
    if len(cv):
        logger.info(f"=== Held-out (individual-split, {len(cv)} folds) — does it transfer? ===")
        logger.info(f"   fitted   F{args.beta}  {cv['f_te'].mean():.3f} ± {cv['f_te'].std():.3f}")
        logger.info(f"   current  F{args.beta}  {cv['f_cur'].mean():.3f} ± {cv['f_cur'].std():.3f}")
        logger.info(f"   delta            {cv['f_te'].mean() - cv['f_cur'].mean():+.3f}")
        logger.info(f"   out-of-fold AUROC {res['auroc_oof']:.3f}")
        if np.isfinite(res["auroc_oof"]) and res["auroc_oof"] < 0.5 and cv["f_te"].mean() > cv["f_cur"].mean():
            logger.info("   IGNORE the F-beta gain above: out-of-fold AUROC is BELOW chance, so the fitted")
            logger.info("   rule orders edges worse than a coin. The gain comes from where the threshold")
            logger.info("   happens to sit against the base rate, not from separating the two classes.")
        logger.info(f"   per-fold sigma {sorted(cv['sigma'].tolist())}  thr {sorted(cv['thr'].tolist())}")
        if cv["sigma"].nunique() > 1 or cv["thr"].nunique() > 2:
            logger.info("   NOTE: the fitted constants move between folds — that instability is the "
                  "result,\n         and it means one global value is not supported by 427 edges.")

    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "gate_calibration"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.pair_kind}" if args.pair_kind else ""
    payload = {k: v for k, v in res.items() if k != "cv"}
    payload["cv"] = cv.to_dict("records")
    payload["dataset"] = ds
    payload["pair_kind"] = args.pair_kind or "all"
    payload["current"] = CURRENT
    (outdir / f"fitted_gate{tag}.json").write_text(json.dumps(payload, indent=2, default=float),
                                                   encoding="utf-8")

    md = [
        "# Edge-gate calibration on human verdicts", "",
        f"- dataset `{ds}` · pair kinds `{args.pair_kind or 'all'}` · beta {args.beta} · "
        f"{res['n']} edges over {res['n_individuals']} individuals "
        f"({res['n_pos']} accepted / {res['n_neg']} rejected)",
        "- Fits the **conditional** gate: given the matcher proposed an edge, should it be kept.",
        "  It cannot address the 415 correspondences the matcher never proposed — that is recall,",
        "  and no threshold recovers it.", "",
        "| constants | sigma_pos | match_thr | AUROC | edge precision | coverage |",
        "|---|---|---|---|---|---|",
        f"| current (hand-set) | {CURRENT['sigma_pos']:.2f} | {CURRENT['match_thr']:.2f} | "
        f"{res['auroc_current']:.3f} | {res['current_precision']:.3f} | {res['current_coverage']:.3f} |",
        f"| fitted (in-sample) | {sig} | {res['best_thr']:.2f} | {res['auroc_best']:.3f} | "
        f"{res['fitted_precision']:.3f} | {res['fitted_coverage']:.3f} |", "",
        "**Separability of the gate's inputs** — the check that decides whether the row above is a",
        "result or noise:", "",
        f"- cosine: accepted {sep['cos_pos']:.3f} ± {sep['cos_pos_sd']:.3f} vs rejected "
        f"{sep['cos_neg']:.3f} ± {sep['cos_neg_sd']:.3f} (AUROC {sep['auroc_cos']:.3f})",
        f"- body distance: accepted {sep['d2_pos']:.4f} vs rejected {sep['d2_neg']:.4f} "
        f"(AUROC {sep['auroc_d2']:.3f})", "",
    ]
    if degenerate:
        md += [f"> **The fitted row is degenerate**: threshold {res['best_thr']:.2f} keeps "
               f"{res['fitted_coverage']:.0%} of edges, i.e.",
               f"> \"keep everything\". Its F-beta advantage is the {base:.0%} base rate, not "
               f"discrimination — read AUROC.", ""]
    if at_chance:
        md += ["> **Null result, and the useful kind.** The gate's inputs are at chance on the human",
               "> verdict, so no setting of `sigma_pos` / `match_thr` / `good_thr` can reproduce the",
               "> reviewer's decision — the constants were never the problem. The reviewer is judging",
               "> on structure the 62-dim embedding does not encode. This is results.md #36 (\"the",
               "> representation is the ceiling\") measured directly, at the level of a single edge",
               "> rather than a whole ranking.", ""]
    if len(cv):
        md += [f"Held-out F{args.beta} (individual-split): fitted "
               f"**{cv['f_te'].mean():.3f} ± {cv['f_te'].std():.3f}** vs current "
               f"{cv['f_cur'].mean():.3f} ± {cv['f_cur'].std():.3f} "
               f"({cv['f_te'].mean() - cv['f_cur'].mean():+.3f}); out-of-fold AUROC "
               f"{res['auroc_oof']:.3f}.", ""]
    md += [f"`good_thr` at 90% edge precision: **{good_s}** "
           f"(currently {CURRENT['good_thr']:.2f}).", ""]
    (outdir / f"RESULTS_gate_calibration{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info(f"wrote {outdir / f'RESULTS_gate_calibration{tag}.md'}")
    logger.info(f"wrote {outdir / f'fitted_gate{tag}.json'}")


if __name__ == "__main__":
    main()
