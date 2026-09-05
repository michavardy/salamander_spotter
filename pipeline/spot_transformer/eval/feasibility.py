"""Is the constellation/comparability plan worth building? — three measurements, no training.

The plan under test has three claims. Each one is cheap to check and expensive to assume, so this
module checks all three before any of the machinery gets written:

**A. Absence is informative.** The proposal is to charge a pair when a distinctive spot goes
   unmatched. But between two photos of the SAME animal the median mutual-match fraction is already
   0.50 (results.md #10) — half the spots vanish in genuine repeats — so absence is weak evidence
   by default. It is only usable if *how surprised we should be* varies with something observable:
   spot size, distinctiveness, whether that stretch of body was even visible, blur, curl. Part A
   fits exactly that and reports how much of the variance is real. If a spot's disappearance is
   unpredictable, absence cannot be weighted intelligently and step 2 of the plan is dead.

**B. Comparability degrades matching.** The proposal is that curl, camera angle and image quality
   say how much to trust a comparison. Part B measures whether they predict how well two photos of
   the same animal actually match. A factor that does not move this number cannot help downstream.
   (``border_frac`` is already a casualty: it is <=0.038 everywhere, so bodies are essentially never
   cut off by the frame and it has no variance to contribute.)

**C. The interaction is real.** This is the decisive one, and the easiest to get wrong. Adding curl
   as a plain feature teaches a linear model "curled animals match less often" — a main effect.
   What the plan actually needs is "**trust the geometry less when the animals are curled**", which
   is an interaction and is invisible to an additive model unless the product term is handed to it.
   Part C tests the premise directly: it splits photo pairs into comparable and incomparable halves
   and asks whether geometric consistency separates true from false pairs BETTER in the comparable
   half. If the two halves score the same, the interaction does not exist in this data and step 4
   should not be built.

Everything here is measurement on labels you already have — which photos are the same animal — so
there is nothing to annotate and nothing to train.

    pixi run feasibility                 # all three parts, real photos only
    pixi run feasibility --include-synth # include Gemini views (they are generator artifacts:
                                         # their curl/quality describe the generator, not a camera)
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

import data as d                                    # noqa: E402
import strict_match as sm                           # noqa: E402
from aggregator import _norm, attach_centroids      # noqa: E402
from census import auroc                            # noqa: E402

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

MATCH_THR = 0.4          # a spot "survived" if its mutual-NN similarity clears this
SURVIVAL_FACTORS = ["size_pct", "distinct", "observable", "blur_min", "d_curl", "d_aspect",
                    "axis_t", "log_n_other"]
COMPARABILITY = ["d_curl", "d_aspect", "d_solidity", "blur_min", "quality_min", "d_log_nspots"]

# MEASURED, not assumed: `blur_quality` behaves BACKWARDS as a comparability factor here, and the
# reason is framing, not a sign error. It tracks `blur_score` at rho +0.997 (so higher really does
# mean sharper), but it also correlates -0.23 with spot count and -0.41 with median spot AREA --
# high scores go with photos full of fine background texture, i.e. the animal small in frame. Part A
# then does the rest: survival is driven by spot size (+0.52), so "sharper" photos inherit smaller
# spots and lose more of them. Use `overall_quality` as the comparability factor; `blur_quality`
# measures the photograph, not the comparability of the two animals in it.


# ----------------------------------------------------------------------------- inputs
def load_image_meta(db_path=None) -> pd.DataFrame:
    """Per-photo geometry + quality: the conditioning variables, all already computed."""
    import duckdb
    con = duckdb.connect(str(db_path or d.DB_PATH), read_only=True)
    try:
        return con.execute(
            "SELECT salamander_id AS sid, curl_deg, aspect_ratio, solidity, border_frac, "
            "blur_quality, overall_quality, lighting_quality, n_spots, spots_outside_frac "
            "FROM image_quality"
        ).df()
    finally:
        con.close()


def load_spot_meta(db_path=None) -> pd.DataFrame:
    """Per-spot geometry: body position and the size percentile the review labels argue in."""
    import duckdb
    from embeddings import _size_percentile                              # noqa: PLC0415
    con = duckdb.connect(str(db_path or d.DB_PATH), read_only=True)
    try:
        df = con.execute(
            "SELECT s.salamander_id AS sid, s.spot_id, s.area_pixels, s.axis_t, s.axis_offset, "
            "b.length_px FROM spots s LEFT JOIN body_axis b USING (salamander_id)"
        ).df()
    finally:
        con.close()
    df["size_pct"] = _size_percentile(df) / 100.0
    return df


def build_context(include_synth: bool = False):
    """``(sets, per-image meta, per-spot lookups)`` — everything the three parts share."""
    sets = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))
    if not include_synth:
        sets = [s for s in sets if not s.is_synth]
    img = load_image_meta().set_index("sid")
    spot = load_spot_meta()

    size = {(r.sid, int(r.spot_id)): float(r.size_pct) if np.isfinite(r.size_pct) else 0.5
            for r in spot.itertuples(index=False)}
    tpos = {(r.sid, int(r.spot_id)): float(r.axis_t) if np.isfinite(r.axis_t) else np.nan
            for r in spot.itertuples(index=False)}

    # Distinctiveness: the hand blend, computed once over the whole population (it is a
    # dataset-relative quantity). Used only to ask whether *striking* spots vanish less often, so
    # the hand version is adequate -- the learned head would circularly involve the same labels.
    frame = spot.rename(columns={"sid": "salamander_id"}).copy()
    frame["local_contour"] = None
    import duckdb
    con = duckdb.connect(str(d.DB_PATH), read_only=True)
    try:
        cts = con.execute("SELECT salamander_id, spot_id, local_contour FROM spots").df()
    finally:
        con.close()
    cmap = {(r.salamander_id, int(r.spot_id)): r.local_contour for r in cts.itertuples(index=False)}
    frame["local_contour"] = [cmap.get((s, int(i))) for s, i in
                              zip(frame["salamander_id"], frame["spot_id"])]
    frame["local_contour"] = frame["local_contour"].map(
        lambda c: np.zeros((0, 2)) if c is None else np.array([list(p) for p in c], float))
    emb = d.get_spot_embeddings()
    E = {(r.salamander_id, int(r.spot_id)): np.asarray(r.embedding, float)
         for r in emb.itertuples(index=False)}
    M = np.array([E.get((s, int(i)), np.zeros(62)) for s, i in
                  zip(frame["salamander_id"], frame["spot_id"])], float)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    dist = sm.distinctiveness(frame, M)
    distinct = {(s, int(i)): float(w) for s, i, w in
                zip(frame["salamander_id"], frame["spot_id"], dist)}
    return sets, img, size, tpos, distinct


def _meta(img: pd.DataFrame, sid: str, col: str, default=np.nan) -> float:
    try:
        v = img.at[sid, col]
    except KeyError:
        return default
    return float(v) if v is not None and np.isfinite(v) else default


# ----------------------------------------------------------------------------- Part A
def spot_survival(sets, img, size, tpos, distinct, *, max_pairs_per_ind: int = 12, seed=0
                  ) -> pd.DataFrame:
    """One row per (spot, other-photo-of-the-same-animal): did it survive, and under what
    conditions.

    ``survived`` = the spot has a mutual nearest neighbour in the other photo clearing
    ``MATCH_THR``. That is the same test the matcher itself applies, so the model fitted here
    predicts the matcher's own behaviour rather than an abstraction of it.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[int]] = {}
    for i, s in enumerate(sets):
        if len(s.spots) >= 3:
            by_label.setdefault(s.label, []).append(i)

    rows = []
    for lbl, idxs in by_label.items():
        if len(idxs) < 2:
            continue
        pairs = [(a, b) for a in idxs for b in idxs if a != b]
        if len(pairs) > max_pairs_per_ind:
            pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs_per_ind, replace=False)]
        for ai, bi in pairs:
            A, B = sets[ai], sets[bi]
            S = _norm(A.spots) @ _norm(B.spots).T
            abest = S.argmax(1); bbest = S.argmax(0)
            mutual = {i for i in range(len(S)) if bbest[abest[i]] == i}

            tb = np.array([tpos.get((B.sid, int(j)), np.nan) for j in B.spot_ids], float)
            span = sm.axial_span(tb)
            ta = np.array([tpos.get((A.sid, int(j)), np.nan) for j in A.spot_ids], float)
            obs = sm.observability(ta, span)

            d_curl = abs(_meta(img, A.sid, "curl_deg", 0) - _meta(img, B.sid, "curl_deg", 0))
            d_asp = abs(_meta(img, A.sid, "aspect_ratio", 0) - _meta(img, B.sid, "aspect_ratio", 0))
            blur_min = min(_meta(img, A.sid, "blur_quality", 0.5),
                           _meta(img, B.sid, "blur_quality", 0.5))

            for k, spid in enumerate(A.spot_ids):
                key = (A.sid, int(spid))
                surv = int(k in mutual and float(S[k, abest[k]]) >= MATCH_THR)
                rows.append(dict(
                    label=lbl, sid_a=A.sid, sid_b=B.sid, spot=int(spid), survived=surv,
                    size_pct=size.get(key, 0.5), distinct=distinct.get(key, 0.5),
                    observable=float(obs[k]),
                    blur_min=blur_min, d_curl=d_curl, d_aspect=d_asp,
                    axis_t=float(ta[k]) if np.isfinite(ta[k]) else 0.5,
                    log_n_other=float(np.log1p(len(B.spots))),
                ))
    return pd.DataFrame(rows)


def fit_survival(df: pd.DataFrame, *, k: int = 5, seed: int = 0) -> dict:
    """Individual-split logistic regression: how predictable is a spot's disappearance?

    Split on individual because two spots on one animal share its extraction quality; an
    row-level split would report a model that memorised the animals rather than the conditions.
    """
    from aggregator import train_aggregator, _prob                       # noqa: PLC0415
    X = df[SURVIVAL_FACTORS].to_numpy(float)
    y = df["survived"].to_numpy(float)
    labels = df["label"].to_numpy()
    uniq = np.array(sorted(set(labels)))
    rng = np.random.default_rng(seed)
    folds = np.array_split(uniq[rng.permutation(len(uniq))], k)

    oof = np.full(len(df), np.nan)
    aurocs = []
    for f in folds:
        te = np.isin(labels, f); tr = ~te
        if y[tr].sum() in (0, tr.sum()) or y[te].sum() in (0, te.sum()):
            continue
        m, sc = train_aggregator(X[tr], y[tr], hidden=0, seed=seed)
        p = _prob(m, sc, X[te])
        oof[te] = p
        aurocs.append(auroc(p, y[te]))
    ok = ~np.isnan(oof)
    m, sc = train_aggregator(X, y, hidden=0, seed=seed)
    w = m.net.weight.detach().numpy().ravel()
    return dict(base_rate=float(y.mean()), n=int(len(df)), n_individuals=int(len(uniq)),
                auroc_mean=float(np.mean(aurocs)) if aurocs else float("nan"),
                auroc_std=float(np.std(aurocs)) if aurocs else float("nan"),
                auroc_oof=float(auroc(oof[ok], y[ok])) if ok.any() else float("nan"),
                weights=dict(zip(SURVIVAL_FACTORS, [float(x) for x in w])))


# ----------------------------------------------------------------------------- Part B
def pair_quality(sets, img, *, seed=0) -> pd.DataFrame:
    """One row per photo pair: how well it matched, plus the comparability factors.

    ``match_frac`` (mutual matches / min(n_a, n_b)) is the outcome — the same quantity results.md
    #10 reports at a median of 0.50 for true pairs. ``same`` marks true pairs so Part C can use the
    same table.
    """
    rng = np.random.default_rng(seed)
    idxs = [i for i, s in enumerate(sets) if len(s.spots) >= 3]
    by_label: dict[str, list[int]] = {}
    for i in idxs:
        by_label.setdefault(sets[i].label, []).append(i)

    pairs = [(a, b) for v in by_label.values() for k, a in enumerate(v) for b in v[k + 1:]]
    n_false = min(len(pairs) * 3, 6000)
    seen = set()
    while len(seen) < n_false:
        a, b = int(rng.choice(idxs)), int(rng.choice(idxs))
        if sets[a].label == sets[b].label:
            continue
        seen.add((min(a, b), max(a, b)))
    allp = [(a, b, 1) for a, b in pairs] + [(a, b, 0) for a, b in seen]

    rows = []
    for a, b, same in allp:
        A, B = sets[a], sets[b]
        S = _norm(A.spots) @ _norm(B.spots).T
        abest = S.argmax(1); bbest = S.argmax(0)
        mutual = [i for i in range(len(S)) if bbest[abest[i]] == i
                  and float(S[i, abest[i]]) >= MATCH_THR]
        denom = max(min(len(A.spots), len(B.spots)), 1)

        geom = np.nan
        if len(mutual) >= 3:
            from scipy.spatial.distance import pdist                     # noqa: PLC0415
            from scipy.stats import spearmanr                            # noqa: PLC0415
            qp = np.asarray(A.centroids, float)[mutual]
            cp = np.asarray(B.centroids, float)[abest[mutual]]
            dq, dc = pdist(qp), pdist(cp)
            if dq.std() > 1e-9 and dc.std() > 1e-9:
                r = spearmanr(dq, dc).correlation
                geom = 0.0 if not np.isfinite(r) else float(r)

        rows.append(dict(
            sid_a=A.sid, sid_b=B.sid, label_a=A.label, label_b=B.label, same=same,
            match_frac=len(mutual) / denom, n_mutual=len(mutual), n_min=denom, geom=geom,
            d_curl=abs(_meta(img, A.sid, "curl_deg", 0) - _meta(img, B.sid, "curl_deg", 0)),
            d_aspect=abs(_meta(img, A.sid, "aspect_ratio", 0) - _meta(img, B.sid, "aspect_ratio", 0)),
            d_solidity=abs(_meta(img, A.sid, "solidity", 0) - _meta(img, B.sid, "solidity", 0)),
            blur_min=min(_meta(img, A.sid, "blur_quality", .5), _meta(img, B.sid, "blur_quality", .5)),
            quality_min=min(_meta(img, A.sid, "overall_quality", .5),
                            _meta(img, B.sid, "overall_quality", .5)),
            d_log_nspots=abs(np.log1p(len(A.spots)) - np.log1p(len(B.spots))),
        ))
    return pd.DataFrame(rows)


def comparability_effects(pairs: pd.DataFrame) -> pd.DataFrame:
    """For TRUE pairs only: does each factor predict how well the pair matched?

    Spearman correlation with ``match_frac``, plus the match rate in the best and worst tercile of
    each factor. Spearman because these relationships need not be linear and the factors are
    heavy-tailed; the terciles are there because a correlation coefficient alone hides whether an
    effect is large enough to act on.
    """
    from scipy.stats import spearmanr                                    # noqa: PLC0415
    t = pairs[pairs["same"] == 1]
    rows = []
    for f in COMPARABILITY:
        v = t[f].to_numpy(float); m = t["match_frac"].to_numpy(float)
        cnt = t["n_mutual"].to_numpy(float); nmin = t["n_min"].to_numpy(float)
        ok = np.isfinite(v) & np.isfinite(m)
        if ok.sum() < 30 or np.std(v[ok]) < 1e-12:
            rows.append(dict(factor=f, rho=np.nan, note="no variance"))
            continue
        rho = spearmanr(v[ok], m[ok]).correlation
        # `match_frac` divides by the spot count, and the spot count is itself driven by image
        # quality -- a sharper photo yields MORE spots, inflating the denominator and deflating the
        # fraction even when matching improved. So the absolute count and the spot count are
        # reported alongside: if a factor's sign flips between the fraction and the count, the
        # "effect" is the denominator moving, not matching degrading.
        rho_cnt = spearmanr(v[ok], cnt[ok]).correlation
        rho_n = spearmanr(v[ok], nmin[ok]).correlation
        q1, q3 = np.percentile(v[ok], [33.3, 66.7])
        lo = m[ok][v[ok] <= q1].mean(); hi = m[ok][v[ok] >= q3].mean()
        rows.append(dict(factor=f, rho=float(rho), rho_count=float(rho_cnt),
                         rho_nspots=float(rho_n), n=int(ok.sum()),
                         match_low_tercile=float(lo), match_high_tercile=float(hi),
                         delta=float(hi - lo),
                         confounded=bool(np.sign(rho) != np.sign(rho_cnt)
                                         and abs(rho_cnt) > 0.05)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Part C
# Which direction of a factor means "these two photos are MORE comparable". The difference factors
# (d_*) are comparable when SMALL; the min-quality factors are comparable when LARGE. Getting this
# backwards silently inverts the interaction's sign and turns "geometry works better on good photos"
# into its opposite, so it is declared once here rather than inferred at the call site.
HIGHER_IS_MORE_COMPARABLE = {"blur_min", "quality_min"}


def interaction_test(pairs: pd.DataFrame, factor: str = "d_curl") -> dict:
    """Is geometric consistency a BETTER true/false discriminator when the pair is comparable?

    Splits pairs at the median of ``factor`` and computes the AUROC of ``geom`` for same-vs-different
    within each half. A **positive** gap means the interaction the plan depends on is real and the
    product term is worth building. A gap near zero means an additive feature is all the data
    supports, and multiplying would add a parameter that fits noise. A **negative** gap is not a
    weaker version of a positive one -- it says geometry is more trustworthy on the WORSE photos,
    which contradicts the premise and needs explaining before anything is built on it.
    """
    p = pairs.dropna(subset=["geom", factor])
    if len(p) < 100:
        return dict(factor=factor, note="too few pairs with geometry")
    cut = float(p[factor].median())
    hi_better = factor in HIGHER_IS_MORE_COMPARABLE
    comparable = p[p[factor] >= cut] if hi_better else p[p[factor] <= cut]
    incomparable = p[p[factor] < cut] if hi_better else p[p[factor] > cut]
    out = dict(factor=factor, cut=cut, higher_is_comparable=bool(hi_better))
    for name, sub in (("comparable", comparable), ("incomparable", incomparable)):
        y = sub["same"].to_numpy(int)
        if len(np.unique(y)) < 2:
            out[f"auroc_{name}"] = float("nan"); out[f"n_{name}"] = int(len(sub)); continue
        out[f"auroc_{name}"] = float(auroc(sub["geom"].to_numpy(float), y))
        out[f"n_{name}"] = int(len(sub))
    a, b = out.get("auroc_comparable", np.nan), out.get("auroc_incomparable", np.nan)
    out["gap"] = float(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")
    return out


# ----------------------------------------------------------------------------- report
def main():
    import argparse
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-synth", action="store_true",
                    help="include Gemini-generated views (their curl/quality describe the "
                         "generator, not a camera -- off by default for that reason)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logger.info(f"dataset {d.dataset_name}   (synthetic views "
          f"{'INCLUDED' if args.include_synth else 'excluded'})")
    logger.info("loading spots, embeddings, per-image geometry + quality ...")
    sets, img, size, tpos, distinct = build_context(args.include_synth)
    logger.info(f" {len(sets)} photos, {sum(len(s.spots) for s in sets):,} spots")

    # ---------------------------------------------------------------- A
    logger.info("=" * 78)
    logger.info(" PART A — is a spot's disappearance predictable? (does 'absence' deserve a weight?)")
    logger.info("=" * 78)
    surv = spot_survival(sets, img, size, tpos, distinct, seed=args.seed)
    fit = fit_survival(surv, seed=args.seed)
    logger.info(f" {fit['n']:,} (spot, other-photo) observations over {fit['n_individuals']} individuals")
    logger.info(f" base survival rate: {fit['base_rate']:.3f}"
          f"   <- a spot found a match this often between photos of the SAME animal")
    logger.info(f" predictability:  AUROC {fit['auroc_mean']:.3f} ± {fit['auroc_std']:.3f}"
          f"   (out-of-fold {fit['auroc_oof']:.3f})")
    logger.info(" what predicts survival (standardized logistic weights):")
    for nm, w in sorted(fit["weights"].items(), key=lambda t: -abs(t[1])):
        logger.info(f"   {nm:14s} {w:+.3f}")
    if fit["auroc_mean"] >= 0.65:
        logger.info(" => Absence IS conditionable. Charging a pair for an unmatched spot can be scaled")
        logger.info("    by how surprising that disappearance is. Step 2 of the plan is supported.")
    else:
        logger.info(" => Absence is close to unpredictable. A missing spot cannot be weighted")
        logger.info("    intelligently from these factors, so charging for it will mostly add noise.")

    # ---------------------------------------------------------------- B
    logger.info("=" * 78)
    logger.info(" PART B — do curl / camera angle / quality predict how well a TRUE pair matches?")
    logger.info("=" * 78)
    pairs = pair_quality(sets, img, seed=args.seed)
    t = pairs[pairs["same"] == 1]
    logger.info(f" {len(t):,} true photo pairs, {int((pairs['same'] == 0).sum()):,} false pairs")
    logger.info(f" median match_frac on TRUE pairs: {t['match_frac'].median():.3f}"
          f"   (results.md #10 reports 0.50)")
    eff = comparability_effects(pairs)
    logger.info(f" {'factor':<14}{'rho(frac)':>10}{'rho(count)':>11}{'rho(nspots)':>12}"
          f"{'match@low':>11}{'match@high':>11}")
    logger.info(" " + "-" * 69)
    for r in eff.sort_values("rho", key=lambda s: -s.abs()).itertuples(index=False):
        if not np.isfinite(getattr(r, "rho", np.nan)):
            logger.info(f" {r.factor:<14}{'—':>10}   (no variance)"); continue
        flag = "  <- CONFOUNDED (sign flips vs count)" if r.confounded else ""
        logger.info(f" {r.factor:<14}{r.rho:>+10.3f}{r.rho_count:>+11.3f}{r.rho_nspots:>+12.3f}"
              f"{r.match_low_tercile:>11.3f}{r.match_high_tercile:>11.3f}{flag}")
    clean = eff[(eff["rho"].abs() >= 0.15) & (~eff["confounded"].fillna(False))]
    dirty = eff[eff["confounded"].fillna(False)]
    if len(clean):
        logger.info(f" => Real effects: {', '.join(clean['factor'])}. Worth conditioning on.")
    else:
        logger.info(" => No factor cleanly moves match quality. Conditioning will not buy anything.")
    if len(dirty):
        logger.info(f" => CONFOUNDED: {', '.join(dirty['factor'])} — the correlation with match_frac has")
        logger.info("    the opposite sign to the correlation with the raw match COUNT, which is what a")
        logger.info("    denominator artefact looks like: better photos yield more spots, inflating")
        logger.info("    min(n) and deflating the fraction even when matching improved. Do not treat")
        logger.info("    these as degradation without a spot-count-adjusted outcome.")

    # ---------------------------------------------------------------- C
    logger.info("=" * 78)
    logger.info(" PART C — is geometric evidence MORE reliable when the two photos are comparable?")
    logger.info("         (the test for whether interaction terms are justified at all)")
    logger.info("=" * 78)
    inter = []
    for f in ("d_curl", "d_aspect", "blur_min", "quality_min"):
        r = interaction_test(pairs, f)
        inter.append(r)
        if "auroc_comparable" not in r:
            logger.info(f" {f:<12} {r.get('note')}"); continue
        logger.info(f" {f:<12} split at {r['cut']:.3f} | geom AUROC  comparable "
              f"{r['auroc_comparable']:.3f} (n={r['n_comparable']})  vs  incomparable "
              f"{r['auroc_incomparable']:.3f} (n={r['n_incomparable']})   gap {r['gap']:+.3f}")
    # Every factor is now oriented so "comparable" means MORE comparable, which makes the sign
    # meaningful: only a POSITIVE gap supports the plan. Judged per factor, because "build the
    # interaction" is a decision about one term at a time, not about the whole idea.
    supported = [r["factor"] for r in inter if r.get("gap", np.nan) >= 0.05]
    contra = [r["factor"] for r in inter if r.get("gap", np.nan) <= -0.05]
    flat = [r["factor"] for r in inter
            if np.isfinite(r.get("gap", np.nan)) and abs(r["gap"]) < 0.05]
    if supported:
        logger.info(f" => Interaction SUPPORTED for: {', '.join(supported)}.")
        logger.info("    Build geometry x comparability products for these factors only.")
    if flat:
        logger.info(f" => No interaction for: {', '.join(flat)}. Add as plain columns; a product term")
        logger.info("    here would be a parameter fitted to noise.")
    if contra:
        logger.info(f" => BACKWARDS for: {', '.join(contra)} — geometry looks MORE reliable on the worse")
        logger.info("    photos, which contradicts the premise. Do not build on this until it is")
        logger.info("    explained; the likely cause is a confound rather than a real effect.")
    geoms = [r.get("auroc_comparable", np.nan) for r in inter]
    if geoms and np.nanmax(geoms) < 0.65:
        logger.warning(f" NOTE: geometry's own discriminative power is weak everywhere "
              f"(best AUROC {np.nanmax(geoms):.3f}).")
        logger.warning(" `geom` here is only the rank-correlation of pairwise distances among matched spots —")
        logger.warning(" the crudest constellation measure available. Read this as 'the current geometry")
        logger.warning(" feature is weak', NOT as 'constellations are weak'; triplet/angle invariants are")
        logger.warning(" the thing actually proposed and they are not measured here.")

    # ---------------------------------------------------------------- write
    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "feasibility"
    outdir.mkdir(parents=True, exist_ok=True)
    surv.to_csv(outdir / "spot_survival.csv", index=False)
    pairs.to_csv(outdir / "pair_quality.csv", index=False)
    eff.to_csv(outdir / "comparability_effects.csv", index=False)
    (outdir / "feasibility.json").write_text(
        json.dumps(dict(survival=fit, interaction=inter,
                        median_match_frac_true=float(t["match_frac"].median()),
                        n_true_pairs=int(len(t))), indent=2, default=float), encoding="utf-8")

    md = ["# Feasibility of the constellation / comparability plan", "",
          f"- dataset `{d.dataset_name}` · synthetic views "
          f"{'included' if args.include_synth else 'excluded'} · no training, no new labels", "",
          "## A. Is a spot's disappearance predictable?", "",
          f"- base survival rate **{fit['base_rate']:.3f}** — how often a spot finds a match "
          f"between photos of the SAME animal", f"- held-out AUROC **{fit['auroc_mean']:.3f} ± "
          f"{fit['auroc_std']:.3f}** over {fit['n_individuals']} individuals "
          f"({fit['n']:,} observations)", "",
          "| factor | weight |", "|---|---|",
          *[f"| `{nm}` | {w:+.3f} |" for nm, w in
            sorted(fit["weights"].items(), key=lambda t: -abs(t[1]))], "",
          "## B. Do comparability factors predict match quality on TRUE pairs?", "",
          f"- median `match_frac` on true pairs: **{t['match_frac'].median():.3f}**", "",
          "| factor | Spearman rho | match@low tercile | match@high | delta |",
          "|---|---|---|---|---|",
          *[f"| `{r.factor}` | {r.rho:+.3f} | {r.match_low_tercile:.3f} | "
            f"{r.match_high_tercile:.3f} | {r.delta:+.3f} |"
            for r in eff.itertuples(index=False) if np.isfinite(getattr(r, "rho", np.nan))], "",
          "## C. Does comparability change how much geometry can be trusted?", "",
          "| factor | geom AUROC (comparable) | geom AUROC (incomparable) | gap |",
          "|---|---|---|---|",
          *[f"| `{r['factor']}` | {r['auroc_comparable']:.3f} | {r['auroc_incomparable']:.3f} | "
            f"{r['gap']:+.3f} |" for r in inter if "auroc_comparable" in r], ""]
    (outdir / "RESULTS_feasibility.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info(f"wrote {outdir / 'RESULTS_feasibility.md'}")
    logger.info(f"      {outdir / 'spot_survival.csv'}  ({len(surv):,} rows, for your own digging)")


if __name__ == "__main__":
    main()
