"""Look at the 76%: photo pairs of ONE animal where the spots fail to find each other.

The measurements say a spot has only a ~24% chance of being found again in another photo of the
same salamander, that conditions and size barely change it (0.24 -> 0.30 under ideal conditions,
0.34 for the biggest spots), and that it is not an assignment problem — 71.5% of spots have *no
similar candidate at all* in the other photo. Those are aggregates, and aggregates cannot tell you
WHY. This renders the actual pairs so the cause is visible.

Cases are chosen to be different from each other on purpose, because one failure mode would
otherwise dominate the sample and look like the whole story:

    ideal_fail    good photos, similar pose and angle, most spots still unmatched. The puzzling
                  case, and the one that decides whether segmentation is to blame — there is no
                  occlusion or viewpoint excuse available.
    big_unmatched large, conspicuous spots with no counterpart. If these are real markings the
                  segmenter should never miss them; if they are merged blobs, they are artifacts.
    count_split   the two photos disagree sharply on how MANY spots the animal has — the
                  fragmentation signature (one spot in A, two in B).
    hard_pose     curled or photographed from a different angle: the failures that DO have an
                  excuse, for contrast.
    success       a pair that matched well. Without it there is no baseline for what "good" looks
                  like and every image looks like a failure.

Each panel draws both animals with body outline (grey) and spot contours (orange), red arcs for
the correspondences the matcher DID find, and magenta for spots left unexplained — the half of the
evidence the arcs cannot show.

    pixi run unmatched-examples                 # 2 of each case -> artifacts/.../unmatched/
    pixi run unmatched-examples --per-case 4
    pixi run unmatched-examples --case ideal_fail --per-case 6
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval", "viz"):
    p = str(_ST / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import data as d                                         # noqa: E402
import embeddings as E                                   # noqa: E402
from aggregator import _norm                             # noqa: E402

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

MATCH_THR = 0.4
CASES = ["ideal_fail", "big_unmatched", "count_split", "hard_pose", "success"]


def pair_table(sets, img: pd.DataFrame, size: dict) -> pd.DataFrame:
    """One row per ordered same-animal photo pair: survival plus the facts used to pick cases."""
    by: dict[str, list[int]] = {}
    for i, s in enumerate(sets):
        if len(s.spots) >= 4 and not s.is_synth:
            by.setdefault(s.label, []).append(i)

    def meta(sid, col, dflt=np.nan):
        try:
            v = img.at[sid, col]
        except KeyError:
            return dflt
        return float(v) if v is not None and np.isfinite(v) else dflt

    rows = []
    for lbl, idxs in by.items():
        if len(idxs) < 2:
            continue
        for k, ai in enumerate(idxs):
            for bi in idxs[k + 1:]:
                A, B = sets[ai], sets[bi]
                S = _norm(A.spots) @ _norm(B.spots).T
                abest = S.argmax(1); bbest = S.argmax(0)
                surv = np.array([bbest[abest[i]] == i and float(S[i, abest[i]]) >= MATCH_THR
                                 for i in range(len(S))])
                sz = np.array([size.get((A.sid, int(j)), 0.5) for j in A.spot_ids], float)
                big_un = int(((~surv) & (sz >= 0.8)).sum())
                rows.append(dict(
                    label=lbl, a=ai, b=bi, sid_a=A.sid, sid_b=B.sid,
                    n_a=len(A.spots), n_b=len(B.spots),
                    survival=float(surv.mean()), n_surv=int(surv.sum()),
                    big_unmatched=big_un,
                    count_ratio=min(len(A.spots), len(B.spots)) / max(len(A.spots), len(B.spots)),
                    d_curl=abs(meta(A.sid, "curl_deg", 0) - meta(B.sid, "curl_deg", 0)),
                    d_aspect=abs(meta(A.sid, "aspect_ratio", 0) - meta(B.sid, "aspect_ratio", 0)),
                    q_min=min(meta(A.sid, "overall_quality", 0), meta(B.sid, "overall_quality", 0)),
                ))
    return pd.DataFrame(rows)


def pick(pairs: pd.DataFrame, case: str, n: int) -> pd.DataFrame:
    """Select ``n`` pairs illustrating ``case``. Deliberately not random: a random sample of
    failures is dominated by the commonest kind and hides the rest."""
    p = pairs
    if case == "ideal_fail":
        q = p[(p.q_min >= p.q_min.quantile(0.6)) & (p.d_curl <= 3.0) & (p.d_aspect <= 1.5)
              & (p.n_a >= 8) & (p.n_b >= 8)]
        return q.nsmallest(n, "survival")
    if case == "big_unmatched":
        return p[p.survival < 0.5].nlargest(n, "big_unmatched")
    if case == "count_split":
        return p[(p.n_a >= 8) & (p.n_b >= 8)].nsmallest(n, "count_ratio")
    if case == "hard_pose":
        return p[p.survival < 0.4].nlargest(n, "d_curl")
    if case == "success":
        return p[(p.n_a >= 8) & (p.n_b >= 8)].nlargest(n, "survival")
    raise SystemExit(f"unknown case {case!r}; expected one of {CASES}")


def render(spots_df, row, size: dict, outdir: Path, case: str) -> Path:
    """Draw one pair, arcs for what matched and magenta for what did not."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s1, s2, S, _ = E.match_spots(spots_df, row.sid_a, row.sid_b, method="mutual")
    abest = S.argmax(1); bbest = S.argmax(0)
    pairs, un_a = [], {}
    for i in range(len(S)):
        j = int(abest[i]); sim = float(S[i, j])
        if bbest[j] == i and sim >= MATCH_THR:
            pairs.append((i, j, sim))
    matched_a = {i for i, _, _ in pairs}
    matched_b = {j for _, j, _ in pairs}
    # Mark intensity = the spot's size percentile, so the eye is drawn to the unmatched spots that
    # SHOULD have been easy -- a big conspicuous blotch with no counterpart is the interesting
    # failure; a tiny one is expected (survival by size decile runs 0.19 -> 0.38).
    for i, r in s1.iterrows():
        if i not in matched_a:
            un_a[int(r["spot_id"])] = float(size.get((row.sid_a, int(r["spot_id"])), 0.5))
    un_b = {int(r["spot_id"]): float(size.get((row.sid_b, int(r["spot_id"])), 0.5))
            for i, r in s2.iterrows() if i not in matched_b}

    fig, ax = plt.subplots(figsize=(13, 7))
    E.visualize_spot_matches(spots_df, row.sid_a, row.sid_b, pairs=pairs,
                             mark={row.sid_a: un_a, row.sid_b: un_b}, mark_thr=0.5, ax=ax)
    ax.set_title(
        f"[{case}]  {row.sid_a}  vs  {row.sid_b}   (same animal: {row.label})\n"
        f"{row.n_surv}/{row.n_a} spots matched = {row.survival:.0%}   |   "
        f"spots {row.n_a} vs {row.n_b}   |   big spots unmatched: {row.big_unmatched}   |   "
        f"Δcurl {row.d_curl:.1f}°  Δangle {row.d_aspect:.2f}  quality {row.q_min:.2f}",
        fontsize=10)
    out = outdir / f"{case}__{row.sid_a}__{row.sid_b}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def main():
    import argparse
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-case", type=int, default=2)
    ap.add_argument("--case", default=None, choices=CASES)
    args = ap.parse_args()

    logger.info("loading spots + embeddings + per-image geometry ...")
    import duckdb
    from feasibility import load_image_meta, load_spot_meta                  # noqa: PLC0415
    sets = [s for s in d.get_image_sets(d.get_spot_embeddings()) if not s.is_synth]
    img = load_image_meta().set_index("sid")
    sm_ = load_spot_meta()
    size = {(r.sid, int(r.spot_id)): (float(r.size_pct) if np.isfinite(r.size_pct) else 0.5)
            for r in sm_.itertuples(index=False)}
    spots_df = E.filter_small_spots(E.get_spots(), min_pct=10)

    pairs = pair_table(sets, img, size)
    logger.info(f" {len(pairs):,} same-animal photo pairs over {pairs['label'].nunique()} individuals")
    logger.info(f" median survival across pairs: {pairs['survival'].median():.3f}")

    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "unmatched"
    outdir.mkdir(parents=True, exist_ok=True)
    cases = [args.case] if args.case else CASES
    written = []
    for case in cases:
        sel = pick(pairs, case, args.per_case)
        logger.info(f"--- {case} ({len(sel)}) ---")
        for row in sel.itertuples(index=False):
            try:
                out = render(spots_df, row, size, outdir, case)
            except Exception as exc:
                logger.error(f"  !! {row.sid_a} vs {row.sid_b}: {type(exc).__name__}: {exc}")
                continue
            written.append(out)
            logger.info(f"  {row.sid_a:>12} vs {row.sid_b:<12} survival {row.survival:.0%}  "
                  f"spots {row.n_a}/{row.n_b}  bigUnmatched {row.big_unmatched}  "
                  f"Δcurl {row.d_curl:4.1f}  q {row.q_min:.2f}  -> {out.name}")

    pairs.to_csv(outdir / "pair_index.csv", index=False)
    logger.info(f"wrote {len(written)} panels -> {outdir}")
    logger.info("  grey = body outline · orange = spot contours · red arcs = matched")
    logger.info("  magenta = UNMATCHED (heavier = larger spot, i.e. the ones that should have been easy)")
    logger.info(f"  full pair index (sortable, for finding your own examples): {outdir/'pair_index.csv'}")


if __name__ == "__main__":
    main()
