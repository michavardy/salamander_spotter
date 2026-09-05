"""Two mechanical data faults, counted across the whole dataset.

Reviewing failed photo pairs by eye turned up two causes that are neither modelling problems nor
segmentation-quality problems, but broken inputs — and both are cheap to find and cheap to fix:

**1. More than one salamander in the frame.** ``ca_14_5`` is two animals overlapping, so its body
   mask fuses them and its 49 "spots" belong to two different individuals. That poisons matching
   *and* the ground truth: some of those spots are labelled with the wrong animal's identity, so
   every metric computed against them is quietly wrong. Calibrated on that confirmed example, the
   signature is a body mask that is strongly non-convex (``solidity`` at the 0th percentile of the
   dataset) carrying an unusually high spot count (100th percentile).

**2. A body axis running the wrong way.** ``sj_1_1`` vs ``sj_1_2`` gains matches 2 -> 5 when one
   image's head/tail is reversed — the reviewer had already flagged it in the app as "its
   upsidedown?". A reversed axis puts every spot's ``axis_t`` at ``1 - t``, so the position half of
   the embedding sends corresponding spots to opposite ends of the body while the shape half still
   points at the right partner; the two halves disagree and the pair scores as unrelated.

Both outputs are **ranked candidate lists for a human to confirm, not classifiers**. There is no
ground truth for either fault, so nothing here is a measurement of how many exist — it is a
worklist ordered by suspicion, with the confirmed example's position shown so the ranking can be
sanity-checked.

    pixi run data-hygiene              # both checks
    pixi run data-hygiene --top 40
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval"):
    p = str(_ST / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import data as d                                     # noqa: E402
import embeddings as E                               # noqa: E402
from aggregator import _norm                         # noqa: E402

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

MATCH_THR = 0.4
KNOWN_MULTI = "ca_14_5"            # confirmed by eye: two overlapping animals
KNOWN_FLIP = "sj_1_1"              # reviewer-flagged "its upsidedown?"


# ------------------------------------------------------------------ 1. multiple animals per frame
def multi_animal_candidates(db_path=None) -> pd.DataFrame:
    """Rank photos by how much they look like the confirmed two-animal case.

    Score = percentile(n_spots) + (1 - percentile(solidity)), i.e. "unusually many spots inside an
    unusually non-convex body". Both halves are needed: a genuinely spotty animal has a high count
    with normal solidity, and a curled animal has low solidity with a normal count. Only the
    combination is the fused-mask signature.
    """
    import duckdb
    con = duckdb.connect(str(db_path or d.DB_PATH), read_only=True)
    try:
        df = con.execute(
            "SELECT salamander_id AS sid, n_spots, solidity, aspect_ratio, body_area_frac, "
            "curl_deg, overall_quality FROM image_quality"
        ).df()
    finally:
        con.close()
    df = df[~df["sid"].str.rsplit("_", n=1).str[-1].str.startswith("g")].reset_index(drop=True)

    def pct(s):
        return s.rank(pct=True)

    df["p_spots"] = pct(df["n_spots"])
    df["p_solidity"] = pct(df["solidity"])
    df["suspicion"] = df["p_spots"] + (1.0 - df["p_solidity"])
    df["individual"] = df["sid"].str.rsplit("_", n=1).str[0]
    # A photo whose spot count dwarfs its own siblings is more suspicious than one that is simply
    # from a spotty animal -- this is the within-individual version of the same signal.
    med = df.groupby("individual")["n_spots"].transform("median")
    df["spots_vs_siblings"] = df["n_spots"] / med.clip(lower=1)
    return df.sort_values("suspicion", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ 2. reversed body axis
def _pos_block(sid, spot_ids, meta, halfwidth, flip_t=False):
    """Rebuild the 25-dim position block, optionally with head/tail reversed."""
    t, u, side = [], [], []
    for i in spot_ids:
        a, off, sd = meta.get((sid, int(i)), (0.5, 0.0, "right"))
        a = 0.5 if not np.isfinite(a) else float(a)
        off = 0.0 if not np.isfinite(off) else float(off)
        t.append(1.0 - a if flip_t else a)
        u.append(float(np.clip(off / halfwidth if halfwidth > 0 else 0.0, -3, 3)))
        side.append(1.0 if sd == "left" else -1.0)
    P = np.concatenate([E._sinusoidal(np.array(t), 6), E._sinusoidal(np.array(u), 6),
                        np.array(side).reshape(-1, 1)], axis=1)
    return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)


def axis_flip_candidates(db_path=None) -> pd.DataFrame:
    """For each photo: does REVERSING its head/tail improve matching against its own siblings?

    Tested per image rather than per pair, because a reversed axis is a property of one photo — if
    image X is backwards, every pair containing X improves when X is flipped, and a pair-level view
    would blame both photos equally.
    """
    import duckdb
    con = duckdb.connect(str(db_path or d.DB_PATH), read_only=True)
    try:
        sp = con.execute("SELECT salamander_id AS sid, spot_id, axis_t, axis_offset, axis_side "
                         "FROM spots").df()
        hw = con.execute("SELECT salamander_id AS sid, avg_width_px FROM image_quality").df()
    finally:
        con.close()
    meta = {(r.sid, int(r.spot_id)): (r.axis_t, r.axis_offset, r.axis_side)
            for r in sp.itertuples(index=False)}
    half = {r.sid: (float(r.avg_width_px) / 2.0 if np.isfinite(r.avg_width_px) else 1.0)
            for r in hw.itertuples(index=False)}

    sets = {s.sid: s for s in d.get_image_sets(d.get_spot_embeddings()) if not s.is_synth}
    by: dict[str, list[str]] = {}
    for sid, s in sets.items():
        if len(s.spots) >= 4:
            by.setdefault(s.label, []).append(sid)

    def norm_rows(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    rows = []
    for lbl, sids in by.items():
        if len(sids) < 2:
            continue
        shape = {s: norm_rows(_norm(sets[s].spots)[:, :37]) for s in sids}
        pos = {s: _pos_block(s, sets[s].spot_ids, meta, half.get(s, 1.0)) for s in sids}
        posf = {s: _pos_block(s, sets[s].spot_ids, meta, half.get(s, 1.0), flip_t=True)
                for s in sids}
        for a in sids:
            base = flip = 0
            for b in sids:
                if a == b:
                    continue
                for P_a in (pos, posf):
                    S = (shape[a] @ shape[b].T + P_a[a] @ pos[b].T) / 2.0
                    ab, bb = S.argmax(1), S.argmax(0)
                    n = sum(1 for i in range(len(S))
                            if bb[ab[i]] == i and float(S[i, ab[i]]) >= MATCH_THR)
                    if P_a is pos:
                        base += n
                    else:
                        flip += n
            nsib = len(sids) - 1
            rows.append(dict(sid=a, individual=lbl, n_siblings=nsib,
                             matches_asis=base, matches_flipped=flip, gain=flip - base,
                             matches_per_sibling=base / max(nsib, 1),
                             n_spots=len(sets[a].spots)))
    df = pd.DataFrame(rows)
    return df.sort_values("gain", ascending=False).reset_index(drop=True)


# A reversed axis is only a plausible EXPLANATION for a photo that matches almost nothing. Tested
# across all 836 photos, flipping helps 19% and hurts 63% -- it looks like a dead end. Restricted to
# photos averaging under half a match per sibling, it helps 67% with a positive mean gain. The
# restriction is what makes the test meaningful: on a photo that already matches well, flipping can
# only destroy a working alignment, and those cases swamp the signal.
LOW_MATCH_PER_SIBLING = 0.5


def holdout_flip_check(db_path=None, min_siblings: int = 4) -> dict:
    """Does "flipping helps" survive being decided on one set of siblings and scored on another?

    Necessary because :func:`axis_flip_candidates` selects on the very quantity it reports: pick the
    photos where flipping raised the match count, and their match count is higher by construction.
    Here the decision is made on half an image's siblings and measured on the other half, so a gain
    can only appear if the axis really is reversed rather than the noise happening to fall that way.
    A positive rate near 0.5 with a mean gain near 0 means noise; well above 0.5 means a real fault.
    """
    import duckdb
    con = duckdb.connect(str(db_path or d.DB_PATH), read_only=True)
    try:
        sp = con.execute("SELECT salamander_id AS sid, spot_id, axis_t, axis_offset, axis_side "
                         "FROM spots").df()
        hw = con.execute("SELECT salamander_id AS sid, avg_width_px FROM image_quality").df()
    finally:
        con.close()
    meta = {(r.sid, int(r.spot_id)): (r.axis_t, r.axis_offset, r.axis_side)
            for r in sp.itertuples(index=False)}
    half = {r.sid: (float(r.avg_width_px) / 2.0 if np.isfinite(r.avg_width_px) else 1.0)
            for r in hw.itertuples(index=False)}
    sets = {s.sid: s for s in d.get_image_sets(d.get_spot_embeddings()) if not s.is_synth}
    by: dict[str, list[str]] = {}
    for sid, s in sets.items():
        if len(s.spots) >= 4:
            by.setdefault(s.label, []).append(sid)

    def nr(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    def cnt(a, b, flip):
        S = (nr(_norm(sets[a].spots)[:, :37]) @ nr(_norm(sets[b].spots)[:, :37]).T
             + _pos_block(a, sets[a].spot_ids, meta, half.get(a, 1.0), flip)
             @ _pos_block(b, sets[b].spot_ids, meta, half.get(b, 1.0)).T) / 2.0
        ab, bb = S.argmax(1), S.argmax(0)
        return sum(1 for i in range(len(S)) if bb[ab[i]] == i and float(S[i, ab[i]]) >= MATCH_THR)

    dec, hold = [], []
    for lbl, sids in by.items():
        if len(sids) < min_siblings:
            continue
        for a in sids:
            sib = [x for x in sids if x != a]
            A, B = sib[::2], sib[1::2]
            if not A or not B:
                continue
            dec.append(sum(cnt(a, b, True) - cnt(a, b, False) for b in A))
            hold.append(sum(cnt(a, b, True) - cnt(a, b, False) for b in B))
    if not dec:
        return {}
    dec, hold = np.array(dec), np.array(hold)
    sel = dec > 0
    return dict(n_images=int(len(dec)), n_selected=int(sel.sum()),
                pos_rate=float((hold[sel] > 0).mean()) if sel.any() else float("nan"),
                mean_gain=float(hold[sel].mean()) if sel.any() else float("nan"))


# ------------------------------------------------------------------ report
def main():
    import argparse
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "hygiene"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 84)
    logger.info(" 1. MORE THAN ONE SALAMANDER IN THE FRAME  (ranked suspicion, needs your eye to confirm)")
    logger.info("=" * 84)
    multi = multi_animal_candidates()
    rank = multi.index[multi["sid"] == KNOWN_MULTI]
    logger.info(f" {len(multi)} real photos scored. The confirmed case {KNOWN_MULTI} ranks "
          f"#{int(rank[0]) + 1 if len(rank) else '?'} of {len(multi)} — "
          f"{'the ranking works' if len(rank) and rank[0] < 20 else 'the ranking is NOT validated'}")
    logger.info(f" {'rank':>5}{'photo':>12}{'spots':>7}{'solidity':>10}{'vs siblings':>13}{'quality':>9}")
    for i, r in multi.head(args.top).iterrows():
        mark = "  <- confirmed two animals" if r["sid"] == KNOWN_MULTI else ""
        logger.info(f" {i + 1:>5}{r['sid']:>12}{int(r['n_spots']):>7}{r['solidity']:>10.3f}"
              f"{r['spots_vs_siblings']:>13.2f}{r['overall_quality']:>9.3f}{mark}")
    multi.to_csv(outdir / "multi_animal_candidates.csv", index=False)

    logger.info("=" * 84)
    logger.info(" 2. BODY AXIS RUNNING THE WRONG WAY  (does reversing head/tail recover matches?)")
    logger.info("=" * 84)
    flips = axis_flip_candidates()
    helped = flips[flips["gain"] > 0]
    logger.info(f" {len(flips)} photos with at least one sibling to test against")
    logger.info(f" reversing head/tail HELPS on {len(helped)} of them ({len(helped) / max(len(flips), 1):.1%})"
          f", hurts on {int((flips['gain'] < 0).sum())}, no change on "
          f"{int((flips['gain'] == 0).sum())}")
    logger.warning(f" NOTE: 'helps' is selected on the same quantity it is scored by, so the apparent total"
          f" gain\n (+{int(helped['gain'].sum()):,} matches) is not claimable. See the held-out"
          f" check below, which is.")
    hv = holdout_flip_check()
    if hv:
        logger.info(" HELD-OUT validation (images with >=4 siblings; decide the flip on half of them,")
        logger.info(" measure on the other half):")
        logger.info(f"   {hv['n_images']} images testable · flip chosen for {hv['n_selected']}")
        logger.info(f"   of those, held-out gain positive: {hv['pos_rate']:.2f}   "
              f"mean held-out gain {hv['mean_gain']:+.2f}")
        if hv["pos_rate"] < 0.5 or hv["mean_gain"] <= 0:
            logger.info("   => the flips DO NOT generalise. Reversed axes are not a systematic fault here;")
            logger.info("      the per-pair improvements are noise selected after the fact. Do not spend")
            logger.info("      time on axis correction on the strength of this.")
        else:
            logger.info("   => the flips generalise: reversed axes are a real, mechanical fault worth fixing.")
    kr = flips.index[flips["sid"] == KNOWN_FLIP]
    if len(kr):
        k = flips.loc[kr[0]]
        logger.info(f" reviewer-flagged {KNOWN_FLIP}: rank #{int(kr[0]) + 1}, "
              f"{int(k['matches_asis'])} -> {int(k['matches_flipped'])} matches (gain {int(k['gain'])})")
    low = flips[flips["matches_per_sibling"] <= LOW_MATCH_PER_SIBLING]
    logger.info(f" RESTRICTED to photos matching almost nothing (<= {LOW_MATCH_PER_SIBLING} matches per"
          f" sibling) —\n a reversed axis is only a plausible explanation for those:")
    if len(low):
        logger.info(f"   {len(low)} photos · flipping helps {(low['gain'] > 0).mean():.0%} of them "
              f"(vs {(flips['gain'] > 0).mean():.0%} across all photos) · "
              f"mean gain {low['gain'].mean():+.2f}")
        logger.info("   But read the size of the win: these go from ~0 matches to 1-2, which is still a")
        logger.info("   failed pair. And most have exactly ONE sibling, where flipping EITHER photo fixes")
        logger.info("   the relative alignment equally — so the data cannot say which of the two is the")
        logger.info("   reversed one. Treat this as a short review list, not a correction pipeline.")
    logger.info(f" {'rank':>5}{'photo':>12}{'as-is':>8}{'flipped':>9}{'gain':>7}{'siblings':>10}")
    for i, r in low.nlargest(args.top, "gain").iterrows():
        mark = "  <- reviewer flagged" if r["sid"] == KNOWN_FLIP else ""
        logger.info(f" {i + 1:>5}{r['sid']:>12}{int(r['matches_asis']):>8}{int(r['matches_flipped']):>9}"
              f"{int(r['gain']):>7}{int(r['n_siblings']):>10}{mark}")
    flips.to_csv(outdir / "axis_flip_candidates.csv", index=False)

    logger.info(f"wrote {outdir/'multi_animal_candidates.csv'}")
    logger.info(f"      {outdir/'axis_flip_candidates.csv'}")
    logger.info(" Neither list is a measurement of how many faults exist — there is no ground truth for")
    logger.info(" either. They are worklists ordered by suspicion. Confirm the top of each by eye, and the")
    logger.info(" hit rate you find in the first 20 tells you whether the rest is worth working through.")


if __name__ == "__main__":
    main()
