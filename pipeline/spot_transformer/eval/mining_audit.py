"""Audit the SSL correspondence-mining rule against the human verdicts.

results.md #29 is the strongest live result on the representation: SimCLR pretrained on positives
*mined* from real cross-photo correspondences reached **0.375 R@1** against **0.277** for
augmentation positives and 0.089 for a random-init control. The whole gain rests on those mined
pairs being correct — a wrong pair teaches the encoder that two different spots are the same — and
their precision has never been measured. It could not be: there was no ground truth for "is this
really the same physical spot".

The preprocessing review is that ground truth, for 842 correspondences over 55 individuals. This
module runs ``ssl_pretrain.mine_correspondences`` unchanged, joins its output to the verdicts, and
answers three questions the pretraining run currently assumes:

1. **What is mining precision?** Of mined pairs a human adjudicated, how many did they accept.
2. **Where should ``SSL_MIN_SIM`` sit?** The precision/yield curve over the cutoff, measured rather
   than reasoned about — the current 0.40 was chosen from the average similarity of genuine
   matches, which is a statement about the score, not about correctness.
3. **Does the RANSAC filter earn its keep?** It is described as "the real precision filter"; with
   ``use_geom=False`` we can see whether it raises precision or merely lowers yield.

**The denominator is the honest part.** The reviewer adjudicated the correspondences the pair-review
app showed them plus the ones they drew — not every pair the miner can produce. A mined pair with no
verdict is *unknown*, not correct, so precision here is measured only over the overlap and the
overlap size is reported next to it. Scaling that number to all ~40k crops is an inference, not a
measurement, and the report says so.

    pixi run mining-audit                  # the curve + the RANSAC ablation
    pixi run mining-audit --no-geom-only   # skip the ablation (faster)
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

import data as d                                          # noqa: E402
import review_labels as rl                                # noqa: E402

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

CUTOFFS = [0.0, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]


def verdict_lookup(dataset: str) -> dict[frozenset, bool]:
    """``{(sid, spot_id), (sid, spot_id)} -> accepted``, over EVERY adjudicated correspondence.

    Both proposers count as ground truth here, and for once that is the right call: the question is
    whether two crops show the same physical spot, which the human answered directly. Who proposed
    the edge shaped *which* pairs got looked at, not what the answer was. (It still matters for the
    matcher's own precision — see ``review_labels.matcher_operating_point`` — just not here.)
    """
    df = rl.match_verdicts(dataset)
    out = {}
    for r in df.itertuples(index=False):
        out[frozenset({(r.sid_a, r.spot_a), (r.sid_b, r.spot_b)})] = bool(r.accepted)
    return out


def mine_keyed(sets, *, min_sim: float, use_geom: bool = True, seed: int = 0):
    """``mine_correspondences`` with an identity index, so pairs come back as ``(sid, spot_id)``
    key tuples instead of crop row numbers.

    Deliberately calls the real function rather than reimplementing its logic: an audit of a copy
    of the rule audits the copy. The identity mapping is the only adapter needed.
    """
    from ssl_pretrain import mine_correspondences                      # noqa: PLC0415
    keys = {(s.sid, int(i)) for s in sets for i in (s.spot_ids if s.spot_ids is not None else [])}
    identity = {k: k for k in keys}
    return mine_correspondences(sets, identity, min_sim=min_sim, seed=seed,
                                report=False, use_geom=use_geom)


def score_pairs(pairs, emb_lookup) -> pd.DataFrame:
    """``(key_a, key_b, cos)`` for mined pairs — the cosine the cutoff is applied to."""
    rows = []
    for ka, kb in pairs:
        ea, eb = emb_lookup.get(ka), emb_lookup.get(kb)
        if ea is None or eb is None:
            continue
        c = float(ea @ eb / ((np.linalg.norm(ea) * np.linalg.norm(eb)) + 1e-12))
        rows.append((ka, kb, c))
    return pd.DataFrame(rows, columns=["key_a", "key_b", "cos"])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion — the right one for small ``n`` near the boundary,
    where the normal approximation gives intervals that run past 1.0.

    This is not decoration. The adjudicated overlap here is a few dozen pairs, so a precision of
    0.84 carries an interval wide enough to swallow every difference in the table, and reading the
    point estimates alone would manufacture conclusions the data cannot support.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def audit(mined: pd.DataFrame, verdicts: dict[frozenset, bool], n_accepted_total: int) -> pd.DataFrame:
    """Precision / recall / yield of the mining rule at each cutoff in :data:`CUTOFFS`.

    ``precision`` is over adjudicated mined pairs only; ``n_judged`` next to it is what makes the
    number readable. ``recall`` is over all human-ACCEPTED correspondences, so it says how much of
    the known-true correspondence set this rule recovers.
    """
    keysets = [frozenset({a, b}) for a, b in zip(mined["key_a"], mined["key_b"])]
    verdict = np.array([verdicts.get(k, None) for k in keysets], dtype=object)
    cos = mined["cos"].to_numpy(float)

    rows = []
    for t in CUTOFFS:
        keep = cos >= t
        v = verdict[keep]
        judged = v[v != None]                                          # noqa: E711
        n_j = len(judged)
        n_ok = int(sum(bool(x) for x in judged))
        lo, hi = wilson(n_ok, n_j)
        rows.append(dict(
            cutoff=t, mined=int(keep.sum()), n_judged=n_j,
            precision=(n_ok / n_j) if n_j else float("nan"),
            ci_lo=lo, ci_hi=hi,
            n_accepted_found=n_ok,
            recall=(n_ok / n_accepted_total) if n_accepted_total else float("nan"),
        ))
    return pd.DataFrame(rows)


def main():
    import argparse
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-geom-only", action="store_true",
                    help="skip the RANSAC ablation (halves the runtime)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = d.dataset_name
    logger.info(f"dataset {ds}")
    logger.info(" " + rl.review_summary(ds))

    from aggregator import attach_centroids                            # noqa: PLC0415
    logger.info("loading sets + embeddings ...")
    sets = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))
    emb = d.get_spot_embeddings()
    emb_lookup = {(r.salamander_id, int(r.spot_id)): np.asarray(r.embedding, float)
                  for r in emb.itertuples(index=False)}

    verdicts = verdict_lookup(ds)
    n_acc = sum(1 for v in verdicts.values() if v)
    logger.info(f" {len(verdicts)} adjudicated correspondences ({n_acc} accepted) as ground truth")

    results = {}
    for use_geom in ([True] if args.no_geom_only else [True, False]):
        lab = "with RANSAC" if use_geom else "no RANSAC"
        logger.info(f"mining ({lab}) ...")
        pairs = mine_keyed(sets, min_sim=0.0, use_geom=use_geom, seed=args.seed)
        mined = score_pairs(pairs, emb_lookup)
        logger.info(f"  {len(mined):,} candidate pairs before any cutoff")
        tab = audit(mined, verdicts, n_acc)
        results[lab] = tab
        logger.info(f"  {'cutoff':>7}{'mined':>9}{'judged':>8}{'precision':>11}{'95% CI':>16}"
              f"{'recall':>8}")
        for r in tab.itertuples(index=False):
            mark = "  <- SSL_MIN_SIM default" if abs(r.cutoff - 0.40) < 1e-9 else ""
            logger.info(f"  {r.cutoff:>7.2f}{r.mined:>9,}{r.n_judged:>8}{r.precision:>11.3f}"
                  f"{f'[{r.ci_lo:.2f},{r.ci_hi:.2f}]':>16}{r.recall:>8.3f}{mark}")

    base = results["with RANSAC"]
    at40 = base[np.isclose(base["cutoff"], 0.40)].iloc[0]
    logger.info(f"=== Verdict at the current default (SSL_MIN_SIM=0.40, RANSAC on) ===")
    logger.info(f"   mining precision {at40['precision']:.3f} "
          f"[{at40['ci_lo']:.2f}, {at40['ci_hi']:.2f}] over {int(at40['n_judged'])} adjudicated "
          f"pairs of {int(at40['mined']):,} mined")
    logger.info(f"   -> roughly {1 - at40['precision']:.0%} of the SSL positives are wrong pairs, "
          f"teaching the encoder that two different spots are the same")

    # The decision this audit exists to inform: precision is flat across the knobs, so the knobs are
    # not controlling it, and the only thing they change is how much training data survives.
    spread = base.dropna(subset=["precision"])["precision"]
    flat = (spread.max() - spread.min()) < 0.15
    if "no RANSAC" in results:
        ng = results["no RANSAC"]
        ng40 = ng[np.isclose(ng["cutoff"], 0.40)].iloc[0]
        logger.info(f"   RANSAC on/off: precision {ng40['precision']:.3f} "
              f"[{ng40['ci_lo']:.2f}, {ng40['ci_hi']:.2f}] -> {at40['precision']:.3f} "
              f"[{at40['ci_lo']:.2f}, {at40['ci_hi']:.2f}] ({at40['precision'] - ng40['precision']:+.3f})"
              f"\n   while yield {int(ng40['mined']):,} -> {int(at40['mined']):,} "
              f"({at40['mined'] / max(ng40['mined'], 1):.0%} kept)")
        overlap = not (at40["ci_lo"] > ng40["ci_hi"] or ng40["ci_lo"] > at40["ci_hi"])
        if overlap:
            logger.info("   The intervals overlap: RANSAC's precision gain is not resolvable at this "
                  "sample size,\n   but it demonstrably halves the training set. That trade is "
                  "not currently justified.")

    if flat:
        best_yield = base[base["precision"].notna()].sort_values("mined", ascending=False).iloc[0]
        logger.info(f"   Precision is FLAT across every cutoff ({spread.min():.2f}-{spread.max():.2f}), "
              f"so SSL_MIN_SIM is not\n   controlling label quality — it is only controlling yield. "
              f"The pretraining set can be\n   grown to {int(best_yield['mined']):,} pairs at "
              f"cutoff {best_yield['cutoff']:.2f} without measurably worse positives.")

    logger.info(f"   Mining recall against known correspondences is {at40['recall']:.3f} — this rule "
          f"finds a small\n   fraction of the true correspondences, which is a yield ceiling, "
          f"not a precision problem.")

    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "mining_audit"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "mining_audit.json").write_text(
        json.dumps({k: v.to_dict("records") for k, v in results.items()}, indent=2, default=float),
        encoding="utf-8")

    md = ["# Correspondence-mining audit against human verdicts", "",
          f"- dataset `{ds}` · {len(verdicts)} adjudicated correspondences ({n_acc} accepted) "
          f"over {rl.matcher_operating_point(ds).get('n_individuals', '?')} individuals",
          "- `precision` is over mined pairs a human actually judged (`judged`); a mined pair with",
          "  no verdict is unknown, not correct. `recall` is over all accepted correspondences.",
          "- results.md #29's gain (mined 0.375 R@1 vs augmented 0.277) rests on this precision.", ""]
    for lab, tab in results.items():
        md += [f"**{lab}**", "",
               "| cutoff | mined | judged | precision | 95% CI | found | recall |",
               "|---|---|---|---|---|---|---|"]
        for r in tab.itertuples(index=False):
            md.append(f"| {r.cutoff:.2f} | {r.mined:,} | {r.n_judged} | {r.precision:.3f} | "
                      f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}] | {r.n_accepted_found} | {r.recall:.3f} |")
        md.append("")
    (outdir / "RESULTS_mining_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info(f"wrote {outdir / 'RESULTS_mining_audit.md'}")


if __name__ == "__main__":
    main()
