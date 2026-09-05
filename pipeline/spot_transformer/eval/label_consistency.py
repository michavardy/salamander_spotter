"""D1 — label-consistency diagnostic: is the ceiling the LABELS or the MODEL?

Every matcher in this repo assumes two photos of one animal carry the same spot constellation.
This measures that assumption directly, with no model involved, and answers the one question
that routes all other effort (see ``docs/sea_improvments.md``):

* If an individual's own photos agree no better than two *random* animals do, no aggregator can
  win — the signal is not in the extracted spots. Go to the data branch (better extraction).
* If they agree clearly better, the signal is there and the matcher is leaving it on the table.
  Go to the model branch.

Three outputs:

1. **The gate.** Same-individual vs different-individual distributions of each consistency
   statistic, plus the AUROC between them. AUROC ~0.5 means repeats of one animal are
   indistinguishable from unrelated animals — the labels/extraction cannot support matching.
2. **Per-individual table** (``per_individual.csv``, worst first): each individual's own
   agreement, and its percentile *within the different-individual null*. An individual sitting
   at the null's median is a concrete suspect — mislabeled, or one of its photos extracted badly.
3. **Duplicate scan** (``suspect_duplicates.csv``): pairs of DIFFERENT labels whose photos agree
   as well as genuine repeats do — i.e. probably one animal filed under two identities. These
   are invisible to every metric but poison them all: the model is scored wrong for finding the
   right animal under its other name.

Consistency statistics per photo pair (all computed on the fixed 62-dim spot embeddings, so this
measures the DATA, not any trained thing):

``ransac_frac``   fraction of mutual-nearest-neighbour spot matches that survive a single
                  similarity transform (scale+rotation+translation) — constellation agreement,
                  the strongest evidence two photos show one animal.
``mutual_frac``   fraction of query spots with a reciprocal nearest neighbour.
``softchamfer``   mean best-match cosine (the original hand score).
``spot_ratio``    min/max spot count — extraction stability, independent of matching.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root, so `pipeline.*` resolves

from pipeline.spot_transformer.core import data as d                        # noqa: E402
from pipeline.spot_transformer.models.aggregator import (                   # noqa: E402
    _norm, _similarity_inliers, attach_centroids,
)

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

STATS = ["ransac_frac", "mutual_frac", "softchamfer", "spot_ratio"]
OUT_ROOT = d.REPO_ROOT / "artifacts" / "label_consistency"


def pair_stats(a, b) -> dict[str, float]:
    """Consistency statistics between two photos' spot sets (order-independent by construction)."""
    qe, ce = _norm(a.spots), _norm(b.spots)
    S = qe @ ce.T
    nq, nc = S.shape
    qmax = S.max(1)
    qbest, cbest = S.argmax(1), S.argmax(0)
    mutual = np.array([i for i in range(nq) if cbest[qbest[i]] == i], dtype=int)

    ransac = 0.0
    if len(mutual) >= 3 and a.centroids is not None and b.centroids is not None:
        qp, cp = np.asarray(a.centroids)[mutual], np.asarray(b.centroids)[qbest[mutual]]
        if np.isfinite(qp).all() and np.isfinite(cp).all():
            ransac = _similarity_inliers(qp, cp)

    return {
        "ransac_frac": float(ransac),
        "mutual_frac": float(len(mutual) / nq) if nq else 0.0,
        "softchamfer": float(qmax.mean()) if nq else 0.0,
        "spot_ratio": float(min(nq, nc) / max(nq, nc)) if max(nq, nc) else 0.0,
        "n_mutual": float(len(mutual)),
        "n_spots_q": float(nq),
        "n_spots_c": float(nc),
    }


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random positive scores above a random negative). Mann-Whitney, ties at 0.5."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def describe(name: str, v: np.ndarray) -> str:
    q = np.percentile(v, [10, 50, 90]) if len(v) else [np.nan] * 3
    return f"  {name:14s} n={len(v):>6}  p10 {q[0]:.3f}   median {q[1]:.3f}   p90 {q[2]:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-null", type=int, default=3000,
                    help="random different-individual photo pairs for the null (default 3000)")
    ap.add_argument("--dup-top", type=int, default=300,
                    help="cross-label photo pairs to fully score in the duplicate scan (default 300)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None,
                    help=f"results dir (default: {OUT_ROOT}/<dataset>)")
    args = ap.parse_args()

    out_dir = args.output or (OUT_ROOT / d.dataset_name)
    rng = np.random.default_rng(args.seed)

    sets = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))
    real = [i for i, s in enumerate(sets) if not s.is_synth]
    by_label: dict[str, list[int]] = {}
    for i in real:
        by_label.setdefault(sets[i].label, []).append(i)
    multi = {lbl: idx for lbl, idx in by_label.items() if len(idx) >= 2}

    logger.info(f"dataset: {d.dataset_name}")
    logger.info(f"real photos: {len(real)}   individuals: {len(by_label)}   "
          f"with >=2 real photos: {len(multi)}")

    # --- 1. same-individual pairs ------------------------------------------------------------
    same_rows = []
    for lbl, idx in multi.items():
        for i, j in itertools.combinations(idx, 2):
            st = pair_stats(sets[i], sets[j])
            st["label"] = lbl
            st["sid_a"], st["sid_b"] = sets[i].sid, sets[j].sid
            same_rows.append(st)
    logger.info(f"same-individual photo pairs: {len(same_rows)}")

    # --- 2. different-individual null --------------------------------------------------------
    labels = sorted(by_label)
    diff_rows = []
    seen = set()
    tries = 0
    while len(diff_rows) < args.n_null and tries < args.n_null * 20:
        tries += 1
        la, lb = rng.choice(len(labels), 2, replace=False)
        la, lb = labels[la], labels[lb]
        i = int(rng.choice(by_label[la])); j = int(rng.choice(by_label[lb]))
        if (i, j) in seen:
            continue
        seen.add((i, j))
        st = pair_stats(sets[i], sets[j])
        st["label_a"], st["label_b"] = la, lb
        st["sid_a"], st["sid_b"] = sets[i].sid, sets[j].sid
        diff_rows.append(st)
    logger.info(f"different-individual photo pairs (null): {len(diff_rows)}")

    # --- 3. the gate -------------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info(" THE GATE — do an individual's own photos agree more than random animals do?")
    logger.info("=" * 72)
    aurocs = {}
    for stat in STATS:
        pos = np.array([r[stat] for r in same_rows])
        neg = np.array([r[stat] for r in diff_rows])
        aurocs[stat] = auroc(pos, neg)
        logger.info(f" {stat}   AUROC(same vs different) = {aurocs[stat]:.3f}")
        logger.info(describe("same", pos))
        logger.info(describe("different", neg))

    best = max(aurocs, key=lambda s: (aurocs[s] if np.isfinite(aurocs[s]) else -1))
    a_best = aurocs[best]
    logger.info("-" * 72)
    logger.info(f" strongest statistic: {best}  AUROC {a_best:.3f}")
    if a_best < 0.65:
        verdict = ("LABELS/EXTRACTION ARE THE CEILING. Repeat photos of one animal barely "
                   "separate from unrelated animals. No aggregator can fix this -> data branch "
                   "(X1: better spot extraction + a hand-verified subset) BEFORE more modeling.")
    elif a_best < 0.80:
        verdict = ("MIXED. There is real signal but it is noisy; expect a low ceiling until "
                   "extraction improves. Worth doing BOTH: model work now, extraction work in "
                   "parallel. Start with the worst individuals listed below.")
    else:
        verdict = ("SIGNAL IS THERE. Same-individual photos are clearly distinguishable at the "
                   "data level, so the matcher is leaving signal on the table -> model branch "
                   "(hard negatives, absence penalties, gallery pre-filtering).")
    logger.info(f" VERDICT: {verdict}")
    logger.info("-" * 72)

    # --- 4. per-individual table -------------------------------------------------------------
    neg_best = np.array([r[best] for r in diff_rows])
    per_ind = []
    for lbl, idx in multi.items():
        rows = [r for r in same_rows if r.get("label") == lbl]
        vals = np.array([r[best] for r in rows])
        med = float(np.median(vals))
        pct = float((neg_best < med).mean())        # where this individual sits inside the null
        per_ind.append({
            "label": lbl, "n_photos": len(idx), "n_pairs": len(rows),
            f"median_{best}": round(med, 4),
            "null_percentile": round(pct, 4),
            "median_mutual_frac": round(float(np.median([r["mutual_frac"] for r in rows])), 4),
            "median_spot_ratio": round(float(np.median([r["spot_ratio"] for r in rows])), 4),
            "spots_min": int(min(len(sets[i].spots) for i in idx)),
            "spots_max": int(max(len(sets[i].spots) for i in idx)),
        })
    per_ind.sort(key=lambda r: r["null_percentile"])

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_individual.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_ind[0]))
        w.writeheader(); w.writerows(per_ind)

    n_at_chance = sum(1 for r in per_ind if r["null_percentile"] < 0.5)
    logger.info(f" individuals whose own photos agree WORSE than the median random pair: "
          f"{n_at_chance}/{len(per_ind)}")
    logger.info(" worst 15 (go look at these images — mislabeled, or badly extracted):")
    logger.info(f"   {'label':10s} {'photos':>6} {'median_'+best:>18s} {'null_pct':>9s} {'spots':>12s}")
    for r in per_ind[:15]:
        logger.info(f"   {r['label']:10s} {r['n_photos']:>6} {r['median_'+best]:>18.3f} "
              f"{r['null_percentile']:>9.2f} {str(r['spots_min'])+'-'+str(r['spots_max']):>12s}")

    # --- 5. duplicate scan -------------------------------------------------------------------
    # Cheap prefilter first: mean spot embedding per photo. This can MISS a duplicate whose mean
    # drifts (different pose/lighting), so it is a lower bound on duplicates, not a clean bill.
    logger.info("=" * 72)
    logger.info(" DUPLICATE SCAN — two labels that may be the same animal")
    logger.info("=" * 72)
    M = _norm(np.stack([sets[i].spots.mean(0) for i in real]))
    Sim = M @ M.T
    lab_of = np.array([sets[i].label for i in real], dtype=object)
    iu = np.triu_indices(len(real), k=1)
    cross = lab_of[iu[0]] != lab_of[iu[1]]
    ci, cj, cs = iu[0][cross], iu[1][cross], Sim[iu][cross]
    top = np.argsort(-cs)[: args.dup_top]

    # The bar is the same-individual p90, NOT the median. These pairs were selected for being the
    # most similar in the dataset, so a median bar would flag nearly all of them and say nothing;
    # p90 asks the sharper question "do these two labels agree better than 90% of genuine repeats".
    same_med = float(np.median([r[best] for r in same_rows]))
    same_p90 = float(np.percentile([r[best] for r in same_rows], 90))
    dups = []
    for t in top:
        i, j = real[int(ci[t])], real[int(cj[t])]
        st = pair_stats(sets[i], sets[j])
        if st[best] >= same_p90:                    # agrees better than 90% of genuine repeats
            dups.append({
                "label_a": sets[i].label, "label_b": sets[j].label,
                "sid_a": sets[i].sid, "sid_b": sets[j].sid,
                best: round(st[best], 4),
                "mutual_frac": round(st["mutual_frac"], 4),
                "softchamfer": round(st["softchamfer"], 4),
                "n_mutual": int(st["n_mutual"]),
            })
    dups.sort(key=lambda r: -r[best])

    logger.info(f" scored the {len(top)} most similar cross-label photo pairs; genuine repeats: "
          f"median {best} {same_med:.3f}, p90 {same_p90:.3f} (the bar)")
    logger.info(f" pairs of DIFFERENT labels above the p90 bar: {len(dups)}")
    if dups:
        with (out_dir / "suspect_duplicates.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dups[0]))
            w.writeheader(); w.writerows(dups)
        logger.info(" top 15:")
        for r in dups[:15]:
            logger.info(f"   {r['sid_a']:16s} ~ {r['sid_b']:16s}  {best} {r[best]:.3f}  "
                  f"mutual {r['mutual_frac']:.2f}")
        logger.info(f" -> every confirmed duplicate here is a query the matcher is scored WRONG on "
              f"for being RIGHT.\n    Merge them before trusting any R@1.")
    else:
        logger.info(" -> no cross-label pair reaches genuine-repeat agreement (by this prefilter).")

    logger.info(f"wrote {out_dir / 'per_individual.csv'}"
          + (f" and {out_dir / 'suspect_duplicates.csv'}" if dups else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
