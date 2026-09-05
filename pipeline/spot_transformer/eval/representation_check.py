"""Does a spot representation see what the human sees? — seconds, not hours.

``gate_calibration`` established the problem: on the 427 spot correspondences a human accepted or
rejected, the 62-dim embedding's cosine is **0.645 vs 0.650 — AUROC 0.484**, a coin flip. Whatever
the reviewer is reacting to when they reject a match is not in those numbers.

That null is also a **measuring instrument**, and this module is it. Any candidate per-spot
descriptor gets one question — do the pairs the human accepted score higher than the pairs they
rejected? — and one number back. It runs in seconds because it needs no folds, no training and no
gallery, which is the entire point: a representation change can be judged before committing to a
multi-hour census run, and twenty ideas fit in an afternoon.

**This is a screen, not a result.** It is one axis (edge-level agreement with a human), measured on
one small hand-labelled set, and a representation could improve it while hurting retrieval. Nothing
here belongs in a paper as a headline; use ``sweeps/sweep_all9.py`` for that. What this earns you is
knowing which candidates deserve the sweep.

Read the numbers against these anchors:

    0.50   chance — the descriptor carries none of the human's decision (where the baseline sits)
    0.60+  the descriptor has started to encode it
    0.70+  strong, for a single edge-level cue

    pixi run repr-check                                   # every built variant + the baseline
    pixi run repr-check --tables spot_embeddings,spot_embeddings_morph
    pixi run repr-check --ssl-tags simclr,corr,corr_audit
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval"):
    p = str(_ST / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import data as d                                      # noqa: E402
import review_labels as rl                            # noqa: E402
from census import auroc                              # noqa: E402

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

BASELINE_TABLE = "spot_embeddings"


# ----------------------------------------------------------------------------- representations
def vectors_from_table(table: str) -> dict[tuple[str, int], np.ndarray]:
    """``(sid, spot_id) -> vector`` from a DuckDB embedding table."""
    df = d.get_spot_embeddings(table=table)
    return {(r.salamander_id, int(r.spot_id)): np.asarray(r.embedding, float)
            for r in df.itertuples(index=False)}


def vectors_from_ssl(tag: str, *, with_position: bool = True
                     ) -> dict[tuple[str, int], np.ndarray]:
    """``(sid, spot_id) -> vector`` from a cached SSL encoder.

    ``with_position`` concatenates the baseline's body-position block, matching
    ``ssl_pretrain.attach_ssl_features``'s default. Without it the SSL row is shape-only while
    every table row is shape+position, and the comparison would be measuring the missing position
    block rather than the encoder.
    """
    from ssl_pretrain import cache_path, POS_SLICE                       # noqa: PLC0415
    path = cache_path(d.dataset_name, tag)
    if not path.is_file():
        raise SystemExit(f"no SSL cache for tag {tag!r} at {path}\n"
                         f"  build it first, e.g.  SSL_MODE=corr SSL_TAG={tag} "
                         f"pixi run python pipeline/spot_transformer/models/ssl_pretrain.py")
    by_spot = pickle.loads(path.read_bytes())["by_spot"]
    out = {k: np.asarray(v, float) for k, v in by_spot.items()}
    if with_position:
        base = vectors_from_table(BASELINE_TABLE)
        merged = {}
        for k, v in out.items():
            b = base.get(k)
            if b is None:
                continue
            v = v / (np.linalg.norm(v) + 1e-12)
            p = b[POS_SLICE]
            p = p / (np.linalg.norm(p) + 1e-12)
            merged[k] = np.concatenate([v, p])
        return merged
    return out


# ----------------------------------------------------------------------------- scoring
def cluster_bootstrap_auroc(scores, y, groups, *, n_boot: int = 2000, seed: int = 0):
    """95% CI for AUROC, resampling **individuals** rather than edges.

    Edges from one animal share its extraction quality and its body-axis fit, so they are not
    independent draws; an edge-level bootstrap would report an interval two or three times too
    narrow and make differences between representations look decisive when they are not.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        yy = y[idx]
        if yy.sum() == 0 or (yy == 0).sum() == 0:
            continue
        vals.append(auroc(scores[idx], yy))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def random_pairs(vectors: dict[tuple[str, int], np.ndarray], n: int, *, seed: int = 0
                 ) -> np.ndarray:
    """Cosines of ``n`` random spot pairs drawn from DIFFERENT individuals.

    The control that makes a low verdict-AUROC interpretable. These pairs are *definitely* not the
    same physical spot, so a representation with any per-spot discriminative power at all must
    score human-accepted correspondences above them. If it separates accepted-from-random but not
    accepted-from-rejected, the representation is not blind — the human's rejections are being
    made on something other than how the two spots look, and no per-spot descriptor can be tuned
    to reproduce them.
    """
    rng = np.random.default_rng(seed)
    keys = list(vectors)
    inds = np.array(["_".join(k[0].split("_")[:2]) for k in keys])
    out = []
    guard = 0
    while len(out) < n and guard < n * 50:
        guard += 1
        i, j = rng.integers(len(keys)), rng.integers(len(keys))
        if inds[i] == inds[j]:
            continue
        va, vb = vectors[keys[i]], vectors[keys[j]]
        out.append(float(va @ vb / ((np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-12)))
    return np.asarray(out, float)


def check(vectors: dict[tuple[str, int], np.ndarray], edges: pd.DataFrame, *,
          n_boot: int = 2000, seed: int = 0) -> dict:
    """Score one representation on the human verdicts. Returns AUROC + CI + class means."""
    cos, keep = [], []
    for r in edges.itertuples(index=False):
        va, vb = vectors.get((r.sid_a, r.spot_a)), vectors.get((r.sid_b, r.spot_b))
        if va is None or vb is None:
            keep.append(False); cos.append(np.nan); continue
        cos.append(float(va @ vb / ((np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-12)))
        keep.append(True)
    cos = np.asarray(cos, float); keep = np.asarray(keep)
    sub = edges[keep]
    s = cos[keep]
    y = sub["accepted"].to_numpy().astype(int)
    if len(np.unique(y)) < 2:
        return dict(n=len(sub), auroc=float("nan"), lo=float("nan"), hi=float("nan"))
    lo, hi = cluster_bootstrap_auroc(s, y, sub["individual"].to_numpy(), n_boot=n_boot, seed=seed)

    # Control: human-ACCEPTED correspondences vs random cross-individual pairs. High here with
    # chance on the verdict AUROC is the informative combination -- it says the descriptor can tell
    # spots apart perfectly well, and the human's rejections simply are not an appearance judgement.
    acc = s[y == 1]
    rnd = random_pairs(vectors, max(len(acc) * 4, 400), seed=seed)
    ctrl = np.concatenate([acc, rnd])
    ctrl_y = np.concatenate([np.ones(len(acc)), np.zeros(len(rnd))])
    return dict(
        n=int(len(sub)), n_missing=int((~keep).sum()),
        auroc=float(auroc(s, y)), lo=lo, hi=hi,
        cos_pos=float(s[y == 1].mean()), cos_neg=float(s[y == 0].mean()),
        cos_rand=float(rnd.mean()),
        auroc_ctrl=float(auroc(ctrl, ctrl_y)),
        n_individuals=int(sub["individual"].nunique()),
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
    ap.add_argument("--tables", default=None,
                    help="comma-separated embedding tables (default: every one that exists)")
    ap.add_argument("--ssl-tags", default=None, help="comma-separated SSL cache tags")
    ap.add_argument("--no-position", action="store_true",
                    help="do NOT append the position block to SSL vectors (shape-only)")
    ap.add_argument("--pair-kind", default=None,
                    choices=["real-real", "real-synth", "synth-synth"])
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (0 to skip)")
    args = ap.parse_args()

    ds = d.dataset_name
    edges = rl.verdict_edges(ds, proposed_by="algorithm", pair_kind=args.pair_kind)
    if not len(edges):
        raise SystemExit("no adjudicated machine-proposed edges — nothing to check against")
    logger.info(f"dataset {ds}")
    logger.info(f" {len(edges)} human-judged correspondences "
          f"({int(edges['accepted'].sum())} accepted / {int((~edges['accepted']).sum())} rejected)"
          + (f"  [pair_kind={args.pair_kind}]" if args.pair_kind else ""))

    if args.tables:
        tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    else:
        import duckdb
        con = duckdb.connect(str(d.DB_PATH), read_only=True)
        try:
            have = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        finally:
            con.close()
        tables = sorted(t for t in have if t.startswith("spot_embeddings"))
    ssl_tags = [t.strip() for t in (args.ssl_tags or "").split(",") if t.strip()]

    rows = []
    for t in tables:
        try:
            res = check(vectors_from_table(t), edges, n_boot=args.boot)
        except Exception as exc:
            logger.error(f"  !! {t}: {type(exc).__name__}: {exc}")
            continue
        res["name"] = t
        rows.append(res)
    for tag in ssl_tags:
        try:
            res = check(vectors_from_ssl(tag, with_position=not args.no_position), edges,
                        n_boot=args.boot)
        except SystemExit as exc:
            logger.error(f"  !! ssl:{tag}: {exc}")
            continue
        res["name"] = f"ssl:{tag}" + ("" if not args.no_position else " (shape only)")
        rows.append(res)

    if not rows:
        raise SystemExit("nothing to compare")

    logger.info(f"{'representation':<30}{'verdict':>9}{'95% CI':>15}{'control':>9}"
          f"{'cos +':>8}{'cos -':>8}{'cos rnd':>9}")
    logger.info(f"{'':30}{'AUROC':>9}{'':15}{'AUROC':>9}")
    logger.info("-" * 88)
    base = next((r for r in rows if r["name"] == BASELINE_TABLE), None)
    for r in sorted(rows, key=lambda x: -(x["auroc"] if np.isfinite(x["auroc"]) else -1)):
        ci = f"[{r['lo']:.2f},{r['hi']:.2f}]"
        logger.info(f"{r['name']:<30}{r['auroc']:>9.3f}{ci:>15}{r['auroc_ctrl']:>9.3f}"
              f"{r['cos_pos']:>8.3f}{r['cos_neg']:>8.3f}{r['cos_rand']:>9.3f}")
    logger.info(" verdict AUROC = accepted vs REJECTED correspondences (what the human decided)")
    logger.info(" control AUROC = accepted vs RANDOM cross-individual pairs (basic discriminative power)")

    logger.info(f" anchors: 0.50 = chance (the descriptor carries none of the human's decision)")
    logger.info(f"          0.60 = it has started to encode it · 0.70 = strong for one edge-level cue")
    if base is not None:
        logger.info(f" baseline `{BASELINE_TABLE}` sits at {base['auroc']:.3f}"
              f" — that null is why this check exists.")
    best = max(rows, key=lambda x: x["auroc"] if np.isfinite(x["auroc"]) else -1)
    if base is not None and np.isfinite(best["auroc"]):
        if best["auroc"] - base["auroc"] < 0.03:
            logger.info(" VERDICT: nothing here separates the human's verdict better than the baseline.")
            if base["auroc_ctrl"] >= 0.70:
                logger.info(f" But look at the control column: these descriptors separate accepted"
                      f" correspondences\n from random cross-animal pairs at "
                      f"{base['auroc_ctrl']:.3f}. They are NOT blind — they tell spots apart fine.")
                logger.info(" What they cannot do is predict the human's REJECTIONS, which means those")
                logger.info(" rejections are not an appearance judgement at all. The review comments say")
                logger.info(" the same thing in words: \"only two matches, which isn't very much\",")
                logger.info(" \"the LACK of this spot should be penalized\" — those are statements about")
                logger.info(" the whole configuration, not about how two spots look.")
                logger.info(" => stop enriching the per-spot descriptor; the missing evidence is")
                logger.info("    RELATIONAL (how many matches, what is absent, is the constellation")
                logger.info("    consistent). That is a scoring-rule problem, not an embedding problem.")
            else:
                logger.info(" The control column is also weak, so these descriptors have little per-spot")
                logger.info(" discriminative power to begin with — a learned encoder is the next step.")
        elif base["lo"] < best["auroc"] < base["hi"]:
            logger.warning(f" NOTE: {best['name']} leads but sits inside the baseline's own CI — suggestive,")
            logger.warning(" not established. Worth one census run, not a conclusion.")
        else:
            logger.info(f" VERDICT: {best['name']} clears the baseline's interval "
                  f"({best['auroc']:.3f} vs {base['auroc']:.3f}). This one has earned a full census")
            logger.info(" run — confirm it on sweep_all9 before it goes anywhere near the paper.")

    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "representation"
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[["name", "auroc", "lo", "hi", "cos_pos", "cos_neg", "n",
                             "n_individuals"]]
    df.to_csv(outdir / "representation_check.csv", index=False)
    md = ["# Does the representation see what the human sees?", "",
          f"- dataset `{ds}` · {len(edges)} human-judged correspondences over "
          f"{edges['individual'].nunique()} individuals · pair kinds `{args.pair_kind or 'all'}`",
          "- **Screen, not a result.** One edge-level axis on a small hand-labelled set; a change",
          "  can improve this and still hurt retrieval. Headline numbers come from `sweep_all9`.",
          "- CI is a cluster bootstrap over individuals (edges within an animal are not "
          "independent).", "",
          "| representation | AUROC | 95% CI | cos accepted | cos rejected | n |",
          "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: -(x["auroc"] if np.isfinite(x["auroc"]) else -1)):
        md.append(f"| `{r['name']}` | **{r['auroc']:.3f}** | [{r['lo']:.2f}, {r['hi']:.2f}] | "
                  f"{r['cos_pos']:.3f} | {r['cos_neg']:.3f} | {r['n']} |")
    (outdir / "RESULTS_representation_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info(f"wrote {outdir / 'RESULTS_representation_check.md'}")


if __name__ == "__main__":
    main()
