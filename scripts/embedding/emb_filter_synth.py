#!/usr/bin/env python3
"""emb-filter-synth — the self-consistency gate on Gemini-generated views (free, no API).

``docs/running.md`` makes this mandatory: *"identity preservation is not guaranteed"*. Gemini can
quietly add, drop or move a spot, and a view whose pattern drifted is not another photo of that
individual — it is **label noise** aimed straight at the training pool. This scores every
synthetic ``<label>_g<k>`` view against that individual's REAL photo(s) with the classical
constellation matcher (inlier fraction in [0, 1], 1.0 = the constellations align perfectly) and
quarantines the ones that fail.

The threshold is not guessed. Each run first **calibrates** on the data already in the DB:

    same individual, two real photos   -> the positive distribution (what "is" looks like)
    different individuals, real photos -> the negative distribution (what "isn't" looks like)

and reports where the synthetic views fall against both, so a rejected view means "less
self-consistent than two genuine photos of one animal", not "below a number someone picked".

    pixi run emb-filter-synth --input all_sasa_norm                       # calibrate + report
    pixi run emb-filter-synth --input all_sasa_norm --threshold 0.30      # + list the rejects
    pixi run emb-filter-synth --input all_sasa_norm --threshold 0.30 --apply

``--apply`` MOVES each rejected view and its derived artifacts (purple, anatomy, mask, QA
overlay) into ``<input>/rejected_synth/``. Nothing is deleted, so it is reversible; re-run
``extract-spot-labels contours`` afterwards to drop the rejects from the DB.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from generate_spot_labels._common import reconfigure_utf8, resolve_input_dir  # noqa: E402
from spot_embedding.data.spot_store import _read_from_db  # noqa: E402
from spot_embedding.models.classical import ConstellationMatcher  # noqa: E402

G_RE = re.compile(r"^(.+)_g(\d+)$")
QUARANTINE = "rejected_synth"


def is_synthetic(salamander_id: str) -> bool:
    """A generated view — the last delimited section is ``g<k>`` (``aa_1_g0``)."""
    return bool(G_RE.match(salamander_id))


def pct(vals: list[float], q: float) -> float:
    return float(np.percentile(vals, q)) if vals else float("nan")


def describe(name: str, vals: list[float]) -> None:
    if not vals:
        print(f"  {name:<28s} (none)")
        return
    print(f"  {name:<28s} n={len(vals):5d}  "
          f"p05={pct(vals, 5):.3f}  p25={pct(vals, 25):.3f}  median={pct(vals, 50):.3f}  "
          f"p75={pct(vals, 75):.3f}  mean={np.mean(vals):.3f}")


def calibrate(matcher, by_label: dict[str, list], rng: np.random.Generator,
              n_neg: int = 1500) -> tuple[list[float], list[float]]:
    """Score real-vs-real pairs: same individual (positives) and different (negatives)."""
    pos: list[float] = []
    for sets in by_label.values():
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                pos.append(1.0 - matcher.distance_matrix([sets[i]], [sets[j]])[0, 0])

    multi = [lbl for lbl, s in by_label.items() if s]
    neg: list[float] = []
    if len(multi) >= 2:
        for _ in range(n_neg):
            a, b = rng.choice(len(multi), size=2, replace=False)
            sa = by_label[multi[a]][0]
            sb = by_label[multi[b]][0]
            neg.append(1.0 - matcher.distance_matrix([sa], [sb])[0, 0])
    return pos, neg


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    p = argparse.ArgumentParser(prog="emb-filter-synth", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="all_sasa_norm",
                   help="image dir under images/ (reads its contours/contours.db)")
    p.add_argument("--db", default=None, help="explicit contours.db path (overrides --input)")
    p.add_argument("--threshold", type=float, default=None,
                   help="reject a view scoring below this against its own real photo(s). "
                        "Omit to calibrate only and see where a threshold would land")
    p.add_argument("--max-ratio", type=float, default=1.5,
                   help="spot-count ratio (synth/real) above which a view is suspect; it is only "
                        "rejected if its score is also below --ratio-score (default 1.5)")
    p.add_argument("--ratio-score", type=float, default=0.30,
                   help="the score below which a wild spot ratio becomes a rejection (default "
                        "0.30). A near-identical view with a high ratio is an extraction "
                        "disagreement, not an identity change, and is kept")
    p.add_argument("--apply", action="store_true",
                   help="MOVE the rejects into <input>/rejected_synth/ (default: report only)")
    p.add_argument("--seed", type=int, default=0, help="seed for the negative-pair sample")
    args = p.parse_args(argv)

    input_dir = resolve_input_dir(args.input)
    db = Path(args.db) if args.db else input_dir / "contours" / "contours.db"
    if not db.is_file():
        print(f"error: no contours.db at {db}", file=sys.stderr)
        return 1

    spotsets = _read_from_db(db)
    real_by_label: dict[str, list] = {}
    synth: list = []
    for ss in spotsets:
        if is_synthetic(ss.salamander_id):
            synth.append(ss)
        elif not ss.is_empty:
            real_by_label.setdefault(ss.label, []).append(ss)

    print(f"{db.relative_to(REPO_ROOT)}: {len(spotsets)} images "
          f"({len(synth)} synthetic, {sum(len(v) for v in real_by_label.values())} real)\n")
    if not synth:
        print("no synthetic views in this DB — nothing to filter.")
        return 0

    matcher = ConstellationMatcher()
    rng = np.random.default_rng(args.seed)

    print("calibration (real photos only — the yardstick):")
    multi = {lbl: s for lbl, s in real_by_label.items() if len(s) >= 2}
    pos, neg = calibrate(matcher, multi if multi else real_by_label, rng)
    describe("same individual (positive)", pos)
    describe("different individuals (neg)", neg)

    # The positive and negative distributions above overlap almost completely on this data (see
    # the note printed below), so "above the positives' p05" is NOT a usable gate — it would pass
    # anything. What IS usable: a synthetic view is a near-COPY of its source, so it should align
    # with it far better than two genuinely different photos do. The overlap point of the two real
    # distributions is the noise floor; a view at or below it aligns with the image it was copied
    # from no better than an unrelated animal would, which means the pattern did not survive.
    floor = float(np.median(neg)) if neg else float("nan")
    if pos and neg and abs(pct(pos, 50) - pct(neg, 50)) < 0.02:
        print("\n  ! positives and negatives overlap almost completely (medians "
              f"{pct(pos, 50):.3f} vs {pct(neg, 50):.3f}) — this matcher does not separate "
              "individuals\n    pair-by-pair on this data, so a threshold calibrated from the "
              "positives would pass everything.\n    Falling back to the NOISE FLOOR "
              f"({floor:.3f}, the real-pair median): a near-copy scoring at or below it failed.")

    # score each synthetic view against the best of its individual's real photos
    scored: list[tuple[float, str]] = []
    ratio_of: dict[str, float] = {}
    orphan = 0
    for ss in synth:
        reals = real_by_label.get(ss.label, [])
        if not reals:
            orphan += 1
            continue
        ratio_of[ss.salamander_id] = ss.n_spots / max(1, max(r.n_spots for r in reals))
        if ss.is_empty:
            scored.append((0.0, ss.salamander_id))   # no spots extracted = cannot be verified
            continue
        best = max(1.0 - matcher.distance_matrix([ss], [r])[0, 0] for r in reals)
        scored.append((best, ss.salamander_id))
    ratios = list(ratio_of.values())

    print("\nsynthetic views vs their own real photo(s):")
    describe("synthetic (self-consistency)", [s for s, _ in scored])
    if orphan:
        print(f"  ({orphan} view(s) skipped: no real photo for that individual in this DB)")

    # An independent, matcher-free sanity check: identity preservation means the SAME NUMBER of
    # spots. A ratio far from 1.0 says Gemini added or dropped spots regardless of how they align.
    if ratios:
        describe("spot-count ratio (synth/real)", ratios)
        off = sum(1 for r in ratios if r < 0.7 or r > 1.4)
        print(f"  {off} of {len(ratios)} view(s) differ from their source by >30% in spot count")

    if not np.isnan(floor):
        for t in (floor, floor * 1.4):
            print(f"\n  threshold {t:.3f} rejects {sum(1 for s, _ in scored if s < t)} "
                  f"of {len(scored)} synthetic views "
                  f"({100 * sum(1 for s in pos if s < t) / max(1, len(pos)):.0f}% of real "
                  f"same-individual pairs would also fall below it)")

    if args.threshold is None:
        print("\nNo --threshold given: calibration only, nothing selected for rejection.")
        return 0

    # Two gates, because neither alone is sufficient (verified by eye on the existing views):
    #   score      catches a redrawn pattern (af_2_g1 0.077, lj_17_g0 0.083) but MISSES a view
    #              invented from an occluded source — ca_21_g0 scored 0.200, above any sane
    #              floor, yet is a different animal with 34 spots where the source showed 10.
    #   spot ratio catches that, but alone it fires on views that are visibly the same animal
    #              where only the purple stage disagreed (jh_4_g0: ratio 2.43, score 0.571).
    # Requiring a mediocre score alongside a wild ratio keeps jh_4_g0 and drops ca_21_g0.
    def rejected(s: float, sid: str) -> bool:
        r = ratio_of.get(sid, 1.0)
        return s < args.threshold or (r > args.max_ratio and s < args.ratio_score)

    rejects = sorted((s, i) for s, i in scored if rejected(s, i))
    print(f"\nthreshold {args.threshold:.3f} (or spot ratio >{args.max_ratio} with score "
          f"<{args.ratio_score}) -> {len(rejects)} reject(s) of {len(scored)} scored "
          f"({100 * len(rejects) / max(1, len(scored)):.1f}%)")
    for s, sid in rejects[:25]:
        print(f"    {sid:20s} score={s:.3f}  ratio={ratio_of.get(sid, 1.0):.2f}")
    if len(rejects) > 25:
        print(f"    ... and {len(rejects) - 25} more")

    if not rejects:
        return 0
    if not args.apply:
        print("\nREPORT ONLY — nothing moved. Re-run with --apply to quarantine these.")
        return 0

    quarantine = input_dir / QUARANTINE
    moved = 0
    for _, sid in rejects:
        # the view itself lives in the dir root; its artifacts in purple/ and anatomy/. Each keeps
        # its subdir under the quarantine root, so restoring is a copy straight back.
        sources = [(input_dir, p) for p in input_dir.glob(f"{sid}.*") if p.is_file()]
        for sub, patterns in (("purple", (f"{sid}.*",)),
                              ("anatomy", (f"{sid}.*", f"{sid}_mask.*"))):
            sources += [(input_dir / sub, p) for p in (input_dir / sub).glob(patterns[0])
                        if p.is_file()]
            if len(patterns) > 1:
                sources += [(input_dir / sub, p) for p in (input_dir / sub).glob(patterns[1])
                            if p.is_file()]
        for base, src in sources:
            dst_dir = quarantine / src.parent.relative_to(input_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst_dir / src.name))
            moved += 1
    print(f"\nquarantined {len(rejects)} view(s), {moved} file(s) -> {quarantine}")

    # Moving the files is NOT enough. `extract-spot-labels contours` upserts per image and never
    # prunes, and `package-dataset` copies the DB wholesale (`COPY FROM DATABASE`) while raw/
    # comes from disk — so a row whose file is gone still travels into the dataset, and
    # load_spotsets keys off the DB. The rows have to be deleted here, or the reject stays in the
    # training pool with only its image missing.
    import duckdb

    ids = [sid for _, sid in rejects]
    con = duckdb.connect(str(db))
    try:
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.columns "
            "WHERE column_name = 'salamander_id'").fetchall()]
        deleted = {}
        for t in sorted(set(tables)):
            before = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
            con.execute(f'DELETE FROM "{t}" WHERE salamander_id IN '
                        f'({",".join("?" * len(ids))})', ids)
            deleted[t] = before - con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
    finally:
        con.close()
    print("removed from the DB: " + ", ".join(f"{n} {t}" for t, n in deleted.items() if n))
    print(f"\nDone. The rejects are recoverable in {QUARANTINE}/ (files) but are no longer in the "
          f"DB.\nRe-package when ready:  pixi run package-dataset --input {input_dir.name} "
          f"--name <dataset>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
