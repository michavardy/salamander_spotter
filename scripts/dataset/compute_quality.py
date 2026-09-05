#!/usr/bin/env python3
"""Populate an ``image_quality`` table of cheap, model-free filter markers — no API, seconds/dir.

Reads what stage 2 already stored (the body mask blob, the spots, the axis) plus the original
photo off disk, and writes one quality row per image: raw measurements AND the four 0..1
composite scores (blur / lighting / spot-extraction / body-extraction) plus an overall. See
``pipeline/generate_spot_labels/quality.py`` for exactly what each field means.

    pixi run compute-quality --input all_sasa_norm            # (re)build the image_quality table
    pixi run compute-quality --input all_sasa_norm --dry-run  # compute + print stats, no writes

Runs AFTER ``contours`` (it reads the spots and the body mask from the DB). It only adds/refreshes
its own table — nothing else in the DB is touched — so it is safe to re-run any time, e.g. after
retuning the weights in ``quality.py``. Prints a distribution summary and the worst images per
score so you can pick thresholds.

Filter later with plain SQL, e.g.::

    SELECT salamander_id FROM image_quality
    WHERE overall_quality < 0.4 OR spots_outside_frac > 0.1 OR border_frac > 0;
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from generate_spot_labels import quality as ql  # noqa: E402
from generate_spot_labels import reconfigure_utf8  # noqa: E402
from generate_spot_labels._common import contours_db_for, resolve_input_dir  # noqa: E402

TABLE = "image_quality"


def ensure_table(con) -> None:
    cols = ",\n                ".join(f'"{n}" {t}' for n, t in ql.RAW_FIELDS + ql.SCORE_FIELDS)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
                salamander_id VARCHAR PRIMARY KEY,
                {cols}
        );
    """)
    have = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        f"WHERE table_name = '{TABLE}'").fetchall()}
    for n, t in ql.RAW_FIELDS + ql.SCORE_FIELDS:      # migrate if the field set grew
        if n not in have:
            con.execute(f'ALTER TABLE {TABLE} ADD COLUMN "{n}" {t}')


def find_original(input_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = input_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="image dir under images/ (with contours.db)")
    p.add_argument("--dry-run", action="store_true", help="compute + summarise, write nothing")
    p.add_argument("--limit", type=int, default=0, help="first N images only (0 = all)")
    args = p.parse_args(argv)

    import duckdb

    input_dir = resolve_input_dir(args.input)
    db_path = contours_db_for(input_dir)
    if not db_path.is_file():
        print(f"error: no DB at {db_path} — run contours first", file=sys.stderr)
        return 1
    con = duckdb.connect(str(db_path))
    ensure_table(con)

    ids = [r[0] for r in con.execute(
        "SELECT salamander_id FROM images ORDER BY salamander_id").fetchall()]
    if args.limit:
        ids = ids[:args.limit]
    print(f"computing quality for {len(ids)} images in {input_dir.name}  "
          f"({'DRY RUN' if args.dry_run else 'WRITE'})")

    # Pass 1: raw metrics for every image (also collects blur to calibrate the composite).
    raws: dict[str, dict] = {}
    missing_orig = missing_mask = 0
    for sid in ids:
        row = con.execute("SELECT body_mask_png FROM images WHERE salamander_id = ?",
                          [sid]).fetchone()
        blob = row[0] if row else None
        if not blob:
            missing_mask += 1
            continue
        op = find_original(input_dir, sid)
        if op is None:
            missing_orig += 1
            continue
        orig = cv2.imread(str(op), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
        if orig is None or mask is None:
            continue
        mask = (mask > 127).astype(np.uint8) * 255

        ax = con.execute("SELECT length_px, midline_x, midline_y, source, judged_ok "
                         "FROM body_axis WHERE salamander_id = ?", [sid]).fetchone()
        length = ax[0] if ax else None
        midline = list(zip(ax[1], ax[2])) if ax and ax[1] else []
        spots = [{"global_centroid": (cx, cy), "area_pixels": a} for cx, cy, a in con.execute(
            "SELECT global_centroid_x, global_centroid_y, area_pixels FROM spots "
            "WHERE salamander_id = ?", [sid]).fetchall()]

        raw = ql.raw_metrics(orig, mask, midline, spots, length_px=length)
        raw["axis_source"] = ax[3] if ax else None
        raw["judged_ok"] = ax[4] if ax else None
        raws[sid] = raw

    if not raws:
        print("no images had both a stored mask and an original photo — nothing to do",
              file=sys.stderr)
        return 1

    # Auto-tune: learn the per-metric scales from THIS dataset, then score against them.
    calib = ql.calibrate(list(raws.values()))
    print("\nauto-calibration (learned from this dataset — no hand-tuning):")
    for k, v in calib.as_dict().items():
        print(f"    {k:14s} {v}")

    # Pass 2: composites + write.
    rows = []
    for sid, raw in raws.items():
        raw.update(ql.composite_scores(raw, calib))
        rows.append((sid, raw))
        if not args.dry_run:
            cols = [n for n, _ in ql.RAW_FIELDS + ql.SCORE_FIELDS]
            con.execute(f"DELETE FROM {TABLE} WHERE salamander_id = ?", [sid])
            con.execute(
                f'INSERT INTO {TABLE} (salamander_id, {", ".join(cols)}) '
                f'VALUES ({", ".join(["?"] * (len(cols) + 1))})',
                [sid, *[raw.get(c) for c in cols]])

    # --- summary so you can pick thresholds ---
    print()
    if missing_mask or missing_orig:
        print(f"skipped: {missing_mask} without a stored mask, {missing_orig} without an original")
    print(f"\n{'score':26s} {'min':>6} {'p10':>6} {'med':>6} {'p90':>6}   worst 3")
    for field, _ in ql.SCORE_FIELDS:
        vals = sorted(r[field] for _, r in rows)
        pct = lambda q: vals[min(int(q * (len(vals) - 1)), len(vals) - 1)]
        worst = sorted(rows, key=lambda kv: kv[1][field])[:3]
        tags = ", ".join(f"{sid}={r[field]:.2f}" for sid, r in worst)
        print(f"  {field:24s} {vals[0]:6.2f} {pct(.1):6.2f} {pct(.5):6.2f} {pct(.9):6.2f}   {tags}")

    if args.dry_run:
        print(f"\nDRY RUN — nothing written. Re-run without --dry-run to populate {TABLE}.")
    else:
        print(f"\nwrote {len(rows)} rows to {TABLE} in {db_path.name}")
        print("filter e.g.:  SELECT salamander_id FROM image_quality "
              "WHERE overall_quality < 0.4 OR spots_outside_frac > 0.1;")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
