#!/usr/bin/env python3
"""Correct mis-tipped body axes from the SAVED MASKS — no model, no image generation, no cost.

When the painter makes a good body mask but drops the head/tail dots in the MIDDLE of it, the
axis collapses: the centre line spans only part of the animal and the spots on the unlabelled
end bin wrong. That is fixable from the mask alone, because the body's two true ends are a fact
about its shape. This walks ``images/<input>/anatomy/``, and for every image whose stored dots
span less than ``--coverage`` of the body length (measured ALONG the body, so curvature is
irrelevant), replaces the tips with the mask's own geodesic ends and re-derives the geometry.

    pixi run correct-axis --input all_sasa_norm               # correct the passing candidates
    pixi run correct-axis --input all_sasa_norm --dry-run     # report + QA only, change nothing
    pixi run correct-axis --input all_sasa_norm --force       # also write the ones still failing

By default this rewrites ``anatomy/<stem>.json`` (and its QA png) for every candidate whose
corrected axis passes every geometric gate — matching the other tasks here, where the bare
command does the work and ``--dry-run`` previews it. ``--dry-run`` writes nothing to the dataset:
it prints the candidates and drops a before/after image for each into
``artifacts/axis_corrections/<input>/`` so you can eyeball them first. ``--force`` also writes the
candidates whose axis still fails the gates (normally skipped). The masks are never touched, so
any run is reversible: re-run to redo, or restore from ``anatomy_*_backup`` if you kept one. Fold
the corrected axes into the DB afterwards (free)::

    pixi run extract-spot-labels contours --input all_sasa_norm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from generate_spot_labels import body_mask as bm  # noqa: E402
from generate_spot_labels import extract_spot_contours as stage2  # noqa: E402
from generate_spot_labels import reconfigure_utf8  # noqa: E402
from generate_spot_labels._common import contours_db_for, resolve_input_dir  # noqa: E402

CORRECTED_SOURCE = "mask_corrected"     # stamped into body_axis.source for provenance


def load_original(img_dir: Path, stem: str) -> "np.ndarray | None":
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = img_dir / f"{stem}{ext}"
        if p.is_file():
            return cv2.imread(str(p), cv2.IMREAD_COLOR)
    return None


def load_mask(anat_dir: Path, stem: str) -> "np.ndarray | None":
    p = anat_dir / f"{stem}_mask.png"
    if not p.is_file():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return None if m is None else (m > 127).astype(np.uint8) * 255


def side_by_side(before: np.ndarray, after: np.ndarray, note: str) -> np.ndarray:
    cv2.putText(before, "BEFORE", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.putText(after, note, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    h = max(before.shape[0], after.shape[0])
    pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    return np.hstack([pad(before), np.full((h, 6, 3), 255, np.uint8), pad(after)])


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="image dir under images/ (with anatomy/)")
    p.add_argument("--coverage", type=float, default=bm.DEFAULT_COVERAGE,
                   help=f"correct only if the dots span LESS than this fraction of the body "
                        f"(default {bm.DEFAULT_COVERAGE}; lower = more conservative)")
    p.add_argument("--dry-run", action="store_true",
                   help="preview only: print candidates + write before/after QA, change nothing")
    p.add_argument("--force", action="store_true",
                   help="also write corrections whose axis still fails the gates (normally skipped)")
    p.add_argument("--no-db", action="store_true",
                   help="don't propagate corrections into contours.db (leave re-binning to a "
                        "later contours run); by default a corrected axis is re-binned in place")
    p.add_argument("--limit", type=int, default=0, help="stop after N candidates (0 = all)")
    p.add_argument("--review-dir", default=None,
                   help="where the before/after images go "
                        "(default artifacts/axis_corrections/<input>/)")
    args = p.parse_args(argv)

    input_dir = resolve_input_dir(args.input)
    anat_dir = input_dir / "anatomy"
    if not anat_dir.is_dir():
        print(f"error: no anatomy dir at {anat_dir}", file=sys.stderr)
        return 1
    review_dir = Path(args.review_dir) if args.review_dir else (
        REPO_ROOT / "artifacts" / "axis_corrections" / input_dir.name)
    review_dir.mkdir(parents=True, exist_ok=True)

    # Corrections are re-binned into the DB in place (fast: only the changed images), so no full
    # contours re-run is needed. Opened only when we will actually write.
    db_path = contours_db_for(input_dir)
    store = None
    if not args.dry_run and not args.no_db and db_path.is_file():
        store = stage2.ContourStore(db_path)

    jsons = sorted(anat_dir.glob("*.json"))
    print(f"scanning {len(jsons)} anatomy files in {input_dir.name}  "
          f"(coverage trigger < {args.coverage:.0%}, "
          f"mode: {'DRY RUN' if args.dry_run else 'APPLY'}, "
          f"db: {'in-place re-bin' if store else 'not updated'})")

    scanned = no_mask = candidates = would_write = written = skipped_fail = rebinned = 0
    rows: list[dict] = []
    for jp in jsons:
        stem = jp.stem
        mask = load_mask(anat_dir, stem)
        if mask is None:
            no_mask += 1
            continue
        try:
            body = bm.Body.from_dict(json.loads(jp.read_text(encoding="utf-8")))
        except Exception:
            continue
        scanned += 1

        new_head, new_tail, did, info = bm.correct_tips_from_shape(
            mask, body.head, body.tail_tip, coverage=args.coverage)
        if not did:
            continue
        candidates += 1

        fixed = bm.build_body(mask, new_head, new_tail)
        fixed.source = CORRECTED_SOURCE
        gain_ok = fixed.passed
        would_write += 1

        rows.append({"stem": stem, "coverage": info.get("coverage"),
                     "moved_px": info.get("moved_px"), "orient": info.get("orient_by"),
                     "agrees": info.get("orient_agrees_taper"),
                     "before_len": round(body.length_px, 0), "after_len": round(fixed.length_px, 0),
                     "after_pass": fixed.passed})

        # before/after QA for the human
        orig = load_original(input_dir, stem)
        if orig is not None:
            note = f"AFTER ({fixed.length_px:.0f}px, pass={fixed.passed})"
            pair = side_by_side(bm.overlay(orig, body, mask),
                                bm.overlay(orig, fixed, mask), note)
            tag = "PASS" if fixed.passed else "FAIL"
            cv2.imwrite(str(review_dir / f"{tag}_{info.get('coverage'):.2f}_{stem}.png"), pair)

        if not args.dry_run and (gain_ok or args.force):
            jp.write_text(json.dumps(fixed.as_dict(), indent=2), encoding="utf-8")
            if orig is not None:
                (anat_dir / f"{stem}.png").write_bytes(
                    cv2.imencode(".png", bm.overlay(orig, fixed, mask))[1].tobytes())
            written += 1
            if store is not None:                       # propagate straight into the DB
                rebinned += bool(store.rebin_axis(
                    stem, fixed.as_dict() if fixed.ok else None, fixed.midline))
        elif not args.dry_run:
            skipped_fail += 1

        if args.limit and candidates >= args.limit:
            break

    if store is not None:
        store.close()

    # a CSV of exactly what was (or would be) touched
    csv = review_dir / "candidates.csv"
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write("image,coverage,moved_px,orient_by,orient_agrees_taper,"
                 "before_len,after_len,after_pass\n")
        for r in rows:
            fh.write(f"{r['stem']},{r['coverage']},{r['moved_px']},{r['orient']},"
                     f"{r['agrees']},{r['before_len']},{r['after_len']},{r['after_pass']}\n")

    print(f"\n{'image':18s} {'cov':>5} {'moved':>7} {'orient':>7} {'after':>7} pass")
    for r in sorted(rows, key=lambda r: r["coverage"] or 0):
        agree = "" if r["agrees"] is None else ("=" if r["agrees"] else "!DISAGREE")
        print(f"  {r['stem']:16s} {r['coverage']:.2f} {r['moved_px']:>7} "
              f"{r['orient']:>7}{agree:>9} {r['after_len']:>6.0f}px  {r['after_pass']}")

    print("\n" + "=" * 64)
    print(f"scanned            : {scanned}   (no mask: {no_mask})")
    print(f"correction candidates (coverage < {args.coverage:.0%}): {candidates}")
    passing = sum(1 for r in rows if r["after_pass"])
    print(f"  of those, axis now passes all gates : {passing}")
    print(f"  still failing (mask likely deficient): {candidates - passing}")
    if not args.dry_run:
        print(f"\nWROTE  : {written}" + ("" if args.force else "  (only the passing ones)"))
        print(f"skipped: {skipped_fail}  (still failing — pass --force to write anyway)")
        if store is not None:
            print(f"DB     : re-binned {rebinned} image(s) in place — no contours re-run needed")
        elif not args.no_db:
            print(f"\nNo DB found; when there is one, re-bin with:")
            print(f"  pixi run extract-spot-labels contours --input {input_dir.name}")
    else:
        print(f"\nDRY RUN — nothing was changed. Review the before/after images:")
        print(f"  {review_dir}")
        print(f"Then re-run without --dry-run to write the {passing} passing correction(s).")
    print(f"candidates.csv -> {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
