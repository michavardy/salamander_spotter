#!/usr/bin/env python3
"""Re-purple the images whose spot extraction scores poorly.

Stage 1 sometimes paints the spots with shading / lighting instead of a flat key colour;
stage 2's key then shatters each shaded spot into a swarm of tiny fragments. This tool scores
every image's CURRENT spots (adaptive key, straight off the existing purple), and for the ones
that score poorly it re-sends the ORIGINAL photo to Gemini with a stronger "flat magenta, no
shadow" instruction, on the SECOND-MOST-EXPENSIVE model in the ladder.

A re-roll is only kept when it actually scores better (the new purple is decoded and re-scored
before anything is overwritten), so a bad reroll can never regress an image. When kept, the new
purple replaces the old one and that image's rows in contours.db are re-extracted (reusing the
existing anatomy + body mask — the geometry does not change, only the spot colour).

    # ALWAYS dry-run first: see which images qualify and which model, spend nothing
    pixi run repurple --input all_sasa_norm --dry-run

    # the real run (BILLED — calls Gemini for each flagged image)
    pixi run repurple --input all_sasa_norm

    # fix stage-1 BLEED instead of fragmentation (magenta flooded off the spots); scored by
    # comparing each purple to its original, so it needs no DB and can run before `contours`:
    pixi run repurple --input all_sasa_norm --select bleed --dry-run
    pixi run repurple --input all_sasa_norm --select bleed --workers 8   # scan + re-rolls in parallel

Default scoring is the fragmentation heuristic stage 2 flags with (`spot_qc`): an image is "not
pretty good" when it has many spots (``--min-spots``) that are almost all tiny (median area
below ``--min-median-area``). Loosen/tighten those to widen or narrow the re-roll set. With
``--select bleed`` the selector is instead the off-spot magenta fraction (``--select-max-bleed``);
that mode only rewrites the purple PNG (no DB touch), so run ``contours`` afterwards.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from generate_spot_labels import extract_spot_contours as stage2  # noqa: E402
from generate_spot_labels import llm_anatomy as stage1b  # noqa: E402
from generate_spot_labels import llm_spot_segmentation as stage1  # noqa: E402
from generate_spot_labels import reconfigure_utf8  # noqa: E402
from generate_spot_labels._common import (  # noqa: E402
    contours_db_for,
    purple_dir_for,
    resolve_input_dir,
)

# Cheapest -> most expensive. The .env keeps no persistent ladder (it is normally passed on the
# CLI), so we default to the documented one and pick the second rung from the top.
DEFAULT_LADDER = ["gemini-2.5-flash-image", "gemini-3.1-flash-image", "gemini-3-pro-image"]

# Re-roll instruction: the canonical inpaint prompt plus a blunt reminder aimed squarely at the
# failure that got the image flagged in the first place — shaded / textured magenta.
REPURPLE_PROMPT = stage1.PROMPT + """

RE-DRAW NOTICE — a previous attempt at THIS image failed because the magenta was shaded,
textured or uneven, which broke the spots into fragments. This time the magenta MUST be a
single, perfectly flat #FF00FF: the exact same colour value across the entire spot, with NO
shadow, NO gradient, NO highlight, NO grain and NO darkening toward the edges. A spot is one
solid uniform fill or it is wrong."""


def load_mask(anatomy_dir: Path, stem: str, shape) -> np.ndarray | None:
    """Binary (0/255) body mask sized to ``shape`` (h, w), or None if there is none."""
    mp = anatomy_dir / f"{stem}_mask.png"
    if not mp.is_file():
        return None
    bm = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    if bm is None:
        return None
    if bm.shape != shape:
        bm = cv2.resize(bm, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(bm > 127, np.uint8(255), np.uint8(0))


def score_bgr(bgr: np.ndarray, body_mask: np.ndarray | None,
              min_spots: int, min_median: float) -> dict:
    """Adaptive-key spot QC for one decoded purple image, with the flag thresholds applied here
    (so the CLI can move the bar without touching stage 2's defaults)."""
    spots = stage2.extract_spots(bgr, body_mask=body_mask, key_mode="adaptive")
    n = len(spots)
    med = float(np.median([s["area_pixels"] for s in spots])) if n else 0.0
    return {"n_spots": n, "median_area": round(med, 1),
            "flagged": n >= min_spots and med < min_median}


def pick_model(ladder: list[str], override: str | None) -> str:
    """The chosen re-roll model: ``override`` if given, else the second-most-expensive rung
    (one below the top). A single-entry ladder just uses that entry."""
    if override:
        return override
    if not ladder:
        raise SystemExit("error: --ladder is empty; give at least one model or use --model")
    return ladder[-2] if len(ladder) >= 2 else ladder[-1]


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="all_sasa_norm",
                    help="input image dir (a name under images/, or a path)")
    ap.add_argument("--select", choices=("fragment", "bleed"), default="fragment",
                    help="which images to re-roll. 'fragment' (default): spots came out "
                         "shattered/tiny (needs no DB — scored live off the purple). 'bleed': "
                         "magenta flooded OFF the spots (scored by comparing each purple to its "
                         "original); use this to fix stage-1 bleed before running contours.")
    ap.add_argument("--select-max-bleed", type=float, default=stage1.DEFAULT_MAX_BLEED,
                    help=f"'--select bleed' only: flag a purple whose off-spot magenta fraction "
                         f"exceeds this (default {stage1.DEFAULT_MAX_BLEED})")
    ap.add_argument("--only", default=None, metavar="CSV",
                    help="restrict the candidate set to the image names/stems listed in this "
                         "file (one per line, '#' comments ok) — e.g. only the newly merged "
                         "images, so old/curated purples are never touched")
    ap.add_argument("--ladder", default=",".join(DEFAULT_LADDER),
                    help="comma-separated model ladder, cheapest first "
                         f"(default: {','.join(DEFAULT_LADDER)})")
    ap.add_argument("--model", default=None,
                    help="force a specific model (default: second-most-expensive in --ladder)")
    ap.add_argument("--min-spots", type=int, default=stage2.QC_MIN_SPOTS,
                    help=f"only images with at least this many spots can be flagged "
                         f"(default {stage2.QC_MIN_SPOTS})")
    ap.add_argument("--min-median-area", type=float, default=stage2.QC_MEDIAN_AREA,
                    help=f"flag when the median spot area is below this in px^2 "
                         f"(default {stage2.QC_MEDIAN_AREA})")
    ap.add_argument("--max-attempts", type=int, default=2,
                    help="Gemini draws per flagged image on the chosen model (default 2)")
    ap.add_argument("--max-bleed", type=float, default=stage1.DEFAULT_MAX_BLEED,
                    help=f"reject a re-roll draft that floods magenta off the spots beyond this "
                         f"fraction (default {stage1.DEFAULT_MAX_BLEED})")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N flagged images (for a small paid trial)")
    ap.add_argument("--dry-run", action="store_true",
                    help="score + list the flagged images and the model, but call nothing")
    ap.add_argument("--workers", type=int, default=1, metavar="N",
                    help="process N images concurrently in a thread pool — parallelises BOTH the "
                         "scan and the Gemini re-rolls (default 1 = serial). Each Gemini call is "
                         "independent and every disk/DB write stays on the main thread, so it is "
                         "safe; start modest (4-8) and watch for 429s.")
    args = ap.parse_args(argv)

    input_dir = resolve_input_dir(args.input)
    purple_dir = purple_dir_for(input_dir)
    anatomy_dir = stage1b.anatomy_dir_for(input_dir)
    if not purple_dir.is_dir():
        print(f"error: no purple/ in {input_dir} — run stage 1 first", file=sys.stderr)
        return 1
    ladder = [m.strip() for m in args.ladder.split(",") if m.strip()]
    model = pick_model(ladder, args.model)

    purples = sorted(p for p in purple_dir.iterdir() if p.suffix.lower() == ".png")
    if not purples:
        print(f"error: no purple PNGs in {purple_dir}", file=sys.stderr)
        return 1

    if args.only:
        only_path = Path(args.only)
        if not only_path.is_file():
            only_path = input_dir / args.only
        if not only_path.is_file():
            print(f"error: --only file not found: {args.only}", file=sys.stderr)
            return 1
        wanted = {Path(ln.strip()).stem for ln in only_path.read_text(encoding="utf-8").splitlines()
                  if ln.strip() and not ln.lstrip().startswith("#")}
        before = len(purples)
        purples = [p for p in purples if p.stem in wanted]
        print(f"--only {only_path.name}: restricted to {len(purples)} of {before} purples "
              f"({len(wanted)} names listed)")
        if not purples:
            print("error: --only left no purples to score", file=sys.stderr)
            return 1

    select_bleed = args.select == "bleed"

    # --- Pass 1: score every candidate purple, collect the flagged ---
    # Pure read + numpy/cv2 (no shared writes), so it parallelises trivially with --workers.
    def _score(pp: Path):
        try:
            bgr = cv2.imread(str(pp), cv2.IMREAD_COLOR)
            if bgr is None:
                return pp, None, f"unreadable purple {pp.name}"
            if select_bleed:
                src = _original_for(input_dir, pp.stem)
                orig = cv2.imread(str(src), cv2.IMREAD_COLOR) if src is not None else None
                if orig is None:
                    return pp, None, f"no original for {pp.stem} — can't score bleed"
                bl = stage1.bleed_fraction(orig, bgr)
                return pp, {"bleed": round(bl, 3), "flagged": bl > args.select_max_bleed}, None
            bm = load_mask(anatomy_dir, pp.stem, bgr.shape[:2])
            return pp, score_bgr(bgr, bm, args.min_spots, args.min_median_area), None
        except Exception as exc:  # one odd image must not kill the whole scan
            return pp, None, f"scoring {pp.name}: {exc}"

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_score, purples))  # map preserves order -> deterministic
    else:
        results = [_score(pp) for pp in purples]

    flagged: list[tuple[Path, dict]] = []
    for pp, qc, warn in results:
        if warn is not None:
            print(f"  warn: {warn}", file=sys.stderr)
            continue
        if qc["flagged"]:
            flagged.append((pp, qc))

    if select_bleed:
        print(f"scored {len(purples)} image(s); {len(flagged)} bleeding "
              f"(off-spot magenta > {args.select_max_bleed:.2f})")
    else:
        print(f"scored {len(purples)} image(s); {len(flagged)} below the bar "
              f"(>= {args.min_spots} spots & median < {args.min_median_area:.0f}px)")
    print(f"ladder      : {' -> '.join(ladder)}  (cheapest -> priciest)")
    print(f"re-roll model: {model}  (second-most-expensive)" if not args.model
          else f"re-roll model: {model}  (forced via --model)")
    for pp, qc in flagged:
        if select_bleed:
            print(f"  FLAG {pp.stem:12s}  bleed {qc['bleed']:.2f}")
        else:
            print(f"  FLAG {pp.stem:12s}  {qc['n_spots']:4d} spots  median {qc['median_area']:.0f}px")

    if args.dry_run:
        print(f"\ndry run — nothing sent. {len(flagged)} image(s) would be re-purpled on {model}.")
        return 0
    if not flagged:
        print("nothing to do.")
        return 0

    todo = flagged[:args.limit] if args.limit is not None else flagged

    # --- Pass 2: re-roll each flagged image, keep the new purple only if it scores better ---
    segmenter = stage1.GeminiSpotSegmenter(
        model=model, escalate_models=[], escalate_attempts=0)  # single model, no further climb
    # 'bleed' selection is a pre-contours fix (the images may not be in the DB yet), so it only
    # overwrites the purple PNG — run `contours` afterwards to fold the cleaned purple into the DB.
    store = None if select_bleed else stage2.ContourStore(contours_db_for(input_dir))
    improved, kept_old, errored = [], [], []
    still_bad: list[str] = []
    n_todo = len(todo)

    def _draw(item):
        """Worker: the BILLED Gemini re-roll only — no disk/DB writes, so it is thread-safe."""
        pp, old_qc = item
        src = _original_for(input_dir, pp.stem)
        if src is None:
            return pp, old_qc, None, None, "no original image found"
        try:
            res = segmenter.segment_file(
                src, max_attempts=args.max_attempts, max_bleed=args.max_bleed,
                prompt=REPURPLE_PROMPT)
            return pp, old_qc, src, res, None
        except Exception as exc:
            return pp, old_qc, src, None, f"gemini error: {exc}"

    def _handle(n, packed):
        """Main thread ONLY: score the draft, decide, and do every write (purple PNG + DB)."""
        pp, old_qc, src, res, err = packed
        if err is not None:
            print(f"  [{n}/{n_todo}] {pp.stem}: {err} — skipped", file=sys.stderr)
            errored.append(pp.stem)
            return

        # bleed mode: keep the re-roll only if its off-spot bleed is lower (no DB touch).
        if select_bleed:
            new_bleed = res.bleed
            if not (new_bleed == new_bleed and new_bleed < old_qc["bleed"]):  # nan-safe
                shown = f"{new_bleed:.2f}" if new_bleed == new_bleed else "n/a"
                print(f"  [{n}/{n_todo}] {pp.stem}: bleed {old_qc['bleed']:.2f} -> {shown} "
                      f"— no gain, KEPT old")
                kept_old.append(pp.stem)
                still_bad.append(pp.stem)
                return
            pp.write_bytes(res.png)
            tag = "clean" if new_bleed <= args.select_max_bleed else "still bleeding"
            print(f"  [{n}/{n_todo}] {pp.stem}: bleed {old_qc['bleed']:.2f} -> {new_bleed:.2f} "
                  f"[{tag}] — purple updated")
            improved.append(pp.stem)
            if new_bleed > args.select_max_bleed:
                still_bad.append(pp.stem)
            return

        # fragment mode: keep if the spots come out bigger/unflagged, then update the DB.
        new_bgr = cv2.imdecode(np.frombuffer(res.png, np.uint8), cv2.IMREAD_COLOR)
        bm = load_mask(anatomy_dir, pp.stem, new_bgr.shape[:2]) if new_bgr is not None else None
        new_qc = (score_bgr(new_bgr, bm, args.min_spots, args.min_median_area)
                  if new_bgr is not None else {"n_spots": 0, "median_area": 0.0, "flagged": True})
        better = (not new_qc["flagged"] and old_qc["flagged"]) or \
                 (new_qc["median_area"] > old_qc["median_area"])
        if not better:
            print(f"  [{n}/{n_todo}] {pp.stem}: {old_qc['n_spots']}->{new_qc['n_spots']} spots, "
                  f"median {old_qc['median_area']:.0f}->{new_qc['median_area']:.0f}px "
                  f"(bleed {res.bleed:.2f}) — no gain, KEPT old")
            kept_old.append(pp.stem)
            still_bad.append(pp.stem)
            return
        pp.write_bytes(res.png)
        anat = stage1b.load_anatomy(anatomy_dir, pp.stem)
        mask_path = anatomy_dir / f"{pp.stem}_mask.png"
        rec = stage2.process_purple_image(
            pp, store, source_image=src.name, anatomy=anat, key_mode="adaptive",
            body_mask_png=mask_path.read_bytes() if mask_path.is_file() else None)
        rq = rec["qc"]
        tag = "clean" if not rq["flagged"] else "still flagged"
        print(f"  [{n}/{n_todo}] {pp.stem}: -> {rq['n_spots']} spots (median {rq['median_area']:.0f}px, "
              f"bleed {res.bleed:.2f}) [{tag}] — purple + db updated")
        improved.append(pp.stem)
        if rq["flagged"]:
            still_bad.append(pp.stem)

    print(f"\nre-purpling {n_todo} image(s) on {model} (BILLED)…"
          + (f"   workers: {args.workers}" if args.workers > 1 else ""))
    try:
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_draw, item) for item in todo]
                for n, fut in enumerate(as_completed(futs), 1):
                    _handle(n, fut.result())
        else:
            for n, item in enumerate(todo, 1):
                _handle(n, _draw(item))
    finally:
        if store is not None:
            store.close()

    if still_bad:
        csv = stage2.write_flagged_spots_csv(purple_dir, sorted(set(still_bad)))
        print(f"\n{len(set(still_bad))} image(s) still flagged after re-roll -> {csv}")
    print(f"\ndone: {len(improved)} improved, {len(kept_old)} kept old (no gain), "
          f"{len(errored)} errored, of {len(todo)} attempted.")
    return 1 if errored else 0


def _original_for(input_dir: Path, stem: str) -> Path | None:
    """The source photo for a purple stem — same stem, any supported extension, in input_dir."""
    from generate_spot_labels._common import IMAGE_EXTS
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"):
        cand = input_dir / f"{stem}{ext}"
        if cand.is_file() and ext in IMAGE_EXTS:
            return cand
    hit = [p for p in input_dir.iterdir()
           if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_EXTS]
    return hit[0] if hit else None


if __name__ == "__main__":
    raise SystemExit(main())
