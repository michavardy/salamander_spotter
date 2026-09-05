#!/usr/bin/env python3
"""Dataset refinement + spot-match curation in a little local web app — one individual at a time.

Opens a browser page showing every photo of one animal side by side. Accept or reject each image,
mark duplicates, toggle whether it is training data, reject bad machine-proposed spot matches, draw
the ones the machine missed, record misses, and queue re-extractions. Every edit is written straight
to::

    artifacts/preprocessing/<dataset>/review.json

Individuals are ordered **lowest mean quality first**, so the pass starts where the edits are, and it
opens at the entry after the last one you edited — stop and resume at will.

    pixi run preprocess-review                          # dataset: all_sasa_norm_2026_23_07
    pixi run preprocess-review --dataset all_sasa_norm_2026_19_07
    pixi run preprocess-review --port 9000 --no-browser
    pixi run preprocess-review --input all_sasa_norm     # the pipeline DB instead of a packaged one
    pixi run preprocess-export                           # legacy interesting_spots.json + queue CSVs
    pixi run preprocess-review interest                  # build the interest cache and exit

**Replaces ``interesting-spot-selector``**: the existing ~8.7k clicks in
``artifacts/interesting_spots/<folder>/interesting_spots.json`` are imported on first run, and
``preprocess-export`` keeps writing that same file, so nothing downstream changes.

Spec: docs/preprocessing_ui.md
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from generate_spot_labels import reconfigure_utf8  # noqa: E402
from preprocessing import ReviewApp, run  # noqa: E402

DEFAULT_DATASET = "all_sasa_norm_2026_23_07"


def folder_for(dataset: str) -> str:
    """``all_sasa_norm_2026_23_07`` -> ``all_sasa_norm`` (the image folder it was packaged from)."""
    return re.sub(r"_\d{4}_\d{2}_\d{2}$", "", dataset)


def resolve_sources(args: argparse.Namespace) -> tuple[Path, list[Path], str, str]:
    """(db_path, photo_dirs, dataset, images_folder) from --dataset / --input / --db.

    Prefers the **packaged dataset** DB: it carries ``is_synthetic`` and ``spot_embeddings``, so the
    UI reads exactly what the experiments read. Falls back to the pipeline DB under
    ``images/<folder>/contours/`` when there is no packaged dataset (no embeddings there, so no
    machine arcs — the page says so).
    """
    if args.input:
        folder = Path(args.input).name
        images_dir = REPO_ROOT / "images" / folder
        if not images_dir.is_dir():
            images_dir = Path(args.input).resolve()
        db = Path(args.db) if args.db else images_dir / "contours" / "contours.db"
        return db, [images_dir, images_dir / "purple"], args.dataset or folder, folder

    dataset = args.dataset or DEFAULT_DATASET
    folder = args.images_folder or folder_for(dataset)
    ds_dir = REPO_ROOT / "datasets" / dataset
    db = Path(args.db) if args.db else ds_dir / "db" / "contours.db"
    photo_dirs = [ds_dir / "raw", REPO_ROOT / "images" / folder,
                  REPO_ROOT / "images" / folder / "purple"]
    return db, photo_dirs, dataset, folder


def build_app(args: argparse.Namespace) -> ReviewApp:
    db, photo_dirs, dataset, folder = resolve_sources(args)
    store = Path(args.store) if args.store else \
        REPO_ROOT / "artifacts" / "preprocessing" / dataset / "review.json"
    if args.legacy in ("none", "off"):
        legacy = None
    elif args.legacy:
        legacy = Path(args.legacy)
    else:
        legacy = REPO_ROOT / "artifacts" / "interesting_spots" / folder / "interesting_spots.json"
    print(f"preprocessing review  ({dataset})")
    return ReviewApp(db_path=db, photo_dirs=photo_dirs, store_path=store,
                     dataset=dataset, images_folder=folder, legacy_labels_path=legacy)


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="serve",
                        choices=["serve", "export", "interest"],
                        help="serve the app (default), write the derived artifacts, or build the "
                             "interest cache and exit")
    parser.add_argument("--dataset", default=None, help=f"packaged dataset (default {DEFAULT_DATASET})")
    parser.add_argument("--input", "--output", dest="input", default=None,
                        help="use the pipeline DB of this image folder instead of a packaged dataset")
    parser.add_argument("--images-folder", default=None,
                        help="image folder name used for the legacy interesting-spots path")
    parser.add_argument("--db", default=None, help="explicit contours.db path")
    parser.add_argument("--store", default=None,
                        help="explicit review.json path (default artifacts/preprocessing/<dataset>/)")
    parser.add_argument("--legacy", default=None,
                        help="interesting_spots.json to import from and export to; 'none' to leave "
                             "the legacy store alone entirely (export unions, never replaces)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--no-interest", action="store_true",
                        help="skip the one-off distinctiveness build (spots get a flat fill)")
    args = parser.parse_args(argv)

    try:
        app = build_app(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        if args.command == "export":
            result = app.export()
            for path in result["written"]:
                print(f"  wrote {path}")
            print(f"  interesting spots: {result['interesting_labels']} labels over "
                  f"{result['interesting_images']} images")
            return 0
        if args.command == "interest":
            app.ensure_interest(background=False)
            return 0 if app.interest_status()["state"] == "ready" else 1
        run(app, host=args.host, port=args.port, open_browser=not args.no_browser,
            build_interest=not args.no_interest)
        return 0
    finally:
        if args.command != "serve":
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
