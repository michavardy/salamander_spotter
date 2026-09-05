#!/usr/bin/env python3
"""Fold a synthetic-views dir INTO a master image dir, bumping the ``_g<k>`` index so nothing
collides — image, purple, anatomy and mask together, so no stage is re-billed.

The build generates augmented views into ``images/synth_<name>/`` and normally keeps them a
separate dataset that is merged only at packaging. This instead copies them straight into the
master dir alongside the real photos (and any synthetic views already there), giving each one the
next free ``_g<k>`` index for its individual. So a fresh ``ca_43_g0`` becomes ``ca_43_g2`` when
``ca_43_g0`` and ``ca_43_g1`` already exist; a view for an individual with no synthetic views yet
simply keeps ``_g0``.

Every derived artefact travels with the image under the new name, so the folded view is as
complete as the ones already there and NOTHING has to be regenerated::

    <into>/<new>.png                 the view itself
    <into>/purple/<new>.png          stage-1 magenta
    <into>/anatomy/<new>.json        stage-1b geometry
    <into>/anatomy/<new>_mask.png    the body mask
    <into>/anatomy/<new>.png         the QA overlay

    pixi run fold-synth --synth synth_all_sasa_norm_2026_15_07 --into all_sasa_norm --dry-run
    pixi run fold-synth --synth synth_all_sasa_norm_2026_15_07 --into all_sasa_norm

Default applies; ``--dry-run`` prints the full rename map and copies nothing. Afterwards the new
spots still need binning into the master DB (free), then package the master dir alone::

    pixi run extract-spot-labels contours --input all_sasa_norm
    pixi run package-dataset --input all_sasa_norm --name <dataset>
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from generate_spot_labels import reconfigure_utf8  # noqa: E402
from generate_spot_labels._common import list_images, resolve_input_dir  # noqa: E402

G_RE = re.compile(r"^(.+)_g(\d+)$")


def label_of(stem: str) -> str:
    """`ca_43_g0` -> `ca_43`; a real `ca_43_1` -> `ca_43` too."""
    m = G_RE.match(stem)
    return m.group(1) if m else stem.rsplit("_", 1)[0] if "_" in stem else stem


def existing_g_indices(into_dir: Path) -> dict[str, int]:
    """label -> highest existing _g index among the .png views already in the master dir."""
    top: dict[str, int] = {}
    for p in into_dir.glob("*_g*.png"):
        m = G_RE.match(p.stem)
        if m:
            top[m.group(1)] = max(top.get(m.group(1), -1), int(m.group(2)))
    return top


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synth", required=True, help="the synthetic dir under images/ to fold in")
    p.add_argument("--into", default="all_sasa_norm", help="the master dir to fold into")
    p.add_argument("--dry-run", action="store_true",
                   help="print the rename map and copy nothing")
    args = p.parse_args(argv)

    synth_dir = resolve_input_dir(args.synth)
    into_dir = resolve_input_dir(args.into)
    if synth_dir == into_dir:
        print("error: --synth and --into are the same dir", file=sys.stderr)
        return 1

    images = list_images(synth_dir)
    if not images:
        print(f"error: no images in {synth_dir}", file=sys.stderr)
        return 1

    next_idx = {lbl: g + 1 for lbl, g in existing_g_indices(into_dir).items()}
    (into_dir / "purple").mkdir(exist_ok=True)
    (into_dir / "anatomy").mkdir(exist_ok=True)

    print(f"folding {len(images)} views from {synth_dir.name}  ->  {into_dir.name}  "
          f"({'DRY RUN' if args.dry_run else 'APPLY'})\n")

    folded = bumped = missing = 0
    for src in sorted(images):
        stem = src.stem
        lbl = label_of(stem)
        idx = next_idx.get(lbl, 0)
        next_idx[lbl] = idx + 1              # reserve, so two synth views of one label don't clash
        new_stem = f"{lbl}_g{idx}"
        if new_stem != stem:
            bumped += 1

        # (source, destination) for the view and each derived artefact
        moves = [
            (src, into_dir / f"{new_stem}{src.suffix}"),
            (synth_dir / "purple" / f"{stem}.png", into_dir / "purple" / f"{new_stem}.png"),
            (synth_dir / "anatomy" / f"{stem}.json", into_dir / "anatomy" / f"{new_stem}.json"),
            (synth_dir / "anatomy" / f"{stem}_mask.png",
             into_dir / "anatomy" / f"{new_stem}_mask.png"),
            (synth_dir / "anatomy" / f"{stem}.png", into_dir / "anatomy" / f"{new_stem}.png"),
        ]
        have = [(s, d) for s, d in moves if s.is_file()]
        missing += sum(1 for s, _ in moves if not s.is_file())
        clash = [d for _, d in have if d.exists()]
        if clash:
            print(f"  SKIP {stem}: target {clash[0].name} already exists (unexpected)",
                  file=sys.stderr)
            continue

        tag = f"{stem:18s} -> {new_stem}" + ("   (bumped)" if new_stem != stem else "")
        if bumped <= 12 or new_stem != stem:
            print(f"  {tag}  [{len(have)} files]")
        if not args.dry_run:
            for s, d in have:
                shutil.copy2(s, d)
        folded += 1

    print("\n" + "=" * 60)
    print(f"views folded : {folded}")
    print(f"index bumped : {bumped}  (the rest kept their _g index — no clash)")
    if missing:
        print(f"missing artefacts skipped: {missing}  (a view lacking purple/anatomy)")
    if args.dry_run:
        print("\nDRY RUN — nothing was copied.")
    else:
        print(f"\nDone. Now bin the new spots into the master DB (free), then package:")
        print(f"  pixi run extract-spot-labels contours --input {into_dir.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
