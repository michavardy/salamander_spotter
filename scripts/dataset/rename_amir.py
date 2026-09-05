#!/usr/bin/env python3
"""Flatten Amir's raw photo dump into one directory of ASCII names.

``images/amir/raw/`` holds one subdirectory per salamander (Hebrew names carrying the
animal's number, e.g. ``סלמנדרה 053``), each with photos and Google-Takeout sidecar
``-מידע.json`` metadata files. This copies every image into a single output directory as::

    amr_<dir_number>_<index>.<ext>        e.g. amr_053_01.jpg, amr_053_02.jpg

Naming
------
``<dir_number>`` is the first run of digits in the subdirectory name, kept **verbatim**
so the zero padding matches the folder (``סלמנדרה 053`` -> ``053``, ``סלמנדרה 08`` -> ``08``,
``00-למיון ישן`` -> ``00``). Two subdirectories that yield the same number are an error —
the number is the label, so a collision would merge two animals.

``<index>`` is 1..N per subdirectory, zero-padded to 2, assigned in natural filename order
(so ``(6)`` sorts before ``(12)``). Extensions are lowercased and otherwise kept as-is.

Rules
-----
* Only image files are copied; videos and any other non-image file are skipped and reported.
* Nothing in ``raw/`` is modified — files are **copied**, not moved.
* Each image's Takeout sidecar JSON is located best-effort (Takeout truncates the sidecar
  name, so it is matched as the longest json stem that prefixes ``<image name>-מידע``) and
  recorded in the map. With ``--with-json`` the sidecar is copied too, as ``<new_stem>.json``.
* ``rename_map.csv`` is written to the output directory: new name, source subdirectory,
  original filename, and the sidecar json (if any).

Input/output directories are hardcoded (relative to the repo root) but may be overridden
on the command line.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

# --- Hardcoded locations (relative to this script) --------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]   # scripts/dataset/ -> repo root
INPUT_DIR = REPO_ROOT / "images" / "amir" / "raw"
OUTPUT_DIR = REPO_ROOT / "images" / "amir" / "processes" / "renamed"

PREFIX = "amr"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic"}

# Google Takeout writes "<original file name>-מידע.json" ("מידע" = metadata), truncating the
# whole name to a fixed length — so the sidecar stem is only a PREFIX of what it should be.
SIDECAR_WORD = "מידע"

_DIGITS = re.compile(r"\d+")


def strip_format_chars(text: str) -> str:
    """Remove bidi/zero-width control characters (Unicode category Cf)."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def dir_number(name: str) -> str:
    """First run of digits in a subdirectory name, verbatim (padding preserved)."""
    m = _DIGITS.search(strip_format_chars(name))
    if not m:
        raise ValueError(f"no digits in directory name: {name!r}")
    return m.group(0)


def natural_key(name: str) -> list:
    """Sort key that orders embedded numbers numerically ("(6)" before "(12)")."""
    parts = _DIGITS.split(name)
    nums = _DIGITS.findall(name)
    key: list = []
    for i, part in enumerate(parts):
        key.append((0, part.lower()))
        if i < len(nums):
            key.append((1, int(nums[i])))
    return key


def find_sidecar(image: Path, jsons: list[Path]) -> Path | None:
    """Best-effort Takeout sidecar for ``image``: the longest json stem prefixing it.

    Takeout truncates ``<image>-מידע.json``, so the sidecar stem is a prefix of the
    untruncated target. Longest match wins; a tie means the truncation is genuinely
    ambiguous and no sidecar is claimed.
    """
    target = f"{image.name}-{SIDECAR_WORD}"
    cands = [j for j in jsons if target.startswith(strip_format_chars(j.stem))]
    if not cands:
        return None
    best = max(len(strip_format_chars(j.stem)) for j in cands)
    top = [j for j in cands if len(strip_format_chars(j.stem)) == best]
    return top[0] if len(top) == 1 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=INPUT_DIR,
                        help=f"source dir of per-salamander subdirectories (default: {INPUT_DIR})")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help=f"destination dir (default: {OUTPUT_DIR})")
    parser.add_argument("--with-json", action="store_true",
                        help="also copy each image's Takeout sidecar as <new_stem>.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the plan without copying files")
    args = parser.parse_args()

    # Print Hebrew readably regardless of the console's default codepage.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    in_dir: Path = args.input
    out_dir: Path = args.output

    if not in_dir.is_dir():
        print(f"error: input directory not found: {in_dir}", file=sys.stderr)
        return 1

    subdirs = sorted((p for p in in_dir.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name))
    if not subdirs:
        print(f"error: no subdirectories found in {in_dir}", file=sys.stderr)
        return 1

    # Number every subdirectory first: a collision means two folders claim one animal.
    numbers: dict[str, list[str]] = defaultdict(list)
    numbered: list[tuple[str, Path]] = []
    for sub in subdirs:
        try:
            num = dir_number(sub.name)
        except ValueError as exc:
            print(f"warning: skipping directory: {exc}", file=sys.stderr)
            continue
        numbers[num].append(sub.name)
        numbered.append((num, sub))

    clashes = {n: names for n, names in numbers.items() if len(names) > 1}
    if clashes:
        for num, names in sorted(clashes.items()):
            print(f"error: number {num} claimed by {len(names)} directories: {names}", file=sys.stderr)
        return 1

    rows: list[tuple[str, str, str, str]] = []   # (new_name, subdir, old_name, sidecar_json)
    copies: list[tuple[Path, Path]] = []         # (src, dst)
    empty: list[str] = []                        # subdirs with no images
    skipped: list[tuple[str, str]] = []          # (subdir, filename) non-image files
    no_sidecar: list[str] = []                   # new names whose sidecar was not found

    for num, sub in sorted(numbered, key=lambda t: natural_key(t[0])):
        files = [p for p in sub.iterdir() if p.is_file()]
        images = sorted((p for p in files if p.suffix.lower() in IMAGE_EXTS),
                        key=lambda p: natural_key(p.name))
        jsons = [p for p in files if p.suffix.lower() == ".json"]
        others = [p for p in files if p.suffix.lower() not in IMAGE_EXTS and p.suffix.lower() != ".json"]
        skipped.extend((sub.name, p.name) for p in sorted(others, key=lambda p: p.name))

        if not images:
            empty.append(f"{sub.name}  ({len(jsons)} json, {len(others)} other)")
            continue

        for idx, img in enumerate(images, start=1):
            stem = f"{PREFIX}_{num}_{idx:02d}"
            new_name = f"{stem}{img.suffix.lower()}"
            sidecar = find_sidecar(img, jsons)
            rows.append((new_name, sub.name, img.name, sidecar.name if sidecar else ""))
            copies.append((img, out_dir / new_name))
            if sidecar is None:
                no_sidecar.append(new_name)
            elif args.with_json:
                copies.append((sidecar, out_dir / f"{stem}.json"))

    per_dir = defaultdict(int)
    for _new, subname, _old, _js in rows:
        per_dir[subname] += 1

    print(f"input : {in_dir}")
    print(f"output: {out_dir}")
    print(f"subdirs: {len(numbered)}   with images: {len(per_dir)}   images: {len(rows)}")
    print("\nimages per subdirectory:")
    for num, sub in sorted(numbered, key=lambda t: natural_key(t[0])):
        n = per_dir.get(sub.name, 0)
        print(f"  {PREFIX}_{num:<4} {n:>3} image(s)   {sub.name}")
    if empty:
        print(f"\nsubdirectories with no images ({len(empty)}) — nothing copied from these:")
        for line in empty:
            print(f"  {line}")
    if skipped:
        print(f"\nskipped non-image files ({len(skipped)}):")
        for subname, fname in skipped:
            print(f"  {subname}/{fname}")
    if no_sidecar:
        print(f"\nno Takeout sidecar json matched ({len(no_sidecar)}): {', '.join(no_sidecar)}")

    if args.dry_run:
        print("\n[dry-run] no files copied. First 40 renames:")
        for new, subname, old, _js in rows[:40]:
            print(f"  {subname}/{old}  ->  {new}")
        if len(rows) > 40:
            print(f"  ... (+{len(rows) - 40} more)")
        return 0

    if not rows:
        print("\nnothing to copy.")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in copies:
        shutil.copy2(src, dst)

    map_path = out_dir / "rename_map.csv"
    with map_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["new_name", "source_dir", "old_name", "sidecar_json"])
        w.writerows(rows)

    print(f"\ncopied {len(copies)} file(s) ({len(rows)} image(s)) to {out_dir}")
    print(f"wrote {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
