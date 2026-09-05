#!/usr/bin/env python3
"""Merge a Roboflow COCO export (haifa_v2) into ``all_sasa_norm``.

The haifa dataset carries each salamander's field identity in the *original*
filename (preserved by Roboflow in ``image.extra.name``), e.g.::

    KF_25-II-089.jpg   ->   site=KF, year=2025, series=II, individual #089

The **individual** is ``(series, number)``; the **year** is the capture occasion,
so a salamander re-photographed in 2023/2024/2025 is ONE animal with three photos.
That maps cleanly onto ``all_sasa_norm``'s ``<code>_<individual>_<instance>`` scheme:

* every distinct ``(series, number)`` individual gets its own fresh two-letter
  ``code`` (unique per individual, so ``individual`` index is always ``1``); if the
  two-letter space fills up, a random three-letter code is used instead;
* the label is ``<code>_1``;
* each of that individual's photos (ordered by year) becomes an ``instance``:
  ``<code>_1_1.jpg``, ``<code>_1_2.jpg``, ...

Images are copied **whole** (the scale coin stays in frame). Every split
(train / valid / test) is folded into the flat output dir. New rows are appended to
``filename_map.csv`` and ``label_map.csv`` (both backed up to ``*.bak`` first); the
KF field id is recorded in the ``hebrew_name`` column for traceability, and the
per-photo original name (``KF_25-II-089.jpg``) is kept in ``old_name``.

Nothing in the input is modified. Existing ``all_sasa_norm`` rows are preserved
verbatim; new rows are appended. Re-running is refused unless ``--force`` is given
(which first removes any previously merged KF rows + their copied files).

    pixi run merge-haifa --dry-run     # ALWAYS first: the plan, no writes
    pixi run merge-haifa               # copy files + update the maps
    pixi run merge-haifa --force       # rebuild the KF portion from scratch
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]   # scripts/dataset/ -> repo root
INPUT_DIR = REPO_ROOT / "images" / "haifa_v2"
OUTPUT_DIR = REPO_ROOT / "images" / "all_sasa_norm"

COCO_NAME = "_annotations.coco.json"
ROBOFLOW_PREFIX = re.compile(r"^[0-9a-f]{8}-")
# KF_<yy>-<series>-<number> ; the leading KF_ may or may not carry the underscore.
KF_ID = re.compile(r"^KF_?(\d{2})-([IVX]+)-(\d+)", re.IGNORECASE)


def orig_name(image: dict) -> str:
    """The pre-Roboflow field filename for a COCO image entry."""
    extra = image.get("extra") or {}
    name = extra.get("name") or image.get("file_name", "")
    return ROBOFLOW_PREFIX.sub("", name)


def parse_kf(name: str):
    """(year, series, number_str, number_int) from a KF filename, or None."""
    stem = re.sub(r"\.[^.]+$", "", ROBOFLOW_PREFIX.sub("", name))
    m = KF_ID.match(stem)
    if not m:
        return None
    yy, series, num = m.group(1), m.group(2).upper(), m.group(3)
    return yy, series, num, int(num)


def two_letter_codes():
    for a in string.ascii_lowercase:
        for b in string.ascii_lowercase:
            yield a + b


def read_csv_rows(path: Path):
    """(header, rows) from a utf-8-sig CSV, or (None, []) if absent."""
    if not path.exists():
        return None, []
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return None, []
    return rows[0], rows[1:]


def write_csv(path: Path, header, rows) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=INPUT_DIR,
                        help=f"Roboflow export root (default: {INPUT_DIR})")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help=f"destination normalized dir (default: {OUTPUT_DIR})")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the plan without copying files or editing maps")
    parser.add_argument("--force", action="store_true",
                        help="if KF rows already exist, delete them (+ their files) and rebuild")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for 3-letter code fallback (default: 0)")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    in_dir: Path = args.input
    out_dir: Path = args.output
    fmap_path = out_dir / "filename_map.csv"
    lmap_path = out_dir / "label_map.csv"

    if not in_dir.is_dir():
        print(f"error: input directory not found: {in_dir}", file=sys.stderr)
        return 1
    if not out_dir.is_dir():
        print(f"error: output directory not found: {out_dir}", file=sys.stderr)
        return 1

    # --- gather every haifa image across all splits --------------------------
    # photo = (year, series, number_str, number_int, source_path, orig_name)
    photos: list[tuple[str, str, str, int, Path, str]] = []
    skipped = 0
    split_dirs = sorted(p for p in in_dir.iterdir() if p.is_dir() and (p / COCO_NAME).exists())
    if not split_dirs:
        print(f"error: no split dirs with {COCO_NAME} under {in_dir}", file=sys.stderr)
        return 1
    for split in split_dirs:
        coco = json.loads((split / COCO_NAME).read_text(encoding="utf-8"))
        for image in coco.get("images", []):
            src = split / image["file_name"]
            orig = orig_name(image)
            parsed = parse_kf(orig)
            if parsed is None:
                print(f"warning: skipping unparseable id: {orig!r}", file=sys.stderr)
                skipped += 1
                continue
            if not src.exists():
                print(f"warning: source file missing: {src}", file=sys.stderr)
                skipped += 1
                continue
            yy, series, num_s, num_i = parsed
            photos.append((yy, series, num_s, num_i, src, orig))

    if not photos:
        print("error: no parseable haifa images found", file=sys.stderr)
        return 1

    # --- group photos into individuals: (series, number) ---------------------
    individuals: dict[tuple[str, int], list] = defaultdict(list)
    for yy, series, num_s, num_i, src, orig in photos:
        individuals[(series, num_i)].append((yy, src, orig, num_s))
    ordered_keys = sorted(individuals)  # series then number, deterministic

    # --- existing maps: keep non-KF rows, learn used codes -------------------
    f_header, f_rows = read_csv_rows(fmap_path)
    l_header, l_rows = read_csv_rows(lmap_path)
    f_header = f_header or ["old_name", "new_name", "label", "hebrew_name"]
    l_header = l_header or ["label", "code", "individual_index", "hebrew_name", "n_instances"]

    def is_kf_frow(r):  # old_name looks like a KF field id
        return bool(r) and r[0].upper().startswith("KF")

    def is_kf_lrow(r):  # hebrew_name column carries the KF field id
        return len(r) >= 4 and r[3].upper().startswith("KF-")

    existing_kf = [r for r in f_rows if is_kf_frow(r)]
    if existing_kf and not args.dry_run and not args.force:
        print(f"error: {len(existing_kf)} KF rows already in {fmap_path.name}; "
              f"re-run with --force to rebuild them.", file=sys.stderr)
        return 1

    kept_frows = [r for r in f_rows if not is_kf_frow(r)]
    kept_lrows = [r for r in l_rows if not is_kf_lrow(r)]
    used_codes = {r[1] for r in kept_lrows if len(r) >= 2}
    existing_new_names = {r[1] for r in kept_frows if len(r) >= 2}

    # --- assign a fresh code per individual ----------------------------------
    rng = random.Random(args.seed)
    two_letter = (c for c in two_letter_codes() if c not in used_codes)

    def next_code() -> str:
        for c in two_letter:
            used_codes.add(c)
            return c
        while True:  # two-letter space exhausted -> random three-letter
            c = "".join(rng.choice(string.ascii_lowercase) for _ in range(3))
            if c not in used_codes:
                used_codes.add(c)
                return c

    # new rows, plus (src -> dst) copy plan
    new_frows: list[list[str]] = []
    new_lrows: list[list[str]] = []
    copy_plan: list[tuple[Path, Path]] = []
    n_3letter = 0
    sample: list[tuple[str, str]] = []
    for series, num_i in ordered_keys:
        code = next_code()
        if len(code) == 3:
            n_3letter += 1
        label = f"{code}_1"
        kf_id = f"KF-{series}-{individuals[(series, num_i)][0][3]}"  # e.g. KF-II-089
        items = sorted(individuals[(series, num_i)], key=lambda t: (t[0], t[1].name))
        for inst, (yy, src, orig, _num_s) in enumerate(items, start=1):
            suffix = src.suffix.lower()
            new_name = f"{label}_{inst}{suffix}"
            if new_name in existing_new_names or (out_dir / new_name).exists():
                print(f"error: name collision on {new_name}", file=sys.stderr)
                return 1
            new_frows.append([orig, new_name, label, kf_id])
            copy_plan.append((src, out_dir / new_name))
            if len(sample) < 20:
                sample.append((orig, new_name))
        new_lrows.append([label, code, "1", kf_id, str(len(items))])

    # --- report --------------------------------------------------------------
    sizes = Counter(len(v) for v in individuals.values())
    n_ind = len(ordered_keys)
    singles = sizes.get(1, 0)
    print(f"input : {in_dir}  ({len(split_dirs)} splits: {', '.join(p.name for p in split_dirs)})")
    print(f"output: {out_dir}")
    print(f"photos: {len(new_frows)}   individuals (new labels): {n_ind}   skipped: {skipped}")
    print(f"codes : {n_ind - n_3letter} two-letter + {n_3letter} three-letter "
          f"(existing codes preserved: {len(used_codes) - n_ind})")
    print("\nphotos-per-individual:")
    for k in sorted(sizes):
        print(f"  {k:>2} photo(s): {sizes[k]:>3} individuals")
    print(f"\nsingletons (1 photo): {singles}   multi (>=2, usable pairs): {n_ind - singles}")

    print("\nsample renames:")
    for old, new in sample:
        print(f"  {old}  ->  {new}")
    if len(new_frows) > len(sample):
        print(f"  ... (+{len(new_frows) - len(sample)} more)")

    if args.dry_run:
        print("\n[dry-run] no files copied, no maps changed.")
        if existing_kf:
            print(f"[dry-run] note: {len(existing_kf)} KF rows already present "
                  f"-> a real run needs --force.")
        return 0

    # --- apply: back up maps, (force) remove old KF files, copy, rewrite maps -
    for p in (fmap_path, lmap_path):
        if p.exists():
            shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))

    if args.force and existing_kf:
        removed = 0
        for r in existing_kf:
            f = out_dir / r[1]
            if f.exists():
                f.unlink()
                removed += 1
        print(f"[force] removed {removed} previously merged KF image files")

    for src, dst in copy_plan:
        shutil.copy2(src, dst)

    write_csv(fmap_path, f_header, kept_frows + new_frows)
    write_csv(lmap_path, l_header, kept_lrows + new_lrows)

    print(f"\ncopied {len(copy_plan)} files to {out_dir}")
    print(f"filename_map.csv: {len(kept_frows)} kept + {len(new_frows)} new = {len(kept_frows) + len(new_frows)} rows")
    print(f"label_map.csv   : {len(kept_lrows)} kept + {len(new_lrows)} new = {len(kept_lrows) + len(new_lrows)} rows")
    print("backups written: filename_map.csv.bak, label_map.csv.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
