#!/usr/bin/env python3
"""Normalize salamander image filenames to ASCII — one label per individual.

Hebrew names like ``סלמנדרה אינגי-1 02.jpg`` are copied to the output dir with clean,
ASCII-only names of the form ``<code>_<individual>_<instance>.jpg`` (e.g. ``aj_1_2``).

Naming
------
A raw stem is ``<hebrew-name>-<individual_index> <instance_index>`` (the instance part is
often absent). The pieces map to::

    <code>       two-letter ASCII code for the Hebrew name (unique per distinct name)
    <individual> the individual index K = which salamander of that name. The LABEL is
                 ``<code>_<individual>`` (e.g. aj_1); aj_1 and aj_2 are DIFFERENT animals.
    <instance>   which photo of that individual, re-numbered 1..N per individual

So two photos of individual ``aj_1`` become ``aj_1_1`` and ``aj_1_2``, and an individual
with a single photo still gets an explicit ``_1`` (``aj_3`` -> ``aj_3_1``). More photos
per individual = more positive pairs for the matching model.

Rules
-----
* Output names contain only ``[a-z0-9_]`` (no Hebrew, no spaces).
* Two-letter code = **position-based** transliteration of the name's initials: each
  Hebrew letter maps to the Latin letter at the same alphabet position (alef=1->a,
  bet=2->b, gimel=3->c, ...); first letter of each of the first two words, or the first
  two letters of a single-word name (``ג-א`` -> ``ca``). Distinct names that collide on a
  code are bumped to the next free code, so every distinct name gets a unique code.
* The individual index K is taken verbatim from the raw name (it identifies the animal);
  the instance index is re-numbered 1..N per individual, so raw gaps / zero-padding /
  duplicates (``01``, missing, repeated) never leak into the label.

Input/output directories are hardcoded (relative to the repo root) but may be overridden
on the command line. Nothing in the input is modified; files are copied. A
``filename_map.csv`` and ``label_map.csv`` are written to the output directory.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import string
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# --- Hardcoded locations (relative to this script) --------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]   # scripts/dataset/ -> repo root
INPUT_DIR = REPO_ROOT / "images" / "sasa"
OUTPUT_DIR = REPO_ROOT / "images" / "sasa_norm"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# --- Hebrew -> Latin, by alphabet position ----------------------------------
# 22 base letters map to a..v; final forms fold onto their base letter.
_HEBREW_ORDER = "אבגדהוזחטיכלמנסעפצקרשת"
POSITION_MAP: dict[str, str] = {
    ch: string.ascii_lowercase[i] for i, ch in enumerate(_HEBREW_ORDER)
}
FINAL_FORMS = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
POSITION_MAP.update({fin: POSITION_MAP[base] for fin, base in FINAL_FORMS.items()})

# The common leading word "סלמנדרה" (salamandra) is dropped from every name.
SALAMANDER_WORD = "סלמנדרה"

# Word separators inside a name: whitespace, ASCII hyphen, Hebrew maqaf.
_WORD_SPLIT = re.compile(r"[\s\-־]+")


def strip_format_chars(text: str) -> str:
    """Remove bidi/zero-width control characters (Unicode category Cf)."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def translit_letter(ch: str) -> str | None:
    """Position-based Latin letter for a single Hebrew letter, else None."""
    return POSITION_MAP.get(ch)


def name_to_code(name: str) -> str:
    """Two-letter (pre-collision) code for a Hebrew name.

    First letter of each of the first two words; a single word uses its first
    two letters. Non-Hebrew characters are skipped.
    """
    words = [w for w in _WORD_SPLIT.split(name) if w]
    letters: list[str] = []
    if len(words) >= 2:
        for word in words:
            for ch in word:
                lat = translit_letter(ch)
                if lat is not None:
                    letters.append(lat)
                    break
            if len(letters) >= 2:
                break
    # Fall back to consecutive letters (single word, or too few word-initials).
    if len(letters) < 2:
        letters = []
        for ch in name:
            lat = translit_letter(ch)
            if lat is not None:
                letters.append(lat)
            if len(letters) >= 2:
                break
    if not letters:
        raise ValueError(f"no Hebrew letters found in name: {name!r}")
    if len(letters) == 1:
        letters.append(letters[0])  # degenerate single-letter name
    return "".join(letters[:2])


def bump_code(base: str, used: set[str]) -> str:
    """Return a free two-letter code, advancing from ``base``."""
    if base not in used:
        return base
    first = base[0]
    for second in string.ascii_lowercase:
        cand = first + second
        if cand not in used:
            return cand
    # First-letter row exhausted: scan the whole 2-letter space.
    for a in string.ascii_lowercase:
        for b in string.ascii_lowercase:
            cand = a + b
            if cand not in used:
                return cand
    raise RuntimeError("exhausted all two-letter codes")


def parse_stem(stem: str) -> tuple[str, str]:
    """Split a filename stem into (hebrew_name, numeric_tail).

    The name is everything up to the first digit (trailing separators trimmed);
    the tail is the index and any sub-index/scan suffix from the first digit on.
    """
    stem = strip_format_chars(stem)
    if stem.startswith(SALAMANDER_WORD):
        stem = stem[len(SALAMANDER_WORD):]
    m = re.search(r"\d", stem)
    if not m:
        raise ValueError(f"no index digit found in stem: {stem!r}")
    name = stem[: m.start()].strip(" -־\t")
    tail = stem[m.start():]
    if not name:
        raise ValueError(f"empty name parsed from stem: {stem!r}")
    return name, tail


def split_indices(tail: str) -> tuple[int, int | None]:
    """Split an index tail into ``(individual_index, instance_index)``.

    The first number is the salamander's individual index (the label); the second, if
    present, is its photo/instance index. ``"1 02"`` -> ``(1, 2)``; ``"3"`` -> ``(3, None)``;
    ``"14 03"`` -> ``(14, 3)``. The instance is only used for ordering — the final instance
    number is re-counted 1..N per individual in :func:`main`.
    """
    nums = re.findall(r"\d+", tail)
    if not nums:
        raise ValueError(f"no index digits in tail: {tail!r}")
    return int(nums[0]), (int(nums[1]) if len(nums) > 1 else None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_DIR,
                        help=f"source dir (default: {INPUT_DIR})")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help=f"destination dir (default: {OUTPUT_DIR})")
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

    files = sorted(
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        print(f"error: no image files found in {in_dir}", file=sys.stderr)
        return 1

    # Parse everything first: (path, hebrew_name, individual_index K, instance_index M).
    parsed: list[tuple[Path, str, int, int | None]] = []
    for path in files:
        try:
            name, tail = parse_stem(path.stem)
            k, m = split_indices(tail)
        except ValueError as exc:
            print(f"warning: skipping {path.name}: {exc}", file=sys.stderr)
            continue
        parsed.append((path, name, k, m))

    # One unique 2-letter code per distinct Hebrew name (bump collisions to a free code),
    # so two different names can never share a label.
    unique_names = sorted({name for _, name, _, _ in parsed})
    used: set[str] = set()
    name_to_code_final: dict[str, str] = {}
    for name in unique_names:
        code = bump_code(name_to_code(name), used)
        used.add(code)
        name_to_code_final[name] = code

    # Group photos by individual = (code, K), then re-number instances 1..N per individual
    # (ordered by the raw instance index, then filename, for determinism).
    groups: dict[tuple[str, int], list[tuple[int | None, Path, str]]] = defaultdict(list)
    for path, name, k, m in parsed:
        groups[(name_to_code_final[name], k)].append((m, path, name))

    rows: list[tuple[str, str, str, str]] = []          # (old, new, label, hebrew_name)
    individuals: list[tuple[str, str, int, str, int]] = []  # (label, code, K, name, n)
    for (code, k), items in groups.items():
        items.sort(key=lambda t: (t[0] if t[0] is not None else -1, t[1].name))
        label = f"{code}_{k}"
        for inst, (_m, path, name) in enumerate(items, start=1):
            rows.append((path.name, f"{label}_{inst}{path.suffix.lower()}", label, name))
        individuals.append((label, code, k, items[0][2], len(items)))

    rows.sort(key=lambda r: r[1])
    individuals.sort()

    # Report: the thing that matters for matching is photos-per-individual.
    n_ind = len(individuals)
    sizes = Counter(n for *_, n in individuals)
    singles = sizes.get(1, 0)
    print(f"input : {in_dir}")
    print(f"output: {out_dir}")
    print(f"files : {len(rows)}   individuals (labels): {n_ind}   distinct names: {len(unique_names)}")
    print("\nphotos-per-individual:")
    for k in sorted(sizes):
        print(f"  {k:>2} photo(s): {sizes[k]:>3} individuals")
    print(f"\nsingletons (1 photo): {singles}   multi (>=2, usable pairs): {n_ind - singles}")

    if args.dry_run:
        print("\n[dry-run] no files copied. First 40 renames:")
        for old, new, _label, _name in rows[:40]:
            print(f"  {old}  ->  {new}")
        if len(rows) > 40:
            print(f"  ... (+{len(rows) - 40} more)")
        return 0

    # Copy files and write maps.
    out_dir.mkdir(parents=True, exist_ok=True)
    for old, new, _label, _name in rows:
        shutil.copy2(in_dir / old, out_dir / new)

    with (out_dir / "filename_map.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["old_name", "new_name", "label", "hebrew_name"])
        w.writerows(rows)

    with (out_dir / "label_map.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "code", "individual_index", "hebrew_name", "n_instances"])
        for label, code, k, name, n in individuals:
            w.writerow([label, code, k, name, n])

    print(f"\ncopied {len(rows)} files to {out_dir}")
    print(f"wrote {out_dir / 'filename_map.csv'} and {out_dir / 'label_map.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
