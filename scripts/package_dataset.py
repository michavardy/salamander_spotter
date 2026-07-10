#!/usr/bin/env python3
"""Package a normalized salamander dataset (images + contours.db) for distribution.

Produces a self-contained, Kaggle-ready folder plus a zip::

    datasets/<name>/
        raw/            every source image, copied unchanged (originals, any ext)
        db/contours.db  a clean, WAL-free snapshot of the spot-contour DuckDB
        README.md       dataset description, label scheme, statistics, usage snippets
    datasets/<name>.zip zipped copy of the folder above

The image id used everywhere (``salamander_id`` in the DB, and the raw file stem) is the
source filename stem, e.g. ``aj_1_2``. The identity LABEL for matching is that id with the
trailing ``_<instance>`` removed (``aj_1_2`` -> ``aj_1``); see :mod:`normalize_names` for how
those names are built. Two photos that share a label are the same animal — a positive pair.

Usage::

    pixi run package-dataset --input all_sasa_norm --name all_sasa_norm_2026_10_07
    pixi run package-dataset --input all_sasa_norm             # name defaults to <input>_<today>
    python scripts/package_dataset.py --input all_sasa_norm --no-zip

The contours.db must NOT be open in another process (stop any running
``extract-spot-labels`` pipeline first) so we can take a consistent snapshot.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from datetime import date
from math import comb
from pathlib import Path

# Reuse the pipeline's path/image helpers (stdlib-only), same trick the other CLIs use.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from generate_spot_labels._common import (  # noqa: E402
    contours_db_for,
    list_images,
    reconfigure_utf8,
    resolve_input_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- label / stats ----------------------------------------------------------
def derive_label(salamander_id: str) -> str:
    """Identity label = image id with the trailing ``_<instance>`` stripped.

    ``aj_1_2`` -> ``aj_1``; ``ca_10_5`` -> ``ca_10``. Photos sharing a label are the
    same individual. Ids without an instance suffix are returned unchanged.
    """
    return salamander_id.rsplit("_", 1)[0] if "_" in salamander_id else salamander_id


def compute_stats(db_path: Path) -> dict:
    """Read the snapshot DB and compute dataset statistics for the README."""
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        ids = [r[0] for r in con.execute(
            "SELECT salamander_id FROM images ORDER BY salamander_id").fetchall()]
        total_spots = con.execute("SELECT count(*) FROM spots").fetchone()[0]
        spot_agg = con.execute(
            "SELECT min(n_spots), max(n_spots), avg(n_spots) FROM images").fetchone()
    finally:
        con.close()

    # Photos per individual, from the identity labels.
    per_individual = Counter(derive_label(sid) for sid in ids)
    n_images = len(ids)
    n_individuals = len(per_individual)
    sizes = list(per_individual.values())
    singletons = sum(1 for n in sizes if n == 1)

    # Positive pairs: unordered same-individual photo pairs, sum over individuals of C(n_i, 2).
    positive_pairs = sum(comb(n, 2) for n in sizes)
    # Triplets: ordered (anchor, positive, negative) with anchor != positive, same label for
    # anchor/positive, and a differently-labelled negative -> n_i*(n_i-1)*(N - n_i) per class.
    triplets = sum(n * (n - 1) * (n_images - n) for n in sizes)

    # Histogram: how many individuals have exactly k photos.
    hist = Counter(sizes)

    return {
        "n_images": n_images,
        "n_individuals": n_individuals,
        "singletons": singletons,
        "multi": n_individuals - singletons,
        "positive_pairs": positive_pairs,
        "triplets": triplets,
        "hist": dict(sorted(hist.items())),
        "total_spots": total_spots,
        "min_spots": spot_agg[0],
        "max_spots": spot_agg[1],
        "avg_spots": spot_agg[2],
    }


# --- copy / snapshot / zip --------------------------------------------------
def copy_images(input_dir: Path, raw_dir: Path) -> tuple[int, int]:
    """Copy every source image (unchanged) into ``raw_dir``. Returns (count, bytes)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(input_dir)
    total_bytes = 0
    for src in images:
        dst = raw_dir / src.name
        shutil.copy2(src, dst)
        total_bytes += dst.stat().st_size
    return len(images), total_bytes


def snapshot_db(src: Path, dst: Path) -> None:
    """Write a clean, WAL-free copy of the DuckDB at ``src`` to ``dst``.

    Uses ``ATTACH ... (READ_ONLY)`` + ``COPY FROM DATABASE`` so the result is a single
    checkpointed file (no ``.wal`` sidecar). Fails clearly if ``src`` is locked by a
    running pipeline.
    """
    import duckdb

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    con = duckdb.connect(":memory:")
    try:
        con.execute(f"ATTACH '{src.as_posix()}' AS src (READ_ONLY)")
        con.execute(f"ATTACH '{dst.as_posix()}' AS dst")
        con.execute("COPY FROM DATABASE src TO dst")
    except duckdb.IOException as exc:
        raise SystemExit(
            f"error: could not open {src} for snapshotting.\n"
            f"       Is the extract-spot-labels pipeline still running? "
            f"Close it and retry.\n       ({exc})"
        )
    finally:
        con.close()


def zip_dataset(dataset_dir: Path) -> Path:
    """Zip ``dataset_dir`` to a sibling ``<name>.zip``. Returns the archive path."""
    archive_base = dataset_dir.parent / dataset_dir.name
    zip_path = shutil.make_archive(
        str(archive_base), "zip",
        root_dir=str(dataset_dir.parent), base_dir=dataset_dir.name,
    )
    return Path(zip_path)


# --- README -----------------------------------------------------------------
def _fmt(n: int) -> str:
    return f"{n:,}"


def render_readme(name: str, stats: dict, n_raw: int, raw_bytes: int) -> str:
    hist_lines = "\n".join(
        f"| {k} | {stats['hist'][k]} |" for k in stats["hist"]
    )
    raw_mb = raw_bytes / (1024 * 1024)
    avg_spots = stats["avg_spots"] or 0.0

    return f"""# {name}

Fire-salamander (*Salamandra salamandra*) belly photos with per-spot contour labels,
packaged for individual re-identification / metric-learning experiments.

Each animal's belly carries a unique pattern of yellow spots. This dataset pairs every
photo with a machine-extracted set of **spots** (contour + full-frame mask + centroid +
area), and the filenames encode which photos belong to the **same individual** — the
supervision signal for training a matching model.

## Layout

```
{name}/
├── raw/            {_fmt(n_raw)} source images ({raw_mb:.0f} MB), named <salamander_id>.<ext>
├── db/
│   └── contours.db DuckDB with per-image + per-spot rows
└── README.md       this file
```

## How the labels work

A raw file stem is `<code>_<individual>_<instance>` (e.g. `aj_1_2`):

| part | meaning |
|------|---------|
| `code` | short id for the animal's source name |
| `individual` | which salamander of that name — part of the identity |
| `instance` | which photo of that individual (1..N) |

The **identity label** is the id with the trailing `_<instance>` removed:

- `aj_1_2` → label `aj_1`
- `ca_10_5` → label `ca_10`

Two photos with the **same label are the same animal** (a *positive pair*); different
labels are different animals. `aj_1` and `aj_2` are DIFFERENT individuals. The
`salamander_id` column in the DB and the `raw/` file stem are this same id, so they join
directly; the label is just `salamander_id` with the last `_N` chopped off.

## Statistics

| metric | value |
|--------|-------|
| images | {_fmt(stats['n_images'])} |
| individuals (labels) | {_fmt(stats['n_individuals'])} |
| singletons (1 photo) | {_fmt(stats['singletons'])} |
| multi-photo individuals (≥2) | {_fmt(stats['multi'])} |
| positive pairs (same individual) | {_fmt(stats['positive_pairs'])} |
| triplets (anchor, positive, negative) | {_fmt(stats['triplets'])} |
| total spots | {_fmt(stats['total_spots'])} |
| spots per image (min / avg / max) | {stats['min_spots']} / {avg_spots:.1f} / {stats['max_spots']} |

- **positive pairs** = Σ over individuals of C(nᵢ, 2) — unordered same-individual photo pairs.
- **triplets** = Σ nᵢ·(nᵢ−1)·(N−nᵢ) — every ordered (anchor, positive) within an
  individual times every negative from another individual (theoretical maximum).

Photos per individual:

| photos | # individuals |
|--------|---------------|
{hist_lines}

## Database schema

`images` — one row per photo:

| column | type | notes |
|--------|------|-------|
| `salamander_id` | VARCHAR (PK) | image id / raw file stem, e.g. `aj_1_2` |
| `width`, `height` | INTEGER | image dimensions in px |
| `n_spots` | INTEGER | number of spot rows for this image |
| `source_image` | VARCHAR | original filename |
| `purple_image` | VARCHAR | intermediate magenta-mask filename |
| `created_at` | TIMESTAMP | when the row was written |

`spots` — one row per detected spot (`n_spots` per image):

| column | type | notes |
|--------|------|-------|
| `salamander_id` | VARCHAR | → `images.salamander_id` |
| `spot_id` | INTEGER | 0-based, ordered largest-area first |
| `global_centroid_x/y` | DOUBLE | spot centroid in image px |
| `area_pixels` | DOUBLE | spot area in px² |
| `local_contour` | DOUBLE[][] | ordered `[[x,y],...]` **relative to the spot centroid** |
| `mask_png` | BLOB | full-frame binary PNG mask (black except this spot, 255) |

Primary key on `spots` is `(salamander_id, spot_id)`.

## Access the labels (DuckDB)

```python
import duckdb

db_path = "/kaggle/input/{name}/db/contours.db"
con = duckdb.connect(database=db_path, read_only=True)

images = con.execute("SELECT * FROM images LIMIT 5").df()
spots  = con.execute("SELECT * FROM spots  LIMIT 5").df()
print(images)
print(spots)

# spots for one photo, largest first
one = con.execute(
    "SELECT spot_id, area_pixels, global_centroid_x, global_centroid_y "
    "FROM spots WHERE salamander_id = 'aj_1_2' ORDER BY spot_id"
).df()
```

Decode a spot mask and its contour:

```python
import cv2, numpy as np

row = con.execute(
    "SELECT mask_png, local_contour, global_centroid_x, global_centroid_y "
    "FROM spots WHERE salamander_id = 'aj_1_2' AND spot_id = 0"
).fetchone()

mask = cv2.imdecode(np.frombuffer(row[0], np.uint8), cv2.IMREAD_GRAYSCALE)  # HxW, 0/255
local = np.array(row[1])                        # [[x,y],...] relative to centroid
absolute = local + np.array([row[2], row[3]])   # back to image pixel coordinates
```

## Access the raw images

Raw files share the DB id as their stem, so join on it:

```python
from pathlib import Path
import cv2

raw = Path("/kaggle/input/{name}/raw")
img_path = next(raw.glob("aj_1_2.*"))    # id -> file (extension may be .jpg/.png/...)
img = cv2.imread(str(img_path))          # BGR HxWx3
```

---

Spot labels were extracted automatically by a Gemini "magenta-spot" segmentation pipeline
followed by OpenCV contour tracing; they are approximate, not hand-verified.
"""


# --- CLI --------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="all_sasa_norm",
                        help="source image dir (a name under images/, or a path). "
                             "default: all_sasa_norm")
    parser.add_argument("--db", default=None,
                        help="path to contours.db (default: <input>/contours/contours.db)")
    parser.add_argument("--name", default=None,
                        help="dataset / output folder name "
                             "(default: <input>_<YYYY_MM_DD>)")
    parser.add_argument("--out", default=None,
                        help="datasets root dir (default: <repo>/datasets)")
    parser.add_argument("--no-zip", action="store_true", help="skip building the .zip")
    args = parser.parse_args(argv)

    input_dir = resolve_input_dir(args.input)
    db_src = Path(args.db).resolve() if args.db else contours_db_for(input_dir)
    if not db_src.is_file():
        print(f"error: contours.db not found: {db_src}", file=sys.stderr)
        return 1

    name = args.name or f"{input_dir.name}_{date.today():%Y_%m_%d}"
    out_root = Path(args.out).resolve() if args.out else REPO_ROOT / "datasets"
    dataset_dir = out_root / name
    raw_dir = dataset_dir / "raw"
    db_dir = dataset_dir / "db"

    print(f"packaging '{name}'")
    print(f"  input : {input_dir}")
    print(f"  db    : {db_src}")
    print(f"  output: {dataset_dir}")

    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("copying raw images ...")
    n_raw, raw_bytes = copy_images(input_dir, raw_dir)
    print(f"  {n_raw} images ({raw_bytes / (1024 * 1024):.0f} MB) -> {raw_dir}")

    print("snapshotting contours.db ...")
    snapshot_db(db_src, db_dir / "contours.db")
    print(f"  -> {db_dir / 'contours.db'}")

    print("computing statistics ...")
    stats = compute_stats(db_dir / "contours.db")
    if stats["n_images"] != n_raw:
        print(f"  warning: {stats['n_images']} images in DB but {n_raw} raw files "
              f"(mismatch)", file=sys.stderr)
    print(f"  {stats['n_images']} images, {stats['n_individuals']} individuals, "
          f"{stats['positive_pairs']} positive pairs, {stats['total_spots']} spots")

    print("writing README.md ...")
    (dataset_dir / "README.md").write_text(
        render_readme(name, stats, n_raw, raw_bytes), encoding="utf-8")

    if not args.no_zip:
        print("zipping ...")
        zip_path = zip_dataset(dataset_dir)
        print(f"  -> {zip_path} ({zip_path.stat().st_size / (1024 * 1024):.0f} MB)")

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
