#!/usr/bin/env python3
"""Package normalized salamander dataset(s) (images + contours.db) for distribution.

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

**Several inputs can be merged into one dataset.** Each must carry its own ``contours.db``;
their images and spot rows are concatenated into a single store. This is how Gemini-augmented
synthetic views are folded in next to the real photos: a synthetic view is named
``<label>_g<k>`` (e.g. ``aj_1_g0``), so it derives the SAME label as the real photos of that
animal — which is exactly what turns it into a training positive. The packaged ``images``
table gets an ``is_synthetic`` flag so consumers can keep synthetic rows in the training pool
and out of gallery/query (they must never be evaluated on).

Usage::

    pixi run package-dataset --input all_sasa_norm --name all_sasa_norm_2026_10_07
    pixi run package-dataset --input all_sasa_norm             # name defaults to <input>_<today>
    pixi run package-dataset --input all_sasa_norm synth_all_sasa_norm_2026_11_07 \
        --name all_sasa_norm_plus_synth_2026_07_13             # merged (--name required)
    python scripts/dataset/package_dataset.py --input all_sasa_norm --no-zip

No contours.db may be open in another process (stop any running ``extract-spot-labels``
pipeline first) so we can take a consistent snapshot.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter
from datetime import date
from math import comb
from pathlib import Path

# Reuse the pipeline's path/image helpers (stdlib-only), same trick the other CLIs use.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

from generate_spot_labels._common import (  # noqa: E402
    contours_db_for,
    list_images,
    reconfigure_utf8,
    resolve_input_dir,
)
from generate_spot_labels.binning import bin_bounds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# A Gemini-augmented view: <label>_g<k>. Mirrors spot_embedding._common.is_synthetic, kept
# literal here so the packager stays importable without the spot_embedding package.
SYNTH_RE = re.compile(r"_g\d+$")
SYNTH_SQL = "regexp_matches(salamander_id, '_g[0-9]+$')"


# --- label / stats ----------------------------------------------------------
def is_synthetic(salamander_id: str) -> bool:
    """True for a Gemini-augmented synthetic view (``<label>_g<k>``); real photos are False."""
    return bool(SYNTH_RE.search(salamander_id))


def derive_label(salamander_id: str) -> str:
    """Identity label = image id with the trailing ``_<instance>`` stripped.

    ``aj_1_2`` -> ``aj_1``; ``ca_10_5`` -> ``ca_10``. Photos sharing a label are the
    same individual. Ids without an instance suffix are returned unchanged.

    A synthetic view ``aj_1_g0`` strips its ``_g0`` the same way and lands on ``aj_1`` — so
    it joins the individual it was generated from, rather than becoming its own singleton.
    """
    return salamander_id.rsplit("_", 1)[0] if "_" in salamander_id else salamander_id


def _pair_stats(ids: list[str]) -> dict:
    """Photos-per-individual derived counts for a set of image ids."""
    per_individual = Counter(derive_label(sid) for sid in ids)
    n_images = len(ids)
    sizes = list(per_individual.values())
    return {
        "n_images": n_images,
        "n_individuals": len(per_individual),
        "singletons": sum(1 for n in sizes if n == 1),
        "multi": sum(1 for n in sizes if n > 1),
        # unordered same-individual photo pairs, sum over individuals of C(n_i, 2)
        "positive_pairs": sum(comb(n, 2) for n in sizes),
        # ordered (anchor, positive, negative): n_i*(n_i-1)*(N - n_i) per class
        "triplets": sum(n * (n - 1) * (n_images - n) for n in sizes),
        "hist": dict(sorted(Counter(sizes).items())),
    }


def compute_stats(db_path: Path) -> dict:
    """Read the snapshot DB and compute dataset statistics for the README.

    Everything is computed twice: over all rows, and over the real photos alone. The delta is
    what the synthetic views actually bought — mostly positive pairs for otherwise-singleton
    individuals, which is the entire point of generating them.
    """
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        ids = [r[0] for r in con.execute(
            "SELECT salamander_id FROM images ORDER BY salamander_id").fetchall()]
        total_spots = con.execute("SELECT count(*) FROM spots").fetchone()[0]
        spot_agg = con.execute(
            "SELECT min(n_spots), max(n_spots), avg(n_spots) FROM images").fetchone()

        # The body grid (may be absent entirely on a DB extracted before stage 1b).
        has_grid = "body_axis" in {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}
        binned = with_axis = judged_ok = judged_bad = 0
        bin_hist: dict = {}
        if has_grid:
            binned = con.execute(
                'SELECT count(*) FROM spots WHERE "bin" IS NOT NULL').fetchone()[0]
            with_axis = con.execute("SELECT count(*) FROM body_axis").fetchone()[0]
            bin_hist = {int(b): int(n) for b, n in con.execute(
                'SELECT "bin", count(*) FROM spots WHERE "bin" IS NOT NULL '
                'GROUP BY "bin" ORDER BY "bin"').fetchall()}
            if "judged_ok" in {r[1] for r in con.execute(
                    "PRAGMA table_info('body_axis')").fetchall()}:
                judged_ok = con.execute(
                    "SELECT count(*) FROM body_axis WHERE judged_ok").fetchone()[0]
                judged_bad = con.execute(
                    "SELECT count(*) FROM body_axis WHERE judged_ok = false").fetchone()[0]
    finally:
        con.close()

    real_ids = [i for i in ids if not is_synthetic(i)]
    stats = _pair_stats(ids)
    stats.update({
        "real": _pair_stats(real_ids),
        "n_synthetic": len(ids) - len(real_ids),
        "total_spots": total_spots,
        "min_spots": spot_agg[0],
        "max_spots": spot_agg[1],
        "avg_spots": spot_agg[2],
        "has_grid": has_grid,
        "binned_spots": binned,
        "images_with_axis": with_axis,
        "bin_hist": bin_hist,
        "judged_ok": judged_ok,
        "judged_bad": judged_bad,
    })
    return stats


# --- copy / snapshot / zip --------------------------------------------------
def check_no_collisions(input_dirs: list[Path]) -> None:
    """Every image id must be unique ACROSS the inputs — the id is the join key.

    Two files sharing a stem would silently overwrite one another in ``raw/`` and collide on
    the ``images`` primary key, so this is a hard error rather than a warning.
    """
    seen: dict[str, Path] = {}
    clashes: list[str] = []
    for d in input_dirs:
        for img in list_images(d):
            prev = seen.get(img.stem)
            if prev is not None:
                clashes.append(f"  {img.stem}: {prev} and {img}")
            else:
                seen[img.stem] = img
    if clashes:
        raise SystemExit("error: duplicate image ids across inputs:\n" + "\n".join(clashes))


def copy_images(input_dirs: list[Path], raw_dir: Path) -> tuple[int, int]:
    """Copy every source image (unchanged) from every input into ``raw_dir``.

    Returns (count, bytes).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    n, total_bytes = 0, 0
    for input_dir in input_dirs:
        for src in list_images(input_dir):
            dst = raw_dir / src.name
            shutil.copy2(src, dst)
            total_bytes += dst.stat().st_size
            n += 1
    return n, total_bytes


def _columns(con, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]


def _tables(con, db: str) -> list[str]:
    """The tables of an attached DB — discovered, not hardcoded, so a DB that predates the
    body-grid tables (or a future one that adds more) merges without a code change."""
    return sorted(r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_catalog = ?",
        [db]).fetchall())


def snapshot_db(srcs: list[Path], dst: Path) -> None:
    """Write one clean, WAL-free DuckDB at ``dst`` holding the rows of every DB in ``srcs``.

    The richest source (most columns) is copied wholesale — ``COPY FROM DATABASE``, which
    brings the schema — and the rest are appended row-wise. The result is a single
    checkpointed file (no ``.wal`` sidecar). Fails clearly if a source is locked by a
    running pipeline, or if two inputs claim the same ``salamander_id``.

    Input DBs of *different vintages* are tolerated: ``extract_spot_contours`` grew its
    columns over time (``mask_png`` is newer), so a column absent from one source is filled
    with NULL and reported — refusing the whole merge over a column the consumer may not
    even read would be worse. It is reported loudly because a half-populated ``mask_png``
    is a real hole in the shipped data.

    An ``is_synthetic`` flag is added to ``images`` — derived from the id, so it is present
    even for a single-input package (where it is simply all-false).
    """
    import duckdb

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    con = duckdb.connect(":memory:")
    try:
        con.execute(f"ATTACH '{dst.as_posix()}' AS dst")
        for i, src in enumerate(srcs):
            con.execute(f"ATTACH '{src.as_posix()}' AS s{i} (READ_ONLY)")

        # The schema donor is the source with the most tables+columns, so nothing that exists
        # anywhere is dropped on the floor — including body_axis/body_bins, which a DB
        # extracted before the body grid will not have at all.
        def richness(i: int) -> tuple[int, int]:
            ts = _tables(con, f"s{i}")
            return len(ts), sum(len(_columns(con, f"s{i}.{t}")) for t in ts)

        donor = max(range(len(srcs)), key=richness)
        con.execute(f"COPY FROM DATABASE s{donor} TO dst")

        tables = _tables(con, "dst")
        base = {t: _columns(con, f"dst.{t}") for t in tables}
        for i in (j for j in range(len(srcs)) if j != donor):
            have = set(_tables(con, f"s{i}"))
            for table in tables:
                if table not in have:
                    print(f"  warning: {srcs[i].parent.parent.name} has no '{table}' table — "
                          f"its images contribute no rows to it (spots stay unbinned). "
                          f"Re-run extract-spot-labels on it.", file=sys.stderr)
                    continue
                cols = base[table]
                got = _columns(con, f"s{i}.{table}")
                unknown = [c for c in got if c not in cols]
                if unknown:
                    raise SystemExit(
                        f"error: {srcs[i]} table '{table}' has column(s) {unknown} that no "
                        f"other input has.\n       Re-run extract-spot-labels so every input "
                        f"DB shares a schema.")
                missing = [c for c in cols if c not in got]
                if missing:
                    print(f"  warning: {srcs[i].parent.parent.name}/{table} is missing "
                          f"{missing} — filled with NULL. Re-run extract-spot-labels on it "
                          f"to populate.", file=sys.stderr)
                names = ", ".join(f'"{c}"' for c in cols)
                select = ", ".join(f'"{c}"' if c in got else f'NULL AS "{c}"' for c in cols)
                con.execute(f"INSERT INTO dst.{table} ({names}) "
                            f"SELECT {select} FROM s{i}.{table}")

        con.execute("ALTER TABLE dst.images ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN")
        con.execute(f"UPDATE dst.images SET is_synthetic = {SYNTH_SQL}")
    except duckdb.IOException as exc:
        raise SystemExit(
            f"error: could not open a source DB for snapshotting.\n"
            f"       Is the extract-spot-labels pipeline still running? "
            f"Close it and retry.\n       ({exc})"
        )
    except duckdb.ConstraintException as exc:
        raise SystemExit(
            f"error: duplicate salamander_id across the input DBs.\n"
            f"       Each image id must appear in exactly one input.\n       ({exc})"
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


def _synthetic_section(stats: dict) -> str:
    """The synthetic-views block — omitted entirely from an all-real dataset."""
    if not stats["n_synthetic"]:
        return ""
    real = stats["real"]
    gained_pairs = stats["positive_pairs"] - real["positive_pairs"]
    return f"""
## Synthetic views

{_fmt(stats['n_synthetic'])} of the {_fmt(stats['n_images'])} images are **Gemini-augmented
synthetic views**: new photorealistic renderings (different lighting / background / angle) of an
individual already in the set, with its spot pattern held fixed. They are named
`<label>_g<k>` (e.g. `aj_1_g0`), so they derive the **same identity label** as that animal's
real photos — which is what makes them training positives.

| | images | individuals | singletons | positive pairs |
|---|---|---|---|---|
| real photos only | {_fmt(real['n_images'])} | {_fmt(real['n_individuals'])} | {_fmt(real['singletons'])} | {_fmt(real['positive_pairs'])} |
| **with synthetic** | {_fmt(stats['n_images'])} | {_fmt(stats['n_individuals'])} | {_fmt(stats['singletons'])} | {_fmt(stats['positive_pairs'])} |

The synthetic views add **{_fmt(gained_pairs)} positive pairs** and take
**{_fmt(real['singletons'] - stats['singletons'])} individuals out of singleton status**.

**Use them for training only.** Query `images.is_synthetic` (or test the id against
`_g\\d+$`) and keep synthetic rows out of the gallery/query sets — evaluating on generated
images would measure the generator, not the matcher. Identity preservation is *not*
guaranteed: Gemini can drift a spot pattern, so these should be self-consistency filtered
(keep a view only if its spot set still matches its source individual) before training.

```sql
SELECT * FROM images WHERE NOT is_synthetic;   -- real photos: safe for gallery/query
SELECT * FROM images WHERE is_synthetic;       -- generated views: training pool only
```
"""


def _grid_section(stats: dict) -> str:
    """The body-grid block — omitted from a dataset extracted without stage 1b."""
    if not stats["has_grid"] or not stats["images_with_axis"]:
        return ""
    pct = 100.0 * stats["binned_spots"] / max(stats["total_spots"], 1)
    rows = "\n".join(
        f"| {b} | {int(bin_bounds(b)[0] * 100)}-{int(bin_bounds(b)[1] * 100)} % | "
        f"{bin_bounds(b)[2]} | {_fmt(stats['bin_hist'].get(b, 0))} |"
        for b in range(1, 9))
    return f"""
## The body grid — where on the animal each spot sits

Every spot carries **two positional labels**, a *relative* key that survives the animal being
photographed at any angle or scale, so two photos of the same individual can be compared
position-by-position:

| column | values | meaning |
|---|---|---|
| `axial_bin` | **1, 2, 3, 4** | which quarter of the body — the centre line cut into 4 equal arc-length segments. **1 is the head end.** |
| `lateral_bin` | **`left`, `right`, `overlap`** | which side of the centre line the spot is on. `overlap` means the spot's *outline* crosses the line, so it is on **neither** side. |

`left` is the image-left of a salamander whose head is at the top of the frame. It is measured
against the **body axis**, not the image axes, so it names the same flank however the animal is
rotated in the photo.

`overlap` is why the spot's outline matters and not just its centroid: a spot lying across the
spine has contour points on both sides, and calling it `left` because its centroid landed a pixel
that way would be a lie. Expect a real share of these — in one sample photo, **12 of 39 spots**
straddled the line, and the whole tail quarter did.

`bin` (1..8) is the two labels multiplied together, kept for convenience — but it is **NULL for an
overlapping spot**, which is in neither box. Prefer `axial_bin` + `lateral_bin`.

It is built from a **body axis** labelled by Gemini on each photo: a green circle on the head, a
red circle on the tail tip, and a red line connecting the two that runs down the **centre** of the
body — at every point equidistant from the left and right edges, so it divides the animal into two
equal halves. That midline is cut at 25 / 50 / 75 % of its **arc length** (so a curled tail bins by
distance *along the body*, not along a chord that cuts across it), and the midline itself splits
each quarter into a left and a right box:

| bin | along the body | side | spots |
|-----|----------------|------|-------|
{rows}

`left` / `right` are relative to the head→tail direction **as it runs in the photo** (the sign of
the 2-D cross product), not to the image axes — so they do not change when the animal is rotated
in frame.

A spot's bin comes from projecting its centroid onto the midline: `axis_t` (0 at the head, 1 at
the tail tip) picks the quarter, and the sign of `axis_offset` picks the side.

**Coverage: {_fmt(stats['images_with_axis'])} of {_fmt(stats['n_images'])} images have an axis,
{_fmt(stats['binned_spots'])} of {_fmt(stats['total_spots'])} spots are binned ({pct:.0f} %).**
Where Gemini could not produce a usable axis, that image has no `body_axis` row and its spots
have `bin IS NULL` — they are never guessed. Filter them out rather than treating NULL as a bin.

### Trust the axis: `judged_ok`

Each drawn line was graded by a second model against three questions — does it run head to tail,
is it entirely inside the body, does it bisect the animal? A line that failed was re-drawn with
that feedback. **{_fmt(stats['judged_ok'])} axes passed; {_fmt(stats['judged_bad'])} were never
accepted** and carry `body_axis.judged_ok = false` plus the reason in `judge_feedback`.

A rejected axis is still binned — the bins are just less trustworthy (typically the line drifted
off-centre, so left/right may be wrong near the flanks). For clean training data, join through
`body_axis` and keep only the accepted ones:

```sql
SELECT s.* FROM spots s
JOIN body_axis a USING (salamander_id)
WHERE a.judged_ok AND s."bin" IS NOT NULL;
```

```sql
-- spots of one photo, by position on the body
SELECT spot_id, axial_bin, lateral_bin, axis_t, area_pixels
FROM spots WHERE salamander_id = 'aj_1_2' AND axial_bin IS NOT NULL
ORDER BY axial_bin, lateral_bin;

-- how the spots of one animal distribute over the body
SELECT axial_bin, lateral_bin, count(*) FROM spots
WHERE salamander_id = 'aj_1_2' GROUP BY 1, 2 ORDER BY 1, 2;

-- the axis itself (midline_x/midline_y are the polyline, head -> tail tip)
SELECT head_x, head_y, tail_tip_x, tail_tip_y, length_px, source, judged_ok, judge_feedback
FROM body_axis WHERE salamander_id = 'aj_1_2';
```
"""


def render_readme(name: str, stats: dict, n_raw: int, raw_bytes: int) -> str:
    hist_lines = "\n".join(
        f"| {k} | {stats['hist'][k]} |" for k in stats["hist"]
    )
    raw_mb = raw_bytes / (1024 * 1024)
    avg_spots = stats["avg_spots"] or 0.0
    synthetic_section = _synthetic_section(stats)
    grid_section = _grid_section(stats)

    return f"""# {name}

Fire-salamander (*Salamandra salamandra*) dorsal (back-view) photos with per-spot contour
labels, packaged for individual re-identification / metric-learning experiments.

Each animal's back carries a unique pattern of yellow spots. This dataset pairs every photo
with machine-extracted labels — the **spots** (contour + full-frame mask + centroid + area),
a **whole-body mask** and its **head→tail centre-line axis** (so every spot also gets a body
grid `bin`), and cheap per-image **quality markers** for filtering. The filenames encode which
photos belong to the **same individual** — the supervision signal for training a matching model.

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
{grid_section}{synthetic_section}
## Database schema

`images` — one row per photo:

| column | type | notes |
|--------|------|-------|
| `salamander_id` | VARCHAR (PK) | image id / raw file stem, e.g. `aj_1_2` |
| `width`, `height` | INTEGER | image dimensions in px |
| `n_spots` | INTEGER | number of spot rows for this image |
| `source_image` | VARCHAR | original filename |
| `purple_image` | VARCHAR | intermediate magenta-spot filename |
| `body_mask_png` | BLOB | full-frame binary PNG of the **whole-body mask** (255 = animal, 0 = background) |
| `created_at` | TIMESTAMP | when the row was written |
| `is_synthetic` | BOOLEAN | true for a Gemini-augmented view (`<label>_g<k>`) — **training pool only** |

`spots` — one row per detected spot (`n_spots` per image):

| column | type | notes |
|--------|------|-------|
| `salamander_id` | VARCHAR | → `images.salamander_id` |
| `spot_id` | INTEGER | 0-based, ordered largest-area first |
| `global_centroid_x/y` | DOUBLE | spot centroid in image px |
| `area_pixels` | DOUBLE | spot area in px² |
| `local_contour` | DOUBLE[][] | ordered `[[x,y],...]` **relative to the spot centroid** |
| `mask_png` | BLOB | full-frame binary PNG mask (black except this spot, 255) |
| `axial_bin` | INTEGER | **1..4** — which quarter of the body, 1 = head end |
| `lateral_bin` | VARCHAR | **`left` / `right` / `overlap`** — side of the centre line |
| `bin` | INTEGER | 1..8 = `axial_bin` × `lateral_bin`; **NULL when `overlap`** |
| `axis_t` | DOUBLE | position along the body: 0 = head, 1 = tail tip |
| `axis_side` | VARCHAR | the *centroid's* side, always `left`/`right` (even when overlapping) |
| `axis_offset` | DOUBLE | signed perpendicular px from the midline (+ = left) |

Primary key on `spots` is `(salamander_id, spot_id)`.

`body_axis` — one row per photo that got a usable axis. The centre line is **derived from the
body mask**, not drawn: it is the line with equal body area on each side, so it bisects the
animal by construction (see the pipeline note at the bottom).

| column | type | notes |
|--------|------|-------|
| `salamander_id` | VARCHAR (PK) | → `images.salamander_id` |
| `head_x/y`, `tail_tip_x/y` | DOUBLE | snout and tail-tip anchors, in image px (0 % and 100 % of the axis) |
| `length_px` | DOUBLE | arc length of the midline |
| `midline_x`, `midline_y` | DOUBLE[] | the centre-line polyline, ordered head → tail tip |
| `left_x/y`, `right_x/y` | DOUBLE[] | the body outline split into its two halves along the midline |
| `source` | VARCHAR | `mask` (derived from the body mask) · `mask_corrected` (axis geometrically re-tipped from the mask) · `none` |
| `judged_ok` | BOOLEAN | passed every geometric quality gate (area, axis length, centre line inside the body); NULL = not evaluated |
| `judge_feedback` | VARCHAR | which gate(s) failed, if any |

`body_bins` — the 8 boxes of each photo, as drawn (one row per box):

| column | type | notes |
|--------|------|-------|
| `bin_id` | VARCHAR (PK) | `<salamander_id>_b<bin>` |
| `salamander_id` | VARCHAR | → `images.salamander_id` |
| `bin`, `quartile`, `side` | INTEGER / INTEGER / VARCHAR | box 1..8, quarter 0..3, `left`/`right` |
| `t_lo`, `t_hi` | DOUBLE | the quarter's bounds along the body (0..1) |
| `polygon_x`, `polygon_y` | DOUBLE[] | the box outline, clipped to the image |

The boxes are the grid *as drawn* (for visualisation); a spot's own `bin` is computed
analytically from `(axis_t, axis_side)`, never by point-in-polygon.

`image_quality` — one row per photo of **cheap, model-free markers for filtering** (blur,
exposure, extraction quality, shape). Two layers: absolute RAW measurements, and four 0..1
composite scores (higher = better) calibrated against this dataset's own distribution.

RAW measurements — *shape / body:*

| column | type | notes |
|--------|------|-------|
| `center_line_length_px` | DOUBLE | body length along the midline (= `body_axis.length_px`) |
| `avg_width_px` | DOUBLE | mean body width (body area ÷ length) |
| `aspect_ratio` | DOUBLE | length ÷ avg width; wild values usually mean a bad mask |
| `body_area_px` | INTEGER | body-mask pixels |
| `body_area_frac` | DOUBLE | body area ÷ frame area |
| `solidity` | DOUBLE | mask ÷ convex-hull area; low = ragged / holed / legs painted |
| `border_frac` | DOUBLE | fraction of the frame edge the mask touches (>0 = animal cropped) |
| `min_dim_px` | INTEGER | min(width, height) — a resolution floor |
| `line_inside_frac` | DOUBLE | fraction of the midline lying on the mask (1 = fully inside) |
| `curl_deg` | DOUBLE | mean turning angle along the midline (pose curl) |

RAW measurements — *exposure / spots (all measured inside the body):*

| column | type | notes |
|--------|------|-------|
| `blur_score` | DOUBLE | variance of the Laplacian; low = out of focus |
| `mean_brightness` | DOUBLE | mean V, 0..255 |
| `underexposed_frac` / `overexposed_frac` | DOUBLE | body pixels crushed to black / blown to white |
| `glare_frac` | DOUBLE | bright, desaturated pixels — specular glare that hides spots |
| `pattern_contrast` | DOUBLE | Otsu split of body brightness: mean(bright) − mean(dark) |
| `n_spots` | INTEGER | spots extracted (= `images.n_spots`) |
| `spot_area_frac` | DOUBLE | spot area ÷ body area (yellow coverage) |
| `spots_outside_frac` | DOUBLE | fraction of spot centroids OFF the body mask — direct "paint bled" alarm |
| `median_spot_area_px` | DOUBLE | typical spot size |
| `axis_source` | VARCHAR | mirror of `body_axis.source` |
| `judged_ok` | BOOLEAN | mirror of `body_axis.judged_ok` |

Composite scores — **0..1, higher = better**, dataset-calibrated:

| column | type | notes |
|--------|------|-------|
| `blur_quality` | DOUBLE | sharpness vs the dataset's median (median → 0.5) |
| `lighting_quality` | DOUBLE | penalised by clipping, glare, and off-band brightness |
| `spot_extraction_quality` | DOUBLE | penalised by bleed / too-few spots / near-zero coverage |
| `body_extraction_quality` | DOUBLE | from gates passed, solidity, line-inside, minus a border penalty |
| `overall_quality` | DOUBLE | geometric mean of the four (any single bad score drags it down) |

Filter, e.g.:

```sql
SELECT salamander_id FROM image_quality
WHERE overall_quality < 0.4 OR spots_outside_frac > 0.1 OR border_frac > 0;
```

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

Decode the whole-body mask (same frame as the raw image — an `(image, mask)` pair):

```python
blob = con.execute(
    "SELECT body_mask_png FROM images WHERE salamander_id = 'aj_1_2'"
).fetchone()[0]
body = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)  # HxW, 0/255
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

How the labels were made (all automatic, approximate, not hand-verified):

1. **Spots** — Gemini repaints the yellow spots flat magenta; OpenCV keys that colour and traces
   each spot's contour, centroid, area and per-spot mask.
2. **Body mask + axis** — Gemini paints the whole trunk-and-tail body one colour (legs excluded);
   the mask is keyed, and the head/tail tips and the centre line are then computed from its
   *shape* — the centre line is the curve with equal body area on each side, so it bisects the
   animal by construction. Axes whose tips were misplaced are geometrically re-tipped from the
   mask (`source = mask_corrected`). No line is ever hand- or model-drawn, and there is no LLM
   judge: the axis is accepted by cheap geometric gates.
3. **Bins** — the midline is cut at 25/50/75 % and split left/right, giving each spot its
   `axial_bin` (1..4) and `lateral_bin`, hence `bin` (1..8).
4. **Quality** — the `image_quality` markers above are computed from the raw photo, the mask and
   the spots (no model), for filtering.
"""


# --- CLI --------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", default=["all_sasa_norm"], metavar="DIR",
                        help="source image dir(s) (a name under images/, or a path). Several "
                             "may be given and are merged into one dataset — each needs its "
                             "own contours.db. default: all_sasa_norm")
    parser.add_argument("--db", default=None,
                        help="path to contours.db (single --input only; "
                             "default: <input>/contours/contours.db)")
    parser.add_argument("--name", default=None,
                        help="dataset / output folder name (default: <input>_<YYYY_MM_DD>; "
                             "REQUIRED when merging several inputs)")
    parser.add_argument("--out", default=None,
                        help="datasets root dir (default: <repo>/datasets)")
    parser.add_argument("--no-zip", action="store_true", help="skip building the .zip")
    args = parser.parse_args(argv)

    input_dirs = [resolve_input_dir(a) for a in args.input]
    if args.db and len(input_dirs) > 1:
        print("error: --db applies to a single --input; with several inputs each dir must "
              "carry its own contours/contours.db", file=sys.stderr)
        return 1
    if not args.name and len(input_dirs) > 1:
        print("error: --name is required when merging several inputs (no sensible default)",
              file=sys.stderr)
        return 1

    db_srcs = ([Path(args.db).resolve()] if args.db
               else [contours_db_for(d) for d in input_dirs])
    for d, db in zip(input_dirs, db_srcs):
        if not db.is_file():
            print(f"error: contours.db not found for '{d.name}': {db}\n"
                  f"       run: pixi run extract-spot-labels all --input {d.name}",
                  file=sys.stderr)
            return 1

    check_no_collisions(input_dirs)

    name = args.name or f"{input_dirs[0].name}_{date.today():%Y_%m_%d}"
    out_root = Path(args.out).resolve() if args.out else REPO_ROOT / "datasets"
    dataset_dir = out_root / name
    raw_dir = dataset_dir / "raw"
    db_dir = dataset_dir / "db"

    print(f"packaging '{name}'")
    for d, db in zip(input_dirs, db_srcs):
        print(f"  input : {d}")
        print(f"  db    : {db}")
    print(f"  output: {dataset_dir}")

    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("copying raw images ...")
    n_raw, raw_bytes = copy_images(input_dirs, raw_dir)
    print(f"  {n_raw} images ({raw_bytes / (1024 * 1024):.0f} MB) -> {raw_dir}")

    print("snapshotting contours.db ...")
    snapshot_db(db_srcs, db_dir / "contours.db")
    print(f"  -> {db_dir / 'contours.db'}")

    print("computing statistics ...")
    stats = compute_stats(db_dir / "contours.db")
    if stats["n_images"] != n_raw:
        print(f"  warning: {stats['n_images']} images in DB but {n_raw} raw files "
              f"(mismatch)", file=sys.stderr)
    synth = f", {stats['n_synthetic']} synthetic" if stats["n_synthetic"] else ""
    print(f"  {stats['n_images']} images{synth}, {stats['n_individuals']} individuals, "
          f"{stats['positive_pairs']} positive pairs, {stats['total_spots']} spots")
    if stats["has_grid"]:
        print(f"  body grid: {stats['images_with_axis']}/{stats['n_images']} images have an "
              f"axis, {stats['binned_spots']}/{stats['total_spots']} spots binned "
              f"{stats['bin_hist']}")
        if stats["judged_bad"]:
            print(f"  warning: {stats['judged_bad']} axis/axes were REJECTED by the judge "
                  f"(judged_ok = false) — their bins are less trustworthy; filter on "
                  f"body_axis.judged_ok", file=sys.stderr)
    else:
        print("  warning: no body grid in this DB — re-run extract-spot-labels to add bins",
              file=sys.stderr)

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
