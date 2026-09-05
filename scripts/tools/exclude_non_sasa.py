"""One-off maintenance: soft-exclude Haifa-KF images from the app's live DB.

The app's DB has no notion of "collection" (sasa vs Haifa-KF) — ``individuals.individual_id`` /
``images.individual_id`` is just the pipeline label (``ae_1``). The authoritative source for which
labels are Haifa-KF is ``images/all_sasa_norm/label_map.csv``: kf rows carry a ``KF-...``
``hebrew_name``, sasa rows carry a real Hebrew name.

This disqualifies (``images.status = 'disqualified'``) every image belonging to a kf label —
nothing is deleted, and dashboard counts / the Names roster / training snapshots already treat
disqualified images as excluded. Fully reversible with a single UPDATE back to another status.

Defaults to a dry run. Pass --apply to actually write.

NOTE: stop the running app server first — DuckDB allows only one read-write connection per file.

    pixi run python scripts/tools/exclude_non_sasa.py           # dry run
    pixi run python scripts/tools/exclude_non_sasa.py --apply   # writes
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.config import load_settings  # noqa: E402
from app.db import open_database  # noqa: E402

LABEL_MAP = REPO_ROOT / "images" / "all_sasa_norm" / "label_map.csv"


def kf_labels() -> set[str]:
    with LABEL_MAP.open(newline="", encoding="utf-8-sig") as f:
        return {row["label"] for row in csv.DictReader(f) if row["hebrew_name"].startswith("KF-")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="actually write; default is a dry run")
    ap.add_argument("--data-dir", default=None, help="override SPOTTER_DATA_DIR")
    args = ap.parse_args()

    if not LABEL_MAP.is_file():
        print(f"error: label map not found: {LABEL_MAP}", file=sys.stderr)
        return 1
    labels = sorted(kf_labels())
    if not labels:
        print(f"error: no KF-marked labels found in {LABEL_MAP}", file=sys.stderr)
        return 1

    settings = load_settings(data_dir=args.data_dir)
    try:
        db = open_database(settings.app_db_path)
    except Exception as exc:  # noqa: BLE001
        print(f"error opening {settings.app_db_path}: {exc}\n"
              f"(stop the running app server first — DuckDB allows one writer per file)",
              file=sys.stderr)
        return 1

    try:
        placeholders = ",".join("?" for _ in labels)
        rows = db.query_dicts(
            f"SELECT individual_id, count(*) AS n, "
            f"count(*) FILTER (WHERE status != 'disqualified') AS n_active "
            f"FROM images WHERE individual_id IN ({placeholders}) GROUP BY 1 ORDER BY 1",
            labels,
        )
        n_individuals = len(rows)
        n_images = sum(r["n"] for r in rows)
        n_active = sum(r["n_active"] for r in rows)
        print(f"label_map: {len(labels)} kf-marked labels")
        print(f"in the app DB: {n_individuals} individuals / {n_images} images match "
              f"({n_active} not yet disqualified)")

        if not args.apply:
            print("dry run — pass --apply to disqualify these images")
            return 0
        if n_active == 0:
            print("nothing to do — already disqualified")
            return 0

        with db.transaction():
            db.execute(
                f"UPDATE images SET status = 'disqualified', rev = rev + 1 "
                f"WHERE individual_id IN ({placeholders}) AND status != 'disqualified'",
                labels,
            )
            db.log_activity(
                kind="maintenance",
                summary=f"Excluded {n_active} non-sasa (Haifa-KF) images across "
                        f"{n_individuals} individuals",
                ref_type="maintenance", ref_id="exclude_non_sasa",
            )
        print(f"disqualified {n_active} images across {n_individuals} individuals")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
