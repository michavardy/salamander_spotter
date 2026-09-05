"""Dataset transfer & incremental ingest (spec §7.9).

The extraction pipeline is **never re-run wholesale**. Data reaches ``/data`` by
two additive paths:

* :func:`transfer_dataset` — the one-time / delta transfer of an existing
  ``datasets/all_sasa_norm_*`` folder: raw images copied, ``contours.db`` copied
  verbatim (or delta-merged), ``app.duckdb`` rows derived from filenames + joins,
  ``corrections.json`` replayed. **Zero LLM calls.** Idempotent.
* :func:`ingest_photo` — one new photo: extracted individually and appended;
  nothing already in either database is touched.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings
from ..db import Database, new_id
from ..ids import (
    code_of,
    display_id,
    individual_id_of,
    is_synthetic_id,
    new_provisional_id,
)
from ..pipeline_bridge import ExtractionBridge
from .quality_ladder import LadderConfig, DEFAULT_LADDER, ladder_tier

IMPORTED_BATCH_ID = "batch_imported_pre_app"
IMPORTED_BATCH_NAME = "Imported (pre-app)"
DEFAULT_SITE_ID = "site_sasa"
DEFAULT_SITE_NAME = "Sasa"

# contours.db tables keyed by salamander_id that the app cares about (§18.3)
_CORE_CONTOUR_TABLES = ("images", "spots", "body_axis", "body_bins", "image_quality")

_CHUNK = 1 << 20


@dataclass
class TransferReport:
    dataset_path: str
    images_added: int = 0
    images_skipped: int = 0          # already present (idempotent re-run / delta)
    images_conflicted: int = 0       # present with a different checksum — left untouched
    individuals_added: int = 0
    synthetic_views_added: int = 0
    merges_applied: int = 0
    exclusions_applied: int = 0
    thumbnails_written: int = 0
    llm_calls: int = 0               # always 0 — transfer never calls a model
    contours_db_copied_verbatim: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _write_thumbnail(src: Path, dst: Path, *, max_side: int = 480) -> bool:
    """Cheap local resize to webp — the only thing computed on import (§7.9 A)."""
    try:
        from PIL import Image  # noqa: PLC0415

        try:
            import pillow_heif  # noqa: PLC0415

            pillow_heif.register_heif_opener()
        except Exception:
            pass

        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "WEBP", quality=80)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
#  first-run scaffolding                                                       #
# --------------------------------------------------------------------------- #
def ensure_default_site(db: Database) -> str:
    if not db.query_one("SELECT id FROM sites WHERE id = ?", [DEFAULT_SITE_ID]):
        db.insert("sites", {"id": DEFAULT_SITE_ID, "name": DEFAULT_SITE_NAME, "code": "SASA"})
    return DEFAULT_SITE_ID


def ensure_batch(db: Database, batch_id: str, name: str, *, status: str = "open") -> str:
    if not db.query_one("SELECT id FROM review_batches WHERE id = ?", [batch_id]):
        db.insert(
            "review_batches",
            {
                "id": batch_id,
                "name": name,
                "status": status,
                "published_at": _utcnow() if status == "published" else None,
            },
        )
    return batch_id


def open_batch(db: Database, name: str = "Autumn survey") -> str:
    """Return the single open batch, creating one if none exists (spec §9.2)."""
    row = db.query_one("SELECT id FROM review_batches WHERE status = 'open' ORDER BY created_at LIMIT 1")
    if row:
        return row["id"]
    bid = new_id("batch")
    db.insert("review_batches", {"id": bid, "name": name, "status": "open"})
    return bid


# --------------------------------------------------------------------------- #
#  A. one-time / delta dataset transfer                                        #
# --------------------------------------------------------------------------- #
def _locate_source_contours(dataset_path: Path) -> Path:
    db_dir = dataset_path / "db"
    cand = db_dir / "contours.db"
    if cand.exists():
        return cand
    hits = sorted(db_dir.glob("*.db")) if db_dir.is_dir() else []
    if not hits:
        raise FileNotFoundError(f"no contours.db under {db_dir}")
    return hits[0]


def _merge_contours(db: Database, src_contours: Path, dest_contours: Path) -> bool:
    """Bring source rows into the working ``contours.db``. First import copies the
    file verbatim; a later snapshot delta-merges only unseen ``salamander_id``s.
    Returns True if the file was copied verbatim."""
    if not dest_contours.exists():
        dest_contours.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_contours, dest_contours)
        return True

    with db.attach(src_contours, "src", read_only=True), db.attach(dest_contours, "dst", read_only=False):
        src_tables = {
            r[0]
            for r in db.query(
                "SELECT table_name FROM information_schema.tables WHERE table_catalog = 'src'"
            )
        }
        dst_tables = {
            r[0]
            for r in db.query(
                "SELECT table_name FROM information_schema.tables WHERE table_catalog = 'dst'"
            )
        }
        for table in _CORE_CONTOUR_TABLES:
            if table not in src_tables:
                continue
            if table not in dst_tables:
                db.execute(f"CREATE TABLE dst.{table} AS SELECT * FROM src.{table}")
                continue
            key = "bin_id" if table == "body_bins" else "salamander_id"
            db.execute(
                f"INSERT INTO dst.{table} SELECT s.* FROM src.{table} s "
                f"WHERE s.{key} NOT IN (SELECT {key} FROM dst.{table})"
            )
    return False


def transfer_dataset(
    db: Database,
    settings: Settings,
    dataset_path: str | Path,
    *,
    actor: str | None = None,
    ladder: LadderConfig = DEFAULT_LADDER,
) -> TransferReport:
    """Transfer ``datasets/all_sasa_norm_*`` into the data dir (spec §7.9 A).

    Safe to re-run; pointed at a newer snapshot it imports only the delta.
    """
    dataset_path = Path(dataset_path)
    raw_dir = dataset_path / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"no raw/ dir under {dataset_path}")
    src_contours = _locate_source_contours(dataset_path)

    settings.ensure_dirs()
    report = TransferReport(dataset_path=str(dataset_path))
    report.contours_db_copied_verbatim = _merge_contours(db, src_contours, settings.contours_db_path)

    site_id = ensure_default_site(db)
    ensure_batch(db, IMPORTED_BATCH_ID, IMPORTED_BATCH_NAME, status="published")

    used_codes = {r[0] for r in db.query("SELECT DISTINCT split_part(individual_id, '_', 1) FROM individuals")}
    used_codes.add("up")  # reserved provisional prefix (§6.1)

    raw_by_stem = {p.stem: p for p in raw_dir.iterdir() if p.is_file()}

    with db.attach(settings.contours_db_path, "cdb", read_only=True):
        source_rows = db.query_dicts(
            """
            SELECT i.salamander_id, i.width, i.height, i.is_synthetic, i.created_at,
                   q.n_spots, q.judged_ok,
                   q.overall_quality, q.blur_quality, q.lighting_quality,
                   q.spot_extraction_quality, q.body_extraction_quality,
                   (a.salamander_id IS NOT NULL) AS has_axis
            FROM cdb.images i
            LEFT JOIN cdb.image_quality q ON q.salamander_id = i.salamander_id
            LEFT JOIN cdb.body_axis a ON a.salamander_id = i.salamander_id
            ORDER BY i.salamander_id
            """
        )

    with db.transaction():
        for row in source_rows:
            image_id: str = row["salamander_id"]
            raw_src = raw_by_stem.get(image_id)
            if raw_src is None:
                report.warnings.append(f"{image_id}: no raw file in dataset — skipped")
                continue

            checksum = sha256_file(raw_src)
            existing = db.query_one(
                "SELECT image_id, source_checksum FROM images WHERE image_id = ?", [image_id]
            )
            if existing is not None:
                if existing["source_checksum"] and existing["source_checksum"] != checksum:
                    report.images_conflicted += 1
                    report.warnings.append(
                        f"{image_id}: already imported with a different checksum — left untouched"
                    )
                else:
                    report.images_skipped += 1
                continue

            raw_dst = settings.raw_images_dir / f"{image_id}{raw_src.suffix.lower()}"
            shutil.copy2(raw_src, raw_dst)

            # purple render only if the dataset actually shipped one — never regenerated
            for cand in (dataset_path / "purple" / f"{image_id}.png", raw_src.with_name(f"{image_id}.purple.png")):
                if cand.exists():
                    shutil.copy2(cand, settings.purple_images_dir / f"{image_id}.png")
                    break

            if _write_thumbnail(raw_dst, settings.thumb_images_dir / f"{image_id}.webp"):
                report.thumbnails_written += 1

            individual_id = individual_id_of(image_id)
            synthetic = bool(row["is_synthetic"]) or is_synthetic_id(image_id)
            created_at = row["created_at"]

            if not db.query_one(
                "SELECT individual_id FROM individuals WHERE individual_id = ?", [individual_id]
            ):
                db.insert(
                    "individuals",
                    {
                        "individual_id": individual_id,
                        "display_id": display_id(individual_id),
                        "site_id": site_id,
                        "status": "published",
                        "first_seen": created_at,
                        "last_seen": created_at,
                        "reference_image_id": None if synthetic else image_id,
                    },
                )
                used_codes.add(code_of(individual_id))
                report.individuals_added += 1

            tier = ladder_tier(
                overall_quality=row["overall_quality"],
                has_axis=bool(row["has_axis"]),
                n_spots=row["n_spots"],
                config=ladder,
            )
            db.insert(
                "images",
                {
                    "image_id": image_id,
                    "individual_id": individual_id,
                    "site_id": site_id,
                    "contributor_id": None,
                    "photographed_at": None,
                    "source_filename": raw_src.name,
                    "source_checksum": checksum,
                    "is_synthetic": synthetic,
                    "status": "confirmed",
                    "ladder_tier": tier,
                    "q_overall": row["overall_quality"],
                    "q_blur": row["blur_quality"],
                    "q_lighting": row["lighting_quality"],
                    "q_spot": row["spot_extraction_quality"],
                    "q_body": row["body_extraction_quality"],
                    "n_spots": row["n_spots"],
                    "review_batch_id": IMPORTED_BATCH_ID,
                    "origin": "import",
                },
            )
            report.images_added += 1
            if synthetic:
                report.synthetic_views_added += 1

        _apply_corrections(db, dataset_path, report, actor=actor)

        db.audit(
            action="import_dataset",
            entity="dataset",
            entity_id=dataset_path.name,
            actor=actor,
            after=report.as_dict(),
        )
        if report.images_added or report.merges_applied or report.exclusions_applied:
            db.log_activity(
                kind="dataset_import",
                summary=(
                    f"Transferred {report.images_added} images / "
                    f"{report.individuals_added} individuals from {dataset_path.name}"
                ),
                detail=json.dumps(report.as_dict()),
                actor=actor,
                ref_type="dataset",
                ref_id=dataset_path.name,
            )

    return report


def _apply_corrections(
    db: Database, dataset_path: Path, report: TransferReport, *, actor: str | None
) -> None:
    """Replay ``corrections.json`` — merges (one animal under two names) and
    exclusions (a photo that must not count). Faithfully applied to app.duckdb;
    the history survives the transfer (spec §7.9 A)."""
    path = dataset_path / "corrections.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))

    for entry in data.get("merge", []):
        keep, alias = entry.get("keep"), entry.get("alias")
        if not keep or not alias:
            continue
        if not db.query_one("SELECT 1 FROM individuals WHERE individual_id = ?", [alias]):
            continue
        db.execute(
            "UPDATE images SET individual_id = ?, rev = rev + 1 WHERE individual_id = ?", [keep, alias]
        )
        db.execute(
            "UPDATE individuals SET status = 'merged_into', merged_into = ?, rev = rev + 1 "
            "WHERE individual_id = ?",
            [keep, alias],
        )
        db.audit(
            action="merge_individual",
            entity="individual",
            entity_id=alias,
            actor=actor,
            after={"merged_into": keep, "source": entry.get("source"), "note": entry.get("note")},
        )
        report.merges_applied += 1

    for entry in data.get("exclude", []):
        sid = entry.get("sid")
        if not sid or not db.query_one("SELECT 1 FROM images WHERE image_id = ?", [sid]):
            continue
        db.execute(
            "UPDATE images SET status = 'disqualified', rev = rev + 1 WHERE image_id = ?", [sid]
        )
        db.insert(
            "extraction_corrections",
            {
                "id": new_id("exc"),
                "image_id": sid,
                "editor_id": actor,
                "diff_json": json.dumps(
                    {"action": "exclude", "reason": entry.get("reason"), "note": entry.get("note")}
                ),
                "prev_snapshot_json": None,
            },
        )
        db.audit(
            action="disqualify_image",
            entity="image",
            entity_id=sid,
            actor=actor,
            after={"reason": entry.get("reason"), "source": entry.get("source")},
        )
        report.exclusions_applied += 1


# --------------------------------------------------------------------------- #
#  B. incremental ingest — one photo appended, nothing existing touched        #
# --------------------------------------------------------------------------- #
@dataclass
class IngestResult:
    image_id: str
    status: str
    ladder_tier: str | None
    n_spots: int | None
    extraction_failed: bool
    reason: str | None = None


def ingest_photo(
    db: Database,
    settings: Settings,
    src_path: str | Path,
    *,
    bridge: ExtractionBridge,
    contributor_id: str | None = None,
    site_id: str | None = None,
    photographed_at: datetime | None = None,
    batch_id: str | None = None,
    source_filename: str | None = None,
    actor: str | None = None,
    ladder: LadderConfig = DEFAULT_LADDER,
) -> IngestResult:
    """Add one new photo (spec §7.9 B / §10.2).

    Extraction runs **only for this photo**; exactly one ``images`` row is
    inserted. Existing extractions, quality scores and individuals are untouched.
    """
    src_path = Path(src_path)
    settings.ensure_dirs()

    image_id = new_provisional_id()
    raw_dst = settings.raw_images_dir / f"{image_id}{src_path.suffix.lower()}"
    shutil.copy2(src_path, raw_dst)
    checksum = sha256_file(raw_dst)

    result = bridge.extract(image_id, raw_dst, settings.contours_db_path)
    _write_thumbnail(raw_dst, settings.thumb_images_dir / f"{image_id}.webp")

    if batch_id is None:
        batch_id = open_batch(db)

    if result.failed or result.reason:
        status, tier = "disqualified", None
    else:
        tier = ladder_tier(
            overall_quality=result.overall_quality,
            has_axis=result.has_axis,
            n_spots=result.n_spots,
            extraction_failed=result.failed,
            config=ladder,
        )
        status = "extracted"

    with db.transaction():
        db.insert(
            "images",
            {
                "image_id": image_id,
                "individual_id": None,
                "site_id": site_id,
                "contributor_id": contributor_id,
                "photographed_at": photographed_at,
                "source_filename": source_filename or src_path.name,
                "source_checksum": checksum,
                "is_synthetic": result.is_synthetic,
                "status": status,
                "ladder_tier": tier,
                "q_overall": result.overall_quality,
                "q_blur": result.quality.get("blur"),
                "q_lighting": result.quality.get("lighting"),
                "q_spot": result.quality.get("spot"),
                "q_body": result.quality.get("body"),
                "n_spots": None if result.failed else result.n_spots,
                "review_batch_id": batch_id,
                "origin": "upload",
            },
        )
        db.audit(
            action="ingest_photo",
            entity="image",
            entity_id=image_id,
            actor=actor,
            after={"status": status, "ladder_tier": tier, "reason": result.reason},
        )
        db.log_activity(
            kind="upload",
            summary=f"New sighting {image_id} ({tier or result.reason or status})",
            actor=actor,
            ref_type="image",
            ref_id=image_id,
        )

    return IngestResult(
        image_id=image_id,
        status=status,
        ladder_tier=tier,
        n_spots=None if result.failed else result.n_spots,
        extraction_failed=result.failed,
        reason=result.reason,
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
