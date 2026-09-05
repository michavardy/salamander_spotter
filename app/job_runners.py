"""Register the background job runners on a :class:`~app.worker.Worker` (spec §5.3)."""

from __future__ import annotations

from pathlib import Path

from .bridges import Bridges
from .config import Settings
from .db import Database
from .settings_store import SettingsStore
from .worker import JobContext, Worker


def register_all(worker: Worker, *, db: Database, settings: Settings, store: SettingsStore, bridges: Bridges) -> None:
    def import_dataset(ctx: JobContext) -> dict:
        from .services import ingest

        report = ingest.transfer_dataset(
            db, settings, Path(ctx.params["dataset_path"]), actor=ctx.params.get("actor")
        )
        return report.as_dict()

    def extract_photo(ctx: JobContext) -> dict:
        from .services import ingest

        res = ingest.ingest_photo(
            db, settings, Path(ctx.params["path"]), bridge=bridges.extraction,
            batch_id=ctx.params.get("batch_id"), actor=ctx.params.get("actor"),
            ladder=store.ladder_config(),
        )
        return res.__dict__

    def match_sighting(ctx: JobContext) -> dict:
        from .services import decision, matching

        match_id = matching.run_match(
            db, store, query_image_id=ctx.params["image_id"], bridge=bridges.matching,
            contours_db_path=settings.contours_db_path,
        )
        triage = decision.triage(db, store, image_id=ctx.params["image_id"], match_id=match_id)
        return {"match_id": match_id, "triage": triage.__dict__}

    def training_run(ctx: JobContext) -> dict:
        from .services import training

        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = settings.logs_dir / f"{ctx.job_id}.log"
        result = training.run_training(
            db, store, bridge=bridges.training, models_dir=settings.models_dir,
            contours_db_path=settings.contours_db_path,
            trigger=ctx.params.get("trigger", "manual"), actor=ctx.params.get("actor"),
            progress=ctx.progress, job_id=ctx.job_id, log_path=log_path,
        )
        return {"run_id": result.run_id, "snapshot_id": result.snapshot_id,
                "models": result.models, "promotion": result.promotion}

    def backup_job(ctx: JobContext) -> dict:
        from . import backup

        dest = Path(ctx.params.get("dest") or (settings.data_dir.parent / "spotter-backups"))
        # live: the server is the one running this job, so use its own connection
        # to snapshot app.duckdb/contours.db instead of a locked raw file copy.
        res = backup.live_backup(settings, db, dest)
        return {"archive": res.archive, "bytes": res.bytes, "files": res.files}

    def export_full_job(ctx: JobContext) -> dict:
        from . import backup

        archive = Path(ctx.params["archive_path"])
        backup.live_export_full(settings, db, archive)
        return {"archive": str(archive), "bytes": archive.stat().st_size}

    worker.register("import_dataset", import_dataset)
    worker.register("extract_photo", extract_photo)
    worker.register("match_sighting", match_sighting)
    worker.register("training_run", training_run)
    worker.register("backup", backup_job)
    worker.register("export_full", export_full_job)
