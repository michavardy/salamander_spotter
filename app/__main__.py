"""``app`` — the project's single console entrypoint (spec §5.4).

    app serve            run the server (what the container runs)
    app migrate          apply DuckDB migrations to the data dir
    app import-dataset   one-time / delta dataset transfer (spec 7.9 A)
    app import-model     register an already-trained checkpoint (spec 7.6)
    app backup           dated tar of the data dir to a second location (spec 2.2)
    app restore <a>      rebuild a fresh data dir from one archive
    app export --full    move a whole data dir between hosts (spec 2.3)
    app import --full <a>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_settings
from .db import open_database


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default=None, help="overrides SPOTTER_DATA_DIR")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve"); _common(p)
    p.add_argument("--host", default=None); p.add_argument("--port", type=int, default=None)
    p.add_argument("--reload", action="store_true")

    p = sub.add_parser("migrate"); _common(p)

    p = sub.add_parser("import-dataset"); _common(p)
    p.add_argument("path"); p.add_argument("--actor", default=None)

    p = sub.add_parser("import-model", help="register an already-trained checkpoint (spec §7.6)")
    _common(p)
    p.add_argument("name")
    p.add_argument("weights", help="path to the weights file (.pt/.pth/...)")
    p.add_argument("--kind", default="custom")
    p.add_argument("--calibration", default=None, help="path to calibration.json, if you have one")
    p.add_argument("--metric", action="append", default=[], metavar="LETTER=VALUE",
                   help="e.g. --metric a=0.25 --metric e=0.59 (a=R@1 b=R@5 c=R@10 d=bal_acc e=novelty_auroc f=review@90)")
    p.add_argument("--make-active", action="store_true")
    p.add_argument("--notes", default=None)
    p.add_argument("--actor", default=None)

    p = sub.add_parser("backup"); _common(p)
    p.add_argument("--dest", default=None)

    p = sub.add_parser("restore"); _common(p)
    p.add_argument("archive")

    p = sub.add_parser("export"); _common(p)
    p.add_argument("--full", action="store_true", required=True)
    p.add_argument("out")

    p = sub.add_parser("import"); _common(p)
    p.add_argument("--full", action="store_true", required=True)
    p.add_argument("archive")

    args = parser.parse_args(argv)
    settings = load_settings(
        data_dir=args.data_dir,
        host=getattr(args, "host", None),
        port=getattr(args, "port", None),
    )

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=args.reload)
        return 0

    if args.cmd == "migrate":
        settings.ensure_dirs()
        db = open_database(settings.app_db_path, migrate=False)
        print(f"applied: {db.migrate() or 'nothing (up to date)'}")
        db.close()
        return 0

    if args.cmd == "import-dataset":
        from .services import ingest

        settings.ensure_dirs()
        db = open_database(settings.app_db_path)
        report = ingest.transfer_dataset(db, settings, Path(args.path), actor=args.actor)
        db.close()
        print(json.dumps(report.as_dict(), indent=2))
        return 0

    if args.cmd == "import-model":
        from .services import models as model_svc
        from .settings_store import SettingsStore

        settings.ensure_dirs()
        db = open_database(settings.app_db_path)
        store = SettingsStore(db)
        metrics: dict[str, float] = {}
        for pair in args.metric:
            letter, _, value = pair.partition("=")
            col = model_svc.METRIC_LETTERS.get(letter.strip())
            if not col:
                print(f"error: unknown metric letter {letter!r} (use a-f)", file=sys.stderr)
                db.close()
                return 1
            metrics[col] = float(value)
        try:
            result = model_svc.import_model(
                db, settings, name=args.name, kind=args.kind,
                source_weights_path=args.weights, source_calibration_path=args.calibration,
                metrics=metrics, coefficients=store.get("score_coefficients"),
                notes=args.notes, actor=args.actor,
            )
            if args.make_active:
                model_svc.promote(db, args.name, actor=args.actor)
        except model_svc.ModelImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            db.close()
            return 1
        db.close()
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "backup":
        from . import backup

        dest = Path(args.dest) if args.dest else settings.data_dir.parent / "spotter-backups"
        try:
            res = backup.backup(settings, dest)
        except backup.DatabaseLockedError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"archive": res.archive, "bytes": res.bytes, "files": res.files}, indent=2))
        return 0

    if args.cmd == "restore":
        from . import backup

        backup.restore(Path(args.archive), settings.data_dir)
        print(f"restored into {settings.data_dir}")
        return 0

    if args.cmd == "export":
        from . import backup

        try:
            backup.export_full(settings, Path(args.out))
        except backup.DatabaseLockedError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"exported {settings.data_dir} -> {args.out}")
        return 0

    if args.cmd == "import":
        from . import backup

        backup.import_full(Path(args.archive), settings.data_dir)
        print(f"imported {args.archive} -> {settings.data_dir}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
