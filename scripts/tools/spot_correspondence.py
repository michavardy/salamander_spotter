#!/usr/bin/env python3
"""Label and analyse human spot correspondences — the label the matcher has never had.

Image-level labels say "these two photos are the same animal". A correspondence says "spot 7 in
photo A is the same physical spot as spot 3 in photo B". Only the second can tell apart the four
failures the pipeline currently sees as one number: the spot was never extracted, it was
shattered, its descriptor drifted, or the matcher picked the wrong neighbour.

Two steps:

    pixi run correspondence                     # label: click LEFT spot, then its RIGHT partner
    pixi run correspondence-analyze             # measure: what the labels say about the pipeline

Labelling is stop-and-resume (every click is saved, and it opens on the first unfinished pair),
so ~25 individuals at ~10 confident spots each is an evening, not a project. That is enough for
the statistics — these are measurements, not training targets.

Store: artifacts/correspondence/<images-folder>/links.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))
sys.path.insert(0, str(REPO_ROOT / "pipeline" / "spot_transformer" / "core"))

DEFAULT_FOLDER = "all_sasa_norm"


def _reconfigure_utf8():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")       # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def store_for(folder: str) -> Path:
    return REPO_ROOT / "artifacts" / "correspondence" / folder / "links.json"


def cmd_label(args) -> int:
    from correspondence.app import CorrespondenceApp
    from correspondence.server import run

    input_dir = REPO_ROOT / "images" / args.folder
    if not input_dir.is_dir():
        print(f"error: no image folder {input_dir}", file=sys.stderr)
        return 1
    only = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None
    app = CorrespondenceApp(input_dir, store_for(args.folder),
                            n_individuals=args.n_individuals, seed=args.seed, only=only)
    if not app.total:
        print("no individuals with >=2 real photos in this folder.", file=sys.stderr)
        return 1
    run(app, host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_analyze(args) -> int:
    import duckdb

    from correspondence.analyze import analyze
    import data as d                                   # spot_transformer/core/data.py

    store = args.store or store_for(args.folder)
    if not Path(store).is_file():
        print(f"error: no correspondence store at {store} — run `pixi run correspondence` first",
              file=sys.stderr)
        return 1

    db_path = Path(args.db) if args.db else d.DB_PATH
    if not db_path.is_file():
        print(f"error: no dataset db at {db_path}", file=sys.stderr)
        return 1

    # Compare every embedding flow that exists, so the descriptor table shows concat vs tensor
    # side by side without a second run.
    con = duckdb.connect(str(db_path), read_only=True)
    have = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    con.close()
    tables = {n: t for n, t in (("concat", "spot_embeddings"),
                                ("tensor", "spot_embeddings_tensor")) if t in have}
    if not tables:
        print(f"error: no spot_embeddings table in {db_path}", file=sys.stderr)
        return 1
    print(f"dataset db : {db_path}")
    print(f"flows      : {', '.join(tables)}\n")
    return analyze(Path(store), db_path, tables)


def main(argv=None) -> int:
    _reconfigure_utf8()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    lb = sub.add_parser("label", help="open the labelling web app")
    lb.add_argument("--folder", default=DEFAULT_FOLDER, help="images/<folder> (purple + masks)")
    lb.add_argument("--n-individuals", type=int, default=25,
                    help="how many individuals to sample (default 25)")
    lb.add_argument("--only", default=None,
                    help="comma list of labels to label instead of a sample, e.g. aj_1,rz_1")
    lb.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
    lb.add_argument("--host", default="127.0.0.1")
    lb.add_argument("--port", type=int, default=8772)
    lb.add_argument("--no-browser", action="store_true")
    lb.set_defaults(func=cmd_label)

    an = sub.add_parser("analyze", aliases=["analyse"], help="what the labels say")
    an.add_argument("--folder", default=DEFAULT_FOLDER)
    an.add_argument("--store", default=None, help="override the links.json path")
    an.add_argument("--db", default=None, help="override the dataset contours.db")
    an.set_defaults(func=cmd_analyze)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
