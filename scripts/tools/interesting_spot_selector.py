#!/usr/bin/env python3
"""Hand-label the *interesting* spots per image in a little local web app.

Opens a browser page that shows one image at a time from ``images/<folder>/``. Click inside
a spot and the whole spot turns green; click it again to unselect (toggle). The click is
mapped to a real spot using the per-spot masks already in ``images/<folder>/contours/contours.db``,
so it is the actual spot that lights up, not just a dot. Move between images with the arrow
keys or the Prev/Next buttons; ``U`` jumps to the next unlabeled image. Every toggle is saved
immediately to::

    artifacts/interesting_spots/<folder>/interesting_spots.json      { "aa_1_1": [3, 7], ... }

The app opens on the first image that has no interesting spots yet, so you can stop and resume.

    pixi run interesting-spot-selector                         # folder: all_sasa_norm
    pixi run interesting-spot-selector --output sasa_norm      # a different image folder
    pixi run interesting-spot-selector --port 9000 --no-browser

``--output`` names the image folder (a bare name is resolved under ``images/``); ``--input`` is
accepted as an alias.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from generate_spot_labels import reconfigure_utf8  # noqa: E402
from generate_spot_labels._common import resolve_input_dir  # noqa: E402
from interesting_spots import SpotSelectorApp, run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", "--input", dest="output", default="all_sasa_norm",
                        help="image folder to label (bare name resolves under images/). "
                             "Default: all_sasa_norm")
    parser.add_argument("--labels", default=None,
                        help="override the labels JSON path (default: "
                             "artifacts/interesting_spots/<folder>/interesting_spots.json)")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="bind port (default 8765)")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not auto-open the browser")
    args = parser.parse_args(argv)

    try:
        input_dir = resolve_input_dir(args.output)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    labels_path = (Path(args.labels) if args.labels
                   else REPO_ROOT / "artifacts" / "interesting_spots" / input_dir.name
                   / "interesting_spots.json")

    try:
        app = SpotSelectorApp(input_dir, labels_path)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if app.total == 0:
        print(f"error: no images with spots found in {input_dir} "
              f"(is {app.db_path} populated?)", file=sys.stderr)
        app.close()
        return 1

    run(app, host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
