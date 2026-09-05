#!/usr/bin/env python3
"""emb-bakeoff — run several matchers through the shared harness and emit a comparison.

Evaluates each matcher on the *same* spotsets + folds and writes
artifacts/spot_embedding/bakeoff/comparison.md — a top-to-bottom ladder from the dummy
(chance floor) to the oracle (perfect ceiling), with the real matchers in between::

    pixi run emb-bakeoff                                  # default: dummy classical cnn oracle
    pixi run emb-bakeoff --models dummy classical oracle  # pick the set
    pixi run emb-bakeoff --models dummy classical cnn oracle --k 5 --mode session

``gemini`` is intentionally excluded from the default set (billed/external — run it opt-in
via `pixi run emb-eval --model gemini --limit N`). Argument parsing only; logic lives in
``spot_embedding.runner``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

from spot_embedding import reconfigure_utf8  # noqa: E402
from spot_embedding._common import DEFAULT_DATASET  # noqa: E402
from spot_embedding import runner  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emb-bakeoff", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--models", nargs="+", default=["dummy", "classical", "cnn", "oracle"],
                   help="matchers to compare (default: dummy classical cnn oracle)")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--mode", choices=("individual", "session"), default="individual")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60, help="training epochs for learned models")
    # quality gate: drop bad images from BOTH eval and training (0 / 1.0 = off). See emb-quality.
    p.add_argument("--min-spots", type=int, default=0, help="drop images with fewer spots than this")
    p.add_argument("--min-blur", type=float, default=0.0, help="drop images blurrier than this (var-Laplacian)")
    p.add_argument("--max-largest-frac", type=float, default=1.0,
                   help="drop images where one spot is >= this fraction of all spot area")
    return p


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    args = build_parser().parse_args(argv)
    runner.bakeoff(args.dataset, args.models, k=args.k, mode=args.mode, seed=args.seed,
                   epochs=args.epochs, min_spots=args.min_spots, min_blur=args.min_blur,
                   max_largest_frac=args.max_largest_frac)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
