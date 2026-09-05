#!/usr/bin/env python3
"""Prepare the spot-embedding dataset — CLI entry point (Phase 0).

Loads the packaged dataset, caches per-photo SpotSets, builds leakage-safe CV splits, runs
the leakage guard, and writes a manifest::

    pixi run emb-prepare                                   # defaults: all_sasa_norm_2026_10_07, 5-fold individual
    pixi run emb-prepare --dataset all_sasa_norm_2026_10_07 --k 5 --mode session --seed 0
    pixi run emb-prepare --rebuild                         # ignore the SpotSet cache

Outputs land in artifacts/spot_embedding/prepared/<dataset>/ (spotsets.pkl, splits.json,
manifest.json). Argument parsing only; logic lives in ``spot_embedding.runner``.
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
    p = argparse.ArgumentParser(prog="emb-prepare", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="dataset name under datasets/ or a path")
    p.add_argument("--k", type=int, default=5, help="number of CV folds (default 5)")
    p.add_argument("--mode", choices=("individual", "session"), default="individual",
                   help="grouping unit for the split (default individual)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rebuild", action="store_true", help="rebuild the SpotSet cache from the DB")
    return p


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    args = build_parser().parse_args(argv)
    manifest = runner.prepare(args.dataset, k=args.k, mode=args.mode, seed=args.seed,
                              rebuild=args.rebuild)
    return 1 if manifest["splits"]["leakage_problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
