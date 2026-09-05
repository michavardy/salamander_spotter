#!/usr/bin/env python3
"""Evaluate a matcher through the shared harness — CLI entry point (Phase 0).

Runs a named matcher over the CV splits and writes a report to
artifacts/spot_embedding/runs/<id>/ (metrics.json, report.md, risk_coverage.csv)::

    pixi run emb-eval --model dummy      # random baseline -> chance
    pixi run emb-eval --model oracle     # label-cheating -> ~perfect (plumbing check)
    pixi run emb-eval --model dummy --dataset all_sasa_norm_2026_10_07 --k 5 --seed 0

Phase 0 ships only the ``dummy`` and ``oracle`` matchers; the learned candidates
(set_transformer, gnn, cnn, hungarian, classical) are added in later phases. Argument
parsing only; logic lives in ``spot_embedding.runner``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

from spot_embedding import reconfigure_utf8  # noqa: E402
from spot_embedding._common import DEFAULT_DATASET  # noqa: E402
from spot_embedding import runner  # noqa: E402
from spot_embedding.models import MATCHERS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emb-eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dummy", choices=sorted(set(MATCHERS) | runner.LEARNED),
                   help="matcher to evaluate (learned: set_transformer/gnn/hungarian train per fold)")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--mode", choices=("individual", "session"), default="individual")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=128, help="dummy embedding dimension")
    p.add_argument("--orient", choices=("a", "b"), default="a",
                   help="spotdesc orientation mode: a=as-is, b=principal-axis canonical (§2a)")
    p.add_argument("--epochs", type=int, default=60, help="training epochs for learned models")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of gallery/query images per fold (gemini cost control)")
    # quality gate: drop bad images from BOTH eval and training (0 / 1.0 = off). See emb-quality.
    p.add_argument("--min-spots", type=int, default=0, help="drop images with fewer spots than this")
    p.add_argument("--min-blur", type=float, default=0.0, help="drop images blurrier than this (var-Laplacian)")
    p.add_argument("--max-largest-frac", type=float, default=1.0,
                   help="drop images where one spot is >= this fraction of all spot area")
    return p


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    args = build_parser().parse_args(argv)
    runner.evaluate_matcher(args.dataset, args.model, k=args.k, mode=args.mode, seed=args.seed,
                            dim=args.dim, limit=args.limit, orient=args.orient, epochs=args.epochs,
                            min_spots=args.min_spots, min_blur=args.min_blur,
                            max_largest_frac=args.max_largest_frac)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
