#!/usr/bin/env python3
"""emb-selfcheck — Phase-0 acceptance doctor (CLI entry point).

One command that prints the DB ground truth vs. what the loader sees, runs the leakage
guard, and runs the oracle + dummy matchers through the harness, asserting oracle ≈ 1.0 and
dummy ≈ chance. Prints PASS/FAIL and exits non-zero on failure::

    pixi run emb-selfcheck
    pixi run emb-selfcheck --dataset all_sasa_norm_2026_10_07 --k 5 --seed 0

Argument parsing only; logic lives in ``spot_embedding.selfcheck``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

from spot_embedding import reconfigure_utf8  # noqa: E402
from spot_embedding._common import DEFAULT_DATASET  # noqa: E402
from spot_embedding import selfcheck  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emb-selfcheck", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    args = build_parser().parse_args(argv)
    return selfcheck.run(args.dataset, k=args.k, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
