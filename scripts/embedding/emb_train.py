#!/usr/bin/env python3
"""emb-train — train one candidate matcher (planned for Phase 3).

Not implemented in Phase 0. The Phase-0 deliverable is the shared harness plus the dummy /
oracle sanity matchers (``pixi run emb-eval``, ``pixi run emb-selfcheck``). Training the
learned candidates (set_transformer, gnn, cnn, hungarian) arrives with Phase 2/3 — see
docs/spot_embedding_aggregation_plan.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

from spot_embedding import reconfigure_utf8  # noqa: E402
from spot_embedding._common import DEFAULT_DATASET  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emb-train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="candidate to train (Phase 2/3)")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--config", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    build_parser().parse_args(argv)
    print("emb-train is not implemented yet — planned for Phase 3 "
          "(see docs/spot_embedding_aggregation_plan.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
