#!/usr/bin/env python3
"""emb-identify — query one photo against a gallery (planned for Phase 4).

Not implemented in Phase 0. The three-way match / new-individual / abstain decision plus the
confidence metric arrive with Phase 4 — see docs/spot_embedding_aggregation_plan.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

from spot_embedding import reconfigure_utf8  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emb-identify", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gallery", default=None)
    p.add_argument("--image", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    build_parser().parse_args(argv)
    print("emb-identify is not implemented yet — planned for Phase 4 "
          "(see docs/spot_embedding_aggregation_plan.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
