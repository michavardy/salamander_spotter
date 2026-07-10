#!/usr/bin/env python3
"""Extract spot labels — CLI entry point for the whole pipeline.

Argument parsing and configuration only; all logic lives in the importable
``pipeline/generate_spot_labels/`` package. One subcommand per stage, plus ``all`` for
the full run::

    pixi run extract-spot-labels all       --input all_sasa_norm   # stage 1 + 2
    pixi run extract-spot-labels segment   --input all_sasa_norm   # stage 1: images  -> purple/
    pixi run extract-spot-labels contours  --input all_sasa_norm   # stage 2: purple/ -> contours.db

    pixi run extract-spot-labels all --input all_sasa_norm  --escalate-models "gemini-3.1-flash-image,gemini-3-pro-image" --max-attempts 1 --escalate-attempts 1

Stage 2 stores per-spot contours *and* per-spot masks (a full-frame PNG, black except
that spot) in the ``spots`` table — there is no separate masks/ directory. ``segment`` /
``contours`` re-run a single stage over an existing dir; ``all`` does the efficient
per-image pass (Gemini purple, then contours) in one go.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from generate_spot_labels import extract_spot_contours as stage2  # noqa: E402
from generate_spot_labels import llm_spot_segmentation as stage1  # noqa: E402
from generate_spot_labels import reconfigure_utf8, runner  # noqa: E402


# --- shared argument groups -------------------------------------------------
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", required=True,
                   help="input image dir (a name under images/, or a path)")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N images (for testing)")


def _add_gemini_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--overwrite", action="store_true",
                   help="re-run Gemini even if the purple image already exists")
    p.add_argument("--model", default=None, help="override GEMINI_MODEL")
    p.add_argument("--keep-gemini-size", action="store_true",
                   help="keep Gemini's output resolution (default: resize back "
                        "to the original image's pixel grid)")
    p.add_argument("--max-attempts", type=int, default=stage1.DEFAULT_MAX_ATTEMPTS,
                   help=f"Gemini re-draws to beat the bleed threshold "
                        f"(default {stage1.DEFAULT_MAX_ATTEMPTS})")
    p.add_argument("--max-bleed", type=float, default=stage1.DEFAULT_MAX_BLEED,
                   help=f"reject a draft if >this fraction of magenta lands off the "
                        f"spots (default {stage1.DEFAULT_MAX_BLEED})")
    p.add_argument("--escalate-models", default=None, metavar="M1,M2,...",
                   help="comma-separated ladder of pricier models (cheapest first) to "
                        "fall back to when a cheaper rung never beats --max-bleed, e.g. "
                        "'gemini-3.1-flash-image,gemini-3-pro-image' "
                        "(default: GEMINI_ESCALATE_MODELS or off)")
    p.add_argument("--escalate-attempts", type=int, default=stage1.DEFAULT_ESCALATE_ATTEMPTS,
                   help=f"draws per escalate model (default {stage1.DEFAULT_ESCALATE_ATTEMPTS})")
    p.add_argument("--temperature", type=float, default=None,
                   help="Gemini sampling temperature (default: model default)")


def _add_contour_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--eps-frac", type=float, default=stage2.DEFAULT_EPS_FRAC,
                   help=f"Douglas-Peucker epsilon / perimeter (default {stage2.DEFAULT_EPS_FRAC})")
    p.add_argument("--min-area", type=float, default=stage2.DEFAULT_MIN_AREA,
                   help=f"drop spots smaller than this in px^2 (default {stage2.DEFAULT_MIN_AREA})")
    p.add_argument("--morph", type=int, default=stage2.DEFAULT_MORPH,
                   help=f"contour mask cleanup kernel size, 0 to disable (default {stage2.DEFAULT_MORPH})")


# --- subcommand handlers ----------------------------------------------------
def _run_all(args: argparse.Namespace) -> int:
    return runner.run(
        args.input,
        overwrite=args.overwrite, limit=args.limit, model=args.model,
        eps_frac=args.eps_frac, min_area=args.min_area, morph=args.morph,
        keep_gemini_size=args.keep_gemini_size, rewrite=args.rewrite,
        max_attempts=args.max_attempts, max_bleed=args.max_bleed,
        temperature=args.temperature, escalate_models=args.escalate_models,
        escalate_attempts=args.escalate_attempts,
    )


def _run_segment(args: argparse.Namespace) -> int:
    return stage1.segment_dir(
        args.input,
        overwrite=args.overwrite, limit=args.limit, model=args.model,
        keep_gemini_size=args.keep_gemini_size,
        max_attempts=args.max_attempts, max_bleed=args.max_bleed,
        temperature=args.temperature, escalate_models=args.escalate_models,
        escalate_attempts=args.escalate_attempts,
    )


def _run_contours(args: argparse.Namespace) -> int:
    return stage2.extract_dir(
        args.input, eps_frac=args.eps_frac, min_area=args.min_area,
        morph=args.morph, limit=args.limit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract-spot-labels",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser("all", help="full pipeline: Gemini purple + contours (+ per-spot masks)",
                           formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(p_all)
    _add_gemini_opts(p_all)
    _add_contour_opts(p_all)
    p_all.add_argument("--rewrite", default=None, metavar="CSV",
                       help="single-column CSV of image names to redo: process only those, "
                            "regenerate their purple, and replace their DB rows")
    p_all.set_defaults(func=_run_all)

    p_seg = sub.add_parser("segment", help="stage 1 only: images -> purple/")
    _add_common(p_seg)
    _add_gemini_opts(p_seg)
    p_seg.set_defaults(func=_run_segment)

    p_con = sub.add_parser("contours",
                           help="stage 2 only: purple/ -> contours/contours.db (+ per-spot masks)")
    _add_common(p_con)
    _add_contour_opts(p_con)
    p_con.set_defaults(func=_run_contours)

    return parser


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
