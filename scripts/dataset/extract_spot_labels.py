#!/usr/bin/env python3
"""Extract spot labels — CLI entry point for the whole pipeline.

Argument parsing and configuration only; all logic lives in the importable
``pipeline/generate_spot_labels/`` package. One subcommand per stage, plus ``all`` for
the full run::

    pixi run extract-spot-labels all       --input all_sasa_norm   # stage 1 + 1b + 2
    pixi run extract-spot-labels count     --input all_sasa_norm   # stage 0:  images  -> animals/
    pixi run extract-spot-labels segment   --input all_sasa_norm   # stage 1:  images  -> purple/
    pixi run extract-spot-labels anatomy   --input all_sasa_norm   # stage 1b: images  -> anatomy/
    pixi run extract-spot-labels contours  --input all_sasa_norm   # stage 2:  purple/ -> contours.db

``count`` is a standalone stage and is NOT part of ``all``: it answers "how many salamanders are
in this frame?", which every other stage simply assumes is one. Run it on its own to find the
frames that break that assumption before spending anything on them.

    pixi run extract-spot-labels all --input all_sasa_norm  --escalate-models "gemini-3.1-flash-image,gemini-3-pro-image" --max-attempts 1 --escalate-attempts 1

Stage 2 stores per-spot contours, per-spot masks (a full-frame PNG, black except that spot)
*and* each spot's body-grid ``bin`` (1..8) in the ``spots`` table, plus the ``body_axis`` and
``body_bins`` tables — there is no separate masks/ directory.

The bin says WHERE ON THE ANIMAL a spot sits: stage 1b has Gemini mark the head, the tail tip
and the CENTRE LINE connecting them, and stage 2 cuts that midline at 25/50/75 % of its arc
length and splits each quarter left/right of it — boxes 1/2 (0-25 % left/right), 3/4, 5/6, 7/8.
Pass ``--no-anatomy`` to skip stage 1b (one fewer billed call per image); the spots are then
stored unbinned.

Stage 1b is **LLM-judged**: a second model looks at the drawn line and answers three questions —
does it go head to tail, is it completely inside the body, does it bisect the animal? A line that
fails is RE-DRAWN with the judge's written feedback and the rejected image fed back in, so the
model corrects a specific mistake instead of resampling blindly (``--anatomy-attempts``, default
3). Images the judge never accepts are listed in ``anatomy/flagged_axis.csv`` and carry
``body_axis.judged_ok = false`` in the DB. ``--no-judge`` turns the judge off.

``segment`` / ``anatomy`` / ``contours`` re-run a single stage over an existing dir; ``all``
does the efficient per-image pass (purple, anatomy, then contours) in one go. Every stage is
resumable — existing purple/anatomy files are reused unless ``--overwrite``, so a re-run only
pays for what is missing. In particular, a dir that already has purple images costs only the
anatomy call to gain bins.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

from generate_spot_labels import extract_spot_contours as stage2  # noqa: E402
from generate_spot_labels import llm_anatomy as stage1b  # noqa: E402
from generate_spot_labels import animal_screen  # noqa: E402
from generate_spot_labels import llm_animal_count as stage0  # noqa: E402
from generate_spot_labels import llm_body_fill as stage1b_mask  # noqa: E402
from generate_spot_labels import llm_spot_segmentation as stage1  # noqa: E402
from generate_spot_labels import reconfigure_utf8, runner  # noqa: E402


# --- shared argument groups -------------------------------------------------
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", required=True,
                   help="input image dir (a name under images/, or a path)")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N images (for testing)")
    p.add_argument("--run-log", default=None, metavar="PATH",
                   help="append every stage's events to this JSONL log instead of the input "
                        "dir's own (used by build-dataset to log a whole build in one file)")
    p.add_argument("--run-id", default=None,
                   help="stamp every log line with this run id (default: a fresh timestamp)")


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
                   help=f"mask OPEN (noise) kernel size, 0 to disable (default {stage2.DEFAULT_MORPH})")
    p.add_argument("--key-mode", choices=stage2.KEY_MODES, default=stage2.DEFAULT_KEY_MODE,
                   help=f"spot key: 'adaptive' (per-image Otsu on redness, lighting-tolerant) "
                        f"or 'hsv' (original fixed S/V floors) (default {stage2.DEFAULT_KEY_MODE})")
    p.add_argument("--close", type=int, default=stage2.DEFAULT_CLOSE,
                   help=f"adaptive-key CLOSE kernel that seals shading pin-holes, 0 to disable "
                        f"(default {stage2.DEFAULT_CLOSE})")


def _add_anatomy_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--anatomy-mode", choices=("mask", "outline"), default="mask",
                   help="how stage 1b finds the body axis. 'mask' (default): Gemini PAINTS the "
                        "body and the outlines, centre line and tips are computed from the "
                        "region — no judge, 1 call per image. 'outline': the original method, "
                        "where the model draws the two flanks and an LLM judge grades them "
                        "(measured 73%% rejected, ~6 calls per image)")
    p.add_argument("--anatomy-attempts", type=int, default=None,
                   help="draws on the PRIMARY model before escalating; a rejected draft is "
                        "re-drawn with feedback (default: 2 in mask mode, 1 in outline mode)")
    p.add_argument("--anatomy-escalate-models", default=None, metavar="M1,M2,...",
                   help=f"ladder of pricier DRAWING models (cheapest first) to climb when the "
                        f"judge keeps rejecting the line, e.g. "
                        f"'gemini-3.1-flash-image,gemini-3-pro-image' (default: "
                        f"GEMINI_ANATOMY_ESCALATE_MODELS or "
                        f"'{stage1b.DEFAULT_ESCALATE_MODELS}'; pass '' to disable)")
    p.add_argument("--anatomy-escalate-attempts", type=int, default=None,
                   help=f"drafts per escalate rung "
                        f"(default {stage1b.DEFAULT_ESCALATE_ATTEMPTS})")
    p.add_argument("--image-retries", type=int, default=stage1b.DEFAULT_IMAGE_RETRIES,
                   help=f"tries to coax an image out of ONE rung when the model replies with "
                        f"text instead of drawing, before escalating "
                        f"(default {stage1b.DEFAULT_IMAGE_RETRIES})")
    p.add_argument("--min-axis-frac", type=float, default=stage1b.DEFAULT_MIN_AXIS_FRAC,
                   help=f"reject an axis shorter than this fraction of the image diagonal, "
                        f"before spending a judge call (default {stage1b.DEFAULT_MIN_AXIS_FRAC})")
    p.add_argument("--no-judge", action="store_true",
                   help="skip the LLM judge (cheaper, but nothing then catches a line that "
                        "runs off the body or hugs one flank)")
    p.add_argument("--judge-model", default=None,
                   help=f"vision model that grades the drawn line (default: "
                        f"GEMINI_JUDGE_MODEL or {stage1b.DEFAULT_JUDGE_MODEL})")


def _run_log(args: argparse.Namespace, input_dir: str, task: str = ""):
    """The shared JSONL log, when --run-log points every stage at one file."""
    if not getattr(args, "run_log", None):
        return None
    from generate_spot_labels.runlog import RunLog
    return RunLog(Path(args.run_log), run_id=getattr(args, "run_id", None),
                  task=task, input=input_dir)


# --- subcommand handlers ----------------------------------------------------
def _run_all(args: argparse.Namespace) -> int:
    return runner.run(
        args.input,
        run_log=_run_log(args, args.input),
        overwrite=args.overwrite, limit=args.limit, model=args.model,
        eps_frac=args.eps_frac, min_area=args.min_area, morph=args.morph,
        key_mode=args.key_mode, close=args.close,
        keep_gemini_size=args.keep_gemini_size, rewrite=args.rewrite,
        max_attempts=args.max_attempts, max_bleed=args.max_bleed,
        temperature=args.temperature, escalate_models=args.escalate_models,
        escalate_attempts=args.escalate_attempts,
        anatomy=not args.no_anatomy, anatomy_mode=args.anatomy_mode,
        anatomy_attempts=args.anatomy_attempts,
        min_axis_frac=args.min_axis_frac,
        judge=not args.no_judge, judge_model=args.judge_model,
        anatomy_escalate_models=args.anatomy_escalate_models,
        anatomy_escalate_attempts=args.anatomy_escalate_attempts,
        image_retries=args.image_retries,
    )


def _run_segment(args: argparse.Namespace) -> int:
    return stage1.segment_dir(
        args.input,
        overwrite=args.overwrite, limit=args.limit, model=args.model,
        keep_gemini_size=args.keep_gemini_size,
        max_attempts=args.max_attempts, max_bleed=args.max_bleed,
        temperature=args.temperature, escalate_models=args.escalate_models,
        escalate_attempts=args.escalate_attempts,
        workers=args.workers,
    )


def _run_anatomy(args: argparse.Namespace) -> int:
    log = _run_log(args, args.input, task="anatomy")
    if args.anatomy_mode == "mask":
        return stage1b_mask.paint_dir(
            args.input, run_log=log,
            overwrite=args.overwrite, limit=args.limit, model=args.model,
            max_attempts=(args.anatomy_attempts
                          if args.anatomy_attempts is not None
                          else stage1b_mask.DEFAULT_MAX_ATTEMPTS),
            min_axis_frac=args.min_axis_frac, temperature=args.temperature,
            escalate_models=args.anatomy_escalate_models,
            escalate_attempts=(args.anatomy_escalate_attempts
                               if args.anatomy_escalate_attempts is not None
                               else stage1b_mask.DEFAULT_ESCALATE_ATTEMPTS),
            image_retries=args.image_retries,
            workers=args.workers,
        )
    if args.workers > 1:
        print("note: --workers only parallelizes 'mask' anatomy mode; "
              "outline mode runs serially.", file=sys.stderr)
    return stage1b.annotate_dir(
        args.input, run_log=log,
        overwrite=args.overwrite, limit=args.limit, model=args.model,
        max_attempts=(args.anatomy_attempts if args.anatomy_attempts is not None
                      else stage1b.DEFAULT_MAX_ATTEMPTS),
        min_axis_frac=args.min_axis_frac,
        temperature=args.temperature,
        judge=not args.no_judge, judge_model=args.judge_model,
        escalate_models=args.anatomy_escalate_models,
        escalate_attempts=(args.anatomy_escalate_attempts
                           if args.anatomy_escalate_attempts is not None
                           else stage1b.DEFAULT_ESCALATE_ATTEMPTS),
        image_retries=args.image_retries,
    )


def _run_count(args: argparse.Namespace) -> int:
    return stage0.count_dir(
        args.input, run_log=_run_log(args, args.input, task="count"),
        overwrite=args.overwrite, limit=args.limit, model=args.count_model,
        max_dim=args.max_dim, temperature=args.temperature,
        workers=args.workers, include_synthetic=args.include_synthetic,
        screen=not args.no_screen, offaxis_cutoff=args.screen_cutoff,
        gap_cutoff=args.screen_gap, refresh_screen=args.refresh_screen,
        dry_run=args.dry_run,
    )


def _run_contours(args: argparse.Namespace) -> int:
    return stage2.extract_dir(
        args.input, eps_frac=args.eps_frac, min_area=args.min_area,
        morph=args.morph, key_mode=args.key_mode, close=args.close, limit=args.limit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract-spot-labels",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser("all",
                           help="full pipeline: Gemini purple + anatomy + contours/bins",
                           formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(p_all)
    _add_gemini_opts(p_all)
    _add_anatomy_opts(p_all)
    _add_contour_opts(p_all)
    p_all.add_argument("--no-anatomy", action="store_true",
                       help="skip stage 1b (one fewer billed call per image); spots are "
                            "stored with NULL bins")
    p_all.add_argument("--rewrite", default=None, metavar="CSV",
                       help="single-column CSV of image names to redo: process only those, "
                            "regenerate their purple + anatomy, and replace their DB rows")
    p_all.set_defaults(func=_run_all)

    p_seg = sub.add_parser("segment", help="stage 1 only: images -> purple/")
    _add_common(p_seg)
    _add_gemini_opts(p_seg)
    p_seg.add_argument("--workers", type=int, default=1, metavar="N",
                       help="run N Gemini purple calls concurrently (default 1 = serial). "
                            "Each image is an independent call writing its own PNG; start "
                            "modest (4-8) and watch for 429s.")
    p_seg.set_defaults(func=_run_segment)

    p_ana = sub.add_parser(
        "anatomy",
        help="stage 1b only: images -> anatomy/ (head, tail tip, midline). Run `contours` "
             "afterwards to fold the bins into the DB.")
    _add_common(p_ana)
    _add_anatomy_opts(p_ana)
    p_ana.add_argument("--overwrite", action="store_true",
                       help="re-run Gemini even if the anatomy already exists")
    p_ana.add_argument("--model", default=None, help="override GEMINI_MODEL")
    p_ana.add_argument("--temperature", type=float, default=None,
                       help="Gemini sampling temperature (default: model default)")
    p_ana.add_argument("--workers", type=int, default=1, metavar="N",
                       help="run N Gemini anatomy calls concurrently (mask mode only; "
                            "default 1 = serial). Start modest (4-8) and watch for 429s.")
    p_ana.set_defaults(func=_run_anatomy)

    p_cnt = sub.add_parser(
        "count",
        help="stage 0 only: images -> animals/ (how many salamanders per frame). Not part of "
             "`all` — run it on its own to find frames that break the one-animal assumption.",
        description=stage0.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(p_cnt)
    p_cnt.add_argument("--dry-run", action="store_true",
                       help="print the plan and the billed-call estimate, send nothing")
    p_cnt.add_argument("--overwrite", action="store_true",
                       help="re-count even if animals/<stem>.json already exists")
    p_cnt.add_argument("--count-model", default=None,
                       help=f"vision model that does the counting (default: GEMINI_COUNT_MODEL "
                            f"or {stage0.DEFAULT_MODEL}). A vision+text model, NOT an image "
                            f"model — this stage draws nothing")
    p_cnt.add_argument("--max-dim", type=int, default=stage0.DEFAULT_MAX_DIM,
                       help=f"downscale the long side to this before sending; counting animals "
                            f"does not need 12 MP (default {stage0.DEFAULT_MAX_DIM})")
    p_cnt.add_argument("--temperature", type=float, default=None,
                       help="Gemini sampling temperature (default: model default)")
    p_cnt.add_argument("--no-screen", action="store_true",
                       help="skip layer 1 and bill EVERY frame. Use when you want the model's "
                            "opinion on all of them (e.g. to measure what layer 1 gets wrong)")
    p_cnt.add_argument("--screen-cutoff", type=float,
                       default=animal_screen.DEFAULT_OFFAXIS_CUTOFF,
                       help=f"layer 1 gate 1: defer a frame whose off-axis mass reaches this "
                            f"(default {animal_screen.DEFAULT_OFFAXIS_CUTOFF}). LOWER = "
                            f"more frames deferred = safer and more expensive")
    p_cnt.add_argument("--screen-gap", type=float,
                       default=animal_screen.DEFAULT_GAP_CUTOFF,
                       help=f"layer 1 gate 2: defer a frame with a spot this far off the body "
                            f"mask, measured in BODY LENGTHS (default "
                            f"{animal_screen.DEFAULT_GAP_CUTOFF}). Counting strays instead of "
                            f"measuring their distance cannot tell a clipped mask edge from a "
                            f"second animal")
    p_cnt.add_argument("--refresh-screen", action="store_true",
                       help="recompute layer 1's metrics instead of reusing animals/screen.csv "
                            "(needed after re-running `anatomy` or `compute-quality`)")
    p_cnt.add_argument("--include-synthetic", action="store_true",
                       help="also count the generated `<label>_g<k>` views. They are "
                            "re-renderings of one source photo of one animal, so their count "
                            "is known already — skipped by default, which is ~a third of the "
                            "corpus not paid for")
    p_cnt.add_argument("--workers", type=int, default=1, metavar="N",
                       help="run N counting calls concurrently (default 1 = serial). Start "
                            "modest (4-8) and watch for 429s.")
    p_cnt.set_defaults(func=_run_count)

    p_con = sub.add_parser(
        "contours",
        help="stage 2 only: purple/ (+ anatomy/) -> contours/contours.db (+ masks, bins)")
    _add_common(p_con)
    _add_contour_opts(p_con)
    p_con.set_defaults(func=_run_contours)

    return parser


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except stage0.FatalCountError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    except stage1b.FatalJudgeError as exc:
        # A misconfigured judge is a setup mistake, not a crash — say so plainly and stop
        # before any more billed images are drawn.
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
