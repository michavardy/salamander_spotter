#!/usr/bin/env python3
"""Orchestration logic for "generate spot labels" — Gemini purple + anatomy + contours.

Pure logic: no argument parsing here. The CLI that configures and calls :func:`run` is
``scripts/dataset/extract_spot_labels.py all`` (``pixi run extract-spot-labels all``).

For every salamander image in ``input`` (one at a time):

    stage 1   image  -> Gemini inpaint    -> <input_dir>/purple/<stem>.png
    stage 1b  image  -> Gemini annotate   -> <input_dir>/anatomy/<stem>.{png,json}
    stage 2   purple -> cv2 magenta keying + per-spot contours & masks + BINS
              -> contours/contours.db

Stage 2 writes one row per spot into the ``spots`` table, each carrying a centroid-local
contour, a full-frame ``mask_png`` (black except that spot), and its body-grid ``bin``
(1..8); there is no separate masks/ directory.

Stage 1b asks Gemini to mark the head, the tail tip and the line down the back on a COPY of
the photo (never on the purple image — a red line down the body would slice the magenta
spots in two). Stage 2 cuts that midline at 25/50/75 % of its arc length and splits each
quarter left/right, giving every spot a bin. An image whose axis cannot be recovered still
gets its spots; they are simply left unbinned.

Stage 1 judges every draft against the original with a bleed check and re-draws; if the
primary model never beats ``max_bleed`` it can escalate through the pricier
``escalate_models`` ladder before giving up and flagging the image.

Stage 1 purple images and stage 1b anatomy are both reused unless ``overwrite`` is set, so
re-runs only pay for what is missing — an input whose purple already exists costs just the
anatomy call. Pass a ``rewrite`` CSV of image names to redo only those (regenerating their
purple + anatomy and replacing their DB rows).
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import extract_spot_contours as stage2
from . import llm_anatomy as stage1b
from . import llm_body_fill as stage1b_mask
from . import llm_spot_segmentation as stage1
from ._common import (
    contours_db_for,
    list_images,
    purple_dir_for,
    read_rewrite_names,
    resolve_input_dir,
    resolve_rewrite_csv,
)
from .runlog import ACCEPTED, DONE, FAILED, REGENERATE, SKIPPED, RunLog

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)


def run(
    input: str,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    model: str | None = None,
    eps_frac: float = stage2.DEFAULT_EPS_FRAC,
    min_area: float = stage2.DEFAULT_MIN_AREA,
    morph: int = stage2.DEFAULT_MORPH,
    key_mode: str = stage2.DEFAULT_KEY_MODE,
    close: int = stage2.DEFAULT_CLOSE,
    keep_gemini_size: bool = False,
    rewrite: str | None = None,
    max_attempts: int = stage1.DEFAULT_MAX_ATTEMPTS,
    max_bleed: float = stage1.DEFAULT_MAX_BLEED,
    temperature: float | None = None,
    escalate_models: str | list[str] | None = None,
    escalate_attempts: int = stage1.DEFAULT_ESCALATE_ATTEMPTS,
    anatomy: bool = True,
    anatomy_mode: str = "mask",
    anatomy_attempts: int | None = None,
    min_axis_frac: float = stage1b.DEFAULT_MIN_AXIS_FRAC,
    judge: bool = True,
    judge_model: str | None = None,
    anatomy_escalate_models: str | list[str] | None = None,
    anatomy_escalate_attempts: int | None = None,
    image_retries: int = stage1b.DEFAULT_IMAGE_RETRIES,
    run_log: "RunLog | None" = None,
) -> int:
    """Run the full pipeline over one input dir. Returns a process exit code.

    ``rewrite`` is a CSV of image names to redo: only those images are processed,
    their purple is regenerated via Gemini (implies ``overwrite``), and their DB
    rows are replaced. ``anatomy=False`` skips stage 1b (one fewer billed call per image);
    the spots are then stored with NULL bins.
    """
    input_dir = resolve_input_dir(input)
    purple_dir = purple_dir_for(input_dir)
    purple_dir.mkdir(parents=True, exist_ok=True)
    anatomy_dir = stage1b.anatomy_dir_for(input_dir)
    if anatomy:
        anatomy_dir.mkdir(parents=True, exist_ok=True)
    db_path = contours_db_for(input_dir)

    images = list_images(input_dir)

    # --- rewrite mode: restrict to the CSV-listed images and force regeneration ---
    if rewrite:
        csv_path = resolve_rewrite_csv(rewrite, input_dir)
        wanted_raw = read_rewrite_names(csv_path)
        wanted = {Path(n).stem for n in wanted_raw}
        by_stem = {p.stem for p in images}
        missing = [n for n in wanted_raw if Path(n).stem not in by_stem]
        images = [p for p in images if p.stem in wanted]
        overwrite = True  # the point of a rewrite is to regenerate + replace
        logger.info(f"rewrite mode: {len(images)} image(s) from {csv_path}")
        if missing:
            logger.warning(f"warning: {len(missing)} name(s) in CSV not found in {input_dir.name}/: "
                  + ", ".join(missing))

    if limit is not None:
        images = images[:limit]
    if not images:
        where = "rewrite CSV" if rewrite else input_dir
        logger.error(f"error: no images to process ({where})")
        return 1

    segmenter = stage1.GeminiSpotSegmenter(
        model=model, temperature=temperature,
        escalate_models=escalate_models, escalate_attempts=escalate_attempts,
    )

    # Stage 1b has two implementations. "mask" (the default) has Gemini PAINT the body and
    # derives the outlines, the centre line and the tips from the region in numpy; "outline" is
    # the original, which asked the model to draw the two flanks and paid a judge to grade them.
    # The judge only exists in outline mode — a computed bisector has nothing to grade.
    mask_mode = anatomy_mode == "mask"
    attempts = anatomy_attempts if anatomy_attempts is not None else (
        stage1b_mask.DEFAULT_MAX_ATTEMPTS if mask_mode else stage1b.DEFAULT_MAX_ATTEMPTS)
    rungs = anatomy_escalate_attempts if anatomy_escalate_attempts is not None else (
        stage1b_mask.DEFAULT_ESCALATE_ATTEMPTS if mask_mode
        else stage1b.DEFAULT_ESCALATE_ATTEMPTS)

    annotator = the_judge = None
    if anatomy:
        cls = (stage1b_mask.GeminiBodyPainter if mask_mode
               else stage1b.GeminiAnatomyAnnotator)
        annotator = cls(model=model, temperature=temperature,
                        escalate_models=anatomy_escalate_models,
                        escalate_attempts=rungs)
        if not mask_mode:
            the_judge = stage1b.make_judge(judge, judge_model)

    # One log for the whole input dir (or the whole build, when the orchestrator supplies one),
    # with a per-task view so every line says which stage wrote it.
    base_log = run_log or RunLog(input_dir / stage1b.JUDGE_LOG_NAME, input=input_dir.name)
    purple_log = base_log.child(task="purple", input=input_dir.name)
    the_log = base_log.child(task="anatomy", input=input_dir.name) if anatomy else None
    contour_log = base_log.child(task="contours", input=input_dir.name)
    store = stage2.ContourStore(db_path)

    total = len(images)
    logger.info(f"processing images in dir {input_dir}")
    logger.info(f"  purple model: {segmenter.model}")
    if annotator is not None:
        logger.info(f"  anatomy     : {anatomy_mode} mode")
        logger.info(f"  draw ladder : {annotator.ladder_str(attempts)}")
        logger.info(f"  judge model : "
              + ("NONE — the mask is graded by free geometric gates" if mask_mode
                 else (the_judge.model if the_judge else "OFF (--no-judge)")))
    logger.info(f"  run log     : {base_log.path}  (run_id={base_log.run_id})")
    failures = 0
    total_spots = 0
    total_binned = 0
    no_axis = 0             # images that reached stage 2 without a usable axis
    escalated = 0            # images whose accepted draft came from the escalate model
    flagged: list[str] = []   # image names whose best purple draft still bled — for review
    rejected: list[str] = []  # image names whose centre line the judge never accepted
    fragmented: list[str] = []  # image names whose spots came out fragmented (QC) — for a re-key
    try:
        for i, src in enumerate(images, start=1):
            logger.info(f"processing image {i} / {total}: {src.name}")
            purple_path = stage1.purple_path_for(purple_dir, src)

            # --- stage 1: purple image from Gemini (judged + retried) ---
            logger.info("stage 1: extracting purple image from gemini")
            if purple_path.exists() and not overwrite:
                logger.info("  purple exists, reusing (use --overwrite to redo)")
                purple_log.write({"image": src.name, "result": SKIPPED,
                                  "detail": "purple already on disk", "billed_calls": 0})
            else:
                try:
                    res = segmenter.segment_file(
                        src, keep_gemini_size=keep_gemini_size,
                        max_attempts=max_attempts, max_bleed=max_bleed,
                    )
                    purple_path.write_bytes(res.png)
                    if res.model != segmenter.model:
                        escalated += 1
                    if res.passed:
                        logger.info(f"  saved {purple_path.relative_to(input_dir)} "
                              f"[bleed={res.bleed:.2f}, {res.attempts} draw(s), {res.model}]")
                    else:
                        flagged.append(src.name)
                        logger.warning(f"  saved {purple_path.relative_to(input_dir)} "
                              f"[bleed={res.bleed:.2f} > {max_bleed}, FLAGGED after "
                              f"{res.attempts} draw(s), {res.model}]")
                    purple_log.write({
                        "image": src.name, "attempt": res.attempts,
                        "max_attempts": max_attempts, "draw_model": res.model,
                        "escalated": res.model != segmenter.model,
                        "bleed": None if res.bleed != res.bleed else round(res.bleed, 3),
                        "max_bleed": max_bleed,
                        "result": ACCEPTED if res.passed else REGENERATE,
                        "detail": "" if res.passed else "bleed over threshold — flagged",
                        "billed_calls": res.attempts,
                    })
                except Exception as exc:
                    failures += 1
                    logger.error(f"  ERROR (stage 1): {exc}")
                    purple_log.write({"image": src.name, "result": FAILED,
                                      "detail": str(exc)[:300]})
                    continue  # can't do stage 2 without a purple image

            # --- stage 1b: head / tail tip / midline, drawn on a COPY of the original ---
            anat = None
            if annotator is not None:
                logger.info("stage 1b: labelling head, tail and centre line from gemini")
                existing = stage1b.load_anatomy(anatomy_dir, src.stem)
                if existing is not None and not overwrite:
                    anat = existing
                    logger.info(f"  anatomy exists, reusing [{anat.source}] "
                          "(use --overwrite to redo)")
                    the_log.write({"image": src.name, "result": SKIPPED,
                                   "detail": "anatomy already on disk",
                                   "axis_source": anat.source,
                                   "judged_ok": anat.judged_ok, "billed_calls": 0})
                else:
                    try:
                        if mask_mode:
                            res = stage1b_mask.paint_file_to_disk(
                                annotator, src, anatomy_dir,
                                max_attempts=attempts, min_axis_frac=min_axis_frac,
                                image_retries=image_retries, log=the_log,
                            )
                        else:
                            res = stage1b.annotate_file_to_disk(
                                annotator, src, anatomy_dir,
                                max_attempts=attempts, min_axis_frac=min_axis_frac,
                                judge=the_judge, log=the_log, image_retries=image_retries,
                            )
                        anat = res.anatomy
                        if not res.passed:
                            rejected.append(src.name)
                    except stage1b.FatalJudgeError:
                        raise   # misconfigured judge: stop, do not burn the rest of the dir
                    except Exception as exc:
                        # A missing axis costs bins, not the image — stage 2 still runs.
                        logger.error(f"  ERROR (stage 1b): {exc}")

            # --- stage 2: contour data + bins from the purple image ---
            logger.info("stage 2: extracting contour data from image")
            try:
                mask_path = anatomy_dir / f"{src.stem}_mask.png"
                rec = stage2.process_purple_image(
                    purple_path, store,
                    eps_frac=eps_frac, min_area=min_area, morph=morph,
                    key_mode=key_mode, close=close,
                    source_image=src.name, anatomy=anat,
                    body_mask_png=mask_path.read_bytes() if mask_path.is_file() else None,
                )
                n = len(rec["spots"])
                total_spots += n
                qc = rec["qc"]
                if qc["flagged"]:
                    fragmented.append(src.name)
                frag = "  [FRAGMENTED?]" if qc["flagged"] else ""
                if rec["body_axis"] is None:
                    no_axis += 1
                    logger.info(f"saved to db ({n} spots + per-spot masks, median "
                          f"{qc['median_area']:.0f}px, spots UNPOSITIONED){frag}")
                    contour_log.write({"image": src.name, "result": DONE, "n_spots": n,
                                       "n_positioned": 0, "median_area": qc["median_area"],
                                       "fragmented": qc["flagged"],
                                       "detail": "no anatomy — spots unpositioned",
                                       "billed_calls": 0})
                else:
                    binned = stage2.n_positioned(rec)
                    total_binned += binned
                    logger.info(f"saved to db ({n} spots + per-spot masks, median "
                          f"{qc['median_area']:.0f}px, {binned} positioned, "
                          f"{stage2.position_summary(rec)}){frag}")
                    contour_log.write({
                        "image": src.name, "result": DONE, "n_spots": n,
                        "n_positioned": binned, "median_area": qc["median_area"],
                        "fragmented": qc["flagged"],
                        "position_summary": stage2.position_summary(rec),
                        "axis_source": rec["body_axis"]["source"],
                        "judged_ok": rec["body_axis"].get("judged_ok"),
                        "billed_calls": 0,
                    })
            except Exception as exc:
                failures += 1
                logger.error(f"  ERROR (stage 2): {exc}")
                contour_log.write({"image": src.name, "result": FAILED,
                                   "detail": str(exc)[:300]})
    finally:
        store.close()

    if flagged:
        # A single-column CSV that feeds straight back into --rewrite, so a follow-up
        # run redoes just these images: pixi run extract-spot-labels all --input <dir>
        #   --rewrite <this file>
        flagged_csv = purple_dir / "flagged_bleed.csv"
        flagged_csv.write_text(
            "# images whose best draft still exceeded --max-bleed; redo with --rewrite\n"
            + "\n".join(flagged) + "\n",
            encoding="utf-8",
        )
        logger.info(f"flagged {len(flagged)} image(s) for bleed -> {flagged_csv}")

    if rejected:
        rejected_csv = anatomy_dir / "flagged_axis.csv"
        rejected_csv.write_text(
            "# images whose centre line the judge never accepted; redo with --rewrite\n"
            + "\n".join(rejected) + "\n",
            encoding="utf-8",
        )
        logger.info(f"flagged {len(rejected)} image(s) for a bad centre line -> {rejected_csv}")

    if fragmented:
        frag_csv = stage2.write_flagged_spots_csv(purple_dir, fragmented)
        logger.info(f"flagged {len(fragmented)} image(s) for fragmented spots -> {frag_csv}")

    done = total - failures
    logger.info(f"done: {done}/{total} images, {total_spots} spots "
          f"({total_binned} positioned) -> {db_path}"
          + (f", {escalated} needed escalation" if escalated else "")
          + (f", {len(flagged)} flagged for bleed" if flagged else "")
          + (f", {len(rejected)} rejected by the axis judge" if rejected else "")
          + (f", {len(fragmented)} flagged for fragmentation" if fragmented else "")
          + (f", {no_axis} without an axis (unpositioned)" if no_axis else "")
          + (f" ({failures} failed)" if failures else ""))
    base_log.write({"task": "extract", "event": "final", "images": total, "done": done,
                    "failed": failures, "spots": total_spots, "positioned": total_binned,
                    "escalated": escalated, "bleed_flagged": len(flagged),
                    "axis_rejected": len(rejected), "spots_fragmented": len(fragmented),
                    "no_axis": no_axis})
    logger.info(f"billed calls this stage: {base_log.billed}   log: {base_log.path}")
    return 1 if failures else 0
