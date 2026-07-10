#!/usr/bin/env python3
"""Orchestration logic for "generate spot labels" — Gemini purple + contours + masks.

Pure logic: no argument parsing here. The CLI that configures and calls :func:`run` is
``scripts/extract_spot_labels.py all`` (``pixi run extract-spot-labels all``).

For every salamander image in ``input`` (one at a time):

    stage 1  image  -> Gemini inpaint -> <input_dir>/purple/<stem>.png
    stage 2  purple -> cv2 magenta keying + per-spot contours & masks -> contours/contours.db

Stage 2 writes one row per spot into the ``spots`` table, each carrying both a
centroid-local contour and a full-frame ``mask_png`` (black except that spot); there is
no separate masks/ directory.

Stage 1 judges every draft against the original with a bleed check and re-draws; if the
primary model never beats ``max_bleed`` it can escalate through the pricier
``escalate_models`` ladder before giving up and flagging the image.

Stage 1 already-done purple images are reused unless ``overwrite`` is set, so re-runs
only pay for images that still need the (billed) Gemini call. Pass a ``rewrite`` CSV of
image names to redo only those (regenerating their purple and replacing their DB rows).
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import extract_spot_contours as stage2
from . import llm_spot_segmentation as stage1
from ._common import (
    contours_db_for,
    list_images,
    purple_dir_for,
    read_rewrite_names,
    resolve_input_dir,
    resolve_rewrite_csv,
)


def run(
    input: str,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    model: str | None = None,
    eps_frac: float = stage2.DEFAULT_EPS_FRAC,
    min_area: float = stage2.DEFAULT_MIN_AREA,
    morph: int = stage2.DEFAULT_MORPH,
    keep_gemini_size: bool = False,
    rewrite: str | None = None,
    max_attempts: int = stage1.DEFAULT_MAX_ATTEMPTS,
    max_bleed: float = stage1.DEFAULT_MAX_BLEED,
    temperature: float | None = None,
    escalate_models: str | list[str] | None = None,
    escalate_attempts: int = stage1.DEFAULT_ESCALATE_ATTEMPTS,
) -> int:
    """Run the full pipeline over one input dir. Returns a process exit code.

    ``rewrite`` is a CSV of image names to redo: only those images are processed,
    their purple is regenerated via Gemini (implies ``overwrite``), and their DB
    rows are replaced.
    """
    input_dir = resolve_input_dir(input)
    purple_dir = purple_dir_for(input_dir)
    purple_dir.mkdir(parents=True, exist_ok=True)
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
        print(f"rewrite mode: {len(images)} image(s) from {csv_path}")
        if missing:
            print(f"warning: {len(missing)} name(s) in CSV not found in {input_dir.name}/: "
                  + ", ".join(missing), file=sys.stderr)

    if limit is not None:
        images = images[:limit]
    if not images:
        where = "rewrite CSV" if rewrite else input_dir
        print(f"error: no images to process ({where})", file=sys.stderr)
        return 1

    segmenter = stage1.GeminiSpotSegmenter(
        model=model, temperature=temperature,
        escalate_models=escalate_models, escalate_attempts=escalate_attempts,
    )
    store = stage2.ContourStore(db_path)

    total = len(images)
    print(f"processing images in dir {input_dir}")
    failures = 0
    total_spots = 0
    escalated = 0            # images whose accepted draft came from the escalate model
    flagged: list[str] = []  # image names whose best draft still bled — for review
    try:
        for i, src in enumerate(images, start=1):
            print(f"\nprocessing image {i} / {total}: {src.name}")
            purple_path = stage1.purple_path_for(purple_dir, src)

            # --- stage 1: purple image from Gemini (judged + retried) ---
            print("stage 1: extracting purple image from gemini")
            if purple_path.exists() and not overwrite:
                print("  purple exists, reusing (use --overwrite to redo)")
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
                        print(f"  saved {purple_path.relative_to(input_dir)} "
                              f"[bleed={res.bleed:.2f}, {res.attempts} draw(s), {res.model}]")
                    else:
                        flagged.append(src.name)
                        print(f"  saved {purple_path.relative_to(input_dir)} "
                              f"[bleed={res.bleed:.2f} > {max_bleed}, FLAGGED after "
                              f"{res.attempts} draw(s), {res.model}]")
                except Exception as exc:
                    failures += 1
                    print(f"  ERROR (stage 1): {exc}", file=sys.stderr)
                    continue  # can't do stage 2 without a purple image

            # --- stage 2: contour data from the purple image ---
            print("stage 2: extracting contour data from image")
            try:
                rec = stage2.process_purple_image(
                    purple_path, store,
                    eps_frac=eps_frac, min_area=min_area, morph=morph,
                    source_image=src.name,
                )
                n = len(rec["spots"])
                total_spots += n
                print(f"saved to db ({n} spots + per-spot masks)")
            except Exception as exc:
                failures += 1
                print(f"  ERROR (stage 2): {exc}", file=sys.stderr)
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
        print(f"flagged {len(flagged)} image(s) for bleed -> {flagged_csv}")

    done = total - failures
    print(f"\ndone: {done}/{total} images, {total_spots} spots -> {db_path}"
          + (f", {escalated} needed escalation" if escalated else "")
          + (f", {len(flagged)} flagged for bleed" if flagged else "")
          + (f" ({failures} failed)" if failures else ""))
    return 1 if failures else 0
