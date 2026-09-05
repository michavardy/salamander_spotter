#!/usr/bin/env python3
"""Build a whole dataset end to end: purple + anatomy + contours, synthetic duplicates, package.

One command, five steps, one log. Every step is resumable — nothing already on disk is paid for
twice — so a run that dies (or runs out of API credits) can simply be re-run::

    pixi run build-dataset --dry-run          # the plan + what it will cost. ALWAYS DO THIS FIRST.
    pixi run build-dataset                    # the real thing (BILLED)
    pixi run build-dataset --limit 3          # a small end-to-end rehearsal
    pixi run build-dataset --rebuild          # NO API: re-derive the DB from on-disk artifacts,
                                              # correct axes, compute quality, package (free)

The steps, in the only order the dependencies allow (BILLED = Gemini calls; the rest are free)::

  1  extract       images/<input>/  -> purple/ + anatomy/ + contours/contours.db  BILLED
  2  correct-axis  fix mis-tipped body axes from the masks + re-bin them in the DB  (free)
  3  quality       cheap per-image markers -> contours.db image_quality table       (free)
  4  package       -> datasets/<name>/    (interim: step 5 reads a packaged dataset to pull
                                           singletons and raw images from)
  5  augment       individuals with only ONE photo -> images/synth_<name>/<label>_g0.png  BILLED
  6  extract       images/synth_<name>/ -> purple/ + anatomy/ + contours/contours.db  BILLED
  7  correct-axis  same, on the synthetic dir                                        (free)
  8  quality       same, on the synthetic dir                                        (free)
  9  package       datasets/<name>/  <- real photos AND synthetic views, merged + zipped

correct-axis and quality run right after each extract — corrections re-bin into that dir's DB in
place (no full contours re-run), and quality must be written before packaging because packaging
snapshots each source DB (so the image_quality table travels into the shipped dataset). Turn them
off with --skip-correct-axis / --skip-quality.

Step 4 is not redundant with step 9: `emb-gen-augment` reads a *packaged* dataset (its raw/ and
db/), so the real photos must be packaged before the synthetic ones can be generated from them.
Step 9 then rewrites the same dataset dir with both halves merged.

Every stage appends to ONE JSONL log — task, image, attempt, which model drew it, which model
judged it, the judgment, the decision, and how many billed calls it cost::

    artifacts/dataset_runs/<name>/pipeline_log.jsonl

so `--report` can tell you afterwards exactly what happened and what it cost.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from generate_spot_labels._common import (  # noqa: E402
    list_images, reconfigure_utf8, resolve_input_dir,
)
from generate_spot_labels.runlog import DONE, FAILED, SKIPPED, RunLog, new_run_id  # noqa: E402


def derive_label(sid: str) -> str:
    """`aj_1_2` -> `aj_1`; a synthetic `aj_1_g0` -> `aj_1` too."""
    return sid.rsplit("_", 1)[0] if "_" in sid else sid


# --- planning ---------------------------------------------------------------
def singletons(input_dir: Path) -> list[str]:
    """Individuals with exactly ONE photo — the ones a synthetic duplicate actually helps."""
    per = Counter(derive_label(p.stem) for p in list_images(input_dir))
    return sorted(lbl for lbl, n in per.items() if n == 1)


def count_missing(input_dir: Path, sub: str, suffix: str) -> int:
    """How many images still lack their `sub/<stem><suffix>` output (i.e. must be paid for)."""
    out = input_dir / sub
    return sum(1 for p in list_images(input_dir)
               if not (out / f"{p.stem}{suffix}").is_file())


def plan(args, real_dir: Path, synth_dir: Path, no_synth: bool = False) -> dict:
    n_real = len(list_images(real_dir))
    singles = singletons(real_dir)
    # --only-dir suppresses the whole synthetic side, so it costs nothing in the estimate.
    n_synth_want = 0 if no_synth else len(singles) * args.n_per
    n_synth_have = len(list_images(synth_dir)) if (synth_dir.is_dir() and not no_synth) else 0

    purple_todo = count_missing(real_dir, "purple", ".png")
    anat_todo = count_missing(real_dir, "anatomy", ".json")
    if args.limit:
        purple_todo = min(purple_todo, args.limit)
        anat_todo = min(anat_todo, args.limit)
        n_synth_want = min(n_synth_want, args.limit)

    augment_todo = max(0, n_synth_want - n_synth_have)
    # each synthetic view then needs its own purple + anatomy
    synth_purple = synth_anat = augment_todo + n_synth_have

    # What one anatomy image costs depends entirely on the mode:
    #   mask    the model paints the body once and the geometry is computed from it. Grading is
    #           free (geometric gates), so a call is a call: 1 best case, and at worst the whole
    #           ladder of re-paints.
    #   outline the model draws two flanks and an LLM JUDGE grades them — so every attempt is
    #           TWO billed calls, and the measured rejection rate was 73 %, meaning the worst
    #           case is very close to the typical case.
    mask_mode = args.anatomy_mode == "mask"
    per_attempt = 1 if mask_mode else 2         # judge call, or not
    attempts = args.anatomy_attempts or (2 if mask_mode else 1)
    rungs = 1 if mask_mode else 2               # escalate models in the default ladder
    anat_worst = per_attempt * (attempts + rungs)

    return {
        "n_real": n_real,
        "n_singletons": len(singles),
        "n_synth_have": n_synth_have,
        "purple_todo": purple_todo,
        "anatomy_todo": anat_todo,
        "augment_todo": augment_todo,
        "synth_purple": synth_purple,
        "synth_anatomy": synth_anat,
        "anatomy_mode": args.anatomy_mode,
        "anatomy_call_best": per_attempt,
        "anatomy_call_worst": anat_worst,
        "calls_min": (purple_todo + anat_todo * per_attempt + augment_todo
                      + synth_purple + synth_anat * per_attempt),
        "calls_max": (purple_todo * args.max_attempts
                      + anat_todo * anat_worst
                      + augment_todo
                      + synth_purple * args.max_attempts
                      + synth_anat * anat_worst),
    }


# --- steps ------------------------------------------------------------------
def run_cmd(cmd: list[str], log: RunLog, step: int, task: str, detail: str = "") -> int:
    """Run one child command, streaming its output, and log the outcome."""
    print("\n" + "=" * 78)
    print(f"STEP {step}  {task}")
    print(f"  $ {' '.join(cmd)}")
    print("=" * 78, flush=True)
    log.write({"step": step, "task": task, "event": "step_start",
               "command": " ".join(cmd), "detail": detail})
    t0 = time.monotonic()
    rc = subprocess.call([sys.executable, *cmd], cwd=str(REPO_ROOT))
    dt = time.monotonic() - t0
    log.write({"step": step, "task": task, "event": "step_end",
               "result": DONE if rc == 0 else FAILED, "exit_code": rc,
               "seconds": round(dt, 1)})
    print(f"\n[step {step}: {task}] exit={rc} in {dt / 60:.1f} min", flush=True)
    return rc


def extract_cmd(args, input_name: str, log_path: Path, run_id: str) -> list[str]:
    cmd = ["scripts/dataset/extract_spot_labels.py", "all", "--input", input_name,
           "--run-log", str(log_path), "--run-id", run_id,
           "--max-attempts", str(args.max_attempts),
           "--anatomy-mode", args.anatomy_mode,
           "--image-retries", str(args.image_retries)]
    if args.anatomy_attempts is not None:
        cmd += ["--anatomy-attempts", str(args.anatomy_attempts)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.model:
        cmd += ["--model", args.model]
    if args.judge_model:
        cmd += ["--judge-model", args.judge_model]
    if args.anatomy_escalate_models is not None:
        cmd += ["--anatomy-escalate-models", args.anatomy_escalate_models]
    return cmd


def correct_axis_cmd(input_name: str) -> list[str]:
    """Geometrically fix mis-tipped axes and re-bin them into the DB in place (free, no API)."""
    return ["scripts/dataset/correct_axis.py", "--input", input_name]


def quality_cmd(input_name: str) -> list[str]:
    """Compute the cheap per-image quality markers into image_quality (free, no API)."""
    return ["scripts/dataset/compute_quality.py", "--input", input_name]


def run_rebuild(args, name: str, synth_name: str, synth_dir: Path, log_path: Path,
                no_synth: bool = False) -> int:
    """Rebuild the DB + package from EXISTING artifacts — no API calls.

    Every Gemini product (purple, anatomy, body masks) is already on disk, so this re-derives
    everything downstream of them for free, per input dir:

        contours      purple + anatomy  -> spots, per-spot masks, bins, the body mask, body_axis
        correct-axis  geometric axis fix, re-binned into the DB in place
        quality       the image_quality markers

    then packages. Use it to rebuild or REFRESH a dataset after the paid stages are done —
    e.g. to fold newly-stored columns (the whole-body mask, image_quality) into a DB that was
    extracted before they existed, without paying to regenerate a single image. Because it calls
    only ``contours`` (never ``all``), it can never reach the network. Skips the augment and both
    extract-from-Gemini steps entirely.
    """
    inputs = ([args.input] if no_synth
              else [args.input] + ([synth_name] if synth_dir.is_dir() else []))
    print("=" * 78)
    print(f"REBUILD (no API)  {name}")
    print("=" * 78)
    print(f"  inputs : {', '.join(inputs)}")
    print(f"  output : datasets/{name}/")
    chain = ["contours"] + (["correct-axis"] if not args.skip_correct_axis else []) \
        + (["quality"] if not args.skip_quality else [])
    print(f"  per dir: {' -> '.join(chain)}   then package")
    print(f"  BILLED CALLS: 0  (re-derived from artifacts already on disk)")
    print("=" * 78)

    steps: list[tuple[int, str, list[str]]] = []
    n = 1
    for inp in inputs:
        steps.append((n, f"contours-{inp}",
                      ["scripts/dataset/extract_spot_labels.py", "contours", "--input", inp]))
        n += 1
        if not args.skip_correct_axis:
            steps.append((n, f"correct-axis-{inp}", correct_axis_cmd(inp)))
            n += 1
        if not args.skip_quality:
            steps.append((n, f"quality-{inp}", quality_cmd(inp)))
            n += 1
    steps.append((n, "package",
                  ["scripts/dataset/package_dataset.py", "--input", *inputs, "--name", name]
                  + (["--no-zip"] if args.no_zip else [])))

    if args.dry_run:
        print("\nplanned steps:")
        for s, task, _ in steps:
            print(f"  {s}. {task}")
        print("\n--dry-run: nothing was run, nothing was billed.")
        return 0

    run_id = new_run_id()
    log = RunLog(log_path, run_id=run_id, echo=False)
    log.write({"event": "rebuild_start", "dataset": name, "inputs": inputs, "args": vars(args)})
    print(f"\nrun_id={run_id}\n")
    for step, task, cmd in steps:
        if step < args.from_step:
            print(f"[step {step}: {task}] skipped (--from-step {args.from_step})")
            log.write({"step": step, "task": task, "result": SKIPPED})
            continue
        rc = run_cmd(cmd, log, step, task)
        if rc != 0:
            print(f"\nSTOPPED at step {step} ({task}) with exit {rc}.")
            print(f"Fix the cause, then resume:  "
                  f"pixi run build-dataset --rebuild --from-step {step}")
            log.write({"event": "run_abort", "step": step, "task": task, "exit_code": rc})
            return rc
    log.write({"event": "rebuild_end", "dataset": name, "result": DONE})
    print("\n" + "=" * 78)
    print(f"REBUILT  -> datasets/{name}/   (0 API calls)")
    print("=" * 78)
    return 0


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="all_sasa_norm", help="source image dir under images/")
    p.add_argument("--name", default=None,
                   help="dataset name (default: <input>_<YYYY_DD_MM>, matching the existing "
                        "datasets/ naming)")
    p.add_argument("--n-per", type=int, default=1,
                   help="synthetic views per singleton individual (default 1)")
    p.add_argument("--limit", type=int, default=0,
                   help="cap images per step — for an end-to-end rehearsal (0 = no cap)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and the billed-call estimate, then stop")
    p.add_argument("--report", action="store_true",
                   help="summarise an existing run's log and exit (no work, no calls)")
    p.add_argument("--from-step", type=int, default=1, help="resume from this step (1-9)")
    p.add_argument("--max-attempts", type=int, default=3, help="purple re-draws per image")
    p.add_argument("--anatomy-mode", choices=("mask", "outline"), default="mask",
                   help="'mask' (default): Gemini paints the body, the centre line is computed "
                        "from it, no judge. 'outline': the old draw-two-flanks + LLM-judge "
                        "method (73%% rejected, ~6 calls/image)")
    p.add_argument("--anatomy-attempts", type=int, default=None,
                   help="anatomy drafts on the PRIMARY model before escalating "
                        "(default: 2 in mask mode, 1 in outline mode)")
    p.add_argument("--image-retries", type=int, default=3,
                   help="re-asks when a model answers with prose instead of drawing")
    p.add_argument("--model", default=None, help="override GEMINI_MODEL (the image model)")
    p.add_argument("--judge-model", default=None, help="override the anatomy judge model")
    p.add_argument("--anatomy-escalate-models", default=None,
                   help="ladder of pricier drawing models ('' disables escalation)")
    p.add_argument("--no-zip", action="store_true", help="skip the final .zip")
    p.add_argument("--only-dir", default=None, metavar="NAME",
                   help="operate on ONLY this image dir: skip synthetic generation/extraction "
                        "and package it alone. Use after you have folded the synthetic views "
                        "INTO the main dir, so the separate synth_* dir must be ignored.")
    p.add_argument("--rebuild", action="store_true",
                   help="NO API: re-derive the whole DB from the artifacts already on disk "
                        "(contours + body masks + bins), correct axes, compute quality, and "
                        "package. Use after the Gemini stages are done to rebuild/refresh the "
                        "database and dataset for free.")
    p.add_argument("--skip-correct-axis", action="store_true",
                   help="don't run the geometric axis-correction step (free; on by default)")
    p.add_argument("--skip-quality", action="store_true",
                   help="don't compute the image_quality markers (free; on by default)")
    args = p.parse_args(argv)

    # --only-dir NAME means "this one dir IS the dataset; there is no separate synth side".
    if args.only_dir:
        args.input = args.only_dir
    no_synth = bool(args.only_dir)

    name = args.name or f"{args.input}_{date.today():%Y_%d_%m}"
    real_dir = resolve_input_dir(args.input)
    synth_name = f"synth_{name}"
    synth_dir = REPO_ROOT / "images" / synth_name

    log_dir = REPO_ROOT / "artifacts" / "dataset_runs" / name
    log_path = log_dir / "pipeline_log.jsonl"

    if args.report:
        return report(log_path)

    if args.rebuild:
        return run_rebuild(args, name, synth_name, synth_dir, log_path, no_synth)

    pl = plan(args, real_dir, synth_dir, no_synth)
    print("=" * 78)
    print(f"BUILD DATASET  {name}")
    print("=" * 78)
    print(f"  source images        : images/{args.input}/  ({pl['n_real']} photos)")
    print(f"  individuals w/ 1 photo: {pl['n_singletons']}  -> {args.n_per} synthetic view each")
    print(f"  synthetic dir        : images/{synth_name}/  ({pl['n_synth_have']} already there)")
    print(f"  output dataset       : datasets/{name}/")
    print(f"  run log              : {log_path}")
    print()
    print("  work remaining (already-done files are never paid for twice):")
    print(f"    step 1  purple      {pl['purple_todo']:>5} images")
    print(f"            anatomy     {pl['anatomy_todo']:>5} images  "
          f"({pl['anatomy_mode']} mode: {pl['anatomy_call_best']} call each, "
          f"up to {pl['anatomy_call_worst']})")
    print(f"    step 5  augment     {pl['augment_todo']:>5} synthetic views")
    print(f"    step 6  purple      {pl['synth_purple']:>5} synthetic")
    print(f"            anatomy     {pl['synth_anatomy']:>5} synthetic")
    ax = "off (--skip-correct-axis)" if args.skip_correct_axis else "on"
    qa = "off (--skip-quality)" if args.skip_quality else "on"
    print(f"    steps 2/3, 7/8   correct-axis: {ax}   quality: {qa}   (both free, no API)")
    print()
    print(f"  BILLED CALLS: ~{pl['calls_min']} best case, up to ~{pl['calls_max']} if every "
          f"draft is rejected and the ladder is climbed every time.")
    if args.limit:
        print(f"  (--limit {args.limit} is capping each step)")
    print("=" * 78)

    if args.dry_run:
        print("\n--dry-run: nothing was run, nothing was billed.")
        return 0

    run_id = new_run_id()
    log = RunLog(log_path, run_id=run_id, echo=False)
    log.write({"event": "run_start", "dataset": name, "input": args.input,
               "plan": pl, "args": vars(args)})
    print(f"\nrun_id={run_id}\n")

    # correct-axis and compute-quality run per input dir, right after that dir is extracted (so
    # corrections re-bin into its DB in place and quality sees the final bins) and before it is
    # packaged (packaging snapshots each source DB, so the image_quality table travels with it).
    # Both are free — no API — so they are on by default; skip with --skip-correct-axis/-quality.
    steps: list[tuple[int, str, list[str]]] = [
        (1, "extract-real", extract_cmd(args, args.input, log_path, run_id)),
        (2, "correct-axis-real", correct_axis_cmd(args.input)),
        (3, "quality-real", quality_cmd(args.input)),
        (4, "package-interim",
         ["scripts/dataset/package_dataset.py", "--input", args.input, "--name", name, "--no-zip"]),
        (5, "augment-singletons",
         ["scripts/embedding/emb_gen_augment.py", "--dataset", name, "--which", "singletons",
          "--n-per", str(args.n_per), "--limit", "0"]),
        (6, "extract-synth", extract_cmd(args, synth_name, log_path, run_id)),
        (7, "correct-axis-synth", correct_axis_cmd(synth_name)),
        (8, "quality-synth", quality_cmd(synth_name)),
        (9, "package-final",
         ["scripts/dataset/package_dataset.py", "--input", args.input]
         + ([] if no_synth else [synth_name])
         + ["--name", name] + (["--no-zip"] if args.no_zip else [])),
    ]

    for step, task, cmd in steps:
        if step < args.from_step:
            print(f"[step {step}: {task}] skipped (--from-step {args.from_step})")
            log.write({"step": step, "task": task, "result": SKIPPED})
            continue
        if task.startswith("correct-axis") and args.skip_correct_axis:
            print(f"[step {step}: {task}] skipped (--skip-correct-axis)")
            log.write({"step": step, "task": task, "result": SKIPPED})
            continue
        if task.startswith("quality") and args.skip_quality:
            print(f"[step {step}: {task}] skipped (--skip-quality)")
            log.write({"step": step, "task": task, "result": SKIPPED})
            continue
        # --only-dir suppresses the whole synthetic side (augment + the synth extract/fix/quality)
        if no_synth and (task == "augment-singletons" or task.endswith("-synth")):
            print(f"[step {step}: {task}] skipped (--only-dir {args.input})")
            log.write({"step": step, "task": task, "result": SKIPPED,
                       "detail": "only-dir: no synthetic side"})
            continue
        if step == 5 and pl["augment_todo"] == 0:
            print(f"[step {step}: {task}] nothing to generate — every singleton already has a view")
            log.write({"step": step, "task": task, "result": SKIPPED,
                       "detail": "all singletons already augmented"})
            continue
        # synth-side steps have nothing to act on until the synthetic dir exists
        if task.endswith("-synth") and not synth_dir.is_dir():
            print(f"[step {step}: {task}] no synthetic dir yet — skipping")
            log.write({"step": step, "task": task, "result": SKIPPED,
                       "detail": "no synthetic dir"})
            continue

        rc = run_cmd(cmd, log, step, task)
        if rc != 0:
            print(f"\nSTOPPED at step {step} ({task}) with exit {rc}.")
            print(f"Fix the cause, then resume:  pixi run build-dataset --from-step {step}")
            log.write({"event": "run_abort", "step": step, "task": task, "exit_code": rc})
            return rc

    log.write({"event": "run_end", "dataset": name, "result": DONE})
    print("\n" + "=" * 78)
    print(f"DONE  -> datasets/{name}/")
    print(f"Report:  pixi run build-dataset --name {name} --report")
    print("=" * 78)
    return 0


# --- reporting --------------------------------------------------------------
def report(log_path: Path) -> int:
    """Summarise a run log: what ran, which models, how many calls, what failed."""
    import json

    if not log_path.is_file():
        print(f"no log at {log_path}", file=sys.stderr)
        return 1
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    print(f"{log_path}   ({len(rows)} events)")
    runs = sorted({r.get("run_id") for r in rows if r.get("run_id")})
    print(f"runs: {', '.join(runs)}\n")

    by_task = Counter(r.get("task") for r in rows if r.get("task"))
    print("events per task:")
    for t, n in by_task.most_common():
        print(f"  {t:<12} {n}")

    calls = sum(int(r.get("billed_calls") or 0) for r in rows)
    print(f"\nBILLED CALLS: {calls}")
    draws = Counter(r["draw_model"] for r in rows if r.get("draw_model"))
    if draws:
        print("\ncalls per drawing model:")
        for m, n in draws.most_common():
            print(f"  {m:<28} {n}")
    judges = Counter(r["judge_model"] for r in rows if r.get("judge_model"))
    if judges:
        print("\ncalls per judge model:")
        for m, n in judges.most_common():
            print(f"  {m:<28} {n}")

    results = Counter(r["result"] for r in rows if r.get("result"))
    print("\noutcomes:")
    for k, n in results.most_common():
        print(f"  {k:<24} {n}")

    # which anatomy criteria failed most — tells you whether to fix the prompt or the ladder
    crit = Counter()
    for r in rows:
        j = r.get("judgment")
        if isinstance(j, dict):
            for k, v in j.items():
                if not v:
                    crit[k] += 1
    if crit:
        print("\nanatomy judge — most-failed criteria:")
        for k, n in crit.most_common():
            print(f"  {k:<20} {n}")

    fails = [r for r in rows if r.get("result") == FAILED]
    if fails:
        print(f"\nFAILURES ({len(fails)}):")
        for r in fails[:15]:
            print(f"  [{r.get('task')}] {r.get('image') or r.get('step')}: "
                  f"{str(r.get('detail'))[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
