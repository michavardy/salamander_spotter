#!/usr/bin/env python3
"""One structured, append-only log for an entire dataset build.

Every stage writes JSONL lines to the SAME file, so a whole run — hundreds of images, thousands
of billed calls — is auditable afterwards from one place: which task, which image, which attempt,
which model drew it, which model judged it, what the judgment was, and what was decided.

    {"ts": "...", "run_id": "20260714_101500", "step": 3, "task": "anatomy",
     "input": "all_sasa_norm", "image": "aa_1_1.jpg", "attempt": 2, "max_attempts": 3,
     "draw_model": "gemini-3.1-flash-image", "judge_model": "gemini-3.5-flash",
     "judgment": {"dots_at_tips": true, ...}, "result": "image accepted",
     "billed_calls": 2, "seconds": 8.4}

Design rules:
  * Never raises. A broken log must not kill a run that is spending real money.
  * Console and file come from the SAME record, so what you watch is what is stored.
  * `billed_calls` is recorded per event, so the cost of a run is a `sum()` over the log rather
    than a guess.
"""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime
from pathlib import Path

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

# result strings, shared so the log stays greppable
ACCEPTED = "image accepted"
REGENERATE = "regenerate required"
SKIPPED = "skipped (already done)"
FAILED = "failed"
DONE = "done"


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class RunLog:
    """Append-only JSONL log shared by every stage of a build."""

    def __init__(self, path: Path, run_id: str | None = None, echo: bool = True,
                 step: int | None = None, task: str = "", input: str = ""):
        self.path = Path(path)
        self.run_id = run_id or new_run_id()
        self.echo = echo
        self.step = step
        self.task = task
        self.input = input
        self.billed = 0
        # Guards the append + billed tally so parallel (--workers) stages can share one log.
        # Children write the SAME file, so they must share the SAME lock (set in child()).
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def child(self, *, step: int | None = None, task: str = "", input: str = "") -> "RunLog":
        """A view on the same file that stamps a different step/task/input."""
        c = RunLog(self.path, run_id=self.run_id, echo=self.echo,
                   step=step if step is not None else self.step,
                   task=task or self.task, input=input or self.input)
        c._lock = self._lock  # same file -> same lock
        return c

    # -- the one write path ---------------------------------------------------
    def write(self, rec: dict) -> dict:
        rec = {"ts": datetime.now().isoformat(timespec="seconds"),
               "run_id": self.run_id,
               **({"step": self.step} if self.step is not None else {}),
               **({"task": rec.pop("task", None) or self.task} if (self.task or rec.get("task")) else {}),
               **({"input": self.input} if self.input else {}),
               **rec}
        line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            self.billed += int(rec.get("billed_calls") or 0)
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line)
            except Exception as exc:  # a broken log must never kill a billed run
                logger.warning(f"  warning: could not write {self.path}: {exc}")
        return rec

    def event(self, message: str = "", **fields) -> dict:
        """A generic line. `message` is echoed; everything else is stored."""
        rec = self.write(fields)
        if self.echo and message:
            logger.info(message)
        return rec

    # -- the draw/judge loop (stage 1b) --------------------------------------
    def attempt(self, *, image: str, attempt: int, max_attempts: int, draw_model: str,
                judge_model: str | None, verdict, result: str, anatomy, note: str = "",
                task: str = "anatomy") -> dict:
        """One draw+judge round: which model drew, which judged, the judgment, the decision."""
        judgment = None
        if verdict is not None:
            judgment = {"dots_at_tips": verdict.dots_at_tips,
                        "goes_head_to_tail": verdict.goes_head_to_tail,
                        "follows_edges": verdict.follows_edges,
                        "opposite_sides": verdict.opposite_sides}
        rec = self.write({
            "task": task,
            "image": image,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "draw_model": draw_model,
            "judge_model": judge_model,
            "judgment": judgment,
            "result": result,
            "feedback": (verdict.feedback if verdict else note),
            "axis_len_px": round(anatomy.length_px, 1) if anatomy.ok else None,
            "axis_source": anatomy.source,
            "billed_calls": 1 + (1 if judgment is not None else 0),
        })

        if self.echo:
            logger.info(f"  attempt {attempt}/{max_attempts}  draw={draw_model}  "
                  f"judge={judge_model or 'OFF'}")
            if judgment is not None:
                logger.info("    judgment: " + "  ".join(
                    f"{k}={'PASS' if v else 'FAIL'}" for k, v in judgment.items()))
            if result == ACCEPTED:
                extra = (f"axis {anatomy.length_px:.0f}px, {len(anatomy.midline)} pts, "
                         f"{anatomy.source}") if anatomy.ok else note
                logger.info(f"    -> IMAGE ACCEPTED ({extra})")
            else:
                why = (verdict.feedback if verdict and verdict.feedback else note)
                logger.info(f"    -> REGENERATE REQUIRED: {why}")
        return rec

    def gated(self, *, image: str, attempt: int, max_attempts: int, draw_model: str,
              checks: dict, result: str, anatomy, note: str = "",
              task: str = "anatomy") -> dict:
        """One paint round, graded by :mod:`body_mask`'s free gates instead of by a judge model.

        The gate results go into the same ``judgment`` field the LLM judge used, so everything
        downstream that reads the log — `build-dataset --report`'s "most-failed criteria" tally
        above all — keeps working unchanged; only the criteria names differ (`line_inside`
        rather than `follows_edges`). ``billed_calls`` is 1, not 2: grading is free now.
        """
        rec = self.write({
            "task": task,
            "image": image,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "draw_model": draw_model,
            "judge_model": None,
            "judgment": dict(checks) if checks else None,
            "result": result,
            "feedback": note,
            "axis_len_px": round(anatomy.length_px, 1) if anatomy.ok else None,
            "axis_source": anatomy.source,
            "billed_calls": 1,
        })
        if self.echo:
            logger.info(f"  attempt {attempt}/{max_attempts}  paint={draw_model}  judge=NONE (gates)")
            if checks:
                logger.info("    gates: " + "  ".join(
                    f"{k}={'PASS' if v else 'FAIL'}" for k, v in checks.items()))
            if result == ACCEPTED:
                logger.info(f"    -> IMAGE ACCEPTED (axis {anatomy.length_px:.0f}px, "
                      f"{len(anatomy.midline)} pts, {anatomy.source})")
            else:
                logger.info(f"    -> REGENERATE REQUIRED: {note}")
        return rec

    def summary(self, *, image: str, attempts: int, accepted: bool, anatomy,
                task: str = "anatomy") -> dict:
        return self.write({
            "task": task,
            "image": image,
            "event": "final",
            "attempts": attempts,
            "accepted": accepted,
            "judged_ok": anatomy.judged_ok,
            "axis_len_px": round(anatomy.length_px, 1) if anatomy.ok else None,
            "axis_source": anatomy.source,
            "feedback": anatomy.judge_feedback,
        })
