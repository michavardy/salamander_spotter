"""Training seam (spec §7.5, M5).

A training run snapshots the reviewed dataset, retrains every registered trainable
model on the full snapshot, and evaluates each on the held eval split. The real
bridge shells out to ``pipeline/spot_transformer/sweeps/train_all13.py`` in a
subprocess so a crash/OOM can't take the server down; tests use the fake.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "pipeline" / "spot_transformer" / "sweeps" / "train_all13.py"
_PROGRESS_RE = re.compile(r"^PROGRESS ([0-9.]+) (.*)$")
_RELAY_TAIL_LINES = 40  # remote-logger snapshots carry only the recent tail, not the whole log

ProgressFn = Callable[[float, str], None]


def _append_relay_entry(relay_path: Path, tail: "deque[str]", *, final: bool, returncode: int | None = None) -> None:
    """Append one JSON-line snapshot for the remote-logger feature (spec: settings
    ``remote_logger_enabled`` / ``remote_logger_interval_minutes``). Best-effort: a relay
    write must never take down a real training run, so failures are swallowed."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tail": list(tail),
        "final": final,
    }
    if returncode is not None:
        entry["returncode"] = returncode
    try:
        relay_path.parent.mkdir(parents=True, exist_ok=True)
        with open(relay_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


@dataclass(frozen=True)
class TrainedModel:
    name: str
    kind: str
    metrics: dict          # r1, r5, r10, bal_acc, novelty_auroc, review_at_90
    weights_path: str | None = None
    calibration_path: str | None = None


class TrainingBridge(Protocol):
    def train_and_eval(
        self, snapshot_id: str, models_dir: Path, contours_db_path: Path,
        *, log_path: Path | None = None, progress: ProgressFn | None = None,
        extra_env: dict[str, str] | None = None,
        relay_path: Path | None = None, relay_interval_s: float | None = None,
    ) -> list[TrainedModel]:
        ...


class FakeTrainingBridge:
    """Returns deterministic metrics; a named model can be made to 'improve'."""

    def __init__(self, models: list[tuple[str, str]] | None = None, boost: dict[str, float] | None = None):
        self.models = models or [("set_transformer", "aggregator"), ("cnn_baseline", "cnn")]
        self.boost = boost or {}
        self.runs = 0

    def train_and_eval(self, snapshot_id, models_dir, contours_db_path,
                       *, log_path=None, progress=None, extra_env=None,
                       relay_path=None, relay_interval_s=None):  # noqa: D102
        self.runs += 1
        if progress:
            progress(1.0, "done")
        out = []
        for name, kind in self.models:
            base = 0.40 + 0.05 * self.runs + self.boost.get(name, 0.0)
            out.append(
                TrainedModel(
                    name=name,
                    kind=kind,
                    metrics={
                        "r1": round(min(0.99, base), 4),
                        "r5": round(min(0.99, base + 0.15), 4),
                        "r10": round(min(0.99, base + 0.22), 4),
                        "bal_acc": round(min(0.99, base + 0.05), 4),
                        "novelty_auroc": round(min(0.99, base + 0.10 + self.boost.get(name, 0.0)), 4),
                        "review_at_90": round(min(0.99, base - 0.05), 4),
                    },
                    weights_path=str(Path(models_dir) / name / "weights.pt"),
                    calibration_path=str(Path(models_dir) / name / "calibration.json"),
                )
            )
        return out


class PipelineTrainingBridge:
    """Spawns ``train_all13.py`` as a subprocess, streams its stdout to ``log_path``,
    parses ``PROGRESS <frac> <message>`` lines into ``progress()`` calls, and — on a
    clean exit — reads the ``manifest.json`` it wrote into ``TrainedModel``\\ s.

    ``snapshot_id`` / ``contours_db_path`` are accepted for the seam's shape and
    provenance (they're what ``run_training`` already snapshots/records) but the
    training data itself comes from the pipeline's own configured research dataset
    (``pipeline/spot_transformer/core/data.py``), not the app's live per-deployment
    ``contours.db`` — matching how every model registered in this app so far was
    produced (see docs/salamander_spotter_spec.md M5).
    """

    def train_and_eval(self, snapshot_id, models_dir, contours_db_path,
                       *, log_path: Path | None = None, progress: ProgressFn | None = None,
                       extra_env: dict[str, str] | None = None,
                       relay_path: Path | None = None, relay_interval_s: float | None = None):
        if not TRAIN_SCRIPT.is_file():
            raise RuntimeError(f"training script not found: {TRAIN_SCRIPT}")

        proc = subprocess.Popen(
            [sys.executable, "-u", str(TRAIN_SCRIPT)],
            cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
            env={**os.environ, **(extra_env or {})},
        )
        log_file = None
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_path, "w", encoding="utf-8")
        # remote-logger (settings: remote_logger_enabled / remote_logger_interval_minutes) — off
        # unless a caller opts in with a positive interval, so this is a no-op by default.
        relay_on = bool(relay_path) and relay_interval_s and relay_interval_s > 0
        relay_tail: deque[str] = deque(maxlen=_RELAY_TAIL_LINES)
        last_relay = time.monotonic()
        manifest_dir: Path | None = None
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if log_file:
                    log_file.write(line)
                    log_file.flush()
                if relay_on:
                    relay_tail.append(line.rstrip("\n"))
                    now = time.monotonic()
                    if now - last_relay >= relay_interval_s:
                        _append_relay_entry(relay_path, relay_tail, final=False)
                        last_relay = now
                m = _PROGRESS_RE.match(line.strip())
                if m and progress:
                    progress(float(m.group(1)), m.group(2))
                wrote = re.match(r"^wrote (.*[/\\]manifest\.json)$", line.strip())
                if wrote:
                    manifest_dir = Path(wrote.group(1)).parent
        finally:
            proc.wait()
            if log_file:
                log_file.close()
            if relay_on:
                _append_relay_entry(relay_path, relay_tail, final=True, returncode=proc.returncode)

        if proc.returncode != 0:
            tail = f" — see log at {log_path}" if log_path else ""
            raise RuntimeError(
                f"training subprocess failed (exit {proc.returncode}){tail}"
            )
        if manifest_dir is None:
            raise RuntimeError(
                f"training subprocess exited cleanly but never reported a manifest.json"
                f" — see log at {log_path}" if log_path else
                "training subprocess exited cleanly but never reported a manifest.json"
            )

        manifest = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
        return [
            TrainedModel(
                name=entry["name"], kind=entry["kind"], metrics=entry["metrics"],
                weights_path=entry.get("weights_path"),
            )
            for entry in manifest
        ]
