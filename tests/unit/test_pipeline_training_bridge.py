"""app/pipeline_bridge/training.py::PipelineTrainingBridge — drives a tiny fixture
subprocess instead of the real hours-long train_all13.py, to check log streaming,
PROGRESS-line parsing, and the manifest.json -> TrainedModel mapping."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline_bridge import training as training_bridge
from app.pipeline_bridge.training import PipelineTrainingBridge


def _write_fixture_script(tmp_path, *, exit_code: int = 0, write_manifest: bool = True) -> Path:
    outdir = tmp_path / "run_out"
    outdir.mkdir()
    manifest = [
        {"name": "logreg", "kind": "feat",
         "metrics": {"r1": 0.5, "r5": None, "r10": None, "bal_acc": 0.6,
                     "novelty_auroc": 0.7, "review_at_90": 0.9},
         "weights_path": str(outdir / "logreg" / "weights.pt")},
    ]
    (outdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    script = tmp_path / "fixture_train.py"
    lines = [
        "print('PROGRESS 0.5000 fold 1 of 2 complete', flush=True)",
        "print('some other log line', flush=True)",
        "print('PROGRESS 1.0000 done', flush=True)",
    ]
    if write_manifest:
        lines.append(f"print('wrote ' + {str(outdir / 'manifest.json')!r})")
    lines.append(f"raise SystemExit({exit_code})")
    script.write_text("\n".join(lines), encoding="utf-8")
    return script


def test_streams_log_parses_progress_and_returns_manifest(tmp_path, monkeypatch):
    script = _write_fixture_script(tmp_path)
    monkeypatch.setattr(training_bridge, "TRAIN_SCRIPT", script)

    seen = []
    log_path = tmp_path / "job.log"
    bridge = PipelineTrainingBridge()
    trained = bridge.train_and_eval(
        "snap1", tmp_path / "models", tmp_path / "contours.db",
        log_path=log_path, progress=lambda frac, msg: seen.append((frac, msg)),
    )

    assert seen == [(0.5, "fold 1 of 2 complete"), (1.0, "done")]
    log_text = log_path.read_text(encoding="utf-8")
    assert "some other log line" in log_text
    assert len(trained) == 1
    assert trained[0].name == "logreg"
    assert trained[0].kind == "feat"
    assert trained[0].metrics["r1"] == 0.5
    assert trained[0].weights_path.endswith("weights.pt")


def test_nonzero_exit_raises_with_log_path(tmp_path, monkeypatch):
    script = _write_fixture_script(tmp_path, exit_code=3)
    monkeypatch.setattr(training_bridge, "TRAIN_SCRIPT", script)
    log_path = tmp_path / "job.log"

    bridge = PipelineTrainingBridge()
    with pytest.raises(RuntimeError, match="exit 3") as exc_info:
        bridge.train_and_eval("snap1", tmp_path / "models", tmp_path / "contours.db", log_path=log_path)
    assert str(log_path) in str(exc_info.value)


def test_clean_exit_without_manifest_raises(tmp_path, monkeypatch):
    script = _write_fixture_script(tmp_path, write_manifest=False)
    monkeypatch.setattr(training_bridge, "TRAIN_SCRIPT", script)

    bridge = PipelineTrainingBridge()
    with pytest.raises(RuntimeError, match="manifest"):
        bridge.train_and_eval("snap1", tmp_path / "models", tmp_path / "contours.db")


def test_remote_logger_off_by_default_writes_no_relay_file(tmp_path, monkeypatch):
    script = _write_fixture_script(tmp_path)
    monkeypatch.setattr(training_bridge, "TRAIN_SCRIPT", script)
    relay_path = tmp_path / "remote_relay" / "job1.jsonl"

    bridge = PipelineTrainingBridge()
    bridge.train_and_eval(
        "snap1", tmp_path / "models", tmp_path / "contours.db",
        log_path=tmp_path / "job.log",
    )  # relay_path/relay_interval_s not passed -> disabled

    assert not relay_path.exists()


def test_remote_logger_writes_a_final_relay_entry_when_enabled(tmp_path, monkeypatch):
    script = _write_fixture_script(tmp_path)
    monkeypatch.setattr(training_bridge, "TRAIN_SCRIPT", script)
    relay_path = tmp_path / "remote_relay" / "job1.jsonl"

    bridge = PipelineTrainingBridge()
    bridge.train_and_eval(
        "snap1", tmp_path / "models", tmp_path / "contours.db",
        log_path=tmp_path / "job.log",
        # a huge interval means the periodic tick never fires in this short-lived fixture run,
        # so the only entry we should see is the guaranteed final one written in `finally`
        relay_path=relay_path, relay_interval_s=3600,
    )

    lines = relay_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["final"] is True
    assert entry["returncode"] == 0
    assert any("some other log line" in l for l in entry["tail"])
