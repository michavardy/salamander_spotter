"""Spec §5.3 — background job queue."""

from __future__ import annotations

import pytest

from app.worker import Worker


def test_job_runs_and_records_result(db):
    w = Worker(db)
    w.register("double", lambda ctx: {"out": ctx.params["n"] * 2})
    result = w.run_sync("double", {"n": 21})
    assert result["status"] == "done"
    assert result["result"] == {"out": 42}
    row = db.query_one("SELECT * FROM jobs WHERE id = ?", [result["job_id"]])
    assert row["status"] == "done" and row["progress"] == 1.0
    w.shutdown()


def test_job_failure_is_captured(db):
    w = Worker(db)
    def boom(ctx):
        raise RuntimeError("kaboom")
    w.register("boom", boom)
    result = w.run_sync("boom")
    assert result["status"] == "failed"
    assert "kaboom" in db.query_one("SELECT error FROM jobs WHERE id = ?", [result["job_id"]])["error"]
    w.shutdown()


def test_progress_events_reach_subscribers(db):
    w = Worker(db)
    q = w.bus.subscribe()
    def stepper(ctx):
        ctx.progress(0.5, "halfway")
        return "ok"
    w.register("step", stepper)
    w.run_sync("step")
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    kinds = {e["event"] for e in events}
    assert "job_progress" in kinds and "job_done" in kinds
    w.shutdown()


def test_unregistered_job_type_raises(db):
    w = Worker(db)
    with pytest.raises(KeyError):
        w.submit("nope")
    w.shutdown()


def test_run_sync_times_out_on_a_genuinely_slow_job(db):
    import time

    w = Worker(db)
    w.register("slow", lambda ctx: time.sleep(1))
    with pytest.raises(TimeoutError):
        w.run_sync("slow", timeout_s=0.2)
    w.shutdown()


def test_status_matches_run_sync_shape(db):
    w = Worker(db)
    w.register("double", lambda ctx: {"out": ctx.params["n"] * 2})
    job_id = w.submit("double", {"n": 5})
    import time

    for _ in range(200):
        s = w.status(job_id)
        if s["status"] == "done":
            break
        time.sleep(0.01)
    assert s["result"] == {"out": 10}
    assert w.status("nope") is None
    w.shutdown()
