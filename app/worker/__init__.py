"""Background job queue (spec §5.3).

A ThreadPoolExecutor + a ``jobs`` row per unit of work — not Celery/Redis
(single machine, keep it simple). Progress and "queue changed" events are
published on an in-process bus that the SSE endpoint (``/api/events``) drains.

Training is the exception: it runs as its own subprocess (see
``app/services/training.py``) so an OOM cannot take the server down.
"""

from __future__ import annotations

import json
import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ..db import Database, new_id

JobFn = Callable[["JobContext"], Any]


@dataclass
class JobContext:
    job_id: str
    params: dict
    _worker: "Worker"

    def progress(self, fraction: float, message: str | None = None) -> None:
        self._worker._update(self.job_id, progress=max(0.0, min(1.0, fraction)), message=message)
        self._worker.publish("job_progress", {"job_id": self.job_id, "progress": fraction, "message": message})


class EventBus:
    """Fan-out of small JSON events to any number of subscriber queues."""

    def __init__(self) -> None:
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def publish(self, event: str, data: dict) -> None:
        payload = {"event": event, "data": data, "ts": _utcnow().isoformat()}
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass


class Worker:
    def __init__(self, db: Database, *, max_workers: int = 2):
        self.db = db
        self.bus = EventBus()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="spotter-job")
        self._registry: dict[str, JobFn] = {}
        self._cancelled: set[str] = set()

    # -- registration ------------------------------------------------------
    def register(self, job_type: str, fn: JobFn) -> None:
        self._registry[job_type] = fn

    def publish(self, event: str, data: dict) -> None:
        self.bus.publish(event, data)

    # -- submission ------------------------------------------------------
    def submit(self, job_type: str, params: dict | None = None) -> str:
        if job_type not in self._registry:
            raise KeyError(f"no job runner registered for {job_type!r}")
        job_id = new_id("job")
        params = params or {}
        self.db.insert(
            "jobs",
            {
                "id": job_id,
                "type": job_type,
                "status": "queued",
                "progress": 0.0,
                "params_json": json.dumps(params),
            },
        )
        self.publish("queue_changed", {"job_id": job_id, "type": job_type, "status": "queued"})
        self._pool.submit(self._run, job_id, job_type, params)
        return job_id

    def run_sync(self, job_type: str, params: dict | None = None, *, timeout_s: float = 600.0) -> dict:
        """Submit and block until done — for tests, the CLI, and short API calls.

        Not for genuinely long jobs (full-dataset import, backup/export of a large
        data dir, training): those should ``submit`` and let the client poll
        :meth:`status` (or ``GET /api/jobs/{id}``) instead, since a client-facing
        HTTP request has no business blocking for minutes.
        """
        job_id = self.submit(job_type, params)
        import time

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            row = self.status(job_id)
            if row and row["status"] in {"done", "failed", "cancelled"}:
                return row
            time.sleep(0.05)
        raise TimeoutError(f"job {job_id} did not finish within {timeout_s}s")

    def status(self, job_id: str) -> dict | None:
        row = self.db.query_one(
            "SELECT id AS job_id, type, status, progress, message, result_json, error, "
            "created_at, updated_at FROM jobs WHERE id = ?",
            [job_id],
        )
        if not row:
            return None
        row["result"] = json.loads(row.pop("result_json")) if row["result_json"] else None
        return row

    def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)
        self._update(job_id, status="cancelled")
        self.publish("queue_changed", {"job_id": job_id, "status": "cancelled"})

    # -- execution ------------------------------------------------------
    def _run(self, job_id: str, job_type: str, params: dict) -> None:
        if job_id in self._cancelled:
            return
        self._update(job_id, status="running")
        self.publish("queue_changed", {"job_id": job_id, "type": job_type, "status": "running"})
        ctx = JobContext(job_id=job_id, params=params, _worker=self)
        try:
            result = self._registry[job_type](ctx)
            self._update(job_id, status="done", progress=1.0,
                         result_json=json.dumps(result, default=str))
            self.publish("job_done", {"job_id": job_id, "type": job_type, "result": result})
        except Exception as exc:  # noqa: BLE001 - jobs must never crash the worker
            self._update(job_id, status="failed",
                         error=f"{exc}\n{traceback.format_exc()}")
            self.publish("job_failed", {"job_id": job_id, "type": job_type, "error": str(exc)})

    def _update(self, job_id: str, **fields: Any) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(
            f"UPDATE jobs SET {sets}, updated_at = now() WHERE id = ?",
            [*fields.values(), job_id],
        )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
