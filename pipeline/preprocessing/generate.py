"""Regenerate synthetic views for one real photo, extract them, and show them in the app.

A bad synthetic (``je_3_g0`` that does not carry ``je_3_1``'s pattern) is not a labelling problem,
it is a *generation* problem — so the review app gets a button that re-runs generation for that
source photo and brings the result back into the family view without a round trip through the
whole packaging pipeline.

The job is three **existing, documented commands** run as subprocesses, so what executes is exactly
what an operator would type and the job log is the real output:

1. ``pixi run emb-gen-augment --dataset <ds> --only <sid> --n-per <n> --out-dir regen_<ds>``
   — **BILLED** (one Gemini image call per view)
2. ``pixi run extract-spot-labels all --input regen_<ds>``
   — **BILLED** (purple + anatomy per new view; resumable, so only new files cost anything)
3. ``pixi run compute-quality --input regen_<ds>``  — free

The output dir is ``images/regen_<dataset>/``, which is a normal image folder: it carries
``purple/``, ``anatomy/`` and its own ``contours/contours.db``, so the new views are **complete**
and ``fold-synth --synth regen_<dataset> --into <folder>`` promotes them with nothing re-billed.

Until they are folded in and repackaged, the app serves them as **provisional** cards read straight
from that scratch DB. Nothing is written to the packaged dataset — an experiment quoting `_q0.4`
must not have its DB mutated under it by a review click.

``mock=True`` clones an already-extracted view (image + purple + anatomy) instead of calling Gemini
and runs only the free steps. It exercises every path here for **zero** billed calls, which is how
this module is tested.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
G_RE = re.compile(r"^(.+)_g(\d+)$")

# One view costs: 1 generate + 1 purple + 1 anatomy (+ up to 2 judge re-draws).
BILLED_CALLS_PER_VIEW = 3


def derive_label(sid: str) -> str:
    return sid.rsplit("_", 1)[0] if "_" in sid else sid


def next_g_index(label: str, dirs: list[Path]) -> int:
    """First free ``_g<k>`` for ``label`` across every directory given.

    Both the master image folder and the regen dir are consulted, so a fresh view can never be
    confused with one that already exists (``je_3_g0`` bad -> the new one is ``je_3_g1``).
    """
    top = -1
    for d in dirs:
        if not d or not d.is_dir():
            continue
        for p in d.glob(f"{label}_g*.png"):
            if m := G_RE.match(p.stem):
                top = max(top, int(m.group(2)))
        for p in d.glob(f"{label}_g*.jpg"):
            if m := G_RE.match(p.stem):
                top = max(top, int(m.group(2)))
    return top + 1


class GenerateJobs:
    """Runs regeneration jobs in daemon threads and reports their progress to the page."""

    def __init__(self, dataset: str, images_folder: str, on_done=None, release_db=None):
        self.dataset = dataset
        self.images_folder = images_folder
        self.regen_dir = REPO_ROOT / "images" / f"regen_{dataset}"
        self.master_dir = REPO_ROOT / "images" / images_folder
        self._on_done = on_done
        self._release_db = release_db
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._seq = 0

    # ------------------------------------------------------------------ status

    def jobs(self) -> list[dict]:
        with self._lock:
            return [dict(j, log=j["log"][-40:]) for j in self._jobs.values()]

    def active_for(self, label: str) -> bool:
        with self._lock:
            return any(j["label"] == label and j["state"] in ("queued", "running")
                       for j in self._jobs.values())

    def _log(self, job: dict, line: str) -> None:
        with self._lock:
            job["log"].append(line.rstrip())
            job["updated_at"] = datetime.now().isoformat(timespec="seconds")

    # ------------------------------------------------------------------ submit

    def submit(self, source_sid: str, n: int = 1, *, mock: bool = False) -> dict:
        """Queue a regeneration for one REAL photo. Returns the job record immediately."""
        label = derive_label(source_sid)
        if not mock and not self._api_key():
            raise RuntimeError("GEMINI_API_KEY is not set (env or .env) — generation is a billed "
                               "Gemini call and cannot run without it")
        with self._lock:
            self._seq += 1
            start = next_g_index(label, [self.master_dir, self.regen_dir])
            job = {
                "id": f"j{self._seq:03d}",
                "source": source_sid,
                "label": label,
                "n": int(n),
                "mock": bool(mock),
                "state": "queued",              # queued | running | ready | failed
                "step": "",
                "steps_done": 0,
                "steps_total": 3,
                "expect": [f"{label}_g{start + k}" for k in range(int(n))],
                "produced": [],
                "billed_calls": 0 if mock else int(n) * BILLED_CALLS_PER_VIEW,
                "error": "",
                "log": [],
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._jobs[job["id"]] = job
        threading.Thread(target=self._run, args=(job,), name="regen-" + job["id"],
                         daemon=True).start()
        return job

    @staticmethod
    def _api_key() -> str:
        import sys

        p = str(REPO_ROOT / "pipeline")
        if p not in sys.path:
            sys.path.insert(0, p)
        try:
            from generate_spot_labels._common import getenv, load_dotenv

            return getenv("GEMINI_API_KEY", "", load_dotenv())
        except Exception:
            return os.environ.get("GEMINI_API_KEY", "")

    # --------------------------------------------------------------------- run

    def _run(self, job: dict) -> None:
        t0 = time.time()
        with self._lock:
            job["state"] = "running"
        try:
            self.regen_dir.mkdir(parents=True, exist_ok=True)
            if job["mock"]:
                self._mock_views(job)
            else:
                self._step(job, "generating", [
                    "pixi", "run", "emb-gen-augment",
                    "--dataset", self.dataset,
                    "--only", job["source"],
                    "--n-per", str(job["n"]),
                    "--limit", str(job["n"]),
                    "--out-dir", self.regen_dir.name,
                ])
            self._step(job, "extracting spots + anatomy", [
                "pixi", "run", "extract-spot-labels", "all", "--input", self.regen_dir.name,
            ])
            self._step(job, "computing quality", [
                "pixi", "run", "compute-quality", "--input", self.regen_dir.name,
            ])

            self._ensure_regen_schema(job)
            produced = [sid for sid in job["expect"] if self._is_complete(sid)]
            with self._lock:
                job["produced"] = produced
                if not produced:
                    job["state"] = "failed"
                    job["error"] = ("no complete view was produced — see the log (the expected "
                                    f"ids were {', '.join(job['expect'])})")
                else:
                    job["state"] = "ready"
                    job["step"] = f"ready in {time.time() - t0:.0f}s"
            self._log(job, f"produced: {', '.join(produced) or 'nothing'}")
        except Exception as exc:
            with self._lock:
                job["state"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
            self._log(job, f"FAILED {job['error']}")
        finally:
            if self._on_done:
                try:
                    self._on_done(job)
                except Exception:
                    pass

    def _step(self, job: dict, label: str, cmd: list[str]) -> None:
        with self._lock:
            job["step"] = label
        self._log(job, "$ " + " ".join(cmd))
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1)
        for line in proc.stdout:                       # stream, so the page shows real progress
            self._log(job, line)
        code = proc.wait()
        with self._lock:
            job["steps_done"] += 1
        if code != 0:
            raise RuntimeError(f"`{' '.join(cmd[:4])}…` exited {code}")

    def _ensure_regen_schema(self, job: dict) -> None:
        """Give the scratch DB the one column a *packaged* dataset has and a pipeline DB does not.

        ``embeddings.get_spots`` selects ``images.is_synthetic``, which ``package-dataset`` adds. To
        compute descriptors for a provisional view through that same tested path — rather than
        re-implementing the 62-d embedding here — the scratch DB gets the column too. Every image in
        a regen dir is a generated view by construction, so ``true`` is right for all of them.

        This is *our* scratch DB, never the packaged dataset. DuckDB will not mix a read-only and a
        read-write connection to one file in one process, so the app's read-only handle is released
        first (``release_db``) and re-opened lazily afterwards.
        """
        import duckdb

        db = self.regen_dir / "contours" / "contours.db"
        if not db.is_file():
            return
        if self._release_db:
            self._release_db()
        con = duckdb.connect(str(db))
        try:
            cols = {r[0] for r in con.execute("DESCRIBE images").fetchall()}
            if "is_synthetic" not in cols:
                con.execute("ALTER TABLE images ADD COLUMN is_synthetic BOOLEAN")
                self._log(job, "added images.is_synthetic to the scratch DB (all views synthetic)")
            con.execute("UPDATE images SET is_synthetic = true WHERE is_synthetic IS NULL")
        finally:
            con.close()

    def _is_complete(self, sid: str) -> bool:
        """A view is usable only with its image, its purple render and its anatomy JSON."""
        img = any((self.regen_dir / f"{sid}{ext}").is_file() for ext in (".png", ".jpg", ".jpeg"))
        return img and (self.regen_dir / "purple" / f"{sid}.png").is_file() \
            and (self.regen_dir / "anatomy" / f"{sid}.json").is_file()

    # -------------------------------------------------------------------- mock

    def _mock_views(self, job: dict) -> None:
        """Clone an already-extracted view instead of paying Gemini — plumbing test only.

        Copies image + purple + anatomy exactly as ``fold-synth`` does, so stages 1 and 1b are
        already satisfied and only the free steps run.
        """
        with self._lock:
            job["step"] = "cloning an existing view (mock: no billed calls)"
        src = self._find_extracted(job["label"])
        if src is None:
            raise RuntimeError(f"mock needs an already-extracted {job['label']}_g* view to clone; "
                               f"none found under images/synth_*/")
        src_dir, src_sid = src
        self._log(job, f"cloning {src_dir.name}/{src_sid} -> {len(job['expect'])} view(s)")
        for sid in job["expect"]:
            for sub, name in [(None, f"{src_sid}.png"), ("purple", f"{src_sid}.png"),
                              ("anatomy", f"{src_sid}.json"), ("anatomy", f"{src_sid}_mask.png")]:
                s = (src_dir / sub / name) if sub else (src_dir / name)
                if not s.is_file():
                    continue
                d_dir = (self.regen_dir / sub) if sub else self.regen_dir
                d_dir.mkdir(parents=True, exist_ok=True)
                d = d_dir / name.replace(src_sid, sid)
                shutil.copy2(s, d)
                self._log(job, f"  {d.relative_to(REPO_ROOT)}")
        with self._lock:
            job["steps_done"] += 1

    def _find_extracted(self, label: str) -> tuple[Path, str] | None:
        for d in sorted((REPO_ROOT / "images").glob("synth_*"), reverse=True):
            for p in sorted(d.glob(f"{label}_g*.png")):
                if (d / "purple" / p.name).is_file() and (d / "anatomy" / f"{p.stem}.json").is_file():
                    return d, p.stem
        return None
