"""State + data layer for the preprocessing review app.

One :class:`ReviewApp` owns everything the HTTP layer needs:

* the **individual index** — every ``salamander_id`` grouped by ``derive_label``, real photos
  first and synthetic ``_g*`` views last, ordered **lowest mean quality first** so the pass
  starts where the edits are (docs/preprocessing_ui.md, Flow 1),
* a read-only DuckDB connection to the packaged dataset (``spots``, ``body_axis``,
  ``image_quality``, ``spot_embeddings``),
* the **review store** (``artifacts/preprocessing/<dataset>/review.json``), written atomically
  after every edit and guarded by a monotonic ``rev``,
* **machine matches** for one individual, computed on demand from the 62-d spot embeddings
  (measured: ~20 ms for a four-photo family),
* **interest** (per-spot distinctiveness in [0, 1]) — a whole-dataset computation, so it is
  cached to ``interest.json`` and built in a background thread on first run.

The page owns the review *shape*: an edit posts the whole individual record back and this
module merges, counts and persists it. Nothing here writes to the DB — it is opened read-only.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
SCHEMA_VERSION = 1

# _q0.4 — the gate every headline number in results.md §6 is quoted at.
Q_GATE = 0.4

DEFAULT_REASONS = [
    {"slug": "blurry", "label": "Blurry", "quick_key": "1"},
    {"slug": "pixelated", "label": "Pixelated", "quick_key": "2"},
    {"slug": "small", "label": "Small", "quick_key": "3"},
    {"slug": "occluded", "label": "Occluded", "quick_key": "4"},
]

DEFAULT_UI = {
    "overlays": {"spots": True, "outline": True, "spine": True, "head_tail": False, "arcs": True},
    "photo_opacity": 1.0,
    "top_n_matches": 5,
    "matcher": "raw",
    "match_cutoff": 0.0,
    "spot_fill": "interest",
    "spot_fill_alpha": 0.45,
    "tooltip_fields": ["ssid", "interest", "spot", "head_dist"],
}

MATCHERS = ("raw", "strict_hand_pos", "logreg")

# Quality columns the footer grid and the family table show, in display order.
SCORE_FIELDS = ["overall_quality", "blur_quality", "spot_extraction_quality",
                "body_extraction_quality", "lighting_quality"]
RAW_FIELDS = [
    "center_line_length_px", "avg_width_px", "aspect_ratio", "body_area_px", "body_area_frac",
    "solidity", "border_frac", "min_dim_px", "line_inside_frac", "curl_deg",
    "blur_score", "mean_brightness", "underexposed_frac", "overexposed_frac", "glare_frac",
    "pattern_contrast", "n_spots", "spot_area_frac", "spots_outside_frac", "median_spot_area_px",
]


def derive_label(salamander_id: str) -> str:
    """Identity label = the id minus its trailing instance (``ac_3_1`` / ``ac_3_g0`` -> ``ac_3``).

    Mirrors ``spot_embedding._common.derive_label`` — kept local so this module imports nothing
    from the model pipeline just to group photos.
    """
    return salamander_id.rsplit("_", 1)[0] if "_" in salamander_id else salamander_id


def is_synthetic_id(salamander_id: str) -> bool:
    """``ac_3_g0`` is a generated view; ``ac_3_1`` is a photograph."""
    tail = salamander_id.rsplit("_", 1)[-1]
    return tail.startswith("g") and tail[1:].isdigit()


def ssid_of(salamander_id: str, spot_id: int) -> str:
    """Display/export form: ``ac_3_1`` + 14 -> ``AC-3-1-14`` (see docs/preprocessing_ui.md#ssid).

    Only the letters segment is upper-cased, so a synthetic view stays ``AC-3-g0-04``.
    """
    parts = salamander_id.split("_")
    parts[0] = parts[0].upper()
    return "-".join(parts) + f"-{int(spot_id):02d}"


def _num(value, nd: int = 4):
    """Round a float for transport; None-safe, NaN-safe."""
    if value is None:
        return None
    try:
        import math

        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, nd)
    except (TypeError, ValueError):
        return None


def _xy_list(seq, nd: int = 1):
    """A DuckDB ``DOUBLE[][]`` contour (or paired arrays) -> ``[[x, y], ...]`` rounded for JSON."""
    if seq is None:
        return []
    out = []
    for p in seq:
        try:
            out.append([round(float(p[0]), nd), round(float(p[1]), nd)])
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _decimate(points: list, cap: int = 250) -> list:
    """Keep at most ``cap`` points, evenly spaced, always including the last one.

    A body outline is ~1000 points and is drawn a few hundred px wide, so every 4th point is
    visually identical and the payload is a quarter of the size.
    """
    n = len(points)
    if n <= cap:
        return points
    step = -(-n // cap)                                # ceil division
    kept = points[::step]
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    return kept


class ReviewApp:
    """Everything the HTTP layer needs; thread-safe over one DuckDB connection."""

    def __init__(self, db_path: Path, photo_dirs: list[Path], store_path: Path,
                 dataset: str, images_folder: str, legacy_labels_path: Path | None = None):
        import duckdb

        self.db_path = Path(db_path)
        self.store_path = Path(store_path)
        self.dataset = dataset
        self.images_folder = images_folder
        self.legacy_labels_path = legacy_labels_path
        self.interest_path = self.store_path.parent / "interest.json"

        if not self.db_path.is_file():
            raise FileNotFoundError(f"no contours.db at {self.db_path}")

        self._lock = threading.RLock()
        try:
            self._con = duckdb.connect(str(self.db_path), read_only=True)
        except duckdb.IOException as exc:              # another process holds it read-write
            raise RuntimeError(
                f"could not open {self.db_path} read-only — is another script using it?\n  {exc}"
            ) from exc

        self._photos = self._index_photos(photo_dirs)
        self._images = self._load_images()             # sid -> {w, h, n_spots, synthetic, quality}
        self._families = self._group_families()        # label -> [sid, ...] real first
        self._q_sorted: dict[str, list[float]] = {}
        self.q_stats = self._quality_stats()           # dataset-wide mean/std/quartiles
        self.review = self._load_or_init_store()

        # Lazily-built, expensive things.
        self._emb: dict[str, "object"] = {}            # sid -> (spot_ids, matrix) once loaded
        self._emb_loaded = False
        self._interest: dict[str, dict[int, float]] = {}
        self._interest_state = "missing"               # missing | building | ready | failed
        self._interest_error = ""
        self._load_interest_cache()

        # Regenerated synthetic views, served straight from the scratch dir until an operator
        # folds them in — the packaged dataset is never written to (see generate.py).
        from .generate import GenerateJobs

        self.jobs = GenerateJobs(dataset, images_folder, on_done=self._on_job_done,
                                 release_db=self._release_regen_db)
        self._regen_con = None
        self._regen_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------ startup

    def _index_photos(self, photo_dirs: list[Path]) -> dict[str, Path]:
        """``sid -> file``. Earlier directories win, so a source photo beats a purple render."""
        exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
        found: dict[str, Path] = {}
        for d in photo_dirs:
            if not d or not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in exts:
                    found.setdefault(f.stem, f)
        return found

    def _load_images(self) -> dict[str, dict]:
        cols = ", ".join(f"q.{c}" for c in SCORE_FIELDS + RAW_FIELDS + ["axis_source", "judged_ok"])
        rows = self._con.execute(f"""
            SELECT i.salamander_id, i.width, i.height, i.n_spots, i.is_synthetic,
                   b.source AS axis_line_source, b.judged_ok AS axis_judged_ok,
                   b.judge_feedback, b.length_px, {cols}
            FROM images i
            LEFT JOIN body_axis b USING (salamander_id)
            LEFT JOIN image_quality q USING (salamander_id)
            ORDER BY i.salamander_id
        """).fetchall()
        names = [d[0] for d in self._con.description]
        out: dict[str, dict] = {}
        for row in rows:
            r = dict(zip(names, row))
            sid = r["salamander_id"]
            synthetic = bool(r["is_synthetic"]) if r["is_synthetic"] is not None \
                else is_synthetic_id(sid)
            quality = {k: _num(r.get(k)) for k in SCORE_FIELDS + RAW_FIELDS}
            out[sid] = {
                "sid": sid,
                "width": int(r["width"] or 0),
                "height": int(r["height"] or 0),
                "n_spots": int(r["n_spots"] or 0),
                "is_synthetic": synthetic,
                "length_px": _num(r.get("length_px")),
                "axis": {
                    "source": r.get("axis_line_source"),
                    "judged_ok": None if r.get("axis_judged_ok") is None
                    else bool(r["axis_judged_ok"]),
                    "judge_feedback": r.get("judge_feedback"),
                },
                "quality": quality,
                "passes_q04": self._passes_gate(quality.get("overall_quality"), synthetic),
                "has_photo": sid in self._photos,
            }
        return out

    @staticmethod
    def _passes_gate(overall, synthetic: bool) -> bool:
        """The `_q0.4` badge — same rule as ``core/data.quality_keep_mask``.

        Synthetic views and images with no ``image_quality`` row are KEPT by the gate, which is
        why the page renders those two cases as inherited (`gen`) / unknown (`?`) rather than as
        an earned tick.
        """
        return True if (synthetic or overall is None) else float(overall) >= Q_GATE

    # Fields the footer and the family table normalize against the dataset.
    NORM_FIELDS = SCORE_FIELDS + ["spots_outside_frac", "curl_deg", "glare_frac", "n_spots"]

    def _quality_stats(self) -> dict:
        """Dataset-wide mean / std / quartiles per quality field, over REAL photos only.

        A bare ``0.83`` is not a decision a reviewer can make; ``0.83, +0.21 above the dataset
        mean, 78th percentile`` is. Synthetic views are excluded — their numbers describe the
        generator, not a photograph, and would drag every mean.
        """
        import math

        vals: dict[str, list[float]] = {f: [] for f in self.NORM_FIELDS}
        for meta in self._images.values():
            if meta["is_synthetic"]:
                continue
            for f in self.NORM_FIELDS:
                v = meta["quality"].get(f)
                if v is not None:
                    vals[f].append(float(v))
        out: dict[str, dict] = {}
        for f, xs in vals.items():
            if not xs:
                continue
            xs.sort()
            n = len(xs)
            mean = sum(xs) / n
            std = math.sqrt(sum((x - mean) ** 2 for x in xs) / max(1, n - 1))
            self._q_sorted[f] = xs
            out[f] = {"mean": round(mean, 4), "std": round(std, 4), "n": n,
                      "quartiles": [round(xs[int(q * (n - 1))], 4) for q in (0, .25, .5, .75, 1)]}
        return out

    def _quality_norm(self, quality: dict) -> dict:
        """Per field: ``z`` (how many SDs off the dataset mean) and ``pct`` (percentile rank)."""
        from bisect import bisect_left

        out: dict[str, dict] = {}
        for f, stats in self.q_stats.items():
            v = quality.get(f)
            if v is None:
                continue
            xs = self._q_sorted.get(f) or []
            z = None if not stats["std"] else round((float(v) - stats["mean"]) / stats["std"], 2)
            pct = round(bisect_left(xs, float(v)) / max(1, len(xs) - 1), 3) if xs else None
            out[f] = {"z": z, "pct": pct, "d": round(float(v) - stats["mean"], 3)}
        return out

    def _group_families(self) -> dict[str, list[str]]:
        fams: dict[str, list[str]] = {}
        for sid, meta in self._images.items():
            if not meta["has_photo"]:
                continue
            fams.setdefault(derive_label(sid), []).append(sid)
        for label, sids in fams.items():
            # real photos first, synthetic last; each block alphabetical (Section C).
            sids.sort(key=lambda s: (self._images[s]["is_synthetic"], s))
        return fams

    def _family_quality(self, label: str) -> tuple[float, float]:
        """(mean, worst) ``overall_quality`` over the REAL photos — the review-order key."""
        vals = [self._images[s]["quality"].get("overall_quality")
                for s in self._families[label] if not self._images[s]["is_synthetic"]]
        vals = [v for v in vals if v is not None]
        if not vals:
            return (-1.0, -1.0)                        # unmeasured sorts FIRST (Flow 1)
        return (sum(vals) / len(vals), min(vals))

    def _build_order(self) -> list[str]:
        """Lowest mean quality first, ties by worst photo, then label. Frozen once written."""
        return sorted(self._families, key=lambda lab: (*self._family_quality(lab), lab))

    # ------------------------------------------------------------- review store

    def _load_or_init_store(self) -> dict:
        store: dict = {}
        if self.store_path.is_file():
            try:
                store = json.loads(self.store_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                backup = self.store_path.with_suffix(".corrupt.json")
                os.replace(self.store_path, backup)
                logger.warning(f"  ! unreadable store moved to {backup.name} ({exc}); starting fresh")
                store = {}

        store.setdefault("schema_version", SCHEMA_VERSION)
        store.setdefault("rev", 0)
        store["dataset"] = self.dataset
        store["images_folder"] = self.images_folder
        store.setdefault("reviewer", os.environ.get("REVIEWER", ""))
        store.setdefault("cursor", {"last_edited": None, "last_edited_at": None, "resume_at": None})
        store.setdefault("rejection_reasons", [dict(r, added_at=self._now()) for r in DEFAULT_REASONS])
        store.setdefault("reprocess_queue", [])
        store.setdefault("individuals", {})
        ui = dict(DEFAULT_UI)
        ui.update(store.get("ui") or {})
        ui["overlays"] = {**DEFAULT_UI["overlays"], **(ui.get("overlays") or {})}
        store["ui"] = ui

        # review_order is persisted so the resume point survives the dataset growing (Flow 1).
        order = [lab for lab in (store.get("review_order") or []) if lab in self._families]
        new = [lab for lab in self._build_order() if lab not in set(order)]
        if not order:
            store["order_policy"] = "overall_quality_asc"
            order = new
        else:
            order = order + new                        # appended, never interleaved
        store["review_order"] = order
        store.setdefault("order_policy", "overall_quality_asc")

        self._import_legacy(store)
        store["interesting_spots"] = self._derive_interesting(store)
        store["counts"] = self._counts(store)
        return store

    def _import_legacy(self, store: dict) -> None:
        """Seed from ``interesting_spots.json`` once — those ~8.7k clicks are load-bearing (#31)."""
        if store.get("legacy_import"):
            return
        path = self.legacy_labels_path
        if not path or not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        labels = (data.get("labels") if isinstance(data, dict) else None) or {}
        # Keep EVERY click, including images that are not in this packaged dataset (renamed or
        # dropped since they were labelled). Silently discarding hand labels is the one thing this
        # import must not do — they round-trip through export() untouched.
        clicks = {sid: sorted({int(i) for i in ids}) for sid, ids in labels.items() if ids}
        outside = sorted(sid for sid in clicks if sid not in self._images)
        store["legacy_interesting_spots"] = clicks
        store["legacy_import"] = {
            "source": str(path), "at": self._now(),
            "images": len(clicks), "clicks": sum(len(v) for v in clicks.values()),
            "images_outside_dataset": outside,
        }
        logger.info(f"  imported {sum(len(v) for v in clicks.values())} interesting-spot clicks over "
              f"{len(clicks)} images from {path.name}"
              + (f" ({len(outside)} of those images are not in this dataset)" if outside else ""))

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _counts(self, store: dict) -> dict:
        inds = store.get("individuals", {})
        imgs = [r for ind in inds.values() for r in ind.get("images", {}).values()]
        n_matches = [m for ind in inds.values() for m in ind.get("matches", [])]
        return {
            "individuals_done": sum(1 for i in inds.values() if i.get("done")),
            "individuals_total": len(self._families),
            "individuals_deleted": sum(1 for i in inds.values() if i.get("deleted")),
            "images_accepted": sum(1 for r in imgs if r.get("decision") == "accept"),
            "images_rejected": sum(1 for r in imgs if r.get("decision") == "reject"),
            "images_deleted": sum(1 for r in imgs if r.get("deleted")),
            "images_total": sum(len(v) for v in self._families.values()),
            "matches_accepted": sum(1 for m in n_matches if m.get("accepted", True)),
            "matches_rejected": sum(1 for m in n_matches if not m.get("accepted", True)),
            "matches_human": sum(1 for m in n_matches if m.get("proposed_by") == "human"),
            "matches_algorithm": sum(1 for m in n_matches if m.get("proposed_by") == "algorithm"),
            "misses": sum(len(ind.get("misses", [])) for ind in inds.values()),
            "reprocess_requested": sum(1 for e in store.get("reprocess_queue", [])
                                       if e.get("status") == "requested"),
            "interesting_spots": sum(len(v) for v in (store.get("interesting_spots") or {}).values()),
        }

    def _derive_interesting(self, store: dict) -> dict[str, list[int]]:
        """Every human match member, in the legacy store's shape, merged with the imported clicks."""
        out: dict[str, set[int]] = {sid: set(ids)
                                    for sid, ids in (store.get("legacy_interesting_spots") or {}).items()}
        for ind in store.get("individuals", {}).values():
            for m in ind.get("matches", []):
                # A machine pair a human drew for themselves counts too — they clicked both spots.
                if m.get("proposed_by") != "human" and not m.get("confirmed_by_human"):
                    continue
                for mem in m.get("members", []):
                    sid, spot = mem.get("image"), mem.get("spot")
                    if sid is None or spot is None:
                        continue
                    out.setdefault(sid, set()).add(int(spot))
        return {sid: sorted(v) for sid, v in sorted(out.items()) if v}

    def _save(self) -> dict:
        """Atomic write — a crash mid-save must never truncate hand-labelled work."""
        store = self.review
        store["rev"] = int(store.get("rev", 0)) + 1
        store["updated_at"] = self._now()
        store["interesting_spots"] = self._derive_interesting(store)
        store["counts"] = self._counts(store)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        tmp.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.store_path)
        return {"rev": store["rev"], "updated_at": store["updated_at"], "counts": store["counts"]}

    def apply_edit(self, rev: int, payload: dict) -> dict:
        """Merge one individual's record (+ optional globals) and persist.

        ``rev`` is the revision the page based its edit on. A mismatch means another tab or
        reviewer wrote in between, so the edit is REFUSED rather than silently winning — hours
        of hand labels are not worth a last-write-wins race.
        """
        with self._lock:
            current = int(self.review.get("rev", 0))
            if int(rev) != current:
                return {"conflict": True, "rev": current, "review": self.review}

            label = payload.get("individual")
            record = payload.get("record")
            if label and record is not None:
                if label not in self._families:
                    raise KeyError(f"unknown individual {label!r}")
                self.review["individuals"][label] = record
                self.review["cursor"] = {
                    "last_edited": label,
                    "last_edited_at": self._now(),
                    "resume_at": self._after(label),
                }
            for key in ("ui", "rejection_reasons", "reprocess_queue", "reviewer"):
                if key in payload and payload[key] is not None:
                    self.review[key] = payload[key]
            return self._save()

    def _after(self, label: str) -> str | None:
        order = self.review.get("review_order", [])
        try:
            i = order.index(label)
        except ValueError:
            return None
        return order[i + 1] if i + 1 < len(order) else None

    # ------------------------------------------------------------- interest map

    @staticmethod
    def _interest_weights() -> dict:
        """The blend ``strict_match.distinctiveness`` will use, as a plain dict for the cache stamp.

        Imported lazily and defaulting to ``{}`` on failure, for the same reason
        :meth:`_build_interest` catches everything: a missing model dependency must not stop the
        review app, and an unstamped cache is treated as stale rather than as a crash.
        """
        try:
            import sys                                              # noqa: PLC0415

            core = REPO_ROOT / "pipeline" / "spot_transformer" / "core"
            if str(core) not in sys.path:
                sys.path.insert(0, str(core))
            from strict_match import DEFAULT_WEIGHTS                # noqa: PLC0415
            return {k: float(v) for k, v in DEFAULT_WEIGHTS.items()}
        except Exception:
            return {}

    def _load_interest_cache(self) -> None:
        """Load ``interest.json``, but only if it was built with the weights in force TODAY.

        The ramp and the `Interest:` tooltip are what a reviewer judges a spot by, so a cache built
        under a superseded blend is worse than no cache — it shows the reviewer a number no part of
        the system computes any more, with nothing on screen saying so. The 2026-07-28 cache
        outlived the 2026-08-15 `rarity` change exactly this way: unstamped, so never invalidated.
        A stamp mismatch (or a stamp-less file) leaves the state ``missing``, which is the signal
        :meth:`ensure_interest` already acts on.
        """
        if not self.interest_path.is_file():
            return
        try:
            data = json.loads(self.interest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        want = self._interest_weights()
        got = data.get("weights")
        if want and got != want:
            logger.warning(f"  interest: cache built {data.get('built_at', '?')} under a different "
                  f"distinctiveness blend ({got or 'unstamped'}) — rebuilding")
            return
        scores = data.get("scores") or {}
        self._interest = {sid: {int(k): float(v) for k, v in per.items()}
                          for sid, per in scores.items()}
        self._interest_state = "ready"
        logger.info(f"  interest: {sum(len(v) for v in self._interest.values())} cached scores "
              f"({self.interest_path.name})")

    def interest_status(self) -> dict:
        return {"state": self._interest_state, "error": self._interest_error,
                "path": str(self.interest_path)}

    def ensure_interest(self, background: bool = True) -> None:
        """Kick off the whole-dataset distinctiveness computation if it has not been cached.

        It is a ~4-minute job over 40k spots (percentile ranks are population-wide, so it cannot
        be done per family without changing what the number means). Runs in a daemon thread; the
        page falls back to a flat fill and picks the scores up on the next individual it loads.
        """
        with self._lock:
            if self._interest_state in ("ready", "building"):
                return
            self._interest_state = "building"
        if background:
            threading.Thread(target=self._build_interest, name="interest", daemon=True).start()
        else:
            self._build_interest()

    def _build_interest(self) -> None:
        t0 = time.time()
        try:
            import sys

            import numpy as np

            core = REPO_ROOT / "pipeline" / "spot_transformer" / "core"
            if str(core) not in sys.path:
                sys.path.insert(0, str(core))
            import embeddings as E                                   # noqa: PLC0415
            from strict_match import distinctiveness                 # noqa: PLC0415

            logger.info("  interest: computing distinctiveness over the whole dataset "
                  "(one-off, ~4 min, cached afterwards)…")
            spots = E.get_spots(self.db_path)
            emb = self._con.execute(
                "SELECT salamander_id, spot_id, embedding FROM spot_embeddings").fetchall()
            emap = {(sid, int(sp)): vec for sid, sp, vec in emb}
            keys = list(zip(spots["salamander_id"], spots["spot_id"].astype(int)))
            keep = [i for i, k in enumerate(keys) if k in emap]
            sub = spots.iloc[keep].reset_index(drop=True)
            M = np.asarray([emap[keys[i]] for i in keep], dtype=float)
            M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)

            scores = distinctiveness(sub, M)
            per: dict[str, dict[int, float]] = {}
            for sid, spot, val in zip(sub["salamander_id"], sub["spot_id"].astype(int), scores):
                per.setdefault(str(sid), {})[int(spot)] = round(float(val), 4)

            self.interest_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.interest_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({
                "dataset": self.dataset, "built_at": self._now(),
                "source": "strict_match.distinctiveness (population = whole dataset)",
                # the blend these scores were produced by — read back by _load_interest_cache, so
                # changing DEFAULT_WEIGHTS invalidates the cache instead of silently outliving it
                "weights": self._interest_weights(),
                "n_spots": sum(len(v) for v in per.values()),
                "scores": per,
            }), encoding="utf-8")
            os.replace(tmp, self.interest_path)
            with self._lock:
                self._interest = per
                self._interest_state = "ready"
            logger.info(f"  interest: {sum(len(v) for v in per.values())} scores in "
                  f"{time.time() - t0:.0f}s -> {self.interest_path}")
        except Exception as exc:                        # a missing model dep must not kill the app
            with self._lock:
                self._interest_state = "failed"
                self._interest_error = f"{type(exc).__name__}: {exc}"
            logger.error(f"  interest: FAILED ({self._interest_error}) — spots render with a flat fill")

    # --------------------------------------------------------------- embeddings

    def _embeddings_for(self, sids: list[str]) -> dict[str, tuple[list[int], "object"]]:
        """``sid -> (spot_ids, (n, 62) L2-normalized matrix)`` for the given images."""
        import numpy as np

        out: dict[str, tuple[list[int], object]] = {}
        with self._lock:
            unknown = [s for s in sids if s not in self._emb and s not in self._images]
        for s in unknown:                              # provisional views: computed, not stored
            emb = self._provisional_embeddings(s)
            with self._lock:
                self._emb[s] = emb
        with self._lock:
            missing = [s for s in sids if s not in self._emb]
            if missing:
                marks = ", ".join("?" * len(missing))
                rows = self._con.execute(
                    f"SELECT salamander_id, spot_id, embedding FROM spot_embeddings "
                    f"WHERE salamander_id IN ({marks}) ORDER BY salamander_id, spot_id",
                    missing).fetchall()
                grouped: dict[str, list[tuple[int, list[float]]]] = {s: [] for s in missing}
                for sid, spot, vec in rows:
                    grouped.setdefault(sid, []).append((int(spot), vec))
                for sid, items in grouped.items():
                    if not items:
                        self._emb[sid] = ([], np.zeros((0, 0)))
                        continue
                    ids = [i for i, _ in items]
                    M = np.asarray([v for _, v in items], dtype=float)
                    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
                    self._emb[sid] = (ids, M)
            for s in sids:
                out[s] = self._emb.get(s, ([], np.zeros((0, 0))))
        return out

    def machine_matches(self, label: str, matcher: str = "raw", top_n: int = 5,
                        cutoff: float = 0.0) -> dict:
        """Proposed edges for one family: adjacent image pairs, best ``top_n`` per pair.

        ``raw`` is mutual-nearest-neighbour on the 62-d descriptors — per-spot, training-free and
        instant, and its arcs are its own. ``strict_hand_pos`` replaces that with the one-to-one
        Hungarian assignment from ``core/strict_match``, so a spot cannot claim three partners.
        ``logreg`` is an image-level ranker with no spot-level output, so its arcs ARE the raw
        correspondences it scored — the page says so rather than letting the reviewer assume.
        """
        import numpy as np

        sids = list(self._families.get(label, [])) + self.provisional_sids(label)
        embs = self._embeddings_for(sids)
        method = matcher if matcher in MATCHERS else "raw"
        edges: list[dict] = []
        note = ""

        pairs = list(zip(sids, sids[1:]))              # the chain the canvas draws
        for a, b in pairs:
            ids_a, Ma = embs.get(a, ([], None))
            ids_b, Mb = embs.get(b, ([], None))
            if not ids_a or not ids_b or Ma is None or Mb is None or Ma.size == 0 or Mb.size == 0:
                continue
            S = Ma @ Mb.T

            if method == "strict_hand_pos":
                cand = self._assign_one_to_one(S)
                if cand is None:
                    note = ("strict_hand_pos unavailable (scipy/strict_match import failed) — "
                            "showing raw correspondences")
                    method = "raw"
                    cand = self._mutual(S)
            else:
                cand = self._mutual(S)

            cand.sort(key=lambda t: -t[2])
            for i, j, s in cand[: max(0, int(top_n))]:
                if s < cutoff:
                    continue
                edges.append({
                    "a": {"image": a, "spot": int(ids_a[i]), "ssid": ssid_of(a, ids_a[i])},
                    "b": {"image": b, "spot": int(ids_b[j]), "ssid": ssid_of(b, ids_b[j])},
                    "score": round(float(s), 4),
                })
        if method == "logreg":
            note = ("logreg ranks whole images and has no spot-level output — these arcs are the "
                    "raw correspondences it scored")
        return {"method": method, "requested": matcher, "edges": edges, "note": note,
                "pairs": [[a, b] for a, b in pairs]}

    @staticmethod
    def _mutual(S) -> list[tuple[int, int, float]]:
        """Mutual nearest neighbours — i's best is j and j's best is i."""
        ja = S.argmax(1)
        jb = S.argmax(0)
        return [(i, int(ja[i]), float(S[i, ja[i]]))
                for i in range(S.shape[0]) if int(jb[ja[i]]) == i]

    @staticmethod
    def _assign_one_to_one(S) -> list[tuple[int, int, float]] | None:
        """Hungarian max-weight assignment via ``core/strict_match.assign_one_to_one``."""
        try:
            import sys

            core = REPO_ROOT / "pipeline" / "spot_transformer" / "core"
            if str(core) not in sys.path:
                sys.path.insert(0, str(core))
            from strict_match import assign_one_to_one                # noqa: PLC0415

            ri, ci = assign_one_to_one(S, 0.0)
            return [(int(i), int(j), float(S[i, j])) for i, j in zip(ri, ci)]
        except Exception:
            return None

    # ----------------------------------------------------------------- payloads

    def bootstrap_payload(self) -> dict:
        """Everything the page needs before it draws anything: order, summaries, the store."""
        order = self.review.get("review_order", [])
        inds = self.review.get("individuals", {})
        summary = []
        for label in order:
            mean, worst = self._family_quality(label)
            rec = inds.get(label, {})
            summary.append({
                "label": label,
                "n_images": len(self._families[label]),
                "n_real": sum(1 for s in self._families[label]
                              if not self._images[s]["is_synthetic"]),
                "mean_quality": None if mean < 0 else round(mean, 3),
                "worst_quality": None if worst < 0 else round(worst, 3),
                "done": bool(rec.get("done")),
                "deleted": bool(rec.get("deleted")),
                "touched": bool(rec),
            })
        return {
            "dataset": self.dataset,
            "images_folder": self.images_folder,
            "db": str(self.db_path),
            "store": str(self.store_path),
            "matchers": list(MATCHERS),
            "score_fields": SCORE_FIELDS,
            "raw_fields": RAW_FIELDS,
            "q_gate": Q_GATE,
            "q_stats": self.q_stats,
            "interest": self.interest_status(),
            "individuals": summary,
            "review": self.review,
        }

    def individual_payload(self, label: str, matcher: str | None = None,
                           top_n: int | None = None, cutoff: float | None = None) -> dict:
        """One family's geometry + machine matches — the only heavy fetch the page makes."""
        if label not in self._families:
            raise KeyError(f"unknown individual {label!r}")
        ui = self.review.get("ui", DEFAULT_UI)
        matcher = matcher or ui.get("matcher", "raw")
        top_n = ui.get("top_n_matches", 5) if top_n is None else top_n
        cutoff = ui.get("match_cutoff", 0.0) if cutoff is None else cutoff

        sids = self._families[label]
        images = []
        with self._lock:
            for sid in list(sids):
                meta = self._images[sid]
                spots = self._con.execute("""
                    SELECT spot_id, global_centroid_x, global_centroid_y, area_pixels,
                           local_contour, axis_t, axis_side, axis_offset, bin
                    FROM spots WHERE salamander_id = ? ORDER BY spot_id
                """, [sid]).fetchall()
                axis = self._con.execute("""
                    SELECT midline_x, midline_y, left_x, left_y, right_x, right_y,
                           head_x, head_y, tail_tip_x, tail_tip_y, length_px, source, judged_ok
                    FROM body_axis WHERE salamander_id = ?
                """, [sid]).fetchone()
                interest = self._interest.get(sid, {})
                images.append({
                    "sid": sid,
                    "label": label,
                    "width": meta["width"],
                    "height": meta["height"],
                    "is_synthetic": meta["is_synthetic"],
                    "n_spots": meta["n_spots"],
                    "quality": meta["quality"],
                    "quality_norm": self._quality_norm(meta["quality"]),
                    "passes_q04": meta["passes_q04"],
                    "quality_known": meta["quality"].get("overall_quality") is not None,
                    "axis_meta": meta["axis"],
                    "length_px": meta["length_px"],
                    "spots": [{
                        "id": int(s[0]),
                        "xy": [_num(s[1], 1), _num(s[2], 1)],
                        "area": _num(s[3], 1),
                        "contour": _xy_list(s[4]),
                        "axis_t": _num(s[5]),
                        "side": s[6],
                        "offset": _num(s[7]),
                        "bin": None if s[8] is None else int(s[8]),
                        "ssid": ssid_of(sid, s[0]),
                        "interest": interest.get(int(s[0])),
                    } for s in spots],
                    "axis": None if axis is None else {
                        "midline": _xy_list(list(zip(axis[0] or [], axis[1] or []))),
                        # Outlines run to ~1000 points each; decimated they are 80 % of the
                        # payload smaller and identical at display scale (Technology).
                        "left": _decimate(_xy_list(list(zip(axis[2] or [], axis[3] or [])))),
                        "right": _decimate(_xy_list(list(zip(axis[4] or [], axis[5] or [])))),
                        "head": [_num(axis[6], 1), _num(axis[7], 1)],
                        "tail": [_num(axis[8], 1), _num(axis[9], 1)],
                        "length_px": _num(axis[10], 1),
                        "source": axis[11],
                        "judged_ok": None if axis[12] is None else bool(axis[12]),
                    },
                })
        # Regenerated views land after the dataset's own images, flagged provisional.
        for sid in self.provisional_sids(label):
            prov = self._provisional_image(sid)
            if prov:
                images.append(prov)
        all_sids = [im["sid"] for im in images]
        return {
            "label": label,
            "images": images,
            "machine": self.machine_matches(label, matcher, top_n, cutoff),
            "interest": self.interest_status(),
            "jobs": [j for j in self.jobs.jobs() if j["label"] == label],
            "regen_dir": str(self.jobs.regen_dir),
            "legacy_clicks": {sid: (self.review.get("legacy_interesting_spots") or {}).get(sid, [])
                              for sid in all_sids},
        }

    def photo_path(self, sid: str) -> Path | None:
        p = self._photos.get(sid)
        if p is not None:
            return p
        for ext in (".png", ".jpg", ".jpeg"):          # a provisional view, not yet folded in
            cand = self.jobs.regen_dir / f"{sid}{ext}"
            if cand.is_file():
                return cand
        return None

    # ------------------------------------------------------- provisional views

    def _on_job_done(self, job: dict) -> None:
        """A finished job invalidates the provisional cache so the new view is served at once."""
        self._release_regen_db()
        with self._lock:
            self._regen_cache.clear()
            for sid in job.get("produced", []):        # recompute descriptors for the new view
                self._emb.pop(sid, None)

    def _release_regen_db(self) -> None:
        """Close the scratch DB handle so the job runner can open it read-write."""
        with self._lock:
            if self._regen_con is not None:
                try:
                    self._regen_con.close()
                except Exception:
                    pass
                self._regen_con = None

    def _regen_db(self):
        """Read-only connection to the regen dir's own contours.db (None until one exists)."""
        import duckdb

        db = self.jobs.regen_dir / "contours" / "contours.db"
        if not db.is_file():
            return None
        if self._regen_con is None:
            try:
                self._regen_con = duckdb.connect(str(db), read_only=True)
            except Exception:
                return None
        return self._regen_con

    def provisional_sids(self, label: str) -> list[str]:
        """Regenerated views of ``label`` that are extracted but not yet in the packaged dataset."""
        con = self._regen_db()
        if con is None:
            return []
        try:
            rows = con.execute(
                "SELECT salamander_id FROM images WHERE salamander_id LIKE ? ORDER BY 1",
                [label + "_g%"]).fetchall()
        except Exception:
            return []
        return [r[0] for r in rows if r[0] not in self._images and self.photo_path(r[0])]

    def _provisional_image(self, sid: str) -> dict | None:
        """One provisional view in the same payload shape as a dataset image, `provisional: true`.

        Its 62-d descriptors are computed here rather than read from ``spot_embeddings`` — the
        concat embedding is a deterministic function of the contour and the body-frame position,
        with no population statistics in it, so a locally computed vector is directly comparable
        to the dataset's and the new view gets matched like any other photo.
        """
        with self._lock:
            if sid in self._regen_cache:
                return self._regen_cache[sid]
        con = self._regen_db()
        if con is None:
            return None
        try:
            img = con.execute("SELECT width, height, n_spots FROM images WHERE salamander_id = ?",
                              [sid]).fetchone()
            if img is None:
                return None
            qcols = SCORE_FIELDS + RAW_FIELDS
            qrow = con.execute(
                f"SELECT {', '.join(qcols)} FROM image_quality WHERE salamander_id = ?",
                [sid]).fetchone()
            quality = {k: _num(v) for k, v in zip(qcols, qrow)} if qrow else {k: None for k in qcols}
            spots = con.execute("""
                SELECT spot_id, global_centroid_x, global_centroid_y, area_pixels,
                       local_contour, axis_t, axis_side, axis_offset, bin
                FROM spots WHERE salamander_id = ? ORDER BY spot_id
            """, [sid]).fetchall()
            axis = con.execute("""
                SELECT midline_x, midline_y, left_x, left_y, right_x, right_y,
                       head_x, head_y, tail_tip_x, tail_tip_y, length_px, source, judged_ok
                FROM body_axis WHERE salamander_id = ?
            """, [sid]).fetchone()
        except Exception as exc:
            logger.error(f"  provisional {sid}: {type(exc).__name__}: {exc}")
            return None

        payload = {
            "sid": sid, "label": derive_label(sid),
            "width": int(img[0] or 0), "height": int(img[1] or 0),
            "is_synthetic": True, "provisional": True,
            "n_spots": int(img[2] or 0),
            "quality": quality, "quality_norm": self._quality_norm(quality),
            "passes_q04": True,                        # synthetic: kept by the gate, not earned
            "quality_known": quality.get("overall_quality") is not None,
            "axis_meta": {"source": axis[11] if axis else None,
                          "judged_ok": None if not axis or axis[12] is None else bool(axis[12]),
                          "judge_feedback": None},
            "length_px": _num(axis[10], 1) if axis else None,
            "spots": [{
                "id": int(s[0]), "xy": [_num(s[1], 1), _num(s[2], 1)], "area": _num(s[3], 1),
                "contour": _xy_list(s[4]), "axis_t": _num(s[5]), "side": s[6],
                "offset": _num(s[7]), "bin": None if s[8] is None else int(s[8]),
                "ssid": ssid_of(sid, s[0]), "interest": None,
            } for s in spots],
            "axis": None if axis is None else {
                "midline": _xy_list(list(zip(axis[0] or [], axis[1] or []))),
                "left": _decimate(_xy_list(list(zip(axis[2] or [], axis[3] or [])))),
                "right": _decimate(_xy_list(list(zip(axis[4] or [], axis[5] or [])))),
                "head": [_num(axis[6], 1), _num(axis[7], 1)],
                "tail": [_num(axis[8], 1), _num(axis[9], 1)],
                "length_px": _num(axis[10], 1), "source": axis[11],
                "judged_ok": None if axis[12] is None else bool(axis[12]),
            },
        }
        with self._lock:
            self._regen_cache[sid] = payload
        return payload

    def _provisional_embeddings(self, sid: str):
        """``(spot_ids, (n, 62) normalized)`` computed locally for a provisional view."""
        import sys

        import numpy as np

        core = REPO_ROOT / "pipeline" / "spot_transformer" / "core"
        if str(core) not in sys.path:
            sys.path.insert(0, str(core))
        db = self.jobs.regen_dir / "contours" / "contours.db"
        try:
            import embeddings as E                                       # noqa: PLC0415

            frame = E.get_spots(db)
            frame = frame[frame["salamander_id"] == sid]
            if frame.empty:
                return ([], np.zeros((0, 0)))
            emb = E.get_spot_embeddings(frame)
            ids = [int(i) for i in frame["spot_id"]]
            M = np.asarray([emb[f"{sid}#{i}"] for i in ids], dtype=float)
            M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
            return (ids, M)
        except Exception as exc:
            logger.error(f"  provisional embeddings {sid}: {type(exc).__name__}: {exc}")
            return ([], np.zeros((0, 0)))

    # ------------------------------------------------------------------ exports

    def export(self) -> dict:
        """Write the two derived artifacts a downstream consumer reads.

        * the legacy ``interesting_spots.json`` (existing shape) so ``strict_hand`` /
          distinctiveness consumers keep reading the file they already read;
        * one single-column CSV per queued re-extraction action, ready for ``--only`` /
          ``--rewrite``.
        """
        written = []
        with self._lock:
            spots = self._derive_interesting(self.review)
            if self.legacy_labels_path:
                # UNION with whatever is on disk, never replace it. The legacy file is
                # hand-labelled and lives outside git (artifacts/ is ignored), so an export must
                # not be able to drop a click somebody made with the old tool in the meantime.
                spots = self._union_legacy(self.legacy_labels_path, spots)
                self.legacy_labels_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "images_folder": self.images_folder,
                    "updated_at": self._now(),
                    "labeled_images": len(spots),
                    "source": f"preprocessing review ({self.store_path.name})",
                    "labels": spots,
                }
                tmp = self.legacy_labels_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                os.replace(tmp, self.legacy_labels_path)
                written.append(str(self.legacy_labels_path))

            by_action: dict[str, list[str]] = {}
            for entry in self.review.get("reprocess_queue", []):
                if entry.get("status") != "requested":
                    continue
                by_action.setdefault(entry.get("action", "unknown"), []).append(entry.get("image", ""))
            out_dir = self.store_path.parent / "reprocess"
            for action, ids in by_action.items():
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"{action}.csv"
                uniq = sorted({i for i in ids if i})
                path.write_text("salamander_id\n" + "\n".join(uniq) + "\n", encoding="utf-8")
                written.append(str(path))
        return {"written": written,
                "interesting_images": len(spots),
                "interesting_labels": sum(len(v) for v in spots.values())}

    @staticmethod
    def _union_legacy(path: Path, spots: dict[str, list[int]]) -> dict[str, list[int]]:
        """``spots`` merged with the clicks already in ``path`` — additive, never destructive."""
        if not path.is_file():
            return spots
        try:
            existing = (json.loads(path.read_text(encoding="utf-8")).get("labels") or {})
        except (json.JSONDecodeError, OSError, AttributeError):
            return spots
        merged = {sid: set(int(i) for i in ids) for sid, ids in existing.items() if ids}
        for sid, ids in spots.items():
            merged.setdefault(sid, set()).update(int(i) for i in ids)
        return {sid: sorted(v) for sid, v in sorted(merged.items()) if v}

    # --------------------------------------------------------------------- misc

    @property
    def n_individuals(self) -> int:
        return len(self._families)

    @property
    def n_images(self) -> int:
        return sum(len(v) for v in self._families.values())

    def resume_label(self) -> str:
        """Where to open: the entry AFTER ``cursor.last_edited``, not the first unreviewed one."""
        order = self.review.get("review_order", [])
        if not order:
            return ""
        last = (self.review.get("cursor") or {}).get("last_edited")
        if last in order:
            i = order.index(last)
            if i + 1 < len(order):
                return order[i + 1]
        for label in order:
            if not self.review["individuals"].get(label, {}).get("done"):
                return label
        return order[-1]

    def close(self) -> None:
        with self._lock:
            try:
                self._con.close()
            except Exception:
                pass
