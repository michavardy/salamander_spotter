"""State for the spot-correspondence labeller: which pairs to show, and what was linked.

Pixel work (click -> spot_id, per-spot masks) is delegated to the interesting-spot selector's
:class:`SpotSelectorApp`, which already solves it — including snapping a near-miss click to the
nearest spot, which matters a lot when spots are small. We use it purely as a pixel service and
never call its ``toggle``, so it never writes to its own label store.

Two things get recorded per pair, and BOTH are needed:

``links``   ``[a_spot, b_spot]`` — confirmed same physical spot. These measure the DESCRIPTOR:
            given a true correspondence, does the embedding actually rank it first?
``misses``  a spot in A whose physical spot is visible in photo B but was never extracted there.
            These measure EXTRACTION RECALL, which no amount of descriptor work can fix.

Without the misses you cannot tell "the embedding is bad" from "the spot was never found", which
is the exact ambiguity this whole exercise exists to remove.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from interesting_spots.app import SpotSelectorApp

# One colour per link, cycled. Chosen to stay distinguishable over the purple segmentation
# image (which is magenta-on-dark), so no magenta and no near-black. cv2 writes B, G, R, A.
LINK_COLORS_BGRA = [
    (0, 210, 0, 235), (255, 200, 0, 235), (0, 165, 255, 235), (255, 255, 0, 235),
    (0, 255, 255, 235), (128, 255, 0, 235), (255, 128, 128, 235), (0, 100, 255, 235),
    (200, 255, 200, 235), (255, 0, 128, 235),
]
PENDING_BGRA = (255, 255, 255, 200)      # first click of a link, waiting for its partner
MISS_BGRA = (60, 60, 220, 220)           # red: visible in B but never extracted there


def is_synth(sid: str) -> bool:
    return sid.split("_")[-1].startswith("g")


def label_of(sid: str) -> str:
    return "_".join(sid.split("_")[:2])


class CorrespondenceApp:
    def __init__(self, input_dir: Path, store_path: Path, *, n_individuals: int = 25,
                 seed: int = 0, only: list[str] | None = None):
        self.input_dir = Path(input_dir)
        self.store_path = Path(store_path)
        self._lock = threading.RLock()

        # Pixel service (masks, click resolution, the purple base image).
        self.pixels = SpotSelectorApp(self.input_dir, self.store_path)
        self._con = self.pixels._con

        # One pair per individual: its first two REAL photos. One pair each keeps the sample
        # broad (many animals) rather than deep (many photos of few animals) -- the statistics
        # we want are per-individual, so breadth is what reduces variance.
        by_label: dict[str, list[str]] = {}
        for img in self.pixels.images:
            sid = img["sid"]
            if not is_synth(sid):
                by_label.setdefault(label_of(sid), []).append(sid)

        labels = sorted(lbl for lbl, sids in by_label.items() if len(sids) >= 2)
        if only:
            wanted = set(only)
            labels = [lbl for lbl in labels if lbl in wanted]
        else:
            # Deterministic shuffle, then take n: an unbiased sample beats "the first 25
            # alphabetically", which would over-sample one naming prefix.
            rng = np.random.default_rng(seed)
            labels = [labels[i] for i in rng.permutation(len(labels))][:n_individuals]

        self.pairs: list[dict] = []
        for lbl in labels:
            sids = sorted(by_label[lbl])
            self.pairs.append({"key": f"{sids[0]}__{sids[1]}", "label": lbl,
                               "a": sids[0], "b": sids[1]})
        self.pairs.sort(key=lambda p: p["label"])

        self.store: dict[str, dict] = self._load()

    # ---- persistence --------------------------------------------------------

    def _load(self) -> dict[str, dict]:
        if not self.store_path.is_file():
            return {}
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        out = {}
        for key, rec in (data.get("pairs") or {}).items():
            out[key] = {
                "links": [[int(a), int(b)] for a, b in rec.get("links", [])],
                "misses": [int(a) for a in rec.get("misses", [])],
                "done": bool(rec.get("done", False)),
            }
        return out

    def _save(self) -> None:
        """Atomic write — a crash mid-save must never truncate hand-labelled work."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        n_links = sum(len(r["links"]) for r in self.store.values())
        payload = {
            "images_folder": self.input_dir.name,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "n_pairs_touched": len(self.store),
            "n_pairs_done": sum(1 for r in self.store.values() if r["done"]),
            "n_links": n_links,
            "pairs": {k: self.store[k] for k in sorted(self.store)},
        }
        tmp = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.store_path)

    def _rec(self, key: str) -> dict:
        return self.store.setdefault(key, {"links": [], "misses": [], "done": False})

    # ---- navigation ---------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.pairs)

    def start_index(self) -> int:
        for i, p in enumerate(self.pairs):
            if not self.store.get(p["key"], {}).get("done"):
                return i
        return 0

    def next_undone(self, after: int) -> int | None:
        for step in range(1, self.total + 1):
            i = (after + step) % self.total
            if not self.store.get(self.pairs[i]["key"], {}).get("done"):
                return i
        return None

    def pair_meta(self, i: int) -> dict:
        p = self.pairs[i]
        rec = self.store.get(p["key"], {"links": [], "misses": [], "done": False})
        wa, ha = self.pixels._dims(p["a"])
        wb, hb = self.pixels._dims(p["b"])
        return {
            "index": i, "total": self.total, "key": p["key"], "label": p["label"],
            "a": {"sid": p["a"], "width": wa, "height": ha, "nSpots": self._n_spots(p["a"])},
            "b": {"sid": p["b"], "width": wb, "height": hb, "nSpots": self._n_spots(p["b"])},
            "links": rec["links"], "misses": rec["misses"], "done": rec["done"],
            "doneCount": sum(1 for r in self.store.values() if r["done"]),
            "linkCount": sum(len(r["links"]) for r in self.store.values()),
        }

    def _n_spots(self, sid: str) -> int:
        with self._lock:
            row = self._con.execute(
                "SELECT count(*) FROM spots WHERE salamander_id = ?", [sid]).fetchone()
        return int(row[0]) if row else 0

    def index_payload(self) -> dict:
        return {
            "folder": self.input_dir.name, "storePath": str(self.store_path),
            "total": self.total, "startIndex": self.start_index(),
            "pairs": [{"key": p["key"], "label": p["label"],
                       "done": bool(self.store.get(p["key"], {}).get("done"))}
                      for p in self.pairs],
        }

    # ---- editing ------------------------------------------------------------

    def add_link(self, key: str, a_spot: int, b_spot: int) -> dict:
        """Link A's spot to B's spot. A spot may appear in only ONE link — re-linking either
        side replaces the older link, so the store can never claim one physical spot is two."""
        with self._lock:
            rec = self._rec(key)
            rec["links"] = [l for l in rec["links"] if l[0] != a_spot and l[1] != b_spot]
            rec["links"].append([int(a_spot), int(b_spot)])
            rec["misses"] = [m for m in rec["misses"] if m != a_spot]   # it wasn't missing
            self._save()
            return dict(rec)

    def toggle_miss(self, key: str, a_spot: int) -> dict:
        """Mark/unmark A's spot as 'visible in B but never extracted there'."""
        with self._lock:
            rec = self._rec(key)
            if a_spot in rec["misses"]:
                rec["misses"] = [m for m in rec["misses"] if m != a_spot]
            else:
                rec["misses"].append(int(a_spot))
                rec["links"] = [l for l in rec["links"] if l[0] != a_spot]
            self._save()
            return dict(rec)

    def undo(self, key: str) -> dict:
        with self._lock:
            rec = self._rec(key)
            if rec["links"]:
                rec["links"].pop()
            elif rec["misses"]:
                rec["misses"].pop()
            self._save()
            return dict(rec)

    def set_done(self, key: str, done: bool) -> dict:
        with self._lock:
            rec = self._rec(key)
            rec["done"] = bool(done)
            self._save()
            return dict(rec)

    def spot_at(self, sid: str, x: int, y: int) -> int | None:
        return self.pixels.spot_at(sid, x, y)

    def photo_path(self, sid: str):
        return self.pixels.photo_path(sid)

    # ---- overlays -----------------------------------------------------------

    def overlay_png(self, sid: str, spot_id: int, kind: str, idx: int = 0) -> bytes | None:
        """Full-frame RGBA PNG: transparent except this spot, filled by role.

        ``kind`` is ``link`` (colour cycles by ``idx`` so a pair shares one colour across the two
        photos), ``pending`` (white, first click awaiting its partner) or ``miss`` (red).
        """
        if kind == "link":
            color = LINK_COLORS_BGRA[idx % len(LINK_COLORS_BGRA)]
        elif kind == "miss":
            color = MISS_BGRA
        else:
            color = PENDING_BGRA
        with self._lock:
            row = self._con.execute(
                "SELECT mask_png FROM spots WHERE salamander_id = ? AND spot_id = ?",
                [sid, spot_id]).fetchone()
        if row is None:
            return None
        m = cv2.imdecode(np.frombuffer(row[0], np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        rgba = np.zeros((*m.shape, 4), np.uint8)
        rgba[m > 127] = color
        ok, buf = cv2.imencode(".png", rgba)
        return buf.tobytes() if ok else None

    def close(self) -> None:
        self.pixels.close()
