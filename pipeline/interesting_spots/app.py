"""State + image logic for the interesting-spot selector.

One :class:`SpotSelectorApp` owns everything the HTTP layer needs:

* the ordered list of images (source files that have a row in ``contours.db``),
* a read-only DuckDB connection to the per-spot masks,
* lazily-built **label maps** (a full-frame int array: pixel -> spot_id+1, 0 = background)
  used to turn a click at (x, y) into the spot that was hit,
* lazily-built **green overlay PNGs** (one per spot: transparent except a green wash on the
  spot's pixels) the browser stacks over the photo,
* the JSON label store (``salamander_id -> [spot_id, ...]``), persisted after every toggle.

Everything that touches the shared DuckDB connection or the label dict goes through
``self._lock`` so the threaded HTTP server is safe with a single connection.
"""
from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from generate_spot_labels._common import (
    contours_db_for,
    list_images,
    purple_dir_for,
)

# Green painted on a selected spot. cv2 encodes 4-channel arrays as B, G, R, A, so this is
# (B=0, G=210, R=0) at alpha 240 — nearly opaque, so a selected magenta spot on the purple
# image clearly *turns green* rather than tinting.
OVERLAY_BGRA = (0, 210, 0, 240)

# How many images' worth of label maps / overlay sets to keep in memory at once.
_CACHE_CAP = 8


class SpotSelectorApp:
    def __init__(self, input_dir: Path, labels_path: Path):
        import duckdb

        self.input_dir = Path(input_dir)
        self.db_path = contours_db_for(self.input_dir)
        self.labels_path = Path(labels_path)
        if not self.db_path.is_file():
            raise FileNotFoundError(
                f"no contours.db under {self.input_dir} — run the spot-label pipeline first "
                f"(expected {self.db_path})"
            )

        self._lock = threading.RLock()
        try:
            self._con = duckdb.connect(str(self.db_path), read_only=True)
        except duckdb.IOException as exc:  # another process holds the DB open read-write
            raise RuntimeError(
                f"could not open {self.db_path} read-only — is another script "
                f"(extract-spot-labels / build-dataset) using it?\n  {exc}"
            ) from exc

        db_ids = {r[0] for r in self._con.execute(
            "SELECT salamander_id FROM images").fetchall()}

        # We DISPLAY the purple image (its flat-magenta spots are exactly the clickable masks,
        # so the base image and the spots line up by construction). Order = filesystem order of
        # the source photos; keep only sids that have both spots in the DB and a purple image.
        self._purple_dir = purple_dir_for(self.input_dir)
        self.images: list[dict] = []
        self._path_by_sid: dict[str, Path] = {}
        for f in list_images(self.input_dir):
            purple = self._purple_dir / f"{f.stem}.png"
            if f.stem in db_ids and purple.is_file():
                self.images.append({"sid": f.stem, "filename": f.name})
                self._path_by_sid[f.stem] = purple
        self._index_by_sid = {img["sid"]: i for i, img in enumerate(self.images)}

        self.labels: dict[str, list[int]] = self._load_labels()

        # Bounded LRU caches (see _CACHE_CAP).
        self._labelmap_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._overlay_cache: "OrderedDict[tuple[str, int], bytes]" = OrderedDict()

    # ---- label store (JSON) -------------------------------------------------

    def _load_labels(self) -> dict[str, list[int]]:
        if not self.labels_path.is_file():
            return {}
        try:
            data = json.loads(self.labels_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        raw = data.get("labels", {}) if isinstance(data, dict) else {}
        out: dict[str, list[int]] = {}
        for sid, ids in raw.items():
            clean = sorted({int(i) for i in ids})
            if clean:
                out[sid] = clean
        return out

    def _save_labels(self) -> None:
        """Atomic write: temp file then os.replace, so a crash never truncates the JSON."""
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "images_folder": self.input_dir.name,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "labeled_images": len(self.labels),
            "labels": {sid: self.labels[sid] for sid in sorted(self.labels)},
        }
        tmp = self.labels_path.with_suffix(self.labels_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.labels_path)

    # ---- navigation helpers -------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.images)

    @property
    def labeled_count(self) -> int:
        return len(self.labels)

    def start_index(self) -> int:
        """First image with no (non-empty) label yet — where the app opens."""
        for i, img in enumerate(self.images):
            if not self.labels.get(img["sid"]):
                return i
        return 0

    def next_unlabeled(self, after: int) -> int | None:
        """Index of the next unlabeled image strictly after ``after`` (wraps once)."""
        n = self.total
        for step in range(1, n + 1):
            i = (after + step) % n
            if not self.labels.get(self.images[i]["sid"]):
                return i
        return None

    def index_meta(self, i: int) -> dict:
        img = self.images[i]
        sid = img["sid"]
        w, h = self._dims(sid)
        return {
            "index": i,
            "sid": sid,
            "filename": img["filename"],
            "width": w,
            "height": h,
            "selected": list(self.labels.get(sid, [])),
            "total": self.total,
            "labeledCount": self.labeled_count,
        }

    def index_payload(self) -> dict:
        return {
            "folder": self.input_dir.name,
            "labelsPath": str(self.labels_path),
            "total": self.total,
            "startIndex": self.start_index(),
            "images": [
                {"sid": img["sid"], "labeled": bool(self.labels.get(img["sid"]))}
                for img in self.images
            ],
        }

    # ---- pixels: click resolution + overlays --------------------------------

    def photo_path(self, sid: str) -> Path | None:
        """The image shown for this sid — the purple (segmentation) image."""
        return self._path_by_sid.get(sid)

    def _dims(self, sid: str) -> tuple[int, int]:
        with self._lock:
            row = self._con.execute(
                "SELECT width, height FROM images WHERE salamander_id = ?", [sid]).fetchone()
        return (int(row[0]), int(row[1])) if row else (0, 0)

    def _label_map(self, sid: str) -> np.ndarray:
        """Full-frame int32 map: 0 = background, else spot_id + 1.

        Spots are painted largest-first so a smaller spot on top wins any overlap — which is
        what a user aiming *inside* a small spot expects.
        """
        with self._lock:
            cached = self._labelmap_cache.get(sid)
            if cached is not None:
                self._labelmap_cache.move_to_end(sid)
                return cached

            w, h = self._dims(sid)
            lmap = np.zeros((h, w), np.int32)
            rows = self._con.execute(
                "SELECT spot_id, mask_png FROM spots WHERE salamander_id = ? "
                "ORDER BY area_pixels DESC", [sid]).fetchall()
            for spot_id, blob in rows:
                m = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
                if m is None or m.shape != (h, w):
                    continue
                lmap[m > 127] = int(spot_id) + 1

            self._labelmap_cache[sid] = lmap
            self._labelmap_cache.move_to_end(sid)
            while len(self._labelmap_cache) > _CACHE_CAP:
                self._labelmap_cache.popitem(last=False)
            return lmap

    @staticmethod
    def _snap_radius(w: int, h: int) -> int:
        """How far a click may miss and still snap to a spot (~3% of the short side)."""
        return max(25, round(0.03 * min(w, h)))

    def spot_at(self, sid: str, x: int, y: int, snap: bool = True) -> int | None:
        """The spot_id under pixel (x, y).

        With ``snap`` (the default), a click that lands on background falls back to the
        *nearest* spot within :meth:`_snap_radius` — the masks are often a few px off the real
        spot, so requiring a pixel-perfect hit makes small spots almost unselectable. Returns
        None only when there is no spot under or near the click.
        """
        lmap = self._label_map(sid)
        h, w = lmap.shape
        if not (0 <= x < w and 0 <= y < h):
            return None
        v = int(lmap[y, x])
        if v > 0:
            return v - 1
        if not snap:
            return None

        r = self._snap_radius(w, h)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        sub = lmap[y0:y1, x0:x1]
        nz = np.argwhere(sub > 0)
        if nz.size == 0:
            return None
        cy, cx = y - y0, x - x0
        d2 = (nz[:, 0] - cy) ** 2 + (nz[:, 1] - cx) ** 2
        j = int(np.argmin(d2))
        if d2[j] > r * r:
            return None
        return int(sub[nz[j, 0], nz[j, 1]]) - 1

    def overlay_png(self, sid: str, spot_id: int) -> bytes | None:
        """A full-frame RGBA PNG: transparent except a solid green fill over this one spot."""
        key = (sid, spot_id)
        with self._lock:
            cached = self._overlay_cache.get(key)
            if cached is not None:
                self._overlay_cache.move_to_end(key)
                return cached

            row = self._con.execute(
                "SELECT mask_png FROM spots WHERE salamander_id = ? AND spot_id = ?",
                [sid, spot_id]).fetchone()
            if row is None:
                return None
            m = cv2.imdecode(np.frombuffer(row[0], np.uint8), cv2.IMREAD_GRAYSCALE)
            if m is None:
                return None
            h, w = m.shape
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[m > 127] = OVERLAY_BGRA
            ok, buf = cv2.imencode(".png", rgba)
            if not ok:
                return None
            png = buf.tobytes()

            self._overlay_cache[key] = png
            self._overlay_cache.move_to_end(key)
            while len(self._overlay_cache) > _CACHE_CAP * 40:
                self._overlay_cache.popitem(last=False)
            return png

    def toggle(self, sid: str, x: int, y: int) -> dict:
        """Toggle the spot under (x, y). Returns the click outcome + the new selection.

        Persists the JSON on every change. An image with an empty selection is dropped from
        the store, so it counts as unlabeled again.
        """
        spot_id = self.spot_at(sid, x, y)
        if spot_id is None:
            with self._lock:
                sel = list(self.labels.get(sid, []))
                labeled_count = self.labeled_count
            return {"spotId": None, "selected": False,
                    "selection": sel, "labeledCount": labeled_count}

        with self._lock:
            sel = set(self.labels.get(sid, []))
            now_selected = spot_id not in sel
            if now_selected:
                sel.add(spot_id)
            else:
                sel.discard(spot_id)
            if sel:
                self.labels[sid] = sorted(sel)
            else:
                self.labels.pop(sid, None)
            self._save_labels()
            selection = sorted(sel)
            labeled_count = self.labeled_count

        return {"spotId": spot_id, "selected": now_selected,
                "selection": selection, "labeledCount": labeled_count}

    def close(self) -> None:
        with self._lock:
            try:
                self._con.close()
            except Exception:
                pass
