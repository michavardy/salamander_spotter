"""preprocessing — the dataset-refinement and spot-match curation app.

One **individual at a time** (all of its photos side by side): accept/reject each image, mark
duplicates, disqualify bad machine-proposed matches, add the ones the machine missed, and queue
re-extractions. Everything lands in a single file that every downstream consumer filters against::

    artifacts/preprocessing/<dataset>/review.json

The whole UI is one HTML page (:mod:`page`); all state and every interaction live in the browser,
including click -> ``spot_id`` (the spot outlines are SVG paths, so the browser does the hit test).
Python only reads the packaged dataset DB and writes the store — see :mod:`app` for the state layer
and :mod:`server` for the stdlib HTTP layer.

**Replaces** ``interesting_spots``: its ~8.7k clicks are imported on first run and its legacy JSON
is kept up to date by ``preprocess-export``, so nothing downstream has to change.

Spec: docs/preprocessing_ui.md. CLI: ``scripts/tools/preprocessing_review.py``
(``pixi run preprocess-review``).
"""
from .app import ReviewApp, derive_label, is_synthetic_id, ssid_of
from .server import run

__all__ = ["ReviewApp", "run", "derive_label", "is_synthetic_id", "ssid_of"]
