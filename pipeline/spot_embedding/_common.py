"""Shared helpers for the spot_embedding pipeline.

Stdlib + numpy only (no cv2 / duckdb / torch here) so any module can import it cheaply.
Paths mirror the existing ``generate_spot_labels`` conventions: the repo root is
``parents[2]`` of this file, datasets live under ``datasets/<name>/`` and this pipeline's
outputs under ``artifacts/spot_embedding/``.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

# Gemini-augmented synthetic views are named "<label>_g<k>" (e.g. aa_1_g0). They join the
# TRAINING pool only — never gallery/query — so evaluation stays on real photos.
_SYNTH_RE = re.compile(r"_g\d+$")


def is_synthetic(salamander_id: str) -> bool:
    """True for a Gemini-augmented synthetic view (``<label>_g<k>``); real photos are False."""
    return bool(_SYNTH_RE.search(salamander_id))

# This file: pipeline/spot_embedding/_common.py  ->  repo root is parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPO_ROOT / "datasets"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "spot_embedding"

DEFAULT_DATASET = "all_sasa_norm_2026_10_07"


def reconfigure_utf8() -> None:
    """Make stdout/stderr UTF-8 regardless of the console codepage (matches the other CLIs)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def set_seed(seed: int) -> None:
    """Seed stdlib + numpy RNGs for reproducible splits / dummy embeddings."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def derive_label(salamander_id: str) -> str:
    """Identity label = image id with the trailing ``_<instance>`` stripped.

    ``aj_1_2`` -> ``aj_1``; ``ca_10_5`` -> ``ca_10``. Photos sharing a label are the same
    individual. Mirrors ``scripts/dataset/package_dataset.derive_label``.
    """
    return salamander_id.rsplit("_", 1)[0] if "_" in salamander_id else salamander_id


def derive_code(salamander_id: str) -> str:
    """Source/session code = the first token of the id (``ca_10_5`` -> ``ca``).

    Used for the session-grouped split: all individuals photographed under one source
    name stay on the same side of a fold, so shared background/lighting can't leak.
    """
    return salamander_id.split("_", 1)[0]


def resolve_dataset(name_or_path: str) -> Path:
    """Resolve a dataset to the directory that contains ``db/contours.db``.

    Accepts a bare name (under ``datasets/``), or a path to the dataset dir.
    """
    p = Path(name_or_path)
    for cand in (p, DATASETS_ROOT / name_or_path, REPO_ROOT / name_or_path):
        if (cand / "db" / "contours.db").is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"dataset not found: {name_or_path!r} — expected a dir with db/contours.db "
        f"(looked at {p}, {DATASETS_ROOT / name_or_path}, {REPO_ROOT / name_or_path})"
    )


def dataset_name(dataset_dir: Path) -> str:
    return Path(dataset_dir).name


def contours_db_path(dataset_dir: Path) -> Path:
    return Path(dataset_dir) / "db" / "contours.db"


def raw_dir(dataset_dir: Path) -> Path:
    return Path(dataset_dir) / "raw"


def prepared_dir(name: str) -> Path:
    return ARTIFACTS_ROOT / "prepared" / name


def runs_dir() -> Path:
    return ARTIFACTS_ROOT / "runs"


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")
