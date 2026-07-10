"""Shared helpers for the generate_spot_labels pipeline.

Kept dependency-free (stdlib only) so both stage modules and the orchestrator can
import it without pulling in cv2 / duckdb / genai when they are not needed.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

# This file: pipeline/generate_spot_labels/_common.py  ->  repo root is parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGES_ROOT = REPO_ROOT / "images"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Subdirectories the pipeline writes *inside* an input dir. Never treated as input.
RESERVED_SUBDIRS = {"purple", "contours"}


def reconfigure_utf8() -> None:
    """Make stdout/stderr UTF-8 regardless of the console codepage."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def load_dotenv(root: Path = REPO_ROOT) -> dict[str, str]:
    """Parse ``root/.env`` into a dict. Does not override the real environment."""
    env: dict[str, str] = {}
    path = root / ".env"
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def getenv(key: str, default: str = "", dotenv: dict[str, str] | None = None) -> str:
    """Real environment wins over the .env file wins over ``default``."""
    dotenv = dotenv if dotenv is not None else load_dotenv()
    return os.environ.get(key, dotenv.get(key, default))


def resolve_input_dir(arg: str | os.PathLike[str]) -> Path:
    """Resolve ``--input`` to a directory of salamander images.

    Accepts a full/relative path, or a bare name resolved under ``images/``.
    """
    p = Path(arg)
    for cand in (p, IMAGES_ROOT / p, REPO_ROOT / p):
        if cand.is_dir():
            return cand.resolve()
    raise FileNotFoundError(
        f"input directory not found: {arg!r} "
        f"(looked at {p}, {IMAGES_ROOT / p}, {REPO_ROOT / p})"
    )


def list_images(input_dir: Path) -> list[Path]:
    """Sorted image files directly under ``input_dir`` (non-recursive).

    Skips the pipeline's own ``purple/`` and ``contours/`` output subdirs.
    """
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and p.parent.name not in RESERVED_SUBDIRS
    )


def purple_dir_for(input_dir: Path) -> Path:
    return input_dir / "purple"


def contours_db_for(input_dir: Path) -> Path:
    return input_dir / "contours" / "contours.db"


def resolve_rewrite_csv(arg: str | os.PathLike[str], input_dir: Path | None = None) -> Path:
    """Resolve a ``--rewrite`` CSV path (as-is, under images/, repo root, or input dir)."""
    p = Path(arg)
    cands = [p, IMAGES_ROOT / p, REPO_ROOT / p]
    if input_dir is not None:
        cands.append(Path(input_dir) / p.name)
    for cand in cands:
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"rewrite CSV not found: {arg!r} (looked at {', '.join(map(str, cands))})")


def read_rewrite_names(csv_path: Path) -> list[str]:
    """Read a single-column CSV of image names (e.g. ``aa_1.jpg``).

    Blank lines and ``#`` comments are skipped; only the first column is used.
    Returns the raw names in file order (duplicates removed, order preserved).
    """
    names: list[str] = []
    seen: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            val = row[0].strip()
            if not val or val.startswith("#"):
                continue
            if val not in seen:
                seen.add(val)
                names.append(val)
    return names
