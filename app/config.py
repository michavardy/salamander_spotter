"""Runtime configuration and data-directory resolution (spec §4.3, §2.4).

Every path and bind address is environment / config driven — nothing hard-codes
``localhost`` or a data location, so relocating the app stays a config change.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ENV_PREFIX = "SPOTTER_"


def _default_data_dir() -> Path:
    """OS-appropriate default when ``SPOTTER_DATA_DIR`` is unset (bare source run)."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "SalamanderSpotter"
        return Path.home() / "AppData" / "Roaming" / "SalamanderSpotter"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "SalamanderSpotter"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "salamander-spotter"
    return Path.home() / ".local" / "share" / "salamander-spotter"


def bootstrap_config_path() -> Path:
    """A tiny pointer file at a fixed, OS-standard location — independent of
    whatever ``data_dir`` is currently chosen, so it can record *which* data
    dir to use next time (spec §4.3's Settings-editable data directory).
    """
    return _default_data_dir() / "config.json"


def read_bootstrap_data_dir() -> Path | None:
    p = bootstrap_config_path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = raw.get("data_dir")
    return Path(value).expanduser() if value else None


def write_bootstrap_data_dir(path: Path) -> None:
    """Persist the chosen data dir for the *next* server start — takes effect
    on restart, not live (DuckDB allows one read-write connection per file, so
    a running server can't hot-swap its own data directory)."""
    p = bootstrap_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if p.is_file():
        try:
            existing = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
    existing["data_dir"] = str(path)
    p.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(ENV_PREFIX + name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one running instance."""

    data_dir: Path
    host: str = "127.0.0.1"
    port: int = 8756
    external_base_url: str | None = None
    """Absolute base URL the SPA should use; ``None`` => same-origin / relative."""
    actor_label: str | None = None
    expose_api_docs: bool = False
    csrf_cookie_name: str = "spotter_csrf"

    # --- derived data-dir paths -------------------------------------------------
    @property
    def app_db_path(self) -> Path:
        return self.data_dir / "app.duckdb"

    @property
    def contours_db_path(self) -> Path:
        return self.data_dir / "contours.db"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def raw_images_dir(self) -> Path:
        return self.images_dir / "raw"

    @property
    def purple_images_dir(self) -> Path:
        return self.images_dir / "purple"

    @property
    def thumb_images_dir(self) -> Path:
        return self.images_dir / "thumb"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def lock_file(self) -> Path:
        return self.data_dir / ".spotter.lock"

    def ensure_dirs(self) -> None:
        """Create the data-dir tree if missing (idempotent)."""
        for p in (
            self.data_dir,
            self.raw_images_dir,
            self.purple_images_dir,
            self.thumb_images_dir,
            self.models_dir,
            self.exports_dir,
            self.logs_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


def load_settings(*, data_dir: str | os.PathLike[str] | None = None, **overrides) -> Settings:
    """Build :class:`Settings` from explicit args, then ``SPOTTER_*`` env, then the
    Settings-page-editable bootstrap file, then defaults."""
    resolved_dir = (
        Path(data_dir)
        if data_dir is not None
        else Path(_env("DATA_DIR")) if _env("DATA_DIR")
        else read_bootstrap_data_dir() or _default_data_dir()
    )
    values = dict(
        data_dir=resolved_dir.expanduser().resolve(),
        host=_env("HOST", "127.0.0.1"),
        port=int(_env("PORT", "8756")),
        external_base_url=_env("EXTERNAL_BASE_URL") or None,
        actor_label=_env("ACTOR_LABEL") or None,
        expose_api_docs=_env_bool("EXPOSE_API_DOCS", False),
    )
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Settings(**values)
