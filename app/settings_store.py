"""Typed application settings + secrets (spec §11, §15).

Tunable knobs (thresholds, ladders, coverage target, notification toggles) live
as a typed key/value store in the ``settings`` table. Secrets (LLM tokens, SMTP
password, map key) live in ``secrets.json`` (0600) on the data volume and are
never returned to the frontend — only "set / not set" + the last 4 chars.

The spec names ``config.toml`` for non-secret settings; a DB table is used
instead so a change is one transaction and shows up over SSE. The public surface
(``get`` / ``set`` / typed accessors) is what the rest of the app depends on.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import Database

# --------------------------------------------------------------------------- #
#  defaults (spec §7.2, §7.3, §7.4, §9.1, §10.11)                              #
# --------------------------------------------------------------------------- #
DEFAULTS: dict[str, Any] = {
    # quality ladder
    "auto_accept_quality": 0.65,
    "needs_a_look_floor": 0.40,
    "min_spots_auto_accept": 3,
    "quality_cutoff": 0.40,
    # matching / decisions
    "match_threshold": 0.55,
    "auto_approve_threshold": 0.95,
    "coverage_target": 0.90,
    "max_candidates": 6,
    "warn_on_override": True,
    # score formula — coefficient map over lettered metrics a..f (spec §7.4)
    "score_coefficients": {"a": 0.5, "e": 0.5},
    # training scheduler (spec §7.5, §10.11)
    "retrain_every_days": 60,
    "retrain_after_images": 200,
    "auto_promote": True,
    # remote logger (Claude Code / phone monitoring during a long training run) — off by
    # default; when on, the training bridge appends a log-tail snapshot to a relay file every
    # N minutes instead of a caller having to poll the growing job log itself
    "remote_logger_enabled": False,
    "remote_logger_interval_minutes": 15,
    # llm budget (spec §7.1, §9.3)
    "daily_llm_budget": 0,          # 0 = unlimited
    "llm_low_warn_at": 40,
    # notifications (spec §9.3)
    "notify_training_complete": True,
    "notify_model_auto_promoted": True,
    "notify_extraction_llm_low": True,
    "notify_hand_correction_spike": True,
    # pipeline logging (pipeline/utils/logger_utils.py) — empty log_dir means stdout only;
    # a training run's console output is still teed to its own job log regardless (job_runners.py)
    "log_dir": "",
    "log_level": "",
    # misc
    "expose_api_docs": False,
    "reason_chips": [
        "distinct spot layout",
        "no candidate above threshold",
        "good photo quality",
        "few corresponding spots",
        "partial match only",
        "pose/curl differs",
        "possible duplicate frame",
    ],
    "active_model": None,
}

SECRET_KEYS = ("llm_api_key", "judge_api_key", "smtp_password", "map_tile_key")


class SettingsStore:
    def __init__(self, db: Database):
        self.db = db

    def all(self) -> dict[str, Any]:
        stored = {
            r["key"]: json.loads(r["value_json"])
            for r in self.db.query_dicts("SELECT key, value_json FROM settings")
        }
        return {**DEFAULTS, **stored}

    def get(self, key: str, default: Any = None) -> Any:
        row = self.db.query_one("SELECT value_json FROM settings WHERE key = ?", [key])
        if row is not None:
            return json.loads(row["value_json"])
        return DEFAULTS.get(key, default)

    def set(self, key: str, value: Any, *, actor: str | None = None) -> None:
        before = self.get(key)
        payload = json.dumps(value)
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO settings (key, value_json) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value_json = excluded.value_json, updated_at = now()",
                [key, payload],
            )
            self.db.audit(
                action="setting_changed", entity="setting", entity_id=key,
                actor=actor, before=before, after=value,
            )

    def update(self, values: dict[str, Any], *, actor: str | None = None) -> None:
        for k, v in values.items():
            self.set(k, v, actor=actor)

    # -- typed helpers used across services ---------------------------------
    def ladder_config(self):
        from .services.quality_ladder import LadderConfig

        return LadderConfig(
            auto_accept_quality=float(self.get("auto_accept_quality")),
            needs_a_look_floor=float(self.get("needs_a_look_floor")),
            min_spots_auto_accept=int(self.get("min_spots_auto_accept")),
        )

    def decision_thresholds(self) -> "DecisionThresholds":
        return DecisionThresholds(
            match_threshold=float(self.get("match_threshold")),
            auto_approve=float(self.get("auto_approve_threshold")),
            coverage_target=float(self.get("coverage_target")),
            max_candidates=int(self.get("max_candidates")),
            warn_on_override=bool(self.get("warn_on_override")),
        )


@dataclass(frozen=True)
class DecisionThresholds:
    match_threshold: float = 0.55
    auto_approve: float = 0.95
    coverage_target: float = 0.90
    max_candidates: int = 6
    warn_on_override: bool = True


# --------------------------------------------------------------------------- #
#  secrets                                                                     #
# --------------------------------------------------------------------------- #
class SecretStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save(self, data: dict[str, str]) -> None:
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            self.path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600 (best effort on win)
        except OSError:
            pass

    def get(self, key: str) -> str | None:
        return os.environ.get(f"SPOTTER_{key.upper()}") or self._load().get(key)

    def set(self, key: str, value: str) -> None:
        data = self._load()
        data[key] = value
        self._save(data)

    def status(self) -> dict[str, dict]:
        data = self._load()
        out = {}
        for key in SECRET_KEYS:
            val = self.get(key)
            out[key] = {"set": bool(val), "hint": (val[-4:] if val else None)}
        return out
