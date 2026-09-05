from __future__ import annotations

from app.settings_store import DEFAULTS, SecretStore, SettingsStore


def test_defaults_and_override(db):
    s = SettingsStore(db)
    assert s.get("auto_approve_threshold") == 0.95
    s.set("auto_approve_threshold", 0.9)
    assert s.get("auto_approve_threshold") == 0.9
    assert s.all()["auto_approve_threshold"] == 0.9
    # untouched keys still come from defaults
    assert s.all()["coverage_target"] == DEFAULTS["coverage_target"]


def test_set_writes_audit(db):
    s = SettingsStore(db)
    s.set("coverage_target", 0.8, actor="dana")
    row = db.query_one("SELECT * FROM audit WHERE action = 'setting_changed'")
    assert row["entity_id"] == "coverage_target"
    assert row["actor"] == "dana"


def test_ladder_and_thresholds_typed(db):
    s = SettingsStore(db)
    s.update({"min_spots_auto_accept": 5, "match_threshold": 0.6})
    assert s.ladder_config().min_spots_auto_accept == 5
    assert s.decision_thresholds().match_threshold == 0.6


def test_secret_store_roundtrip_and_status(tmp_path):
    store = SecretStore(tmp_path / "secrets.json")
    assert store.status()["llm_api_key"]["set"] is False
    store.set("llm_api_key", "sk-abcd1234")
    assert store.get("llm_api_key") == "sk-abcd1234"
    status = store.status()
    assert status["llm_api_key"]["set"] is True
    assert status["llm_api_key"]["hint"] == "1234"


def test_secret_store_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SPOTTER_MAP_TILE_KEY", "from-env")
    store = SecretStore(tmp_path / "secrets.json")
    assert store.get("map_tile_key") == "from-env"
