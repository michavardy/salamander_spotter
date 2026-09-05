"""Spec §7.4 / §7.5 — model registry, training runs, promotion / rollback."""

from __future__ import annotations

import pytest

from app.pipeline_bridge.training import FakeTrainingBridge
from app.services import ingest, models as model_svc, training
from app.settings_store import SettingsStore


@pytest.fixture
def store(db):
    return SettingsStore(db)


def test_register_and_best_per_kind(db, store):
    coeffs = store.get("score_coefficients")
    model_svc.register_model(db, name="st_v1", kind="agg", metrics={"r1": 0.5, "novelty_auroc": 0.5}, coefficients=coeffs)
    model_svc.register_model(db, name="st_v2", kind="agg", metrics={"r1": 0.7, "novelty_auroc": 0.7}, coefficients=coeffs)
    model_svc.register_model(db, name="cnn_v1", kind="cnn", metrics={"r1": 0.6, "novelty_auroc": 0.4}, coefficients=coeffs)
    model_svc.recompute_best(db)
    best = {r["name"] for r in model_svc.list_models(db) if r["status"] == "best"}
    assert best == {"st_v2", "cnn_v1"}


def test_promote_and_rollback(db, store):
    coeffs = store.get("score_coefficients")
    model_svc.register_model(db, name="m1", kind="agg", metrics={"r1": 0.5}, coefficients=coeffs)
    model_svc.register_model(db, name="m2", kind="agg", metrics={"r1": 0.9}, coefficients=coeffs)
    model_svc.promote(db, "m1")
    assert model_svc.active_model(db)["name"] == "m1"
    model_svc.promote(db, "m2")
    assert model_svc.active_model(db)["name"] == "m2"
    assert store.get("active_model") == "m2"
    model_svc.rollback(db)
    assert model_svc.active_model(db)["name"] == "m1"


def test_training_run_evaluates_and_auto_promotes(db, settings, store):
    store.set("auto_promote", True)
    bridge = FakeTrainingBridge(models=[("set_transformer", "agg")], boost={"set_transformer": 0.3})
    result = training.run_training(
        db, store, bridge=bridge, models_dir=settings.models_dir,
        contours_db_path=settings.contours_db_path, trigger="manual",
    )
    assert result.models[0]["name"] == "set_transformer"
    assert result.promotion is not None
    assert model_svc.active_model(db)["name"] == "set_transformer"
    # a snapshot + training_run row + notification were written
    assert db.scalar("SELECT count(*) FROM dataset_snapshots") == 1
    assert db.query_one("SELECT status FROM training_runs")["status"] == "done"
    assert db.scalar("SELECT count(*) FROM notifications WHERE type = 'training_complete'") == 1


def test_training_run_persists_job_id_and_log_path(db, settings, store, tmp_path):
    log_path = tmp_path / "job123.log"
    result = training.run_training(
        db, store, bridge=FakeTrainingBridge(), models_dir=settings.models_dir,
        contours_db_path=settings.contours_db_path, job_id="job123", log_path=log_path,
    )
    row = db.query_one("SELECT job_id, log_path FROM training_runs WHERE id = ?", [result.run_id])
    assert row["job_id"] == "job123"
    assert row["log_path"] == str(log_path)


def test_training_run_no_promote_when_disabled(db, settings, store):
    store.set("auto_promote", False)
    training.run_training(
        db, store, bridge=FakeTrainingBridge(), models_dir=settings.models_dir,
        contours_db_path=settings.contours_db_path,
    )
    assert model_svc.active_model(db) is None


def test_training_due_by_time(db, store):
    store.update({"retrain_every_days": 60, "retrain_after_images": 0})
    assert training.training_due(db, store)["due"] is True  # never run
    db.insert("training_runs", {"id": "r1", "trigger": "manual", "status": "done"})
    assert training.training_due(db, store)["due"] is False


def test_singleton_hint(db, settings, dataset_dir):
    ingest.transfer_dataset(db, settings, dataset_dir)
    # bb_1 and dd_1 have a single real photo (dd_1_1 is disqualified though)
    assert model_svc.singleton_hint(db)["singletons"] >= 1


class TestImportModel:
    """Spec §7.6 — bring an already-trained checkpoint onto the volume."""

    @pytest.fixture
    def weights_file(self, tmp_path):
        p = tmp_path / "source" / "e2e_ckpt_fold0.pt"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"not-really-a-checkpoint")
        return p

    def test_copies_weights_and_registers(self, db, settings, store, weights_file):
        result = model_svc.import_model(
            db, settings, name="e2e_transformer_fold0", kind="aggregator",
            source_weights_path=weights_file,
            metrics={"r1": 0.19, "novelty_auroc": 0.59},
            coefficients=store.get("score_coefficients"),
        )
        dest = settings.models_dir / "e2e_transformer_fold0" / "weights.pt"
        assert dest.exists()
        assert dest.read_bytes() == weights_file.read_bytes()
        assert result["score"] == pytest.approx(0.5 * 0.19 + 0.5 * 0.59)

        row = db.query_one("SELECT * FROM models WHERE name = ?", ["e2e_transformer_fold0"])
        assert row["status"] == "candidate"
        assert row["weights_path"] == str(dest)
        assert db.scalar("SELECT count(*) FROM audit WHERE action = 'import_model'") == 1

    def test_copies_calibration_when_given(self, db, settings, store, weights_file, tmp_path):
        cal = tmp_path / "calibration.json"
        cal.write_text('{"a": 6.0, "b": -3.0}')
        model_svc.import_model(
            db, settings, name="m1", kind="agg", source_weights_path=weights_file,
            source_calibration_path=cal, coefficients=store.get("score_coefficients"),
        )
        dest = settings.models_dir / "m1" / "calibration.json"
        assert dest.read_text() == cal.read_text()

    def test_missing_weights_file_rejected(self, db, settings, store, tmp_path):
        with pytest.raises(model_svc.ModelImportError, match="not found"):
            model_svc.import_model(
                db, settings, name="m1", kind="agg",
                source_weights_path=tmp_path / "nope.pt",
                coefficients=store.get("score_coefficients"),
            )

    def test_bad_name_rejected(self, db, settings, store, weights_file):
        with pytest.raises(model_svc.ModelImportError, match="invalid model name"):
            model_svc.import_model(
                db, settings, name="../escape", kind="agg", source_weights_path=weights_file,
                coefficients=store.get("score_coefficients"),
            )

    def test_metrics_optional_scores_zero(self, db, settings, store, weights_file):
        result = model_svc.import_model(
            db, settings, name="m1", kind="agg", source_weights_path=weights_file,
            coefficients=store.get("score_coefficients"),
        )
        assert result["score"] == 0.0

    def test_reimport_updates_in_place(self, db, settings, store, weights_file):
        model_svc.import_model(db, settings, name="m1", kind="agg", source_weights_path=weights_file,
                               metrics={"r1": 0.1}, coefficients=store.get("score_coefficients"))
        model_svc.import_model(db, settings, name="m1", kind="agg", source_weights_path=weights_file,
                               metrics={"r1": 0.9}, coefficients=store.get("score_coefficients"))
        assert db.scalar("SELECT count(*) FROM models WHERE name = 'm1'") == 1
        assert db.query_one("SELECT r1 FROM models WHERE name = 'm1'")["r1"] == 0.9
