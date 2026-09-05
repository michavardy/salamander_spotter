"""End-to-end API smoke over every router (spec §10)."""

from __future__ import annotations

import io
import time

import pytest
from PIL import Image


def _wait_for_job(client, job_id: str, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in {"done", "failed", "cancelled"}:
            return body
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish in test")


def _jpeg() -> io.BytesIO:
    buf = io.BytesIO()
    Image.new("RGB", (160, 200), (20, 90, 40)).save(buf, "JPEG")
    buf.seek(0)
    return buf


@pytest.fixture
def loaded(client, dataset_dir):
    client.post("/api/imports", json={"dataset_path": str(dataset_dir)})
    return client


def test_dashboard_and_activity(loaded):
    body = loaded.get("/api/dashboard").json()
    assert body["stats"]["images"] == 6
    assert "census" in body and "map" in body
    assert loaded.get("/api/activity").json()["activity"]


def test_setup_status(loaded):
    s = loaded.get("/api/setup/status").json()
    assert s["has_data"] is True
    assert s["has_api_keys"] is False


def test_full_review_flow(loaded):
    up = loaded.post("/api/uploads", files={"file": ("f.jpg", _jpeg(), "image/jpeg")}).json()
    image_id = up["image_id"]

    m = loaded.post(f"/api/review/{image_id}/match").json()
    assert "candidates" in m and "triage" in m

    detail = loaded.get(f"/api/review/{image_id}").json()
    assert detail["image"]["image_id"] == image_id
    assert detail["reason_chips"]

    dec = loaded.post(f"/api/review/{image_id}/decision", json={"verdict": "new"}).json()
    assert dec.get("new_individual_id") or dec.get("needs_confirmation")


def test_batches_and_census(loaded):
    bid = loaded.post("/api/batches", json={"name": "Autumn"}).json()["batch_id"]
    pub = loaded.post(f"/api/batches/{bid}/publish").json()
    assert pub["batch_id"] == bid
    assert "published_individuals" in loaded.get("/api/census").json()


def test_settings_roundtrip_and_validation(loaded):
    s = loaded.get("/api/settings").json()
    assert s["settings"]["auto_approve_threshold"] == 0.95

    ok = loaded.patch("/api/settings", json={"values": {"coverage_target": 0.8}})
    assert ok.status_code == 200
    assert loaded.get("/api/settings").json()["settings"]["coverage_target"] == 0.8

    bad = loaded.patch("/api/settings", json={"values": {"score_coefficients": {"z": 1}}})
    assert bad.status_code == 400

    unknown = loaded.patch("/api/settings", json={"values": {"nope": 1}})
    assert unknown.status_code == 400


def test_score_preview(loaded):
    r = loaded.post("/api/settings/score-preview", json={
        "metrics": {"r1": 0.6, "novelty_auroc": 0.8}, "coefficients": {"a": 0.5, "e": 0.5},
    }).json()
    assert r["score"] == pytest.approx(0.7)


def test_models_and_training(loaded):
    run = loaded.post("/api/models/retrain").json()
    assert "job_id" in run
    done = _wait_for_job(loaded, run["job_id"])
    assert done["status"] == "done"
    models = loaded.get("/api/models").json()
    assert models["models"]
    if models["active"]:
        loaded.post("/api/models/rollback")  # tolerated even if nothing to roll back to


def test_import_model_endpoint(loaded, tmp_path):
    weights = tmp_path / "e2e_ckpt_fold0.pt"
    weights.write_bytes(b"checkpoint-bytes")

    r = loaded.post(
        "/api/models/import",
        json={
            "name": "e2e_transformer_fold0",
            "kind": "aggregator",
            "source_weights_path": str(weights),
            "metrics": {"r1": 0.19, "novelty_auroc": 0.59},
            "make_active": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "e2e_transformer_fold0"
    assert body["score"] == pytest.approx(0.5 * 0.19 + 0.5 * 0.59)

    listed = loaded.get("/api/models").json()
    assert listed["active"]["name"] == "e2e_transformer_fold0"
    assert any(m["name"] == "e2e_transformer_fold0" for m in listed["models"])


def test_import_model_missing_file_is_400(loaded, tmp_path):
    r = loaded.post(
        "/api/models/import",
        json={"name": "m1", "source_weights_path": str(tmp_path / "nope.pt")},
    )
    assert r.status_code == 400


def test_extraction_editor_endpoint(loaded):
    r = loaded.post("/api/images/aa_1_1/extraction/edit",
                    json={"ops": [{"op": "join_spots", "spot_ids": [1, 2]}]})
    assert r.status_code == 200
    body = r.json()
    assert body["n_spots"] == 7  # was 8

    rev = loaded.post("/api/images/aa_1_1/extraction/revert",
                      json={"correction_id": body["correction_id"]})
    assert rev.status_code == 200


def test_exports(loaded):
    r = loaded.post("/api/exports/census?fmt=xlsx").json()
    assert r["name"].endswith(".xlsx")
    dl = loaded.get(r["download"])
    assert dl.status_code == 200
    assert loaded.get("/api/exports").json()["exports"]


def test_estimate_endpoint(loaded):
    r = loaded.post("/api/uploads/estimate", json={"count": 5}).json()
    assert r["billed_calls_estimate"] == 20


def test_fully_disqualified_individual_hidden_from_roster(loaded):
    # dd_1's only image (dd_1_1) is disqualified by the fixture's exclude correction —
    # it should not clutter the Names roster with an all-zero row.
    ids = {r["individual_id"] for r in loaded.get("/api/individuals").json()["individuals"]}
    assert "dd_1" not in ids
    assert "aa_1" in ids


def test_individual_patch_optimistic_concurrency(loaded):
    ind = loaded.get("/api/individuals/aa_1").json()["individual"]
    ok = loaded.patch("/api/individuals/aa_1", json={"nickname": "Spot", "rev": ind["rev"]})
    assert ok.status_code == 200
    stale = loaded.patch("/api/individuals/aa_1", json={"nickname": "X", "rev": ind["rev"]})
    assert stale.status_code == 409


def test_sse_events_stream_opens(loaded):
    r = loaded.get("/api/events?once=1")
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "event: hello" in r.text
