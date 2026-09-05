"""API surface (spec §10 subset) + CSRF guard (§15)."""

from __future__ import annotations

import io

from PIL import Image


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_runtime_config(client):
    r = client.get("/runtime-config.json")
    assert r.status_code == 200
    body = r.json()
    assert "features" in body and "version" in body


def test_csrf_blocks_mutating_without_header(settings, db):
    from fastapi.testclient import TestClient

    from app.api import create_app

    app = create_app(settings, db=db)
    with TestClient(app) as bare:
        bare.get("/api/health")  # sets the cookie
        # cookie present but no X-CSRF-Token header
        r = bare.post("/api/imports", json={"dataset_path": "/nope"})
        assert r.status_code == 403


def test_import_then_browse_roster(client, dataset_dir):
    r = client.post("/api/imports", json={"dataset_path": str(dataset_dir), "actor": "tester"})
    assert r.status_code == 200
    report = r.json()
    assert report["images_added"] == 7
    assert report["llm_calls"] == 0

    r = client.get("/api/individuals")
    assert r.status_code == 200
    ids = {i["individual_id"] for i in r.json()["individuals"]}
    assert {"aa_1", "bb_1", "cc_1"} <= ids

    r = client.get("/api/individuals/aa_1")
    assert r.status_code == 200
    body = r.json()
    assert body["individual"]["display_id"] == "AA-1"
    assert {img["image_id"] for img in body["images"]} == {"aa_1_1", "aa_1_2", "aa_1_g0"}

    r = client.get("/api/images/aa_1_1")
    assert r.json()["image"]["ladder_tier"] == "auto_accept"

    r = client.get("/api/images/does_not_exist")
    assert r.status_code == 404


def test_dashboard_counts(client, dataset_dir):
    client.post("/api/imports", json={"dataset_path": str(dataset_dir)})
    body = client.get("/api/dashboard").json()
    # 7 imported, dd_1_1 disqualified -> 6 counted; cc_2 merged -> not counted as an individual
    assert body["stats"]["images"] == 6
    assert body["stats"]["individuals"] == 4  # aa_1, bb_1, cc_1, dd_1
    assert body["stats"]["synthetic_views"] == 1
    assert body["tiers"]["auto_accept"] >= 1


def test_upload_endpoint_appends_sighting(client):
    buf = io.BytesIO()
    Image.new("RGB", (120, 160), (10, 80, 40)).save(buf, "JPEG")
    buf.seek(0)
    r = client.post("/api/uploads", files={"file": ("field.jpg", buf, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["image_id"].startswith("up_")
    assert body["status"] == "extracted"
    assert body["ladder_tier"] in {"auto_accept", "needs_a_look", "hand_correction"}

    got = client.get(f"/api/images/{body['image_id']}").json()["image"]
    assert got["origin"] == "upload"
