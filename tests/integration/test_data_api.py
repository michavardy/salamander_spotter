"""Spec §4.3 (paths) and §2.2/§2.3 (live backup/export via the API + job polling)."""

from __future__ import annotations

import time

import pytest


def _wait_for_job(client, job_id: str, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in {"done", "failed", "cancelled"}:
            return body
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish in test")


def test_settings_paths_reports_resolved_locations(client, settings, dataset_dir):
    client.post("/api/imports", json={"dataset_path": str(dataset_dir)})
    r = client.get("/api/settings/paths")
    assert r.status_code == 200
    body = r.json()
    assert body["data_dir"]["path"] == str(settings.data_dir)
    assert body["app_db"]["exists"] is True
    assert body["images_raw"]["n_files"] == 7  # the 7 fixture images
    assert body["env_var"] == "SPOTTER_DATA_DIR"


def test_backup_endpoint_runs_live_and_is_pollable(client, dataset_dir, settings):
    client.post("/api/imports", json={"dataset_path": str(dataset_dir)})
    r = client.post("/api/backup", json={})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    done = _wait_for_job(client, job_id)
    assert done["status"] == "done"
    assert done["result"]["archive"].endswith(".tar.gz")

    # the app is still fully responsive during/after — same client, same connection
    assert client.get("/api/dashboard").status_code == 200


def test_export_full_endpoint_produces_a_downloadable_archive(client, dataset_dir):
    client.post("/api/imports", json={"dataset_path": str(dataset_dir)})
    r = client.post("/api/exports/full", json={})
    assert r.status_code == 200
    body = r.json()
    job_id, download = body["job_id"], body["download"]

    done = _wait_for_job(client, job_id)
    assert done["status"] == "done"

    dl = client.get(download)
    assert dl.status_code == 200
    assert dl.headers["content-type"] in (
        "application/gzip", "application/x-gzip", "application/x-tar", "application/octet-stream",
    )


def test_job_not_found(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404
