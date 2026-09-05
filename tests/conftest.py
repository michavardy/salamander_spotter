from __future__ import annotations

from pathlib import Path

import pytest

from app.bridges import Bridges
from app.config import Settings, load_settings
from app.db import Database, open_database
from app.settings_store import SettingsStore
from tests.fixtures import build_fixture_dataset


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = load_settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    return s


@pytest.fixture
def db(settings: Settings) -> Database:
    database = open_database(settings.app_db_path)
    yield database
    database.close()


@pytest.fixture
def store(db: Database) -> SettingsStore:
    return SettingsStore(db)


@pytest.fixture
def bridges() -> Bridges:
    return Bridges.fakes()


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    return build_fixture_dataset(tmp_path / "datasets")


@pytest.fixture
def client(settings: Settings, db: Database, bridges: Bridges):
    from fastapi.testclient import TestClient

    from app.api import create_app

    app = create_app(settings, db=db, bridges=bridges)
    with TestClient(app) as c:
        c.get("/api/health")
        token = c.cookies.get(settings.csrf_cookie_name)
        if token:
            c.headers.update({"x-csrf-token": token})
        yield c
