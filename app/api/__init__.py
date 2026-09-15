"""FastAPI application factory (spec §4).

Assembles: settings + secret stores, the DuckDB handle, the background worker
with its job runners, the pipeline-bridge bundle, the CSRF guard, every API
router, and the SPA/static mount.
"""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..bridges import Bridges
from ..config import Settings, load_settings
from ..db import Database, open_database
from ..job_runners import register_all
from ..settings_store import SecretStore, SettingsStore
from ..worker import Worker
from .routes import router as api_router

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def create_app(
    settings: Settings | None = None,
    *,
    db: Database | None = None,
    bridges: Bridges | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    settings.ensure_dirs()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.worker.shutdown()
        if app.state.owns_db:  # pragma: no cover - lifecycle
            app.state.db.close()

    app = FastAPI(
        title="Salamander Spotter",
        version=__version__,
        docs_url="/api/docs" if settings.expose_api_docs else None,
        redoc_url=None,
        openapi_url="/api/openapi.json" if settings.expose_api_docs else None,
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.db = db or open_database(settings.app_db_path)
    app.state.owns_db = db is None
    app.state.settings_store = SettingsStore(app.state.db)
    app.state.secret_store = SecretStore(settings.data_dir / "secrets.json")
    if bridges is None:
        from ..services.models import active_model as _active_model_row

        _active = _active_model_row(app.state.db)
        app.state.bridges = Bridges.production(
            active_model=_active["name"] if _active else None,
            weights_path=_active.get("weights_path") if _active else None,
        )
    else:
        app.state.bridges = bridges
    app.state.worker = Worker(app.state.db)
    register_all(
        app.state.worker,
        db=app.state.db,
        settings=settings,
        store=app.state.settings_store,
        bridges=app.state.bridges,
    )

    @app.middleware("http")
    async def csrf_guard(request: Request, call_next):
        cookie_name = settings.csrf_cookie_name
        token = request.cookies.get(cookie_name)
        path = request.url.path
        exempt = path.startswith("/api/health") or path.startswith("/api/events")
        if request.method not in SAFE_METHODS and not exempt:
            header = request.headers.get("x-csrf-token")
            if not token or not header or not secrets.compare_digest(header, token):
                return JSONResponse({"detail": "missing or invalid CSRF token"}, status_code=403)
        response = await call_next(request)
        if not token:
            response.set_cookie(
                cookie_name, secrets.token_urlsafe(32), samesite="strict", httponly=False
            )
        return response

    @app.get("/runtime-config.json")
    def runtime_config() -> dict:
        return {
            "external_base_url": settings.external_base_url,
            "actor_label": settings.actor_label,
            "features": {"upload": True, "review": True, "models": True, "map": True},
            "version": __version__,
        }

    app.include_router(api_router, prefix="/api")
    _mount_media(app, settings)
    _mount_spa(app, settings)
    return app


def _mount_media(app: FastAPI, settings: Settings) -> None:
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=str(settings.images_dir)), name="media")


def _mount_spa(app: FastAPI, settings: Settings) -> None:
    """Serve the static Next export (spec §4.2). Real files win; a bare path like
    ``/review`` falls back to ``review.html``; anything else to ``index.html`` so
    deep links / query-string routes resolve client-side."""
    web_root = Path(__file__).resolve().parent.parent / "resources" / "web"
    index = web_root / "index.html"

    if not index.exists():
        @app.get("/")
        def placeholder() -> dict:
            return {
                "app": "salamander-spotter",
                "version": __version__,
                "note": "web bundle not built (run `pixi run ui:build && pixi run ui:bake`); API under /api",
            }
        return

    if (web_root / "_next").is_dir():
        app.mount("/_next", StaticFiles(directory=str(web_root / "_next")), name="next-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith(("api/", "media/", "_next/")) or full_path == "runtime-config.json":
            return JSONResponse({"detail": "not found"}, status_code=404)
        candidate = (web_root / full_path).resolve()
        if web_root in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        html = (web_root / f"{full_path}.html").resolve() if full_path else index
        if full_path and web_root in html.parents and html.is_file():
            return FileResponse(html)
        return FileResponse(index)
