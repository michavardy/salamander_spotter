"""Aggregated API router (spec §10 route map)."""

from __future__ import annotations

from fastapi import APIRouter

from . import (
    dashboard,
    data,
    events,
    exports,
    health,
    imports,
    individuals,
    jobs,
    models,
    review,
    settings as settings_routes,
    setup,
    upload,
)

router = APIRouter()
router.include_router(health.router, tags=["health"])
router.include_router(dashboard.router, tags=["dashboard"])
router.include_router(imports.router, tags=["imports"])
router.include_router(upload.router, tags=["upload"])
router.include_router(review.router, tags=["review"])
router.include_router(individuals.router, tags=["individuals"])
router.include_router(models.router, tags=["models"])
router.include_router(settings_routes.router, tags=["settings"])
router.include_router(exports.router, tags=["exports"])
router.include_router(data.router, tags=["data"])
router.include_router(jobs.router, tags=["jobs"])
router.include_router(setup.router, tags=["setup"])
router.include_router(events.router, tags=["events"])
