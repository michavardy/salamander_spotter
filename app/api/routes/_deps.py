"""Shared request accessors for the route modules."""

from __future__ import annotations

from fastapi import Request

from ...config import Settings
from ...db import Database
from ...settings_store import SecretStore, SettingsStore


def db(request: Request) -> Database:
    return request.app.state.db


def settings(request: Request) -> Settings:
    return request.app.state.settings


def store(request: Request) -> SettingsStore:
    return request.app.state.settings_store


def secrets_store(request: Request) -> SecretStore:
    return request.app.state.secret_store


def worker(request: Request):
    return request.app.state.worker


def bridges(request: Request):
    return request.app.state.bridges


def actor(request: Request) -> str | None:
    return request.headers.get("x-actor") or request.app.state.settings.actor_label
