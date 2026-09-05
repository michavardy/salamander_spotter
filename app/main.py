"""Entry point for ``uvicorn app.main:app`` and ``app serve`` (spec §5.1, §5.4)."""

from __future__ import annotations

from .api import create_app

app = create_app()
