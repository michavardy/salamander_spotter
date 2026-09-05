"""Salamander Spotter application backend.

This package implements the application that wraps the research pipeline
(``pipeline/``) — see ``docs/salamander_spotter_spec.md``.

The slice implemented so far covers the foundation (spec M0) and the data
layer + dataset transfer / incremental ingest (spec M1 core, §7.9):

* :mod:`app.config`          — settings + data-dir resolution
* :mod:`app.ids`             — the pipeline ID conventions (§6.1)
* :mod:`app.db`              — DuckDB schema, migrations, single-writer access
* :mod:`app.services.ingest` — one-time dataset transfer + incremental append (§7.9)
* :mod:`app.api`             — FastAPI app factory + read/import endpoints
"""

__version__ = "0.1.0"
