"""DuckDB access: schema migrations + a single-writer connection (spec §4.1c).

For this foundation slice, *all* DB access — reads and writes — is serialized
through one connection guarded by a re-entrant lock. The spec's richer model
(a writer queue + short-lived read connections) is a later refinement; the
public surface here (``Database.execute`` / ``Database.transaction``) does not
change when that lands.
"""

from __future__ import annotations

import json
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import duckdb

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def new_id(prefix: str = "") -> str:
    """A short unique id for a table row."""
    token = uuid.uuid4().hex[:16]
    return f"{prefix}_{token}" if prefix else token


class Database:
    """A serialized DuckDB handle for one data dir."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.RLock()
        self._con = duckdb.connect(self.path)
        self._main_catalog: str | None = None

    # -- lifecycle ------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._con.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- migrations ---------------------------------------------------------
    def migrate(self) -> list[str]:
        """Apply any pending ``migrations/*.sql`` in filename order. Returns the
        names applied this call. Idempotent."""
        with self._lock:
            self._con.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(name VARCHAR PRIMARY KEY, applied_at TIMESTAMP DEFAULT now())"
            )
            done = {
                r[0]
                for r in self._con.execute("SELECT name FROM schema_migrations").fetchall()
            }
            applied: list[str] = []
            for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
                if sql_file.name in done:
                    continue
                self._con.execute("BEGIN")
                try:
                    self._con.execute(sql_file.read_text(encoding="utf-8"))
                    self._con.execute(
                        "INSERT INTO schema_migrations (name) VALUES (?)", [sql_file.name]
                    )
                    self._con.execute("COMMIT")
                except Exception:
                    self._con.execute("ROLLBACK")
                    raise
                applied.append(sql_file.name)
            return applied

    # -- queries ----------------------------------------------------------
    def execute(self, sql: str, params: Sequence[Any] | None = None):
        with self._lock:
            return self._con.execute(sql, list(params) if params else None)

    def query(self, sql: str, params: Sequence[Any] | None = None) -> list[tuple]:
        with self._lock:
            return self._con.execute(sql, list(params) if params else None).fetchall()

    def query_dicts(self, sql: str, params: Sequence[Any] | None = None) -> list[dict]:
        with self._lock:
            cur = self._con.execute(sql, list(params) if params else None)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def query_one(self, sql: str, params: Sequence[Any] | None = None) -> dict | None:
        rows = self.query_dicts(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        row = self.query(sql, params)
        return row[0][0] if row else None

    @contextmanager
    def transaction(self) -> Iterator["Database"]:
        """A write transaction. The lock is held for its whole duration, so
        nested :meth:`execute` calls run on the same connection atomically."""
        with self._lock:
            self._con.execute("BEGIN")
            try:
                yield self
                self._con.execute("COMMIT")
            except Exception:
                self._con.execute("ROLLBACK")
                raise

    @contextmanager
    def attach(self, other_path: str | Path, alias: str, *, read_only: bool = True) -> Iterator[str]:
        """ATTACH another DuckDB file for the duration of the block."""
        ro = " (READ_ONLY)" if read_only else ""
        with self._lock:
            self._con.execute(f"ATTACH '{other_path}' AS {alias}{ro}")
            try:
                yield alias
            finally:
                self._con.execute(f"DETACH {alias}")

    # -- hot (live) snapshots ------------------------------------------------
    # DuckDB locks its file exclusively for the life of a connection (notably on
    # Windows, an external process — even read-only — is refused). A **live**
    # connection can still produce an independent, immediately-reopenable copy
    # via `COPY FROM DATABASE`, which is how backup/export stay usable while the
    # server keeps running (spec §2.2's "consistent copy", without a stop-the-
    # server requirement).
    @property
    def main_catalog(self) -> str:
        if self._main_catalog is None:
            with self._lock:
                self._main_catalog = self._con.execute("PRAGMA database_list").fetchone()[1]
        return self._main_catalog

    def snapshot_to(self, dest_path: str | Path) -> None:
        """Write a consistent, standalone copy of *this* database to ``dest_path``
        while staying open — safe to call from a running server."""
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists():
            dest_path.unlink()
        with self._lock:
            self._con.execute(f"ATTACH '{dest_path}' AS snap_target")
            try:
                self._con.execute(f"COPY FROM DATABASE {self.main_catalog} TO snap_target")
            finally:
                self._con.execute("DETACH snap_target")

    def snapshot_attached_to(self, source_path: str | Path, dest_path: str | Path) -> None:
        """Same idea for a file this connection does not otherwise hold open
        (e.g. ``contours.db``) — attach it, copy it, detach, all through the one
        live connection so nothing ever opens the raw file with a second handle."""
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists():
            dest_path.unlink()
        with self._lock:
            self._con.execute(f"ATTACH '{source_path}' AS snap_source (READ_ONLY)")
            self._con.execute(f"ATTACH '{dest_path}' AS snap_target")
            try:
                self._con.execute("COPY FROM DATABASE snap_source TO snap_target")
            finally:
                self._con.execute("DETACH snap_target")
                self._con.execute("DETACH snap_source")

    # -- helpers ----------------------------------------------------------
    def insert(self, table: str, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        placeholders = ", ".join("?" for _ in row)
        self.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", list(row.values())
        )

    def audit(
        self,
        *,
        action: str,
        entity: str,
        entity_id: str | None = None,
        actor: str | None = None,
        before: Any = None,
        after: Any = None,
    ) -> None:
        self.insert(
            "audit",
            {
                "id": new_id("aud"),
                "actor": actor,
                "action": action,
                "entity": entity,
                "entity_id": entity_id,
                "before_json": json.dumps(before) if before is not None else None,
                "after_json": json.dumps(after) if after is not None else None,
            },
        )

    def log_activity(
        self,
        *,
        kind: str,
        summary: str,
        detail: str | None = None,
        actor: str | None = None,
        ref_type: str | None = None,
        ref_id: str | None = None,
    ) -> None:
        self.insert(
            "activity",
            {
                "id": new_id("act"),
                "kind": kind,
                "summary": summary,
                "detail": detail,
                "actor": actor,
                "ref_type": ref_type,
                "ref_id": ref_id,
            },
        )


def open_database(path: str | Path, *, migrate: bool = True) -> Database:
    db = Database(path)
    if migrate:
        db.migrate()
    return db
