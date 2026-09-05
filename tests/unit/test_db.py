from __future__ import annotations

import threading

import duckdb

from app.db import Database, open_database


def test_migrate_creates_tables_and_is_idempotent(settings):
    db = open_database(settings.app_db_path, migrate=False)
    applied = db.migrate()
    assert "0001_initial.sql" in applied
    assert db.migrate() == []  # nothing pending on the second call

    tables = {r[0] for r in db.query("SELECT table_name FROM information_schema.tables")}
    assert {"images", "individuals", "review_batches", "audit", "activity"} <= tables
    db.close()


def test_insert_and_query_helpers(db):
    db.insert("contributors", {"id": "c1", "name": "Dana", "short_name": "D"})
    row = db.query_one("SELECT * FROM contributors WHERE id = ?", ["c1"])
    assert row["name"] == "Dana"
    assert db.scalar("SELECT count(*) FROM contributors") == 1


def test_transaction_rolls_back_on_error(db):
    try:
        with db.transaction():
            db.insert("sites", {"id": "s1", "name": "A"})
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert db.scalar("SELECT count(*) FROM sites") == 0


def test_audit_and_activity_helpers(db):
    db.audit(action="test", entity="thing", entity_id="x", after={"k": 1})
    db.log_activity(kind="test", summary="did a thing")
    assert db.scalar("SELECT count(*) FROM audit") == 1
    assert db.query_one("SELECT * FROM activity")["summary"] == "did a thing"


def test_snapshot_to_is_independent_and_hot(db, tmp_path):
    """The live connection stays open (as it would in a running server) and the
    snapshot must still be a standalone file another connection can open."""
    db.insert("contributors", {"id": "c1", "name": "Dana"})
    snap_path = tmp_path / "app_snapshot.duckdb"
    db.snapshot_to(snap_path)

    assert snap_path.exists()
    other = open_database(snap_path, migrate=False)
    try:
        assert other.scalar("SELECT count(*) FROM contributors") == 1
    finally:
        other.close()
    # the source connection is still perfectly usable
    assert db.scalar("SELECT count(*) FROM contributors") == 1


def test_snapshot_attached_to_copies_a_second_file_live(db, tmp_path):
    src = tmp_path / "contours.db"
    con = duckdb.connect(str(src))
    con.execute("CREATE TABLE images (salamander_id VARCHAR)")
    con.execute("INSERT INTO images VALUES ('aa_1_1')")
    con.close()

    dest = tmp_path / "contours_snapshot.db"
    db.snapshot_attached_to(src, dest)

    check = duckdb.connect(str(dest), read_only=True)
    try:
        assert check.execute("SELECT count(*) FROM images").fetchone()[0] == 1
    finally:
        check.close()


def test_concurrent_writes_are_serialized(db):
    def worker(n: int):
        for i in range(20):
            db.insert("activity", {"id": f"{n}-{i}", "kind": "k", "summary": "s"})

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert db.scalar("SELECT count(*) FROM activity") == 80
