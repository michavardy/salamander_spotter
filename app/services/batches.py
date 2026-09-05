"""Batches / surveys and the published census (spec §9.2)."""

from __future__ import annotations

from datetime import datetime, timezone

from ..db import Database, new_id


def open_batches(db: Database) -> list[dict]:
    return db.query_dicts("SELECT * FROM review_batches ORDER BY created_at DESC")


def current_open_batch(db: Database) -> dict | None:
    return db.query_one(
        "SELECT * FROM review_batches WHERE status = 'open' ORDER BY created_at DESC LIMIT 1"
    )


def create_batch(db: Database, name: str, *, actor: str | None = None) -> str:
    """Open a new batch, closing any currently-open one (exactly one open at a time)."""
    bid = new_id("batch")
    with db.transaction():
        db.execute("UPDATE review_batches SET status = 'closed' WHERE status = 'open'")
        db.insert("review_batches", {"id": bid, "name": name, "status": "open"})
        db.audit(action="open_batch", entity="batch", entity_id=bid, actor=actor, after={"name": name})
        db.log_activity(kind="batch", summary=f"Opened batch “{name}”", actor=actor,
                        ref_type="batch", ref_id=bid)
    return bid


def batch_counts(db: Database, batch_id: str) -> dict:
    rows = db.query_dicts(
        "SELECT status, count(*) AS n FROM images WHERE review_batch_id = ? GROUP BY status",
        [batch_id],
    )
    by_status = {r["status"]: r["n"] for r in rows}
    total = sum(by_status.values())
    decided = sum(
        by_status.get(s, 0)
        for s in ("confirmed", "enrolled_new", "disqualified", "flagged_uncertain")
    )
    return {"total": total, "decided": decided, "undecided": total - decided, "by_status": by_status}


def publish_batch(db: Database, batch_id: str, *, actor: str | None = None) -> dict:
    """Freeze decisions, promote provisional individuals to published, stamp the census.
    Does not require every sighting decided (spec §9.2)."""
    batch = db.query_one("SELECT * FROM review_batches WHERE id = ?", [batch_id])
    if not batch:
        raise ValueError(f"unknown batch {batch_id}")
    if batch["status"] == "published":
        return {"batch_id": batch_id, "already_published": True}

    with db.transaction():
        # provisional individuals with a confirmed sighting in this batch -> published
        # (any still-uncertain sighting holds its individual back)
        db.execute(
            """
            UPDATE individuals SET status = 'published', rev = rev + 1
            WHERE status = 'provisional'
              AND individual_id IN (
                  SELECT individual_id FROM images
                  WHERE review_batch_id = ? AND status IN ('confirmed','enrolled_new')
              )
              AND individual_id NOT IN (
                  SELECT individual_id FROM images
                  WHERE status = 'flagged_uncertain' AND individual_id IS NOT NULL
              )
            """,
            [batch_id],
        )
        db.execute(
            "UPDATE review_batches SET status = 'published', published_at = now() WHERE id = ?",
            [batch_id],
        )
        counts = batch_counts(db, batch_id)
        db.audit(action="publish_batch", entity="batch", entity_id=batch_id, actor=actor, after=counts)
        db.log_activity(kind="batch", summary=f"Published batch “{batch['name']}”", actor=actor,
                        ref_type="batch", ref_id=batch_id)
    return {"batch_id": batch_id, **counts}


def unpublish_batch(db: Database, batch_id: str, *, actor: str | None = None) -> None:
    with db.transaction():
        db.execute(
            "UPDATE review_batches SET status = 'open', published_at = NULL WHERE id = ?", [batch_id]
        )
        db.execute("UPDATE review_batches SET status = 'closed' WHERE status = 'open' AND id != ?", [batch_id])
        db.audit(action="unpublish_batch", entity="batch", entity_id=batch_id, actor=actor)


def census(db: Database) -> dict:
    """The published census = confirmed sightings in published batches, counting
    published individuals (spec §9.2)."""
    published_individuals = db.scalar(
        """
        SELECT count(DISTINCT i.individual_id)
        FROM individuals i
        JOIN images img ON img.individual_id = i.individual_id
        JOIN review_batches b ON b.id = img.review_batch_id
        WHERE i.status = 'published' AND i.merged_into IS NULL
          AND b.status = 'published'
          AND img.status IN ('confirmed','enrolled_new')
        """
    )
    confirmed_sightings = db.scalar(
        """
        SELECT count(*)
        FROM images img JOIN review_batches b ON b.id = img.review_batch_id
        WHERE b.status = 'published' AND img.status IN ('confirmed','enrolled_new')
        """
    )
    contributors = db.scalar(
        """
        SELECT count(DISTINCT img.contributor_id)
        FROM images img JOIN review_batches b ON b.id = img.review_batch_id
        WHERE b.status = 'published' AND img.contributor_id IS NOT NULL
        """
    )
    provisional = db.scalar("SELECT count(*) FROM individuals WHERE status = 'provisional'")
    return {
        "published_individuals": published_individuals or 0,
        "confirmed_sightings": confirmed_sightings or 0,
        "contributors": contributors or 0,
        "provisional_individuals": provisional or 0,
    }
