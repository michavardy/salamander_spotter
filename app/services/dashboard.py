"""Dashboard aggregations + sightings-map data (spec §10.1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..db import Database
from . import batches as batch_svc
from . import notify


def _delta_this_month(db: Database, table: str, ts_col: str) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    return db.scalar(f"SELECT count(*) FROM {table} WHERE {ts_col} >= ?", [cutoff]) or 0


def overview(db: Database) -> dict:
    stats = {
        "individuals": db.scalar(
            "SELECT count(*) FROM individuals WHERE merged_into IS NULL AND status != 'merged_into'"
        ) or 0,
        "images": db.scalar("SELECT count(*) FROM images WHERE status != 'disqualified'") or 0,
        "contributors": db.scalar("SELECT count(*) FROM contributors") or 0,
        "synthetic_views": db.scalar("SELECT count(*) FROM images WHERE is_synthetic") or 0,
    }
    deltas = {
        "individuals": _delta_this_month(db, "individuals", "created_at"),
        "images": _delta_this_month(db, "images", "created_at"),
        "contributors": _delta_this_month(db, "contributors", "created_at"),
    }
    # individuals sparkline — count per week for the last 7 weeks
    sparkline = []
    now = datetime.now(timezone.utc)
    for wk in range(6, -1, -1):
        hi = now - timedelta(weeks=wk)
        sparkline.append(db.scalar("SELECT count(*) FROM individuals WHERE created_at <= ?", [hi]) or 0)

    activity = db.query_dicts(
        "SELECT kind, summary, detail, actor, ref_type, ref_id, created_at "
        "FROM activity ORDER BY created_at DESC LIMIT 25"
    )
    tiers = {
        r["ladder_tier"] or "unknown": r["n"]
        for r in db.query_dicts("SELECT ladder_tier, count(*) AS n FROM images GROUP BY 1")
    }
    queue = db.scalar("SELECT count(*) FROM images WHERE status = 'in_review'") or 0
    alert = db.query_one(
        "SELECT * FROM notifications WHERE dismissed_at IS NULL "
        "ORDER BY (severity = 'warning') DESC, created_at DESC LIMIT 1"
    )
    return {
        "stats": stats,
        "deltas": deltas,
        "sparklines": {"individuals": sparkline},
        "tiers": tiers,
        "review_queue": queue,
        "census": batch_svc.census(db),
        "activity": activity,
        "alert": alert,
        "notifications_unread": notify.unread_count(db),
        "map": map_data(db),
    }


def map_data(db: Database) -> dict:
    sites = db.query_dicts("SELECT id, name, lat, lon FROM sites WHERE lat IS NOT NULL")
    recent_cut = datetime.now(timezone.utc) - timedelta(days=14)
    markers = db.query_dicts(
        """
        SELECT img.image_id, img.individual_id, img.status, img.photographed_at,
               s.lat, s.lon,
               (img.created_at >= ?) AS is_recent
        FROM images img JOIN sites s ON s.id = img.site_id
        WHERE s.lat IS NOT NULL AND img.status != 'disqualified'
        ORDER BY img.created_at DESC LIMIT 500
        """,
        [recent_cut],
    )
    return {"sites": sites, "markers": markers}
