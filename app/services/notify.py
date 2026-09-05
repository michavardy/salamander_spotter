"""Notifications + alerts (spec §9.3) and optional SMTP email (spec §10.12 WF4)."""

from __future__ import annotations

import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText

from ..db import Database, new_id


def raise_notification(
    db: Database,
    *,
    type: str,
    title: str,
    body: str | None = None,
    severity: str = "info",
    copy_text: str | None = None,
    action_label: str | None = None,
    action_href: str | None = None,
) -> str:
    nid = new_id("ntf")
    db.insert(
        "notifications",
        {
            "id": nid,
            "type": type,
            "severity": severity,
            "title": title,
            "body": body,
            "copy_text": copy_text,
            "action_label": action_label,
            "action_href": action_href,
        },
    )
    return nid


def list_notifications(db: Database, *, include_dismissed: bool = False) -> list[dict]:
    where = "" if include_dismissed else "WHERE dismissed_at IS NULL"
    return db.query_dicts(
        f"SELECT * FROM notifications {where} ORDER BY created_at DESC LIMIT 100"
    )


def unread_count(db: Database) -> int:
    return db.scalar("SELECT count(*) FROM notifications WHERE read_at IS NULL AND dismissed_at IS NULL") or 0


def mark_read(db: Database, nid: str) -> None:
    db.execute("UPDATE notifications SET read_at = now() WHERE id = ?", [nid])


def dismiss(db: Database, nid: str) -> None:
    db.execute("UPDATE notifications SET dismissed_at = now() WHERE id = ?", [nid])


def low_token_block(model: str, calls_left: int, resets: str, reviewer: str | None) -> str:
    """The copy-pastable text for the research team (spec §9.3)."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    who = reviewer or "a reviewer"
    return (
        f"Salamander Spotter: extraction LLM ({model}) low on tokens — "
        f"~{calls_left} calls left, resets {resets}. Reported by {who}, {ts}."
    )


def send_email(
    *, host: str, port: int, user: str | None, password: str | None,
    sender: str, recipients: list[str], subject: str, body: str, use_tls: bool = True,
) -> None:
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    with smtplib.SMTP(host, port, timeout=20) as s:
        if use_tls:
            s.starttls()
        if user and password:
            s.login(user, password)
        s.send_message(msg)
