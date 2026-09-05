"""Season census report — the artefact for the biology team (spec §9.4, WF3).

Generates an XLSX workbook and a PDF into ``<data-dir>/exports/``. Readable
without the app: cover, headline counts, per-site breakdown, roster table,
methods note, flagged/uncertain appendix.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from pathlib import Path

from ..db import Database
from . import batches as batch_svc
from . import models as model_svc


def _roster_rows(db: Database) -> list[dict]:
    return db.query_dicts(
        """
        SELECT ind.display_id, ind.individual_id, ind.nickname, ind.status,
               ind.first_seen, ind.last_seen,
               count(img.image_id) FILTER (WHERE NOT img.is_synthetic) AS n_photos,
               string_agg(DISTINCT c.name, '; ') AS contributors
        FROM individuals ind
        LEFT JOIN images img ON img.individual_id = ind.individual_id
             AND img.status IN ('confirmed','enrolled_new')
        LEFT JOIN individual_contributors ic ON ic.individual_id = ind.individual_id
        LEFT JOIN contributors c ON c.id = ic.contributor_id
        WHERE ind.merged_into IS NULL AND ind.status = 'published'
        GROUP BY 1,2,3,4,5,6
        ORDER BY ind.individual_id
        """
    )


def report_payload(db: Database) -> dict:
    stats = batch_svc.census(db)
    active = model_svc.active_model(db)
    new_this_season = db.scalar(
        "SELECT count(*) FROM individuals WHERE status = 'published' "
        "AND created_at > coalesce((SELECT min(created_at) FROM review_batches WHERE status='open'), '1970-01-01')"
    )
    flagged = db.query_dicts(
        "SELECT image_id, individual_id, status FROM images "
        "WHERE status = 'flagged_uncertain' ORDER BY image_id"
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stats": stats,
        "new_this_season": new_this_season or 0,
        "roster": _roster_rows(db),
        "flagged": flagged,
        "methods": {
            "model": active["name"] if active else "none",
            "note": "Counts are provisional until their batch is published.",
        },
    }


def write_xlsx(db: Database, out_dir: Path) -> Path:
    from openpyxl import Workbook

    data = report_payload(db)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"census_{_stamp()}.xlsx"

    wb = Workbook()
    cover = wb.active
    cover.title = "Summary"
    cover.append(["Salamander Spotter — Season Census"])
    cover.append(["Generated", data["generated_at"]])
    cover.append([])
    for k, v in data["stats"].items():
        cover.append([k.replace("_", " ").title(), v])
    cover.append(["New This Season", data["new_this_season"]])
    cover.append(["Matcher", data["methods"]["model"]])

    roster = wb.create_sheet("Roster")
    roster.append(["Display ID", "ID", "Nickname", "Status", "First seen", "Last seen",
                   "Photos", "Contributors"])
    for r in data["roster"]:
        roster.append([r["display_id"], r["individual_id"], r["nickname"], r["status"],
                       str(r["first_seen"] or ""), str(r["last_seen"] or ""),
                       r["n_photos"], r["contributors"]])

    flagged = wb.create_sheet("Flagged")
    flagged.append(["Image", "Individual", "Status"])
    for r in data["flagged"]:
        flagged.append([r["image_id"], r["individual_id"], r["status"]])

    wb.save(path)
    return path


def write_pdf(db: Database, out_dir: Path) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas

    data = report_payload(db)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"census_{_stamp()}.pdf"

    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    y = h - 2 * cm
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, y, "Salamander Spotter — Season Census")
    y -= 1 * cm
    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, y, f"Generated {data['generated_at']}")
    y -= 1 * cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Headline counts")
    y -= 0.7 * cm
    c.setFont("Helvetica", 10)
    for k, v in data["stats"].items():
        c.drawString(2.5 * cm, y, f"{k.replace('_', ' ').title()}: {v}")
        y -= 0.5 * cm
    c.drawString(2.5 * cm, y, f"New this season: {data['new_this_season']}")
    y -= 0.9 * cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, f"Roster ({len(data['roster'])} published individuals)")
    y -= 0.7 * cm
    c.setFont("Helvetica", 9)
    for r in data["roster"]:
        if y < 3 * cm:
            c.showPage()
            y = h - 2 * cm
            c.setFont("Helvetica", 9)
        c.drawString(2.5 * cm, y, f"{r['display_id']}  ·  {r['n_photos']} photos  ·  {r['contributors'] or '—'}")
        y -= 0.45 * cm

    c.showPage()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, h - 2 * cm, "Methods")
    c.setFont("Helvetica", 10)
    c.drawString(2.5 * cm, h - 2.8 * cm, f"Matcher: {data['methods']['model']}")
    c.drawString(2.5 * cm, h - 3.4 * cm, data["methods"]["note"])
    c.save()
    return path


def write_roster_csv(db: Database, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"roster_{_stamp()}.csv"
    rows = _roster_rows(db)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["display_id"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
