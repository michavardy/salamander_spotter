"""In-place extraction correction (spec §10.6, M4).

Edits touch ``contours.db`` only; the app records an ``extraction_corrections``
row with a ``prev_snapshot`` for undo, and the correction feeds the next
training snapshot as a high-value label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb

from ..db import Database, new_id
from ..pipeline_bridge.editor import EditorBridge
from .quality_ladder import LadderConfig, DEFAULT_LADDER, ladder_tier

_SNAPSHOT_TABLES = ("spots", "body_axis")


def snapshot_extraction(contours_db_path: Path, image_id: str) -> dict:
    con = duckdb.connect(str(contours_db_path), read_only=True)
    try:
        out: dict[str, list] = {}
        for table in _SNAPSHOT_TABLES:
            cur = con.execute(f"SELECT * FROM {table} WHERE salamander_id = ?", [image_id])
            cols = [d[0] for d in cur.description]
            out[table] = [dict(zip(cols, row)) for row in cur.fetchall()]
        return out
    finally:
        con.close()


def _restore_snapshot(contours_db_path: Path, image_id: str, snapshot: dict) -> None:
    con = duckdb.connect(str(contours_db_path))
    try:
        for table, rows in snapshot.items():
            con.execute(f"DELETE FROM {table} WHERE salamander_id = ?", [image_id])
            for row in rows:
                cols = ", ".join(row)
                ph = ", ".join("?" for _ in row)
                con.execute(f"INSERT INTO {table} ({cols}) VALUES ({ph})", list(row.values()))
    finally:
        con.close()


@dataclass
class EditResult:
    correction_id: str
    n_spots: int
    ladder_tier: str | None
    overall_quality: float | None


def apply_edit(
    db: Database,
    contours_db_path: Path,
    *,
    image_id: str,
    ops: list[dict],
    bridge: EditorBridge,
    editor_id: str | None = None,
    ladder: LadderConfig = DEFAULT_LADDER,
) -> EditResult:
    """Apply a list of edit ops, then re-bin + re-score.

    Supported ops (``{"op": ..., ...}``):
      * ``join_spots``  ``{"spot_ids": [a, b, ...]}``  — merge into the lowest id
      * ``split_spot``  ``{"spot_id": a}``             — duplicate as a new id (placeholder geometry)
      * ``delete_spot`` ``{"spot_id": a}``
      * ``set_head_tail`` ``{"head": [x,y], "tail": [x,y]}``
      * ``set_axis_ok`` ``{"judged_ok": bool}``
    """
    before = snapshot_extraction(contours_db_path, image_id)

    con = duckdb.connect(str(contours_db_path))
    try:
        for op in ops:
            kind = op["op"]
            if kind == "join_spots":
                ids = sorted(op["spot_ids"])
                keep, drop = ids[0], ids[1:]
                con.executemany(
                    "DELETE FROM spots WHERE salamander_id = ? AND spot_id = ?",
                    [(image_id, d) for d in drop],
                )
            elif kind == "delete_spot":
                con.execute(
                    "DELETE FROM spots WHERE salamander_id = ? AND spot_id = ?",
                    [image_id, op["spot_id"]],
                )
            elif kind == "split_spot":
                src = con.execute(
                    "SELECT * FROM spots WHERE salamander_id = ? AND spot_id = ?",
                    [image_id, op["spot_id"]],
                )
                cols = [d[0] for d in src.description]
                row = src.fetchone()
                if row:
                    data = dict(zip(cols, row))
                    new_spot_id = con.execute(
                        "SELECT coalesce(max(spot_id), 0) + 1 FROM spots WHERE salamander_id = ?",
                        [image_id],
                    ).fetchone()[0]
                    data["spot_id"] = new_spot_id
                    data["area_pixels"] = (data.get("area_pixels") or 0) / 2
                    ph = ", ".join("?" for _ in data)
                    con.execute(
                        f"INSERT INTO spots ({', '.join(data)}) VALUES ({ph})", list(data.values())
                    )
            elif kind == "set_head_tail":
                hx, hy = op["head"]
                tx, ty = op["tail"]
                con.execute(
                    "UPDATE body_axis SET head_x = ?, head_y = ?, tail_tip_x = ?, tail_tip_y = ?, "
                    "source = 'mask_corrected' WHERE salamander_id = ?",
                    [hx, hy, tx, ty, image_id],
                )
            elif kind == "set_axis_ok":
                con.execute(
                    "UPDATE body_axis SET judged_ok = ? WHERE salamander_id = ?",
                    [bool(op["judged_ok"]), image_id],
                )
            else:
                raise ValueError(f"unknown edit op {kind!r}")
    finally:
        con.close()

    recompute = bridge.rebin_and_score(image_id, contours_db_path)
    tier = ladder_tier(
        overall_quality=recompute.overall_quality,
        has_axis=recompute.has_axis,
        n_spots=recompute.n_spots,
        config=ladder,
    )

    correction_id = new_id("corr")
    with db.transaction():
        db.execute(
            "UPDATE images SET n_spots = ?, ladder_tier = ?, q_overall = ?, q_blur = ?, "
            "q_lighting = ?, q_spot = ?, q_body = ?, rev = rev + 1 WHERE image_id = ?",
            [
                recompute.n_spots, tier, recompute.overall_quality,
                recompute.quality.get("blur"), recompute.quality.get("lighting"),
                recompute.quality.get("spot"), recompute.quality.get("body"), image_id,
            ],
        )
        db.insert(
            "extraction_corrections",
            {
                "id": correction_id,
                "image_id": image_id,
                "editor_id": editor_id,
                "diff_json": json.dumps({"ops": ops}, default=str),
                "prev_snapshot_json": json.dumps(before, default=str),
            },
        )
        db.audit(action="edit_extraction", entity="image", entity_id=image_id, actor=editor_id,
                 after={"ops": ops, "n_spots": recompute.n_spots, "tier": tier})
        db.log_activity(kind="correction", summary=f"Extraction corrected for {image_id}",
                        actor=editor_id, ref_type="image", ref_id=image_id)

    return EditResult(correction_id, recompute.n_spots, tier, recompute.overall_quality)


def revert_extraction(
    db: Database, contours_db_path: Path, *, image_id: str, correction_id: str,
    bridge: EditorBridge, editor_id: str | None = None,
) -> EditResult:
    row = db.query_one(
        "SELECT * FROM extraction_corrections WHERE id = ? AND image_id = ?",
        [correction_id, image_id],
    )
    if not row or not row["prev_snapshot_json"]:
        raise ValueError("no snapshot to revert to")
    _restore_snapshot(contours_db_path, image_id, json.loads(row["prev_snapshot_json"]))
    recompute = bridge.rebin_and_score(image_id, contours_db_path)
    tier = ladder_tier(overall_quality=recompute.overall_quality, has_axis=recompute.has_axis,
                       n_spots=recompute.n_spots)
    with db.transaction():
        db.execute(
            "UPDATE images SET n_spots = ?, ladder_tier = ?, q_overall = ?, rev = rev + 1 WHERE image_id = ?",
            [recompute.n_spots, tier, recompute.overall_quality, image_id],
        )
        db.audit(action="revert_extraction", entity="image", entity_id=image_id, actor=editor_id,
                 after={"correction_id": correction_id})
    return EditResult(correction_id, recompute.n_spots, tier, recompute.overall_quality)


def label_correspondence(
    db: Database, *, query_image_id: str, query_spot_id: int, other_image_id: str,
    other_spot_id: int, verdict: str, editor: str | None = None,
) -> str:
    """Record a HUMAN-verified spot pair (spec §10.6, D24)."""
    if verdict not in {"same", "different", "unsure"}:
        raise ValueError(f"bad verdict {verdict!r}")
    cid = new_id("cl")
    db.insert(
        "correspondence_labels",
        {
            "id": cid,
            "query_image_id": query_image_id,
            "query_spot_id": query_spot_id,
            "other_image_id": other_image_id,
            "other_spot_id": other_spot_id,
            "verdict": verdict,
            "editor": editor,
        },
    )
    return cid
