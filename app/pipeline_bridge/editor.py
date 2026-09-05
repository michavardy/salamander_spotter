"""Extraction-editor seam (spec §10.6, M4).

After a human edits spots / spine / head-tail / outline in ``contours.db`` the
body grid must be re-cut and the cheap quality markers recomputed. That is
``pipeline/generate_spot_labels/binning.py`` + ``quality.py`` — no model, no cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import duckdb


@dataclass(frozen=True)
class RecomputeResult:
    n_spots: int
    overall_quality: float | None
    quality: dict[str, float]
    has_axis: bool


class EditorBridge(Protocol):
    def rebin_and_score(self, image_id: str, contours_db_path: Path) -> RecomputeResult:
        """Re-derive bins from the (possibly edited) axis + spots, then recompute
        the ``image_quality`` composites. Writes back into ``contours.db``."""
        ...


class FakeEditorBridge:
    """Recomputes ``n_spots`` from the live rows and nudges quality by ±spot count
    so tests can see a tier/confidence move after an edit."""

    def rebin_and_score(self, image_id: str, contours_db_path: Path) -> RecomputeResult:
        con = duckdb.connect(str(contours_db_path))
        try:
            n = con.execute(
                "SELECT count(*) FROM spots WHERE salamander_id = ?", [image_id]
            ).fetchone()[0]
            axis = con.execute(
                "SELECT judged_ok FROM body_axis WHERE salamander_id = ?", [image_id]
            ).fetchone()
            has_axis = axis is not None
            q = min(1.0, 0.4 + 0.05 * n)
            con.execute(
                "UPDATE image_quality SET n_spots = ?, overall_quality = ?, "
                "spot_extraction_quality = ? WHERE salamander_id = ?",
                [n, q, q, image_id],
            )
            con.execute("UPDATE images SET n_spots = ? WHERE salamander_id = ?", [n, image_id])
        finally:
            con.close()
        return RecomputeResult(
            n_spots=n, overall_quality=q,
            quality={"blur": 0.5, "lighting": 0.6, "spot": q, "body": 0.7},
            has_axis=has_axis,
        )


class PipelineEditorBridge:
    def rebin_and_score(self, image_id: str, contours_db_path: Path) -> RecomputeResult:  # noqa: D102
        raise NotImplementedError(
            "PipelineEditorBridge wiring lands in M4 — calls "
            "pipeline/generate_spot_labels/binning.py + quality.py."
        )
