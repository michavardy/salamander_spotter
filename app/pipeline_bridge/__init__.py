"""Thin adapters onto ``pipeline/*`` — no logic, just wiring (spec §7).

The application never re-implements ML. For the ingest slice the only pipeline
touch-point is *per-photo extraction*: given one raw image, write its rows into
``contours.db`` and report what came out.

``ExtractionBridge`` is the seam. Tests use :class:`FakeExtractionBridge`;
:class:`PipelineExtractionBridge` (spec M2) will call
``pipeline/generate_spot_labels``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import duckdb


@dataclass(frozen=True)
class ExtractionResult:
    """What one photo's extraction produced (spec §7.1)."""

    image_id: str
    n_spots: int
    has_axis: bool
    overall_quality: float | None
    quality: dict[str, float] = field(default_factory=dict)  # blur/lighting/spot/body composites
    width: int | None = None
    height: int | None = None
    is_synthetic: bool = False
    failed: bool = False
    reason: str | None = None  # e.g. "multi-animal frame" (auto-disqualify, §7.1)


@runtime_checkable
class ExtractionBridge(Protocol):
    def extract(self, image_id: str, raw_path: Path, contours_db_path: Path) -> ExtractionResult:
        """Extract one photo, writing rows into ``contours_db_path``."""
        ...


# --------------------------------------------------------------------------- #
#  Fake bridge for tests                                                       #
# --------------------------------------------------------------------------- #
class FakeExtractionBridge:
    """Deterministic stand-in: writes minimal ``images`` / ``image_quality`` /
    ``spots`` / ``body_axis`` rows so the ingest path can be exercised without
    the LLM pipeline. Configure per-image outcomes via ``outcomes``."""

    def __init__(self, outcomes: dict[str, ExtractionResult] | None = None):
        self.outcomes = outcomes or {}
        self.calls: list[str] = []

    def result_for(self, image_id: str) -> ExtractionResult:
        if image_id in self.outcomes:
            return self.outcomes[image_id]
        return ExtractionResult(
            image_id=image_id,
            n_spots=5,
            has_axis=True,
            overall_quality=0.72,
            quality={"blur": 0.6, "lighting": 0.7, "spot": 0.75, "body": 0.8},
            width=800,
            height=1000,
        )

    def extract(self, image_id: str, raw_path: Path, contours_db_path: Path) -> ExtractionResult:
        self.calls.append(image_id)
        res = self.result_for(image_id)
        _ensure_contours_schema(contours_db_path)
        con = duckdb.connect(str(contours_db_path))
        try:
            con.execute("DELETE FROM images WHERE salamander_id = ?", [image_id])
            con.execute("DELETE FROM image_quality WHERE salamander_id = ?", [image_id])
            con.execute("DELETE FROM spots WHERE salamander_id = ?", [image_id])
            con.execute("DELETE FROM body_axis WHERE salamander_id = ?", [image_id])
            if res.failed:
                return res
            con.execute(
                "INSERT INTO images (salamander_id, width, height, n_spots, is_synthetic) "
                "VALUES (?, ?, ?, ?, ?)",
                [image_id, res.width, res.height, res.n_spots, res.is_synthetic],
            )
            q = res.quality
            con.execute(
                "INSERT INTO image_quality (salamander_id, n_spots, judged_ok, blur_quality, "
                "lighting_quality, spot_extraction_quality, body_extraction_quality, overall_quality) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    image_id,
                    res.n_spots,
                    res.has_axis,
                    q.get("blur"),
                    q.get("lighting"),
                    q.get("spot"),
                    q.get("body"),
                    res.overall_quality,
                ],
            )
            for spot_id in range(1, res.n_spots + 1):
                con.execute(
                    "INSERT INTO spots (salamander_id, spot_id, global_centroid_x, "
                    "global_centroid_y, area_pixels, axial_bin, lateral_bin) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [image_id, spot_id, 10.0 * spot_id, 20.0 * spot_id, 100.0, 1, "left"],
                )
            if res.has_axis:
                con.execute(
                    "INSERT INTO body_axis (salamander_id, head_x, head_y, tail_tip_x, "
                    "tail_tip_y, length_px, source, judged_ok) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [image_id, 0.0, 0.0, 100.0, 100.0, 141.4, "mask", True],
                )
        finally:
            con.close()
        return res


def _ensure_contours_schema(path: Path) -> None:
    """Create the subset of the ``contours.db`` schema (§18.3) the app reads."""
    con = duckdb.connect(str(path))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                salamander_id VARCHAR PRIMARY KEY, width INTEGER, height INTEGER,
                n_spots INTEGER, source_image VARCHAR, purple_image VARCHAR,
                created_at TIMESTAMP DEFAULT now(), body_mask_png BLOB,
                is_synthetic BOOLEAN DEFAULT FALSE
            );
            CREATE TABLE IF NOT EXISTS spots (
                salamander_id VARCHAR, spot_id INTEGER, global_centroid_x DOUBLE,
                global_centroid_y DOUBLE, area_pixels DOUBLE, local_contour DOUBLE[][],
                mask_png BLOB, bin INTEGER, axial_bin INTEGER, lateral_bin VARCHAR,
                axis_t DOUBLE, axis_side VARCHAR, axis_offset DOUBLE,
                PRIMARY KEY (salamander_id, spot_id)
            );
            CREATE TABLE IF NOT EXISTS body_axis (
                salamander_id VARCHAR PRIMARY KEY, head_x DOUBLE, head_y DOUBLE,
                tail_tip_x DOUBLE, tail_tip_y DOUBLE, length_px DOUBLE,
                midline_x DOUBLE[], midline_y DOUBLE[], left_x DOUBLE[], left_y DOUBLE[],
                right_x DOUBLE[], right_y DOUBLE[], source VARCHAR, judged_ok BOOLEAN,
                judge_feedback VARCHAR
            );
            CREATE TABLE IF NOT EXISTS image_quality (
                salamander_id VARCHAR PRIMARY KEY, n_spots INTEGER, judged_ok BOOLEAN,
                blur_quality DOUBLE, lighting_quality DOUBLE, spot_extraction_quality DOUBLE,
                body_extraction_quality DOUBLE, overall_quality DOUBLE
            );
            """
        )
    finally:
        con.close()


class PipelineExtractionBridge:
    """Real extraction via ``pipeline/generate_spot_labels`` — spec M2.

    Deferred: wiring the resumable multi-stage LLM pipeline (animal screen ->
    magenta repaint -> contours -> body mask/axis -> binning -> quality) is the
    body of milestone M2. This placeholder documents the seam.
    """

    def extract(self, image_id: str, raw_path: Path, contours_db_path: Path) -> ExtractionResult:  # noqa: D102
        raise NotImplementedError(
            "PipelineExtractionBridge is implemented in milestone M2 "
            "(see docs/salamander_spotter_spec.md §7.1, §17)."
        )
