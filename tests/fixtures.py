"""Helpers that build a tiny on-disk ``datasets/all_sasa_norm_*`` folder — a real
DuckDB ``contours.db`` (the §18.3 schema) + raw image files + ``corrections.json``
— so the transfer path can be exercised end-to-end without the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
from PIL import Image

CONTOURS_SCHEMA = """
CREATE TABLE images (
    salamander_id VARCHAR PRIMARY KEY, width INTEGER, height INTEGER, n_spots INTEGER,
    source_image VARCHAR, purple_image VARCHAR, created_at TIMESTAMP DEFAULT now(),
    body_mask_png BLOB, is_synthetic BOOLEAN DEFAULT FALSE
);
CREATE TABLE spots (
    salamander_id VARCHAR, spot_id INTEGER, global_centroid_x DOUBLE, global_centroid_y DOUBLE,
    area_pixels DOUBLE, local_contour DOUBLE[][], mask_png BLOB, bin INTEGER, axial_bin INTEGER,
    lateral_bin VARCHAR, axis_t DOUBLE, axis_side VARCHAR, axis_offset DOUBLE,
    PRIMARY KEY (salamander_id, spot_id)
);
CREATE TABLE body_axis (
    salamander_id VARCHAR PRIMARY KEY, head_x DOUBLE, head_y DOUBLE, tail_tip_x DOUBLE,
    tail_tip_y DOUBLE, length_px DOUBLE, midline_x DOUBLE[], midline_y DOUBLE[],
    left_x DOUBLE[], left_y DOUBLE[], right_x DOUBLE[], right_y DOUBLE[],
    source VARCHAR, judged_ok BOOLEAN, judge_feedback VARCHAR
);
CREATE TABLE body_bins (
    bin_id VARCHAR PRIMARY KEY, salamander_id VARCHAR, bin INTEGER, quartile INTEGER,
    side VARCHAR, t_lo DOUBLE, t_hi DOUBLE, polygon_x DOUBLE[], polygon_y DOUBLE[]
);
CREATE TABLE image_quality (
    salamander_id VARCHAR PRIMARY KEY, n_spots INTEGER, judged_ok BOOLEAN,
    blur_quality DOUBLE, lighting_quality DOUBLE, spot_extraction_quality DOUBLE,
    body_extraction_quality DOUBLE, overall_quality DOUBLE
);
"""


@dataclass
class FixtureImage:
    image_id: str
    n_spots: int = 6
    overall_quality: float = 0.72
    has_axis: bool = True
    is_synthetic: bool = False
    ext: str = ".jpg"
    width: int = 200
    height: int = 260


DEFAULT_IMAGES: list[FixtureImage] = [
    FixtureImage("aa_1_1", n_spots=8, overall_quality=0.80),                 # auto_accept
    FixtureImage("aa_1_2", n_spots=6, overall_quality=0.55),                 # needs_a_look
    FixtureImage("aa_1_g0", n_spots=6, overall_quality=0.30, is_synthetic=True, ext=".png"),
    FixtureImage("bb_1_1", n_spots=2, overall_quality=0.28),                 # hand_correction
    FixtureImage("cc_1_1", n_spots=7, overall_quality=0.70),
    FixtureImage("cc_2_1", n_spots=5, overall_quality=0.66),                 # merged away
    FixtureImage("dd_1_1", n_spots=4, overall_quality=0.61),                 # excluded
]

DEFAULT_CORRECTIONS = {
    "dataset": "fixture",
    "merge": [{"keep": "cc_1", "alias": "cc_2", "source": "test", "note": "same animal"}],
    "exclude": [{"sid": "dd_1_1", "reason": "extraction", "source": "test", "note": "bad extraction"}],
}


def _write_image(path: Path, w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (90, 120, 60)).save(path)


def build_fixture_dataset(
    dest: Path,
    *,
    name: str = "all_sasa_norm_2099_01_01",
    images: list[FixtureImage] | None = None,
    corrections: dict | None = DEFAULT_CORRECTIONS,
) -> Path:
    """Create ``dest/<name>/`` with ``raw/``, ``db/contours.db`` and ``corrections.json``.
    Returns the dataset path."""
    images = images if images is not None else DEFAULT_IMAGES
    ds = dest / name
    raw = ds / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (ds / "db").mkdir(parents=True, exist_ok=True)

    for im in images:
        _write_image(raw / f"{im.image_id}{im.ext}", im.width, im.height)

    con = duckdb.connect(str(ds / "db" / "contours.db"))
    try:
        con.execute(CONTOURS_SCHEMA)
        for im in images:
            con.execute(
                "INSERT INTO images (salamander_id, width, height, n_spots, is_synthetic) "
                "VALUES (?, ?, ?, ?, ?)",
                [im.image_id, im.width, im.height, im.n_spots, im.is_synthetic],
            )
            con.execute(
                "INSERT INTO image_quality (salamander_id, n_spots, judged_ok, blur_quality, "
                "lighting_quality, spot_extraction_quality, body_extraction_quality, overall_quality) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [im.image_id, im.n_spots, im.has_axis, 0.5, 0.6, 0.7, 0.8, im.overall_quality],
            )
            for spot_id in range(1, im.n_spots + 1):
                con.execute(
                    "INSERT INTO spots (salamander_id, spot_id, global_centroid_x, "
                    "global_centroid_y, area_pixels, axial_bin, lateral_bin) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [im.image_id, spot_id, float(spot_id), float(spot_id), 50.0, 1, "left"],
                )
            if im.has_axis:
                con.execute(
                    "INSERT INTO body_axis (salamander_id, head_x, head_y, tail_tip_x, tail_tip_y, "
                    "length_px, source, judged_ok) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [im.image_id, 0.0, 0.0, 10.0, 10.0, 14.0, "mask", True],
                )
    finally:
        con.close()

    if corrections is not None:
        (ds / "corrections.json").write_text(json.dumps(corrections, indent=2), encoding="utf-8")
    return ds
