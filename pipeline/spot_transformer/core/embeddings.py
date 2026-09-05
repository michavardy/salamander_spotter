from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

dataset_name = "all_sasa_norm_2026_23_07"
dataset_path = f"datasets/{dataset_name}"
REPO_ROOT = Path(__file__).resolve().parents[3]   # core/ -> spot_transformer/ -> pipeline/ -> repo root
DB_PATH = REPO_ROOT / dataset_path / "db" / "contours.db"
SHAPE_HARMONICS = 10
POS_BANDS = 6

def _to_xy(seq) -> np.ndarray:
    """Clean a DuckDB ``DOUBLE[][]`` (jagged object array of [x, y]) into ``(k, 2)`` float."""
    if seq is None or np.ndim(seq) == 0 or len(seq) == 0:  # None / NA scalar / empty
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray([np.asarray(p, dtype=np.float64) for p in seq], dtype=np.float64)

def _pair_xy(x, y) -> np.ndarray:
    """Fold DuckDB paired ``DOUBLE[]`` columns into ``(m, 2)`` float, NA/None-safe."""
    if x is None or np.ndim(x) == 0 or len(x) == 0:  # unmatched LEFT JOIN -> pd.NA scalar
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack([x, y]).astype(np.float64)

def get_spots(
    db_path: Path | str = DB_PATH,
    *,
    decode_masks: bool = False,
    include_body_outline: bool = False,
) -> pd.DataFrame:
    """Every spot in ``contours.db`` as one wide DataFrame — one row per spot, carrying
    its shape, its location, and its photo's quality + body-axis context.

    Ordered by ``(salamander_id, spot_id)`` so a photo's spots are contiguous;
    ``df.groupby("salamander_id")`` yields per-image token sequences.

    Column groups
    -------------
    Shape (per spot)
        ``area_pixels``; ``local_contour`` (k, 2) float, centroid-relative as stored;
        ``contour_xy`` (k, 2) float, absolute image px; ``mask_png`` (raw PNG bytes,
        full-frame). With ``decode_masks=True`` an extra ``mask`` column holds the
        decoded H×W uint8 array (0/255) — heavy (~GBs over the whole set), hence opt-in.
    Location (per spot)
        ``global_centroid_x/y``; ``bin`` (1..8, NaN if the spot straddles the midline),
        ``axial_bin`` (1..4), ``lateral_bin`` ('left'|'right'|'overlap'); position along
        the body line: ``axis_t`` (0=head..1=tail), ``axis_side``, ``axis_offset`` (signed
        px from the midline).
    The body line (per image, joined onto every spot)
        ``head_x/y``, ``tail_tip_x/y``, ``length_px``, ``midline_xy`` (m, 2) float,
        ``axis_line_source``, ``axis_judged_ok``. With ``include_body_outline=True`` also
        ``left_xy`` / ``right_xy`` — the drawn body outlines (~1024 pts each, duplicated
        per spot, so opt-in).
    Quality (per image — there is no per-spot quality; the spot inherits its photo's)
        every ``image_quality`` column: ``blur_quality``, ``lighting_quality``,
        ``spot_extraction_quality``, ``body_extraction_quality``, ``overall_quality``,
        plus the raw measures (``blur_score``, ``pattern_contrast``, ``solidity``, ...).
    Image
        ``width``, ``height``, ``is_synthetic``.

    The two heavy raw columns dropped from the eager path are ``local_contour``'s DB blob
    twin ``mask_png`` (kept as bytes) — everything the spot's shape needs is in
    ``contour_xy``; ``mask_png`` is there for rasterized use.
    """
    outline_cols = (
        ", ax.left_x, ax.left_y, ax.right_x, ax.right_y" if include_body_outline else ""
    )
    sql = f"""
        SELECT
            s.*,
            iq.* EXCLUDE (salamander_id),
            img.width, img.height, img.is_synthetic,
            ax.head_x, ax.head_y, ax.tail_tip_x, ax.tail_tip_y, ax.length_px,
            ax.midline_x, ax.midline_y{outline_cols},
            ax.source    AS axis_line_source,
            ax.judged_ok AS axis_judged_ok
        FROM spots s
        LEFT JOIN image_quality iq USING (salamander_id)
        LEFT JOIN images        img USING (salamander_id)
        LEFT JOIN body_axis     ax  USING (salamander_id)
        ORDER BY s.salamander_id, s.spot_id
    """
    con = duckdb.connect(database=str(db_path), read_only=True)
    try:
        df = con.execute(sql).df()
    finally:
        con.close()

    # Shape: clean the jagged contour, add absolute-coord contour.
    df["local_contour"] = df["local_contour"].map(_to_xy)
    centroid = df[["global_centroid_x", "global_centroid_y"]].to_numpy()
    df["contour_xy"] = [c + o for c, o in zip(df["local_contour"], centroid)]

    # The line: fold the paired x/y arrays into (m, 2) point sequences.
    df["midline_xy"] = [_pair_xy(x, y) for x, y in zip(df.pop("midline_x"), df.pop("midline_y"))]
    if include_body_outline:
        for side in ("left", "right"):
            xs, ys = df.pop(f"{side}_x"), df.pop(f"{side}_y")
            df[f"{side}_xy"] = [_pair_xy(x, y) for x, y in zip(xs, ys)]

    if decode_masks:
        import cv2  # lazy: only when asked to rasterize

        def _decode(b):
            if b is None:
                return None
            return cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_GRAYSCALE)

        df["mask"] = df["mask_png"].map(_decode)

    return df

def visualize_spot(spots: pd.DataFrame, row=0, save_path: Path | str | None = None):
    """Plot one spot four ways: on the whole body (``midline_xy`` + ``contour_xy``), the
    decoded ``mask_png`` cropped with ``contour_xy`` overlaid, and the raw ``local_contour``.

    ``row`` is either a positional index (``0``) or a ``(salamander_id, spot_id)`` tuple.
    Shows an interactive window; pass ``save_path`` to write a PNG instead.
    """
    import cv2
    import matplotlib.pyplot as plt

    if isinstance(row, tuple):
        sid, spot_id = row
        r = spots[(spots.salamander_id == sid) & (spots.spot_id == spot_id)].iloc[0]
    else:
        r = spots.iloc[row]

    contour = np.vstack([r.contour_xy, r.contour_xy[:1]])          # close the polygon
    local = np.vstack([r.local_contour, r.local_contour[:1]])
    mask = cv2.imdecode(np.frombuffer(bytes(r.mask_png), np.uint8), cv2.IMREAD_GRAYSCALE)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(f"{r.salamander_id} · spot {r.spot_id}  "
                 f"(bin={r['bin']}, axis_t={r.axis_t:.2f}, area={r.area_pixels:.0f}px²)")

    # 1) the spot in body context: midline (the line) + spot outline + centroid
    ax1.set_title("on the body: midline_xy + contour_xy")
    if len(r.midline_xy):
        ax1.plot(r.midline_xy[:, 0], r.midline_xy[:, 1], "-", color="tab:blue", lw=1.5, label="midline")
        ax1.scatter(*r.midline_xy[0], c="green", s=40, zorder=5, label="head")
        ax1.scatter(*r.midline_xy[-1], c="red", s=40, zorder=5, label="tail")
    ax1.fill(contour[:, 0], contour[:, 1], color="tab:orange", alpha=0.7)
    ax1.scatter(r.global_centroid_x, r.global_centroid_y, c="black", s=15, zorder=6, label="centroid")
    ax1.set_xlim(0, r.width); ax1.set_ylim(r.height, 0)           # image orientation (y down)
    ax1.set_aspect("equal"); ax1.legend(loc="upper right", fontsize=8)

    # 2) mask_png (decoded) cropped to the spot, with contour_xy overlaid to confirm they agree
    ax2.set_title("mask_png (cropped) + contour_xy")
    ys, xs = np.nonzero(mask)
    m = 12
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, mask.shape[1])
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, mask.shape[0])
    ax2.imshow(mask[y0:y1, x0:x1], cmap="gray", extent=(x0, x1, y1, y0))
    ax2.plot(contour[:, 0], contour[:, 1], "-", color="tab:orange", lw=1.5)
    ax2.set_aspect("equal")

    # 3) local_contour: the raw centroid-relative shape (0,0 = centroid)
    ax3.set_title("local_contour (centroid-relative)")
    ax3.fill(local[:, 0], local[:, 1], color="tab:orange", alpha=0.7)
    ax3.scatter(0, 0, c="black", s=15, label="centroid")
    ax3.axhline(0, color="gray", lw=0.5); ax3.axvline(0, color="gray", lw=0.5)
    ax3.set_aspect("equal"); ax3.invert_yaxis(); ax3.legend(fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info(f"wrote {save_path}")
    else:
        plt.show()
    plt.close(fig)

def _efd(contour: np.ndarray, harmonics: int = 10) -> np.ndarray:
    """Normalized Elliptic Fourier Descriptor of a closed contour (Kuhl & Giardina, 1982).

    Invariant to translation (DC term dropped), scale, rotation, and contour start point.
    High-frequency boundary jaggedness lives in the top harmonics, so keeping only the
    first ``harmonics`` low-pass-filters pixelation ("±squares") noise.

    Returns a flat ``harmonics*4 - 3`` vector (the first harmonic normalizes to the
    constants a1=1, b1=0, c1=0, which are dropped; d1 = fundamental-ellipse eccentricity
    is kept). Degenerate contours return zeros.
    """
    dim = harmonics * 4 - 3
    if contour is None or len(contour) < 3:
        return np.zeros(dim)

    pts = np.vstack([contour, contour[:1]]).astype(np.float64)  # close the loop
    d = np.diff(pts, axis=0)
    dt = np.hypot(d[:, 0], d[:, 1])
    keep = dt > 1e-9                                            # drop duplicate points
    if keep.sum() < 3:
        return np.zeros(dim)
    d, dt = d[keep], dt[keep]
    t = np.concatenate([[0.0], np.cumsum(dt)])
    T = t[-1]
    phi = 2.0 * np.pi * t / T                                   # phase at each vertex

    coeffs = np.zeros((harmonics, 4))
    for n in range(1, harmonics + 1):
        dcos = np.diff(np.cos(n * phi))
        dsin = np.diff(np.sin(n * phi))
        f = T / (2.0 * n * n * np.pi * np.pi)
        coeffs[n - 1] = [
            f * np.sum(d[:, 0] / dt * dcos),   # a_n
            f * np.sum(d[:, 0] / dt * dsin),   # b_n
            f * np.sum(d[:, 1] / dt * dcos),   # c_n
            f * np.sum(d[:, 1] / dt * dsin),   # d_n
        ]

    # Start-point normalization: rotate the parameterization so harmonic 1 starts canonically.
    a1, b1, c1, d1 = coeffs[0]
    theta1 = 0.5 * np.arctan2(2 * (a1 * b1 + c1 * d1),
                              a1 ** 2 - b1 ** 2 + c1 ** 2 - d1 ** 2)
    for n in range(harmonics):
        arg = (n + 1) * theta1
        rot = np.array([[np.cos(arg), -np.sin(arg)], [np.sin(arg), np.cos(arg)]])
        coeffs[n] = (coeffs[n].reshape(2, 2) @ rot).ravel()
    # Rotation normalization: rotate the shape so harmonic-1 major axis is horizontal.
    psi1 = np.arctan2(coeffs[0, 2], coeffs[0, 0])
    rpsi = np.array([[np.cos(psi1), np.sin(psi1)], [-np.sin(psi1), np.cos(psi1)]])
    for n in range(harmonics):
        coeffs[n] = (rpsi @ coeffs[n].reshape(2, 2)).ravel()
    # Scale normalization: divide by the fundamental ellipse's major-axis magnitude.
    e = abs(coeffs[0, 0])
    if e > 1e-12:
        coeffs /= e

    return coeffs.ravel()[3:]   # drop the now-constant a1, b1, c1

def _sinusoidal(z: np.ndarray, bands: int, base: float = np.pi) -> np.ndarray:
    """NeRF/Transformer-style Fourier features of a scalar coordinate: (n,) -> (n, 2*bands).

    Distance in feature space grows monotonically with |Δz|, so positions far apart on the
    body (e.g. head vs tail) map far apart in embedding space.
    """
    freqs = base * (2.0 ** np.arange(bands))          # octaves
    ang = np.outer(np.asarray(z, dtype=np.float64), freqs)
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)

def _augment(block: np.ndarray, c: float) -> np.ndarray:
    """Unit-norm rows + a constant column ``c``, renormalized: kernel -> ``(cos + c²)/(1 + c²)``.

    Lifting the kernel above zero is what makes the outer product safe to take -- otherwise two
    spots that DISagree on a block (negative cosine) would multiply to a positive score.
    """
    n = len(block)
    out = np.concatenate([block, np.full((n, 1), float(c))], axis=1)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def _tensor_combine(shape: np.ndarray, pos: np.ndarray, c_shape: float, c_pos: float) -> np.ndarray:
    """``vec(shape_hat ⊗ pos_hat)`` per spot -> (N, 38*26). See :func:`get_spot_embeddings`.

    The point of this construction: the dot product of two such vectors equals
    ``k_shape * k_pos``, so shape and position must BOTH agree for a match to score.
    """
    s_hat, p_hat = _augment(shape, c_shape), _augment(pos, c_pos)
    return np.einsum("ni,nj->nij", s_hat, p_hat).reshape(len(s_hat), -1)


def _morph_block(spots: pd.DataFrame, bands: int) -> np.ndarray:
    """The **morphology** block: the properties a human demonstrably uses that the EFD discards.

    ``distinctiveness.py`` fitted the human interesting-spot clicks and found the two dominant
    factors to be **irregularity (+0.824)** and **size (+0.707)** — and neither survives into the
    shape block:

    * ``_efd`` normalizes every spot to unit scale, so **size is absent by construction**: a big
      blotch and a small dot with the same outline are the same 37 numbers. The reviewer's own
      notes argue in size percentiles ("spot 96", "spot 100"), about a quantity the matcher cannot
      represent.
    * only the first ``SHAPE_HARMONICS`` harmonics are kept, deliberately, to low-pass pixelation
      noise — but lobes and notches (what ``irregularity`` measures) live in the same band. The
      noise filter and the signal filter are the same filter.

    So this block re-admits four scalars — size percentile, irregularity, elongation,
    noncircularity — as Fourier features, on the same footing as the position block. It is the
    cheapest available test of whether ``gate_calibration``'s null (embedding cosine at chance on
    human accept/reject verdicts, AUROC 0.484) is caused by exactly this omission.
    """
    try:
        from strict_match import shape_descriptors                          # noqa: PLC0415
    except ImportError:                                     # package path (no core/ on sys.path)
        from pipeline.spot_transformer.core.strict_match import shape_descriptors  # noqa: PLC0415

    n = len(spots)
    pct = _size_percentile(spots) / 100.0                   # 0..1, NaN where no body length
    pct = np.nan_to_num(pct, nan=float(np.nanmedian(pct)) if np.isfinite(pct).any() else 0.5)

    desc = np.array([shape_descriptors(c) for c in spots["local_contour"]])  # (n, 3)
    elong, noncirc, irreg = desc[:, 0], desc[:, 1], desc[:, 2]

    # Rank-normalize the shape descriptors before encoding: they are heavy-tailed (noncircularity
    # is unbounded above), and a raw value would put almost every spot in one Fourier bin.
    def _rank01(x):
        x = np.nan_to_num(np.asarray(x, float))
        order = x.argsort(); r = np.empty(len(x)); r[order] = np.arange(len(x))
        return r / max(len(x) - 1, 1)

    cols = [pct, _rank01(irreg), _rank01(elong), _rank01(noncirc)]
    block = np.concatenate([_sinusoidal(c, bands) for c in cols], axis=1)
    return block / (np.linalg.norm(block, axis=1, keepdims=True) + 1e-8)


def get_spot_embeddings(
    spots: pd.DataFrame,
    *,
    harmonics: int = SHAPE_HARMONICS,
    pos_bands: int = POS_BANDS,
    alpha: float = 1.0,          # weight on the shape block   (combine="concat")
    beta: float = 1.0,           # weight on the position block (combine="concat")
    gamma: float = 1.0,          # weight on the morphology block (combine="concat", morph=True)
    morph: bool = False,         # append the size/irregularity block -- see _morph_block
    combine: str = "concat",     # "concat" (additive) | "tensor" (conjunctive)
    c_shape: float = 1.0,        # veto strength of the shape block   (combine="tensor")
    c_pos: float = 1.0,          # veto strength of the position block (combine="tensor")
) -> dict[str, np.ndarray]:
    """Embed every spot as shape x position, keyed by ``"{salamander_id}#{spot_id}"``.

    Shape block (``_efd``, 37-dim): translation/scale/rotation/start-point-invariant,
    pixelation-tolerant — a bow-tie matches a rotated, resized, jagged bow-tie.

    Position block (``_sinusoidal``, 25-dim): body-intrinsic, so identical shapes at different
    body locations land far apart. Built from ``axis_t`` (0=head..1=tail), the midline-offset
    normalized by half the body width (``axis_offset / (avg_width_px/2)``, scale-invariant),
    and a left/right ``axis_side`` sign.

    ``combine`` decides how the two blocks meet, and it is the whole ballgame for what a
    "match" means:

    ``"concat"`` (62-dim, the original)
        ``[alpha*shape | beta*pos]``. Each block is L2-normalized, so the cosine between two
        spots is ``(cos_shape + cos_pos)/2`` -- an ADDITIVE, compensatory rule. A perfect shape
        match in the wrong place (1.0, 0.0) scores the same 0.5 as two mediocre halves
        (0.5, 0.5): neither block can veto the other. Squared distance decomposes as
        ``alpha²·d²_shape + beta²·d²_pos``, so alpha/beta only re-mix, never gate.

    ``"tensor"`` (988-dim, conjunctive)
        The outer product of the two blocks, each first augmented with a constant and
        renormalized. Because ``<a1⊗b1, a2⊗b2> = <a1,a2>·<b1,b2>``, the dot product of two
        such embeddings is the PRODUCT of the block similarities -- so a spot only matches when
        shape AND position agree, and a penalty in either drags the whole score down. Still one
        vector compared with one plain dot product, so every downstream consumer is unchanged.

        The constant does two jobs. It lifts each block's kernel toward
        ``(cos + c²)/(1 + c²)``, and it tunes veto strength: ``c=0`` is the raw cosine (maximum
        veto), ``c=1`` maps to ``(cos+1)/2``, and large ``c`` pushes the kernel toward 1 so that
        block stops vetoing. ``‖a⊗b‖ = ‖a‖·‖b‖``, so the result is already unit-norm and
        cosine == dot.

        KNOWN LIMITATION: a bilinear form cannot implement a ReLU, so at small ``c`` two spots
        that DISagree on both blocks (both cosines negative) multiply to a spuriously POSITIVE
        score -- at ``c=0``, (-0.5, -0.5) scores +0.25, above a perfect shape in the wrong place
        (-0.5). Raising ``c`` suppresses this (0.04 at c=0.5, 0.06 at c=1) at the cost of veto
        strength, which is the trade-off ``--c-shape``/``--c-pos`` exist to explore. It matters
        less than it looks here because position cosines cluster near 0.5 in practice, but it is
        the reason not to default ``c`` to 0.
    """
    n = len(spots)

    # --- shape: normalized EFD per spot, z-scored across the dataset, then unit-normed ---
    shape = np.vstack([_efd(c, harmonics) for c in spots["local_contour"]])
    shape = np.nan_to_num(shape)
    shape = (shape - shape.mean(0)) / (shape.std(0) + 1e-8)
    shape /= np.linalg.norm(shape, axis=1, keepdims=True) + 1e-8

    # --- position: body-intrinsic (t, u, side) -> Fourier features, then unit-normed ---
    t = np.nan_to_num(spots["axis_t"].to_numpy(np.float64), nan=0.5)
    half_w = spots["avg_width_px"].to_numpy(np.float64) / 2.0
    offset = spots["axis_offset"].to_numpy(np.float64)
    u = np.divide(offset, half_w, out=np.zeros(n), where=half_w > 0)
    u = np.clip(np.nan_to_num(u), -3.0, 3.0)
    side = np.where(spots["axis_side"].to_numpy() == "left", 1.0, -1.0).reshape(-1, 1)
    pos = np.concatenate([_sinusoidal(t, pos_bands), _sinusoidal(u, pos_bands), side], axis=1)
    pos /= np.linalg.norm(pos, axis=1, keepdims=True) + 1e-8

    if combine == "concat":
        blocks = [alpha * shape, beta * pos]
        if morph:
            # A third unit-normed block makes the cosine (cos_shape + cos_pos + cos_morph)/3, so
            # every block's influence drops from 1/2 to 1/3. That dilution is the price of the
            # extra evidence and is exactly what the sweep is measuring; gamma is the dial if the
            # trade needs tuning.
            blocks.append(gamma * _morph_block(spots, pos_bands))
        emb = np.concatenate(blocks, axis=1)
    elif combine == "tensor":
        if morph:
            raise ValueError("morph=True is only implemented for combine='concat' -- the tensor "
                             "flow is an outer product of exactly two blocks")
        emb = _tensor_combine(shape, pos, c_shape, c_pos)
    else:
        raise ValueError(f"combine must be 'concat' or 'tensor', got {combine!r}")

    ids = spots["salamander_id"].to_numpy()
    sids = spots["spot_id"].to_numpy()
    return {f"{i}#{s}": e for i, s, e in zip(ids, sids, emb)}


def _ensure_embeddings(spots: pd.DataFrame, **emb_kwargs) -> pd.DataFrame:
    """Return ``spots`` guaranteed to carry an ``embedding`` column."""
    if "embedding" in spots.columns:
        return spots
    emb = get_spot_embeddings(spots, **emb_kwargs)
    spots = spots.copy()
    spots["embedding"] = [emb[f"{s}#{i}"] for s, i in zip(spots.salamander_id, spots.spot_id)]
    return spots


def _fetch_outline(db_path: Path | str, sid: str) -> tuple[np.ndarray, np.ndarray]:
    """Fetch a salamander's drawn body outline (left, right halves) from ``body_axis``."""
    con = duckdb.connect(database=str(db_path), read_only=True)
    try:
        row = con.execute(
            "SELECT left_x, left_y, right_x, right_y FROM body_axis WHERE salamander_id = ?",
            [sid],
        ).fetchone()
    finally:
        con.close()
    if row is None:
        return np.empty((0, 2)), np.empty((0, 2))
    lx, ly, rx, ry = row
    return _pair_xy(lx, ly), _pair_xy(rx, ry)


def match_spots(
    spots: pd.DataFrame, id1: str, id2: str, *, method: str = "mutual",
    emb_col: str = "embedding", **emb_kwargs
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[tuple[int, int, float]]]:
    """Cosine-similarity match between the two salamanders' spot embeddings.

    Returns ``(s1, s2, S, pairs)``: the two per-salamander spot sub-frames (row order
    aligned to ``S``), the (n1, n2) cosine-similarity matrix, and ``pairs`` — a list of
    ``(i, j, sim)`` local-index matches (pre-cutoff).

    ``method``: ``"mutual"`` (reciprocal nearest neighbours — clean, few lines),
    ``"hungarian"`` (one-to-one optimal assignment), or ``"threshold"`` (all pairs).

    ``emb_col`` picks which per-spot vector the arcs are computed from. Default ``"embedding"``
    (the 62-d hand-engineered representation, built on demand). Pass a column already present on
    ``spots`` — e.g. a learned SSL per-spot embedding — to draw the correspondences ANOTHER
    representation would make. Any per-spot method can be visualised this way; only image-level
    aggregators (logreg, deepsets) cannot, because they never emit spot-to-spot pairs.
    """
    if emb_col == "embedding":
        spots = _ensure_embeddings(spots, **emb_kwargs)
    elif emb_col not in spots.columns:
        raise ValueError(f"emb_col {emb_col!r} not in spots columns {list(spots.columns)}")
    s1 = spots[spots.salamander_id == id1].reset_index(drop=True)
    s2 = spots[spots.salamander_id == id2].reset_index(drop=True)
    if len(s1) == 0 or len(s2) == 0:
        raise ValueError(f"no spots for {id1!r} ({len(s1)}) or {id2!r} ({len(s2)})")

    e1 = np.vstack(s1[emb_col]); e2 = np.vstack(s2[emb_col])
    e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12)
    e2 = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12)
    S = e1 @ e2.T                                             # cosine similarity in [-1, 1]

    if method == "threshold":
        pairs = [(i, j, float(S[i, j])) for i in range(len(s1)) for j in range(len(s2))]
    elif method == "hungarian":
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(-S)
        pairs = [(int(i), int(j), float(S[i, j])) for i, j in zip(ri, ci)]
    else:  # mutual nearest neighbour
        b2 = S.argmax(1); b1 = S.argmax(0)
        pairs = [(i, int(b2[i]), float(S[i, b2[i]])) for i in range(len(s1)) if b1[b2[i]] == i]
    return s1, s2, S, pairs


def naive_match_all(
    spots: pd.DataFrame,
    *,
    cutoff: float = 0.75,
    include_same_salamander: bool = False,
    chunk: int = 4000,
    **emb_kwargs,
) -> pd.DataFrame:
    """Brute-force compare every spot against every other spot; keep pairs above ``cutoff``.

    Returns one row per surviving pair::

        salamander_source_id, salamander_target_id, spot_source_id, spot_target_id, similarity

    Similarity is cosine similarity of the spot embeddings (same metric as
    :func:`match_spots`). Each unordered pair appears once (source index < target index);
    self-comparisons are excluded, and same-salamander pairs are excluded unless
    ``include_same_salamander=True``. Computed in row-blocks of ``chunk`` so the full
    N×N matrix (≈2.7 GB at 26k spots) is never materialised.

    Note: this embedding's cosine scale is low (~0.3–0.5 even for the same individual), so
    a 0.75 cutoff typically returns very few rows — lower it to match the metric.
    """
    spots = _ensure_embeddings(spots, **emb_kwargs)
    e = np.vstack(spots["embedding"]).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
    sal = spots["salamander_id"].to_numpy()
    spot = spots["spot_id"].to_numpy()
    n = len(e)

    si, ti, sims_out = [], [], []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = e[start:end] @ e.T                    # (b, n) cosine sims
        for local, i in enumerate(range(start, end)):
            sims = block[local]
            j = np.nonzero(sims[i + 1:] >= cutoff)[0] + (i + 1)   # upper triangle only
            if not include_same_salamander:
                j = j[sal[j] != sal[i]]
            if len(j):
                si.append(np.full(len(j), i))
                ti.append(j)
                sims_out.append(sims[j])

    if not si:
        return pd.DataFrame(columns=["salamander_source_id", "salamander_target_id",
                                     "spot_source_id", "spot_target_id", "similarity"])
    si = np.concatenate(si); ti = np.concatenate(ti); sims_out = np.concatenate(sims_out)
    out = pd.DataFrame({
        "salamander_source_id": sal[si],
        "salamander_target_id": sal[ti],
        "spot_source_id": spot[si],
        "spot_target_id": spot[ti],
        "similarity": sims_out.astype(float),
    })
    return out.sort_values("similarity", ascending=False, ignore_index=True)


def write_spot_embeddings(
    spots: pd.DataFrame,
    db_path: Path | str = DB_PATH,
    *,
    table: str = "spot_embeddings",
    **emb_kwargs,
) -> None:
    """Persist every spot's embedding to ``table`` (replacing it if it exists).

    Columns: ``salamander_id`` (VARCHAR), ``spot_id`` (INTEGER), ``embedding`` (``DOUBLE[]``)
    — keyed the same as ``spots`` so it joins straight back on. This is what
    :func:`transformer.get_spot_embeddings` reads.
    """
    spots = _ensure_embeddings(spots, **emb_kwargs)
    df = pd.DataFrame({
        "salamander_id": spots["salamander_id"].to_numpy(),
        "spot_id": spots["spot_id"].astype(int).to_numpy(),
        "embedding": [np.asarray(e, dtype=np.float64).tolist() for e in spots["embedding"]],
    })
    con = duckdb.connect(database=str(db_path))
    try:
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(f"CREATE TABLE {table} "
                    "(salamander_id VARCHAR, spot_id INTEGER, embedding DOUBLE[])")
        con.register("_emb_df", df)
        con.execute(f"INSERT INTO {table} SELECT * FROM _emb_df")
        con.unregister("_emb_df")
    finally:
        con.close()
    logger.info(f"wrote {len(df)} spot embeddings (dim {len(df['embedding'].iloc[0])}) -> {table}")


def write_naive_match_all(
    matches: pd.DataFrame,
    db_path: Path | str = DB_PATH,
    *,
    table: str = "naive_match_all",
) -> None:
    """Persist :func:`naive_match_all` output to ``table`` (replacing it if it exists).

    Columns mirror the returned frame: ``salamander_source_id``, ``salamander_target_id``,
    ``spot_source_id``, ``spot_target_id``, ``similarity``.
    """
    con = duckdb.connect(database=str(db_path))
    try:
        con.register("_match_df", matches)
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM _match_df")
        con.unregister("_match_df")
    finally:
        con.close()
    logger.info(f"wrote {len(matches)} matches -> {table}")


def visualize_spot_matches(
    spots: pd.DataFrame,
    id1: str,
    id2: str,
    *,
    cutoff: float = 0.7,
    method: str = "mutual",
    db_path: Path | str = DB_PATH,
    gap_frac: float = 0.5,
    rad: float = 0.2,
    save_path: Path | str | None = None,
    ax=None,
    emb_col: str = "embedding",
    pairs: list[tuple[int, int, float]] | None = None,
    mark: dict[str, dict[int, float]] | None = None,
    mark_thr: float = 0.25,
    **emb_kwargs,
):
    """Draw the two salamanders side by side and connect matched spots with red arcs.

    Body outlines (grey), spot contours (orange) + centroids, each body labelled by name
    (blue) and each spot by its id (green). For every embedding match with cosine
    similarity ``>= cutoff`` a red arc links the two centroids with the score above it.

    ``method`` is passed to :func:`match_spots`. Shows a window unless ``save_path`` is set.
    Pass ``ax`` to draw into an existing axis (for composing a grid); then the function does not
    create, save, show or close a figure — the caller owns it.

    Two hooks let a *scorer* show its own reasoning instead of this function's default matching:

    ``pairs``   draw these correspondences — ``(i, j, sim)`` in the two id-filtered, index-reset
                row orders — rather than recomputing them here. Use it so the arcs are the ones
                the score was actually computed from (e.g. a gated one-to-one assignment).
    ``mark``    ``{salamander_id: {spot_id: intensity}}``. Spots at or above ``mark_thr`` are drawn
                in magenta, heavier the higher the intensity — for showing which spots a score
                PENALIZED: pattern on one animal with nothing to answer it on the other. Unmatched
                spots are the half of the evidence arcs cannot show.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    given = pairs is not None
    s1, s2, S, auto_pairs = match_spots(spots, id1, id2, method=method, emb_col=emb_col,
                                        **emb_kwargs)
    pairs = pairs if given else auto_pairs
    l1, r1 = _fetch_outline(db_path, id1)
    l2, r2 = _fetch_outline(db_path, id2)

    # place salamander 2 to the right of salamander 1
    dx = float(s1.iloc[0]["width"]) * (1.0 + gap_frac)
    shift = np.array([dx, 0.0])

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 9))
    else:
        fig = ax.figure

    marked = 0

    def _draw(sub, outline_l, outline_r, off, name):
        nonlocal marked
        marks = (mark or {}).get(name, {})
        for out in (outline_l, outline_r):
            if len(out):
                ax.plot(out[:, 0] + off[0], out[:, 1] + off[1], "-", color="0.6", lw=1.0)
        for _, row in sub.iterrows():
            cc = np.vstack([row["contour_xy"], row["contour_xy"][:1]]) + off
            # Two tiers, because on real photos MOST unmatched spots carry some penalty and a flat
            # highlight just turns the animal magenta: outline = charged, filled = charged enough
            # to decide the pair on its own.
            m = float(marks.get(int(row["spot_id"]), 0.0))
            if m >= 2 * mark_thr:                                # a conspicuous spot, unanswered
                ax.fill(cc[:, 0], cc[:, 1], color="magenta", alpha=0.12 + 0.33 * min(m, 1.0))
                ax.plot(cc[:, 0], cc[:, 1], "-", color="magenta", lw=1.0 + 2.5 * min(m, 1.0))
                marked += 1
            elif m >= mark_thr:
                ax.plot(cc[:, 0], cc[:, 1], "-", color="magenta", lw=1.1, alpha=0.7)
            else:
                ax.plot(cc[:, 0], cc[:, 1], "-", color="tab:orange", lw=0.9)
            cx = row["global_centroid_x"] + off[0]
            cy = row["global_centroid_y"] + off[1]
            ax.plot(cx, cy, ".", color="0.25", ms=4)
            ax.text(cx + 4, cy, str(row["spot_id"]), color="green", fontsize=7,
                    ha="left", va="center")
        pts = [o for o in (outline_l, outline_r) if len(o)] or \
              [sub[["global_centroid_x", "global_centroid_y"]].to_numpy()]
        allpts = np.vstack(pts)
        ax.text(allpts[:, 0].mean() + off[0], allpts[:, 1].min() + off[1] - 25, name,
                color="tab:blue", fontsize=14, fontweight="bold", ha="center", va="bottom")

    _draw(s1, l1, r1, np.array([0.0, 0.0]), id1)
    _draw(s2, l2, r2, shift, id2)

    # red arcs for matches above the cutoff
    c1 = s1[["global_centroid_x", "global_centroid_y"]].to_numpy()
    c2 = s2[["global_centroid_x", "global_centroid_y"]].to_numpy() + shift
    n_drawn = 0
    for i, j, sim in pairs:
        if sim < cutoff:
            continue
        p1, p2 = c1[i], c2[j]
        ax.add_patch(FancyArrowPatch(
            p1, p2, connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
            color="red", lw=1.0, alpha=0.6))
        d = p2 - p1
        perp = np.array([-d[1], d[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-9)
        mid = (p1 + p2) / 2 + 0.5 * rad * np.linalg.norm(d) * perp
        ax.text(mid[0], mid[1], f"{sim:.2f}", color="red", fontsize=7, ha="center", va="center")
        n_drawn += 1

    ax.set_aspect("equal")
    ax.invert_yaxis()                                        # image orientation (head up)
    ax.axis("off")
    how = "scorer's assignment" if given else f"cutoff {cutoff}, {method}"
    extra = (f"  ·  {marked} conspicuous spots with nothing to answer them (filled magenta)"
             if mark else "")
    ax.set_title(f"{id1}  vs  {id2}   —   {n_drawn} matches ({how}){extra}", fontsize=11)
    if not own_fig:
        return n_drawn                                       # caller owns the figure/layout
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        logger.info(f"wrote {save_path}  ({n_drawn} matches drawn)")
    else:
        plt.show()
    plt.close(fig)
    return n_drawn

def _size_percentile(spots: pd.DataFrame) -> np.ndarray:
    """Global rank percentile (0 = smallest .. 100 = largest) of each spot's *scale-invariant*
    size: equivalent-circle diameter as a fraction of body length, ``sqrt(4*area/pi)/length_px``.

    Scale-invariant because body ``length_px`` varies ~32x across the set, so raw ``area_pixels``
    is not comparable between a big and a small individual. Rank-based, so colours/filters spread
    evenly over the data. Spots with no body length get ``NaN``.
    """
    L = spots["length_px"].to_numpy(np.float64)
    area = spots["area_pixels"].to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        diam_frac = np.sqrt(4.0 * area / np.pi) / L
    pct = np.full(len(spots), np.nan)
    valid = np.isfinite(diam_frac)
    v = diam_frac[valid]
    order = v.argsort()
    ranks = np.empty(len(v))
    ranks[order] = np.arange(len(v))
    pct[valid] = 100.0 * ranks / max(len(v) - 1, 1)
    return pct


def filter_small_spots(spots: pd.DataFrame, *, min_pct: float = 5.0) -> pd.DataFrame:
    """Drop spots below the ``min_pct`` *global* size percentile (a number in [0, 100]).

    Use :func:`visualize_spot_sizes` to pick ``min_pct`` by eye, then apply it here right after
    ``get_spots()`` so the filter flows through embedding + matching. Scale-invariant (see
    :func:`_size_percentile`), so it treats large and small individuals alike; e.g. ``min_pct=5``
    removes the smallest ~5% of spots dataset-wide (catches junk like ca_29_2 #21 @ p4).
    """
    pct = _size_percentile(spots)
    keep = np.isfinite(pct) & (pct >= min_pct)   # NaN length -> dropped
    return spots[keep].reset_index(drop=True)


def visualize_spot_sizes(
    spots: pd.DataFrame,
    sid: str,
    *,
    cutoff_pct: float | None = None,
    db_path: Path | str = DB_PATH,
    cmap: str = "coolwarm",
    save_path: Path | str | None = None,
    ax=None,
    colorbar: bool = True,
):
    """Heat-map one salamander's spots by size, FEA-style: largest -> dark red, smallest -> dark blue.

    Colour is each spot's **global** size percentile across every spot in ``spots`` (0 = smallest,
    100 = largest), using the same scale-invariant metric as :func:`filter_small_spots`. Colours are
    therefore comparable across individuals, and a percentile you pick here means the same thing
    dataset-wide. Each spot is labelled with its percentile; a colourbar gives the scale.

    Pass ``cutoff_pct`` to preview a filter: spots below that percentile are hatched + black-edged
    (they'd be dropped), counted in the title, and printed. Explore a few individuals, settle on a
    number, then feed it to :func:`filter_small_spots` as ``min_pct``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    spots = spots.copy()
    spots["_size_pct"] = _size_percentile(spots)
    sub = spots[spots.salamander_id == sid].reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(f"no spots for {sid!r}")
    l, r = _fetch_outline(db_path, sid)

    cm = plt.get_cmap(cmap)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 9))
    else:
        fig = ax.figure
    for out in (l, r):
        if len(out):
            ax.plot(out[:, 0], out[:, 1], "-", color="0.6", lw=1.0)

    below = []
    for _, row in sub.iterrows():
        p = row["_size_pct"]
        cc = np.vstack([row["contour_xy"], row["contour_xy"][:1]])
        color = "0.7" if not np.isfinite(p) else cm(p / 100.0)
        is_below = cutoff_pct is not None and np.isfinite(p) and p < cutoff_pct
        if is_below:
            below.append(row)
            ax.fill(cc[:, 0], cc[:, 1], facecolor=color, edgecolor="black", lw=1.4,
                    hatch="///", alpha=0.9)
        else:
            ax.fill(cc[:, 0], cc[:, 1], facecolor=color, edgecolor="0.3", lw=0.8, alpha=0.9)
        cx, cy = row["global_centroid_x"], row["global_centroid_y"]
        ax.text(cx, cy, "-" if not np.isfinite(p) else f"{p:.0f}", color="black", fontsize=6,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))

    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    title = f"{sid} — {len(sub)} spots by global size percentile"
    if cutoff_pct is not None:
        title += f"\ncutoff p{cutoff_pct:g}:  {len(below)} spot(s) below (hatched = dropped)"
    ax.set_title(title, fontsize=11)

    if colorbar:
        sm = ScalarMappable(norm=Normalize(0, 100), cmap=cm); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("global size percentile   (0 = smallest / blue … 100 = largest / red)")
        if cutoff_pct is not None:
            cb.ax.axhline(cutoff_pct, color="black", lw=1.5)

    if below:
        logger.info(f"{sid}: {len(below)} spot(s) below p{cutoff_pct:g} ->")
        for row in sorted(below, key=lambda x: x["_size_pct"]):
            logger.info(f"  spot {int(row['spot_id']):>2}: p{row['_size_pct']:4.1f}  "
                  f"area={row['area_pixels']:.0f}px^2")

    if not own_fig:
        return                                               # caller owns the figure/layout
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        logger.info(f"wrote {save_path}")
    else:
        plt.show()
    plt.close(fig)


EMB_TABLES = {"concat": "spot_embeddings", "tensor": "spot_embeddings_tensor"}


def build_flow(combine: str = "concat", *, c_shape: float = 1.0, c_pos: float = 1.0,
               harmonics: int = SHAPE_HARMONICS, morph: bool = False, gamma: float = 1.0,
               table: str | None = None, naive_matches: bool = True):
    """Build ONE embedding flow end-to-end and persist it to its own table.

    The flows live side by side (``spot_embeddings`` / ``spot_embeddings_tensor`` / any
    ``--table`` you name) so a bakeoff can switch between them with ``EMB_TABLE`` and nothing has
    to be rebuilt to compare. ``harmonics`` and ``morph`` are the representation knobs the
    ``run_representation.sh`` sweep varies:

    * ``harmonics`` raises the EFD cutoff. The default 10 low-passes pixelation noise, and lobes
      with it; more harmonics re-admit both, so this is a signal-vs-noise trade to measure, not a
      dial to turn up.
    * ``morph`` appends size + irregularity/elongation/noncircularity (see :func:`_morph_block`) —
      the two properties the human labels say matter most and the EFD provably discards.
    """
    table = table or EMB_TABLES[combine]
    spots = filter_small_spots(get_spots(), min_pct=10)
    emb = get_spot_embeddings(spots, combine=combine, c_shape=c_shape, c_pos=c_pos,
                              harmonics=harmonics, morph=morph, gamma=gamma)
    spots["embedding"] = [emb[f"{s}#{i}"] for s, i in zip(spots.salamander_id, spots.spot_id)]
    write_spot_embeddings(spots, table=table)
    if naive_matches and combine == "concat":       # the reference table is keyed to the original
        matches = naive_match_all(spots, cutoff=0.7)
        matches["src"] = matches.salamander_source_id.apply(lambda x: x.split("_")[:2])
        matches["tgt"] = matches.salamander_target_id.apply(lambda x: x.split("_")[:2])
        matches["is_match"] = matches.apply(lambda x: x["src"] == x["tgt"], axis=1)
        write_naive_match_all(matches)
    return spots


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build spot embeddings into their own table.")
    ap.add_argument("--combine", choices=["concat", "tensor", "both"], default="concat",
                    help="concat = additive 62-dim (original); tensor = conjunctive 988-dim")
    ap.add_argument("--c-shape", type=float, default=1.0, help="shape veto strength (tensor)")
    ap.add_argument("--c-pos", type=float, default=1.0, help="position veto strength (tensor)")
    ap.add_argument("--harmonics", type=int, default=SHAPE_HARMONICS,
                    help=f"EFD harmonics kept (default {SHAPE_HARMONICS}); higher re-admits lobes "
                         f"AND pixelation noise -- a trade to measure, not a dial to turn up")
    ap.add_argument("--morph", action="store_true",
                    help="append the size + irregularity/elongation/noncircularity block "
                         "(the properties the human labels weight most and the EFD discards)")
    ap.add_argument("--gamma", type=float, default=1.0, help="weight on the morphology block")
    ap.add_argument("--table", default=None, help="override the destination table name")
    _args = ap.parse_args()

    for _mode in (["concat", "tensor"] if _args.combine == "both" else [_args.combine]):
        logger.info(f"=== building '{_mode}' -> {_args.table or EMB_TABLES[_mode]} "
              f"(harmonics={_args.harmonics}, morph={_args.morph}) ===")
        build_flow(_mode, c_shape=_args.c_shape, c_pos=_args.c_pos, harmonics=_args.harmonics,
                   morph=_args.morph, gamma=_args.gamma, table=_args.table,
                   # the naive-match reference table is keyed to the ORIGINAL embedding; a variant
                   # must not overwrite it or every downstream consumer silently changes meaning
                   naive_matches=(_args.table is None and _args.harmonics == SHAPE_HARMONICS
                                  and not _args.morph))
    raise SystemExit(0)


def _legacy_main():
    spots = get_spots()
    # explore spot sizes on a few individuals, then choose a percentile cutoff:
    #visualize_spot_sizes(spots, "ca_29_2", cutoff_pct=5)
    spots = filter_small_spots(spots, min_pct=10)
    emb = get_spot_embeddings(spots)
    spots["embedding"] = [emb[f"{s}#{i}"] for s, i in zip(spots.salamander_id, spots.spot_id)]
    write_spot_embeddings(spots)
    matches = naive_match_all(spots, cutoff=0.7)
    matches['src'] = matches.salamander_source_id.apply(lambda x: x.split('_')[:2])
    matches['tgt'] = matches.salamander_target_id.apply(lambda x: x.split('_')[:2])
    matches['is_match'] = matches.apply(lambda x: x['src'] == x['tgt'], axis=1)
    write_naive_match_all(matches)
 
                  
    #visualize_spot_matches(spots, "aa_1_g1", "lj_21_1", cutoff=0.6, method="mutual")
    #visualize_spot_sizes(spots, "mr_3_1")
    
