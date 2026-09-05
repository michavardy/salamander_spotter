"""Where a spot is LOST between two photos of one animal — painted, extracted, matched.

``unmatched_examples.py`` draws the pairs that fail and marks what the matcher could not
explain. It cannot say *why*, because it only ever sees the spots that reached the database.
A spot can drop out at three stages, and only the first two are visible upstream of the
embedding:

    1. PAINT      a human painted the spot in one photo and not in the other. Nothing
                  downstream can invent it, and no amount of descriptor work will help.
    2. EXTRACT    the paint is there but no row reached ``spots``: the blob fell outside the
                  body mask (the key is AND-ed with it), was eaten by the OPEN kernel, fell
                  under ``min_area``, or was CLOSE-d into a neighbour.
    3. MATCH      the spot exists in both photos' ``spots`` rows and the matcher still failed
                  to pair them — the only failure the descriptor is actually responsible for.

This renders one panel per pair with all three visible at once: both photos with the body
mask outline, every painted blob, which of them became rows, and which rows found a partner;
plus the two animals unrolled into the shared body frame (t along the body, u across it) so
a missing spot is a visible hole rather than a number.

    pixi run missing-spots                          # a spread of cases -> artifacts/.../missing/
    pixi run missing-spots --pairs ca_14_1:ca_14_2,ke_1_1:ke_1_2
    pixi run missing-spots --auto 8 --case paint_loss

Colours:  green  = extracted here AND matched in the other photo
          red    = extracted here, NO partner found          (stage 3, descriptor/matcher)
          cyan   = painted here, never became a spot row     (stage 2, extraction)
          grey   = the body mask the key is confined to
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ST = Path(__file__).resolve().parents[1]
_ROOT = _ST.parents[1]
for _sub in ("core", "models", "eval", "viz"):
    p = str(_ST / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import data as d                                                    # noqa: E402
import embeddings as E                                              # noqa: E402
from pipeline.generate_spot_labels import binning as B              # noqa: E402
from pipeline.generate_spot_labels import extract_spot_contours as X  # noqa: E402

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

MATCH_THR = 0.4
CASES = ["paint_loss", "extract_loss", "match_loss", "success"]

C_MATCHED = "#1a9850"      # extracted + matched
C_UNMATCHED = "#d73027"    # extracted, no partner
C_DROPPED = "#00b3d6"      # painted, never extracted
C_BODY = "#9a9a9a"


# --------------------------------------------------------------------------- loading

def painted_blobs(bgr: np.ndarray, min_area: float = 40.0) -> list[dict]:
    """Every blob a human painted, keyed WITHOUT the body mask — the upstream truth.

    Deliberately the plain magenta key (:func:`extract_spot_contours.magenta_mask` with
    ``body_mask=None``): the point is to see what the pipeline's body-mask AND removes, so the
    reference must not be confined by the same mask. Red-band spots are not keyed here for the
    same reason — outside the body they are indistinguishable from leaf litter.
    """
    import cv2

    mask = X.magenta_mask(bgr, morph=X.DEFAULT_MORPH, body_mask=None)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        out.append({"area": area,
                    "centroid": (m["m10"] / m["m00"], m["m01"] / m["m00"]),
                    "xy": cnt.reshape(-1, 2).astype(float)})
    return sorted(out, key=lambda s: -s["area"])


def body_mask_of(db_path, sid: str) -> np.ndarray | None:
    import cv2
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute("SELECT body_mask_png FROM images WHERE salamander_id = ?",
                          [sid]).fetchone()
    finally:
        con.close()
    if not row or row[0] is None:
        return None
    return cv2.imdecode(np.frombuffer(bytes(row[0]), np.uint8), cv2.IMREAD_GRAYSCALE)


def classify_blobs(blobs: list[dict], spots: pd.DataFrame, mask: np.ndarray | None,
                   shape: tuple[int, int]) -> list[dict]:
    """Tag each painted blob KEPT / DROPPED by AREA OVERLAP with the ``spots`` rows.

    Overlap, not centroid distance: the rows are often a different shape from the paint (the
    adaptive key eats a shaded rim, CLOSE fuses two blobs into one row), and a centroid test
    calls those losses when the pixels are plainly covered. Rasterized once per photo into a
    downscaled label canvas, so the cost is one fill per spot rather than a polygon test per
    pair.
    """
    import cv2

    h, w = shape
    scale = min(1.0, 900.0 / max(h, w))
    H, W = max(int(h * scale), 1), max(int(w * scale), 1)
    labels = np.zeros((H, W), np.int32)
    for k, r in enumerate(spots.itertuples(index=False), start=1):
        poly = np.round(np.asarray(r.contour_xy, float) * scale).astype(np.int32)
        cv2.fillPoly(labels, [poly], k)
    row_ids = spots["spot_id"].to_numpy()
    row_area = np.bincount(labels.ravel(), minlength=len(row_ids) + 1)

    for b in blobs:
        canvas = np.zeros((H, W), np.uint8)
        cv2.fillPoly(canvas, [np.round(b["xy"] * scale).astype(np.int32)], 1)
        px = int(canvas.sum())
        hit = labels[canvas > 0]
        counts = np.bincount(hit, minlength=len(row_ids) + 1)
        covered = int(counts[1:].sum())
        b["covered_frac"] = covered / px if px else 0.0
        b["kept"] = b["covered_frac"] >= 0.5
        touching = [i for i in range(1, len(counts)) if counts[i] >= 0.15 * px]
        b["rows"] = [int(row_ids[i - 1]) for i in touching]
        if b["kept"]:
            j = int(counts[1:].argmax()) + 1
            b["spot_id"] = int(row_ids[j - 1])
            # one row spanning several painted blobs = CLOSE fused them
            b["merged"] = bool(row_area[j] > 1.6 * px)
            b["split"] = len(touching) > 1
            b["reason"] = ("split into several rows" if b["split"] else
                           "fused with a neighbour" if b["merged"] else "kept")
            continue
        b["spot_id"], b["merged"], b["split"] = None, False, False
        inside = np.nan
        if mask is not None:
            mh, mw = mask.shape[:2]
            pts = np.round(b["xy"]).astype(int)
            pts[:, 0] = np.clip(pts[:, 0], 0, mw - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, mh - 1)
            inside = float((mask[pts[:, 1], pts[:, 0]] > 127).mean())
        b["inside_frac"] = inside
        if np.isfinite(inside) and inside < 0.5:
            b["reason"] = "outside the body mask"
        elif b["area"] < 120:
            b["reason"] = "small — OPEN kernel / min_area"
        else:
            b["reason"] = "keyed away (redness below the Otsu cut)"
    return blobs


def body_frame(pts: np.ndarray, midline, half_w: float) -> np.ndarray:
    """(k,2) image px -> (k,2) body frame: t along the body (0=head), u across it in half-widths."""
    out = np.empty((len(pts), 2), float)
    for i, p in enumerate(pts):
        t, off = B.project(p, midline)
        out[i] = (t, off / half_w if half_w else 0.0)
    return out


# --------------------------------------------------------------------------- the pair

def _head_up(sub: pd.DataFrame) -> float:
    """Rotation (degrees) that lays this animal's head->tail chord left-to-right."""
    r = sub.iloc[0]
    v = np.array([r.tail_tip_x - r.head_x, r.tail_tip_y - r.head_y], float)
    if not np.isfinite(v).all() or np.hypot(*v) < 1e-6:
        return 0.0
    return float(np.degrees(np.arctan2(v[1], v[0])))


class Pair:
    """Everything one panel needs about two photos of one animal.

    Both photos are rotated head-left before anything is drawn. Without it the two animals sit
    at whatever angle they were photographed at, and the eye has to do the alignment that the
    whole panel exists to make unnecessary; with it the photos read in the same direction as
    the unrolled strip below them.
    """

    def __init__(self, sid_a: str, sid_b: str, spots: pd.DataFrame, emb: pd.DataFrame,
                 images_dir: Path, db_path: Path):
        import cv2

        self.sid_a, self.sid_b = sid_a, sid_b
        self.spots = {s: spots[spots.salamander_id == s].sort_values("spot_id")
                      for s in (sid_a, sid_b)}
        for s, sub in self.spots.items():
            if sub.empty:
                raise SystemExit(f"no spots for {s}")
        self.emb = {s: emb[emb.salamander_id == s].set_index("spot_id") for s in (sid_a, sid_b)}
        self.img, self.mask, self.blobs, self.rot = {}, {}, {}, {}
        for s in (sid_a, sid_b):
            path = images_dir / "purple" / f"{s}.png"
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise SystemExit(f"cannot read purple image {path}")
            sub = self.spots[s]
            self.mask[s] = body_mask_of(db_path, s)
            self.blobs[s] = classify_blobs(painted_blobs(bgr), sub, self.mask[s], bgr.shape[:2])
            img, M = self._rotate(bgr, _head_up(sub))
            self.rot[s] = M
            self.img[s] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.mask[s] is not None:
                self.mask[s] = self._rotate(self.mask[s], _head_up(sub), M=M)[0]
            for b in self.blobs[s]:
                b["xy_rot"] = self.warp(s, b["xy"])
        self.matched = self._match()

    @staticmethod
    def _rotate(img: np.ndarray, deg: float, M=None):
        """Rotate about the image centre into a canvas that still holds every corner."""
        import cv2

        h, w = img.shape[:2]
        if M is None:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            W, H = int(h * sin + w * cos), int(h * cos + w * sin)
            M[0, 2] += W / 2 - w / 2
            M[1, 2] += H / 2 - h / 2
            M = np.vstack([M, [0, 0, 1]])
            M = (M, (W, H))
        (mat, size) = M
        return cv2.warpAffine(img, mat[:2], size, flags=cv2.INTER_AREA,
                              borderValue=(255, 255, 255) if img.ndim == 3 else 0), M

    def warp(self, sid: str, pts: np.ndarray) -> np.ndarray:
        """Image points -> the rotated frame the panel draws in."""
        mat = self.rot[sid][0]
        p = np.asarray(pts, float).reshape(-1, 2)
        return (p @ mat[:2, :2].T) + mat[:2, 2]

    def _match(self) -> dict[str, dict[int, tuple[int, float]]]:
        """spot_id -> (partner spot_id, similarity) for mutual nearest neighbours over MATCH_THR."""
        ids, vecs = {}, {}
        for s in (self.sid_a, self.sid_b):
            sub = self.spots[s]
            keep = [i for i in sub.spot_id.to_numpy() if i in self.emb[s].index]
            ids[s] = np.array(keep, int)
            vecs[s] = np.vstack([np.asarray(self.emb[s].loc[i, "embedding"], float) for i in keep])
            vecs[s] /= np.linalg.norm(vecs[s], axis=1, keepdims=True) + 1e-12
        S = vecs[self.sid_a] @ vecs[self.sid_b].T
        ab, ba = S.argmax(1), S.argmax(0)
        out = {self.sid_a: {}, self.sid_b: {}}
        for i in range(len(S)):
            j = int(ab[i])
            if int(ba[j]) == i and float(S[i, j]) >= MATCH_THR:
                out[self.sid_a][int(ids[self.sid_a][i])] = (int(ids[self.sid_b][j]), float(S[i, j]))
                out[self.sid_b][int(ids[self.sid_b][j])] = (int(ids[self.sid_a][i]), float(S[i, j]))
        return out

    def stats(self, sid: str) -> dict:
        sub, blobs = self.spots[sid], self.blobs[sid]
        return {"painted": len(blobs),
                "dropped": sum(1 for b in blobs if not b["kept"]),
                "extracted": len(sub),
                "matched": len(self.matched[sid])}


# --------------------------------------------------------------------------- drawing

def _draw_photo(ax, pair: Pair, sid: str):
    import cv2

    img, mask = pair.img[sid], pair.mask[sid]
    ax.imshow(img)
    if mask is not None:
        cs, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            c = c.reshape(-1, 2)
            ax.plot(*np.vstack([c, c[:1]]).T, color=C_BODY, lw=1.4, alpha=0.9)

    for b in pair.blobs[sid]:
        if not b["kept"]:
            xy = np.vstack([b["xy_rot"], b["xy_rot"][:1]])
            ax.plot(*xy.T, color=C_DROPPED, lw=2.0)

    for r in pair.spots[sid].itertuples(index=False):
        spot_id = int(r.spot_id)
        col = C_MATCHED if spot_id in pair.matched[sid] else C_UNMATCHED
        xy = pair.warp(sid, np.asarray(r.contour_xy, float))
        ax.plot(*np.vstack([xy, xy[:1]]).T, color=col, lw=1.4)
        c = pair.warp(sid, [[r.global_centroid_x, r.global_centroid_y]])[0]
        ax.text(c[0], c[1], str(spot_id), color=col, fontsize=6.5, ha="center", va="center",
                weight="bold")

    # crop to the animal plus a margin, so the litter around it does not dominate
    pts = [pair.warp(sid, np.asarray(r, float)) for r in pair.spots[sid].contour_xy]
    pts += [b["xy_rot"] for b in pair.blobs[sid]]
    allxy = np.vstack(pts)
    pad = 0.10 * max(np.ptp(allxy[:, 0]), np.ptp(allxy[:, 1])) + 20
    ax.set_xlim(allxy[:, 0].min() - pad, allxy[:, 0].max() + pad)
    ax.set_ylim(allxy[:, 1].max() + pad, allxy[:, 1].min() - pad)
    ax.set_xticks([]); ax.set_yticks([])
    st = pair.stats(sid)
    ax.set_title(f"{sid}   painted {st['painted']}  ->  extracted {st['extracted']}"
                 f"  ->  matched {st['matched']}", fontsize=10)


def _draw_unrolled(ax, pair: Pair):
    """Both animals in the shared body frame: t along the body, u across it, A above B.

    This is the comparison the photos cannot make: pose, scale and rotation are divided out, so
    two spots of the same physical marking land at the same place on the page and a spot present
    in one photo only shows up as a hole in the other lane.
    """
    lanes = {pair.sid_a: 2.3, pair.sid_b: -2.3}
    for sid, y0 in lanes.items():
        sub = pair.spots[sid]
        half = float(sub["avg_width_px"].iloc[0]) / 2.0 if "avg_width_px" in sub else 1.0
        mid = np.asarray(sub["midline_xy"].iloc[0], float)
        ax.axhline(y0, color=C_BODY, lw=0.8, alpha=0.5, zorder=0)
        ax.text(-0.035, y0, sid, ha="right", va="center", fontsize=9, weight="bold")
        for r in sub.itertuples(index=False):
            bf = body_frame(np.asarray(r.contour_xy, float), mid, half)
            if not np.isfinite(bf).all():
                continue
            col = C_MATCHED if int(r.spot_id) in pair.matched[sid] else C_UNMATCHED
            ax.fill(bf[:, 0], y0 + np.clip(bf[:, 1], -1.9, 1.9), color=col, alpha=0.75, lw=0.0,
                    zorder=2)
        for b in pair.blobs[sid]:
            if b["kept"]:
                continue
            bf = body_frame(b["xy"], mid, half)
            ax.fill(bf[:, 0], y0 + np.clip(bf[:, 1], -1.9, 1.9), facecolor="none", lw=1.4,
                    edgecolor=C_DROPPED, zorder=3)
    ax.set_xlim(-0.06, 1.02)
    ax.set_ylim(-4.9, 4.9)
    ax.set_yticks([])
    ax.set_xlabel("position along the body   (0 = head,  1 = tail tip);  "
                  "vertical = across the body, in half-widths", fontsize=9)
    ax.set_title("both animals unrolled into the body frame — a hole in one lane is a lost spot",
                 fontsize=10)


def render(pair: Pair, outdir: Path, case: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], hspace=0.16, wspace=0.06)
    _draw_photo(fig.add_subplot(gs[0, 0]), pair, pair.sid_a)
    _draw_photo(fig.add_subplot(gs[0, 1]), pair, pair.sid_b)
    _draw_unrolled(fig.add_subplot(gs[1, :]), pair)

    sa, sb = pair.stats(pair.sid_a), pair.stats(pair.sid_b)
    surv = sa["matched"] / max(sa["extracted"], 1)
    head = f"[{case}]  " if case else ""
    fig.suptitle(
        f"{head}{pair.sid_a}  vs  {pair.sid_b}   (same animal)\n"
        f"paint {sa['painted']}/{sb['painted']}   ->   extracted {sa['extracted']}/"
        f"{sb['extracted']}  (dropped {sa['dropped']}/{sb['dropped']})   ->   "
        f"matched {sa['matched']} = {surv:.0%} of A's spots",
        fontsize=12)
    fig.legend(handles=[
        Line2D([], [], color=C_MATCHED, lw=2, label="extracted + matched in the other photo"),
        Line2D([], [], color=C_UNMATCHED, lw=2, label="extracted, no partner found (matcher)"),
        Line2D([], [], color=C_DROPPED, lw=2, label="painted, never extracted (segmentation)"),
        Line2D([], [], color=C_BODY, lw=2, label="body mask (the key is AND-ed with it)"),
    ], loc="lower center", ncol=4, fontsize=9, frameon=False)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{(case + '__') if case else ''}{pair.sid_a}__{pair.sid_b}.png"
    fig.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)
    return out


def report(pair: Pair) -> str:
    """Per-blob text for the pair — what was lost and the likely reason."""
    lines = []
    for sid in (pair.sid_a, pair.sid_b):
        st = pair.stats(sid)
        lines.append(f"  {sid}: painted {st['painted']}  extracted {st['extracted']}  "
                     f"matched {st['matched']}")
        for b in pair.blobs[sid]:
            if b["kept"]:
                continue
            inside = b.get("inside_frac", np.nan)
            lines.append(f"      LOST paint  area {b['area']:7.0f}px  "
                         f"inside-body {inside:.0%}" if np.isfinite(inside) else
                         f"      LOST paint  area {b['area']:7.0f}px")
            lines[-1] += f"   {b['reason']}"
    return "\n".join(lines)


# --------------------------------------------------------------------------- selection

def auto_pairs(spots: pd.DataFrame, emb: pd.DataFrame, n: int, case: str | None) -> list:
    """Pick pairs by the failure they illustrate, from the cheap stats (no images read)."""
    idx = _ROOT / "artifacts" / "spot_transformer" / "unmatched" / "pair_index.csv"
    if not idx.is_file():
        raise SystemExit(f"no pair index at {idx} — run `pixi run unmatched-examples` first")
    p = pd.read_csv(idx)
    p = p[(p.n_a >= 8) & (p.n_b >= 8)]
    picks = []
    cases = [case] if case else CASES
    for c in cases:
        k = max(1, n // len(cases))
        if c == "paint_loss":                       # photos that disagree wildly on spot count
            sel = p.nsmallest(k, "count_ratio")
        elif c == "extract_loss":                   # low survival, similar counts, decent quality
            sel = p[(p.count_ratio >= 0.85) & (p.q_min >= p.q_min.median())].nsmallest(k, "survival")
        elif c == "match_loss":                     # near-identical counts and pose, still failing
            sel = p[(p.count_ratio >= 0.9) & (p.d_curl <= 2.0)].nsmallest(k, "survival")
        elif c == "success":
            sel = p.nlargest(k, "survival")
        else:
            raise SystemExit(f"unknown case {c!r}; expected {CASES}")
        picks += [(r.sid_a, r.sid_b, c) for r in sel.itertuples(index=False)]
    return picks


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=None,
                    help="comma list of A:B photo ids, e.g. ca_14_1:ca_14_2,ke_1_1:ke_1_2")
    ap.add_argument("--auto", type=int, default=8, help="how many pairs to pick automatically")
    ap.add_argument("--case", default=None, choices=CASES)
    ap.add_argument("--images", default="all_sasa_norm", help="images/<folder> with purple/")
    ap.add_argument("--out", default=None, help="output dir (default artifacts/.../missing)")
    args = ap.parse_args()

    images_dir = _ROOT / "images" / args.images
    outdir = Path(args.out) if args.out else _ROOT / "artifacts" / "spot_transformer" / "missing"

    logger.info("loading spots + embeddings ...")
    spots = E.get_spots()
    emb = d.get_spot_embeddings()

    if args.pairs:
        picks = [(*pair.split(":"), "") for pair in args.pairs.split(",") if pair.strip()]
    else:
        picks = auto_pairs(spots, emb, args.auto, args.case)

    written = []
    for a, b, case in picks:
        try:
            pair = Pair(a, b, spots, emb, images_dir, d.DB_PATH)
            out = render(pair, outdir, case)
        except Exception as exc:                                  # noqa: BLE001
            logger.error(f"  !! {a} vs {b}: {type(exc).__name__}: {exc}")
            continue
        written.append(out)
        logger.info(f"[{case or 'pair'}] {a} vs {b} -> {out.name}")
        logger.info(report(pair))

    logger.info(f"wrote {len(written)} panels -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
