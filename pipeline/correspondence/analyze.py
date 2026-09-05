"""What human spot correspondences buy: four separable measurements the pipeline cannot make.

Every metric in this repo collapses four different failures into one number. Given ground-truth
links, they come apart:

1. **Extraction recall** — was the spot even found in the second photo? Nothing downstream can
   recover a spot that was never extracted.
2. **Descriptor quality** — given a TRUE correspondence, does the embedding rank the real partner
   first? This scores the 62-dim vector directly, with no matcher and no training involved, and
   it is reported per block (shape / position / full / tensor) so a failure is localised.
3. **Matching precision & recall** — of the mutual-nearest-neighbour links the matcher actually
   makes, how many are right; and of the true links, how many does it find? This separates "the
   representation is bad" from "the matching rule is bad".
4. **Body-frame drift** — how far apart are the body coordinates of the SAME physical spot? This
   calibrates the position gate from data instead of a sweep, and it detects a head/tail axis
   inversion directly (a flipped axis makes |Δ(1−t)| smaller than |Δt|).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

SHAPE, POS = slice(0, 37), slice(37, 62)


def _norm(x):
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + 1e-12)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def load_links(store_path: Path) -> tuple[list[dict], dict]:
    """-> ([{key,a_sid,b_sid,links,misses}, ...], header) for pairs marked done."""
    data = json.loads(Path(store_path).read_text(encoding="utf-8"))
    out = []
    for key, rec in (data.get("pairs") or {}).items():
        if not rec.get("done") or not rec.get("links"):
            continue
        a_sid, b_sid = key.split("__", 1)
        out.append({"key": key, "a_sid": a_sid, "b_sid": b_sid,
                    "links": [(int(a), int(b)) for a, b in rec["links"]],
                    "misses": [int(m) for m in rec.get("misses", [])]})
    return out, {k: v for k, v in data.items() if k != "pairs"}


def load_spots(db_path: Path, table: str) -> tuple[dict, dict]:
    """-> (emb[sid] -> {spot_id: vector}, geom[sid] -> {spot_id: (axis_t, u, side)})."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        e = con.execute(f"SELECT salamander_id, spot_id, embedding FROM {table}").fetchall()
        g = con.execute(
            "SELECT s.salamander_id, s.spot_id, s.axis_t, s.axis_offset, s.axis_side, "
            "       q.avg_width_px "
            "FROM spots s LEFT JOIN image_quality q USING (salamander_id)").fetchall()
    finally:
        con.close()
    emb: dict[str, dict[int, np.ndarray]] = {}
    for sid, spot_id, vec in e:
        emb.setdefault(sid, {})[int(spot_id)] = np.asarray(vec, np.float64)
    geom: dict[str, dict[int, tuple]] = {}
    for sid, spot_id, t, off, side, w in g:
        half = (float(w) / 2.0) if w else None
        u = (float(off) / half) if (off is not None and half) else np.nan
        geom.setdefault(sid, {})[int(spot_id)] = (
            float(t) if t is not None else np.nan, u, 1.0 if side == "left" else -1.0)
    return emb, geom


def rank_of_partner(qv, gallery: dict[int, np.ndarray], true_id: int, sl=slice(None)) -> int | None:
    """Rank of the TRUE partner among all spots of the other photo (1 = best). None if absent."""
    if true_id not in gallery:
        return None
    ids = list(gallery)
    M = _norm(np.vstack([gallery[i][sl] for i in ids]))
    sims = M @ _norm(np.asarray(qv)[sl])
    order = np.argsort(-sims)
    return int(np.where(np.array(ids)[order] == true_id)[0][0]) + 1


def mutual_nn(a_emb: dict, b_emb: dict) -> set[tuple[int, int]]:
    """The correspondences the MATCHER would make: mutual nearest neighbours by cosine."""
    ai, bi = list(a_emb), list(b_emb)
    if not ai or not bi:
        return set()
    A, B = _norm(np.vstack([a_emb[i] for i in ai])), _norm(np.vstack([b_emb[i] for i in bi]))
    S = A @ B.T
    a_best, b_best = S.argmax(1), S.argmax(0)
    return {(ai[i], bi[int(a_best[i])]) for i in range(len(ai))
            if int(b_best[int(a_best[i])]) == i}


def describe(name: str, v: np.ndarray, unit: str = "") -> str:
    if not len(v):
        return f"  {name:22s} (no data)"
    return (f"  {name:22s} median {np.median(v):7.3f}{unit}   "
            f"p90 {np.percentile(v, 90):7.3f}{unit}   mean {v.mean():7.3f}{unit}")


def analyze(store_path: Path, db_path: Path, tables: dict[str, str]) -> int:
    pairs, header = load_links(store_path)
    if not pairs:
        logger.info(f"no completed pairs in {store_path} — label some first "
              f"(mark a pair Done with Enter).")
        return 1

    n_links = sum(len(p["links"]) for p in pairs)
    n_miss = sum(len(p["misses"]) for p in pairs)
    logger.info("=" * 74)
    logger.info(" SPOT-CORRESPONDENCE ANALYSIS")
    logger.info("=" * 74)
    logger.info(f" store : {store_path}")
    logger.info(f" pairs : {len(pairs)} completed   links: {n_links}   marked-missing: {n_miss}")

    # ---- 1. extraction recall -----------------------------------------------------------
    logger.info("-" * 74)
    logger.info(" 1. EXTRACTION RECALL — was the spot even found in the second photo?")
    logger.info("-" * 74)
    denom = n_links + n_miss
    if denom:
        logger.info(f"  spots with a partner extracted in B : {n_links}/{denom} = "
              f"{n_links / denom:.1%}")
        logger.info(f"  visible in B but NEVER extracted     : {n_miss}  "
              f"({n_miss / denom:.1%}) <- no descriptor can recover these")
    else:
        logger.info("  (nothing marked missing — recall not measurable)")

    # ---- 2. descriptor quality ----------------------------------------------------------
    logger.info("-" * 74)
    logger.info(" 2. DESCRIPTOR QUALITY — given a TRUE link, does the partner rank first?")
    logger.info("-" * 74)
    logger.info(f"  {'representation':22s} {'rank-1':>8s} {'rank<=3':>8s} {'median':>8s} "
          f"{'MRR':>7s} {'norm.rank':>10s} {'n':>5s}")

    for tname, table in tables.items():
        try:
            emb, geom = load_spots(db_path, table)
        except duckdb.CatalogException:
            continue
        blocks = ({"full": slice(None)} if tname != "concat"
                  else {"full": slice(None), "shape [:37]": SHAPE, "position [37:]": POS})
        for bname, sl in blocks.items():
            ranks, norm_ranks = [], []
            for p in pairs:
                ae, be = emb.get(p["a_sid"], {}), emb.get(p["b_sid"], {})
                for a_id, b_id in p["links"]:
                    if a_id not in ae:
                        continue
                    r = rank_of_partner(ae[a_id], be, b_id, sl)
                    if r is None:
                        continue
                    ranks.append(r)
                    norm_ranks.append(r / max(len(be), 1))
            if not ranks:
                continue
            r = np.array(ranks)
            label = f"{tname}:{bname}" if len(blocks) > 1 or len(tables) > 1 else bname
            logger.info(f"  {label:22s} {(r == 1).mean():>8.3f} {(r <= 3).mean():>8.3f} "
                  f"{np.median(r):>8.1f} {(1 / r).mean():>7.3f} "
                  f"{np.mean(norm_ranks):>10.3f} {len(r):>5d}")

    # ---- 3. matching precision / recall --------------------------------------------------
    emb, geom = load_spots(db_path, tables.get("concat", next(iter(tables.values()))))
    logger.info("-" * 74)
    logger.info(" 3. AUTOMATIC MATCHING — mutual nearest neighbour vs the truth")
    logger.info("-" * 74)
    tp = fp = fn = 0
    for p in pairs:
        truth = set(p["links"])
        got = mutual_nn(emb.get(p["a_sid"], {}), emb.get(p["b_sid"], {}))
        # Only judge machine links whose A-spot a human actually ruled on; the rest are unknown.
        judged_a = {a for a, _ in truth} | set(p["misses"])
        got_j = {(a, b) for a, b in got if a in judged_a}
        tp += len(got_j & truth)
        fp += len(got_j - truth)
        fn += len(truth - {(a, b) for a, b in got})
    if tp + fp:
        logger.info(f"  precision : {tp}/{tp + fp} = {tp / (tp + fp):.1%}   "
              f"of the links the matcher makes, this many are the same physical spot")
    if tp + fn:
        logger.info(f"  recall    : {tp}/{tp + fn} = {tp / (tp + fn):.1%}   "
              f"of true links, this many are recovered")

    # ---- 4. body-frame drift -------------------------------------------------------------
    logger.info("-" * 74)
    logger.info(" 4. BODY FRAME — how far apart are the coordinates of the SAME spot?")
    logger.info("-" * 74)
    dt, du, flips = [], [], []
    for p in pairs:
        ga, gb = geom.get(p["a_sid"], {}), geom.get(p["b_sid"], {})
        d_norm, d_flip = [], []
        for a_id, b_id in p["links"]:
            if a_id not in ga or b_id not in gb:
                continue
            ta, ua, _ = ga[a_id]
            tb, ub, _ = gb[b_id]
            if np.isfinite(ta) and np.isfinite(tb):
                dt.append(abs(ta - tb))
                d_norm.append(abs(ta - tb))
                d_flip.append(abs(ta - (1.0 - tb)))       # B's axis read head<->tail
            if np.isfinite(ua) and np.isfinite(ub):
                du.append(abs(ua - ub))
        if d_norm and np.mean(d_flip) < np.mean(d_norm):
            flips.append(p["key"])
    logger.info(describe("|delta axis_t|", np.array(dt), "  (0..1 along the body)"))
    logger.info(describe("|delta u| (width)", np.array(du)))
    if dt:
        logger.info(f"  pairs whose axis looks HEAD/TAIL INVERTED: {len(flips)}/{len(pairs)}"
              + (f"  -> {', '.join(flips[:6])}" if flips else ""))
        logger.info("  (inverted = the same spots line up better against 1-t than against t)")

    # ---- verdict -------------------------------------------------------------------------
    logger.info("=" * 74)
    logger.info(" WHERE THE LOSS IS")
    logger.info("=" * 74)
    if denom and n_miss / denom > 0.25:
        logger.info(f"  EXTRACTION: {n_miss / denom:.0%} of spots visible in both photos are missing "
              f"from one.\n  Fix segmentation first — the descriptor cannot see what was never "
              f"extracted.")
    if 'r' in dir() and len(ranks):
        r1 = (np.array(ranks) == 1).mean()
        if r1 < 0.5:
            logger.info(f"  DESCRIPTOR: the true partner ranks first only {r1:.0%} of the time. The "
                  f"62-dim\n  embedding does not identify the same physical spot across photos.")
        else:
            logger.info(f"  DESCRIPTOR is healthy ({r1:.0%} rank-1) — the loss is in matching or "
                  f"aggregation.")
    if flips:
        logger.info(f"  BODY FRAME: {len(flips)} pair(s) look head/tail inverted. That alone would "
              f"cripple\n  every position-based rule, including the conjunctive embedding.")
    return 0
