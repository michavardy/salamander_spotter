"""Are two photos labelled as one animal actually the same animal? — ask the GPS, not the model.

Reviewing residual failures turned up ``bf_1_1`` vs ``bf_1_2``: same identity label, 1 of 16 spots
matching, both photos good quality and similar pose. The photographs carry burned-in coordinates
**7 km apart**, taken a year apart. *Salamandra salamandra* has a home range of tens to a few
hundred metres and is strongly site-faithful, so those are almost certainly two different animals
filed under one identity.

That matters far beyond one pair. results.md #9 already found the mirror-image fault — 54 pairs of
*different* labels whose photos agree as well as genuine repeats — and noted that such errors are
"invisible to every metric but poison them all". This is the same fault in the other direction, and
unlike #9 it can be checked **without the matcher**: 61% of the photos carry EXIF GPS, which is
evidence the model never sees and therefore cannot be circular about.

Two outputs:

* a worklist of individuals whose photos are implausibly far apart, ranked by distance;
* the cross-check that says whether this matters — do those individuals also show low spot
  survival? If mislabelled individuals match badly, then part of the 24% survival rate is not a
  segmentation or representation failure at all. It is the ground truth being wrong.

    pixi run geo-check
    pixi run geo-check --max-metres 300
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval"):
    p = str(_ST / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import data as d                                     # noqa: E402
from aggregator import _norm                         # noqa: E402

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

MATCH_THR = 0.4
IMAGE_DIR = "images/all_sasa_norm"
# Salamandra salamandra is strongly site-faithful; typical home ranges are tens to a few hundred
# metres and individuals return to the same shelters across years. A few hundred metres is generous;
# kilometres are not a movement, they are a labelling error.
DEFAULT_MAX_M = 500.0


def _dms(v) -> float:
    dd, mm, ss = [float(x) for x in v]
    return dd + mm / 60.0 + ss / 3600.0


def read_exif_geo(image_dir: str = IMAGE_DIR) -> pd.DataFrame:
    """``sid, lat, lon, taken`` for every photo carrying EXIF GPS. Missing GPS is simply absent —
    it is not evidence of anything, so those photos are excluded rather than defaulted."""
    from PIL import Image, ExifTags                                      # noqa: PLC0415
    rows = []
    for p in sorted((d.REPO_ROOT / image_dir).glob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            ex = Image.open(p)._getexif() or {}
        except Exception:
            continue
        tags = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
        gps = tags.get("GPSInfo")
        if not gps or 2 not in gps or 4 not in gps:
            continue
        try:
            lat = _dms(gps[2]) * (-1 if str(gps.get(1, "N")).upper().startswith("S") else 1)
            lon = _dms(gps[4]) * (-1 if str(gps.get(3, "E")).upper().startswith("W") else 1)
        except Exception:
            continue
        rows.append(dict(sid=p.stem, lat=lat, lon=lon, taken=tags.get("DateTimeOriginal")))
    return pd.DataFrame(rows)


def haversine_m(a_lat, a_lon, b_lat, b_lon) -> float:
    R = 6371000.0
    p1, p2 = np.radians(a_lat), np.radians(b_lat)
    dp, dl = p2 - p1, np.radians(b_lon - a_lon)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(h)))


def survival_by_pair() -> pd.DataFrame:
    """Spot survival per same-label photo pair — the quantity we cross-check the GPS against."""
    sets = [s for s in d.get_image_sets(d.get_spot_embeddings()) if not s.is_synth]
    by: dict[str, list] = {}
    for s in sets:
        if len(s.spots) >= 4:
            by.setdefault(s.label, []).append(s)
    rows = []
    for lbl, ss in by.items():
        for i, A in enumerate(ss):
            for B in ss[i + 1:]:
                S = _norm(A.spots) @ _norm(B.spots).T
                ab, bb = S.argmax(1), S.argmax(0)
                n = sum(1 for k in range(len(S))
                        if bb[ab[k]] == k and float(S[k, ab[k]]) >= MATCH_THR)
                rows.append(dict(label=lbl, sid_a=A.sid, sid_b=B.sid,
                                 survival=n / min(len(A.spots), len(B.spots)), matched=n))
    return pd.DataFrame(rows)


def main():
    import argparse
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-metres", type=float, default=DEFAULT_MAX_M)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    logger.info("reading EXIF GPS ...")
    geo = read_exif_geo()
    if not len(geo):
        raise SystemExit("no EXIF GPS found — this check needs the original photo files")
    geo["label"] = geo["sid"].str.rsplit("_", n=1).str[0]
    logger.info(f" {len(geo)} photos carry GPS, over {geo['label'].nunique()} individuals")

    pairs = survival_by_pair()
    g = geo.set_index("sid")
    dist, taken_gap = [], []
    for r in pairs.itertuples(index=False):
        if r.sid_a in g.index and r.sid_b in g.index:
            a, b = g.loc[r.sid_a], g.loc[r.sid_b]
            dist.append(haversine_m(a.lat, a.lon, b.lat, b.lon))
            try:
                ta = pd.to_datetime(a.taken, format="%Y:%m:%d %H:%M:%S")
                tb = pd.to_datetime(b.taken, format="%Y:%m:%d %H:%M:%S")
                taken_gap.append(abs((tb - ta).days))
            except Exception:
                taken_gap.append(np.nan)
        else:
            dist.append(np.nan); taken_gap.append(np.nan)
    pairs["metres"] = dist
    pairs["days_apart"] = taken_gap
    have = pairs.dropna(subset=["metres"])
    logger.info(f" {len(have)} same-label photo pairs have GPS on BOTH sides")

    far = have[have["metres"] > args.max_metres].sort_values("metres", ascending=False)
    logger.info(f"{'=' * 82}")
    logger.info(f" PAIRS LABELLED AS ONE ANIMAL BUT PHOTOGRAPHED > {args.max_metres:.0f} m APART")
    logger.info(f"{'=' * 82}")
    logger.info(f" {len(far)} of {len(have)} pairs ({len(far) / max(len(have), 1):.1%}), "
          f"covering {far['label'].nunique()} individuals")
    logger.info(f" {'individual':>12}{'photo A':>12}{'photo B':>12}{'metres':>10}{'days':>7}"
          f"{'survival':>10}")
    for r in far.head(args.top).itertuples(index=False):
        logger.info(f" {r.label:>12}{r.sid_a:>12}{r.sid_b:>12}{r.metres:>10,.0f}"
              f"{('' if np.isnan(r.days_apart) else int(r.days_apart)):>7}{r.survival:>10.2f}")

    # ---- the cross-check that decides whether this matters ----
    logger.info(f"{'=' * 82}")
    logger.info(" DOES IT EXPLAIN ANYTHING? — spot survival vs distance between the two photos")
    logger.info(f"{'=' * 82}")
    bins = [(0, 50), (50, 200), (200, 500), (500, 2000), (2000, 1e9)]
    logger.info(f" {'distance apart':>18}{'pairs':>8}{'mean survival':>15}{'median':>9}")
    for lo, hi in bins:
        s = have[(have["metres"] >= lo) & (have["metres"] < hi)]
        if not len(s):
            continue
        lab = f"{lo:,}-{hi:,.0f} m" if hi < 1e9 else f">{lo:,} m"
        logger.info(f" {lab:>18}{len(s):>8}{s['survival'].mean():>15.3f}{s['survival'].median():>9.3f}")
    near = have[have["metres"] <= args.max_metres]["survival"]
    fars = have[have["metres"] > args.max_metres]["survival"]
    if len(near) and len(fars):
        logger.info(f"  <= {args.max_metres:.0f} m : mean survival {near.mean():.3f}  (n={len(near)})")
        logger.info(f"  >  {args.max_metres:.0f} m : mean survival {fars.mean():.3f}  (n={len(fars)})")
        logger.info(f"  difference {fars.mean() - near.mean():+.3f}")
        if fars.mean() < near.mean() - 0.03:
            logger.info("  => Distant 'same animal' pairs match WORSE. Consistent with them being")
            logger.info("     different animals mislabelled as one — which means part of the low spot")
            logger.info("     survival is a GROUND TRUTH fault, not a segmentation or model fault.")
            logger.info("     Confirm a few by eye, then decide whether to split those identities.")
        else:
            logger.info("  => Distance does NOT predict worse matching, so mislabelling on this scale is")
            logger.info("     not a major contributor to the survival rate. The far pairs still deserve")
            logger.info("     a look, but they are not the explanation for the 76%.")

    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "geo"
    outdir.mkdir(parents=True, exist_ok=True)
    have.to_csv(outdir / "pair_distances.csv", index=False)
    geo.to_csv(outdir / "photo_gps.csv", index=False)
    logger.info(f"wrote {outdir/'pair_distances.csv'}  ({len(have)} pairs with GPS on both sides)")
    logger.info(f"      {outdir/'photo_gps.csv'}")
    logger.warning(" NOTE: only 61% of photos carry GPS, so absence of a flag is not evidence of")
    logger.warning(" correctness — this can only ever confirm faults, never clear an individual.")


if __name__ == "__main__":
    main()
