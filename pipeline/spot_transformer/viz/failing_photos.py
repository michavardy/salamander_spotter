"""The failing pairs as PHOTOGRAPHS, side by side — because the contour plots hide the cause.

The contour renderer (``unmatched_examples.py``) draws outlines, and outlines are exactly what a
human cannot diagnose from: reviewing them produced two confident wrong readings (a "merged band
pattern" that was really a side-vs-top viewpoint difference, and a "sliver body mask" that was
really two salamanders in one frame). Both were obvious in one second from the photographs.

So this shows the photographs, with the diagnostic numbers in the caption, and deliberately selects
pairs that the known causes do NOT explain — good quality on both sides, similar curl and aspect,
plenty of spots on both, and still almost nothing matches. Those are the residual mysteries, and
they are where a new cause would show up.

    pixi run failing-photos                 # 6 residual-mystery pairs
    pixi run failing-photos --n 12
    pixi run failing-photos --mode worst    # simply the worst pairs, causes included
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval", "viz"):
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
IMAGE_DIRS = ["images/all_sasa_norm"]


def find_photo(sid: str) -> Path | None:
    for rel in IMAGE_DIRS:
        for ext in (".jpg", ".jpeg", ".png", ".JPG"):
            p = d.REPO_ROOT / rel / f"{sid}{ext}"
            if p.is_file():
                return p
    return None


def pair_stats() -> pd.DataFrame:
    """Same-animal photo pairs with survival and the conditions, for selection."""
    import duckdb
    con = duckdb.connect(str(d.DB_PATH), read_only=True)
    try:
        q = con.execute("SELECT salamander_id AS sid, curl_deg, aspect_ratio, solidity, "
                        "overall_quality, n_spots FROM image_quality").df().set_index("sid")
    finally:
        con.close()
    sets = [s for s in d.get_image_sets(d.get_spot_embeddings()) if not s.is_synth]
    by: dict[str, list] = {}
    for s in sets:
        if len(s.spots) >= 8:                      # both sides must have real pattern to compare
            by.setdefault(s.label, []).append(s)

    def m(sid, col, dflt=np.nan):
        try:
            v = q.at[sid, col]
        except KeyError:
            return dflt
        return float(v) if v is not None and np.isfinite(v) else dflt

    rows = []
    for lbl, ss in by.items():
        for i, A in enumerate(ss):
            for B in ss[i + 1:]:
                S = _norm(A.spots) @ _norm(B.spots).T
                ab, bb = S.argmax(1), S.argmax(0)
                n = sum(1 for k in range(len(S))
                        if bb[ab[k]] == k and float(S[k, ab[k]]) >= MATCH_THR)
                rows.append(dict(
                    label=lbl, sid_a=A.sid, sid_b=B.sid, n_a=len(A.spots), n_b=len(B.spots),
                    matched=n, survival=n / min(len(A.spots), len(B.spots)),
                    d_curl=abs(m(A.sid, "curl_deg", 0) - m(B.sid, "curl_deg", 0)),
                    d_aspect=abs(m(A.sid, "aspect_ratio", 0) - m(B.sid, "aspect_ratio", 0)),
                    q_min=min(m(A.sid, "overall_quality", 0), m(B.sid, "overall_quality", 0)),
                    sol_min=min(m(A.sid, "solidity", 1), m(B.sid, "solidity", 1)),
                ))
    return pd.DataFrame(rows)


def select(pairs: pd.DataFrame, mode: str, n: int) -> pd.DataFrame:
    """``residual`` = failures the known causes do not explain; ``worst`` = simply the worst."""
    if mode == "worst":
        return pairs.nsmallest(n, "survival")
    # Strip out everything already accounted for: poor photos, big posture/viewpoint gaps, and the
    # non-convex masks that flagged as possible multi-animal frames. Whatever still fails is new.
    ok = pairs[(pairs.q_min >= pairs.q_min.quantile(0.5))
               & (pairs.d_curl <= 3.0)
               & (pairs.d_aspect <= 1.5)
               & (pairs.sol_min >= pairs.sol_min.quantile(0.35))]
    return ok.nsmallest(n, "survival")


def render(row, outdir: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    pa, pb = find_photo(row.sid_a), find_photo(row.sid_b)
    if pa is None or pb is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, p, sid, ns in ((axes[0], pa, row.sid_a, row.n_a), (axes[1], pb, row.sid_b, row.n_b)):
        ax.imshow(mpimg.imread(p))
        ax.set_title(f"{sid}   ({ns} spots found)", fontsize=11)
        ax.axis("off")
    fig.suptitle(
        f"same animal '{row.label}' — only {row.matched} of {min(row.n_a, row.n_b)} spots matched "
        f"({row.survival:.0%})\n"
        f"Δcurl {row.d_curl:.1f}°   Δaspect {row.d_aspect:.2f}   "
        f"quality(worse of two) {row.q_min:.2f}   solidity(worse) {row.sol_min:.2f}",
        fontsize=12)
    out = outdir / f"{row.survival:.2f}__{row.sid_a}__{row.sid_b}.png"
    fig.tight_layout(); fig.savefig(out, dpi=95); plt.close(fig)
    return out


def main():
    import argparse
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--mode", default="residual", choices=["residual", "worst"])
    args = ap.parse_args()

    logger.info("scoring same-animal photo pairs ...")
    pairs = pair_stats()
    logger.info(f" {len(pairs)} pairs (both photos with >=8 spots) over "
          f"{pairs['label'].nunique()} individuals; median survival {pairs['survival'].median():.2f}")
    sel = select(pairs, args.mode, args.n)
    if args.mode == "residual":
        logger.info(f" residual mysteries: good quality both sides, Δcurl<=3°, Δaspect<=1.5, mask not "
              f"unusually\n non-convex — every known cause excluded, and they still fail.")

    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "failing_photos"
    outdir.mkdir(parents=True, exist_ok=True)
    for row in sel.itertuples(index=False):
        out = render(row, outdir)
        status = out.name if out else "PHOTO NOT FOUND"
        logger.info(f"  {row.sid_a:>10} vs {row.sid_b:<10} matched {row.matched:>2}/{min(row.n_a, row.n_b):<3}"
              f" ({row.survival:.0%})  Δcurl {row.d_curl:4.1f}  q {row.q_min:.2f}  -> {status}")
    pairs.to_csv(outdir / "pair_stats.csv", index=False)
    logger.info(f"wrote panels -> {outdir}")


if __name__ == "__main__":
    main()
