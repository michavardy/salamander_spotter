#!/usr/bin/env python3
"""emb-quality — review image quality and preview what a threshold would drop.

Scores every image (n_spots, blur = variance-of-Laplacian, largest-spot fraction) and reports
how many would be dropped from BOTH eval and training at the given thresholds. Nothing is
deleted — it writes a ranked CSV so you can eyeball the worst images before committing::

    pixi run emb-quality --dataset all_sasa_norm_2026_11_07                 # defaults, review only
    pixi run emb-quality --dataset all_sasa_norm_2026_11_07 --min-blur 100  # preview blur cutoff
    pixi run emb-quality --dataset all_sasa_norm_2026_11_07 --min-spots 5 --top 30

Then apply the SAME thresholds to training/eval:
    pixi run emb-bakeoff --dataset ... --min-spots 5 --min-blur 100 --models ...

Argument parsing only; logic lives in ``spot_embedding.data.quality``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

from spot_embedding import reconfigure_utf8  # noqa: E402
from spot_embedding._common import DEFAULT_DATASET, prepared_dir, dataset_name, resolve_dataset  # noqa: E402
from spot_embedding.data import load_spotsets  # noqa: E402
from spot_embedding.data.quality import QualityConfig, assess  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emb-quality", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--min-spots", type=int, default=3)
    p.add_argument("--min-blur", type=float, default=0.0, help="0 = blur filter off")
    p.add_argument("--max-largest-frac", type=float, default=0.95)
    p.add_argument("--no-blur", action="store_true", help="skip the (slower) blur computation")
    p.add_argument("--top", type=int, default=20, help="print this many worst images")
    return p


def _pct(vals, q):
    import numpy as np
    v = [x for x in vals if x == x]
    return float(np.percentile(v, q)) if v else float("nan")


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    args = build_parser().parse_args(argv)
    cfg = QualityConfig(min_spots=args.min_spots, min_blur=args.min_blur,
                        max_largest_frac=args.max_largest_frac, use_blur=not args.no_blur)
    spotsets = load_spotsets(args.dataset)
    rows = assess(spotsets, args.dataset, cfg)
    dropped = [r for r in rows if r["drop"]]

    print(f"dataset {args.dataset}: {len(rows)} non-empty images")
    if cfg.blur_on or not args.no_blur:
        blurs = [r["blur"] for r in rows]
        print(f"blur (var-Laplacian) percentiles: p5={_pct(blurs,5):.0f}  p25={_pct(blurs,25):.0f}  "
              f"p50={_pct(blurs,50):.0f}  p75={_pct(blurs,75):.0f}")
    nsp = sorted(r["n_spots"] for r in rows)
    print(f"n_spots: min={nsp[0]} p10={nsp[len(nsp)//10]} median={nsp[len(nsp)//2]} max={nsp[-1]}")
    print(f"\nWOULD DROP {len(dropped)} / {len(rows)} at "
          f"min_spots={args.min_spots}, min_blur={args.min_blur:g}, max_largest_frac={args.max_largest_frac:g}")

    from collections import Counter
    reason_counts = Counter(r for row in dropped for r in row["reasons"])
    for reason, n in reason_counts.most_common():
        print(f"    {n:4}  {reason}")

    print(f"\nworst {min(args.top, len(rows))} images:")
    print(f"  {'id':16}{'n_spots':>8}{'blur':>9}{'largest':>9}  reasons")
    for r in rows[:args.top]:
        print(f"  {r['id']:16}{r['n_spots']:>8}{r['blur']:>9.0f}{r['largest_frac']:>9.2f}  "
              f"{','.join(r['reasons']) or '-'}")

    out = prepared_dir(dataset_name(resolve_dataset(args.dataset))) / "quality.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label", "n_spots", "blur", "largest_frac", "drop", "reasons"])
        for r in rows:
            w.writerow([r["id"], r["label"], r["n_spots"], f"{r['blur']:.1f}",
                        f"{r['largest_frac']:.3f}", r["drop"], ";".join(r["reasons"])])
    print(f"\nfull ranked list -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
