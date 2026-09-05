"""emb-selfcheck — the Phase-0 acceptance doctor.

One command that (1) prints the DB ground-truth vs. what the loader sees, (2) runs the
leakage guard, and (3) runs the oracle and dummy matchers through the harness, asserting
**oracle ≈ 1.0** and **dummy ≈ chance**. Prints PASS/FAIL and returns an exit code.

Nothing here is science — it is the tripwire that proves the harness is neither broken
(oracle would fail) nor leaking (dummy would beat chance).
"""
from __future__ import annotations

import sys
from pathlib import Path

from ._common import contours_db_path, resolve_dataset
from .data import (
    check_leakage,
    dataset_summary,
    load_mask,
    load_spotsets,
    make_folds,
    reconcile_raw,
)
from .eval import evaluate
from .models import DummyMatcher, OracleMatcher

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)


def _db_ground_truth(dataset: str) -> dict:
    """Counts straight from the DB — the loader must reproduce these."""
    import duckdb

    from collections import Counter

    db = contours_db_path(resolve_dataset(dataset))
    con = duckdb.connect(database=str(db), read_only=True)
    try:
        ids = [r[0] for r in con.execute("SELECT salamander_id FROM images").fetchall()]
        spots = con.execute("SELECT count(*) FROM spots").fetchone()[0]
        zero = con.execute("SELECT count(*) FROM images WHERE n_spots=0").fetchone()[0]
    finally:
        con.close()
    per_label = Counter(i.rsplit("_", 1)[0] for i in ids)
    return {
        "images": len(ids),
        "labels": len(per_label),
        "singletons": sum(1 for v in per_label.values() if v == 1),
        "multi": sum(1 for v in per_label.values() if v >= 2),
        "empty_images": int(zero),
        "total_spots": int(spots),
    }


def _row(name: str, ok: bool, detail: str = "") -> str:
    return f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else "")


def run(dataset: str, *, k: int = 5, seed: int = 0) -> int:
    logger.info(f"=== emb-selfcheck: {dataset} ===")
    checks: list[bool] = []

    # --- 1. loader vs DB ground truth ---
    gt = _db_ground_truth(dataset)
    spotsets = load_spotsets(dataset, use_cache=False)   # bypass cache: check the real read
    summ = dataset_summary(spotsets)

    logger.info("Ground truth (DB) vs loader:")
    logger.info(f"  {'quantity':<16}{'DB':>8}{'loader':>10}")
    match_all = True
    for key in ("images", "labels", "singletons", "multi", "empty_images", "total_spots"):
        same = gt[key] == summ[key]
        match_all &= same
        flag = "" if same else "  <-- MISMATCH"
        logger.info(f"  {key:<16}{gt[key]:>8}{summ[key]:>10}{flag}")
    checks.append(match_all)
    logger.info(_row("loader reproduces DB counts", match_all))

    # --- 2. raw/DB reconciliation ---
    recon = reconcile_raw(dataset, spotsets)
    raw_ok = recon["db_only"] == []      # every DB id must have a raw file
    checks.append(raw_ok)
    logger.info(_row("every DB image has a raw file", raw_ok,
               f"raw={recon['raw_files']}, db={recon['db_images']}, "
               f"raw-only={recon['raw_only']}"))

    # --- 3. spot-count sanity (ca_5 -> 8) + mask decode ---
    from collections import Counter

    per_label = Counter(ss.label for ss in spotsets)
    ca5_ok = per_label.get("ca_5") == 8
    checks.append(ca5_ok)
    logger.info(_row("ca_5 has 8 photos", ca5_ok, f"got {per_label.get('ca_5')}"))

    probe = next(ss for ss in spotsets if not ss.is_empty)
    mask = load_mask(dataset, probe.salamander_id, 0)
    unique_vals = set(int(v) for v in set(mask.ravel().tolist()))
    mask_ok = (
        mask.shape == (probe.height, probe.width)
        and unique_vals <= {0, 255}
        and int(mask.max()) == 255
    )
    checks.append(mask_ok)
    logger.info(_row("mask decodes to binary HxW, correct dims", mask_ok,
               f"{probe.salamander_id} spot0 -> {mask.shape}, dims=({probe.height},{probe.width})"))

    # --- 4. leakage guard ---
    folds = make_folds(spotsets, k=k, seed=seed)
    problems = check_leakage(folds, spotsets)
    checks.append(not problems)
    logger.info(_row("splits pass the leakage guard", not problems,
               "clean" if not problems else f"{len(problems)} problem(s): {problems[:2]}"))

    # every scored (multi) individual has gallery + query
    gallery_query_ok = all(len(f.gallery_ids) > 0 and len(f.query_closed_ids) > 0 for f in folds)
    checks.append(gallery_query_ok)
    logger.info(_row("every fold has gallery + closed queries", gallery_query_ok))

    # --- 5. oracle ~ perfect ---
    oracle = evaluate(OracleMatcher(), spotsets, folds)["aggregate"]
    oracle_ok = (
        oracle["rank1"] > 0.999
        and oracle["mAP"] > 0.999
        and oracle["verify_auc"] > 0.999
        and (oracle["openset_auroc"] != oracle["openset_auroc"] or oracle["openset_auroc"] > 0.999)
    )
    checks.append(oracle_ok)
    logger.info(_row("oracle scores ~1.0", oracle_ok,
               f"rank1={oracle['rank1']:.3f} mAP={oracle['mAP']:.3f} "
               f"auc={oracle['verify_auc']:.3f} open={oracle['openset_auroc']:.3f}"))

    # --- 6. dummy ~ chance + reproducible ---
    dummy_a = evaluate(DummyMatcher(seed=seed), spotsets, folds)["aggregate"]
    dummy_b = evaluate(DummyMatcher(seed=seed), spotsets, folds)["aggregate"]
    reproducible = all(
        abs(dummy_a[k2] - dummy_b[k2]) < 1e-12
        for k2 in ("rank1", "mAP", "verify_auc")
        if dummy_a[k2] == dummy_a[k2]
    )
    chance_like = dummy_a["rank1"] < 0.20 and 0.35 < dummy_a["verify_auc"] < 0.65
    checks.append(reproducible)
    checks.append(chance_like)
    logger.info(_row("dummy is reproducible (same seed)", reproducible))
    logger.info(_row("dummy is at chance", chance_like,
               f"rank1={dummy_a['rank1']:.3f} auc={dummy_a['verify_auc']:.3f}"))

    ok = all(checks)
    logger.info(f"=== {'PASS' if ok else 'FAIL'} — {sum(checks)}/{len(checks)} checks ===")
    return 0 if ok else 1
