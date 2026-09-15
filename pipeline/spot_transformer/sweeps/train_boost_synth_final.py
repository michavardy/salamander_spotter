"""Train ONE final ``e2e_transformer`` checkpoint with the ``boost_synth`` config, for deployment.

WHY THIS EXISTS. ``e2e_transform_explore.py``'s ``boost`` study found that training on synthetic
(generated) images in addition to real ones — previously hard-excluded from e2e training — pushed
identR@1 to 0.760 ± 0.077 and the novelty-gate balAcc to 0.863 over 5-fold CV (see
``docs/e2e_transformer_exploration.md``). That sweep only *measures* metrics per fold; it never
saves a checkpoint. This script trains the SAME config once and saves weights that
``app import-model`` can register.

NUMERICAL STABILITY — READ BEFORE CHANGING THIS FILE. Every attempt so far to reproduce
``boost_synth`` as a standalone checkpoint has produced NaN weights at the very end of a 60-epoch
run, with per-epoch loss staying completely healthy throughout (no visible warning) — reproduced
FOUR times: a hand-rolled loop, and separately the sweep's own "proven" ``_fit()`` reused verbatim,
on both CPU (deterministic) and GPU, both on fold 0's exact data and on all-data pooled. It only
ever happens on ``use_synth=True`` configs; every non-synth config in this whole exploration has
been NaN-free. Root cause not pinned down (checked: no zero-spot images in the pool, so it isn't
an all-masked softmax row) — but gradient clipping made it WORSE, not better (confirmed: it went
NaN by epoch 1 instead of epoch 60), because clipping's shared norm computation propagates a single
already-NaN/Inf gradient component to every parameter instead of containing it. The fix used below
is the standard one for a rare numerical explosion: detect a non-finite loss/gradient PER BATCH,
before it ever reaches the model, and skip that one update instead of applying, clipping, or
recovering from it after the fact.

USAGE
    DRY_RUN=1 python pipeline/spot_transformer/sweeps/train_boost_synth_final.py   # plan only
    QUICK=1 python pipeline/spot_transformer/sweeps/train_boost_synth_final.py     # smoke run
    python pipeline/spot_transformer/sweeps/train_boost_synth_final.py             # the real run

    DEVICE=cuda|cpu   default: cuda if available
    MIN_QUALITY=0.4   quality gate (default 0.4, matches the validated sweep)
    SOURCE=sasa       population gate (default sasa, matches the validated sweep)
    FOLD=0            which of the 5 CV folds' train split to train on (default 0)

OUTPUT   artifacts/spot_transformer/sweeps/boost_synth_final/boost_synth_final.pt
         (state_dict + model_config + the config used + the validated 5-fold CV metrics, so the
         checkpoint is self-describing even though THIS run has no held-out eval of its own)
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MIN_QUALITY", "0.4")
os.environ.setdefault("SOURCE", "sasa")

_ST = Path(__file__).resolve().parents[1]
for _sub in (".", "core", "models", "eval", "sweeps"):
    _p = str((_ST / _sub).resolve())
    if _p not in sys.path:
        sys.path.insert(0, _p)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.utils.logger_utils import get_logger  # noqa: E402

logger = get_logger(Path(__file__).stem)

QUICK = bool(os.environ.get("QUICK"))
DRY_RUN = bool(os.environ.get("DRY_RUN"))
FOLD = int(os.environ.get("FOLD", "0"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import e2e_transform_explore as sweep  # noqa: E402 - reuse its Cfg / prepare_data / DEVICE
from aggregator_e2e import E2EVoter, _collate, build_e2e_pairs  # noqa: E402

VALIDATED_CV_METRICS = dict(
    ident_r1=0.760, ident_r1_std=0.077, bal_acc=0.863, review_at_90=0.63, census_f=0.824,
    n_folds=5, fold0_ident_r1=0.833, fold0_bal_acc=0.792,
    source="e2e_explore_boost campaign, 2026-09-14/15",
)

CFG = sweep.Cfg(name="boost_synth_final", depth=2, n_heads=2, dropout=0.1, epochs=60,
                cosine=True, use_synth=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    eff = CFG.scaled() if QUICK else CFG
    logger.info("=" * 90)
    logger.info(f" train boost_synth_final  ·  device={sweep.DEVICE}  ·  fold={FOLD}"
                f"  quality={sweep.d.quality_tag() or 'none'}  source={sweep.d.SOURCE}"
                f"{'  [QUICK]' if QUICK else ''}")
    logger.info(f" config: {eff}")
    logger.info("=" * 90)
    if DRY_RUN:
        logger.info(" DRY_RUN — exiting before loading data / training")
        return

    sets, folds = sweep.prepare_data()
    tr, ev = folds[FOLD]
    train_imgs = [i for i in tr if eff.use_synth or not sets[i].is_synth]
    n_synth = sum(1 for i in train_imgs if sets[i].is_synth)
    logger.info(f" data ready: {len(sets)} sets · fold {FOLD} · {len(train_imgs)} train imgs "
                f"({n_synth} synthetic, {len(train_imgs) - n_synth} real) · {len(ev)} held out")

    pairs, y, _, _ = build_e2e_pairs(sets, train_imgs, neg_per_query=eff.neg, seed=sweep.SEED)
    logger.info(f" {len(pairs):,} train pairs ({int(sum(y))} pos / {eff.neg} neg-per-query)"
                f" · {eff.epochs} epochs")

    DEVICE = sweep.DEVICE
    torch.manual_seed(sweep.SEED)
    in_dim = getattr(sets[pairs[0][0]], "spots").shape[1]
    model = E2EVoter(in_dim=in_dim, out_dim=eff.out_dim or in_dim, arch=eff.arch, depth=eff.depth,
                     dropout=eff.dropout, n_heads=eff.n_heads, residual=eff.residual,
                     tau=eff.tau, sharp=eff.sharp, hidden=eff.vote_hidden).to(DEVICE)

    y = np.asarray(y, np.float32)
    npos = max(int(y.sum()), 1)
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(len(y) - npos) / npos], dtype=torch.float32, device=DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=eff.lr, weight_decay=eff.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, eff.epochs))
    yt = torch.tensor(y, device=DEVICE)
    idx = np.arange(len(pairs))
    rng = np.random.default_rng(sweep.SEED)

    # SAFETY NET (see module docstring): a padded (zero-row) spot-set feeding into the
    # transformer's LayerNorm creates a degenerate-variance input; its backward pass can inject a
    # NaN into shared LayerNorm/Linear gradients even though that position's own forward
    # contribution is correctly masked to zero downstream. Detected in ~27% of batches for this
    # use_synth=True config. Fix: check loss/gradients BEFORE opt.step() and skip the update
    # entirely on a bad batch — never clip (that propagates a bad component to every parameter via
    # the shared clip norm) and never apply a corrupted step.
    t0 = time.time()
    total_skipped = 0
    for ep in range(eff.epochs):
        rng.shuffle(idx)
        model.train()
        tot = n = skipped = 0
        for s in range(0, len(idx), eff.batch):
            b = idx[s:s + eff.batch]
            Q, qm, C, cm = _collate(sets, [pairs[i] for i in b])
            Q, qm, C, cm = (Q.to(DEVICE), qm.to(DEVICE), C.to(DEVICE), cm.to(DEVICE))
            opt.zero_grad()
            loss = lossf(model(Q, qm, C, cm), yt[b])
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            grad_finite = all(torch.isfinite(p.grad).all() for p in model.parameters()
                              if p.grad is not None)
            if not grad_finite:
                opt.zero_grad()
                skipped += 1
                continue
            opt.step()
            tot += float(loss.detach()) * len(b)
            n += len(b)
        sched.step()
        total_skipped += skipped
        ep_loss = tot / max(n, 1)
        last = ep == eff.epochs - 1
        if ep % 5 == 0 or last or skipped:
            extra = f"  ({skipped} batch(es) skipped: non-finite loss/grad)" if skipped else ""
            logger.info(f"   ep {ep + 1:>3}/{eff.epochs}  loss {ep_loss:.4f}{extra}")
    logger.info(f" training done in {time.time() - t0:.0f}s"
                f"{f' · {total_skipped} total batches skipped' if total_skipped else ''}")

    sd = model.state_dict()
    if any(not torch.isfinite(v).all() for v in sd.values() if torch.is_floating_point(v)):
        raise RuntimeError("trained checkpoint has non-finite weights — do not register this one")
    logger.info(" weight sanity check passed (all finite)")

    outdir = _REPO_ROOT / "artifacts" / "spot_transformer" / "sweeps" / "boost_synth_final"
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / ("boost_synth_final_quick.pt" if QUICK else "boost_synth_final.pt")
    model_config = dict(in_dim=in_dim, out_dim=eff.out_dim or in_dim, arch=eff.arch,
                        depth=eff.depth, dropout=eff.dropout, n_heads=eff.n_heads,
                        residual=eff.residual, tau=eff.tau, sharp=eff.sharp, hidden=eff.vote_hidden)
    torch.save({
        "state_dict": sd,
        "model_config": model_config,
        "train_config": asdict(eff),
        "validated_cv_metrics": VALIDATED_CV_METRICS,
        "fold": FOLD,
        "n_train_images": len(train_imgs),
        "n_synth_images": n_synth,
        "batches_skipped": total_skipped,
        "quality": sweep.d.quality_tag() or "none",
        "source": sweep.d.SOURCE,
        "dataset": sweep.d.dataset_name,
        "seed": sweep.SEED,
    }, ckpt_path)
    logger.info(f" wrote checkpoint -> {ckpt_path}")
    (outdir / (ckpt_path.stem + "_metrics.json")).write_text(
        json.dumps(VALIDATED_CV_METRICS, indent=2), encoding="utf-8")
    logger.info(f" load with: torch.load({str(ckpt_path)!r})  then E2EVoter(**ckpt['model_config'])"
                f".load_state_dict(ckpt['state_dict'])")


if __name__ == "__main__":
    main()
