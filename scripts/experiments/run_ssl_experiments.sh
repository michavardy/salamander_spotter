#!/usr/bin/env bash
# Everything outstanding on the self-supervised track, in one pass.
#
#   A. build the SSL variants that do not exist yet   (skips ones already cached)
#        aug0.5 / aug2.0   augmentation-STRENGTH ablation
#        ep100             longer cosine schedule
#   B. one 5-fold sweep over every SSL condition + the baselines
#
# Existing caches (simclr / random / corr) are reused, not rebuilt.
#
#   bash scripts/experiments/run_ssl_experiments.sh         # full: ~4-5 h  (ep100 alone is ~2.5 h)
#   FAST=1 bash scripts/experiments/run_ssl_experiments.sh  # skip ep100 and use QUICK folds: ~1.5 h
#   nohup bash scripts/experiments/run_ssl_experiments.sh &  # detached
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1   # scripts/experiments/ -> repo root; all paths below are repo-relative

SSL_DIR="artifacts/spot_transformer/ssl"
LOG="$SSL_DIR/experiments_$(date +%m%d_%H%M).log"
mkdir -p "$SSL_DIR"
say() { echo ""; echo "===== [$(date +%H:%M:%S)] $*"; echo ""; }

# build a pretraining cache only if it is missing, so the script is safely re-runnable
build() {  # build <tag> <env-assignments...>
  local tag="$1"; shift
  if [ -f "$SSL_DIR/all_sasa_norm_2026_23_07_${tag}_64d.pkl" ]; then
    echo "  [skip] $tag — already cached"
    return 0
  fi
  say "building SSL cache: $tag  ($*)"
  env "$@" SSL_TAG="$tag" pixi run python pipeline/spot_transformer/ssl_pretrain.py
}

{
  say "A. SSL variants"

  # augmentation strength. NOT a "more is better" dial: the encoder becomes invariant to whatever
  # the augmentations vary, so this trades away signal. Judge it on identR@1, never on NT-Xent —
  # pretext loss and downstream utility are known to diverge on this data.
  build "aug0.5" SSL_AUG=0.5 &&
  build "aug2.0" SSL_AUG=2.0 &&

  # Longer SCHEDULE, not just more epochs: the 30-epoch run ended fully converged (deltas 0.002,
  # LR at 0), so re-running it longer on the same cosine changes nothing. Stretching the cosine
  # over 100 is a genuinely different optimisation path. ~2.5 h — skipped under FAST.
  { [ -n "$FAST" ] && echo "  [skip] ep100 — FAST set" || build "ep100" SSL_EPOCHS=100; } &&

  say "B. sweep — coupling, positives, augmentation, and the baselines" &&
  ROWS="ssl_simclr,ssl_finetune,ssl_corr,ssl_corr_finetune,ssl_random,e2e_pretrained,logreg"
  [ -z "$FAST" ] && ROWS="$ROWS,ssl_aug0.5,ssl_aug2.0,ssl_ep100"
  env ${FAST:+QUICK=1} MIN_QUALITY=0.4 ONLY="$ROWS" \
    pixi run python pipeline/spot_transformer/sweep_all9.py &&

  say "DONE — table in artifacts/spot_transformer/sweeps/all9_q0.4/"
} 2>&1 | tee "$LOG"

echo "log: $LOG"
