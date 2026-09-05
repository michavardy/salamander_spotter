#!/usr/bin/env bash
# Self-supervised pretraining ablation, end to end.
#
#   1. SimCLR with augmentation positives          (the baseline SSL condition)
#   2. the SAME encoder, never trained             (isolates architecture from pretraining)
#   3. SimCLR with correspondence-mined positives  (isolates the choice of positive pair)
#   4. all three vs the hand-engineered baseline, under one census protocol
#
# Every run shares the crop cache, so only the first pays the ~40k PNG decode.
# Steps are &&-chained: a failure stops the chain instead of running the sweep on stale caches.
#
#   bash scripts/experiments/run_ssl_ablation.sh                 # foreground
#   nohup bash scripts/experiments/run_ssl_ablation.sh &         # detached; tail the log below
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1   # scripts/experiments/ -> repo root; all paths below are repo-relative

LOG="artifacts/spot_transformer/ssl/ablation_$(date +%m%d_%H%M).log"
mkdir -p "$(dirname "$LOG")"
say() { echo ""; echo "===== [$(date +%H:%M:%S)] $*"; echo ""; }

{
  say "1/4  SimCLR — augmentation positives (~30-60 min, builds the crop cache)"
  pixi run python pipeline/spot_transformer/ssl_pretrain.py &&

  say "2/4  random-init control (~2 min)" &&
  SSL_RANDOM_INIT=1 SSL_TAG=random \
    pixi run python pipeline/spot_transformer/ssl_pretrain.py &&

  say "3/4  SimCLR — correspondence-mined positives (~30-60 min)" &&
  SSL_MODE=corr SSL_TAG=corr \
    pixi run python pipeline/spot_transformer/ssl_pretrain.py &&

  say "4/4  ablation sweep" &&
  MIN_QUALITY=0.4 QUICK=1 ONLY=ssl_simclr,ssl_random,ssl_corr,e2e_pretrained \
    pixi run python pipeline/spot_transformer/sweep_all9.py &&

  say "DONE — results in artifacts/spot_transformer/sweeps/all9_q0.4/"
} 2>&1 | tee "$LOG"

echo "log: $LOG"
