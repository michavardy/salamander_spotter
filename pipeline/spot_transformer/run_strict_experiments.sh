#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# Strict / special-spot matcher experiments for spot_transformer.
# Results are written to  artifacts/spot_transformer/sweeps/strict/RESULTS_*.md
# and also streamed to the console (the e2e run prints a per-epoch train/test table).
#
# Run ONE command and pick a mode:
#
#   bash pipeline/spot_transformer/run_strict_experiments.sh smoke        # ~2-3 min, validates plumbing
#   bash pipeline/spot_transformer/run_strict_experiments.sh all          # the full set (long, ~40-60 min)
#
# or a single experiment:
#   bash pipeline/spot_transformer/run_strict_experiments.sh strict       # 5-fold strict comparison, no filter
#   bash pipeline/spot_transformer/run_strict_experiments.sh strict-q     # 5-fold strict comparison, MIN_QUALITY=0.4
#   bash pipeline/spot_transformer/run_strict_experiments.sh e2e          # train e2e_transformer, q0.4, per-epoch log
#   bash pipeline/spot_transformer/run_strict_experiments.sh e2e-nofilter # same, no quality filter (bigger test set)
#   bash pipeline/spot_transformer/run_strict_experiments.sh e2e-compare  # DOES LEARNING THE EMBEDDING HELP?
#                                    transformer (re-embeds) vs frozen (uses 62-dim as-is) vs no-gate ablation
#
# Every mode reports: census F0.5, R@1/R@5/R@10 (is the right animal in the top-k shortlist),
# open-set AUROC, and cov@P90 = the ABSTAIN operating point (fraction of photos answered while
# emitted matches stay >=90% precise; the rest it declines to guess).
#
# Env overrides pass straight through, e.g.:
#   EPOCHS=60 E2E_NEG=30 FOLD=2 GATE_LAMBDA=0.5 SIGMA_POS=0.12 ARCH=transformer \
#     bash pipeline/spot_transformer/run_strict_experiments.sh e2e
# ---------------------------------------------------------------------------------------------
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # -> .../salamander_spotter
cd "$REPO"
PIXI="/c/Users/michav/.pixi/bin/pixi.exe"
CS="pipeline/spot_transformer/sweeps/compare_strict.py"
E2E="pipeline/spot_transformer/sweeps/train_e2e_transformer.py"
E2EC="pipeline/spot_transformer/sweeps/compare_e2e_strict.py"

EPOCHS="${EPOCHS:-50}"; E2E_NEG="${E2E_NEG:-25}"; FOLD="${FOLD:-0}"
export EPOCHS E2E_NEG FOLD

run(){ echo; echo "======================================================================"; \
       echo "=== $*"; echo "======================================================================"; \
       eval "$@"; }

mode="${1:-usage}"
case "$mode" in
  smoke)
    run "QUICK=1 ONLY=raw_voting,strict_hand_pos $PIXI run python $CS"
    run "QUICK=1 $PIXI run python $E2E"
    run "QUICK=1 $PIXI run python $E2EC"
    ;;
  strict)       run "$PIXI run python $CS" ;;
  strict-q)     run "MIN_QUALITY=0.4 $PIXI run python $CS" ;;
  e2e)          run "MIN_QUALITY=0.4 $PIXI run python $E2E" ;;
  e2e-nofilter) run "$PIXI run python $E2E" ;;
  e2e-compare)  run "MIN_QUALITY=0.4 $PIXI run python $E2EC" ;;
  all)
    run "$PIXI run python $CS"
    run "MIN_QUALITY=0.4 $PIXI run python $CS"
    run "MIN_QUALITY=0.4 $PIXI run python $E2E"
    run "MIN_QUALITY=0.4 $PIXI run python $E2EC"
    ;;
  *)
    echo "usage: bash pipeline/spot_transformer/run_strict_experiments.sh {smoke|strict|strict-q|e2e|e2e-nofilter|e2e-compare|all}"
    exit 1
    ;;
esac

echo; echo "done -> artifacts/spot_transformer/sweeps/strict/"
