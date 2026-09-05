#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# Training regimes 1-5 on the hand labels from the preprocessing web app.
#
# Every regime reads artifacts/preprocessing/<dataset>/review.json through
# pipeline/spot_transformer/core/review_labels.py, so no two runs can disagree about which labels
# they used. Results land in artifacts/spot_transformer/{distinctiveness,gate_calibration,
# mining_audit,sweeps/strict}/ and are also streamed to the console + a log file.
#
#   bash scripts/experiments/run_review_regimes.sh audit    # R2+R3 only: ~10 min, no training
#   bash scripts/experiments/run_review_regimes.sh quick    # every regime at QUICK size: ~40 min
#   bash scripts/experiments/run_review_regimes.sh all      # the real thing: several hours
#
# or one regime at a time:
#   bash scripts/experiments/run_review_regimes.sh r1       # learned distinctiveness + ablation
#   bash scripts/experiments/run_review_regimes.sh r2       # edge-gate calibration
#   bash scripts/experiments/run_review_regimes.sh r3       # mining audit (+ SSL rebuild in `all`)
#   bash scripts/experiments/run_review_regimes.sh r4       # e2e-strict gate, fresh vs stale labels
#   bash scripts/experiments/run_review_regimes.sh r5       # human accept/reject as the eval gate
#
# R2 and R3 are pure measurement and finish in minutes. R1, R4 and R5 retrain and are the long
# ones. `all` runs R3's SSL rebuild (~30-60 min on CPU) before the sweep that uses it.
#
# Env overrides pass through, e.g.:
#   MIN_QUALITY=0.4 bash scripts/experiments/run_review_regimes.sh r1
# ---------------------------------------------------------------------------------------------
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1     # scripts/experiments/ -> repo root

MODE="${1:-quick}"
case "$MODE" in
  audit|quick|all|r1|r2|r3|r4|r5) ;;
  *) echo "unknown mode '$MODE' — expected audit|quick|all|r1|r2|r3|r4|r5" >&2; exit 2 ;;
esac

OUT="artifacts/spot_transformer"
LOG="$OUT/review_regimes_$(date +%m%d_%H%M)_${MODE}.log"
mkdir -p "$OUT"

PIXI="${PIXI:-pixi}"
say() { echo ""; echo "===== [$(date +%H:%M:%S)] $*"; echo ""; }

# QUICK shrinks folds/epochs everywhere. It validates plumbing; it is NOT a result, and every
# artifact written under QUICK is labelled as such so a smoke run cannot be mistaken for one.
case "$MODE" in
  quick) Q=1 ;;
  *)     Q=  ;;
esac
q_env() { if [ -n "$Q" ]; then echo "QUICK=1"; fi; }

# --------------------------------------------------------------------------------- R1
r1() {
  say "R1a  learned distinctiveness — is 'special' learnable? (+ legacy-vs-live label refresh)"
  $PIXI run distinctiveness 2>&1

  # The ablation the previous runs never had: every strict number in results.md was produced with
  # WEIGHT_MODE=learned and no baseline, so "distinctiveness-weighted" was never measured against
  # NOT weighting. uniform is the control; hand is the eyeballed blend it replaced.
  for wm in learned hand uniform; do
    say "R1b  compare_strict  WEIGHT_MODE=$wm"
    env $(q_env) WEIGHT_MODE="$wm" ONLY=strict_hand_pos,strict_logreg,logreg \
      $PIXI run python pipeline/spot_transformer/sweeps/compare_strict.py 2>&1
  done

  say "R1c  compare_strict  LABELS=legacy (the stale export) — does the refresh move the matcher?"
  env $(q_env) LABELS=legacy ONLY=strict_hand_pos,strict_logreg,logreg \
    $PIXI run python pipeline/spot_transformer/sweeps/compare_strict.py 2>&1
}

# --------------------------------------------------------------------------------- R2
r2() {
  say "R2a  edge-gate calibration — all 427 machine-proposed verdicts"
  $PIXI run gate-calibration 2>&1
  say "R2b  edge-gate calibration — real-real pairs only (n=59, the fully honest subset)"
  $PIXI run gate-calibration --pair-kind real-real 2>&1
}

# --------------------------------------------------------------------------------- R3
r3() {
  say "R3a  mining audit — precision of the SSL positives against the human verdicts"
  $PIXI run mining-audit 2>&1

  if [ -z "$Q" ] && [ "$MODE" = "all" ]; then
    # The audit's recommendation, built: same ~16% noise rate, 3-4x the pairs. Skipped in quick/r3
    # because it is a real pretraining run (~30-60 min CPU) and the audit alone answers whether it
    # is worth starting.
    say "R3b  rebuilding the SSL cache on the audit's settings (SSL_GEOM=0 SSL_MIN_SIM=0.30)"
    env SSL_MODE=corr SSL_GEOM=0 SSL_MIN_SIM=0.30 SSL_TAG=corr_audit \
      $PIXI run python pipeline/spot_transformer/models/ssl_pretrain.py 2>&1

    say "R3c  downstream: ssl_corr (current) vs ssl_corr_audit (re-mined) on the census protocol"
    env ONLY=ssl_corr,ssl_corr_audit,ssl_random,e2e_pretrained \
      $PIXI run python pipeline/spot_transformer/sweeps/sweep_all9.py 2>&1
  else
    echo "  [skip] R3b/R3c — SSL rebuild + downstream sweep only run in \`all\` mode."
    echo "         To run them alone:"
    echo "           SSL_MODE=corr SSL_GEOM=0 SSL_MIN_SIM=0.30 SSL_TAG=corr_audit \\"
    echo "             pixi run python pipeline/spot_transformer/models/ssl_pretrain.py"
    echo "           ONLY=ssl_corr,ssl_corr_audit,ssl_random,e2e_pretrained \\"
    echo "             pixi run python pipeline/spot_transformer/sweeps/sweep_all9.py"
  fi
}

# --------------------------------------------------------------------------------- R4
r4() {
  # results.md #31 measured the gate ablation on the STALE labels (0.609 -> 0.493 without the human
  # supervision). Re-running both label sources says whether the extra 92 images buy anything in
  # the one place the clicks have a measured causal effect.
  for lb in review legacy; do
    say "R4  compare_e2e_strict  LABELS=$lb  (e2e_strict_nogate is the no-supervision ablation)"
    env $(q_env) LABELS="$lb" \
      $PIXI run python pipeline/spot_transformer/sweeps/compare_e2e_strict.py 2>&1
  done
}

# --------------------------------------------------------------------------------- R5
r5() {
  # MIN_QUALITY=0.4 is the single largest effect in results.md (census F0.5 0.347 -> 0.642) and the
  # cutoff was never validated against a human. These three rows do that, with UNREVIEWED=keep so
  # the review gate is a clean SUBTRACTION of the 37 photos a human rejected.
  #
  # UNREVIEWED=drop (score only human-accepted photos) is deliberately NOT run: measured, it leaves
  # 3 eligible individuals and two folds with zero eval images, because only ~8% of the dataset was
  # reviewed. data.assert_evaluable now aborts on it rather than returning NaNs that read like a
  # result. It becomes available if the review ever covers enough of the dataset.
  for gate in quality review both; do
    say "R5  compare_strict  EVAL_GATE=$gate  (MIN_QUALITY=${MIN_QUALITY:-unset})"
    env $(q_env) EVAL_GATE="$gate" UNREVIEWED=keep \
      $PIXI run python pipeline/spot_transformer/sweeps/compare_strict.py 2>&1
  done
}

# --------------------------------------------------------------------------------- driver
{
  say "review regimes — mode=$MODE  dataset from data.dataset_name"
  $PIXI run review-labels 2>&1

  case "$MODE" in
    audit)        r2; r3 ;;
    r1)           r1 ;;
    r2)           r2 ;;
    r3)           r3 ;;
    r4)           r4 ;;
    r5)           r5 ;;
    quick|all)    r2; r3; r1; r4; r5 ;;
  esac

  say "done — results under $OUT/{distinctiveness,gate_calibration,mining_audit,sweeps/strict}/"
  if [ -n "$Q" ]; then
    echo "  MODE=quick: these are PLUMBING CHECKS, not results. Re-run with 'all' for numbers."
  fi
} 2>&1 | tee "$LOG"

echo "log: $LOG"
