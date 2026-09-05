#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# Does the geometry x quality INTERACTION earn its place?
#
# eval/feasibility.py Part C measured that geometric consistency separates same-from-different at
# 0.603 on high-quality photo pairs and 0.500 -- exactly chance -- on low-quality ones (gap +0.103).
# This runs the matcher with and without the three features that encode that, changing nothing else,
# and prints the two results side by side.
#
#   bash scripts/experiments/run_quality_interaction.sh          # QUICK, 2 folds, ~10 min
#   bash scripts/experiments/run_quality_interaction.sh full     # 5 folds -- the paper number
#
# What to read:
#   strict_logreg    the ONLY row that should move. It is the model using the new features.
#   strict_hand_pos  the control. It never touches the interaction, so if IT moves between the two
#                    arms, something leaked and the comparison is void.
#   geom_x_quality   the learned coefficient. A clear positive value IS the finding: the model
#                    learned to trust the constellation more on good photos. Near zero means the
#                    aggregator was already compensating some other way, and the honest response is
#                    to default QUALITY_INTERACTION=0 rather than carry three dead features.
#
# RESULT (2026-08-15, 2 folds): a null, and a diagnostic one. strict_logreg moved **-0.005** against
# a +/-0.07 fold spread; strict_hand_pos was identical in both arms, so the test was clean. The
# model DID learn the predicted pattern (geom_consistency -0.112 -> -0.222 with geom_x_quality
# **+0.163**) but it buys nothing, because geometry carries a NEGATIVE weight either way -- the
# aggregator has already concluded the current geometric feature is not evidence, and re-scaling
# trust in something already distrusted is a no-op. QUALITY_INTERACTION now defaults to 0.
#
# Re-run this AFTER the constellation work (triplet/angle invariants, spine-relative coordinates).
# The interaction only becomes meaningful once geometry earns a positive weight; the ordering is the
# reverse of how it was planned.
# ---------------------------------------------------------------------------------------------
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1

MODE="${1:-quick}"
case "$MODE" in
  quick) export QUICK=1; FOLDS="2 folds [QUICK -- plumbing check, not a result]" ;;
  full)  unset QUICK;    FOLDS="5 folds" ;;
  *) echo "unknown mode '$MODE' -- expected quick|full" >&2; exit 2 ;;
esac

OUT="artifacts/spot_transformer"
STAMP="$(date +%m%d_%H%M)"
LOG="$OUT/quality_interaction_${STAMP}_${MODE}.log"
ON_OUT="$OUT/.qi_on_${STAMP}.txt"
OFF_OUT="$OUT/.qi_off_${STAMP}.txt"
mkdir -p "$OUT"
PIXI="${PIXI:-pixi}"
MODELS="strict_logreg,strict_hand_pos"

say() { echo ""; echo "===== [$(date +%H:%M:%S)] $*"; echo ""; }

# Pull one model's summary row out of a run's console output: a line that is the model name
# followed by a number. The separating whitespace is OPTIONAL on purpose -- the table pads names to
# a fixed 15-column field and 'strict_hand_pos' is exactly 15 characters, so it comes out flush
# against its first number whenever the number needs no padding of its own.
row() {  # row <file> <model>
  grep -E "^[[:space:]]*$2[[:space:]]*[0-9]" "$1" | tail -1
}

{
  say "geometry x quality interaction -- A/B  ($FOLDS)"
  echo "  models: $MODELS"
  echo "  everything is identical between the arms except the three interaction features."

  say "ARM 1/2 -- QUALITY_INTERACTION=1  (geometry discounted by photo quality)"
  QUALITY_INTERACTION=1 ONLY="$MODELS" \
    $PIXI run python pipeline/spot_transformer/sweeps/compare_strict.py 2>&1 | tee "$ON_OUT"

  say "ARM 2/2 -- QUALITY_INTERACTION=0  (the previous behaviour)"
  QUALITY_INTERACTION=0 ONLY="$MODELS" \
    $PIXI run python pipeline/spot_transformer/sweeps/compare_strict.py 2>&1 | tee "$OFF_OUT"

  say "COMPARISON"
  echo "                          censusF0.5      R@1     R@5    R@10   AUROC  cov@P90"
  for m in strict_logreg strict_hand_pos; do
    on="$(row "$ON_OUT" "$m")"; off="$(row "$OFF_OUT" "$m")"
    printf '  %-18s ON   %s\n' "$m" "$(echo "$on"  | sed "s/^[[:space:]]*$m[[:space:]]*//")"
    printf '  %-18s OFF  %s\n' "$m" "$(echo "$off" | sed "s/^[[:space:]]*$m[[:space:]]*//")"
    echo ""
  done

  echo "  learned coefficients on the interaction (ON arm, last fold):"
  grep -E "geom_x_quality|ransac_x_quality|^\s+quality\s" "$ON_OUT" || \
    echo "    (none printed -- strict_logreg may not have run; check the ARM 1 output above)"

  say "done"
  echo "  results files:"
  ls -1 "$OUT"/sweeps/strict/RESULTS_strict*.md 2>/dev/null | tail -6
  echo ""
  echo "  strict_logreg is the row that should move; strict_hand_pos is the control that must not."
  if [ "$MODE" = "quick" ]; then
    echo "  MODE=quick: 2 folds. Confirms the plumbing and shows the direction, but re-run with"
    echo "  'full' before believing the size of any difference."
  fi
} 2>&1 | tee "$LOG"

rm -f "$ON_OUT" "$OFF_OUT"
echo "log: $LOG"
