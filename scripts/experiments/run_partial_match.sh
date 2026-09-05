#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# Partial-match scoring: does weighing each spot by what it is WORTH beat explaining the pattern?
#
# Measured on this dataset:
#     P(a spot finds a match | SAME animal)      = 0.247
#     P(a spot finds a match | DIFFERENT animal) = 0.159
# so a match is +0.44 log-odds of evidence and a miss only -0.11 -- **one match is worth about four
# misses**. At 24.7% survival, "most of the pattern is missing" is what a TRUE pair looks like, so a
# score that charges for the missing 75% is charging for the normal case.
#
# `llr` (core/likelihood.py) scores a pair as the summed log-likelihood ratio over the query's
# spots, with both probabilities fitted as functions of the spot's size and whether its body region
# was visible -- so a big spot missing from a clearly-shown flank costs real evidence while a small
# one costs almost nothing. Fitted on TRAINING individuals only.
#
#   bash scripts/experiments/run_partial_match.sh          # QUICK, 2 folds, ~10 min
#   bash scripts/experiments/run_partial_match.sh full     # 5 folds -- the paper number
#
# Compared against the two rules it is meant to replace:
#   strict_hand_pos  coverage x support -- divides explained mass by the TOTAL pattern of both
#                    animals, so a perfect true match is structurally capped near the survival rate
#   strict_logreg    the same features, weights learned
#   logreg           the long-standing champion (equal-weight summary features)
#
# What to read: census F0.5 is the headline, but look hardest at **cov@P90** -- the fraction of
# photos answerable while staying 90% precise. That is the deliverable-shaped number, and a
# calibrated log-odds score is exactly what an abstention threshold wants. A win on cov@P90 with a
# flat census F0.5 would still be the more useful result.
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
LOG="$OUT/partial_match_$(date +%m%d_%H%M)_${MODE}.log"
mkdir -p "$OUT"
PIXI="${PIXI:-pixi}"

{
  echo "===== [$(date +%H:%M:%S)] partial-match scoring -- llr vs the existing rules ($FOLDS)"
  echo ""
  ONLY=llr,strict_hand_pos,strict_logreg,logreg \
    $PIXI run python pipeline/spot_transformer/sweeps/compare_strict.py 2>&1
  echo ""
  echo "===== [$(date +%H:%M:%S)] done"
  echo ""
  echo "  How to read it:"
  echo "    llr beats strict_hand_pos          -> partial-match scoring works; make it the default."
  echo "    llr ties                           -> the existing rules were already tolerating"
  echo "                                          absence well enough; keep the simpler one."
  echo "    llr wins on cov@P90 only           -> still a win, and the one that matters for a"
  echo "                                          shortlist product: more photos answerable at"
  echo "                                          90% precision."
  echo ""
  echo "  The 'llr -- what a match and a miss are WORTH' block prints the fitted base rates and"
  echo "  how size / visibility change them. If size carries weight under SAME but ~0 under"
  echo "  DIFFERENT, that asymmetry is what makes a big spot's absence informative -- it is the"
  echo "  mechanism the whole scorer rests on, so check it survived the fold split."
} 2>&1 | tee "$LOG"

echo "log: $LOG"
