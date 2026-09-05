#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# How many photos actually contain more than one salamander? — measure BEFORE building a detector.
#
# `ca_14_5` is two animals overlapping: the body mask fuses them, so its 49 "spots" belong to two
# individuals, which corrupts the matching AND the ground truth those spots are labelled under.
#
# The geometric detector for this failed completely. Calibrated on that one confirmed case,
# `solidity` at the 0th percentile looked decisive; checked against the photographs, two of its top
# three (af_2_1, ri_1_2) are single animals whose masks are non-convex because the limbs are splayed
# or leaves cover part of the body. Six candidate signals were tested against one positive and two
# negatives and none separated them. The quality table describes a fused blob and cannot see inside
# it.
#
# So this asks the vision model already used to draw the spots. One cheap call per photo.
#
#   bash scripts/experiments/run_multi_animal.sh              # dry-run: the plan + call estimate
#   bash scripts/experiments/run_multi_animal.sh sample       # 150 photos -- the prevalence estimate
#   bash scripts/experiments/run_multi_animal.sh all          # every real photo (BILLED)
#   bash scripts/experiments/run_multi_animal.sh report       # analyse the cache, no calls
#
# START WITH `sample`. 150 photos estimates prevalence to roughly +/-3%, and that number is what
# decides the response:
#
#     under ~3%   -> exclude the flagged frames and move on. No detector.
#     3-10%       -> exclude them, and check whether the affected individuals need re-labelling.
#     over ~10%   -> an instance-segmentation front end (YOLO / Mask R-CNN) is justified.
#
# The run also cross-checks the counts against spot survival: a photo holding two animals should
# match its own individual's other photos WORSE, since half its spots belong to somebody else. If
# the flagged photos match no worse, the fault is real but not worth building for -- and that is a
# result too, reached for the price of 150 calls instead of a detector.
#
# Every answer is cached (artifacts/spot_transformer/multi_animal/counts.json), so a run that dies
# on a quota resumes for free and nothing is re-billed.
# ---------------------------------------------------------------------------------------------
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1

MODE="${1:-dry-run}"
case "$MODE" in
  dry-run) ARGS="--dry-run" ;;
  sample)  ARGS="--limit ${LIMIT:-150}" ;;
  all)     ARGS="" ;;
  report)  ARGS="--report" ;;
  *) echo "unknown mode '$MODE' -- expected dry-run|sample|all|report" >&2; exit 2 ;;
esac

OUT="artifacts/spot_transformer"
LOG="$OUT/multi_animal_$(date +%m%d_%H%M)_${MODE}.log"
mkdir -p "$OUT/multi_animal"
PIXI="${PIXI:-pixi}"

{
  echo "===== [$(date +%H:%M:%S)] multi-animal prevalence -- mode=$MODE"
  echo ""
  if [ "$MODE" = "all" ]; then
    echo "  NOTE: this bills one vision call per uncached photo (~1,267 of them)."
    echo "        'sample' answers the prevalence question for a tenth of that."
    echo ""
  fi
  $PIXI run count-animals $ARGS 2>&1
  echo ""
  echo "===== [$(date +%H:%M:%S)] done"
  echo ""
  echo "  Outputs:"
  echo "    $OUT/multi_animal/counts.json        every answer, resumable"
  echo "    $OUT/multi_animal/animal_counts.csv  counts joined to spots/solidity/quality"
  echo "    $OUT/multi_animal/multi_animal.csv   the flagged photos, one id per line"
  echo ""
  echo "  Read the prevalence line first, then the 'does it matter?' block. A fault that does not"
  echo "  predict worse matching is not worth a pipeline stage however common it turns out to be."
  echo ""
  echo "  Spot-check a few flagged photos by eye before acting -- the model has not been validated"
  echo "  on this task either, and one confirmed example (ca_14_5) is not a validation set."
} 2>&1 | tee "$LOG"

echo "log: $LOG"
