#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# Step 1: is the constellation / comparability plan worth building?
#
# Three measurements, no training, no new labels — each one gates a later stage, so a negative
# here saves building machinery that cannot pay off:
#
#   PART A  Is a spot's disappearance predictable?
#           Half the spots already vanish between two photos of the SAME animal, so "a distinctive
#           spot is missing" is weak evidence by default. It only becomes usable if how SURPRISING
#           the disappearance is varies with something observable (size, distinctiveness, whether
#           that body region was visible, blur, curl). If it does not, absence cannot be weighted
#           and there is nothing to build.
#
#   PART B  Do curl / camera angle / image quality predict how well two photos of the same animal
#           actually match? A factor that does not move this cannot help downstream.
#           (border_frac is already out: <=0.038 everywhere, so bodies are never cut off by the
#           frame and it has no variance to contribute.)
#
#   PART C  THE DECIDING ONE. Is geometric consistency a better true/false discriminator when the
#           two photos are comparable than when they are not? That gap is the whole justification
#           for interaction terms (geometry x comparability). No gap => add the factors as plain
#           columns and skip the products, which would otherwise be parameters fitted to noise.
#
#   bash scripts/experiments/run_feasibility.sh            # real photos only  (~5-10 min)
#   bash scripts/experiments/run_feasibility.sh synth      # include Gemini views as well
#
# Real photos only is the default on purpose: a generated view's curl and blur describe the
# generator, not a camera, so including them would answer a question about Gemini rather than
# about photography.
#
# Results -> artifacts/spot_transformer/feasibility/
#   RESULTS_feasibility.md   the three verdicts
#   spot_survival.csv        one row per (spot, other photo) — for your own digging
#   pair_quality.csv         one row per photo pair with its comparability factors
# ---------------------------------------------------------------------------------------------
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1

MODE="${1:-real}"
case "$MODE" in
  real)  EXTRA="" ;;
  synth) EXTRA="--include-synth" ;;
  *) echo "unknown mode '$MODE' — expected real|synth" >&2; exit 2 ;;
esac

OUT="artifacts/spot_transformer"
LOG="$OUT/feasibility_$(date +%m%d_%H%M)_${MODE}.log"
mkdir -p "$OUT"
PIXI="${PIXI:-pixi}"

{
  echo "===== [$(date +%H:%M:%S)] feasibility — mode=$MODE"
  echo ""
  $PIXI run feasibility $EXTRA 2>&1
  echo ""
  echo "===== [$(date +%H:%M:%S)] done"
  echo ""
  echo "  How to read it:"
  echo "    PART A AUROC >= 0.65   -> absence can be weighted; build the survival-conditioned"
  echo "                              penalty. Below that, charging for missing spots adds noise."
  echo "    PART B |rho| >= 0.15   -> that factor really does degrade matching; condition on it."
  echo "    PART C |gap| >= 0.05   -> the interaction is real; build geometry x comparability"
  echo "                              PRODUCTS. Otherwise plain columns are all the data supports."
} 2>&1 | tee "$LOG"

echo "log: $LOG"
