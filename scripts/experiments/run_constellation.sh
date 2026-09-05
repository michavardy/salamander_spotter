#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# Phase 0 of the constellation plan: is the relational evidence simply in the wrong frame?
#
# The matcher already computes constellation evidence -- `geom_consistency` (Spearman of pairwise
# distances) and `ransac_frac` (inliers of a fitted similarity transform), which the learned rule
# weights at +0.51. Both run on `global_centroid_x/y`: IMAGE PIXELS, under a RIGID transform. Pixel
# distances contract and expand when a salamander curls, which is the deformation the geometry is
# supposed to survive. Meanwhile the body-intrinsic frame the dataset already carries (`axis_t`,
# `axis_offset/length_px`) is only ever used one spot at a time, as the position gate in
# `strict_pair_score`. Nothing measures a relation BETWEEN two spots in the frame where relations
# are stable.
#
# Three measurements decide whether the rest of the plan is worth building:
#
#   M1  THE FRAME SWAP. The same features, computed in image pixels and in body coordinates on the
#       same photo pairs. The mechanism being bet on is conditioning as much as invariance: a 4-DOF
#       similarity fitted from 2 points leaves almost nothing over at the n=3-5 correspondences a
#       true pair actually has, whereas in body coordinates the expected transform is the identity,
#       so a 0- or 1-DOF model keeps every correspondence as evidence.
#       GATE: delta >= +0.05 AUROC with the cluster-bootstrap CI clear of zero -> recompute the
#       features in the body frame and confirm on sweep_all9. Flat -> the frame was not the problem
#       and the constellation work does not start here.
#
#   M2  UNDEFINED vs CONTRADICTORY. `match_features` initialises both geometry features to 0.0 and
#       leaves them there below 3 mutual matches, so "not computable" reaches the classifier as
#       "geometrically inconsistent". With median 21.5 spots and 24% survival that is not a corner
#       case. Also asks the honest counter-question: if undefined fires mostly on FALSE pairs, the
#       conflation is accidentally load-bearing and splitting it out must keep the count.
#       GATE: >20% of TRUE pairs affected -> add an explicit `geom_valid` column first.
#
#   M3  THE EDGE-LEVEL CEILING. `representation_check` measures the 62-dim cosine at 0.484 on the
#       417 human-judged correspondences (chance) against a 0.997 control, and concludes the
#       reviewer is judging configuration. This tests that conclusion instead of assuming it: give
#       each judged edge a RELATIONAL score -- does it agree with the other correspondences in its
#       own photo pair -- and see whether it recovers the verdict appearance cannot.
#       GATE: a relational score >= 0.60 confirms the hypothesis at the level of one decision.
#
# A shuffled-correspondence control runs beside every feature and must sit at ~0.50. A
# "constellation" feature that scores well with the correspondence permuted is reading spot
# spacing, not constellation -- read that column before anything else (results.md #18).
#
#   bash scripts/experiments/run_constellation.sh          # real photos only  (~2-4 min)
#   bash scripts/experiments/run_constellation.sh synth    # include Gemini views
#
# Real photos only is the default for the same reason as run_feasibility.sh: a generated view's
# geometry describes the generator, not a camera.
#
# Results -> artifacts/spot_transformer/constellation/
#   RESULTS_constellation_check.md   the three verdicts
#   pair_frames.csv                  one row per photo pair, every feature in both frames
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
LOG="$OUT/constellation_$(date +%m%d_%H%M)_${MODE}.log"
mkdir -p "$OUT"
PIXI="${PIXI:-pixi}"

{
  echo "===== [$(date +%H:%M:%S)] constellation phase 0 — mode=$MODE"
  echo ""
  $PIXI run constellation-check $EXTRA 2>&1
  echo ""
  echo "===== [$(date +%H:%M:%S)] done"
  echo ""
  echo "  How to read it:"
  echo "    CONTROL first  -> the shuffled column must be ~0.50. If it is not, stop: the feature"
  echo "                      is reading spot spacing rather than constellation."
  echo "    M1 delta >= +0.05 (CI clear of 0) -> the frame was the problem. Recompute"
  echo "                      geom_consistency / ransac_frac on (axis_t, axis_offset/length) and"
  echo "                      confirm on sweep_all9 before believing it."
  echo "    M2 undefined-on-true > 20%        -> add \`geom_valid\` before any new feature; the"
  echo "                      classifier is currently told that thinly-matched genuine pairs are"
  echo "                      geometrically contradictory."
  echo "    M3 relational AUROC >= 0.60       -> the human's rejections really are configurational"
  echo "                      and are recoverable from coordinates already in the DB."
} 2>&1 | tee "$LOG"

echo "log: $LOG"
