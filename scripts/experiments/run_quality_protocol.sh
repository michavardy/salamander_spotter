#!/usr/bin/env bash
# The image-quality protocol: settle it, then apply it.  ~5-9 h, step 3 is nearly all of it.
#
#   bash scripts/experiments/run_quality_protocol.sh            # foreground
#   nohup bash scripts/experiments/run_quality_protocol.sh &    # detached; log path is printed at the end
#
# Background: the original sweep read identR@1 0.153 -> 0.464 as "filtering helps". Part real,
# part artefact -- an eval fold needs >=2 images per animal, so filtering deletes INDIVIDUALS and
# the gallery a query competes against shrank 105 -> 27 along the way. sweep_quality_control.py
# pins the gallery at a fixed 14 to separate the two, and reports bootstrap CIs over individuals
# rather than a std over two fold means (which ran ~10x too tight).
#
#   1. control sweep, filtered training   -- all levels, natural vs fixed gallery + difference tests
#   2. control sweep, TRAIN_POOL=full     -- gate the scored side only, keep every image to train on
#   3. all-9 at the settled protocol      -- MIN_QUALITY=0.2 + TRAIN_POOL=full, full epochs
#
# Steps are INDEPENDENT, so unlike run_ssl_ablation.sh they are not &&-chained: that script chains
# because its sweep consumes caches its earlier steps build, whereas here a transient failure in
# the 40-minute step 1 should not cost you the multi-hour step 3. Every step runs, exit codes are
# collected, and the script exits non-zero if any of them failed.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1   # scripts/experiments/ -> repo root; all paths below are repo-relative

LOG="artifacts/spot_transformer/quality/protocol_$(date +%m%d_%H%M).log"
mkdir -p "$(dirname "$LOG")"

STATUS=()
say() { echo ""; echo "===== [$(date +%H:%M:%S)] $*"; echo ""; }

# step <n> <label> <command...>
step() {
  local n="$1" label="$2"; shift 2
  say "$n/3  $label"
  local t0 rc; t0=$(date +%s)
  "$@"; rc=$?
  local mins=$(( ($(date +%s) - t0) / 60 ))
  if [ $rc -eq 0 ]; then STATUS+=("$n  ok         $label  (${mins}m)")
  else                   STATUS+=("$n  FAILED=$rc  $label  (${mins}m)"); fi
  return 0
}

{
  echo "quality protocol — started $(date)"

  # ---- 1. control sweep, filtered training (the historical data regime) --------------------
  # All five levels: none, 0.2, 0.3, 0.4, 0.5. 5 repeats x 2 folds x 5 gallery draws, 2000 cluster
  # bootstraps. The unfiltered level dominates the cost -- its natural-gallery reference scores
  # ~1700 queries against 105 candidates, tens of thousands of RANSAC feature builds -- which is
  # why that reference row is held to 2 repeats while the fixed-gallery experiment gets all 5.
  # This also refills the q0.5 fixed row the previous sweep could not build: that run predated
  # novel_count_for(), which now inverts census.make_openset_split's own rounding (14 known +
  # 7 novel = 21) instead of asking for 22 individuals from a fold that has 21.
  step 1 "control sweep, TRAIN_POOL=filtered (~20-60 min)" \
    pixi run python -u pipeline/spot_transformer/sweep_quality_control.py

  # ---- 2. the same, gating the SCORED side only -------------------------------------------
  # Filtering the image list gates training too: at MIN_QUALITY=0.4 that is 362 training images
  # instead of 1136. Keeping the failing photos as training data and gating only what is scored
  # measured better on R@1, AUROC, census F0.5 and balanced accuracy alike. Slower than step 1
  # precisely because the training pools are ~3x larger.
  step 2 "control sweep, TRAIN_POOL=full (~40-90 min)" \
    env TRAIN_POOL=full \
      pixi run python -u pipeline/spot_transformer/sweep_quality_control.py

  # ---- 3. all-9 under the settled protocol ------------------------------------------------
  # MIN_QUALITY=0.2, not 0.4: the gain saturates at 0.2 (identR@1 0.417 / 0.415 / 0.424 across
  # 0.2-0.4, flat within noise) while 0.4 costs a third of the evaluable individuals (126 -> 85)
  # for nothing measurable. TRAIN_POOL=full then trains on all 1867 images rather than 1273.
  # The slow one: 11 models x 5 folds, set nets at 250 epochs, e2e at 30, 60 negatives per query.
  step 3 "all-9 @ MIN_QUALITY=0.2 TRAIN_POOL=full (HOURS)" \
    env MIN_QUALITY=0.2 TRAIN_POOL=full \
      pixi run python -u pipeline/spot_transformer/sweep_all9.py

  say "SUMMARY"
  printf '  %s\n' "${STATUS[@]}"
  cat <<'EOF'

  control results : artifacts/spot_transformer/sweeps/quality_control/
                    RESULTS_gal14_filtered_r5f2d5_lvnone-0.2-0.3-0.4-0.5.md
                    RESULTS_gal14_full_r5f2d5_lvnone-0.2-0.3-0.4-0.5.md
  all-9 results   : artifacts/spot_transformer/sweeps/all9_q0.2/RESULTS_all9_trainfull.md

  Read the FIXED-gallery table and the DIFFERENCE table; 'natural' only documents the original
  artefact. A '*' in the difference table means that 95% CI excludes zero.

  CAVEAT: sweep_all9 still reports fold-mean +/- fold-std, which measures agreement between folds,
  NOT sampling uncertainty -- on the same quantity it ran ~10x tighter than the bootstrap CI.
  Treat step 3's error bars as decorative until that is ported over.
EOF

  # The exit test lives INSIDE the braces on purpose: `{ ...; } | tee` runs the block in a
  # subshell, so STATUS does not survive the pipe. With `set -o pipefail` this becomes the
  # pipeline's exit code, and the script's.
  printf '%s\n' "${STATUS[@]}" | grep -q FAILED && exit 1
  exit 0
} 2>&1 | tee "$LOG"
rc=$?

echo "log: $LOG"
exit $rc
