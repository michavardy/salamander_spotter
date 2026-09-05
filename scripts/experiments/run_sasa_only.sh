#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# Re-run the matcher comparison on the SASA POPULATION ONLY.
#
# `datasets/all_sasa_norm_2026_23_07` is two populations that `scripts/dataset/merge_haifa.py`
# merged into one gallery:
#
#     sasa   290 individuals · 443 real photos ·  87 eval-eligible   -- the original study group
#     kf     461 individuals · 822 real photos · 237 eval-eligible   -- the Haifa/KF field export
#
# Every number in `results.md` (R@1 0.603, etc.) predates that merge and is a sasa-only figure.
# Scoring a sasa animal against 461 extra Haifa distractors measures a harder problem for free, so
# this script re-runs the three comparisons with `SOURCE=sasa`, which restricts BOTH the queries
# and the census gallery to the sasa individuals (`data.source_keep_mask`). The sasa/kf split comes
# from `images/all_sasa_norm/label_map.csv` -- kf individuals carry a `KF-<series>-<number>` in the
# `hebrew_name` column, every sasa individual an actual name.
#
#   bash scripts/experiments/run_sasa_only.sh quick     # 2 folds, few epochs -- plumbing check (~20 min)
#   bash scripts/experiments/run_sasa_only.sh full      # 5 folds, full epochs -- the real numbers (HOURS)
#   bash scripts/experiments/run_sasa_only.sh 1|2|3     # a single stage
#
# By default the Haifa photos stay in the TRAINING pool (the models here are individual-agnostic,
# so they may still teach something that transfers -- and whether they do is a measurement). Set
# TRAIN_GONE=1 for the literal "that data was never collected" run: Haifa is dropped from training
# as well, and results land in separate `_traingone` files.
#
# Env overrides pass through:  MIN_QUALITY=0.2  TRAIN_POOL=full  SEED=0  ONLY=logreg,raw_voting
# ---------------------------------------------------------------------------------------------
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1   # scripts/experiments/ -> repo root; paths below are repo-relative

MODE="${1:-full}"
case "$MODE" in
  quick) export QUICK=1; STAGES="1 2 3"; NOTE="[QUICK -- plumbing check, not a result]" ;;
  full)  unset QUICK;    STAGES="1 2 3"; NOTE="" ;;
  1|2|3) unset QUICK;    STAGES="$MODE"; NOTE="" ;;
  *) echo "unknown mode '$MODE' -- expected quick|full|1|2|3" >&2; exit 2 ;;
esac

# The source gate. SOURCE is read inside data.py; exporting it here puts every stage on the sasa
# population. TRAIN_GONE=1 also removes Haifa from training (SASA_TRAIN_ONLY, understood by
# sweep_all9 / compare_strict, and --sasa-train-only for confidence-bands).
export SOURCE=sasa
if [ -n "${TRAIN_GONE:-}" ]; then
  export SASA_TRAIN_ONLY=1
  CB_TRAIN_FLAG=(--sasa-train-only)
  TG_NOTE="  (Haifa also removed from training)"
else
  unset SASA_TRAIN_ONLY
  CB_TRAIN_FLAG=()
  TG_NOTE=""
fi

WORKERS="${WORKERS:-14}"
PIXI="${PIXI:-pixi}"
OUT="artifacts/spot_transformer"
LOG="$OUT/sasa_only_$(date +%m%d_%H%M)_${MODE}.log"
mkdir -p "$OUT"

STATUS=()
say() { echo ""; echo "===== [$(date +%H:%M:%S)] $*"; echo ""; }

# stage <n> <label> <command...>
stage() {
  local n="$1" label="$2"; shift 2
  say "$n/3  $label  $NOTE"
  local t0 rc; t0=$(date +%s)
  "$@"; rc=$?
  local mins=$(( ($(date +%s) - t0) / 60 ))
  if [ $rc -eq 0 ]; then STATUS+=("$n  ok         $label  (${mins}m)")
  else                   STATUS+=("$n  FAILED=$rc  $label  (${mins}m)"); fi
  return 0   # stages are independent -- a failure in one must not abort the rest
}

{
  echo "sasa-only comparison -- started $(date)"
  echo "  SOURCE=$SOURCE${TG_NOTE}   MIN_QUALITY=${MIN_QUALITY:-none}   TRAIN_POOL=${TRAIN_POOL:-filtered}   mode=$MODE"
  echo "  the sasa/kf split: $(grep -c 'KF-' images/all_sasa_norm/label_map.csv) kf individuals, "\
       "$(( $(tail -n +2 images/all_sasa_norm/label_map.csv | wc -l) - $(grep -c 'KF-' images/all_sasa_norm/label_map.csv) )) sasa"

  for s in $STAGES; do
    case "$s" in
      # ---- 1. the all-9 model census comparison -----------------------------------------------
      # formulation 1 (summary features -> classifier), 2 (match set -> net), 3 (encoder+voting),
      # the raw-voting baseline, and the SSL rows -- all on the same folds / census split / metrics.
      # Headline identR@1 + novelty AUROC. Writes RESULTS_all9<...>_sasa.md.
      1) stage 1 "all-9 model census (sweep_all9)" \
           $PIXI run python -u pipeline/spot_transformer/sweeps/sweep_all9.py ;;

      # ---- 2. strict vs equal-weight voting --------------------------------------------------
      # raw_voting / logreg / strict_hand / strict_hand_pos, with R@1/R@5/R@10 and cov@P90 (the
      # abstain operating point). Writes RESULTS_strict<...>_sasa.md.
      2) stage 2 "strict vs equal-weight (compare_strict)" \
           $PIXI run python -u pipeline/spot_transformer/sweeps/compare_strict.py ;;

      # ---- 3. accuracy@1/5/10 sliced by confidence band ------------------------------------
      # The triage table: which queries can be accepted unseen, which need a top-5 look, which are
      # nobody's time. --source sasa already gates queries + gallery here. ~15 min at 14 workers.
      3) stage 3 "confidence-band triage table (confidence-bands)" \
           $PIXI run confidence-bands --source sasa "${CB_TRAIN_FLAG[@]}" --workers "$WORKERS" ;;
    esac
  done

  say "SUMMARY"
  printf '  %s\n' "${STATUS[@]}"
  TG_SUFFIX=""; [ -n "${SASA_TRAIN_ONLY:-}" ] && TG_SUFFIX="_traingone"
  QSUFFIX=""; [ -n "${MIN_QUALITY:-}" ] && QSUFFIX="_q${MIN_QUALITY}"
  cat <<EOF

  results:
    stage 1  ->  $OUT/sweeps/all9${QSUFFIX}/RESULTS_all9*_sasa${TG_SUFFIX}.md
    stage 2  ->  $OUT/sweeps/strict/RESULTS_strict*_sasa${TG_SUFFIX}.md
    stage 3  ->  artifacts/confidence_bands/<dataset>/  (per_query.csv + the printed band table)

  Read stage 1's identR@1 against results.md's 0.603 and stage 2's R@1/R@5/R@10 against the
  clean-regime figures -- same population now, so the gap (if any) is the model, not the merge.
  For stage 2 read cov@P90 before census F0.5: F0.5 rewards labelling everything 'new'.
EOF

  printf '%s\n' "${STATUS[@]}" | grep -q FAILED && exit 1
  exit 0
} 2>&1 | tee "$LOG"
rc=$?

echo "log: $LOG"
exit $rc
