#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# Do the wins stack? — every measured gain, put together, under one protocol.
#
# Six weeks of experiments produced six things that each beat their own control. Every one of them
# was measured ALONE, against the baseline, which is correct single-variable practice and is also
# why nobody knows whether they ADD UP or OVERLAP:
#
#   descriptor   morphology block            +0.134 identR@1  (e2e)     measured with the plain scorer
#   scoring      distinctiveness weighting   +0.098 census F  (unfilt)  measured on the OLD descriptor
#   scoring      partial-match log-odds      built, never run to completion
#   combination  learned aggregation         +0.200 census F            measured on the OLD descriptor
#   evaluation   quality-gated scoring       +0.21  census F (controlled)
#   novelty      relative statistics (b')    matches or beats a trained classifier
#
# This script crosses the two axes that have never met — DESCRIPTOR x SCORING RULE — on one
# protocol, then finishes the two sweeps that were left half-run, then records the one number the
# repo has never written down (the deployment-sized gallery).
#
#   bash scripts/experiments/run_stack.sh quick    # 2 folds, plumbing check, ~20-30 min
#   bash scripts/experiments/run_stack.sh full     # the paper numbers, several hours
#   bash scripts/experiments/run_stack.sh 1        # stage 1 only (the headline), etc.
#
# Stages are independent and re-runnable; each writes its own RESULTS file, so a run that dies
# resumes by re-invoking the stage that failed. Env overrides pass through:
#   MIN_QUALITY=0.4  TRAIN_POOL=full  SIGMA_POS=0.12  WEIGHT_MODE=learned  SEED=0
# ---------------------------------------------------------------------------------------------
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1

MODE="${1:-full}"
case "$MODE" in
  quick) export QUICK=1; STAGES="1 2 3"; NOTE="[QUICK — plumbing check, not a result]" ;;
  full)  unset QUICK;    STAGES="1 2 3"; NOTE="" ;;
  1|2|3) unset QUICK;    STAGES="$MODE"; NOTE="" ;;
  *) echo "unknown mode '$MODE' — expected quick|full|1|2|3" >&2; exit 2 ;;
esac

OUT="artifacts/spot_transformer"
LOG="$OUT/stack_$(date +%m%d_%H%M)_${MODE}.log"
mkdir -p "$OUT"
PIXI="${PIXI:-pixi}"
say() { echo ""; echo "===== [$(date +%H:%M:%S)] $*"; echo ""; }

# The protocol every stage is scored under. Exported once, here, because arms measured under
# different gates are not comparable and comparing them is the only thing this script is for.
# MIN_QUALITY gates the SCORED side only — bad photos stay in the training pool, which is the
# setting sweep_quality_control measured as better on every metric at every threshold.
export MIN_QUALITY="${MIN_QUALITY:-0.4}"
export TRAIN_POOL="${TRAIN_POOL:-full}"

# The two descriptors that have both been measured end-to-end. h20/h20morph are stage 2's job.
BASE_EMB="spot_embeddings"
MORPH_EMB="spot_embeddings_morph"

# ------------------------------------------------------------------------------ preflight
# Four descriptor tables should already exist (run_representation.sh built them on 15 Aug). Check
# before spending hours: a missing table fails 40 minutes into a fold, not at the start.
preflight() {
  say "preflight — descriptor tables and review labels"
  $PIXI run python - <<'PY' || exit 3
import sys
sys.path.insert(0, "pipeline/spot_transformer/core")
import duckdb                                               # noqa: E402
import data as d                                            # noqa: E402
con = duckdb.connect(str(d.DB_PATH), read_only=True)
have = {r[0] for r in con.execute(
    "select table_name from information_schema.tables").fetchall()}
need = ["spot_embeddings", "spot_embeddings_morph",
        "spot_embeddings_h20", "spot_embeddings_h20morph"]
missing = [t for t in need if t not in have]
for t in need:
    n = con.execute(f"select count(*) from {t}").fetchone()[0] if t in have else 0
    print(f"  {t:28s} {n if t in have else 'MISSING':>8}")
print(f"  dataset {d.dataset_name}")
if missing:
    print("\n  build the missing ones first:")
    print("    pixi run build-embeddings --table spot_embeddings_morph     --morph")
    print("    pixi run build-embeddings --table spot_embeddings_h20       --harmonics 20")
    print("    pixi run build-embeddings --table spot_embeddings_h20morph  --harmonics 20 --morph")
    raise SystemExit(1)
PY
}

# ------------------------------------------- STAGE 1: the headline — descriptor x scoring rule
# One call per descriptor, all five scoring rules inside it, so the rules share folds, negatives
# and seed and the rows are directly comparable. compare_strict names its RESULTS file after the
# active EMB_TABLE, so the two arms sit side by side instead of overwriting each other.
#
#   raw_voting       equal-weight soft-chamfer — the free baseline every claim is read against
#   logreg           17 summary features -> logistic regression — the long-standing champion
#   strict_hand_pos  distinctiveness-weighted coverage x support, hand blend (rarity now 0)
#   strict_logreg    the same strict features, weights learned
#   llr              partial-match log-odds: a match is +0.44, a miss only -0.11, so absence is
#                    charged at what it is actually worth instead of at full price
stage1() {
  for emb in "$BASE_EMB" "$MORPH_EMB"; do
    say "STAGE 1 — scoring rules on EMB_TABLE=$emb  $NOTE"
    env EMB_TABLE="$emb" ONLY=raw_voting,logreg,strict_hand_pos,strict_logreg,llr \
      $PIXI run python pipeline/spot_transformer/sweeps/compare_strict.py 2>&1
  done
}

# ------------------------------------------- STAGE 2: finish the descriptor sweep
# ARM A of run_representation.sh got through baseline and morph, then stopped mid-fold on the
# 20-harmonic arm; h20morph never ran at all. Same two models as those completed arms so all four
# descriptor rows are comparable. The tables already exist — no rebuild, which also protects the
# results already measured against them.
stage2() {
  for emb in spot_embeddings_h20 spot_embeddings_h20morph; do
    say "STAGE 2 — census sweep on EMB_TABLE=$emb  (logreg + e2e_pretrained)  $NOTE"
    env EMB_TABLE="$emb" ONLY=logreg,e2e_pretrained \
      $PIXI run python pipeline/spot_transformer/sweeps/sweep_all9.py 2>&1
  done
}

# ------------------------------------------- STAGE 3: the deployment-honest gallery
# Every headline in the repo ranks against ~61 candidates. Deployment ranks against the whole
# population (~687). The command has existed for weeks and the number has never been recorded.
# Prints to stdout only — it lands in this run's log, nowhere else.
stage3() {
  for emb in "$BASE_EMB" "$MORPH_EMB"; do
    say "STAGE 3 — full-gallery aggregator on EMB_TABLE=$emb  (~687 candidates/query)  $NOTE"
    env EMB_TABLE="$emb" $PIXI run aggregator --full-gallery 2>&1
  done
}

# ------------------------------------------------------------------------------ driver
{
  say "stack experiment — mode=$MODE  MIN_QUALITY=$MIN_QUALITY  TRAIN_POOL=$TRAIN_POOL"
  preflight
  for s in $STAGES; do
    case "$s" in
      1) stage1 ;;
      2) stage2 ;;
      3) stage3 ;;
    esac
  done

  say "done"
  echo "  stage 1 -> $OUT/sweeps/strict/RESULTS_strict_q${MIN_QUALITY}.md            (baseline descriptor)"
  echo "             $OUT/sweeps/strict/RESULTS_strict_q${MIN_QUALITY}_morph.md      (+ morphology)"
  echo "  stage 2 -> $OUT/sweeps/all9_q${MIN_QUALITY}/RESULTS_all9_subset_trainfull_h20*.md"
  echo "  stage 3 -> this log only (the aggregator prints, it does not write a RESULTS file)"
  echo ""
  echo "  How to read stage 1 — the question is whether the gains ADD:"
  echo "    morph column beats baseline for EVERY rule   -> the descriptor is a free win; adopt it"
  echo "                                                    and re-baseline everything on it."
  echo "    morph helps raw_voting but not strict_*      -> they were finding the SAME signal;"
  echo "                                                    distinctiveness already encoded what the"
  echo "                                                    morphology block adds. Pick one, not both."
  echo "    llr beats strict_hand_pos on census F0.5     -> partial-match scoring works; make it"
  echo "                                                    the default rule."
  echo "    llr wins on cov@P90 only                     -> still the win that matters most: more"
  echo "                                                    photos answerable at 90% precision is"
  echo "                                                    exactly the shortlist deliverable."
  echo "    nothing moves outside +/-0.07                -> that is the fold spread; you are at the"
  echo "                                                    representation ceiling and the next"
  echo "                                                    lever is label hygiene, not scoring."
  echo ""
  echo "  Read cov@P90 before census F0.5. F0.5 rewards labelling everything 'new'."
} 2>&1 | tee "$LOG"

echo "log: $LOG"
