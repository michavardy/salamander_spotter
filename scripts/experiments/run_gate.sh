#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# The correspondence gate — the three lines every other experiment runs downstream of.
#
# Before any spot is weighted, scored or voted on, `strict_pair_score` decides WHICH query spot
# corresponds to WHICH candidate spot, in three lines that have never been ablated:
#
#     qbest = S.argmax(1); cbest = S.argmax(0)
#     mutual = [(i, qbest[i]) for i in range(nq) if cbest[qbest[i]] == i]   # must pick each other
#     if s < 0.4: continue                                                  # and clear 0.4
#
# Three assumptions live there — one partner per spot, the partner must reciprocate, and 0.4 —
# and six weeks of experiments (distinctiveness, llr, morphology, quality interaction, the
# constellation screen) all re-rank the OUTPUT of them. If the gate is throwing away correct
# correspondences, no downstream scoring rule can recover them, which is a candidate explanation
# for why every one of those experiments landed inside the same fold noise.
#
# This does NOT need labels. The 415 human-drawn "missed correspondences" cannot settle it —
# only 20 are real-real (296 are real-synth, 99 synth-synth), and on real photos alone the
# matcher's correspondence recall is <=0.710 (49/69), not the <=0.454 quoted in results.md #43.
# So the question is asked of the metrics instead, on the folds and seed already in use.
#
#   bash scripts/experiments/run_gate.sh quick   # 2 folds, plumbing check, ~30 min
#   bash scripts/experiments/run_gate.sh full    # 5 folds, the result, several hours
#   bash scripts/experiments/run_gate.sh full baseline hungarian    # named arms only
# ---------------------------------------------------------------------------------------------
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1

MODE="${1:-full}"
shift || true
case "$MODE" in
  quick) export QUICK=1; NOTE="[QUICK — plumbing check, not a result]" ;;
  full)  unset QUICK;    NOTE="" ;;
  *) echo "unknown mode '$MODE' — expected quick|full" >&2; exit 2 ;;
esac

OUT="artifacts/spot_transformer"
GDIR="$OUT/gate"
LOG="$OUT/gate_$(date +%m%d_%H%M)_${MODE}.log"
mkdir -p "$GDIR"
PIXI="${PIXI:-pixi}"
say() { echo ""; echo "===== [$(date +%H:%M:%S)] $*"; echo ""; }

# Same protocol as every other headline in the repo, so the rows are comparable to them and to
# each other. MIN_QUALITY gates the SCORED side only (A4: bad photos are bad queries, good
# training data).
export MIN_QUALITY="${MIN_QUALITY:-0.4}"
export TRAIN_POOL="${TRAIN_POOL:-full}"

# raw_voting is the CONTROL and it is the point of including it (B4: run an arm that must not
# move). It is soft-chamfer over aggregator.match_features and never calls strict_pair_score, so
# the gate cannot reach it. If raw_voting moves between arms, the harness is leaking and no other
# row in this run may be believed.
export ONLY="${ONLY:-raw_voting,strict_hand_pos,strict_logreg}"

#            name        GATE_ASSIGN  GATE_MATCH_THR   what it asks
ARMS=(
  "baseline    mutual      0.4"   # as shipped — the reference row
  "best        mutual      0.3"   # is 0.4 simply too high?
  "loose       mutual      0.2"   # how far does lowering it keep paying?
  "tight       mutual      0.5"   # or was it too LOW all along?
  "nomutual    best        0.4"   # drop reciprocity: every query spot takes its argmax
  "hungarian   hungarian   0.4"   # one-to-one: "explaining a spot costs a spot"
)

# ------------------------------------------------------------------ the decision rule, up front
# Written before any number exists. The fold spread on census F0.5 in this protocol is +/-0.06 to
# +/-0.09 (stack_0816_0035_full.log), so anything under ~0.07 is not a result.
preregister() {
  cat <<'RULE'

  PRE-REGISTERED DECISION RULE  (fixed before the run; B8)
  --------------------------------------------------------------------------------------
  Read cov@P90 first, then census F0.5. Compare every arm to `baseline`, on strict_hand_pos.

  1. An arm beats baseline by >= +0.07 census F0.5, or >= +0.05 cov@P90
       -> THE GATE WAS THE PROBLEM. Correct correspondences were being discarded by a
          constant nobody had tested. Adopt that arm as the default in strict_match.py and
          re-baseline every headline in results.md against it. This is the good outcome and
          it is a configuration change, not a model.

  2. Every arm sits within +/-0.07 of baseline
       -> THE GATE IS NOT THE CEILING. The correspondences the matcher can form are already
          all the ones it is going to find, and the missing evidence is in the 62-dim
          descriptor (#36, #44). Stop building scoring rules, ship the shortlist tool
          (R@10 0.93 at _q0.4), and spend the remaining effort on label hygiene: the 8
          confirmed duplicate identities and the multi-animal frames.

  3. `hungarian` wins but `nomutual` loses
       -> the problem is one-to-many explanation, exactly as strict_match's residual docstring
          predicts: a candidate spot explaining five query spots is how a NON-match keeps a
          high score. Adopt hungarian; the cost is O(n^3) on ~20 spots, i.e. nothing.

  4. `nomutual` wins on R@1/R@5 but LOSES on census F0.5
       -> the gate is trading recall for precision and currently sits on the wrong side for a
          shortlist product but the right side for a census. Keep both, switch by use case,
          and say so in the deliverable.

  5. raw_voting differs between arms by more than +/-0.001
       -> STOP. The control moved. The gate is reaching a path it must not reach; fix that
          before reading any other number here.

RULE
}

# ------------------------------------------------------------------------------------- driver
{
  say "correspondence gate ablation — mode=$MODE  MIN_QUALITY=$MIN_QUALITY  models=$ONLY  $NOTE"
  preregister

  WANT=("$@")
  for spec in "${ARMS[@]}"; do
    set -- $spec
    name="$1"; assign="$2"; thr="$3"
    if [ ${#WANT[@]} -gt 0 ] && ! printf '%s\n' "${WANT[@]}" | grep -qx "$name"; then
      continue
    fi
    say "ARM $name — assign=$assign match_thr=$thr  $NOTE"
    env GATE_ASSIGN="$assign" GATE_MATCH_THR="$thr" \
      $PIXI run python pipeline/spot_transformer/sweeps/compare_strict.py 2>&1 \
      | tee "$GDIR/arm_${name}.txt"
  done

  # ---------------------------------------------------------------------------- the comparison
  say "SUMMARY — census F0.5 / R@1 / cov@P90 by arm"
  printf '\n %-12s %-16s %8s %7s %7s %9s\n' arm model censusF R@1 R@10 cov@P90
  printf ' %s\n' "----------------------------------------------------------------------"
  for spec in "${ARMS[@]}"; do
    set -- $spec
    name="$1"
    f="$GDIR/arm_${name}.txt"
    [ -f "$f" ] || continue
    awk -v arm="$name" '
      /^ (raw_voting|strict_hand_pos|strict_logreg) / {
        # compare_strict prints: model censusF±sd R@1 R@5 R@10 AUROC cov@P90
        printf " %-12s %-16s %8s %7s %7s %9s\n", arm, $1, $2, $3, $5, $7
      }' "$f"
  done
  echo ""
  echo "  Columns are as compare_strict prints them: censusF is 'mean±sd' in one token."
  echo "  Re-read the PRE-REGISTERED DECISION RULE above before interpreting these."
  echo ""
  echo "  per-arm RESULTS files: $GDIR/../sweeps/strict/RESULTS_strict_q${MIN_QUALITY}_*.md"
  echo "  (the gate is in the filename — arms cannot overwrite each other)"
  say "done"
} 2>&1 | tee "$LOG"

echo "log: $LOG"
