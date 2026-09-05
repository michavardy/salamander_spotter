#!/usr/bin/env bash
# ============================================================================================
# Additive vs conjunctive spot matching — build, measure, sweep, review. One script.
#
#   concat  = [shape|position]     ->  cosine = (cos_shape + cos_pos)/2   ADDITIVE (original)
#   tensor  = shape (x) position   ->  cosine =  cos_shape * cos_pos      CONJUNCTIVE
#
# The tensor flow is the outer product of the two blocks. Because
# <a1(x)b1, a2(x)b2> = <a1,a2> * <b1,b2>, a spot only matches when shape AND position agree,
# and a penalty in either drags the whole score down — while still being ONE vector compared
# with ONE dot product, so every matcher downstream is unchanged.
#
# USAGE
#   bash scripts/run_bakeoff.sh smoke     # ~minutes. Run this FIRST: proves the wiring works.
#   bash scripts/run_bakeoff.sh full      # the real sweep. LONG (hours) — see the grid below.
#   bash scripts/run_bakeoff.sh penalty   # how much should UNMATCHED pattern cost? (lam/gamma)
#   bash scripts/run_bakeoff.sh veto      # how conjunctive? sweeps c_shape/c_pos only.
#   bash scripts/run_bakeoff.sh review    # re-open the web app on the most recent run.
#
# Each mode creates its own run dir and chains straight into the review app, so nothing has to
# be copy-pasted between steps:
#
#   artifacts/bakeoff/<dataset>/<mode>_<timestamp>/
#       metrics.jsonl    one row per (run, fold, EPOCH) — the curves
#       summary.csv      one row per (embedding, model, config)
#       RESULTS.md       the leaderboard
#       ranked_pairs/    per (flow x matcher), consumed by pair-review-gen --from-bakeoff
# ============================================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# pixi is not always on PATH in every shell on this machine; fall back to the usual install.
PIXI="$(command -v pixi || echo "$HOME/.pixi/bin/pixi")"
[ -x "$PIXI" ] || { echo "error: pixi not found (looked on PATH and in $HOME/.pixi/bin)" >&2; exit 1; }

DATASET="$(sed -n 's/^dataset_name = "\(.*\)"/\1/p' pipeline/spot_transformer/core/data.py)"
[ -n "$DATASET" ] || { echo "error: could not read dataset_name from core/data.py" >&2; exit 1; }

MODE="${1:-smoke}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="artifacts/bakeoff/${DATASET}/${MODE}_${STAMP}"

banner() { echo; echo "=============== $* ==============="; }

# Render the finished run in the web app. Generating the PNGs is the slow part, so it is a
# separate step from serving, and `review` mode re-enters here without recomputing anything.
open_review() {
  local dir="$1"
  banner "review: rendering pairs from $dir"
  "$PIXI" run pair-review-gen --from-bakeoff "$dir"
  banner "opening the review app  (ctrl-c to stop)"
  "$PIXI" run pair-review
}

case "$MODE" in

  smoke)
    banner "[1/3] building BOTH embedding flows (concat 62-dim, tensor 988-dim)"
    "$PIXI" run build-embeddings --combine both

    banner "[2/3] bakeoff — quick grid, 2 folds, every epoch logged"
    "$PIXI" run bakeoff --space quick --folds 2 --eval-every 1 --out "$RUN_DIR"

    banner "[3/3] results"
    head -25 "$RUN_DIR/RESULTS.md" || true
    open_review "$RUN_DIR"
    ;;

  full)
    banner "[1/4] building BOTH embedding flows"
    "$PIXI" run build-embeddings --combine both

    # Free, model-free gate BEFORE spending hours on training: if two photos of one animal do
    # not separate from two random animals under a flow, no matcher built on it can win. Run it
    # per flow so the comparison starts at the data, not at the model.
    banner "[2/4] label-consistency per flow (the data ceiling)"
    EMB_TABLE=spot_embeddings        "$PIXI" run label-consistency
    EMB_TABLE=spot_embeddings_tensor "$PIXI" run label-consistency

    banner "[3/4] full bakeoff — both flows x {raw,logreg,mlp,strict,strict_logreg,e2e_*} x grid, 5 folds"
    "$PIXI" run bakeoff --space full --folds 5 --eval-every 1 --out "$RUN_DIR"

    # The fold-only gallery is ~65 candidates; deployment is ~687. Re-measure the two cheap
    # matchers at population scale so the headline number is the honest one.
    banner "[4/4] full-gallery re-measure (~687 candidates/query) — raw + logreg + strict"
    "$PIXI" run bakeoff --space full --folds 5 --models raw,logreg,strict --full-gallery \
        --eval-every 5 --out "${RUN_DIR}_fullgallery"

    banner "results"
    head -30 "$RUN_DIR/RESULTS.md" || true
    echo; echo "full-gallery:"
    head -15 "${RUN_DIR}_fullgallery/RESULTS.md" || true
    open_review "$RUN_DIR"
    ;;

  penalty)
    # How hard should pattern that does NOT match count against a pair? Review said the matcher
    # is blind to it: unrelated animals keep a high score because their handful of plausible spot
    # matches is all the score looks at. `strict` charges for the rest -- distinctiveness-weighted,
    # gated on whether the other photo even showed that body region -- and lam/gamma say how much.
    # lam=0, gamma=0 is in the grid as the no-penalty control, so the leaderboard shows the effect
    # of the penalty on the SAME matching rather than against a differently-built baseline.
    banner "[1/3] embeddings (concat flow)"
    "$PIXI" run build-embeddings --combine concat

    banner "[2/3] raw + logreg (no penalty) vs strict + strict_logreg (penalized), 5 folds"
    "$PIXI" run bakeoff --space full --folds 5 --embeddings concat \
        --models raw,logreg,strict,strict_logreg --eval-every 5 --out "$RUN_DIR"

    banner "[3/3] results"
    head -30 "$RUN_DIR/RESULTS.md" || true
    open_review "$RUN_DIR"
    ;;

  veto)
    # How conjunctive should the match be? c=0 is maximum veto but lets two MISmatches multiply
    # into a positive score (a bilinear form cannot ReLU); large c switches the veto off
    # entirely. This finds the middle empirically instead of by argument.
    for C in 0.25 0.5 1.0 2.0; do
      banner "c_shape = c_pos = $C"
      "$PIXI" run build-embeddings --combine tensor --c-shape "$C" --c-pos "$C"
      EMB_TABLE=spot_embeddings_tensor "$PIXI" run label-consistency
      "$PIXI" run bakeoff --space quick --folds 3 --embeddings tensor \
          --models raw,logreg --out "${RUN_DIR}_c${C}"
    done
    banner "veto sweep leaderboards"
    for C in 0.25 0.5 1.0 2.0; do
      echo "--- c=$C ---"; sed -n '6,12p' "${RUN_DIR}_c${C}/RESULTS.md" || true
    done
    echo
    echo "NOTE: the tensor table is left built at c=2.0 (the last sweep step)."
    echo "      Rebuild at your chosen c before any further run, e.g.:"
    echo "        pixi run build-embeddings --combine tensor --c-shape 0.5 --c-pos 0.5"
    ;;

  review)
    LAST="$(ls -1dt artifacts/bakeoff/"$DATASET"/*/ 2>/dev/null | head -1 || true)"
    [ -n "$LAST" ] || { echo "error: no bakeoff runs under artifacts/bakeoff/$DATASET/" >&2; exit 1; }
    open_review "${LAST%/}"
    ;;

  *)
    echo "unknown mode '$MODE' — want: smoke | full | penalty | veto | review" >&2
    exit 2
    ;;
esac
