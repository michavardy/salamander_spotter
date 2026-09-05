#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# Can we beat the hand-built 62-dim spot descriptor?
#
# Three arms, one baseline, one protocol. Everything is scored by the same census evaluation
# (sweep_all9 / compare_strict), split by individual, so the rows are directly comparable and the
# table is paper-ready rather than a pile of incomparable runs.
#
#   BASELINE   spot_embeddings          37-dim EFD shape + 25-dim body position   (the current 62)
#
#   ARM A  cheap descriptor variants — no training, minutes to build
#     spot_embeddings_morph      + size, irregularity, elongation, noncircularity  (110-dim)
#     spot_embeddings_h20        EFD cutoff 10 -> 20 harmonics                     (102-dim)
#     spot_embeddings_h20morph   both                                              (150-dim)
#
#   ARM B  learned descriptors — SimCLR over spot crops, hours to build
#     ssl_corr         positives = mined cross-photo correspondences  (the published recipe)
#     ssl_corr_audit   same, with the filters mining_audit showed cost data and buy no precision
#     ssl_corr_human   same, plus the human-verified correspondences  (~0% wrong positives)
#     ssl_random       the SAME encoder, never trained — the control that makes ARM B readable
#
#   ARM C  reference — the relational scorers, on the unchanged baseline descriptor.
#     representation_check found the per-spot descriptors already separate true correspondences
#     from random pairs at ~0.997 while predicting human REJECTIONS at chance, i.e. the missing
#     evidence is configurational, not per-spot. ARM C is here so any descriptor gain is read
#     against what changing the SCORING RULE buys on the same data.
#
#   bash scripts/experiments/run_representation.sh check    # ARM A build + fast screen, ~10 min
#   bash scripts/experiments/run_representation.sh a        # ARM A, full census        ~1-2 h
#   bash scripts/experiments/run_representation.sh b        # ARM B, build + census     ~4-8 h
#   bash scripts/experiments/run_representation.sh all      # everything                ~6-12 h
#
# Recommended first run is `check`: it builds ARM A and screens every descriptor against the 427
# human-judged correspondences in seconds, so you know what is worth the long runs.
#
# Env overrides pass through:  MIN_QUALITY=0.4 QUICK=1 SSL_EPOCHS=60 ...
# ---------------------------------------------------------------------------------------------
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1

MODE="${1:-check}"
case "$MODE" in check|a|b|c|all) ;; *)
  echo "unknown mode '$MODE' — expected check|a|b|c|all" >&2; exit 2 ;;
esac

OUT="artifacts/spot_transformer"
LOG="$OUT/representation_$(date +%m%d_%H%M)_${MODE}.log"
mkdir -p "$OUT"
PIXI="${PIXI:-pixi}"
say() { echo ""; echo "===== [$(date +%H:%M:%S)] $*"; echo ""; }

# The census protocol these runs are scored under. MIN_QUALITY gates the SCORED side only (see
# data.quality_keep_mask) -- exported once here so every arm is measured identically. Without a
# single setting the arms are not comparable, which is the only thing this script is for.
export MIN_QUALITY="${MIN_QUALITY:-0.4}"
export TRAIN_POOL="${TRAIN_POOL:-full}"

# ------------------------------------------------------------------- ARM A: build the variants
build_variants() {
  say "ARM A — building descriptor variants (no training)"
  # --table keeps each variant in its own DuckDB table, so EMB_TABLE swaps between them and
  # nothing has to be rebuilt to compare. The baseline table is never rewritten.
  $PIXI run build-embeddings --table spot_embeddings_morph     --morph                2>&1
  $PIXI run build-embeddings --table spot_embeddings_h20       --harmonics 20         2>&1
  $PIXI run build-embeddings --table spot_embeddings_h20morph  --harmonics 20 --morph 2>&1
}

# ------------------------------------------------------------------- the fast screen
screen() {
  say "fast screen — does each descriptor predict the human's accept/reject verdicts?"
  echo "  (seconds, no training. A screen, NOT a result: it is one edge-level axis on 427"
  echo "   hand-labelled pairs. Headline numbers come from the census runs below.)"
  local tags=""
  for t in simclr random corr corr_audit corr_human; do
    [ -f "$OUT/ssl/all_sasa_norm_2026_23_07_${t}_64d.pkl" ] && tags="${tags:+$tags,}$t"
  done
  if [ -n "$tags" ]; then
    $PIXI run repr-check --ssl-tags "$tags" 2>&1
  else
    $PIXI run repr-check 2>&1
  fi
}

# ------------------------------------------------------------------- ARM A: census
arm_a() {
  say "ARM A — census evaluation of each descriptor variant"
  for tbl in spot_embeddings spot_embeddings_morph spot_embeddings_h20 spot_embeddings_h20morph; do
    say "  EMB_TABLE=$tbl  (logreg champion + e2e_pretrained, 5 folds)"
    # emb_tag() puts the table name in the RESULTS filename, so these do not overwrite each other.
    env EMB_TABLE="$tbl" ONLY=logreg,e2e_pretrained \
      $PIXI run python pipeline/spot_transformer/sweeps/sweep_all9.py 2>&1
  done
}

# ------------------------------------------------------------------- ARM B: learned descriptors
arm_b() {
  say "ARM B — building learned descriptors (SimCLR over spot crops)"
  # Only builds what is missing; each cache is ~30-60 min on CPU. The random-init control is
  # cheap and non-negotiable: without it a learned row cannot be told apart from the architecture.
  build_ssl() {  # build_ssl <tag> <env...>
    local tag="$1"; shift
    if [ -f "$OUT/ssl/all_sasa_norm_2026_23_07_${tag}_64d.pkl" ]; then
      echo "  [skip] $tag — already cached"; return 0
    fi
    say "  building SSL cache: $tag  ($*)"
    env "$@" SSL_TAG="$tag" $PIXI run python pipeline/spot_transformer/models/ssl_pretrain.py 2>&1
  }

  build_ssl random     SSL_RANDOM_INIT=1
  build_ssl corr       SSL_MODE=corr
  build_ssl corr_audit SSL_MODE=corr SSL_GEOM=0 SSL_MIN_SIM=0.30
  build_ssl corr_human SSL_MODE=corr SSL_GEOM=0 SSL_MIN_SIM=0.30 SSL_HUMAN=1

  say "ARM B — census evaluation of the learned descriptors vs the hand-built baseline"
  env ONLY=e2e_pretrained,ssl_random,ssl_corr,ssl_corr_audit,ssl_corr_human \
    $PIXI run python pipeline/spot_transformer/sweeps/sweep_all9.py 2>&1
}

# ------------------------------------------------------------------- ARM C: relational reference
arm_c() {
  say "ARM C — reference: what changing the SCORING RULE buys on the unchanged descriptor"
  echo "  strict_hand_pos / strict_logreg already score configuration rather than appearance"
  echo "  (coverage x support, unexplained distinctive mass, position gate). This is the number"
  echo "  any descriptor improvement has to be read against."
  env ONLY=raw_voting,logreg,strict_hand,strict_hand_pos,strict_logreg \
    $PIXI run python pipeline/spot_transformer/sweeps/compare_strict.py 2>&1
}

# ------------------------------------------------------------------- driver
{
  say "representation experiment — mode=$MODE  MIN_QUALITY=$MIN_QUALITY TRAIN_POOL=$TRAIN_POOL"
  case "$MODE" in
    check) build_variants; screen ;;
    a)     build_variants; screen; arm_a ;;
    b)     arm_b; screen ;;
    c)     arm_c ;;
    all)   build_variants; screen; arm_a; arm_b; arm_c; screen ;;
  esac

  say "done"
  echo "  fast screen   -> $OUT/representation/RESULTS_representation_check.md"
  echo "  ARM A/B       -> $OUT/sweeps/all9_q${MIN_QUALITY}*/RESULTS_all9*.md  (one per descriptor)"
  echo "  ARM C         -> $OUT/sweeps/strict/RESULTS_strict*.md"
  echo ""
  echo "  For the paper, the comparable headline across every arm is census F0.5 with its"
  echo "  fold standard deviation, plus R@1/R@5 and cov@P90. Read the ssl_random row before"
  echo "  believing any learned row: it is the same architecture untrained."
} 2>&1 | tee "$LOG"

echo "log: $LOG"
