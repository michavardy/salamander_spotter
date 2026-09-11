import os

QUICK = bool(os.environ.get("QUICK"))
ONLY = {s.strip() for s in os.environ.get("ONLY", "").split(",") if s.strip()}
# Where the quality gate applies. 'filtered' (default, the historical behaviour) drops failing
# photos outright, which gates TRAINING too -- at MIN_QUALITY=0.4 that is ~3x fewer training
# images. 'full' keeps them as training data and gates only what is scored, which measured better
# on R@1, AUROC, census F0.5 and balanced accuracy alike (sweep_quality_control.py).
TRAIN_POOL = os.environ.get("TRAIN_POOL", "filtered")
# Which COLLECTION to score. ``all_sasa_norm`` is two merged populations -- the original sasa study
# (290 individuals, 87 eval-eligible) and the Haifa-KF field export (461, 237). Every number in
# results.md predates that merge and is a sasa-only figure, so ``SOURCE=sasa`` is the like-for-like
# comparison rather than an easier subset. The env var is read in ``data.py``; ``d.source_keep_mask``
# and ``d.source_tag()`` follow it. The gate always restricts the SCORED side (queries + gallery =
# the population the metric is about). ``SASA_TRAIN_ONLY=1`` additionally drops the other collection
# from the training pool (the literal "that data is gone" run) -- off by default because the models
# here are individual-agnostic and the extra photos may still teach something that transfers.
SASA_TRAIN_ONLY = bool(os.environ.get("SASA_TRAIN_ONLY"))
K_FOLDS = 2 if QUICK else 5
SEED = 0
NEG_PER_QUERY = 10 if QUICK else 60
N_BANDS = 16
NOVEL_FRAC = 0.35

# epochs per family (QUICK cuts them hard -- this is a smoke path, not a result)
EP_SET = 15 if QUICK else 250
EP_E2E = 3 if QUICK else 30
