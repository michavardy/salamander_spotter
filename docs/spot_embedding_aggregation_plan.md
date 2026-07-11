# Spot Embedding + Aggregation — Implementation Plan

An actionable, checklist-driven build plan for the matcher designed in
[per_spot_embedding_aggregation.md](per_spot_embedding_aggregation.md). It follows the phases and
decision gates of that document's §9, and it is wired into the **existing `salamander_spotter`
framework** — thin `scripts/*.py` CLIs delegating to a pure-logic `pipeline/<stage>/` package,
exposed as `pixi run` tasks, reading the packaged `datasets/<name>/db/contours.db`.

- **How to read this:** each task has `- [ ]` sub-tasks, an **Explanation**, a **Definition of
  done (DoD)**, and a **Test / evaluation** you can actually run.
- **Naming is a proposal** — package `spot_embedding`, tasks `emb-*` — rename freely.
- **Order matters:** phases gate each other. Do not start Phase _n_+1 before Phase _n_'s gate.

---

## Folder structure

New code sits beside the existing `generate_spot_labels` pipeline. Nothing here touches the
label-generation code; it only *reads* the packaged dataset.

```
salamander_spotter/
├── pixi.toml                         # + new [tasks]: emb-prepare / emb-train / emb-eval / emb-identify / emb-bakeoff
│
├── scripts/                          # thin CLIs only (argparse + config), same pattern as extract_spot_labels.py
│   ├── emb_prepare.py                # pixi run emb-prepare  — dataset → cached spot-sets + CV splits
│   ├── emb_train.py                  # pixi run emb-train    — train one candidate matcher → artifacts/runs/<id>/
│   ├── emb_eval.py                   # pixi run emb-eval     — run a matcher through the shared harness → metrics + report
│   ├── emb_identify.py               # pixi run emb-identify — one photo vs a gallery → match / new / abstain + confidence
│   └── emb_bakeoff.py                # pixi run emb-bakeoff  — all candidates through the harness → comparison.md
│
├── pipeline/
│   └── spot_embedding/               # pure logic, no argparse (mirrors generate_spot_labels/)
│       ├── __init__.py               # exports reconfigure_utf8, runner; package docstring
│       ├── _common.py                # stdlib-only: paths, artifact roots, seed, config load, derive_label (reused)
│       ├── runner.py                 # orchestrate prepare → train → embed → eval for one candidate
│       │
│       ├── data/
│       │   ├── spot_store.py         # read contours.db → per-image SpotSet (masks, centroids, areas, label)
│       │   ├── splits.py             # by-individual + by-session k-fold CV; leakage guard; gallery/query roles
│       │   └── dataset.py            # torch Dataset → positive pairs / triplets (real + augmentation-manufactured)
│       │
│       ├── augment/
│       │   ├── geometry.py           # spot-set aug: TPS/affine warp, dropout, spurious inject, jitter, per-spot rotate/stretch
│       │   ├── appearance.py         # pixel aug: blur, lighting/colour, background randomization (uses masks)
│       │   └── pipeline.py           # compose into a two-view sampler + curriculum; enforces "no mirror flips"
│       │
│       ├── encoders/
│       │   ├── spot_encoder.py       # small CNN on normalized spot mask → shape token; A/B orientation switch
│       │   ├── handfeatures.py       # Fourier descriptors, Hu moments, solidity, skeleton branch/endpoint counts
│       │   └── geometry_encoding.py  # relative-position features: kNN graph, edge (dist, rel-angle), attention bias
│       │
│       ├── models/                   # every model implements the same Matcher interface (base.py)
│       │   ├── base.py               # Matcher: embed(spotset)->vec  OR  score(a,b)->float ; capability flags
│       │   ├── set_transformer.py    # 4.1  ISAB + PMA over spot tokens
│       │   ├── gnn.py                # 4.2  GAT / message-passing over the kNN spot graph
│       │   ├── cnn_baseline.py       # 4.3  holistic masked-image CNN + metric-learning head
│       │   ├── hungarian.py          # 4.4  per-spot embed + optimal-assignment score (+ coarse-embed prefilter)
│       │   ├── classical.py          # 4.5  SIFT/ORB + RANSAC and constellation/triangle matching (non-DNN)
│       │   └── gemini_ref.py         # Gemini-as-matcher reference (cached, cost-capped)
│       │
│       ├── train/
│       │   ├── losses.py             # NT-Xent (SSL), triplet+hard-mining, SupCon, spot-level auxiliary
│       │   └── trainer.py            # training loop, curriculum, checkpointing, early stop
│       │
│       ├── match/
│       │   ├── gallery.py            # DuckDB vector store: enroll / query embeddings
│       │   ├── retrieve.py           # top-k search; coarse-embed prefilter for hungarian/classical
│       │   ├── verify.py             # geometric verifier (RANSAC over spot correspondences) → re-rank
│       │   ├── confidence.py         # quality + TTA-variance + NN-margin → calibrated confidence + abstain
│       │   └── identify.py           # end-to-end: photo → match / new / abstain + confidence
│       │
│       └── eval/
│           ├── metrics.py            # rank-k, mAP, ROC-AUC, TPR@FPR, open-set AUROC/DIR@FAR, risk-coverage
│           ├── harness.py            # run any Matcher over splits → metrics dict
│           └── report.py             # write metrics.json + report.md (+ risk_coverage.csv)
│
└── artifacts/                        # gitignored (add to .gitignore next to datasets/, images/)
    └── spot_embedding/
        ├── prepared/<dataset>/       # spotsets cache (.npz/duckdb) + splits.json + hand-features
        ├── runs/<run_id>/            # config.yaml · checkpoints/best.pt · gallery.duckdb · metrics.json · report.md
        └── bakeoff/comparison.md     # all candidates side-by-side
```

### `pixi.toml` additions

```toml
[tasks]
# --- spot-embedding matcher (see docs/spot_embedding_aggregation_plan.md) ---
emb-prepare  = "python scripts/emb_prepare.py"    # dataset -> cached spot-sets + CV splits
emb-train    = "python scripts/emb_train.py"      # train one candidate  --model <name> --config <yaml>
emb-eval     = "python scripts/emb_eval.py"       # evaluate a matcher    --model <name> --run <id>
emb-identify = "python scripts/emb_identify.py"   # query one photo       --gallery <> --image <>
emb-bakeoff  = "python scripts/emb_bakeoff.py"    # run all candidates + emit comparison.md

[dependencies]              # add on top of the existing ones
pytorch = "*"               # or pytorch-cpu — CPU-first per project constraint
torchvision = "*"
scikit-learn = "*"          # metrics, calibration, clustering (shape vocabulary)
scikit-image = "*"          # skeleton / region props for hand-features
scipy = "*"                 # linear_sum_assignment (Hungarian), spatial kNN
```

> Every CLI mirrors `extract_spot_labels.py`: `sys.path.insert(0, .../pipeline)`, import from
> `spot_embedding`, `reconfigure_utf8()`, `main(argv) -> int`, `raise SystemExit(main())`. All
> logic lives in the package; scripts only parse args.

---

## Phase 0 — Scaffolding & data harness  ·  **P0, blocks everything**

The bake-off is only fair if every candidate is judged by identical data and metrics. Build that
scaffolding first.

### 0.1 Package skeleton + pixi wiring
- [ ] Create `pipeline/spot_embedding/` with `__init__.py`, `_common.py`, `runner.py` and the
      sub-package dirs above (empty stage modules with docstrings).
- [ ] `_common.py`: artifact-root paths, `set_seed()`, a small YAML/JSON config loader, and reuse
      `derive_label()` (`aj_1_2 → aj_1`) and `resolve_input_dir()` from the existing `_common`.
- [ ] Add the five `emb-*` tasks + deps to `pixi.toml`; stub each script to print usage.

**Explanation.** Establish the module boundaries and the CLI/pixi surface before any modeling, so
later work just fills modules in.
**DoD.** `pixi run emb-prepare --help` (and the other four) print help; `python -c "import
spot_embedding"` succeeds.
**Test / eval.** Import smoke test in CI/local; each `--help` exits 0.

### 0.2 SpotSet data access
- [ ] `data/spot_store.py`: open `datasets/<name>/db/contours.db` (read-only), join `images` +
      `spots`, decode `mask_png`, expose a `SpotSet` (centroids, areas, per-spot masks, `label`).
- [ ] Cache to `artifacts/spot_embedding/prepared/<dataset>/` (`.npz` per image) so training doesn't
      re-hit DuckDB.

**Explanation.** One clean, fast in-memory representation of a photo's spots that every model reads.
**DoD.** Loads the packaged `all_sasa_norm_2026_10_07`; a `SpotSet` round-trips a decoded mask as an
`HxW` 0/255 array.
**Test / eval.** Assert dataset totals match the README (445 images, 290 labels); assert `ca_5`
resolves to 8 photos and `aj_1` ≠ `aj_2`; decode one `mask_png` and check it is binary and non-empty.

### 0.3 Splits (by-individual + by-session)
- [ ] `data/splits.py`: k-fold CV grouped by identity **label**; a second grouping by
      **source/session** (the `code` prefix, e.g. `ca`); assign gallery vs. query roles per fold.
- [ ] Leakage guard: no label appears in two folds; singletons go to train/gallery only (can't be a
      query positive).

**Explanation.** With only 87 multi-photo individuals, a single split is too noisy — CV is
mandatory, and session-grouping proves the model isn't cheating on shared backgrounds.
**DoD.** `splits.json` written; guard passes.
**Test / eval.** Assert train/test label sets are disjoint; assert every eval fold has ≥1 gallery
and ≥1 query photo for each multi-photo individual it contains.

### 0.4 Metrics library
- [ ] `eval/metrics.py`: rank-1 / rank-5 / mAP, ROC-AUC, TPR@FPR, open-set AUROC + DIR@FAR, and a
      risk–coverage curve.

**Explanation.** The single source of truth for scoring; must be trustworthy before any model uses
it.
**DoD.** Each metric is a pure function over a distance/score matrix + labels.
**Test / eval.** Unit tests on hand-built toy inputs with known answers (e.g. a 3×3 distance matrix
whose rank-1 you computed by hand); a perfect matcher scores 1.0, a random one ≈ chance.

### 0.5 Evaluation harness + Matcher interface + dummy
- [ ] `models/base.py`: `Matcher` interface — capability flag for `embed()` (→ vector) and/or
      `score(a, b)` (→ float); the harness adapts both into retrieval + verification metrics.
- [ ] `eval/harness.py` + `eval/report.py`: run any matcher over the splits → `metrics.json` +
      `report.md` (+ `risk_coverage.csv`) under `artifacts/.../runs/<id>/`.
- [ ] A `DummyMatcher` (random embeddings) to smoke-test the whole path.

**Explanation.** This is the rig the entire study runs on; the dummy proves it end-to-end.
**DoD.** `pixi run emb-eval --model dummy` produces a report with chance-level numbers,
reproducibly under a fixed seed.
**Test / eval.** Two dummy runs with the same seed produce identical metrics; a "perfect oracle"
matcher scores rank-1 = 1.0 through the harness.

> **GATE 0 →** metrics are reproducible on the dummy matcher. Only then build real matchers.

---

## Phase 1 — Baselines / floors  ·  **P0**

Cheap models that set the score every learned candidate must beat.

### 1.1 Classical matcher (4.5)
- [ ] `models/classical.py`: SIFT/ORB keypoints on the masked spot image + RANSAC geometric
      verification, and/or constellation/triangle matching on centroids. `score(a, b)` only.

**Explanation.** The non-DNN floor and a ready re-rank verifier; may be surprisingly strong.
**DoD.** Runs through the harness deterministically.
**Test / eval.** On a curated set, known positive pairs out-score known negatives; rank-1 > chance.

### 1.2 CNN holistic baseline (4.3)
- [ ] `models/cnn_baseline.py`: masked-image CNN + triplet/ArcFace head → `embed()`; trains via the
      shared `trainer.py`.

**Explanation.** The honest control — if a plain masked-image embedding ties the per-spot models,
the extra machinery isn't justified.
**DoD.** Trains to a decreasing loss; embeds; evaluated.
**Test / eval.** Overfit-tiny sanity (memorize 10 individuals → near-perfect train rank-1); test
rank-1 recorded vs. the 4.5 floor.

### 1.3 Gemini-as-matcher reference
- [ ] `models/gemini_ref.py`: prompt "same individual?" → score; **cache** responses; cost cap.

**Explanation.** The worst-case fallback and a difficulty reference for hard pairs.
**DoD.** Runs on a small, fixed subset within a cost budget; numbers logged.
**Test / eval.** Agreement with ground truth on a 20–50 pair curated set.

> **GATE 1 →** baseline/floor report exists; every later model is compared against it.

---

## Phase 2 — Spot encoder + orientation A/B  ·  **P1**

Build the per-spot front-end and settle the orientation question by experiment.

### 2.1 Spot normalization + hand-features
- [ ] `encoders/handfeatures.py`: Fourier descriptors, Hu moments, solidity, skeleton
      branch/endpoint counts.
- [ ] Normalize a spot mask to a canonical size; implement orientation **mode A** (as-is) and
      **mode B** (rotate to principal axis).

**Explanation.** Guarantees distinctive shapes (L/M/fork) are representable and provides the A/B
switch for §2a of the design doc.
**DoD.** Features computed for all spots in the dataset; A vs B produce visibly different crops.
**Test / eval.** Rotation-invariant hand-features stay ~constant when a spot is rotated 90°; an
`L`/forked mask yields higher branch/endpoint counts than a round blob.

### 2.2 Spot encoder CNN
- [ ] `encoders/spot_encoder.py`: small CNN → shape embedding; assemble the token
      `[shape ⊕ area ⊕ orientation ⊕ (optional) handfeatures]`.

**Explanation.** Turns each spot into the token the aggregators consume.
**DoD.** Fixed-dim token; both orientation modes and the hand-feature switch wired.
**Test / eval.** Forward-pass shape check; after brief training, the same spot under augmentation
embeds with high cosine similarity to itself.

### 2.3 Augmentation pipeline
- [ ] `augment/geometry.py`, `augment/appearance.py`, `augment/pipeline.py`: TPS/affine warp,
      **spot dropout + spurious inject**, jitter, per-spot rotate/stretch; blur, lighting,
      background randomization; two-view sampler + curriculum; **assert no mirror flips**.

**Explanation.** The training-signal generator — it manufactures positive pairs for the 203
singletons and drives the rotation/stretch/blur/missing-spot invariances.
**DoD.** Sampler yields two identity-preserving augmented views per spot-set.
**Test / eval.** Dump augmented pairs as PNGs to `artifacts/.../prepared/_aug_preview/` for eyeball;
assert spot count varies under dropout; assert no reflection is ever applied.

### 2.4 Orientation A/B experiment
- [ ] Train one fixed aggregator (Set Transformer) twice — encoder mode A vs. B — same data/seed;
      compare through the harness.

**Explanation.** Directly answers "did we need geometric normalization?" with same-everything-else
rigor.
**DoD.** Two runs + a comparison; the chosen front-end recorded in the run config.
**Test / eval.** Harness metrics for A vs B side-by-side; winner documented and frozen for Phase 3.

> **GATE 2 →** shape front-end chosen (A or B, hand-features on/off).

---

## Phase 3 — Aggregator bake-off  ·  **P1, the main study**

Same front-end, same harness; swap only the aggregator.

### 3.1 Shared losses & trainer
- [ ] `train/losses.py`: NT-Xent (SSL pretrain), triplet + hard-mining / SupCon (finetune),
      spot-level auxiliary loss.
- [ ] `train/trainer.py`: SSL-pretrain → supervised-finetune, curriculum, checkpoint best-on-val.

**Explanation.** Two-stage objective from the design doc; shared so model comparisons are fair.
**DoD.** Both stages run; best checkpoint saved.
**Test / eval.** Overfit-tiny sanity per loss; loss curves logged.

### 3.2 Set Transformer (4.1)
- [ ] `models/set_transformer.py`: ISAB self-attention + PMA pooling, relative-position attention
      bias → `embed()`.

**Explanation.** Global all-pairs reasoning over spots.
**DoD / Test.** Trains, converges, beats the Phase-1 floor; overfit-tiny passes.

### 3.3 GNN / GAT (4.2)
- [ ] `models/gnn.py`: kNN spot graph, edge `(dist, rel-angle)`, message passing → pooled `embed()`.

**Explanation.** The most literal encoding of relative geometry.
**DoD / Test.** Trains, converges, beats the floor; robust as spots are dropped at eval.

### 3.4 Hungarian matcher (4.4)
- [ ] `models/hungarian.py`: per-spot embeddings + `scipy.optimize.linear_sum_assignment`; dummy
      node for unassigned spots; coarse pooled embedding for top-k prefilter.

**Explanation.** Explicit set-matching — most interpretable, natural partial-match; the learned
cousin of 4.5.
**DoD / Test.** `score(a,b)` works; prefilter → Hungarian re-rank retrieval runs in acceptable time;
correspondences are inspectable; beats the floor.

### 3.5 Bake-off run + comparison
- [ ] `emb_bakeoff.py` runs 4.1/4.2/4.4 (+ Phase-1 baselines) through the harness →
      `artifacts/.../bakeoff/comparison.md`.

**Explanation.** The deliverable of the study: one table, identical conditions.
**DoD.** `comparison.md` with all candidates on all metrics.
**Test / eval.** Same-seed reproducibility; missing-spot stress column (metrics at 25/50% dropout).

> **GATE 3 →** primary matcher chosen (and whether it is embedding- or matching-based — this decides
> the retrieval path in Phase 4).

---

## Phase 4 — Confidence & open-set  ·  **P2**

### 4.1 Confidence signals
- [ ] `match/confidence.py`: input-quality (blur var-of-Laplacian, visible-spot count, OOD),
      TTA-variance, NN top-1/top-2 margin, distance-to-threshold, verifier inliers.

**Explanation.** The raw signals behind "how sure am I / can I even tell."
**DoD.** Each signal computed per query.
**Test / eval.** Blur/occlusion synthetic inputs score low quality; ambiguous pairs show small NN
margin.

### 4.2 Calibration + abstain + open-set threshold
- [ ] Fuse signals → calibrated confidence (logistic + temperature/Platt on held-out pairs);
      three-way output **match / new / abstain**; calibrate the match-vs-new threshold on CV.
- [ ] Emit the risk–coverage curve.

**Explanation.** Turns signals into the decision rule and the honest accuracy-vs-coverage report.
**DoD.** `identify()` returns one of three states + confidence; `risk_coverage.csv` written.
**Test / eval.** Held-out **novel individuals** are flagged *new* at the target FAR; abstain fires
on degraded inputs; risk–coverage is monotone (accuracy rises as coverage drops).

> **GATE 4 →** reliable "cannot evaluate" + a calibrated match/new threshold.

---

## Phase 5 — Verification re-rank  ·  **P2**

### 5.1 Geometric verifier
- [ ] `match/verify.py`: RANSAC over spot correspondences (reuse 4.5) re-ranks the winner's top-k.

**Explanation.** Precision lift for the final accept/reject.
**DoD.** Verifier plugs in behind retrieval as an ablation switch.
**Test / eval.** Verifier on vs. off: TPR@FPR and rank-1 improvement at fixed coverage.

---

## Phase 6 — Revisit extraction  ·  **P3, only if it becomes the bottleneck**

### 6.1 Deterministic extractor
- [ ] `pipeline/spot_embedding` consumes a non-Gemini extractor: classical threshold+CC, and/or a
      tiny U-Net/YOLO-Seg distilled from Gemini labels. (Extractor itself may live beside
      `generate_spot_labels`.)

**Explanation.** Only worth doing once matching works and Gemini's cost/latency/offline limits bite
(design doc §1).
**DoD.** Offline, deterministic, CPU extractor feeding the same `SpotSet`.
**Test / eval.** End-to-end ID metrics with the new extractor stay within tolerance of the
Gemini-labelled pipeline.

---

## Milestones recap

| Gate | Meaning | Blocks |
|------|---------|--------|
| 0 | Harness + metrics reproducible on a dummy | all modeling |
| 1 | Classical + CNN + Gemini floors established | fair comparison |
| 2 | Shape front-end chosen (orientation A/B) | aggregator study |
| 3 | Primary matcher chosen (4.1 / 4.2 / 4.4) | confidence + retrieval path |
| 4 | Calibrated confidence + open-set threshold | field-usable output |
| 5 | Verifier precision lift measured | final accept/reject |
| 6 | Deterministic extractor (if needed) | offline/CPU deployment |

**Guiding principles** (from the design doc): baselines before models; change one thing at a time
(harness → front-end → aggregator); ablate the switches (hand-features, geometry decoupled vs.
fused, local vs. global relations, verifier on/off); and **report abstention honestly** — the
risk–coverage curve, not a single accuracy number, picks the winner.
