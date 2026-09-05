# Salamander Spotter — Results

*Repo-level summary as of 2026-07-28.* What the project is trying to solve, what data exists, how
it is engineered, what was modelled, what the numbers actually are, and — the longest section —
**what every experiment taught**.

This supersedes [pipeline/spot_transformer/results.md](pipeline/spot_transformer/results.md), which
remains the detailed record of the aggregator track only. Design intent lives in
[docs/project_goal.md](docs/project_goal.md),
[docs/modeling_strategy.md](docs/modeling_strategy.md) and
[docs/per_spot_embedding_aggregation.md](docs/per_spot_embedding_aggregation.md).

---

## 1. The problem

Given a photograph of a fire salamander (*Salamandra salamandra*), decide **whether this animal is
already in the database** — and if so, which individual — using nothing but its natural yellow-spot
pattern. No tags, no capture.

This is **open-set re-identification**, not classification:

1. **Match** — is this the same individual as one already enrolled? Which one? (1-to-many search + verification)
2. **Enroll-as-new** — if nothing clears a calibrated threshold, register a new individual.
3. **Abstain** — "this photo is too blurred / obscured / weird, I can't tell" is a valid, wanted answer.

The end product is a **population census**: point a pipeline at a season's photos and get an
individual count, without a human adjudicating every pair.

**Why it is hard**
- Only the spot pattern is stable. Pose, body curvature, angle, lighting, wetness and background all vary.
- Long-tailed and few-shot: most individuals have 2–3 photos, many have one. A per-individual classifier is impossible.
- Labels are machine-extracted (Gemini + OpenCV), not hand-verified — the supervision itself carries noise.
- The set of individuals is open and grows; the model must handle animals it never trained on.

---

## 2. The datasets

Everything is packaged as `datasets/<name>/{raw/, db/contours.db, README.md}` and versioned by
build date (`all_sasa_norm_2026_23_07` = 23 July 2026). Seven dataset versions exist; the growth is
itself part of the record.

| dataset | images | individuals | singletons | spots | note |
|---|---|---|---|---|---|
| `..._2026_10_07` | 444 (445 raw) | 289 | 202 | 16,536 | Kibbutz Sasa only; Phase 0–3 harness numbers |
| `..._2026_11_07` | ~500 | — | — | — | + first Gemini synthetic views |
| `..._2026_16_07` | 733 (≈443 real) | 272 | — | 17,735 | aggregator track's headline run |
| `..._2026_19_07` | ~1,300 | — | — | — | + Amir + Haifa/KF merges; scaling + census sweeps |
| **`..._2026_23_07`** | **1,869** | **751** | **75** | **40,185** | current; 602 synthetic views, 1,616 positive pairs |

Current set (`2026_23_07`), the one all recent results use:

| metric | value |
|---|---|
| images | 1,869 (1,267 real + 602 synthetic) |
| individuals (labels) | 751 |
| singletons | 75 (427 before synthetic views) |
| multi-photo individuals | 676 |
| positive pairs | 1,616 (761 before synthetic) |
| spots | 40,185 · min/avg/max per image 1 / 21.5 / 83 |
| photos per individual | 75×1, 268×2, 387×3, 13×4, 6×5, 1×7, 1×8 |
| images with a body axis | 1,867 / 1,869 · 70 % of spots binned |

**Three real sources, one naming scheme.** Identity is encoded in the filename:
`<code>_<individual>_<instance>`, and the label is the stem minus the trailing instance
(`aj_1_2` → `aj_1`). Same label = same animal = a positive pair. That convention *is* the
supervision signal and the ground truth.

- **Sasa** — the original field collection (Kibbutz Sasa, Israel).
- **Amir** — a raw dump of Hebrew-named per-animal folders with Google-Takeout sidecars, flattened
  to ASCII `amr_<dir>_<idx>` by [scripts/dataset/rename_amir.py](scripts/dataset/rename_amir.py).
  The folder number *is* the label, so two folders yielding the same number is a hard error.
- **Haifa / KF** — a Roboflow COCO export whose original filenames carry the field identity
  (`KF_25-II-089.jpg` → site KF, year 2025, series II, individual 089). The **individual is
  (series, number)**; the **year is the capture occasion**, so an animal re-photographed in
  2023/24/25 is one individual with three photos —
  [scripts/dataset/merge_haifa.py](scripts/dataset/merge_haifa.py).

**Synthetic views.** 602 images are Gemini-generated re-renderings of an animal already in the set
(different lighting/background/angle, spot pattern held fixed), named `<label>_g<k>` so they derive
the same label. They add **855 positive pairs** and take **352 individuals out of singleton
status**. They are flagged `images.is_synthetic` and are **training-only** — evaluating on them
would measure the generator, not the matcher.

---

## 3. Data engineering and the database

One command builds a dataset end-to-end (`pixi run build-dataset`), in the only order the
dependencies allow: extract-real → package-interim → augment-singletons → extract-synth →
package-final. Every stage is **resumable** (nothing already on disk is billed twice) and every
event appends to one JSONL log — task, image, attempt, which model drew it, which judged it, the
judgment, the decision, and the billed-call count. `--report` rolls that up into calls-per-model
and most-failed criteria.

### The extraction pipeline

| stage | what | model? |
|---|---|---|
| 1. spots | Gemini repaints the yellow spots **flat magenta (#FF00FF)**; OpenCV keys the colour and traces each spot's contour, centroid, area, per-spot mask | Gemini image model |
| 1b. body | Gemini paints the **whole trunk-and-tail body** one flat colour (legs excluded), plus a green snout dot and a red tail dot | Gemini image model |
| 2. geometry | outline, left/right halves, **centre line**, true tips, and the 1..8 body bins — all derived from the mask in OpenCV | **no model** |
| 3. quality | ~25 cheap, model-free per-image markers + four 0..1 composites | **no model** |

The magenta trick is the whole reason labels are affordable: the model does a *masking* task it is
good at, and the output is re-conformed onto the original pixel grid so contours stay in source
coordinates. Output is PNG — lossless, so no JPEG artefacts on the flat key colour.

### The body grid

Every spot carries a **relative** positional key that survives rotation and scale:

- `axial_bin` **1..4** — which quarter of the body along the centre line's **arc length** (1 = head end).
- `lateral_bin` **left / right / overlap** — which side of the centre line, measured against the
  **body axis**, not the image axes.
- `bin` 1..8 = the product, **NULL when `overlap`** — a spot straddling the spine is in neither box.

`overlap` is why the spot's *outline* is used and not just its centroid: in one real photo **12 of
39 spots** straddled the midline, and every spot in the tail quarter did.

### Database schema (`db/contours.db`, DuckDB)

| table | contents |
|---|---|
| `images` | one row per photo: dimensions, `n_spots`, source/purple filenames, **full-frame body mask PNG**, `is_synthetic` |
| `spots` | one row per spot: centroid, area, centroid-relative contour, full-frame mask PNG, `axial_bin` / `lateral_bin` / `bin`, `axis_t` (0=head..1=tail), `axis_side`, `axis_offset` |
| `body_axis` | per photo: head/tail anchors, midline polyline, arc length, the two outline halves, `source` (`mask` / `mask_corrected` / `none`), `judged_ok` + `judge_feedback` |
| `body_bins` | the 8 boxes as polygons (visualisation only — a spot's bin is computed analytically from `(axis_t, axis_side)`, never point-in-polygon) |
| `image_quality` | raw markers (blur, exposure, glare, solidity, border, spots-outside-body, pattern contrast, curl…) + four 0..1 composites and `overall_quality` |
| `spot_embeddings` / `spot_embeddings_tensor` | the two embedding flows, built as separate tables so comparing is a table swap |

**Nothing is guessed.** No axis → no `body_axis` row and `bin IS NULL`; a failed quality gate is
recorded rather than hidden. Filtering is the consumer's job.

### Supporting tooling

- `correct-axis` — geometrically re-tips a bad axis **from the saved mask** (no model, no cost, reversible).
- `repurple` — re-sends only the poorly-scoring images with a flat-magenta/no-shadow prompt on a better model, and keeps the new result **only if it scores better**.
- `compute-quality` — the `image_quality` table.
- `interesting-spot-selector` — local web app: page through photos, click the spots that "draw the eye". Produced **~8.7k per-spot binary labels over 362 images**.
- `pair-review` / `pair-review-gen` — full-screen review of ranked match pairs with a free-text comment per pair. This is where the strict-matching design came from.
- `label-consistency` — the D1 diagnostic (§7).

---

## 4. Preprocessing / the representation

A spot becomes a **62-dim vector**, computed once and stored:

```
[ 37-dim shape | 25-dim position ]      each block L2-normalised
```

- **Shape (37-d)** — elliptic Fourier descriptors (10 harmonics) of the contour: translation-,
  scale-, rotation- and start-point-invariant, pixelation-tolerant. A bow-tie matches a rotated,
  resized, jagged bow-tie.
- **Position (25-d)** — **body-intrinsic**, never image `(x, y)`: sinusoidal features of `axis_t`
  (head→tail), the midline offset normalised by half the body width, and a left/right sign. Two
  identical shapes at different body locations land far apart.

Two ways the blocks meet, built as **separate tables** so they can be compared without rebuilding:

| flow | dim | cosine | semantics |
|---|---|---|---|
| `spot_embeddings` (concat) | 62 | `(cos_shape + cos_pos) / 2` | **additive** — a perfect shape in the wrong place (1.0, 0.0) ties two mediocre halves (0.5, 0.5); neither block can veto |
| `spot_embeddings_tensor` | 988 | `cos_shape · cos_pos` | **conjunctive** — outer product, so a spot matches only when shape **and** position agree |

Other preprocessing in the stack: small-spot filtering (bottom 10 % by area), optional
`image_quality` gating (`MIN_QUALITY`, `MAX_SPOTS_OUTSIDE`), spot-crop caches (32×32 masks) for the
CNN/SSL tracks, and a two-tier augmentation pipeline — constellation warps + **spot dropout and
spurious injection** + per-spot mask transforms — with a determinant guard that makes a **mirror
flip impossible** (a reflected pattern is a *different* identity).

---

## 5. Models tried

Three tracks, each answering a different question.

### Track A — the harness and the classic ladder (`pipeline/spot_embedding`, Phases 0–3)

`dummy` (chance floor) · `oracle` (perfect ceiling, proves the plumbing) · `classical`
(RANSAC constellation matching on centroids) · `spotdesc` (classical + hand shape descriptors,
orientation modes A/B) · `cnn` (frozen ImageNet ResNet18 on the background-free spot-union image) ·
`set_transformer` (ISAB + PMA over spot tokens, SupCon-trained per fold) · `gnn` (GCN over the kNN
spot graph) · `hungarian` (per-spot embeddings + optimal assignment) · `st_cnn` (trained per-spot
CNN → Set Transformer) · `gemini` (ask Gemini "same individual?", opt-in and billed).

### Track B — learned aggregation (`pipeline/spot_transformer`)

Keep the 62-dim embeddings **fixed** and learn the **voting rule** instead:

- **Formulation 1 — summary features → classifier.** 17 individual-agnostic match features
  (soft-chamfer sum/mean, max_sim, `frac≥.6/.7/.8/.9`, counts, mutual NN, `log_nq`/`log_nc`,
  geometric consistency, RANSAC inlier fraction) → logistic regression / MLP.
- **Formulation 2 — the raw match set → a permutation-invariant net.** DeepSets / set-attention /
  axial CNN over per-spot match records.
- **Formulation 3 — end-to-end.** A transformer re-embeds each spot *in the context of its image's
  other spots*, trained by backprop **through a differentiable voting rule** against the pairwise
  same/different label.
- Plus `pretrained_cnn` (frozen ResNet18 over 32×32 spot-mask crops) and `ssl_simclr` / `ssl_corr` /
  `ssl_random` (self-supervised spot encoders, §7).

### Track C — strict, distinctiveness-weighted matching

Built directly from the manual pair review. Equal-weight voting cannot express what a human sees, so
each spot gets a **distinctiveness** weight (size, elongation, non-circularity, irregularity, rarity,
isolation — either hand-blended or a logistic regression fitted to the human interesting-spot
clicks), and the score becomes:

```
score = coverage × support
coverage = weighted fraction of query distinctiveness explained by position-gated matches
support  = 1 − exp(−n_good / τ)          # collapses when only 1–3 spots corroborate
unexplained_q                            # distinctive query spots that matched nothing → novelty
```

`strict_hand` / `strict_hand_pos` (training-free), `strict logreg` (learned combination of the same
features), and `e2e_strict_{xf, frozen, nogate}` (the differentiable version, with the per-spot gate
supervised by the human clicks).

---

## 6. Current results

> **Read the protocol before the number.** The single biggest source of confusion in this project is
> that the same matcher scores 0.60 or 0.15 depending on **how many candidates it ranks against**,
> **which quality filter is on**, and **which dataset version** was used. Every table below states
> all three. Chance is `1 / n_candidates`.

### Track A ladder — 5-fold by individual, `all_sasa_norm_2026_11_07`

| model | rank-1 | rank-5 | mAP | verify AUC | open-set AUROC |
|---|---|---|---|---|---|
| dummy (floor) | 0.060 | 0.285 | 0.180 | 0.503 | 0.499 |
| classical | 0.169 | 0.474 | 0.306 | 0.565 | 0.569 |
| spotdesc A | 0.114 | 0.468 | 0.280 | 0.573 | 0.521 |
| gnn | 0.137 | 0.516 | 0.298 | 0.591 | 0.519 |
| hungarian | 0.167 | 0.498 | 0.293 | 0.563 | 0.501 |
| st_cnn | 0.194 | 0.460 | 0.279 | 0.597 | **0.651** |
| cnn (frozen ResNet18) | 0.235 | 0.564 | 0.326 | 0.573 | 0.578 |
| **set_transformer** | **0.266** | **0.585** | **0.366** | **0.638** | 0.542 |
| oracle (ceiling) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Every real matcher clears the dummy and sits far below the oracle. **GATE 3 was a split decision**:
the Set Transformer led verification/open-set, the frozen CNN led retrieval.

### Track B — learned aggregation, `all_sasa_norm_2026_16_07`, fold-only gallery (~65 candidates)

| scheme | learns | 5-fold R@1 | verdict |
|---|---|---|---|
| **summary features + logistic regression** | the voting rule | **0.603 ± 0.104** | ✅ best; +0.17 over raw voting, positive on all 5 folds |
| raw spot voting (soft-chamfer) | — | 0.434 ± 0.092 | strong free baseline |
| set-attention (DeepSets) | the voting rule | 0.431 ± 0.186 | tied, **unstable** (fold 3 collapsed to 0.087) |
| image-level SupCon transformer | image embedding | ~0.37 (fold 0) | ≈ mean-pool, no gain |
| image-level mean-pool | — | 0.352 (fold 0) | image baseline |
| spot-level MLP | spot embedding | ~0.33 (fold 0) | below raw voting |
| spot-level SupCon transformer | spot embedding | ~0.31 (fold 0) | below raw voting |

Learned weights (what the rule discovered): `frac≥.6` **+1.16**, `max_sim` **−0.68**, `log_nq`
−0.64, `softchamfer_sum` +0.61, `mutual_count` +0.52, `ransac_frac` +0.51, `qmax_std` +0.43.

### Track B/C — census protocol, `all_sasa_norm_2026_23_07`, 5 folds

Headline is **census F0.5**, not R@1 (§7). `_q0.4` = eval restricted to `overall_quality ≥ 0.4`.

**All-9 comparison, quality `_q0.4`** (5 folds, 60 neg/query):

| model | formulation | census F0.5 | P | R | count bias | R@1 (diag) |
|---|---|---|---|---|---|---|
| e2e_transformer | encoder + voting | **0.668 ± 0.098** | 0.929 | 0.323 | +24.6 | 0.423 |
| e2e_pretrained (frozen 62-d + head) | encoder + voting | 0.658 ± 0.098 | 0.913 | 0.318 | +24.6 | 0.418 |
| logreg | summary → clf | 0.649 ± 0.075 | 0.905 | 0.307 | +16.2 | 0.265 |
| mlp_deep | summary → clf | 0.587 ± 0.109 | 0.948 | 0.249 | +18.2 | 0.226 |
| set_transformer | match-set → net | 0.581 ± 0.091 | 0.856 | 0.257 | +17.4 | 0.232 |
| axial_cnn | match-set → net | 0.581 ± 0.083 | 0.883 | 0.247 | +17.8 | 0.286 |
| deepsets_cons | match-set → net | 0.542 ± 0.100 | 0.795 | 0.251 | +16.4 | 0.232 |
| raw_voting | none | 0.449 ± 0.134 | 0.565 | 0.250 | +13.8 | 0.211 |
| e2e_mlp · pretrained_cnn | encoder + voting | **nan** (recall 0) | — | 0.000 | +37.6 | 0.053 |

**Strict, distinctiveness-weighted matching** (`cov@P90` = fraction of photos answered while emitted
matches stay ≥90 % precise — the abstain operating point):

| quality | model | census F0.5 | R@1 | R@5 | R@10 | AUROC | cov@P90 |
|---|---|---|---|---|---|---|---|
| none | strict_hand_pos | **0.347 ± 0.124** | 0.192 | 0.282 | 0.362 | 0.602 | 0.059 |
| none | strict_hand | 0.315 ± 0.108 | 0.185 | 0.297 | 0.391 | 0.613 | 0.055 |
| none | logreg | 0.300 ± 0.153 | 0.183 | 0.342 | 0.466 | 0.601 | 0.055 |
| none | raw_voting | 0.249 ± 0.136 | 0.118 | 0.192 | 0.253 | 0.505 | 0.024 |
| `_q0.4` | strict_hand_pos | **0.642 ± 0.079** | 0.365 | 0.624 | 0.929 | 0.668 | 0.179 |
| `_q0.4` | logreg | 0.634 ± 0.084 | **0.489** | **0.756** | 0.935 | **0.688** | 0.179 |
| `_q0.4` | strict_hand | 0.631 ± 0.082 | 0.376 | 0.592 | 0.897 | 0.697 | 0.150 |
| `_q0.4` | raw_voting | 0.449 ± 0.134 | 0.319 | 0.442 | 0.764 | 0.515 | 0.038 |

**End-to-end strict vs. hand strict** (3 folds, 15 epochs, `_q0.4`; `gate` = how well the learned
per-spot gate recovers the human interesting-spot labels):

| model | census F0.5 | R@1 | R@5 | R@10 | AUROC | cov@P90 | gate |
|---|---|---|---|---|---|---|---|
| strict_hand_pos | **0.622 ± 0.045** | 0.397 | 0.501 | 0.740 | 0.664 | 0.180 | — |
| e2e_strict_xf | 0.609 ± 0.089 | 0.372 | 0.564 | 0.750 | 0.666 | 0.171 | 0.723 |
| e2e_strict_frozen | 0.600 ± 0.100 | 0.380 | 0.611 | 0.850 | 0.672 | 0.161 | 0.734 |
| **e2e_strict_nogate** (ablation) | **0.493 ± 0.183** | 0.293 | 0.459 | 0.588 | **0.516** | 0.128 | 0.518 |

### Embedding-flow bakeoff — additive vs. conjunctive (smoke run, 2 folds, eval-fold gallery, median 162 candidates/query)

| embedding | dim | model | R@1 | MRR | pair AUROC | margin AUROC |
|---|---|---|---|---|---|---|
| tensor | 988 | logreg | **0.145 ± 0.030** | 0.184 | 0.620 | 0.873 |
| concat | 62 | logreg | 0.140 ± 0.040 | 0.180 | 0.618 | 0.878 |
| tensor | 988 | mlp | 0.137 ± 0.029 | 0.175 | 0.635 | 0.538 |
| concat | 62 | mlp | 0.133 ± 0.033 | 0.170 | 0.616 | 0.510 |
| concat | 62 | e2e_frozen | 0.122 ± 0.032 | 0.157 | 0.523 | 0.869 |
| tensor | 988 | raw | 0.116 ± 0.031 | 0.142 | 0.492 | 0.898 |
| concat | 62 | raw | 0.111 ± 0.034 | 0.138 | 0.478 | 0.902 |
| concat / tensor | — | e2e_xf | 0.052 / 0.008 | — | 0.575 / 0.328 | 0.867 / 0.555 |

The full grid started 2026-07-27 and has **not finished** — `artifacts/bakeoff/.../full_20260727_230801/`
holds `metrics.jsonl` but no `summary.csv` / `RESULTS.md`. Conjunctive ≈ additive so far; the
difference is inside the noise.

### Where this leaves the system

- **As a shortlist tool for a human:** usable on good photos — R@5 0.62–0.76 and R@10 0.93 at `_q0.4`.
- **As an automated census:** precision-first thresholds reach P ≈ 0.91–0.95, but recall 0.25–0.32,
  i.e. **two-thirds of genuine re-sights are missed** and the population count runs ~16–25
  individuals per fold too high.
- **As an unattended labour saver:** not yet. `cov@P90` peaks at **0.18** — only ~18 % of photos can
  be answered while keeping emitted matches 90 % precise; `review@90` (fraction a human must check
  to reach 90 % end-to-end accuracy) sits at 68–100 %.
- **Novelty detection is the weak half:** open-set AUROC 0.55–0.70 against 0.5 for a coin flip.

---

## 7. What the tests taught

### About the data and labelling

1. **The magenta trick is the reason labels are affordable.** Asking the model to *recolour* spots
   flat magenta turns annotation into a masking task it is genuinely good at, and OpenCV does the
   rest deterministically. Same trick, reused for the body mask.
2. **Ask for a region, never a line.** The first body-axis design asked Gemini for two flank
   outlines and averaged them. Over a 56-image run the judge rejected **73 %** of drafts, and
   escalating the model did not help (**93 % / 87 % / 89 %** rejected across the three rungs). The
   model reliably finds the body *boundary* but will not decompose it into two side-curves. Asking
   for a filled mask and **deriving** outline, halves, centre line and tips in OpenCV removed three
   of the four failure modes **and the LLM judge entirely** — replaced by cheap geometric gates.
3. **The centre line must be equal-*area*, per slice.** A single straight head→tail chord runs
   outside a C-curled animal, through the gravel. Slice the mask along the body and take the
   **median** signed offset per band — the *mean* is the centroid and a filled leg drags it sideways.
4. **Use the outline, not the centroid, for left/right.** 12 of 39 spots straddled the midline in one
   photo, and the entire tail quarter did. Calling those `left` because a centroid landed a pixel
   that way would be a lie — hence the third value, `overlap`, and `bin IS NULL`.
5. **Bin by arc length, not by chord**, so a curled tail bins by distance *along the body*.
6. **Never guess a label.** No axis → no row; failed gate → recorded, not hidden. Every downstream
   consumer filters explicitly. This is why `judged_ok` and `spots_outside_frac` exist.
7. **Free repairs beat paid ones.** `correct-axis` re-tips a collapsed axis from the saved mask
   geometry — no model, no cost, reversible. `repurple` re-bills only the images that score poorly
   and **keeps the result only if it improves**.
8. **Resumability is a cost feature, not a convenience.** Every stage skips what is already on disk,
   so a run that dies on an API quota resumes for free. The JSONL run log records billed calls per
   model, which is what makes the ladder's cost auditable.
9. **Label noise is real and measurable.** The D1 diagnostic found **54 pairs of *different* labels
   whose photos agree as well as genuine repeats do** — probably one animal filed under two
   identities. These are invisible to every metric but poison them all: the model is scored *wrong*
   for finding the right animal under its other name.
10. **Extraction is unstable between repeat photos.** Across 324 multi-photo individuals, the median
    spot-count ratio between an individual's own photos is **0.786** (≈20 % of spots appear or
    vanish), 5.6 % of individuals lose more than half, and the median mutual-match fraction is 0.50.
    **39 % of individuals sit at or below the different-individual null median** — their own repeats
    agree no better than two random animals do. Roughly a third of the data cannot support matching
    at all, and no aggregator can fix that.
11. **Synthetic views buy pairs, not truth.** They took singletons from 427 → 75 and added 855
    positive pairs, but identity preservation is not guaranteed, so they are training-only and must
    pass a self-consistency gate (does the generated view still match its source's spot pattern?)
    before use.

### About measurement — the lessons that changed which numbers we believe

12. **Gallery size dominates everything.** The headline R@1 0.603 was measured against a
    **fold-only gallery (~65 candidates)**. Against the full population (~687) the same matcher
    scores a fraction of that. Never quote R@1 without candidates-per-query — chance itself moves
    from 0.010 to 0.071 across the protocols used here.
13. **A "quality filter improves the model" result was half an artefact.** Filtering shrinks the
    gallery, so R@1 rises partly because there are fewer wrong answers available. The controlled
    rerun pinned the gallery to **14 individuals** and the effect survived: **+0.15–0.20 R@1,
    +0.08–0.11 AUROC, +0.17–0.25 census F0.5, +0.06–0.09 balanced accuracy**. The 95 % bootstrap CI
    excludes zero in every threshold × metric cell with the full training pool, and in 15 of 16 with
    the filtered one (the exception: census F0.5 at `MIN_QUALITY=0.5`, where only 42 individuals
    survive). Photo quality genuinely matters — but only the controlled protocol proves it.
14. **Filter the *evaluation*, not the *training pool*.** `TRAIN_POOL=full` (train on everything,
    score only good photos) beats `TRAIN_POOL=filtered` on every metric at every threshold — at
    `MIN_QUALITY=0.4`, filtering the training set costs ~3× the training images. Bad photos are bad
    queries but perfectly good training data.
15. **Bootstrap over individuals, not queries.** Queries from the same animal are correlated; CIs
    computed over queries are optimistically narrow. All CIs here are 2,000 cluster bootstraps over
    individuals.
16. **F0.5, not R@1, is the census headline.** A false MATCH collapses two animals into one profile
    and **deflates** the count — mark-recapture cannot undo it. A false NEW only inflates, which
    capture-probability models tolerate. Precision therefore gets 2× the weight, and R@1 is demoted
    to a ranking diagnostic.
17. **But do not tune thresholds on F0.5.** It rewards labelling everything "new" — a degenerate
    policy scores ~0.95 on novel animals while missing 65–86 % of genuine re-sights. Threshold on
    **balanced accuracy** and report **count bias** (inflation − deflation) alongside.
18. **Anchor every harness with a dummy and an oracle.** Random-embedding dummy must sit at chance
    and the label-reading oracle at 1.0. If the dummy ever scores well, something leaks and no other
    number can be believed. `pixi run emb-selfcheck` is that check in one command.
19. **Split by individual, with a leakage guard — and keep a session split.** `aj_1` and `aj_2` are
    different animals that may share a shoot and a background; the session-grouped split is what
    proves the model is not reading the gravel.
20. **`nan` is a result.** `e2e_mlp` and `pretrained_cnn` produced *no* matches at `_q0.4` — a
    precision of `nan` over an empty set, not a good score. Recall 0 is the tell.

### About modelling

21. **Spots ≫ whole images.** Voting over per-spot matches (0.43–0.56 R@1) beats every image-level
    embedding tried (0.35). A learned image embedding converges to roughly a *mean* of the spot
    embeddings and no further.
22. **Do not learn the spot representation with SupCon-by-individual.** It is not a tuning failure —
    the objective fights the inference rule. SupCon pulls one animal's *genuinely different* spots
    together, destroying the per-spot correspondence that voting depends on. Measured directly:
    spot match-AUC rose **0.51 → 0.72** while retrieval got **worse**. Both a contextual transformer
    and a context-free MLP failed the same way, which is what makes it the objective's fault rather
    than the architecture's.
23. **Do learn the aggregation.** Every (query, candidate) pair is a labelled example, so supervision
    is abundant (~6k pairs/fold) for a *tiny* model over *individual-agnostic* match statistics —
    which is exactly why it transfers to unseen animals. +0.17 R@1 over raw voting, positive on all
    five folds, trains in seconds.
24. **Simplest wins, repeatedly.** logreg > MLP > set-attention. The expressive models lose to
    variance at this data size — DeepSets tied raw voting *on average* while collapsing to 0.087 on
    one fold. Capacity is not the constraint.
25. **Breadth beats peak.** The learned rule discovered that many moderate, mutually-consistent,
    geometrically-coherent matches beat one strong one: `frac≥.6` **+1.16** while `max_sim` is
    **−0.68**. It also learned to **size-normalise** (`log_nq` −0.64), correcting soft-chamfer's
    known flaw, and it uses **RANSAC constellation consistency** (+0.51) as real evidence.
26. **Per-spot principal-axis canonicalisation loses** (Mode A 0.19 vs Mode B 0.14). Most salamander
    spots are near-round, so their principal axis is unstable and 180°-ambiguous; canonicalising
    injects noise. Augmentation-based invariance is the right prior — *for spot shape*.
    (Canonicalising the whole *constellation* is a different, still-open idea.)
27. **Hand-engineered shape alone does not beat pure geometry** (`spotdesc` ≤ `classical`). Shape
    needs either a learned encoder or a distinctiveness weighting to pay for itself.
28. **ImageNet features do not transfer to binary spot masks.** Frozen ResNet18 over 32×32 mask crops
    was the worst model in the sweep (census F0.5 0.051 quick; recall 0 at `_q0.4`). The domain gap
    is total — this says nothing about CNNs in general, only about these features.
29. **Self-supervision on spots works, mildly, and the control proves it.** SimCLR over 40,185 spot
    crops (2-fold quick run): **correspondence-mined positives 0.375 R@1 > augmentation positives
    0.277 > frozen hand-features 0.227 ≫ random-init encoder 0.089**. The random-init control is
    what makes this readable — the encoder is learning something real. Mining positives from actual
    cross-photo correspondences beats synthesising them, and the augmentations deliberately vary
    everything *except* gross shape, because shape is the signal.
30. **The end-to-end formulation is safe where SupCon was not** — because the training objective
    *is* the inference rule, the failure mode of #22 cannot occur. It reaches parity with the hand
    rule (0.609 vs 0.622 census F0.5) but does not beat it.
31. **Human intuition, encoded, is load-bearing.** Manual pair review named six recurring reasons a
    match looks wrong: unmatched characteristic spots, round/non-distinctive spots, too few matches,
    outright mismatches, positional disagreement, shape disagreement. Turning those into
    `coverage × support` with a distinctiveness weight beat equal-weight voting on the census metric
    (0.347 vs 0.300 unfiltered). The **ablation is the proof**: removing the human special-spot
    supervision from the learned gate drops census F0.5 **0.609 → 0.493** and novelty AUROC
    **0.666 → 0.516** (chance). The ~8.7k interesting-spot clicks are worth more than the extra model
    capacity.
32. **Novelty is structurally different from ranking, and much harder.** All 17 aggregator features
    are absolute properties of a single (query, candidate) pair, so the model never sees how the top
    match compares to its competition — and "this scored 0.7" is uninterpretable on its own.
    **Relative** features (`margin`, `margin_ratio`, `z_top1`, `top1_minus_med`, `top1_minus_top5`,
    `score_std`) fix the framing, and the best *single* relative statistic used untrained (b′) often
    beats a trained novelty classifier (c′). Which statistic wins varies by fold — margin-type
    features are the most consistent.
33. **Conjunctive vs. additive spot matching is not yet decided.** The tensor (outer-product) flow
    edges out concat in the smoke run (R@1 0.145 vs 0.140) — inside the noise. Its known limitation
    is structural: a bilinear form cannot implement a ReLU, so two spots that *dis*agree on both
    blocks multiply to a spuriously positive score; the `c_shape`/`c_pos` constants trade veto
    strength against that.

### About scale — where the ceiling actually is

34. **More images of the same quality do not help.** The controlled scaling curve (eval fold held
    fixed, training on nested 25/50/75/100 % subsets of that fold's *individuals*) is **flat** on
    unfiltered data: marginal slope **−0.006 census F0.5 per 329 extra images**. On quality-filtered
    data the same curve is **still climbing (+0.014 per 191 images)**. More *good* photos help; more
    photos do not.
35. **Real data cannot separate "more training animals" from "more candidates to be confused by"** —
    collecting moves both at once. Synthetic populations (spots generated directly in the 62-dim
    space, so the matcher runs unmodified) make them independent knobs, and the answer is lopsided:
    growing the gallery 25 → 200 individuals at fixed training collapses census F0.5 **0.92 → 0.35**,
    while growing the training pool past ~50 individuals is **flat or declining**. The bottleneck is
    task difficulty, not training-set size.
36. **The method is sound; the *representation* is the ceiling.** Under an idealised spot vocabulary
    (distinct shape codes, low noise) the *unchanged* pipeline is near-perfect — R@1 0.95–1.00 at a
    50-animal gallery. Under the "realistic" preset (crowded shape vocabulary, large body-axis
    jitter and non-rigid warp) it lands exactly where the real data lands. The named highest-leverage
    knobs, in order: **shape-vocabulary crowding**, then **body-axis misfit** — "the single biggest
    reason real matching is hard" — then alignment noise, detection rate, spurious spots.
37. **Calibrate a simulator on statistics you did not fit.** The first synthetic population scored
    R@1 = 1.000 and described a task nobody has; the presets are now tuned so R@1 at a real-sized
    gallery lands near the measured 0.37–0.49, and each report prints spots-per-image and
    nearest-neighbour spacing (real vs. synthetic) so divergence is visible. Relative claims survive
    calibration error; absolute ones do not.
38. **No pre-match score can tell you a photo is hopeless.** Of the signals available *before*
    matching, the best (`n_spots`) reaches AUROC 0.62 and the rest are at or below chance; the
    oracle quality reaches 0.90 and the *post*-match `top1_score` / `margin` reach 0.96 / 0.85. The
    consequence is a design decision: **abstain after matching, do not pre-filter in the field.**
39. **Abstention is the honest deliverable.** A model answering 18 % of photos at 90 % precision is
    more useful than one answering 100 % at 45 %. `cov@P90` and `review@90` are reported everywhere
    for that reason — and they say plainly that the system is a shortlist assistant today, not an
    unattended census.

### Meta

40. **Diagnose before optimising.** D1 (are the labels or the model the ceiling?) was designed as a
    routing gate, and it earned it: with a third of individuals at chance-level self-consistency and
    54 suspected duplicate identities, effort belongs on extraction quality and label hygiene at
    least as much as on architecture.
41. **Change one thing at a time, behind one harness.** Every model in this repo — training-free,
    classical, learned, LLM — implements the same interface and is scored by the same folds and the
    same metrics. That is what makes "the frozen CNN beat the Set Transformer on retrieval but lost
    on open-set" a usable finding rather than an anecdote.
42. **A negative result is a result, if it is diagnosed.** The two biggest time sinks — image-level
    embeddings and spot-level metric learning — are now permanently closed, *with reasons* (#21,
    #22), which is what let the end-to-end formulation be re-opened safely (#30).

### About the hand labels (the preprocessing review)

43. **The matcher has an operating point, and it is now measured.** The review holds 842
    adjudicated spot correspondences over 55 individuals. Of the 427 the machine proposed, a human
    accepted **342 — 80.1% precision**. They then drew **415** correspondences the machine missed,
    putting recall at **≤46%** (an upper bound: a true correspondence nobody drew counts here as a
    success for the machine). This is what #10's "median mutual-match fraction 0.50" looks like with
    ground truth attached: half the correspondences exist, and one in five of those is wrong.

44. **Retuning the edge gate cannot work, and the reason is the interesting part.** Fitting
    `sigma_pos` / `match_thr` / `good_thr` on the 427 verdicts moves held-out F0.5 by **+0.002**,
    and the fitted constants swing between folds (thr 0.02–0.42). The diagnosis is one line:
    embedding cosine on accepted edges is **0.645 ± 0.154** and on rejected edges **0.650 ± 0.167** —
    identical, with rejected marginally *higher*, AUROC 0.484. The reviewer is judging these
    correspondences on structure the 62-dim embedding does not encode, so no threshold over it can
    reproduce their decision. This is #36 ("the representation is the ceiling") measured at the
    level of a single edge instead of a whole ranking, and it says the constants were never the bug.

45. **The SSL positives are ~16% wrong, and neither knob controls that.** #29's gain rests on mined
    correspondences being correct; against the human verdicts, mining precision is **0.844
    [0.68, 0.93]** at the current `SSL_MIN_SIM=0.40`. It is **flat (0.75–0.88) across every cutoff
    from 0.0 to 0.8**, and the RANSAC filter — described as "the real precision filter" — moves it
    by **+0.010 with overlapping intervals while discarding 49% of the pairs**. So the cutoff and
    the geometric check are controlling *yield*, not label quality. The pretraining set can be grown
    3–4× at the same noise rate (`SSL_GEOM=0 SSL_MIN_SIM=0.30`), which is the one lever here that
    scales past the 842 labels. Mining recall against known correspondences is 0.036 — a yield
    ceiling, not a precision problem.

46. **"Special" is learnable, the hand blend was half wrong, and more clicks buy nothing.** A
    six-factor logistic regression recovers the human interesting-spot labels at **AUROC 0.748 ±
    0.020** (individual-split, 10,613 labeled spots, 2,749 positive) against **0.707** for the
    hand-tuned `DEFAULT_WEIGHTS` — so fitting the weight beats eyeballing it, modestly. The
    coefficients are the real payload: **irregularity +0.824** and **size +0.707** carry the whole
    score, `elongation` is small (+0.103), and the two the hand blend bet on are wrong —
    `rarity` (hand weight **1.5**, the second-highest) fits at **−0.070**, i.e. nothing, and
    `noncircularity` (hand weight 1.0) fits **negative** at −0.337. Once you know a spot is big and
    lobed, "curved" argues *against* special.
    And the refresh is a flat line: the legacy export (362 images, 8,691 labeled spots) scores
    **0.750 ± 0.018** and the live store (454 images, 10,613 spots) scores **0.748 ± 0.020** —
    **−0.002 for a quarter more labels**. This is #34 ("more images of the same quality do not
    help") reappearing at the *label* level, and it is the number to quote before anyone spends
    another session clicking spots. What is scarce is not clicks; it is animals (#35) and
    representation (#36).
    **Shipped 2026-08-15**: `rarity` is now weighted **0** in `strict_match.DEFAULT_WEIGHTS` (was
    1.5, its second-heaviest term) and dropped from the learned feature set. The A/B runs on every
    `pixi run distinctiveness`: the learned path is unchanged (0.748 → 0.748, it had already learned
    to ignore the factor) while the **hand blend improves 0.707 → 0.726**, so the weight was
    actively costing accuracy. Skipping the factor also removes the only population-wide neighbour
    search in the pipeline — factor computation went **318 s → 76 s**. `isolation` (fits −0.061) and
    `noncircularity` (fits −0.385, i.e. backwards) are the next candidates, left in place so the
    next measurement stays readable.
    **Correction, 2026-08-16: that ship was three-quarters undone by defaults, and the shape of the
    mistake is the lesson.** Changing `DEFAULT_WEIGHTS` changed what *reads the default* — and three
    consumers did not.
    (i) `distinctiveness.load_spot_factors` defaulted to `LEGACY_WEIGHTS` so its own A/B had a real
    `rarity` column, and that frame's `distinctiveness` column *is* what `hand_weight_lookup` hands
    the matcher — so `WEIGHT_MODE=hand` in `compare_strict` and `sweep_bakeoff`'s unlabelled
    fallback both kept matching at rarity 1.5, while paying the ~240 s neighbour search the change
    was supposed to delete. The A/B now asks for `LEGACY_WEIGHTS` by name and everything else gets
    the shipped blend.
    (ii) The preprocessing UI's `interest.json` cache was built 2026-07-28, carried no record of the
    weights that produced it, and so was never invalidated: for a month the interest ramp and the
    `Interest:` tooltip — what a reviewer judges a spot by — showed the legacy blend. The cache is
    now stamped with the blend and a mismatch rebuilds it.
    (iii) `synthetic._oracle_distinctiveness` was `0.5·rarity + 0.5·isolation`, i.e. **entirely** the
    two factors the clicks fit at ≈0 (−0.070, −0.069), so every synthetic sweep of a
    distinctiveness-weighted matcher was calibrated against a salience the real system had
    discarded. It now mirrors `DEFAULT_WEIGHTS` (size 2.0, isolation 1.0) over a per-spot `size`
    generated at identity level; `oracle_blend="legacy"` reproduces the old numbers and tags its own
    artifacts. **Synthetic results predating this need re-calibrating** (`MODE=calibrate`).
    The generalisable form: **a default is not a decision until every reader of it is checked.** A
    constant renamed in one module, a cache with no provenance stamp, and a simulator holding its
    own copy of the same idea are three separate ways for a shipped change not to ship.

47. **You cannot re-weight evidence the model has already rejected — and that reverses a build
    order.** ``feasibility.py`` Part C measured that geometric consistency separates same-from-
    different at **0.603** on high-quality photo pairs and **0.500 — chance — on low-quality ones**
    (gap +0.103), while the same test on curl (+0.026) and camera angle (+0.005) found nothing. That
    looked like a cheap, well-evidenced win: add `geometry × quality` as a product term, since an
    additive model provably cannot express "weight this other feature by that one". Built and
    A/B'd (`run_quality_interaction.sh`), it moved census F0.5 by **−0.005** against a ±0.07 fold
    spread, with the untouched `strict_hand_pos` control identical in both arms.
    The coefficients explain it, and they are the result. The model *did* learn the predicted
    pattern — `geom_consistency` went −0.112 → −0.222 with `geom_x_quality` at **+0.163**, i.e.
    "geometry argues against a match, less so when the photos are good". It buys nothing because
    **geometry carries a negative weight in both arms**: the aggregator had already concluded the
    current geometric feature is not evidence, and scaling trust in something already distrusted is
    a no-op. The lesson generalises past this one term: an interaction is only worth building on a
    main effect the model actually uses. Quality-modulation was scheduled *before* the constellation
    work as the cheaper item; it is in fact **downstream** of it, and worth re-running only once
    triplet/angle invariants earn geometry a positive weight. Default is now
    `QUALITY_INTERACTION=0`, plumbing kept.

48. **The pair verdict is a reliable label, and it says a quarter of the "confident errors" are not
    errors.** First complete pass of the new `pair-review` verdict (2026-08-16): **70 judgments over
    56 unique photo-pairs** — 25 match, 38 different, 7 unsure.
    **Self-consistency, measured for free: 13/14.** The generator had rendered 14 photo-pairs in
    both directions (`a→b` and `b→a`), which look different enough on screen not to be recognised,
    so the reviewer was blind-retested 14 times and gave the same verdict on **13** of them. Set
    that against #10 — where 39 % of individuals' own repeat photos agree no better than two random
    animals do — and against #44, where the per-edge verdict was unlearnable: **the pair is an
    altitude at which this human is repeatable.** That is the precondition for using these as
    labels at all, and it is why the verdict replaced the click rather than joining it.
    **10 of the 40 pairs sampled as `CONFIDENT-WRONG` are matches**, plus 6 `unsure`. At most 24/40
    of that band are genuine errors. The band is a deliberately enriched sample of the matcher's
    confident mistakes, so this does not extrapolate to a dataset-wide error rate — but every
    headline number is computed against filenames that are wrong on a quarter of the cases where
    the matcher and the labels disagree most confidently.

49. **The duplicate scan has a recall problem, and hand review found it in one sitting.** The 14
    conflicting rows (verdict `match`, filenames `different`) reduce to **10 unique photo-pairs over
    9 individual-pairs**. Cross-checked against `suspect_duplicates.csv` (53 pairs, from the
    mean-embedding prefilter in `label_consistency.py`):

    | | individual-pairs |
    |---|---|
    | flagged by the scan **and** confirmed by the human | 2 — `ac_3/ca_62`, `ca_70/lj_13` |
    | found by the human, **never flagged by the scan** | **7** |
    | scan suspects still unadjudicated | 51 |

    The scan's own docstring calls itself a lower bound ("can MISS a duplicate whose mean drifts");
    this measures how loose. A 70-pair review found **7 duplicates the 53-pair scan did not**, so
    the true duplicate count is well above 53 and the scan cannot be treated as the work queue. One
    of the 9 (`ca_10/sj_3`) carries the reviewer's note *"not extracted correctly so it shouldn't be
    counted"* — a bad extraction masquerading as a duplicate — leaving **8 to adjudicate**.

50. **Within a band, the matcher's score runs BACKWARDS against the human's call — suggestive, not
    yet established.** AUROC of score → "the human says these are the same animal", on the pairs
    with a decided verdict:

    | stratum | n (match) | AUROC | 95 % CI (bootstrap) |
    |---|---|---|---|
    | pooled | 63 (25) | 0.412 | [0.253, 0.574] |
    | `CONFIDENT-WRONG` | 34 (10) | 0.342 | [0.150, 0.552] |
    | `MID` | 19 (5) | **0.129** | **[0.000, 0.350]** |

    Pooled and `CONFIDENT-WRONG` **span 0.5 and establish nothing**; only `MID` excludes it, on 19
    pairs and 5 positives. Treat this as a hypothesis with one supporting stratum, not a finding.
    Pooling across bands is invalid anyway — `pair_review generate` samples by score band, so the
    frame is stratified and any score statistic that ignores `sources[0]["band"]` is partly
    measuring the sampling design (`review_labels.pair_verdicts` documents this).
    The hypothesis worth testing: a true duplicate is two photos of one animal in different poses —
    genuinely hard, only ~24 % of spots surviving (#21) — whereas a high-scoring false match is two
    different animals that look alike *easily* (round spots, similar layout). If so the score is
    tracking easy visual similarity rather than identity, which is #36 ("the representation is the
    ceiling") and #44 (the human judges on structure the embedding does not encode) showing up a
    third time. **To settle it, judge a random sample rather than a band-stratified one.**

51. **The reason chips went unused — 2 of 70.** No conclusion can be drawn from them. Either they
    cost more than they are worth mid-pass, or the verdict alone is what the reviewer wanted to
    give. Worth one deliberate trial before either removing them or asking for them.

---

## 8. Open threads

| # | thread | status |
|---|---|---|
| 1 | Full additive-vs-conjunctive bakeoff grid | started 2026-07-27, no `summary.csv` yet |
| 2 | Full-gallery (~687 candidate) numbers for the champion | `pixi run aggregator --full-gallery` exists; not recorded in `artifacts/` |
| 3 | Extraction quality / hand-verified validation subset | **done for matching** (#43): 842 verdicts over 55 individuals, 80.1% precision / ≤46% recall |
| 4 | Duplicate-identity cleanup | **8 confirmed** by the pair-review pass (#49), 2 of them also on the scan's list; 51 scan suspects still unadjudicated, and the scan is known to miss ~7-in-9 so the real count is higher. Merging them is unstarted — `duplicate_of` is still empty |
| 8 | Is the matcher's score inverted within a band? (#50) | one stratum significant, 19 pairs — needs a **randomly sampled** verdict pass, not a band-stratified one |
| 5 | Body-axis alignment | flagged by the simulator as the highest-leverage remaining fix (#36) |
| 6 | Calibrated three-way `match / new / abstain` output | signals and `cov@P90` exist; the fused calibrated head does not |
| 7 | RANSAC spatial-verification re-rank (Phase 5) | designed, unbuilt |

---

## Reproducing anything here

```bash
pixi run emb-selfcheck                 # harness acceptance check (oracle ≈ 1.0, dummy ≈ chance)
pixi run label-consistency             # D1: are the labels or the model the ceiling?
pixi run aggregator                    # the summary-features + logreg champion (fold-only gallery)
pixi run aggregator --full-gallery     # the deployment-honest gallery (~687 candidates, ~35 min)
pixi run bakeoff --space quick         # embedding-flow × matcher smoke test
pixi run build-dataset --dry-run       # dataset rebuild plan + billed-call estimate (ALWAYS first)

# training on the hand labels from the preprocessing web app (regimes 1-5)
pixi run review-labels                 # what supervision exists in review.json at all
pixi run distinctiveness               # R1: is "special" learnable from the six factors?
pixi run gate-calibration              # R2: fit the edge gate on 427 human verdicts
pixi run mining-audit                  # R3: precision of the SSL positives, vs human truth
bash scripts/experiments/run_review_regimes.sh audit   # R2+R3, ~10 min, no training
bash scripts/experiments/run_review_regimes.sh all     # every regime (hours)
```

Artifacts land under `artifacts/<track>/<dataset>/`; every sweep writes its own `RESULTS*.md` next
to the logs and plots that produced it.
