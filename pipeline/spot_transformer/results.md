# Spot Transformer — Salamander Re-Identification: Results

Investigation into learning a salamander identification model on top of the per-spot
embeddings in `spot_embeddings`. This document records **every scheme tried**, its result,
and *why* it worked or failed.

---

## TL;DR — Best result

**Learned aggregation (match-features + logistic regression):**
**5-fold held-out R@1 = 0.603 ± 0.104**, a **+0.17** lift over raw-spot voting (0.434) and
**~+0.25** over image-level embedding. Stable (positive on all 5 folds), trains in seconds,
generalizes to unseen individuals, ships with a calibrated confidence.
→ [`aggregator.py`](aggregator.py)

The whole investigation's lesson: **the simplest model with the most direct supervision won.**
Learning the *representation* failed (too few identity labels for a big encoder); learning the
*aggregation* with a *tiny* model on abundant pair-labels won; making that aggregator *more
expressive* (set-attention) pushed it back into the over-parameterized failure mode.

---

## Setup

- **Data:** `all_sasa_norm_2026_16_07` / `contours.db`. 733 images (≈443 real + synthetic),
  272 individuals, **17,735 spots**. Each spot = a hand-engineered **62-dim embedding**
  (EFD shape [37] + body-intrinsic position [25]).
- **The binding constraint:** ~**1.4 real photos per individual**. Identity supervision is
  scarce even though spots/tokens are abundant.
- **Eval protocol:** 5-fold CV **split by individual** (leakage-guarded); eval folds drawn
  from the ~83 individuals with ≥2 real photos; synthetic images are train-only. Leave-one-out
  retrieval, reported as **R@1 / R@5 / R@10**.
- **Reference baseline (pre-existing):** spot-to-spot `naive_match_all`, P(pair is same
  individual | cosine ≥ cutoff): 0.70→0.24, 0.75→0.43, 0.80→0.64, 0.85→0.82, 0.90→0.91.

---

## All approaches, ranked

| # | Scheme | What is learned | Eval R@1 | Verdict |
|---|---|---|---|---|
| **6** | **Aggregation: summary features + logistic regression** | the voting rule | **0.603 ± 0.104** (5-fold) | ✅ **BEST** |
| 2 | Raw-spot **voting** (no training) | — | 0.434 ± 0.092 (5-fold) | strong free baseline |
| 7 | Aggregation: set-attention (Deep Sets) | the voting rule | 0.431 ± 0.186 (5-fold) | tied w/ raw, **unstable** |
| 1 | Image-level SupCon transformer (v1) | image embedding | ~0.37 (fold-0 best) | ≈ mean-pool, no gain |
| 0 | Image-level **mean-pool** (no training) | — | 0.352 (fold-0) | image baseline |
| 4 | Spot-level MLP (context-free) | spot embedding | ~0.33 (fold-0 best) | below raw voting |
| 3 | Spot-level SupCon transformer (v2) | spot embedding | ~0.31 (fold-0 best) | below raw voting |

> Note: rows 2/6/7 have full 5-fold numbers; rows 0/1/3/4 are single-fold (fold-0, an *easy*
> fold where raw voting itself scores 0.556 vs its 0.434 average) — so their true 5-fold
> numbers would be **lower**, widening the gap to the winner.

---

## Detailed results

### 0–1. Image-level embedding (aggregate spots → one fingerprint)
`transformer.py` (CLS readout) + `train.py` / `eval.py` / `sweep.py`.

- **Mean-pool baseline (fold-0):** R@1 0.352, R@5 0.574, R@10 0.667, mAP 0.348.
- **SupCon transformer sweep (fold-0):** best config `F_all_reg` R@1 **0.370** — inside the
  noise of the mean-pool baseline. Training reliably pulls an *untrained* model (~0.15) up to
  ≈ the mean-pool level and **no further**; 2-layer configs overfit by epoch 1–2.
- **Cross-validated image match-AUC:** mean-pool 0.693 → trained 0.738 (a real but small
  representational gain that did **not** translate into better R@1).
- **Verdict:** the learned image embedding ≈ a mean of the spot embeddings. Ceiling set by the
  data/features, not the model. Archived in [`deprecated/sweep_results/RESULTS_image_track.md`](deprecated/sweep_results/RESULTS_image_track.md).

### 2. Raw-spot voting (the pivot)
`eval_spot.py::spot_vote_retrieval` — soft-chamfer: each candidate individual scores
`Σ_query-spots max cosine to any of its gallery spots`.

- **fold-0 R@1 = 0.556** (R@5 0.722, R@10 0.815) — **+0.20 over image-level (0.352)**, with
  zero training.
- **5-fold R@1 = 0.434 ± 0.092.**
- **Key insight:** cross-image spot **match-AUC ≈ 0.51 (near chance)**, yet voting is strong —
  because "same individual" ≠ "same spot"; most same-individual spot pairs are dissimilar, but
  voting keys off the *nearest* match, where the truly-corresponding spots match strongly
  (precision 0.77–1.0 at cutoff ≥0.75). **Spots + voting is the right representation.**

### 3–4. Spot-level metric learning (learn a better spot embedding)
`transformer.py::forward_spots` / `SpotMLP` + `train_spot.py` / `eval_spot.py` / `sweep_spot.py`.
Spot-level SupCon on individual labels (same-image pairs excluded), identification by voting.

- **Contextual transformer (fold-0):** untrained 0.222 → best **~0.31**; match-AUC rose
  0.51→0.72 but **voting got worse**.
- **Context-free MLP (fold-0):** best **~0.33** (slightly better than the transformer → context
  hurts), but still **well below raw voting 0.556**.
- **Verdict — decisive:** both architectures fail, so it is the **objective**, not context.
  SupCon-by-individual pulls together an animal's *genuinely different* spots, destroying the
  per-spot correspondence that voting relies on. The hand-engineered embedding already
  preserves it. **Do not learn the spot representation for this task.**

### 6. Learned aggregation — summary features + logistic regression ✅
`aggregator.py`. Keep raw embeddings **fixed**; learn the function that turns a
(query image → candidate individual) match into a score. Every `(query, candidate)` pair is a
labeled example → **abundant supervision (~6k pairs / fold)** for a **tiny** model over 17
**individual-agnostic** match features → generalizes to unseen animals.

**5-fold CV:**

| fold | raw R@1 | learned R@1 | ΔR@1 |
|---|---|---|---|
| 0 | 0.556 | 0.685 | +0.130 |
| 1 | 0.327 | 0.423 | +0.096 |
| 2 | 0.386 | 0.568 | +0.182 |
| 3 | 0.370 | 0.717 | +0.348 |
| 4 | 0.533 | 0.622 | +0.089 |
| **mean** | **0.434 ± 0.092** | **0.603 ± 0.104** | **+0.169** |

(R@5 mean 0.757. Logistic regression > MLP variant, 0.685 vs 0.648 on fold-0 — simpler wins.)

**Learned weights (what the voting rule discovered):**

| feature | weight | reading |
|---|---|---|
| `frac≥.6` | **+1.16** | **breadth dominates** — many decent matches |
| `max_sim` | **−0.68** | a single strong peak is *negative* → "1×0.9 vs 4×0.7" ⇒ 4×0.7 wins |
| `log_nq` | −0.64 | learned to **size-normalize** (soft-chamfer's flaw) |
| `softchamfer_sum` | +0.61 | the old hand score still helps, but isn't dominant |
| `mutual_count` | +0.52 | reciprocal nearest-neighbour matches |
| `ransac_frac` | +0.51 | **geometric constellation consistency** (RANSAC inliers) |
| `qmax_std` | +0.43 | having *some* standout matches |

**Confidence** (delivers the "voting strategy + confidence" ask):
- `top1−top2` **margin** → identification accuracy **0.33 / 0.68 / 0.80** (low / mid / high) —
  monotonic, usable as-is.
- **Platt scaling** fixes probability calibration: top bin went from predicting 0.72 (actual
  0.22) to predicting 0.18 (actual 0.22).

### 7. Learned aggregation — set-attention (Deep Sets)
`aggregator_set.py`. More expressive: feed the variable-length **set of per-spot match
records** (sim1/2/3, margin, mutual, geom-inlier, log_nc) into an attention-pooling net.

**5-fold CV:** R@1 **0.431 ± 0.186** — tied with raw voting, and **unstable** (fold 3 collapsed
to 0.087, below raw's 0.370). Weaker confidence (margin accuracy 0.34/0.41/0.56).
**Verdict:** over-parameterized for ~187 positive pairs/fold. Revisit only with more data.

---

## Key findings

1. **Spots ≫ images.** Voting over spots (0.43–0.56) beats every image-level embedding (0.35).
2. **Don't learn the representation here** — image *or* spot. Too few identity labels; the
   hand-engineered 62-dim is already near its ceiling for matching.
3. **Do learn the aggregation.** It's the one place supervision is abundant (pair labels) and
   the model can stay tiny and individual-agnostic → it generalizes.
4. **Simplest wins, repeatedly.** logreg > MLP > set-attention; the expressive models lose to
   variance on this data size.
5. **Breadth beats peak.** The learned rule: many moderate, mutually-consistent, geometrically-
   coherent matches, size-normalized — not one lucky strong match.

---

## Best system & how to run

Fixed raw spot embeddings → 17 match features (incl. RANSAC geometry) → logistic regression →
ranked candidates + calibrated confidence.

```bash
pixi run python pipeline/spot_transformer/aggregator.py     # 5-fold CV + weights + calibration
```

Modules: `aggregator.py` (winner), `aggregator_set.py` (Deep-Sets, parked), `eval_spot.py`
(voting + spot match tables), `train_spot.py` / `sweep_spot.py` (spot metric-learning, negative
result), `train.py` / `eval.py` / `sweep.py` (image track, negative result),
`transformer.py` / `losses.py` / `data.py` (shared).

---

## Next steps

- **Hard-negative mining** — train against confusable individuals, not random negatives
  (most likely further lift on the winning aggregator).
- **Per-image** (not per-individual) candidate scoring.
- **Inference/persistence path** — score a new photo vs a gallery, emit id + confidence.
- **More data** (open-source / multi-view synthetic) — the only thing that unlocks the
  expressive models (set-attention) and lifts the whole ceiling; watch **multi-photo
  individuals**, not raw image count.
