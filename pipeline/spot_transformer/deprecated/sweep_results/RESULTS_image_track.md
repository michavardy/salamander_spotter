# Image-level track (v1) — results archive

**Approach:** `SpotSetTransformer` aggregates a photo's spot set → one image fingerprint
(CLS readout), trained with SupCon on **individual** labels. Retrieval = image-to-image.
Eval = held-out individuals (fold 0 unless noted), leave-one-out retrieval.

**Verdict:** the learned image embedding **matches but does not clearly beat the mean-pool
baseline** on held-out individuals. Differences between configs are within single-fold noise
(eval = 54 imgs, 1 image ≈ 0.019 R@1). Most likely limited by data scarcity
(~1.4 real images/individual) and synthetic-dominated positives. → pivot to spot-level (v2).

## Baseline (fold 0, mean-pool, zero training)
R@1 = 0.352 · R@5 = 0.574 · R@10 = 0.667 · mAP = 0.348

## Regularization sweep (fold 0, patience=3, 30 epochs max)
| config | best_ep | R@1 untrained→best | ΔR@1 vs baseline | R@5 | mAP | match-AUC(eval) | verdict |
|---|---|---|---|---|---|---|---|
| F_all_reg (1L d96, drop.4 jit.03 mdrop.2) | 5 | 0.148 → 0.370 | +0.019 | 0.611 | 0.357 | 0.694 | BEATS (noise) |
| E_small   (1L d96, drop.3 mdrop.2)        | 5 | 0.148 → 0.352 | +0.000 | 0.667 | 0.352 | 0.694 | tie |
| A_ref     (2L d128, drop.2)               | 2 | 0.167 → 0.333 | -0.019 | 0.667 | 0.329 | 0.696 | below |
| C_jitter  (2L d128, drop.3 jit.03)        | 1 | 0.167 → 0.315 | -0.037 | 0.556 | 0.335 | 0.679 | below |
| D_shallow (1L d128, drop.3)               | 2 | 0.241 → 0.315 | -0.037 | 0.611 | 0.335 | 0.669 | below |
| B_heavy_drop (2L d128, drop.4 mdrop.2)    | 2 | 0.167 → 0.296 | -0.056 | 0.574 | 0.298 | 0.665 | below |

Pattern: heavier regularization + smaller model wins; 2-layer configs overfit by epoch 1–2.
Training pulls untrained (~0.15) up to ≈ baseline (~0.35) and no further → data/feature ceiling.

## Cross-validated held-out image matching (all 5 folds pooled, 273 true pairs)
Threshold-free **match AUC: mean-pool 0.693 → trained 0.738 (+0.045)**.
Fixed-cutoff precision is NOT comparable across the two spaces (trained shifts cosines up);
read at matched recall. Lower temperature (0.07) populates the 0.85–0.90 cutoffs
(precision 0.78 @ 0.90) that temp 0.10 left empty.

## Reference: spot-level raw matching (`naive_match_all`) — becomes the v2 baseline
P(pair is same individual | cosine > cutoff), over all spot pairs:
| cutoff | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 |
|---|---|---|---|---|---|
| P(same individual) | 0.243 | 0.425 | 0.636 | 0.821 | 0.908 |

## Artifacts
- Learning curves: `curve_<config>.png` (train vs eval R@1 + loss, per config)
- Overlay: `overlay_eval_r1.png`
