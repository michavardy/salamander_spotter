# SEA improvements — Spot Embedding + Aggregation

A prioritised backlog of proposed improvements to the salamander re-identification matcher
(the **S**pot **E**mbedding + **A**ggregation study). It complements
[per_spot_embedding_aggregation.md](per_spot_embedding_aggregation.md) (the design) and
[spot_embedding_aggregation_plan.md](spot_embedding_aggregation_plan.md) (the build plan) — this
doc is the *"what to do next and in what order"* list, scored for impact vs. effort.

## Where we are

Best current numbers (5-fold `individual`, dataset `all_sasa_norm_2026_11_07`, see
[bakeoff/comparison.md](../artifacts/spot_embedding/bakeoff/comparison.md)):

| model | rank-1 | mAP | verify AUC | open-set AUROC |
|-------|--------|-----|------------|----------------|
| set_transformer | 0.27 | 0.37 | 0.64 | 0.54 |
| cnn (frozen ResNet18) | 0.23 | 0.33 | 0.57 | 0.58 |
| classical (geometry) | 0.17 | 0.31 | 0.56 | 0.57 |

**Target:** rank-1 ≈ 0.85, mAP ≈ 0.95, plus a calibrated confidence output.

> **Reality check.** rank-1 0.2 → 0.85 is a change of regime, not a knob-turn. Published wildlife
> re-ID on clean, curated data lands rank-1 ≈ 0.6–0.9; mAP 0.95 means "almost every query ranks its
> true match first" — essentially *solved* re-ID. We are attempting this with ~169 multi-photo
> individuals (mostly 2–3 photos) on **machine-extracted, unverified spots**. 0.85 rank-1 is a
> stretch-but-maybe goal *if the labels hold up*; 0.95 mAP is likely above the current data ceiling
> until extraction quality and/or dataset size improve.

## How to read the scores

- **Influence (1–10)** — predicted contribution toward the target metrics (rank-1 / mAP), or, for
  the confidence work, toward *field usefulness*. 10 = potentially transformative, 1 = negligible.
- **Investment (1–10)** — engineering + compute + risk. 1 = an hour, 10 = weeks / external effort.
- A high-influence / low-investment item is a "do it now"; the recommended order (below) is roughly
  ROI = influence ÷ investment, subject to dependencies and the diagnostic branch.

---

## The improvements

| # | Improvement | What it changes | Influence | Investment |
|---|-------------|-----------------|:---------:|:----------:|
| **D1** | **Label-consistency diagnostic** — for each multi-photo individual, align its photos (RANSAC similarity) and measure spot-count + spatial-inlier consistency across repeats | Nothing in the model; **decides whether the ceiling is labels or model** and routes all other effort | **9** (decision gate) | **2** |
| **D2** | **PMA attention inspection** — dump the Set-Transformer pooling attention per spot, overlay on photos | Diagnostic only; confirms whether the model already focuses on distinctive spots (informs G3) | **2** | **1** |
| **M1** | **Relative-geometry encoding** — feed pairwise `(distance, relative-angle)` as GNN edge features / Set-Transformer attention bias instead of absolute `normalize_pos(x,y)` tokens | Implements the design's core "never feed raw (x,y)" rule; hands the model rotation invariance instead of hoping augmentation teaches it | **8** | **4** |
| **M2** | **RANSAC spatial-verification re-rank** (Phase 5) — learned per-spot descriptors → correspondences → geometric-consistency check → re-rank top-k | Adds the HotSpotter/Wildbook spatial check the Hungarian matcher currently omits (it matches appearance only) | **8** | **4** |
| **M3** | **CNN ⊕ Set-Transformer score fusion** — combine the frozen-CNN appearance embedding with the constellation embedding | Fuses two signals that win *different* metrics (CNN→rank-1, ST→open-set); near-free lift | **5** | **2** |
| **G3** | **TF-IDF / rarity distinctiveness weighting** — cluster spot shapes into a vocabulary, weight each spot by inverse frequency in pooling **and** the verifier | Directly implements "bias harder on unique spots" (§3), training-free so it won't overfit at n≈500 | **5** | **3** |
| **M4** | **Constellation canonicalisation** — align each whole spot-set to an anatomical frame (head→tail axis) before encoding | Removes the rotation nuisance *structurally* (complements or replaces M1); needs a head/orientation cue | **6** | **6** |
| **M5** | **Spot-level auxiliary contrastive loss** — add a per-spot term (corresponding spots close) alongside the salamander-level SupCon | Keeps per-spot embeddings discriminative (helps M2/Hungarian); designed in §5 but unbuilt | **4** | **3** |
| **X1** | **Better / cleaner spot extraction** — aggressive filtering, improved extractor, and a small hand-verified validation subset | Raises the **true ceiling** if D1 shows label noise dominates; without it every model change plateaus | **9** *(if D1 bad)* | **7** |
| **X2** | **Synthetic data (Gemini views) + self-consistency filter** — manufacture new views of singletons, keep only those whose spot-set still matches the source | Adds real-ish positives for the 121 singletons (helps open-set/verification, little for closed rank-1); identity-drift risk | **3** | **7** |
| **X3** | **Collect more real multi-photo individuals** — field/label effort | The highest true-ceiling raiser; metric learning is data-starved at 169 multi-photo IDs | **9** | **9** *(external)* |
| **C1** | **Confidence metric** (Phase 4) — fuse input quality (blur, #spots) + TTA embedding variance + top1/top2 margin + verifier inliers → calibrated `[0,1]`, three-way **match / new / abstain**, risk–coverage report | ~0 effect on rank-1/mAP, but **high effect on usefulness** — lets the system answer the easy fraction at high precision and abstain on the rest | **rank-1: 1 / usefulness: 9** | **5** |

### Explicitly deprioritised (low or negative ROI here)

| item | why not |
|------|---------|
| Bigger model / more layers / higher d_model | ~500 images punish capacity → overfits, doesn't generalise |
| More epochs past convergence | SupCon already converges (1.7→0.5); more just memorises |
| Loss temperature / LR sweeps | second-order; won't move a 4× gap |
| Principal-axis **per-spot** canonicalisation | already tested and **lost** (Mode A beat Mode B, GATE 2) — near-round spots have unstable axes. (Note: M4 canonicalises the **whole constellation**, which is a different, still-open idea.) |

---

## Suggested order of implementation

The order is **branch-dependent**: D1 decides whether you're on the *model* path or the *data* path.

### Phase A — Diagnose (do first, ~1 day, no modeling)

1. **D1 — label-consistency diagnostic.** This is the gate. If same-individual photos don't share a
   spatially-consistent spot set, **stop and go to the data branch (X-items)** — no matcher can win
   against inconsistent labels. If the inliers are healthy, the signal is there and the model is
   leaving it on the table → continue to Phase B.
2. **D2 — PMA attention inspection.** Cheap, and tells you whether G3 (distinctiveness weighting) is
   needed or the model already learned it.

### Phase B — Cheap model wins (if D1 is healthy)

Ordered by ROI:

3. **M3 — CNN ⊕ ST fusion.** Cheapest lift; validates that complementary signals stack.
4. **M2 — RANSAC spatial-verification re-rank.** Likely the single biggest rank-1/mAP jump; reuses
   RANSAC already present in the classical path.
5. **M1 — relative-geometry encoding.** The core design fix; makes M2's correspondences cleaner too.
6. **G3 — TF-IDF distinctiveness weighting.** Directly addresses "bias harder on unique spots";
   feeds both pooling and the M2 verifier.

### Phase C — Structural (if Phase B stalls)

7. **M4 — constellation canonicalisation** — only if rotation invariance is still the failure mode
   after M1 (measure with a rotation-augmented eval).
8. **M5 — spot-level auxiliary loss** — sharpens the per-spot descriptors M2 relies on.

### Phase D — Data (front-load this branch if D1 is bad)

9. **X1 — better extraction + hand-verified validation subset.** If D1 says labels are the ceiling,
   this becomes item #1 of the whole plan, ahead of Phase B.
10. **X2 — synthetic singleton views** (with the self-consistency filter as the safeguard).
11. **X3 — collect more real multi-photo individuals** — the durable ceiling-raiser; runs in
    parallel with everything, longest lead time.

### Cross-cutting

12. **C1 — confidence metric.** Build once retrieval is "decent" (after Phase B). It doesn't raise
    rank-1, but it's what makes the system *useful even at moderate rank-1*: answer the ~40% of
    photos you're sure about at high precision, abstain on the rest. The harness already computes a
    risk–coverage curve, so much of the scaffolding exists.

---

## One-line summary

> **Diagnose first (D1).** If labels are the ceiling → data branch (X1 → X2 → X3). If the model is
> leaving signal on the table → M3 → M2 → M1 → G3, then C1 for usefulness. 0.85 rank-1 is plausible
> if the labels hold and the model fixes land; 0.95 mAP likely also needs cleaner/more data.
