# Per-Spot Embedding + Aggregation — Design Deep-Dive

The chosen architecture (Option B from [modeling_strategy.md](modeling_strategy.md)): a **two-level
hybrid**. A small encoder embeds each isolated spot's *shape* into a token; the tokens are combined
with **relative** positional information and fed to a set/graph network that outputs one
**salamander** embedding — while the per-spot embeddings remain available for detailed matching.

The **current focus of study is the architecture/technology bake-off** ([§6](#6-architecture-and-technology-the-candidate-bake-off-q5)):
several candidate matchers evaluated head-to-head through one shared harness. Spot **extraction is
deliberately deferred** ([§1](#1-spot-extraction-deferred-q7)) — worst case, Gemini already does it
— so the effort goes into *matching*. This document answers the original seven questions and adds
the new considerations, ending with a **priority and evaluation plan** ([§9](#9-priority-and-evaluation-plan)).

| Topic | Section |
|-------|---------|
| Spot extraction — **deferred**, Gemini is the fallback (Q7) | [§1](#1-spot-extraction-deferred-q7) |
| Shape + relative position; **orientation as an A/B experiment** (Q2) | [§2](#2-encoding-shape-and-relative-position-q2) |
| Biasing characteristic spot shapes (L/M/blobs) (Q4) | [§3](#3-making-distinctive-spots-count-more-q4) |
| Robustness to missing spots (Q3) | [§4](#4-robustness-to-missing-spots-q3) |
| **Confidence & the "cannot evaluate" state** (new) | [§5](#5-confidence-and-the-cannot-evaluate-state) |
| **Architecture bake-off: Set Transformer / GNN / CNN / Hungarian / classical** (Q5, main focus) | [§6](#6-architecture-and-technology-the-candidate-bake-off-q5) |
| Augmentation (Q1) | [§7](#7-augmentation-q1) |
| Design considerations (Q6) | [§8](#8-design-considerations-q6) |
| **Priority & evaluation plan** (new) | [§9](#9-priority-and-evaluation-plan) |

```
                          ┌─────────────── one salamander photo ───────────────┐
  raw image ─► SPOT EXTRACTOR ─► { spot mask_i, centroid_i, area_i }  (a set)
                                          │
                    ┌─────────────────────┼─────────────────────┐
              per spot i:                                    the set:
        spot mask ─► small CNN ─► shape embed                centroids ─► relative geometry
        (+ optional topology hand-features)                  (edges / attention / matching)
                    │                                                 │
                    └──────► spot token_i  ◄──────────────────────────┘
                                          │
              {token_1 … token_N} ─► [ Set Transformer | GNN | Hungarian match ] 
                                          │
                                          ▼
                     salamander embedding (or pairwise match score)
                     + per-spot embeddings  +  confidence
```

---

## 1. Spot extraction, deferred (Q7)

**Decision: defer extraction. Keep using the current Gemini pipeline for now, and only revisit
extraction once matching works well.** The reasoning the user gave is sound: the *worst case* is
that Gemini keeps producing spots (and can even serve as a matching fallback), so extraction is not
on the critical path. Effort should go into the matcher.

When matching *does* work and extraction becomes the bottleneck (cost, latency, offline/CPU use),
revisit it then:

- **Start classical.** Spots are high-contrast yellow-on-near-black → color threshold (HSV/Lab
  yellow channel) → connected components → contour trace. Nearly free; the `threshold + CC` route
  noted for the sibling `salamander_id`.
- **Distill if needed.** Train a tiny U-Net / YOLO-Seg on the **Gemini labels as supervision** — a
  cheap, deterministic student distilled from the expensive teacher.

Because extraction is deferred, the matcher **must be trained to tolerate whatever spots Gemini
gives it** — missed, false, and ragged spots included — via the augmentations in
[§7](#7-augmentation-q1) and the robustness design in [§4](#4-robustness-to-missing-spots-q3). That
tolerance is what makes deferring extraction safe: the identifier is decoupled from perfect spots.

---

## 2. Encoding shape and relative position (Q2)

Guiding rule: **the token carries "what this spot looks like"; the geometry (edges / attention /
matching cost) carries "how spots relate."** Absolute pixel `(x, y)` never enters either.

### 2a. Spot shape — and the orientation experiment

Requirement: the spot-shape embedding should be **rotation-, stretch-, and blur-invariant**. There
are two ways to get that invariance, and **which one wins is itself a design question to evaluate**,
not a settled choice:

| | **A. Augmentation-based invariance** (preferred prior) | **B. Geometric canonicalization** |
|---|---|---|
| How | feed the spot mask *as-is*; train the CNN to be invariant by heavily augmenting rotation, stretch/shear, and blur | physically rotate each spot to its principal axis and normalize scale *before* encoding, so the CNN sees a canonical shape |
| Pros | no fragile pre-processing; handles stretch **and** blur (which shift a PCA axis); no sign ambiguity; robust to ragged/occluded contours | invariance "for free", nothing to learn, more sample-efficient with little data |
| Cons | spends model capacity learning invariance; invariance only approximate | principal axis is **unstable for near-round spots**, has a **180° sign ambiguity**, and **breaks under stretch/blur/occlusion** — the very nuisances we care about |

**Plan: implement both behind the same spot-encoder interface and evaluate head-to-head.** The
prior is A (augmentation-heavy, no geometric normalization of shape); B (canonicalize-to-principal-
axis) is the control it must beat. This is a clean ablation — same aggregator, same data, swap only
the shape front-end — and it directly answers "did we actually need geometric normalization?"

> **Note the stretch trade-off.** Full stretch-invariance treats a foreshortened spot as identical
> to its head-on view (good for pose robustness) but also discards genuine aspect-ratio cues. The
> A/B experiment measures whether that trade is net-positive; it is not obvious a priori.

Optionally concatenate **topology hand-features** (Fourier descriptors, Hu moments, solidity,
skeleton branch/endpoint counts) into the token — cheap insurance that distinctive shapes are
represented (see [§3](#3-making-distinctive-spots-count-more-q4)). Treat these as *another ablation
switch*, not a commitment.

### 2b. Relative position

Only **relations between spots** are identity — made invariant by construction:

- **Translation** — relative to the constellation centroid or to each spot's local neighbors.
- **Scale** — divide distances by a robust global scale (body length, or the *median* pairwise
  distance — median so it survives missing spots).
- **Rotation** — either canonicalize the constellation to its principal axis, or use rotation-
  invariant edge features (pairwise **distance** + **relative angle** differences).

Represent this as a **kNN graph** over centroids with edge features `(normalized_distance,
relative_angle)`, consumed by whichever aggregator ([§6](#6-architecture-and-technology-the-candidate-bake-off-q5))
is under test. How position enters differs per candidate: as **attention bias** (Set Transformer),
as **edge features** (GNN), as part of a **per-spot token** (Hungarian matching, 4.4). Prefer
**local** relations (kNN) over global ones — local neighborhoods survive occlusion
([§4](#4-robustness-to-missing-spots-q3)).

> The principal-axis **180° head/tail ambiguity** applies here too — resolve with a head cue or
> augment over it so the embedding is invariant to the flip.

---

## 3. Making distinctive spots count more (Q4)

A rare `L`/`M`/forked spot is a strong identity anchor; a generic round dot is weak. Three levers:

1. **Let the encoder *see* the distinction.** Optional topology hand-features
   ([§2a](#2-encoding-shape-and-relative-position-q2)) — skeleton branch/endpoint counts, Fourier
   descriptors — make forked/elbowed shapes explicit so they can't be averaged away.
2. **Weight by rarity (TF-IDF), HotSpotter-style.** Cluster spot shape-embeddings into a
   vocabulary, give each spot an **inverse-frequency weight** (rare shapes weigh more), and use
   those weights in pooling and in the geometric verifier.
3. **Learn distinctiveness.** A small head scores a spot high when it is **consistent across an
   individual's own photos** yet **rare across the population** — the definition of a good feature.

> **Caveat:** leaning on one distinctive spot hurts when grass hides it. Combine distinctiveness
> weighting with the redundancy of [§4](#4-robustness-to-missing-spots-q3).

---

## 4. Robustness to missing spots (Q3)

Spots vanish (grass, angle, extractor misses). Four layers:

1. **Cardinality-invariant matcher.** Set/graph aggregators and Hungarian matching all accept a
   variable spot count — a missing spot is just a missing token/node. Degrades smoothly.
2. **Spot-dropout augmentation** (core mechanism). Randomly delete spots from one view of a
   positive pair so the matcher is forced to align partial constellations; also inject spurious
   spots. See [§7](#7-augmentation-q1).
3. **Local, redundant descriptors.** Many overlapping kNN sub-constellations, not one global
   signature — lose a region, the rest still match (like a partial fingerprint).
4. **Partial matching at decision time.** The verifier scores a geometrically consistent *subset*
   (RANSAC / Hungarian with unassigned spots), not all spots. Below a **minimum visible-spot
   count**, the system **abstains** — which is exactly the confidence mechanism of
   [§5](#5-confidence-and-the-cannot-evaluate-state).

---

## 5. Confidence and the "cannot evaluate" state

**Requirement: the system must be able to say "this photo is too obscured / blurred / weird — I
can't tell," and attach a confidence to every decision.** The output is therefore *three-way*:
**match(X)** · **new individual** · **cannot evaluate (abstain)**, each with a calibrated
confidence in [0, 1].

Confidence is assembled from signals at four points in the pipeline:

1. **Input quality (before embedding).**
   - **Blur** — variance-of-Laplacian or a small learned quality head.
   - **Visible spots / body area** — too few spots or too much of the body out-of-frame/occluded →
     low confidence (this is the [§4](#4-robustness-to-missing-spots-q3) minimum-spot gate).
   - **Out-of-distribution input** — is this even a salamander dorsal view? Reject non-salamander
     or extreme-angle frames early.

2. **Embedding stability (after embedding, before matching).**
   - **Test-time-augmentation variance** — embed several augmented views of the same photo; a large
     spread in the resulting embeddings means the representation is unstable → low confidence. This
     is model-agnostic and cheap, and pairs naturally with the augmentation stack.
   - Optionally MC-dropout / small-ensemble variance for the same signal.

3. **Match decision (at retrieval).**
   - **Nearest-neighbour margin** — the gap between the top-1 and top-2 gallery distances; a tiny
     margin means "ambiguous between candidates."
   - **Absolute similarity vs. the open-set threshold** — a top-1 sitting right at the threshold is
     inherently uncertain (match-vs-new coin-flip).
   - **Verifier inliers** — how many spots geometrically agree; and whether independent local
     neighborhoods *vote for the same individual*.

4. **Calibration & the abstain rule.** Fuse the signals into one calibrated score (a small logistic
   model over the signals, temperature/Platt-scaled on held-out pairs). **Abstain** whenever input
   quality or embedding stability falls below its operating threshold — *before* forcing a
   match-vs-new decision. Report a **risk–coverage curve** (accuracy vs. fraction of photos
   answered) so a field user can dial precision against coverage. This becomes a first-class
   evaluation axis in [§9](#9-priority-and-evaluation-plan).

---

## 6. Architecture and technology: the candidate bake-off (Q5)

**This is the current main focus.** Five matchers, evaluated head-to-head through one shared
harness ([§9](#9-priority-and-evaluation-plan)). Every candidate must expose the same interface —
either produce a **salamander embedding** (fast nearest-neighbour retrieval) or a **pairwise match
score** — so all are scored on the same metrics.

| # | Approach | Produces | Missing spots | Interpretability | Retrieval cost | Role |
|---|----------|----------|---------------|------------------|----------------|------|
| **4.5** | **SIFT / classical** (local-feature + RANSAC, and/or constellation/triangle matching) | pairwise score | RANSAC subset | high | O(N) per pair | non-DNN **floor** |
| **4.3** | **CNN baseline** — masked whole-image → CNN + metric learning | embedding | implicit | low | O(1) ANN | "is decomposition even needed?" floor |
| **4.1** | **Set Transformer** over spot tokens (ISAB + PMA), relative-position attention bias | embedding | graceful | attention weights | O(1) ANN | core candidate |
| **4.2** | **GNN / GAT** over kNN spot graph, edge = relative geometry | embedding | graceful | edge attention | O(1) ANN | core candidate |
| **4.4** | **Per-spot embed + Hungarian matching** — no pooling; optimal assignment between spot sets | pairwise score (+ coarse embed) | unassigned spots | **very high** | O(N) match (coarse-filter first) | core candidate |

**4.1 Set Transformer.** Spot tokens → self-attention (ISAB) → pooling-by-multihead-attention (PMA)
→ one embedding. Global all-pairs reasoning over spots; relative geometry injected as attention
bias. Tests: *does global attention over spots beat a local graph or explicit matching?*

**4.2 GNN / GAT.** kNN spot graph, edge features = `(distance, relative angle)`, message passing →
pooled embedding. The most literal encoding of "relative position on edges." Tests: *does explicit
local relative-geometry beat the Set Transformer's global attention?*

**4.3 CNN baseline (holistic).** Masked image → CNN + ArcFace/triplet → embedding; no explicit
spots. Cheap, standard, and the honest control: **if a plain masked-image embedding matches the
per-spot models, the extra machinery isn't justified.** Must be beaten before 4.1/4.2/4.4 earn
their complexity.

**4.4 Per-spot embedding + Hungarian matching.** Embed each spot as `(shape ⊕ relative-position)`
and **do not pool**. Match two salamanders by **optimal bipartite assignment** (Hungarian) between
their spot-embedding sets; the match score is the assignment cost, with a dummy/threshold node so
unmatched spots are simply *unassigned* (handles missing/extra spots directly). Pros: **the most
interpretable** (explicit spot-to-spot correspondences) and the most natural partial-match. Cons:
**no single vector → 1-to-many retrieval is O(N) Hungarian solves**; mitigate with a two-stage
scheme — a coarse pooled embedding pre-filters top-k, then Hungarian re-ranks. This is essentially
a *learned* version of the classical constellation match (4.5). Tests: *does explicit set-matching
beat learned pooling?*

**4.5 SIFT / non-DNN traditional.** Two flavours: (a) SIFT/ORB keypoints on the masked spot image +
RANSAC geometric verification; (b) classical constellation/triangle matching on centroids
(HotSpotter / I3S / star-matching). No training, fast to stand up, interpretable — and possibly
strong. It is the **floor every learned model must clear**, and doubles as a ready verifier/re-rank
stage.

Plus a **Gemini-as-matcher reference**: literally ask Gemini "same individual?" This is the
worst-case fallback the user noted, and a useful (if slow, non-deterministic) upper reference for
how hard each pair is.

---

## 7. Augmentation (Q1)

Augmentation is the **training-signal generator**: with 203/290 individuals appearing once, it
**manufactures positive pairs** — many plausible views of one animal from a single photo. Applied
on the fly, two views per sample per epoch, at both levels.

**Geometry-space (on the spot set — primary):** global similarity + mild affine + **thin-plate-
spline** warp (pose/body-bend); per-spot rotate/stretch/shear (angle, foreshortening); centroid
jitter (annotation noise); **spot dropout + spurious-spot injection** (*the single most important
one* — drives missing-spot robustness and the [§2a](#2-encoding-shape-and-relative-position-q2)
rotation/stretch/blur invariance); partial contour occlusion (grass, mud).

**Appearance-space (on the mask/patch pixels):** brightness/contrast/gamma, colour/white-balance
jitter (lighting); gaussian/motion/defocus blur + down-then-up resample (**blur invariance for
[§2a](#2-encoding-shape-and-relative-position-q2)**); background randomization via the mask (forces
background invariance).

**Wiring:** positives come from **both** augmentations of one photo **and** different real photos of
the same individual; curriculum from mild → strong; oversample the 87 multi-photo individuals for
real positives. **Cautions:** **no mirror flips** (reflection reverses pattern chirality = a
different identity); don't warp so hard that different individuals become confusable.

---

## 8. Design considerations (Q6)

- **Orientation is an experiment, not a setting.** Ship both the augmentation-invariant and the
  principal-axis-canonicalized spot encoder behind one interface and let the harness decide
  ([§2a](#2-encoding-shape-and-relative-position-q2)).
- **Confidence is a first-class output.** Every decision carries a calibrated score and an
  **abstain** path ([§5](#5-confidence-and-the-cannot-evaluate-state)); "cannot evaluate" is a
  valid, desirable answer, and coverage-vs-accuracy is a reported metric.
- **Extraction is off the critical path.** Gemini stays until matching is solid
  ([§1](#1-spot-extraction-deferred-q7)); design the matcher to tolerate its noise.
- **Keep shape and geometry decoupled** — absolute `(x, y)` never enters a token; that discipline
  delivers the pose/translation invariance.
- **Retrieval-cost trade-off.** Embedding matchers (4.1/4.2/4.3) give O(1) ANN search; Hungarian
  (4.4) and classical (4.5) are O(N) per comparison — plan the coarse-embed pre-filter if a
  matching-based candidate wins.
- **Split honestly** — by individual, and ideally by **source/session** (`aj_1` and `aj_2` are
  different animals that may share a shoot/background); background masking mitigates leakage,
  session splits prove it.
- **Label noise is real** (Gemini spots unverified) — defend with add/remove-spot augmentation,
  robust losses, and a **small hand-checked validation set**.
- **Open-set calibration** — the 87 multi-photo individuals are the entire budget for setting the
  match/new threshold; use cross-validation, not a single split.
- **Enrollment strategy** — multiple embeddings per individual or a running centroid; DuckDB fits
  the existing stack for the gallery/vector store.
- **Interpretability builds trust** — per-spot embeddings (esp. 4.4) let you *show which spots
  matched*; invaluable for a field researcher.
- **CPU-first, modular** — small models; extraction → embed → aggregate → match/verify as four
  swappable stages.

---

## 9. Priority and evaluation plan

The goal is a **fair bake-off**: build the shared scaffolding first, then plug each candidate in and
compare on identical data and metrics. Priorities run **P0 (blocking) → P3**.

### The shared harness (P0 — build first, blocks everything)

- **Splits:** by individual, and a second split by source/session; k-fold CV over the 87 multi-
  photo individuals (too few for a single split).
- **Metrics, computed identically for every candidate:**
  - *Identification:* rank-1 / rank-5 / mAP.
  - *Verification:* ROC-AUC, TPR @ fixed low FPR.
  - *Open-set:* AUROC / DIR@FAR for novel-individual detection (hold out whole individuals).
  - *Confidence:* **risk–coverage curve** (accuracy vs. fraction answered) — from
    [§5](#5-confidence-and-the-cannot-evaluate-state).
- **Candidate interface:** each returns *either* an embedding *or* a pairwise score; the harness
  wraps both into the same retrieval/verification evaluation.

### Phased plan

| Phase | Priority | Work | Decision gate |
|-------|----------|------|---------------|
| **0** | **P0** | Harness, splits, metrics, augmentation pipeline, DuckDB gallery | metrics reproducible on a dummy matcher |
| **1** | **P0** | **Baselines / floors:** 4.5 classical, 4.3 CNN, Gemini-as-matcher reference | establishes the score every learned model must beat |
| **2** | **P1** | **Spot encoder + orientation A/B** ([§2a](#2-encoding-shape-and-relative-position-q2)): augmentation-invariant vs. principal-axis-canonical, same aggregator | pick the shape front-end |
| **3** | **P1** | **Aggregator bake-off:** 4.1 Set Transformer vs. 4.2 GNN/GAT vs. 4.4 Hungarian, on the chosen front-end | pick the primary matcher (and whether it's embedding- or matching-based) |
| **4** | **P2** | **Confidence & open-set:** quality/stability/margin signals, calibration, abstain, risk–coverage | reliable "cannot evaluate"; calibrated match/new threshold |
| **5** | **P2** | **Verification re-rank:** geometric verifier (4.5-style) over the top-k of the winner | precision lift at fixed coverage |
| **6** | **P3** | **Revisit extraction** ([§1](#1-spot-extraction-deferred-q7)) — replace Gemini with classical/distilled — *only if* it is now the bottleneck | offline/CPU determinism without accuracy loss |

### Guiding principles

- **Baselines before models.** 4.5 and 4.3 are cheap and set the floor; if classical CV already
  matches the neural candidates, that is the most important result of the study.
- **Change one thing at a time.** Fix the harness, then vary the front-end (Phase 2), then the
  aggregator (Phase 3) — so every comparison is attributable.
- **Ablate the switches** flagged inline: topology hand-features on/off, geometry decoupled vs.
  fused, local vs. global relations, verifier on/off.
- **Report abstention honestly.** A model that answers 70% of photos at 98% accuracy may beat one
  that answers 100% at 90% — the risk–coverage curve, not a single accuracy number, decides the
  winner.
