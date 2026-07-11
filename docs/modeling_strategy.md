# Modeling Strategy — Deep Learning for Salamander Identification

A high-level design discussion for the identification model. For *what* we are building and
*why*, see [project_goal.md](project_goal.md); this document is about *how* — the representation,
the augmentation strategy, the candidate models, and the trade-offs between them.

---

## 1. The core modeling insight

A salamander's identity **is** its constellation of yellow spots. Two things carry the signal,
and the model must use **both**:

1. **Spot shape** — the local appearance of each individual spot (its contour: elongated,
   round, forked, ragged).
2. **Relative arrangement** — where each spot sits **in relation to the other spots**, not its
   absolute pixel coordinates. The pattern `A is above-left of B, which is twice as far from C`
   is identity; the raw `(x, y)` of `A` is not.

Everything else in the photo is **nuisance** and must be discarded: the background, the body
outline/shape, the pose, the lighting, the camera angle. The whole design problem is:

> **Build an embedding that is sensitive to (spot shape + relative geometry) and invariant to
> (background, body shape, pose, lighting, scale, translation, and in-plane rotation).**

Because the arrangement must survive a salamander photographed head-up vs. head-down and near
vs. far, the required invariances are at minimum **translation, scale, and full 360° in-plane
rotation**, plus tolerance to **non-rigid deformation** (a salamander is a curved 3-D body, so
its 2-D spot layout warps with pose).

This is the same problem class as **fingerprint minutiae matching**, **star-constellation
matching**, and **spot-patterned wildlife re-ID** (whale sharks, manta rays, leopards). Notably,
whale-shark ID adapted an *astronomy* star-pattern algorithm (Groth) to match spot
constellations, and Wild Me's **HotSpotter/Wildbook** does patterned-animal ID via local
features + spatial verification. That prior art shapes the recommendations below.

---

## 2. Design principle: *remove* nuisance, don't just *hope* the network ignores it

We have a major asset most re-ID projects lack: **the spots are already segmented** (contours +
per-spot masks in `db/contours.db`, produced by this repo's pipeline). So we can **structurally**
strip out nuisance instead of hoping metric learning discovers invariance from limited data:

- **Mask the input.** Blacken everything except the spots (or spots on a neutral body). This
  physically removes background and most body-shape cues before the network sees a pixel. (This
  mirrors the sibling `salamander_id` "blacken the background" stage.)
- **Or drop pixels entirely** and feed **geometry**: each spot as a shape descriptor + a
  relative-position encoding. Then background is irrelevant *by construction*.
- **Canonicalize the constellation** (center on its centroid, scale by body length / mean
  inter-spot distance, optionally align to the principal axis) to hand the model translation +
  scale (+ rotation) invariance for free.
- **Then** use augmentation to mop up the residual nuisance (lighting, blur, deformation) that
  masking and canonicalization don't remove.

Masking + geometry does the heavy lifting; augmentation makes it robust. Relying on augmentation
alone (feeding raw photos and praying) is the weakest option given only ~445 images.

---

## 3. Representation strategy — three options

| Option | What it embeds | Pros | Cons |
|--------|----------------|------|------|
| **A. Holistic** | One CNN/ViT embedding of the whole (masked) image | Simplest; fast NN search; standard re-ID | Must learn spot-focus + invariances implicitly; weak with few-shot; no natural spot-level output |
| **B. Per-spot + aggregation** | Embed each spot, then aggregate into a salamander embedding via set/graph/transformer | Directly encodes *shape + relative geometry*; yields **both** spot and salamander embeddings; robust to missing spots | More moving parts; needs the spot set as input (we have it) |
| **C. Classical geometric matching** | Spot centroids/shapes as a labeled point set; match via graph/triangle matching + RANSAC | Interpretable; invariant by construction; robust to occlusion; strong non-DL baseline | Not a compact embedding; slower 1-to-many; hard to scale retrieval |

**Recommendation: Option B as the primary model, with C as the verification stage.** Option B is
the natural fit for the user's requirement that *both spot shape and relative position* matter,
and it produces exactly what's wanted: **an embedding function that can match spots or match
salamanders, preferably salamanders** — the per-spot embeddings fall out as a by-product of the
salamander-level embedding.

### The two-level architecture (recommended)

```
per spot:   spot patch/mask ─► small CNN ─┐
            contour shape features ───────┤─► spot token (shape embedding)
            relative-position encoding ───┘        │
                                                    ▼
constellation:   {spot tokens} ─► Set Transformer / GNN ─► attention pool ─► salamander embedding
                                     (spots = nodes/tokens;
                                      edges/attention bias = relative geometry)
```

- **Spot level** gives per-spot embeddings → used for spot-to-spot correspondence and geometric
  verification (Option C), interpretability, and partial matches when few spots are visible.
- **Salamander level** aggregates them (attention pooling / NetVLAD / DeepSets) into the primary
  identity vector used for fast nearest-neighbor retrieval.

Because aggregation is **permutation-invariant** and tolerant to a variable number of tokens, the
model is naturally robust to spots being **added, missed, or removed** — which is exactly the
failure mode augmentation (§5) will train against.

### Encoding "relative, not absolute" position

This is the crux. Options, roughly increasing in power:

- **Invariant hand features** — pairwise inter-spot distances and angles, or
  affine/similarity-invariant **triangle features** (the star-matching trick). Inherently
  translation/rotation/scale invariant.
- **Relative positional bias in attention** — feed spot tokens with attention biased by the
  relative offset/distance between spot *i* and *j* (like relative position encodings in
  transformers). The model learns which relations matter.
- **Graph over the constellation** — spots = nodes, edges = k-nearest neighbors with edge
  features = relative geometry; a message-passing GNN reads the arrangement directly.

Feed the model **relations between spots**, never raw `(x, y)`. Center + scale-normalize first;
handle rotation either by an invariant feature set or by learning it through rotation
augmentation.

---

## 4. Augmentation strategy

Augmentation is not a regularization afterthought here — **it is the mechanism that makes the
project viable.** With **203 of 290 individuals appearing in a single photo**, we cannot learn
"same vs. different" from real positive pairs alone. Augmentation lets us **manufacture positive
pairs**: many synthetic "views" of the same animal from one photo. Combined with the 87
multi-photo individuals (real positives), this is what supplies the training signal.

We augment at **two levels**, matching the two-level representation.

### 4a. Spot-level augmentation (operates on the spot set / masks / geometry)

Applied to the spots themselves — cheap, and the mask labels make it exact.

| Augmentation | Models / simulates | Notes |
|--------------|--------------------|-------|
| **Rotate each spot** | body curvature, viewing angle | small per-spot rotation of the contour |
| **Stretch / shear each spot** | foreshortening on a curved 3-D body | anisotropic scale + shear of the contour |
| **Translate spots slightly** | annotation jitter + real micro-deformation | small centroid jitter |
| **Occlude spots partially** | mud, water sheen, debris, shadow | erase part of a spot's contour/mask |
| **Remove spots entirely** | missed detections, spots hidden by pose | drop a random subset — **critical** for missing-spot robustness |
| **Add spurious spots** | false-positive detections | inject a few fake spots to model detector noise |
| **Warp the whole constellation** | overall pose / body bending | global similarity + mild affine, and **thin-plate-spline / elastic** warp of all centroids together |

The global TPS/elastic constellation warp is the single most important one for pose: it turns one
photo into a family of plausibly-deformed-but-same-identity constellations.

### 4b. Image-level augmentation (operates on pixels)

Applied when a pixel branch is used, to produce the "same salamander, different photo" views.

| Augmentation | Models / simulates |
|--------------|--------------------|
| **Rotation (full 360°), perspective/homography, elastic/TPS warp** | different angles and poses |
| **Brightness / contrast / gamma, color & white-balance jitter, synthetic shadows** | lighting conditions |
| **Gaussian / motion / defocus blur, down-then-up resample** | blur and low resolution |
| **Background randomization** | different backgrounds — use the mask to composite the salamander onto random scenes (or blacken it) so the network *cannot* rely on background |
| **Gaussian/ISO noise, JPEG compression** | sensor and compression artifacts |
| **Random erasing / cutout** | a hand, leaf, or twig covering part of the body |

**Background randomization via the existing masks is the highest-value image augmentation** — it
directly enforces the "ignore the background" requirement.

### 4c. Two cautions (identity-preserving vs. label noise)

- **Reflections/mirror flips are *not* safe.** A left-right flip of a dorsal photo produces a
  *chirally reversed* pattern that no real view of that animal can produce — spot-arrangement
  handedness is part of identity. Use **rotations** (which are genuine transforms; head-up vs.
  head-down is a real 180° rotation) and **non-reflective warps**; avoid mirror flips unless you
  first confirm the pattern is bilaterally symmetric enough to tolerate them.
- **Don't over-warp.** Push deformation/dropout far enough for robustness but not so far that two
  *different* individuals become confusable — that injects label noise into the positive pairs.

---

## 5. Training objective

Given few-shot + open-set data, use **metric learning**, not classification, in two stages:

1. **Self-supervised pretrain (augmentation-defined positives).** SimCLR/DINO-style: two
   augmentations of the *same* photo are a positive pair; everything else is negative. This
   needs **no identity labels**, so it uses all 445 photos including the 203 singletons, and it
   directly rewards the invariances we designed the augmentations around.
2. **Supervised fine-tune (real positives).** Use the filename identity labels (the 87
   multi-photo individuals) with **triplet loss + hard/semi-hard mining** or **supervised
   contrastive** (both handle 1–2 examples per class, unlike a softmax classifier). Proxy losses
   like **Sub-center ArcFace** are an option but strain under many singleton classes — keep them
   as a comparison, not the default.

Train **both levels**: a spot-level contrastive term (corresponding spots close) plus the
salamander-level identity term, so the per-spot embeddings stay usable for verification.

Positives therefore come from **both** augmentations of one photo **and** different real photos
of the same individual — the model can't tell which, which is exactly the point.

---

## 6. Model / backbone options — the "best potential" discussion

| Component | Candidates | Recommendation |
|-----------|-----------|----------------|
| **Spot appearance encoder** | small CNN (ResNet-lite / EfficientNet-lite) on the spot patch or mask; ViT patch; DINOv2 features | small CNN — cheap, CPU-friendly, enough for a single spot; DINOv2 if we want zero-train features |
| **Constellation aggregator** | DeepSets (simplest) · **Set Transformer** (attention over spot tokens) · **GNN / GAT** (edges = relative geometry) · PointNet++ | **Set Transformer or GNN** — both encode relative geometry and are dropout-robust; GNN is the most literal fit for "relative position" |
| **Pooling to one vector** | mean, max, **attention pooling**, NetVLAD | attention pooling / NetVLAD (spot-count invariant, learns which spots matter) |
| **Holistic baseline (Option A)** | ResNet/EfficientNet or DINOv2 + ArcFace/triplet on the masked image | build as a **baseline** to beat |
| **Geometric verifier (Option C)** | graph/triangle matching + RANSAC on spot correspondences; HotSpotter-style spatial verification | use as the **re-rank + accept/reject** stage |

**Best overall bet:** per-spot small-CNN descriptors → Set-Transformer/GNN with relative-geometry
encoding → attention-pooled salamander embedding, trained with the two-stage objective in §5, and
paired with a classical geometric verifier for the final accept/reject. This gives the "match
spots or match salamanders, preferably salamanders" flexibility, stays CPU-friendly, and degrades
gracefully when spots are missing. Keep the holistic CNN (Option A) as the baseline that this
architecture must outperform before the extra complexity is justified.

---

## 7. Matching and open-set inference

The end product is an **embedding function** plus a decision rule:

1. **Enroll**: store each known individual's salamander embedding(s) in a gallery (a DuckDB
   vector store fits the existing stack).
2. **Query**: embed the new photo → **cosine nearest-neighbor** search for top-k candidates.
3. **Verify** (optional but recommended): re-rank the top-k with the geometric spot-constellation
   matcher (Option C) for precision.
4. **Decide**: if the best similarity clears a **calibrated threshold** → "same as candidate X";
   otherwise → **new individual**, enroll it. The threshold is what makes this open-set.

This "**fast embedding retrieval → geometric verification → threshold**" shape is the same
philosophy as the sibling `salamander_id` ("deep spot embeddings + geometric validation"), so the
two projects stay aligned: `salamander_spotter` produces the labeled spot data, and this strategy
consumes it.

---

## 8. Evaluation protocol

- **Split by individual** (no identity shared across train/val) so retrieval scores can't leak.
- **Identification**: rank-1 / rank-5 accuracy and mAP on the 87 multi-photo individuals.
- **Verification**: ROC / AUC and TPR at a fixed low FPR on positive vs. negative photo pairs.
- **Open-set**: novel-individual detection (AUROC, or DIR@FAR) — hold out entire individuals as
  "unseen" and check they're flagged new rather than forced onto the nearest gallery entry.
- **Ablations worth running**: masked vs. raw input; geometry-only vs. pixel-only vs. both; with
  vs. without each augmentation family; holistic (A) vs. two-level (B); with vs. without the
  geometric verifier.

---

## 9. Open questions and risks

- **Label noise.** Spots are machine-extracted, not hand-verified — false/missed spots are real.
  The add/remove-spot augmentations train against this, but a small hand-checked validation set
  would let us measure it.
- **How rigid is the pattern under pose?** If 2-D arrangement deforms a lot with body bending, we
  must lean harder on deformation-tolerant matching (soft geometric verification, TPS) and less
  on rigid invariants.
- **Reflection symmetry.** Whether mirror-flip augmentation helps or hurts depends on how
  bilaterally symmetric the dorsal pattern is — worth a quick empirical check before enabling it.
- **Threshold calibration** needs enough held-out positive/negative pairs to set the open-set
  operating point reliably; the 87 multi-photo individuals are the main budget for this.
