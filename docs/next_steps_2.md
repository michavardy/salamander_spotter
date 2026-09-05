# Improvements and Next Steps

1. **Multi-Stage Comparison & Hierarchical Cascade**
   * **Stage 1 (Coarse Exclusion):** Rapidly prune candidate pairs that are definitely not a match using light, global criteria (e.g., severe aspect/size mismatch, coarse feature filters).
   * **Stage 2 (Detailed Match & Uncertainty Modeling):** For surviving candidates, apply fine-grained matching. Compute explicit uncertainty estimates (e.g., confidence scores per spot cluster, overall match variance) to distinguish ambiguous pairs from definite matches.

2. **Global Configuration & Constellation Matching**
   * **Finding:** Local spot descriptors are exceptionally strong ($0.997$ separation on true vs. random pairs), but human rejections ($0.484$ score) were driven by overall body posture and global spot layouts rather than individual spot appearance.
   * **Action:** Shift from pure local-feature matching to  constellation-matching framework that enforces spatial relationships and global geometric consistency across all detected spots.
    * Strategy 1: Learn Deformation Invariants from Pairs of the Same AnimalIf you have a small     dataset of identical salamanders photographed in both straight and curled postures, you don’t     need to force a rigid unwarping step—you can learn the deformation manifold directly.1.     Geodesic / Along-Spine Distance MatrixInstead of computing standard Euclidean distance between     spots:Spine Curve Fitting: Fit a smooth spline through the head-to-tail body axis.Curvilinear     Coordinates: Project each spot onto the spine ($s_i = \text{distance along spine}$) and measure     its lateral offset ($d_i = \text{perpendicular distance from spine}$).Result: While Euclidean     distances contract or expand when a salamander curls, $(s_i, d_i)$ coordinates remain virtually     invariant regardless of posture.       Straight Posture                         Curled Posture
      
        • (s1, d1)                             • (s1, d1)
        |                                     /
      ====== Spine ======                  ~~~~~~ Curved Spine ~~~~~~
        |                                   /
        • (s2, d2)                         • (s2, d2)
    . Pairwise Deformation Learned WeightsWith real straight vs. curled pairs:Calculate distance     atrices $D_{\text{straight}}$ and $D_{\text{curled}}$ for the same animal.Learn a Deformation     ernel $K(i, j)$ that measures how much the distance between spot $i$ and spot $j$ is expected to     hange given their relative positions along the body axis (e.g., spots near the tail move more     han spots along the mid-dorsal trunk).Strategy 2: Synthetic Curling Augmentation & UnwarpingIf     eal straight vs. curled pairs are scarce, you can synthetically warp straight salamanders to     imulate realistic curling.A. Synthetic Data Generation PipelineTake well-aligned "straight"     alamander images and spot maps.Fit a skeletal spine (1D curve with 3–5 control points).Apply a     olar / Bending Transformation along the spine (varying curvature $\kappa$ and bending angle     \theta$).Generate thousands of synthetic $(I_{\text{straight}}, I_{\text{curled}})$ spot-map     airs to train or validate your structural ranker.B. Thin Plate Splines (TPS) / Constellation     lignmentWith synthetic or learned curling parameters, you can run a Posture-Aware Constellation     erification:[Top-10 Candidates] ──► [Align Spine / Unwarp] ──► [Measure Spot Residuals] ──►     Re-rank Top 1]


3. **Low Spot Survival & Partial-Match Scoring**
   * **Finding:** Only $24\%$ of individual spots persist across two photos of the same animal due to occlusion, body posture, or spot dynamics.
   * **Action:** Redesign the match scoring function to allow partial matching, preventing the algorithm from heavily penalizing valid pairs that share only a small subset of persistent spots.

4. **Multi-Animal Detection & Image Pre-Filtering**
   * **Finding:** Failure cases frequently occur when multiple salamanders occupy a single frame, and the legacy multi-animal detector failed completely.
   * **Action:** Implement a dedicated object detection/instance segmentation pipeline (e.g., YOLO or Mask R-CNN) as a front-end pre-processor to isolate individual animals—or exclude unsegmentable, crowded frames—before feature extraction.

5. **Dynamic Geometric Weighting Based on Image Quality**
   * **Finding:** Image quality directly modulates how much geometric structure can be trusted, whereas body curl and camera angle impact matchability far less than assumed.
   * **Action:** Build an automated image quality scoring module that dynamically relaxes or down-weights geometric rigidness constraints on blurry or low-resolution photos.

6. **Feature Weighting Pruning & Labeling Refocus**
   * **Finding:** "Rarity" weighting introduced noise by correlating with temporary/vanishing spots, and recent spot-click annotations yielded diminishing returns (the last 92 clicks added zero signal).
   * **Action:** Strip rarity weighting from the matching logic. Redirect annotation efforts away from individual spot clicks and toward high-level pair alignment, multi-animal bounding boxes, and coarse candidate verification.
   * **Status (2026-08-16):** rarity is **out** — `DEFAULT_WEIGHTS` had it at 0 since 2026-08-15, but three consumers still read the legacy blend (the sweeps' hand arm, the review UI's interest cache, the synthetic oracle) and are now fixed; see [results.md](../results.md) #46 *Correction*. Annotation redirect: targets ranked in [labeling_priorities.md](labeling_priorities.md); `interesting-spot-selector` is marked deprecated, and **coarse candidate verification is built** — `pair-review` now records `match`/`different`/`unsure` + reason chips per pair (70 pairs rendered, 0 judged), read back via `review_labels.pair_verdicts`. It also feeds §4: its `multi-animal` chip is confirmed ground truth for the frames `data_hygiene.multi_animal_candidates` can only guess at. Still unbuilt: bounding-box drawing (§4) and body-axis anchor editing.

