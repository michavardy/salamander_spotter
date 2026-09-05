# Tricks

Techniques that have measurably moved a number in this project, kept in one place so they can be
reused instead of rediscovered. Every entry cites where it was measured — a lesson number in
[results.md](../results.md) §7, or an artifact under `artifacts/`.

They come in two families, and the distinction matters:

- **§A raises the number.** Real gains on retrieval, census F0.5 or coverage.
- **§B keeps the number honest.** These usually *lower* a headline — that is the point. They are
  what make the §A gains believable rather than a rediscovery of the protocol.

A trick only earns a place here once something measured it. "It should help" belongs in
[next_steps_2.md](next_steps_2.md).

---

## A. Tricks that raise the number

| # | trick | measured |
|---|---|---|
| A1 | **The magenta trick.** Ask the model to *recolour* spots flat magenta and key the colour in OpenCV, instead of asking for coordinates. Annotation becomes a masking task, which the model is genuinely good at, and the rest is deterministic. | #1 |
| A2 | **Ask for a region, never a line.** Requesting two flank curves got 73–93% rejected across three model rungs. Requesting a filled mask and *deriving* outline, halves, centre line and tips removed three of four failure modes and the LLM judge entirely. | #2 |
| A3 | **Learn the aggregation, not the representation.** Keep the spot embeddings fixed, learn the *voting rule* over individual-agnostic match statistics. +0.17 R@1 over raw voting, positive on all five folds, trains in seconds. | #23 |
| A4 | **Filter the evaluation, not the training pool.** `TRAIN_POOL=full` beats `TRAIN_POOL=filtered` on every metric at every threshold. Bad photos are bad queries but perfectly good training data. | #14 |
| A5 | **Weight spots by distinctiveness, and take the weights from a human.** `coverage × support` with a distinctiveness weight beat equal-weight voting (0.347 vs 0.300). Removing the human special-spot supervision drops census F0.5 0.609 → 0.493 and novelty AUROC to chance. | #31 |
| A6 | **Fit the weight, don't eyeball it — then check what it says.** A six-factor logistic regression recovers the human "interesting spot" labels at 0.748 vs 0.707 for the hand blend, and revealed that two hand-chosen factors (`rarity` 1.5, `noncircularity` 1.0) fit at ~0 and *negative*. `rarity` is now 0.0 in `DEFAULT_WEIGHTS`. | #46 |
| A7 | **Mine self-supervised positives from real correspondences, not augmentations.** Correspondence-mined 0.375 R@1 > augmentation positives 0.277 > frozen hand-features 0.227 ≫ random-init 0.089. | #29 |
| A8 | **Prefer breadth to peak.** The learned rule put +1.16 on `frac≥.6` and **−0.68** on `max_sim`: many moderate, mutually consistent matches beat one strong one. | #25 |
| A9 | **Free repairs before paid ones.** `correct-axis` re-tips a collapsed axis from the saved mask — no model, no cost, reversible. `repurple` re-bills only poorly-scoring images and keeps the result *only if it improves*. | #7 |
| A10 | **Buy positive pairs with synthetic views.** Gemini re-renderings took singletons 427 → 75 and added 855 positive pairs. Training-only, behind a self-consistency gate — they buy pairs, not truth. | #11 |
| A11 | **Subtract each comparison's own permutation null.** Any "how consistent is this?" score computed over a *variable number* of items is confounded with that number. Correcting it lifted `ransac_frac` 0.544 → 0.621 and `nbr_agree` 0.424 → 0.619 with no new data. **Written up in full below.** | [constellation](../artifacts/spot_transformer/constellation/RESULTS_constellation_check.md) |

## B. Tricks that keep the number honest

| # | trick | measured |
|---|---|---|
| B1 | **Anchor every harness with a dummy and an oracle.** Random embeddings must sit at chance, a label-reading oracle at 1.0. If the dummy ever scores well, something leaks and no other number can be believed. `pixi run emb-selfcheck`. | #18 |
| B2 | **Never quote R@1 without the gallery size.** The same matcher scores 0.603 against ~65 candidates and a fraction of that against ~687. Chance itself moves from 0.071 to 0.010. | #12 |
| B3 | **Bootstrap over individuals, not queries.** Queries from one animal are correlated; query-level CIs are two to three times too narrow. | #15 |
| B4 | **Run a control arm that must fail.** A random-init encoder next to the trained one (#29); a shuffled correspondence next to the real one (A11). The control is what makes a modest number readable — and in A11 it is what caught the bug. | #29 |
| B5 | **Report F0.5, but don't tune on it.** A false MATCH deflates the population count irreversibly, so precision gets 2× weight — but thresholding on F0.5 rewards labelling everything "new". Threshold on balanced accuracy, report count bias alongside. | #16, #17 |
| B6 | **`nan` is a result, and recall 0 is the tell.** `e2e_mlp` and `pretrained_cnn` produced no matches at all: precision `nan` over an empty set, not a good score. | #20 |
| B7 | **Screen before you sweep.** `repr-check` judges a descriptor against 417 human verdicts in *seconds*, so twenty ideas fit in an afternoon and only the survivors cost a multi-hour census run. | #43, #44 |
| B8 | **Pre-register the gate before running the measurement.** `run_feasibility.sh` states "AUROC ≥ 0.65 → build it, below → don't" in the script, before any number exists. It is much harder to talk yourself into a 0.61 afterwards. | `run_feasibility.sh` |
| B9 | **A quality filter shrinks the gallery, so it flatters itself.** Pin the gallery size and re-measure; here the effect survived (+0.15–0.20 R@1 at a fixed 14-individual gallery), but only the controlled protocol proves it. | #13 |
| B10 | **Diagnose before optimising.** D1 found 54 suspected duplicate identities and a third of individuals at chance-level self-consistency — routing effort to label hygiene rather than architecture. | #40 |
| B11 | **Calibrate a simulator on statistics you did not fit.** The first synthetic population scored R@1 = 1.000 and described a task nobody has. Relative claims survive calibration error; absolute ones do not. | #37 |

---

# A11 in full — subtract each comparison's own permutation null

## The finding

Two of the matcher's geometry features are **confounded with the number of matched spots**, badly
enough that one of them scored *worse than chance* before correction.

Measured on 3,020 photo pairs (755 true / 2,265 false), real photos only:

| feature (image frame) | scrambled control | raw | corrected |
|---|---|---|---|
| `ransac_frac` | **0.377** | 0.544 | **0.621** |
| `nbr_agree` | **0.374** | 0.424 | **0.619** |
| `geom_spearman` | 0.523 | 0.555 | 0.555 |

All numbers are AUROC on true-vs-false photo pairs: 0.5 is a coin flip, 1.0 is perfect.

The **scrambled control** is the same pair, the same spots, the same feature — with the
correspondence deliberately permuted into nonsense. It should score 0.5. `ransac_frac` scored
**0.377**: with all real information destroyed, it still separated the classes, *backwards*.

## Why it happens

`ransac_frac` asks "do the matched spots form the same arrangement in both photos?" — measured as
the fraction of points that a fitted transform explains. That question gets easier the fewer points
there are, the same way any two dots lie on some straight line while five rarely do. So the score
partly measures **how few matches were found**, and the fitted transform spends 4 degrees of freedom
on 2 sampled points before any point is left over as evidence.

Straight from `pair_frames.csv`, both classes pooled:

| matched spots | mean `ransac_frac` |
|---|---|
| 3 | 0.693 |
| 4 | 0.567 |
| 5 | 0.503 |
| 6 | 0.471 |

More corroboration, *lower* consistency score.

And the confound points the wrong way, because true pairs match more spots than false ones — mean
**6.27 vs 4.30** on the 1,741 pairs where geometry is defined. So the thin, easy-to-satisfy
comparisons are disproportionately the *wrong* animals, and the raw feature rewards them.

`geom_spearman` is nearly clean (null 0.523) and barely moves under correction. That is the useful
contrast: it is a **rank correlation over all pairwise distances**, which is far less sensitive to
how many points there are than a **fitted-model inlier fraction** is. Rank statistics resist this;
"fraction explained by a fitted model" invites it.

## The correction

For each pair: compute the feature, then recompute it a few times with that same pair's
correspondence permuted, and subtract the mean. What is left is *how much better than luck, for a
comparison with exactly this many matches* — the count advantage cancels because the null carries it
too.

```python
score  = feature(Q, C)
null   = mean(feature(Q, C[permutation]) for _ in range(n_perm))
excess = score - null                       # what the gate reads
```

Three permutations is enough here; the whole screen runs in ~3 minutes over 3,020 pairs
(`pixi run constellation-check`, `--perm` to change it). The per-pair null is doing the work — a
single global control arm can *detect* this confound but cannot remove it, because the correction
has to be conditional on each comparison's own item count.

## Why it matters beyond the screen

[results.md](../results.md) #25 records that the learned aggregator puts **+0.51** on `ransac_frac`
and reads it as *"it uses RANSAC constellation consistency as real evidence."* That reading is now
doubtful: some of what the feature supplies is a match count wearing a geometry costume. The model
does receive `log_nq` / `log_nc` / `mutual_count` separately and #25 notes it learned to
size-normalise with them — but that is a *linear* correction to a relationship that is a curve
(0.69 → 0.47 → 0.58 across counts 3–8).

Normalising the feature is a small edit to something already shipping: no new coordinates, no new
labels. One `sweep_all9` run says whether the screen-level +0.08 survives on the census metric.

## Where it bites hardest — check the regime before assuming the gain

The confound is strongest where the item counts are small and vary a lot. The numbers above come
from **image-to-image** pairs with correspondences thresholded at cosine ≥ 0.4, where the median
true pair has 3–6 matches — the steep part of the curve.

The deployed aggregator runs in a different regime: a candidate is an **individual**, so its spots
are pooled across all its photos, and `match_features` thresholds nothing. Median mutual-match
count there is **10**, and the curve is much flatter above ~6. So the correction is expected to buy
less in `match_features` than the screen showed, and the sweep — not the screen — is what settles
how much.

The transferable rule is the diagnosis, not the effect size: **measure the null in the regime you
will deploy in.** A confound that dominates at n=3 can be nearly absent at n=10, and the reverse.

## Where else to look for it

The signature is: **a score that is a fraction or a rate, computed over a set whose size varies
between the things being compared.** Candidates worth the same control:

- `ransac_frac`, `nbr_agree` — confirmed above.
- `mutual_frac` — a ratio to `nq`, so it is normalised by construction, but the *numerator's*
  difficulty still varies with candidate spot count.
- Any future constellation feature. Triplet and angle invariants have exactly this shape, and would
  inherit the fault silently.
- The strict scorer's `support = 1 − exp(−n_good/τ)` is the deliberate, opposite move: it exists
  *because* few corroborating matches should score low. Worth knowing the two terms pull against
  each other.

The general rule, which is why this sits in §A rather than §B: when a consistency score can be
satisfied by having less evidence, measure it against a null built from the same amount of
evidence — otherwise the feature will quietly reward sparse comparisons, and sparse comparisons are
usually the wrong answer.

---

*A11 came out of Phase 0 of the constellation plan
([RESULTS_constellation_check.md](../artifacts/spot_transformer/constellation/RESULTS_constellation_check.md)),
whose actual hypothesis — that the geometry features were failing because they run in image pixels
rather than body coordinates — was **not supported**: every frame-swap delta came in under +0.02
with a CI spanning zero. The control that was supposed to be a formality was the finding.*
