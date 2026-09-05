# Where an annotation hour goes

The per-spot click is finished as a source of signal. This doc says what to label instead, in
order, with the measurement behind each. It is the second half of
[next_steps_2.md](next_steps_2.md) §6; the first half (stripping `rarity`) is done.

## The finding that closed the old loop

[results.md](../results.md) #46: growing the interesting-spot clicks from **362 → 454 images**
(8,691 → 10,613 labeled spots) moved held-out distinctiveness AUROC **0.750 → 0.748**. The last 92
images of clicking produced nothing. That is #34 ("more images of the same quality do not help")
reappearing at the label level.

And it is not only the *interesting*-spot click that is saturated. #44 measured the other per-spot
label — the accept/reject on a spot↔spot correspondence — and found embedding cosine on accepted
edges is **0.645 ± 0.154** against **0.650 ± 0.167** on rejected ones (AUROC **0.484**, i.e.
nothing). The reviewer is judging those edges on structure the 62-dim embedding does not encode, so
no amount of edge supervision can move a scorer built over it. Fitting the gate on all 427 verdicts
moved F0.5 by **+0.002**.

**Both per-spot label kinds are blocked on the representation, not on volume.** What is scarce is
animals (#35) and representation (#36).

## Stop

| label | tool | why stopped |
|---|---|---|
| interesting-spot clicks | `interesting-spot-selector` (deprecated), the ring gesture in `preprocess-review` | −0.002 AUROC for a quarter more labels (#46) |
| spot↔spot correspondence edges | `correspondence` | the signal that separates accept from reject is not in the embedding (#44) |

Neither tool is deleted — the existing labels stay load-bearing (#31: removing the human
special-spot supervision drops census F0.5 0.609 → 0.493), the *stores* are read by
`review_labels.py`, and `interesting_spots.json` is the `LABELS=legacy` ablation arm. What stops is
collecting **more** of them.

## Start, in this order

Ranked by measured value per hour. The first two need **no new tooling** — their candidate lists are
already computed and the review app already has the field to record the verdict. Target 3 is built
(2026-08-16) and, usefully, feeds both of them as a by-product; if you would rather label than plan,
start there and targets 1 and 2 partly fill themselves.

### 1. Adjudicate the 54 duplicate identities — highest value per click in the repo

- **The label:** for each suspect pair, "same animal" or "different animals".
- **8 are already confirmed** by the 2026-08-16 pair-review pass (results.md #49) and need merging,
  not adjudicating: `ac_3/ca_62` and `ca_70/lj_13` (also on the scan's list), plus `ca_41/ca_47`,
  `ca_6/ca_74`, `en_1/sd_1`, `lw_1/rt_1`, `mk_1/nf_1`, `nb_1/nh_1` — which the scan **never
  flagged**. A ninth, `ca_10/sj_3`, is a bad extraction rather than a duplicate (the reviewer's own
  note) and should be sent to the reprocess queue instead.
- **The scan is not the work queue.** 7 of those 9 were invisible to `suspect_duplicates.csv`, so
  its 53 rows are a floor, not the population. Working the scan list alone will miss most of them.
- **Why first:** one animal filed under two labels is scored as a model **error on every metric**
  the project quotes, so this changes numbers that are already published rather than feeding a
  future model. It is bounded — **54 pairs**, not a corpus — and it is the oldest open thread
  ([results.md](../results.md) open threads #4: *54 suspect pairs listed, **0 adjudicated***).
- **Candidates exist:** `pixi run label-consistency` writes `suspect_duplicates.csv`.
- **Recording exists:** the review app has `duplicate_of` and the ⓓ button
  ([preprocessing_ui.md](preprocessing_ui.md) Flow 9). Every field is empty.
- **Cost:** an afternoon, once.

### 2. Multi-animal bounding boxes

- **The label:** a box per salamander on frames holding more than one, or a "crowded, unusable"
  verdict on frames that cannot be separated.
- **Why:** [next_steps_2.md](next_steps_2.md) §4 — failures cluster on multi-animal frames and the
  legacy multi-animal detector failed completely. A fused two-animal mask produces one body axis
  spanning two bodies, which corrupts `axis_t` for every spot in the frame — so this is upstream of
  the position gate, the observability gate and the normalized view at once.
- **Candidates exist:** `data_hygiene.multi_animal_candidates()` ranks photos by the confirmed
  two-animal signature (unusually many spots inside an unusually non-convex body), so the pass is
  over a ranked shortlist, not the dataset.
- **Tooling needed:** a box-drawing surface. Nothing in the repo draws boxes today.
- **Feeds:** the YOLO/Mask R-CNN front-end pre-processor of §4, and immediately a
  reject-these-frames list even before any detector is trained.

### 3. Coarse pair verification — a verdict, not prose — **built, ready to label**

- **The label:** per candidate pair, `match` / `different` / `unsure`, plus reason chips and the
  existing free-text note.
- **Why:** it is the label [open thread #6](../results.md#8-open-threads) needs — a calibrated
  three-way `match / new / abstain` head — and it is judged at the level the human is actually
  reliable at (#44 says they are judging *the pair*, on global structure, not the edge). One verdict
  covers a whole pair, where the old loop spent one click per spot. `unsure` is kept as a first-class
  class, because it is the abstain target rather than a missing answer.
- **Run it:** `pixi run pair-review-gen` then `pixi run pair-review`. Keys `1` / `2` / `3` judge
  **and advance**; `U` jumps to the next unjudged pair. 70 pairs are already rendered and unjudged.
- **Read it back:** `pixi run review-labels`, or `review_labels.pair_verdicts(dataset)` →
  one row per pair with `verdict`, `reasons`, `disagrees_with_label`, `comment`.
- **Two side effects worth as much as the labels.** A verdict that conflicts with the filename label
  is flagged `disagrees_with_label` — that is target 1 arriving as a by-product of ordinary review.
  And the `multi-animal` reason chip accumulates confirmed multi-animal frames, which is target 2's
  ground truth, collected by someone who is looking at the photo anyway.
- **Cost per label:** seconds, against ~a minute per spot-click image.

### 4. Pair alignment / body-axis anchors

- **The label:** head and tail anchors on a photo, or a coarse alignment between two photos of one
  animal.
- **Why:** the simulator names body-axis alignment the **highest-leverage remaining fix** (#36,
  open thread #5), and axis corrections already arrive unprompted through the reprocess queue —
  reviewers are reporting this bug without being asked for it.
- **Tooling needed:** the anchors are drawable in `preprocess-review`'s normalized view, which
  already renders head-green / tail-red discs; today they are read-only.
- **Ordering note:** ranked below #2 because a multi-animal frame's axis is unfixable by anchoring —
  the mask spans two animals. Fix the mask first, then the axis.

## Why not "just label more of everything"

#44 and #46 are the same result at two altitudes: **a label can only pay for itself if a model
downstream can represent what it says.** Before adding a label kind, name the model that consumes it
and the number it is expected to move. Each item above names both.
