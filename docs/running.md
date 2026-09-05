# Running the salamander matcher — runbook

Everything is a `pixi run` task. This covers the two phases built so far: **Phase 0** (the
shared evaluation harness) and **Phase 1** (the baseline matchers). For the design and the full
roadmap see [per_spot_embedding_aggregation.md](per_spot_embedding_aggregation.md) and
[spot_embedding_aggregation_plan.md](spot_embedding_aggregation_plan.md).

## One-time setup

```bash
cd salamander_spotter
pixi install          # resolves the env; Phase 1 pulls pytorch (CPU) — a few-hundred-MB download
```

The first `emb-eval --model cnn` also downloads ResNet18 ImageNet weights (~45 MB) once and
builds a spot-image cache; later runs are fast.

## The 30-second acceptance check

```bash
pixi run emb-selfcheck
```

Prints a DB-vs-loader table and runs the oracle + dummy matchers, asserting **oracle ≈ 1.0** and
**dummy ≈ chance**. Expect `=== PASS — 9/9 checks ===` (exit 0). This is the single command that
proves the harness is trustworthy; run it after any change to the data/eval code.

## Everyday commands

| Command | What it does |
|---------|--------------|
| `pixi run emb-prepare` | Load the dataset, cache SpotSets, build leakage-safe CV splits, write a manifest. Run once (and after changing `--k`/`--mode`/`--seed`). |
| `pixi run emb-eval --model <name>` | Run one matcher through the harness → a report under `artifacts/spot_embedding/runs/<id>/`. |
| `pixi run emb-bakeoff` | Run several matchers on the *same* folds → `artifacts/spot_embedding/bakeoff/comparison.md`. |
| `pixi run emb-selfcheck` | The acceptance doctor (above). |

Common flags (all optional): `--dataset <name>` (default `all_sasa_norm_2026_10_07`),
`--k 5`, `--mode individual|session`, `--seed 0`.

### Matchers available now

| `--model` | What it is | Needs |
|-----------|------------|-------|
| `dummy` | random, id-seeded embeddings → **chance floor** | — |
| `oracle` | reads labels → **perfect ceiling** (plumbing probe) | — |
| `classical` | constellation matching (RANSAC similarity transform over spot centroids) — the non-DNN floor | numpy/cv2 |
| `spotdesc` | classical + **spot shape**: correspondences from shape descriptors, RANSAC-verified. `--orient a\|b` (Phase 2) | scikit-image |
| `cnn` | frozen ImageNet ResNet18 features on the background-free spot-union image | pytorch |
| `set_transformer` | **learned** aggregator (Phase 3): attention over spot tokens + PMA pool, trained per fold with SupCon + augmentation | pytorch |
| `gnn` | **learned** aggregator (Phase 3): GCN over the kNN spot graph | pytorch |
| `hungarian` | **learned** (Phase 3): per-spot embeddings from the trained encoder, matched by optimal assignment | pytorch, scipy |
| `st_cnn` | **learned** (Phase 3+): a trained per-spot **CNN** encoder (fed OpenCV-augmented crops) → Set Transformer. The improved model | pytorch |
| `gemini` | asks Gemini "same individual?" — **opt-in, billed** (see below) | `GEMINI_API_KEY` |

> The learned models (`set_transformer`, `gnn`, `hungarian`) **train a fresh encoder on each CV
> fold's train split** before embedding that fold's gallery/query — so `emb-eval`/`emb-bakeoff`
> print per-fold training progress and take minutes. Tune with `--epochs` (default 60).

### Examples

```bash
pixi run emb-prepare                                   # build cache + splits
pixi run emb-eval --model classical                    # the non-DNN floor
pixi run emb-eval --model cnn                           # deep-features floor
pixi run emb-eval --model spotdesc --orient a          # shape-aware, as-is orientation (Phase 2)
pixi run emb-eval --model set_transformer --epochs 60  # learned aggregator (Phase 3, per-fold training)
pixi run emb-eval --model gnn --epochs 60              # learned GCN
pixi run emb-eval --model hungarian --epochs 60        # learned per-spot + optimal assignment
pixi run emb-bakeoff                                    # dummy vs classical vs cnn vs oracle
pixi run emb-bakeoff --models dummy classical cnn set_transformer gnn hungarian oracle --epochs 60  # full Phase 0-3 ladder
pixi run emb-bakeoff --mode session                    # session-grouped split (background-leakage check)
pixi run emb-eval --model dummy --seed 1               # reproducibility: rerun same seed → identical
```

> The first `spotdesc` run of each orientation builds a per-spot shape-descriptor cache
> (decodes the 16k masks + computes skeleton topology), ~5 min; later runs read the cache.

### Gemini reference (opt-in, costs money)

Never runs automatically and refuses without `--limit` (a hard cap on API calls). Needs
`GEMINI_API_KEY` in `.env`. Responses are cached, so re-runs are free:

```bash
pixi run emb-eval --model gemini --limit 40
```

## Where results land

```
artifacts/spot_embedding/
├── prepared/<dataset>/
│   ├── spotsets.pkl        # cached SpotSets (delete to force a rebuild)
│   ├── splits.json         # the CV folds
│   ├── manifest.json       # counts + reconciliation + leakage-guard result
│   └── cnn_inputs.pkl      # 224² spot-union images for the CNN baseline
├── runs/<id>_<model>/
│   ├── metrics.json
│   ├── report.md           # aggregate + per-fold table
│   └── risk_coverage.csv
└── bakeoff/comparison.md   # the side-by-side ladder
```

## How to read the numbers

The metrics only mean something **relative to the two anchors**:

- **`dummy`** is the chance floor — a real matcher must clearly beat it.
- **`oracle`** is the perfect ceiling — it exists to prove the harness isn't broken or leaking.

Metrics: **rank-1 / rank-5 / mAP** (retrieval — is the right individual near the top?),
**verification AUC / TPR@1%FPR** (same-vs-different pair separability), **open-set AUROC** (are
novel individuals separable from known ones?). If `dummy` ever scores well above chance, stop —
something leaks and no other number can be believed.

## Results so far (Phases 0–3)

5-fold `individual` split, seed 0, dataset `all_sasa_norm_2026_10_07`. Sorted by rank-1;
**bold** = best in column among real models:

| model | rank-1 | rank-5 | mAP | verify AUC | open-set AUROC |
|-------|--------|--------|-----|------------|----------------|
| dummy (floor) | 0.05 | 0.31 | 0.16 | 0.49 | 0.42 |
| gnn *(learned)* | 0.12 | 0.34 | 0.23 | 0.55 | 0.51 |
| set_transformer *(learned)* | 0.13 | 0.45 | 0.26 | **0.59** | **0.59** |
| spotdesc A (Phase 2) | 0.19 | 0.40 | 0.28 | 0.55 | 0.54 |
| hungarian *(learned)* | 0.19 | 0.41 | 0.27 | 0.58 | 0.51 |
| classical (position only) | 0.21 | 0.44 | 0.30 | 0.57 | 0.55 |
| cnn (frozen ResNet18) | **0.25** | **0.50** | **0.32** | 0.58 | 0.52 |
| oracle (ceiling) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**Reading:** every real matcher clears the dummy and sits far below the oracle. The result is a
**split decision** (see the GATE 3 finding below), not a clean win for the learned models.

### Phase 3 finding — the aggregator bake-off (GATE 3)

The learned models train correctly (supcon loss 1.7 → 0.5; capacity check memorizes a tiny set
at rank-1 1.0). But the honest CV outcome is nuanced:

- **On top-1 retrieval** (find the exact matching photo) the **frozen CNN baseline still wins**
  (rank-1 0.25). The learned aggregators *underperform* it — with only 87 multi-photo
  individuals, they don't yet justify their complexity for ranking.
- **On the product-critical metrics** — verification AUC and **open-set AUROC** ("is this a *new*
  salamander?", the actual goal) — the **Set Transformer leads every model** (0.59 / 0.59 vs the
  CNN's 0.58 / 0.52). This is the first model that meaningfully separates novel individuals from
  known ones.

**GATE 3 decision:** carry the **Set Transformer** forward as the primary matcher — it wins the
open-set/verification metrics that define the product — while keeping the **frozen CNN as the
retrieval bar to beat**. The learned direction is validated for open-set; closing the retrieval
gap is the main open task (a *trained* per-spot CNN encoder instead of hand-features, more epochs,
and — most of all — more labeled multi-photo individuals). GNN was the weakest learned model;
Hungarian led the learned models on rank-1 but not on open-set.

### Phase 2 finding — the orientation A/B experiment (§2a)

Two decisions came out of Phase 2, both **honest and slightly counter to the naïve prior**:

1. **Mode A (as-is) beats Mode B (principal-axis canonical)** on retrieval/verification
   (rank-1 0.19 vs 0.14). Canonicalising each spot to its principal axis *hurts*, because most
   salamander spots are near-round — their principal axis is unstable and 180°-ambiguous, so
   canonicalisation injects noise. This is exactly the failure the design doc warned about, and
   it **confirms the augmentation-first prior**: don't geometrically canonicalise spot shape.
   → **GATE 2: front-end = Mode A (as-is).**
2. **Hand-feature shape alone does not beat pure geometry** (spotdesc ≤ classical on retrieval).
   Adding shape via a training-free descriptor + correspondence gating doesn't help over the
   constellation geometry. This is the motivation for Phase 3: shape needs a **learned** encoder
   + augmentation to pay off, not hand-features. The Phase 2 encoder (`encoders/`) and
   augmentation pipeline (`augment/`) are the inputs that Phase 3's trainer will use.

This is the bar the learned per-spot models (Phase 3) must beat to justify their complexity.

## Augmentation pipeline (two tiers)

Augmentation is the engine that manufactures positive pairs for the 202 single-photo
individuals. It runs at two levels:

1. **Constellation / mask augmentation** (`pipeline/spot_embedding/augment/`, always on during
   learned training) — spot-set warps (similarity, affine, elastic/TPS-stand-in), **spot dropout
   + spurious injection** (missing/false-spot robustness), centroid jitter; and per-spot mask
   transforms (blur, rotate, stretch, occlude). All orientation-preserving — **never a mirror
   flip** (a reflection is a *different* identity), enforced by a determinant guard.
2. **OpenCV spot-crop augmentation** (`augment/spot_crops.py`, on when training `st_cnn`) — each
   spot's mask crop is rotated/stretched/blurred/occluded/speckled before the learned **SpotCNN**
   sees it, so the encoder learns appearance invariance from pixels.
3. **Gemini whole-image augmentation** (`augment/gemini_view.py`, opt-in, offline, billed) — see
   the workflow below.

### Gemini augmentation workflow (opt-in, costs money)

Generates *new photorealistic views of the same individual* (different lighting/background/angle,
spot pattern held fixed) to add real training positives — most valuable for singletons. It never
runs automatically, refuses without `--limit`, skips already-generated files, and needs
`GEMINI_API_KEY`. **Identity preservation is not guaranteed**, so generated views must be
spot-extracted and self-consistency-filtered before use:

```bash
# 1. generate synthetic views (BILLED; caps at --limit API calls)
pixi run emb-gen-augment --dataset all_sasa_norm_2026_11_07 --limit 20 --which singletons --n-per 2
#    -> images/synth_<dataset>/<label>_g<k>.png   (named as new instances of each individual)

# 2. extract spots from the synthetic images (BILLED; reuses the existing spotter pipeline)
pixi run extract-spot-labels all --input synth_all_sasa_norm_2026_11_07

# 3. package the real photos AND the synthetic views into ONE dataset. Both dirs are passed
#    to --input and merged (each carries its own contours.db); --name is required.
pixi run package-dataset \
    --input all_sasa_norm synth_all_sasa_norm_2026_11_07 \
    --name all_sasa_norm_plus_synth_2026_07_13
```

A synthetic view is named `<label>_g<k>` (`aj_1_g0`), so `derive_label` strips the `_g0` and it
lands on **the same individual as that animal's real photos** — which is what turns it into a
training positive. The merged `images` table carries an **`is_synthetic`** flag, and the
generated dataset README reports exactly what the synthetic views bought (positive pairs gained,
singletons eliminated).

The synthetic individuals then join the **training pool only** (never gallery/query, to avoid
leaking synthetic data into evaluation):

```sql
SELECT * FROM images WHERE NOT is_synthetic;   -- safe for gallery/query
SELECT * FROM images WHERE is_synthetic;       -- training pool only
```

Then **FILTER**: keep a synth view only if its spot-set still matches its source individual
(e.g. classical-matcher similarity above a threshold). This self-consistency filter is the
safeguard against Gemini silently changing the spot pattern and injecting label noise.

## The body grid — relative bin positional matching

Every spot carries **two positional labels** — a *relative* key, invariant to how the animal was
rotated or scaled in the photo, so two photos of the same individual can be compared
position-by-position:

| column | values | meaning |
|---|---|---|
| `axial_bin` | **1, 2, 3, 4** | which quarter of the body — the centre line cut into 4 equal arc-length segments. **1 = head end.** |
| `lateral_bin` | **`left` / `right` / `overlap`** | which side of the centre line. `overlap` = the spot's *outline* crosses the line, so it is on **neither** side. |

`left` is the image-left of a salamander whose head is at the top of the frame — measured against
the **body axis**, not the image axes, so it names the same flank however the animal is rotated.

**`overlap` is why the spot's outline is used, not just its centroid.** A spot lying across the
spine has contour points on both sides; calling it `left` because its centroid landed a pixel that
way would be a lie. This is not a rare edge case — in one real photo **12 of 39 spots** straddled
the line, and every spot in the tail quarter did (the tail is narrow, so spots sit across it).

`bin` (1..8) is the two labels multiplied, kept for convenience — but **NULL when `overlap`**,
since a straddling spot is in neither box. Prefer `axial_bin` + `lateral_bin`.

```sql
SELECT axial_bin, lateral_bin, count(*) FROM spots
WHERE salamander_id = 'aa_1_1' GROUP BY 1, 2 ORDER BY 1, 2;
```

Stage 1b (`extract-spot-labels anatomy`) has Gemini draw **four marks** on a *copy* of each photo:

| mark | colour | meaning |
|---|---|---|
| dot | green `#00FF00` | the **front-most tip of the snout** |
| dot | red `#FF0000` | the **back-most tip of the tail** |
| line | cyan `#00FFFF` | one edge of the body, head → tail |
| line | magenta `#FF00FF` | the other edge of the body, head → tail |

The two dots are the axis's **0 % and 100 %**, so their placement is not cosmetic: a green dot
on the *middle* of the head rather than on the snout tip shortens the axis and shifts every bin
boundary backwards along the body. The prompt insists on the extreme tips and the judge checks it
(`dots_at_tips`).

**The centre line is not drawn by the model — we compute it**, as the mean of the two outlines.

That is the whole trick. Asking a model for a line that *bisects* the body asks it to compute a
medial axis in its head, and it cannot: across every model in the ladder, every attempt failed the
"does it bisect" check, always hugging one flank. Tracing the two visible **edges** is a
perceptual task it *is* good at — and the mean of two edges bisects **by construction**. On a
synthetic curved body the derived line tracks the true centre to within 2px.

Two consequences worth knowing:

- **It does not matter which outline lands on which side.** The midpoint of two edges is
  symmetric, so swapping cyan and magenta yields the identical centre line.
- **The legs are ignored on purpose.** The prompt tells the model to pass straight across the base
  of a limb: we want the trunk-and-tail axis, and toes would only drag the outlines — and with
  them the midline — sideways.

A true bisector is load-bearing, not cosmetic: a bin is (quarter along the body) × (side of the
line), so a line hugging one flank would push spots onto the wrong side and bins 1/3/5/7 would
stop being comparable with 2/4/6/8.

The derived centre line is drawn in **white** on the saved `anatomy/<stem>.png`, so one glance
tells you whether the derivation worked: if the white line does not run down the middle, the two
outlines were wrong.

Stage 2 cuts that derived midline at 25 / 50 / 75 % of its **arc length** (so a curled tail bins
by distance *along the body*, not along a chord across it) and splits each quarter left/right of
the midline:

| along the body | left | right |
|---|---|---|
| 0–25 %   | **1** | **2** |
| 25–50 %  | **3** | **4** |
| 50–75 %  | **5** | **6** |
| 75–100 % | **7** | **8** |

A spot's bin comes from projecting its centroid onto the midline: `axis_t` (0 = head, 1 = tail
tip) picks the quarter, the sign of `axis_offset` picks the side. `left`/`right` are relative to
the head→tail direction *as it runs in the photo*, not to the image axes.

The marks are drawn on a **copy**, never on the purple image — stage 2's whole method is keying
flat `#FF00FF`, and a red line down the body would slice every magenta spot it crosses in two.

New DB columns/tables: `spots.bin` / `axis_t` / `axis_side` / `axis_offset`, plus `body_axis`
(head, tail tip, midline polyline, length, source) and `body_bins` (the 8 boxes as polygons).
Where Gemini could not produce a usable axis, the image has no `body_axis` row and its spots have
**`bin IS NULL`** — never a guess. Filter those out rather than treating NULL as a bin.

```sql
SELECT spot_id, "bin", axis_t, axis_side FROM spots
WHERE salamander_id = 'aj_1_2' AND "bin" IS NOT NULL ORDER BY "bin";
```

### The LLM judge (stage 1b)

Drawing the outlines is the easy part; drawing them *correctly* is not. The ways they go wrong —
starting mid-back instead of at the head, wandering off onto the ground, both landing on the same
flank — are all **invisible to a pixel metric**: the marks are perfectly well-formed flat colour
in every one of those cases. So a second model grades each drawing:

1. **are the dots at the extreme tips** — green on the front-most point of the snout, red on the
   back-most point of the tail, with no salamander left in front of one or behind the other?
2. **do both lines go head to tail** — starting at the green dot, ending at the red dot, unbroken?
3. **does each follow an edge** — on the body's outline, not out on the background or cutting
   across the animal? (Ignoring the legs is correct and is not penalised.)
4. **are they on opposite sides** — one down each edge, not both on the same flank, not crossing?

Note what is *no longer* asked: whether the line bisects. It cannot fail that check any more,
because we build the midline ourselves.

A line that fails any of the three is **re-drawn with the judge's written feedback and the
rejected image itself fed back in**, so the model corrects a named mistake rather than
resampling blindly. A cheap local check runs first — were the marks found at all, is the line
long enough — so an obviously broken draft is re-drawn without spending a judge call.

**The drawing model escalates**, exactly like stage 1's `--escalate-models` — **one draft per
rung**, so a rejected line is retried by a *better* model rather than by the one that just
failed:

```
attempt 1/3  gemini-2.5-flash-image
attempt 2/3  gemini-3.1-flash-image
attempt 3/3  gemini-3-pro-image
```

The feedback and the rejected image carry **across** the rung change, so escalating is a
correction, not a fresh start. The pricier rungs are only paid for on images the cheap one could
not get right — an image accepted on attempt 1 never touches them.

Two ways a rung can fail without sinking the image:

- **It 404s** (your key cannot call it) → skipped with a warning, ladder continues.
- **It replies with prose instead of drawing** ("no image part"). This is common and *not*
  configurable away — measured across all three rungs and every `response_modalities` setting,
  they all intermittently answer in text. So the rung is simply re-asked (`--image-retries`,
  default 3) and, if it still refuses, the ladder escalates past it.

Knobs: `--anatomy-attempts` (drafts on the primary, default 1), `--anatomy-escalate-models`
(the ladder; `''` disables it), `--anatomy-escalate-attempts` (drafts per rung, default 1),
`--image-retries`, or `GEMINI_ANATOMY_ESCALATE_MODELS` in `.env`.

Images the judge never accepts are listed in `anatomy/flagged_axis.csv` (feeds straight back
into `--rewrite`) and carry **`body_axis.judged_ok = false`** with the reason in
`judge_feedback`. Their spots are still binned — but the bins are less trustworthy, so filter
them out for clean training data:

```sql
SELECT s.* FROM spots s JOIN body_axis a USING (salamander_id)
WHERE a.judged_ok AND s."bin" IS NOT NULL;
```

The judge is a cheap vision model (`GEMINI_JUDGE_MODEL`, default `gemini-3.5-flash`), not the
image model — `--judge-model` overrides it, `--no-judge` turns it off. It is **preflighted**
before any drawing, so a model id your key cannot call fails immediately instead of after the
first draw of every image. Note that a model appearing in `client.models.list()` does *not* mean
you may call it: `gemini-2.5-flash` lists but 404s with "no longer available to new users".

**Cost:** a line accepted on the first draw costs 1 draw + 1 judge. A stubborn one costs up to
4 draws + 4 judges (2 cheap + 1 mid + 1 pro).

## Building a whole dataset in one command

`build-dataset` runs the entire chain: spot labels + anatomy + contours for the real photos, a
synthetic duplicate for every individual that has only ONE photo, the same extraction on those
synthetic views, and finally the merged, zipped dataset.

```bash
pixi run build-dataset --dry-run     # ALWAYS FIRST: the plan + the billed-call estimate
pixi run build-dataset               # the real run (BILLED)
pixi run build-dataset --limit 3     # small end-to-end rehearsal
pixi run build-dataset --report      # afterwards: what ran, which models, what it cost
```

The five steps, in the only order the dependencies allow:

| # | step | billed | what |
|---|---|---|---|
| 1 | `extract-real` | ✅ | `images/<input>/` → `purple/` + `anatomy/` + `contours.db` |
| 2 | `package-interim` | — | → `datasets/<name>/` |
| 3 | `augment-singletons` | ✅ | single-photo individuals → `images/synth_<name>/<label>_g0.png` |
| 4 | `extract-synth` | ✅ | the same extraction on the synthetic views |
| 5 | `package-final` | — | merge real + synthetic → `datasets/<name>/` + `.zip` |

**Step 2 is not redundant with step 5.** `emb-gen-augment` reads a *packaged* dataset (its `raw/`
and `db/`) to decide which individuals are singletons and to pull their source images, so the real
photos must be packaged before synthetic ones can be generated from them. Step 5 then rewrites the
same dataset dir with both halves merged.

**Everything is resumable.** Nothing already on disk is paid for twice — an existing `purple/`
image or `anatomy/` JSON is reused, so a run that dies (or hits an API quota) is re-run with the
same command and picks up where it stopped. `--from-step N` forces a restart at a given step.

### The run log

Every stage of every step appends to **one** JSONL file:

```
artifacts/dataset_runs/<name>/pipeline_log.jsonl
```

One line per event — the task, the image, the attempt, **which model drew it, which model judged
it**, the judgment, the decision, and how many billed calls it cost:

```json
{"ts":"...","run_id":"20260714_101500","step":1,"task":"anatomy","input":"all_sasa_norm",
 "image":"aa_1_1.jpg","attempt":2,"max_attempts":3,
 "draw_model":"gemini-3.1-flash-image","judge_model":"gemini-3.5-flash",
 "judgment":{"dots_at_tips":true,"goes_head_to_tail":true,"follows_edges":false,"opposite_sides":true},
 "result":"regenerate required","feedback":"the magenta line traced around the rear leg...",
 "billed_calls":2}
```

`--report` rolls it up: events per task, **calls per model**, outcomes, the judge's most-failed
criteria (which tells you whether to fix the prompt or climb the ladder), and every failure.

### Building a dataset with bins, by hand

Stage 1b is one extra billed call per image, and every stage is resumable — a dir that already
has `purple/` pays only for the anatomy call:

```bash
# real photos: purple already exists, so this only pays for anatomy, then re-bins the DB
pixi run extract-spot-labels all --input all_sasa_norm

# synthetic views (after emb-gen-augment): purple + anatomy + contours
pixi run extract-spot-labels all --input synth_all_sasa_norm_2026_11_07

# merge both into one published dataset
pixi run package-dataset \
    --input all_sasa_norm synth_all_sasa_norm_2026_11_07 \
    --name all_sasa_norm_2026_14_07
```

`--no-anatomy` skips stage 1b if you want spots without bins.

## Data notes baked into every run

- The loader keys off the DB (**444 images**); the one raw file absent from the DB (`jt_1_1`)
  is reported, never invented.
- **13 zero-spot images** (e.g. `ca_1_1`, `jd_1_7`) are flagged and excluded from gallery/query.
- Splits are checked by a **leakage guard** every prepare; `manifest.json` records any problem.
