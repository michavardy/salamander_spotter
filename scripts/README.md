# `scripts/` — command-line entry points

Every script here is a **thin CLI**: it parses arguments and configuration, then delegates the
real work to a pure-logic package under [`pipeline/`](../pipeline). Nothing here holds algorithm
state — that lives in `pipeline/generate_spot_labels/`, `pipeline/spot_embedding/`,
`pipeline/spot_transformer/`, and `pipeline/interesting_spots/`.

## How to run

Scripts are invoked through **pixi tasks** (defined in [`pixi.toml`](../pixi.toml)), never with a
bare `python` (the repo's pyenv shim is broken):

```bash
pixi run <task> [options]          # e.g.  pixi run build-dataset --dry-run
```

The task name is listed with each script below. You *can* run a file directly
(`pixi run python scripts/<subdir>/<name>.py …`), but the task is the supported path.

## Conventions

- **`--dry-run` first.** Any script that spends money (Gemini API — marked **BILLED** below) or
  edits data in place takes `--dry-run`. Always run it first: it prints the plan and, for billed
  steps, an API-call estimate, and writes nothing.
- **Resumable.** The billed pipeline stages skip work already on disk, so a run that dies (or runs
  out of credits) can just be re-run.
- **Outputs go to [`artifacts/`](../artifacts) or [`datasets/`](../datasets)**, never next to the
  code. `artifacts/` is git-ignored.
- **Repo-root resolution.** Each script finds the repo root as `Path(__file__).resolve().parents[2]`
  (it lives two levels down: `scripts/<subdir>/<name>.py`). Keep scripts one level deep inside a
  subdirectory or that assumption breaks.

## Subdirectory map

| Subdirectory | What lives there |
|---|---|
| [`dataset/`](#dataset--build-a-packaged-dataset-from-raw-photos) | The raw-photos → packaged-dataset pipeline: normalize names, Gemini spot/anatomy labelling, geometric fixes, quality markers, packaging. This is how a shippable `datasets/<name>/` gets built. |
| [`embedding/`](#embedding--the-spot-embedding-identification-matcher) | The spot-embedding **identification matcher** research pipeline (`emb-*`): prepare CV splits, train/evaluate/compare matchers, identify a query photo, and the Gemini synthetic-view augmentation + its self-consistency filter. |
| [`tools/`](#tools--standalone-interactive-tools) | Standalone interactive tools. Currently the local web app for hand-labelling "interesting" spots. |
| [`experiments/`](#experiments--spot_transformer-sweep-drivers) | Bash drivers that chain the `pipeline/spot_transformer/` sweeps into multi-hour experiment runs, logging to `artifacts/spot_transformer/`. Includes `run_review_regimes.sh` — the five training regimes on the review app's hand labels. |

---

## `dataset/` — build a packaged dataset from raw photos

The stages, in dependency order (**BILLED** = Gemini calls; everything else is free). `build-dataset`
runs the whole chain in one command; the individual scripts let you run or re-run a single stage.

```
normalize / merge-haifa   raw photos      -> images/<dir>/          (normalized names + maps)
extract-spot-labels       images/<dir>/   -> purple/ + anatomy/ + contours/contours.db   BILLED
repurple                  (optional) re-roll only the images whose spots came out badly   BILLED
correct-axis              fix mis-tipped body axes from the masks, re-bin in place        (free)
compute-quality           per-image quality markers -> contours.db image_quality table   (free)
fold-synth                fold a synthetic-views dir into the master image dir            (free)
package-dataset           images/<dir>/ + contours.db -> datasets/<name>/ (+ .zip)        (free)
```
(Synthetic-view *generation* and *filtering* live in [`embedding/`](#embedding--the-spot-embedding-identification-matcher): `emb-gen-augment`, `emb-filter-synth`.)

### `build-dataset` — `pixi run build-dataset`
`build_dataset.py`. **One command, nine steps, one log.** Runs extract → correct-axis → quality →
package → augment (BILLED) → extract synth → correct-axis → quality → package-merged. Every step is
resumable; the whole run appends to `artifacts/dataset_runs/<name>/pipeline_log.jsonl`.

```bash
pixi run build-dataset --dry-run     # ALWAYS FIRST: the plan + billed-call estimate
pixi run build-dataset               # the real run (BILLED)
pixi run build-dataset --limit 3     # small end-to-end rehearsal
pixi run build-dataset --report      # summarise an existing run's log; no work
pixi run build-dataset --rebuild     # NO API: re-derive DB from on-disk artifacts, then package
```

| Option | Default | Meaning |
|---|---|---|
| `--input DIR` | `all_sasa_norm` | source image dir under `images/` |
| `--name NAME` | `<input>_<YYYY_DD_MM>` | dataset / output folder name |
| `--n-per N` | `1` | synthetic views per singleton individual |
| `--limit N` | `0` (no cap) | cap images per step — for a rehearsal |
| `--dry-run` | off | print plan + billed-call estimate, then stop |
| `--report` | off | summarise an existing run's log and exit |
| `--rebuild` | off | **NO API** — rebuild DB + dataset from on-disk artifacts (free) |
| `--from-step N` | `1` | resume from step N (1–9) |
| `--only-dir NAME` | — | operate on ONLY this dir; skip synth gen/extract, package it alone |
| `--skip-correct-axis` | off | don't run the geometric axis-correction step |
| `--skip-quality` | off | don't compute the `image_quality` markers |
| `--no-zip` | off | skip the final `.zip` |
| `--max-attempts N` | `3` | purple re-draws per image |
| `--anatomy-mode {mask,outline}` | `mask` | `mask`: paint body, compute midline, no judge. `outline`: old draw-two-flanks + LLM-judge |
| `--anatomy-attempts N` | 2 (mask) / 1 (outline) | anatomy drafts on the primary model before escalating |
| `--image-retries N` | `3` | re-asks when a model answers with prose instead of drawing |
| `--model M` | `GEMINI_MODEL` | override the image model |
| `--judge-model M` | — | override the anatomy judge model |
| `--anatomy-escalate-models M1,M2` | — | ladder of pricier drawing models (`''` disables) |

### `extract-spot-labels` — `pixi run extract-spot-labels <sub> --input <dir>`
`extract_spot_labels.py`. The core labelling CLI. Pick a **stage** via a subcommand; every stage is
resumable (existing `purple/`/`anatomy/` reused unless `--overwrite`).

```bash
pixi run extract-spot-labels all      --input all_sasa_norm   # BILLED: purple + anatomy + contours
pixi run extract-spot-labels segment  --input all_sasa_norm   # stage 1:  images  -> purple/
pixi run extract-spot-labels anatomy  --input all_sasa_norm   # stage 1b: images  -> anatomy/
pixi run extract-spot-labels contours --input all_sasa_norm   # stage 2:  purple/ + anatomy/ -> DB
```

Subcommands:
- **`all`** — full pipeline. Adds `--no-anatomy` (skip stage 1b; spots stored with NULL bins) and
  `--rewrite CSV` (single-column CSV of image names to redo: regenerate their purple + anatomy and
  replace their DB rows).
- **`segment`** — stage 1 only (images → `purple/`). Adds `--workers N` (concurrent Gemini calls).
- **`anatomy`** — stage 1b only (images → `anatomy/`: head, tail tip, midline). Adds `--overwrite`,
  `--model`, `--temperature`, `--workers N`.
- **`contours`** — stage 2 only (`purple/` + `anatomy/` → `contours.db` + masks + bins). Free.

Common / stage options (available on the relevant subcommands): `--input` (required), `--limit N`,
`--run-log PATH`, `--run-id ID`, `--overwrite`, `--model`, `--keep-gemini-size`,
`--max-attempts N`, `--max-bleed FRAC`, `--escalate-models M1,M2`, `--escalate-attempts N`,
`--temperature T`; anatomy: `--anatomy-mode {mask,outline}`, `--anatomy-attempts N`,
`--anatomy-escalate-models`, `--anatomy-escalate-attempts N`, `--image-retries N`,
`--min-axis-frac F`, `--no-judge`, `--judge-model`; contours: `--eps-frac`, `--min-area`,
`--morph`, `--key-mode`, `--close`.

### `repurple` — `pixi run repurple --input <dir>`
`repurple_bad_spots.py`. **BILLED.** Re-send *only* the images whose spots scored poorly to Gemini
with a flat-magenta / no-shadow prompt, and keep the re-roll only if it actually scores better
(then re-key that image's DB rows).

```bash
pixi run repurple --input all_sasa_norm --dry-run   # ALWAYS FIRST: which images + which model
pixi run repurple --input all_sasa_norm             # the real run (BILLED)
```

| Option | Default | Meaning |
|---|---|---|
| `--input DIR` | `all_sasa_norm` | input image dir (name under `images/`, or a path) |
| `--select {fragment,bleed}` | `fragment` | `fragment`: spots came out shattered/tiny (needs no DB). `bleed`: magenta flooded off the spots |
| `--select-max-bleed FRAC` | stage default | `--select bleed` only: flag purples whose off-spot magenta exceeds this |
| `--only CSV` | — | restrict candidates to names listed in this file (one per line) |
| `--ladder M1,M2,…` | default ladder | model ladder, cheapest first |
| `--model M` | 2nd-priciest in ladder | force a specific model |
| `--min-spots N` | stage default | only images with ≥ N spots can be flagged |
| `--min-median-area PX²` | stage default | flag when median spot area is below this |
| `--max-attempts N` | `2` | Gemini draws per flagged image |
| `--max-bleed FRAC` | stage default | reject a re-roll that floods magenta off the spots beyond this |
| `--limit N` | — | process at most N flagged images |
| `--workers N` | `1` | process N images concurrently (scan + re-rolls) |
| `--dry-run` | off | score + list flagged images and the model, call nothing |

### `correct-axis` — `pixi run correct-axis --input <dir>`
`correct_axis.py`. **Free, no model.** Geometrically re-derive body axes that collapsed when the
painter dropped the head/tail dots mid-body, using the saved masks. Reversible (masks untouched).
Follow with a `contours` run (or let it re-bin in place).

```bash
pixi run correct-axis --input all_sasa_norm --dry-run   # report + before/after QA, no writes
pixi run correct-axis --input all_sasa_norm             # rewrite the passing corrections
```

| Option | Default | Meaning |
|---|---|---|
| `--input DIR` | required | image dir under `images/` (must have `anatomy/`) |
| `--coverage FRAC` | mask default | correct only if the dots span LESS than this fraction of the body (lower = more conservative) |
| `--dry-run` | off | preview: candidates + before/after QA, change nothing |
| `--force` | off | also write corrections whose axis still fails the gates |
| `--no-db` | off | don't propagate into `contours.db` (leave re-binning to a later contours run) |
| `--limit N` | `0` (all) | stop after N candidates |
| `--review-dir DIR` | `artifacts/axis_corrections/<input>/` | where before/after images go |

### `compute-quality` — `pixi run compute-quality --input <dir>`
`compute_quality.py`. **Free.** Cheap, model-free per-image markers (blur, lighting, spot- and
body-extraction quality + raw numbers) into a `contours.db` `image_quality` table for filtering
later. Run after `contours`.

```bash
pixi run compute-quality --input all_sasa_norm --dry-run   # compute + stats, no writes
pixi run compute-quality --input all_sasa_norm             # (re)build the table
```
Options: `--input DIR` (required), `--dry-run`, `--limit N` (first N images, `0` = all).

### `package-dataset` — `pixi run package-dataset --input <dir> --name <name>`
`package_dataset.py`. **Free.** Package a normalized image dir + its `contours.db` into
`datasets/<name>/` (`raw/` + `db/` + README) and zip it. Several `--input` dirs are **merged** into
one dataset (each needs its own `contours.db`) — this is how Gemini-augmented synthetic views get
folded in next to the real photos.

```bash
pixi run package-dataset --input all_sasa_norm --name all_sasa_norm_2026_10_07
pixi run package-dataset --input all_sasa_norm synth_all_sasa_norm_2026_11_07 --name <name>
```
| Option | Default | Meaning |
|---|---|---|
| `--input DIR…` | `all_sasa_norm` | one or more source dirs (merged; each needs its own DB) |
| `--db PATH` | `<input>/contours/contours.db` | explicit DB path (single `--input` only) |
| `--name NAME` | `<input>_<YYYY_MM_DD>` | output folder name (**required** when merging inputs) |
| `--out DIR` | `<repo>/datasets` | datasets root dir |
| `--no-zip` | off | skip building the `.zip` |

### `fold-synth` — `pixi run fold-synth --synth <dir> --into <dir>`
`fold_synth.py`. **Free.** Fold a synthetic-views dir INTO the master image dir, bumping each view's
`_g<k>` index so it sits next to the real photos without colliding (image + purple + anatomy + mask
copied together, nothing re-billed).

```bash
pixi run fold-synth --synth synth_all_sasa_norm_2026_15_07 --into all_sasa_norm --dry-run
pixi run fold-synth --synth synth_all_sasa_norm_2026_15_07 --into all_sasa_norm
```
Options: `--synth DIR` (required), `--into DIR` (default `all_sasa_norm`), `--dry-run`.

### `normalize` — `pixi run normalize`
`normalize_names.py`. Copy raw photos into a normalized directory with canonical filenames.
Options: `--input DIR`, `--output DIR`, `--dry-run` (report the plan without copying).

### `merge-haifa` — `pixi run merge-haifa`
`merge_haifa.py`. Merge a Roboflow COCO export (`haifa_v2`) into `all_sasa_norm`: each KF field
individual becomes one label with a fresh code, yearly photos become instances, maps updated
(backed up first).

```bash
pixi run merge-haifa --dry-run   # ALWAYS FIRST: the plan, no writes
pixi run merge-haifa             # copy files + update the maps
pixi run merge-haifa --force     # rebuild the KF portion from scratch
```
Options: `--input DIR`, `--output DIR`, `--dry-run`, `--force` (delete existing KF rows + files and
rebuild), `--seed N` (RNG seed for the 3-letter code fallback).

---

## `embedding/` — the spot-embedding identification matcher

The `emb-*` research pipeline (see [`docs/spot_embedding_aggregation_plan.md`](../docs/spot_embedding_aggregation_plan.md)).
Prepare cached spot-sets + CV splits once, then train / evaluate / compare matchers on them. Two of
these scripts (`emb-gen-augment`, `emb-filter-synth`) produce and vet the synthetic views the
`dataset/` pipeline folds in.

### `emb-prepare` — `pixi run emb-prepare`
`emb_prepare.py`. Turn a dataset into cached `SpotSet`s + CV splits + a manifest under
`artifacts/spot_embedding/prepared/<dataset>/`. Run once before eval/train.

| Option | Default | Meaning |
|---|---|---|
| `--dataset NAME` | default dataset | dataset name under `datasets/` or a path |
| `--k N` | `5` | number of CV folds |
| `--mode {individual,session}` | `individual` | how folds are split |
| `--seed N` | `0` | RNG seed |
| `--rebuild` | off | rebuild the SpotSet cache from the DB |

### `emb-eval` — `pixi run emb-eval --model <name>`
`emb_eval.py`. Run one matcher through the shared eval harness → `metrics.json`, `report.md`,
`risk_coverage.csv` under `artifacts/spot_embedding/runs/<id>/`.

```bash
pixi run emb-eval --model dummy       # sanity matcher
pixi run emb-eval --model oracle      # upper bound
pixi run emb-eval --model classical   # constellation matcher
```
| Option | Default | Meaning |
|---|---|---|
| `--model NAME` | `dummy` | matcher to evaluate (learned ones — `set_transformer`/`gnn`/`hungarian` — train per fold) |
| `--dataset` / `--k` / `--mode` / `--seed` | as `emb-prepare` | eval split config |
| `--dim N` | `128` | dummy embedding dimension |
| `--orient {a,b}` | `a` | spotdesc orientation: `a`=as-is, `b`=principal-axis canonical |
| `--epochs N` | `60` | training epochs for learned models |
| `--limit N` | — | cap gallery/query images per fold (Gemini cost control) |
| `--min-spots N` | `0` (off) | drop images with fewer spots than this (eval **and** training) |
| `--min-blur F` | `0.0` (off) | drop images blurrier than this (var-Laplacian) |
| `--max-largest-frac F` | `1.0` (off) | drop images where one spot ≥ this fraction of all spot area |

### `emb-bakeoff` — `pixi run emb-bakeoff`
`emb_bakeoff.py`. Run several matchers and emit a top-to-bottom ladder in
`artifacts/spot_embedding/bakeoff/comparison.md`. Options mirror `emb-eval`, plus
`--models m1 m2 …` (default `dummy classical cnn oracle`).

### `emb-train` — `pixi run emb-train --model <name>`
`emb_train.py`. Train one candidate matcher (Phase 2/3). Options: `--model NAME`, `--dataset`,
`--config PATH` (candidate config).

### `emb-identify` — `pixi run emb-identify`
`emb_identify.py`. Query one photo against a gallery (Phase 4). Options: `--gallery PATH`,
`--image PATH`.

### `emb-selfcheck` — `pixi run emb-selfcheck`
`emb_selfcheck.py`. **Doctor**: build the ground-truth table and run oracle/dummy sanity checks.
Options: `--dataset`, `--k`, `--seed`.

### `emb-quality` — `pixi run emb-quality`
`emb_quality.py`. Review image quality and preview what a filter threshold would drop before you
apply it in `emb-eval`.

| Option | Default | Meaning |
|---|---|---|
| `--dataset NAME` | default dataset | dataset to inspect |
| `--min-spots N` | `3` | preview dropping images with fewer spots |
| `--min-blur F` | `0.0` (off) | preview blur threshold |
| `--max-largest-frac F` | `0.95` | preview the largest-spot-dominance threshold |
| `--no-blur` | off | skip the (slower) blur computation |
| `--top N` | `20` | print this many worst images |

### `emb-gen-augment` — `pixi run emb-gen-augment --dataset <ds> --limit <n>`
`emb_gen_augment.py`. **BILLED.** Gemini whole-image augmentation →
`images/synth_<dataset>/<label>_g<k>.png`. Grows individuals that have too few photos.

```bash
pixi run emb-gen-augment --dataset <ds> --which real-singletons --n-per 2 --top-up \
    --limit 0 --dry-run          # ALWAYS FIRST: per-individual plan + billed-call estimate
```
| Option | Default | Meaning |
|---|---|---|
| `--dataset NAME` | default dataset | dataset to augment |
| `--limit N` | **required** | hard cap on billed API calls (`0` = no cap: generate everything remaining) |
| `--which {singletons,real-singletons,multi,all}` | `singletons` | which individuals to augment. `real-singletons` counts REAL photos only, so ones that already have synthetic views still qualify |
| `--n-per N` | `2` | views to generate per source photo |
| `--top-up` | off | treat `--n-per` as a TARGET TOTAL per individual and subtract existing views |
| `--min-source-spots N` | `0` | skip individuals whose source shows fewer than N spots (occluded sources make Gemini invent a pattern) |
| `--out-dir NAME` | `synth_<dataset>` | write views into `images/<NAME>` instead (point at the master dir to generate in place) |
| `--model M` | `GEMINI_MODEL` | override the image model |
| `--dry-run` | off | per-individual plan + billed estimate, then stop |

### `emb-filter-synth` — `pixi run emb-filter-synth --input <dir>`
`emb_filter_synth.py`. **Free, no API. The mandatory self-consistency gate on Gemini views.** Score
every `<label>_g<k>` against that individual's REAL photo(s) with the classical constellation
matcher and quarantine the ones whose spot pattern drifted. Calibrates the threshold on real
same/different-individual pairs first.

```bash
pixi run emb-filter-synth --input all_sasa_norm                     # calibrate + report only
pixi run emb-filter-synth --input all_sasa_norm --threshold 0.30    # + list the rejects
pixi run emb-filter-synth --input all_sasa_norm --threshold 0.30 --apply   # quarantine them
```
| Option | Default | Meaning |
|---|---|---|
| `--input DIR` | `all_sasa_norm` | image dir under `images/` (reads its `contours.db`) |
| `--db PATH` | `<input>/contours/contours.db` | explicit DB path (overrides `--input`) |
| `--threshold F` | — (calibrate only) | reject views scoring below this vs their own real photo(s) |
| `--max-ratio F` | `1.5` | spot-count ratio (synth/real) above which a view is suspect |
| `--ratio-score F` | `0.30` | the score below which a wild ratio becomes a rejection |
| `--apply` | off | MOVE rejects into `<input>/rejected_synth/` (default: report only) |
| `--seed N` | `0` | seed for the negative-pair sample |

---

## `tools/` — standalone interactive tools

### `preprocess-review` — `pixi run preprocess-review`
`preprocessing_review.py` + `pipeline/preprocessing/`. A local **web app** that reviews the dataset
**one individual at a time**, all of its photos side by side. Accept/reject each image, toggle whether
it is training data, delete duplicates, reject bad machine-proposed spot matches, draw the ones the
machine missed, record misses ("visible there, never extracted"), and queue re-extractions. Every edit
is written ~300 ms later to `artifacts/preprocessing/<dataset>/review.json` — the one file every
downstream consumer filters against. Individuals come **lowest mean quality first**, and it opens at
the entry after the last one you edited, so it is stop-and-resume.

**Replaces `interesting-spot-selector`**: its clicks are imported on first run (2521 clicks over 385
images) and `preprocess-export` keeps writing that legacy JSON — by **union**, never replacing it — so
no downstream consumer changes. Spec + flows: [docs/preprocessing_ui.md](../docs/preprocessing_ui.md).

```bash
pixi run preprocess-review                                # dataset: all_sasa_norm_2026_23_07
pixi run preprocess-review --dataset all_sasa_norm_2026_19_07
pixi run preprocess-interest                              # build the interest cache (~4 min, once)
pixi run preprocess-export                                # legacy interesting_spots.json + queue CSVs
```
| Option | Default | Meaning |
|---|---|---|
| `--dataset NAME` | `all_sasa_norm_2026_23_07` | packaged dataset (`datasets/<name>/db/contours.db` + `raw/`) — the only DB with `is_synthetic` **and** `spot_embeddings`, so machine arcs need it |
| `--input NAME` | — | use `images/<folder>/contours/contours.db` instead; no embeddings there, so no machine arcs |
| `--store PATH` | `artifacts/preprocessing/<dataset>/review.json` | override the review store |
| `--legacy PATH` \| `none` | `artifacts/interesting_spots/<folder>/interesting_spots.json` | interesting-spot store to import from / export to; `none` leaves it alone |
| `--no-interest` | off | skip the one-off distinctiveness build (spots get a flat fill) |
| `--host H` / `--port P` | `127.0.0.1` / `8770` | bind address |
| `--no-browser` | off | do not auto-open the browser |

`?label=ac_3` on the URL opens (and links to) one animal. Keys: `←/→` individual, `U` next unreviewed,
`A`/`R` accept/reject, `T` train, `D` delete, `1–9` reason, `M` miss, `N` normalized view, `Esc` clear,
`?` help. Re-extractions are **queued, never run** — they are billed Gemini calls, so the operator
launches them.

**One exception: the ⟳ regenerate button** on a real card re-runs synthetic generation for that photo
and *does* run — `emb-gen-augment --only <sid>` → `extract-spot-labels all` → `compute-quality`, into
`images/regen_<dataset>/`. It states the cost first (~3 Gemini calls per view) and refuses without
`GEMINI_API_KEY`. The new view appears in the family view on its own as a **provisional** card read
from that scratch dir; nothing is written to the packaged dataset. Promote it when you are happy:

```bash
pixi run fold-synth --synth regen_<dataset> --into all_sasa_norm
pixi run extract-spot-labels contours --input all_sasa_norm     # free re-bin
pixi run package-dataset --input all_sasa_norm --name <dataset>
```

### `interesting-spot-selector` — `pixi run interesting-spot-selector` (superseded)
Kept until a full pass through `preprocess-review` exists; prefer that.
`interesting_spot_selector.py`. A local **web app** to hand-label "interesting" spots. Page through
`images/<folder>/` one image at a time; click inside a spot to toggle it green (mapped from
`contours.db` masks). Arrow keys / buttons move between images; **U** jumps to the next unlabeled
one. Every toggle is saved to `artifacts/interesting_spots/<folder>/interesting_spots.json`
(`{ "aa_1_1": [3, 7], … }`), so it is stop-and-resume — it opens on the first unlabeled image.

```bash
pixi run interesting-spot-selector                     # folder: all_sasa_norm
pixi run interesting-spot-selector --output sasa_norm   # a different image folder
```
| Option | Default | Meaning |
|---|---|---|
| `--output NAME` / `--input NAME` | `all_sasa_norm` | image folder to label (bare name resolves under `images/`) |
| `--labels PATH` | `artifacts/interesting_spots/<folder>/interesting_spots.json` | override the labels JSON path |
| `--host H` | `127.0.0.1` | bind host |
| `--port P` | `8765` | bind port |
| `--no-browser` | off | do not auto-open the browser |

---

## `experiments/` — `spot_transformer` sweep drivers

Bash drivers that chain the `pipeline/spot_transformer/` sweeps into long, resumable experiment
runs. Each `cd`s to the repo root, tees everything to a timestamped log under
`artifacts/spot_transformer/`, and is safe to re-run (caches are reused). Run them from the repo
root:

```bash
bash scripts/experiments/<name>.sh          # foreground
nohup bash scripts/experiments/<name>.sh &  # detached; the log path is printed at the end
```

### `run_ssl_experiments.sh`
Everything outstanding on the self-supervised track in one pass: **(A)** build the SSL variants that
don't exist yet (`aug0.5`/`aug2.0` augmentation-strength ablation, `ep100` longer cosine schedule —
existing simclr/random/corr caches reused), then **(B)** one 5-fold sweep over every SSL condition +
the baselines. Log → `artifacts/spot_transformer/ssl/experiments_<ts>.log`; table →
`artifacts/spot_transformer/sweeps/all9_q0.4/`.
- `FAST=1` — skip the ~2.5 h `ep100` build and use QUICK folds (~1.5 h vs ~4–5 h).

### `run_ssl_ablation.sh`
The tighter SimCLR ablation: SimCLR-with-augmentation vs the same encoder never trained vs
SimCLR-with-correspondence positives, all against the hand-engineered baseline under one census
protocol. `&&`-chained (a failure stops the chain rather than sweeping on stale caches). Log →
`artifacts/spot_transformer/ssl/ablation_<ts>.log`.

### `run_feasibility.sh`
**Step 1 of the constellation / comparability plan — run this before building any of it.** Three
measurements, no training and no new labels, each gating a later stage:
- **A** Is a spot's disappearance predictable? Half the spots already vanish between two photos of
  the *same* animal, so "a distinctive spot is missing" is weak evidence by default. It is only
  usable if how *surprising* the disappearance is varies with something observable (size,
  distinctiveness, visibility of that body region, blur, curl). AUROC ≥ 0.65 → absence can be
  weighted; below that, charging for it adds noise.
- **B** Do `curl_deg` / `aspect_ratio` / quality predict how well true pairs match? |rho| ≥ 0.15 →
  worth conditioning on. (`border_frac` is already ruled out — ≤0.038 everywhere, no variance.)
- **C** The deciding one: is geometric consistency a better true/false discriminator on *comparable*
  photo pairs? That gap is the entire justification for **interaction** terms
  (geometry × comparability) rather than plain extra columns. |gap| ≥ 0.05 → build the products.

`real` (default) excludes Gemini views deliberately: a generated view's curl and blur describe the
generator, not a camera. Results → `artifacts/spot_transformer/feasibility/`, including
`spot_survival.csv` and `pair_quality.csv` for your own analysis.

### `run_representation.sh`
Can a different spot descriptor beat the hand-built 62-dim? Three arms on one census protocol:
**A** cheap descriptor variants (`--morph` adds size + irregularity/elongation/noncircularity;
`--harmonics` raises the EFD cutoff), **B** learned descriptors (SimCLR over spot crops, with
`ssl_random` as the untrained control that makes the arm readable), **C** the relational scorers on
the unchanged descriptor, as the reference any descriptor gain must be read against. Modes:
`check` (build + fast screen, ~10 min), `a`, `b`, `c`, `all`.

Start with `check` — it screens every descriptor against the 427 human-judged correspondences in
seconds via `pixi run repr-check`, so you learn what deserves the long runs. Read the **control**
column next to the verdict column: high control with a chance verdict means the descriptor is fine
and the human's rejections are configurational, not appearance-based.

### `run_review_regimes.sh`
Training regimes 1–5 on the hand labels the preprocessing web app collects, all reading
`artifacts/preprocessing/<dataset>/review.json` through one loader
(`pipeline/spot_transformer/core/review_labels.py`), so no two runs can disagree about which labels
they used. Modes: `audit` (R2+R3 only, ~10 min, no training), `quick` (every regime at QUICK size,
plumbing check), `all` (the real thing, hours), or a single `r1`…`r5`.
- **R1** learned distinctiveness + the ablation the strict numbers never had
  (`WEIGHT_MODE=learned|hand|uniform`) and the label refresh (`LABELS=review|legacy`).
- **R2** edge-gate calibration on the 427 machine-proposed correspondences a human judged.
- **R3** audit of the SSL correspondence-mining rule against those verdicts; in `all` mode it also
  rebuilds the cache on the audit's settings and sweeps it downstream.
- **R4** e2e-strict gate supervision, fresh vs stale labels (`e2e_strict_nogate` is the control).
- **R5** human accept/reject as the eval gate (`EVAL_GATE=quality|review|both`), which is what
  finally validates the `MIN_QUALITY=0.4` cutoff against a person.

Log → `artifacts/spot_transformer/review_regimes_<ts>_<mode>.log`; results →
`artifacts/spot_transformer/{distinctiveness,gate_calibration,mining_audit,sweeps/strict}/`.

### `run_quality_protocol.sh`
The image-quality protocol (~5–9 h): **(1)** control sweep with filtered training, **(2)** the same
with `TRAIN_POOL=full`, **(3)** all-9 at the settled protocol (`MIN_QUALITY=0.2`, `TRAIN_POOL=full`).
Steps are **independent** (not `&&`-chained) so a transient failure in the short step 1 doesn't cost
the multi-hour step 3; exit codes are collected and the script exits non-zero if any step failed.
Results → `artifacts/spot_transformer/sweeps/quality_control/` and `.../sweeps/all9_q0.2/`.
