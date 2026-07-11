# Project Goal — Salamander Spotter

## In one sentence

Given a new photograph of a fire salamander, decide **whether it is an animal we have
already photographed** (and if so, which individual) **or a never‑before‑seen individual** —
using nothing but the animal's natural yellow‑spot pattern, no physical tags.

## Why this matters

Fire salamanders (*Salamandra salamandra*) carry a **unique, lifelong pattern of yellow
spots** on a black body — effectively a natural fingerprint. That makes it possible to track
individual animals over time **without capturing, tagging, or otherwise disturbing them**:
you photograph the animal, and the pattern tells you who it is.

We have a field collection of **445 photos taken at Kibbutz Sasa, Israel**. Some of those
photos are repeat sightings of the *same* salamander on different days, from different angles,
in different light. The value of the collection is unlocked only if we can reliably answer, for
each photo, *"have we seen this one before?"* — the foundation for non‑invasive population
monitoring and mark‑recapture studies.

## The problem, precisely: open‑set re‑identification

This is **not** an ordinary image classifier with a fixed list of classes. The set of
individuals is **open** — new salamanders keep appearing, and the system must handle animals it
was never trained on. The task decomposes into two decisions on every incoming photo:

1. **Match** — is this photo the same individual as one already enrolled in the database? If so,
   which one? (a 1‑to‑many search + verification)
2. **Enroll‑as‑new** — if no enrolled individual is a confident enough match, register this
   photo as a **new individual**.

Concretely, the system should turn a query photo into a **ranked list of candidate matches with
a similarity score in [0, 1]**, and apply a **decision threshold**: above it → "same individual
as candidate *X*"; below it → "new individual, add to database."

## Why it is hard

- **Only the spot pattern is stable.** Pose, body curvature, camera angle, lighting, skin
  wetness, and background all vary wildly between photos of the same animal. The model has to be
  invariant to everything *except* the spot pattern.
- **Long‑tailed, few‑shot data.** Of **290 labeled individuals, 203 appear in only a single
  photo**; most of the rest have just 2–3. There is nowhere near enough data to train a
  per‑individual classifier — the model must learn a general notion of *"same vs. different
  pattern"* that transfers to individuals it has never seen.
- **Approximate labels.** The per‑spot annotations are machine‑extracted (see below), not
  hand‑verified, so the supervision signal contains noise.

## The data we have

The dataset is packaged in
[datasets/all_sasa_norm_2026_10_07/](../datasets/all_sasa_norm_2026_10_07/) (see its
[README](../datasets/all_sasa_norm_2026_10_07/README.md)).

| metric | value |
|--------|-------|
| photos (raw) | 445 |
| labeled individuals | 290 |
| individuals with a single photo (singletons) | 203 |
| individuals with ≥ 2 photos | 87 |
| most‑photographed individual | 8 photos (`ca_5`) |
| extracted spots (total) | ~16,500 |

**Identity is encoded in the filename.** A file stem is `<code>_<individual>_<instance>`
(e.g. `aj_1_2`), and the **identity label** is the stem with the trailing `_<instance>` removed:

- `aj_1_2` → individual `aj_1`
- `ca_5_8` → individual `ca_5`

Two photos with the **same label are the same animal** (a *positive pair*); different labels are
different animals. This filename convention is the **supervision signal** for training and the
**ground truth** for evaluation.

Each photo is also paired with a set of machine‑extracted **spots** — contour, full‑frame mask,
centroid, and area — stored in `db/contours.db` (DuckDB). These were produced by this repo's
Gemini "magenta‑spot" segmentation pipeline followed by OpenCV contour tracing, and give the
identification model a clean, pose‑normalized representation of the pattern to work from.

## What success looks like

A working system, given a **held‑out** photo of an individual, should:

- **Rank the true match highly** among enrolled candidates — measured by **rank‑1 / rank‑5
  identification accuracy** on the 87 multi‑photo individuals.
- **Separate same‑from‑different reliably** — measured by verification metrics (**ROC / AUC,
  true‑positive rate at a fixed false‑positive rate**) on positive vs. negative photo pairs.
- **Correctly flag genuinely new individuals** as novel rather than forcing them onto the
  closest existing entry — the **open‑set** behavior that a plain classifier cannot provide.

## Intended approach (direction, not commitment)

Because the data is few‑shot and open‑set, the plan is **metric learning**, not classification:
learn an **embedding** of each salamander's spot pattern such that photos of the same individual
land close together and different individuals land far apart. Identification then becomes
**nearest‑neighbor search in embedding space plus a distance threshold** — new individuals are
simply queries whose nearest neighbor is too far away. The filename labels supply the
positive/negative pairs (and triplets) needed to train this.

## Non‑goals

- **Not species classification** — every animal here is a fire salamander; we identify *which
  individual*, not *which species*.
- **Not a closed, fixed roster** — the database grows over time; the design must assume unseen
  individuals are the norm.
- **Not physical tagging or capture** — the entire point is identification from a photograph
  alone.
