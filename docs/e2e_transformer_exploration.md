# e2e_transformer exploration — results

Two RunPod campaigns hyperparameter-searching `e2e_transformer`, the aggregator that tied for
the top of the `all9_q0.4` bake-off (identR@1 0.717 ± 0.062, alongside `e2e_pretrained` 0.730 and
`logreg` 0.717) but had never had a real search — `EP_E2E` was pinned at 30 and a fold cost ~3h on
CPU, so `sweep_all9` could only afford one config. On a GPU a fold-train drops to ~1-2 min, making
these searches feasible.

Code: [`pipeline/spot_transformer/sweeps/e2e_transform_explore.py`](../pipeline/spot_transformer/sweeps/e2e_transform_explore.py)
(the search + all `Cfg` knobs) and
[`run_e2e_runpod.py`](../pipeline/spot_transformer/sweeps/run_e2e_runpod.py) (the RunPod launcher).
Raw output: [`artifacts/spot_transformer/sweeps/e2e_explore/`](../artifacts/spot_transformer/sweeps/e2e_explore/).

All runs share the same folds / open-set protocol / metric code as `sweep_all9`
(`_census_metrics` / `_novelty_block` / `_r1` copied verbatim), so every row below is directly
comparable to `all9_q0.4/RESULTS_all9_sasa.md` and to each other.

| column | meaning |
|---|---|
| `identR@1` | of re-sight queries, top-1 named the right animal (**the identification task**) |
| `balAcc` | balanced accuracy of the known-vs-novel call at the recalibrated cut (**the novelty gate**) |
| `review@90` | fraction of photos a human must check for 90% end-to-end accuracy |

---

## Campaign 1 — five studies (2026-09-06)

24-config architecture/optimiser grid (`identr`), a novelty-gate variant with an auxiliary margin
loss (`gate`), hard-negative mining (`hardneg`), and capacity/arch probes (`general`) — each
re-running `baseline` as a self-check. 6 RunPod pods, ~$9 total.

| study | winner | key metric | Δ vs baseline |
|---|---|---|---|
| identr | `id_d2_h2_p0.1` (depth 2, heads 2, dropout 0.1) | identR@1 0.743 ± 0.068 | +0.043 |
| gate | `hn_skip3_mix0.2_r2` | balAcc 0.808 | — |
| hardneg | *(all configs worse than baseline)* | identR@1 0.466–0.661 | **−0.04 to −0.23** |
| general | `gen_long_ep150` | identR@1 0.731 | +0.030 |

**Baseline reproduced:** identR@1 0.701 ± 0.063 vs the 0.717 ± 0.062 reference — within fold std.

**Hard-negative mining backfired uniformly** — every config scored below baseline, worse with more
mining rounds (`r1` → `r2` made it worse, not better), consistent with the miner surfacing label
noise rather than genuine hard negatives.

Full table: [`RESULTS_e2e_explore_MERGED.md`](../artifacts/spot_transformer/sweeps/e2e_explore/RESULTS_e2e_explore_MERGED.md).

---

## Campaign 2 — `boost` study (2026-09-14/15)

Campaign 1's `identr`/`gate` probes only ever varied lr/wd/neg *one-at-a-time off the depth-3
baseline* — never combined with `identr`'s own winning architecture, and never touched several
knobs that existed in the codebase but were never wired into the sweep: the `SoftVote` head's
temperature/sharpness/capacity (`tau`/`sharp`/`hidden` — `E2EVoter` didn't even forward `sharp`
until this campaign), the feature-jitter augmentation (`data.feature_jitter`, implemented but
unused), and training on synthetic images (hard-excluded from e2e training). `boost` also added
two new mechanisms: decoupled weight decay (AdamW vs Adam-with-L2) and leak-free early stopping
(checkpoint on a train-carved probe instead of always the last epoch).

All 8 configs build on the `identr` winner (depth 2, heads 2, dropout 0.1) instead of the
depth-3 baseline. 1 RunPod pod (NVIDIA L4, $0.49/hr community), 45 fold-trains, 7.1h, **$3.48**.

**Targets going in:** push identR@1 into 0.75–0.80, balAcc to 0.82.

| config | identR@1 | balAcc | review@90 | what it tests |
|---|---|---|---|---|
| **`boost_synth`** ⭐ | **0.760 ± 0.077** | **0.863** | **63%** | include synthetic images in e2e training |
| `boost_adamw` | 0.743 ± 0.068 | 0.786 | 87% | decoupled weight decay |
| `boost_warmup` | 0.743 ± 0.068 | 0.779 | 85% | linear LR warmup before cosine decay |
| `boost_vote_cap` | 0.743 ± 0.068 | 0.797 | 87% | SoftVote head capacity 32→64 |
| `boost_vote_sharp` | 0.731 ± 0.071 | 0.810 | 77% | SoftVote temperature/sharpness |
| `boost_lr_neg` | 0.701 ± 0.063 | 0.771 | 84% | best lr + best neg, combined (interaction test) |
| `boost_jitter` | 0.701 ± 0.063 | 0.779 | 85% | feature-space Gaussian noise |
| `boost_earlystop` | 0.606 ± 0.121 | 0.821 | 77% | checkpoint on train-carved probe, not last epoch |
| baseline (this campaign) | 0.714 ± 0.074 | 0.768 | 86% | — |

Full table: [`RESULTS_e2e_explore_q0.4_sasa_boost.md`](../artifacts/spot_transformer/sweeps/e2e_explore/RESULTS_e2e_explore_q0.4_sasa_boost.md).

### Headline: `boost_synth` clears both targets

`arch=transformer · depth=2 · n_heads=2 · dropout=0.1 · lr=0.001 · wd=0.0001 · neg=60 · epochs=60 · cosine=True · use_synth=True`

- identR@1 **0.760 ± 0.077**, inside the 0.75–0.80 target band, +0.046 over this campaign's baseline
  and the single largest gain across both campaigns.
- balAcc **0.863**, well past the 0.82 target.
- review@90 dropped to **63%** (vs 86% baseline) — a large practical reduction in human-review
  load, not just a metric win.

**Caveat.** The ±0.077 fold std is wide, and one fold (f4) scored a suspicious ceiling —
identR@1/balAcc/os_auroc all 1.000 simultaneously. The held-out eval split itself stays real-only
(synthetic images only enter the *training* pool, per `cfg.use_synth` in `_eval_fold`), so this
isn't definitionally a leak, but per-fold eval sets here are small (~20 queries), so one easy fold
can plausibly hit a ceiling by chance. **Before treating this as settled, rerun `boost_synth` alone
with more folds/seeds** to confirm the gain is real and not one lucky fold — cheap to check with
`STUDY=boost` + a config filter, or a small standalone script pinned to `use_synth=True`.

### Negative / null results (also useful)

- **`boost_lr_neg`** — combining `identr`'s individually-best lr (0.713 alone) and neg (0.731
  alone) did **not** stack; ties baseline. Confirms these knobs don't compose additively off the
  `id_d2_h2_p0.1` architecture — no further joint-grid search off this base is likely to help
  without a different lever.
- **`boost_jitter`** — feature-space noise augmentation had **no measurable effect** (identical to
  `boost_lr_neg`'s baseline-tying number). Not worth pursuing further at `jitter_std=0.02`; a much
  larger sweep of the noise scale is low-priority given the null result at a reasonable default.
- **`boost_earlystop`** — the **worst** result of the campaign (identR@1 0.606 ± 0.121, highest
  variance) despite reasonable balAcc (0.821). The checkpoint-selection score combined
  R@1 and balAcc from a leak-free train-carved probe (`0.5 · r1 + 0.5 · bal`); the likely failure
  mode is that this composite let the selector chase the balAcc half at R@1's expense, or that the
  40-individual train-carved probe is simply too small/noisy a signal for reliable model selection
  here. Worth revisiting the selection criterion (e.g. R@1-only, or a larger probe subsample)
  before concluding early stopping doesn't help at all.

### Ties that didn't move the ceiling

`boost_adamw`, `boost_warmup`, and `boost_vote_cap` all landed at exactly the `identr`-campaign's
prior best (0.743 ± 0.068) — none *beat* it, they matched it. Decoupled weight decay, LR warmup,
and a wider vote head are all safe-to-keep defaults but none is independently the lever that
explains `boost_synth`'s gain.

---

## Recommended next step

1. **Confirm `boost_synth`** with a repeat run (more folds and/or seeds) given the wide fold std
   and the one ceiling fold — don't wire it into the app as the new default off a single 5-fold run.
2. If confirmed, **register it via `app import-model`** (spec §7.6) — the current registered model
   (`e2e_transformer_fold0`) predates this whole exploration and was never updated with any winner
   from either campaign.
3. Revisit `boost_earlystop`'s selection criterion rather than discarding early stopping outright —
   the mechanism ran correctly (see the `early_stop: restored best-checkpoint weights` log lines),
   the composite metric it optimized was likely the problem.
4. `axial_cnn` / `deepsets_cons` still have no cloud sweep of their own (tracked separately,
   see `.TODO`) — out of scope for this doc.
