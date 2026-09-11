# Training manual

Runs the 13-model / 5-fold census bake-off, then final-fits each surviving model on
100% of the real images and writes a checkpoint per model + a `manifest.json`. Same
script the app calls (`PipelineTrainingBridge`).

## Best run so far

`MIN_QUALITY=0.4`, `SOURCE=sasa`, `TRAIN_POOL=filtered`, 5 folds, seed 0, neg/query 60.
Full table: [`artifacts/spot_transformer/sweeps/all9_q0.4/RESULTS_all9_sasa.md`](../artifacts/spot_transformer/sweeps/all9_q0.4/RESULTS_all9_sasa.md).

| model | R@1 | novelty AUROC | balAcc | review@90 | censusF0.5 |
|---|---|---|---|---|---|
| e2e_pretrained | **0.730 ± 0.061** | 0.729 | 0.780 | 86% | 0.812 |
| e2e_transformer | 0.717 ± 0.062 | 0.731 | 0.771 | 85% | 0.807 |
| logreg | 0.717 ± 0.071 | 0.732 | **0.831** | **50%** | 0.796 |
| ssl_corr | 0.674 | 0.723 | 0.872 | 66% | 0.784 |
| deepsets_cons | 0.658 | 0.796 | 0.811 | 75% | 0.794 |
| set_transformer | 0.619 | 0.715 | 0.822 | 82% | 0.773 |
| axial_cnn | 0.601 | 0.796 | 0.816 | 78% | 0.787 |
| mlp_deep | 0.592 ± 0.156 | 0.649 | 0.774 | 96% | 0.775 |
| raw_voting | 0.529 | 0.561 | 0.799 | 84% | 0.583 |

models_to_keep: e2e_pretrained, e2e_transformer, logreg, raw_voting, axial_cnn, deepsets_cons

- **R@1** — of queries that are re-sights, top-1 named the right animal.
- **novelty AUROC** — can the top-1 score tell a known animal from a novel one (0.5 = coin flip).
- **balAcc** — balanced accuracy of the known/novel call after recalibrating the cut.
- **review@90** — fraction of photos a human must check to hit 90% end-to-end accuracy (lower better).
- **censusF0.5** — population-count accuracy, precision weighted 2×.
- R@5/R@10 are not produced by this eval (open-set census, not closed retrieval) — they log as null.

Take: `logreg` is the pick — ties the top on R@1, best novelty call, needs the least review,
trains in seconds. `e2e_pretrained` edges it on raw R@1 only.

## Run it

```bash
# full run — 5 folds, real epochs, hours
MIN_QUALITY=0.4 pixi run train-all13


# just the models that matter (fast)
MIN_QUALITY=0.4 ONLY="logreg,e2e_pretrained,raw_voting" pixi run train-all13

# just the models that matter (fast)
MIN_QUALITY=0.4 ONLY=logreg,e2e_pretrained,e2e_transformer,axial_cnn, deepsets_cons, raw_voting pixi run train-all13

# smoke test — 2 folds, few epochs, minutes (NOT a result)
QUICK=1 MIN_QUALITY=0.4 pixi run train-all13
```

## Env knobs

| var | meaning | default |
|---|---|---|
| `MIN_QUALITY=0.4` | drop photos below quality 0.4 from eval **and** train. Required to match the historical numbers (`RESULTS_all9_sasa.md`, logreg R@1 0.717). Without it you score on unidentifiable photos and R@1 falls ~0.13. | unset (no gate) |
| `SOURCE=sasa` | score the sasa study population only (like-for-like with old results). | `sasa` (set by the script) |
| `ONLY=a,b,c` | restrict to these model names (comma-separated, no spaces). | all 13 |
| `TRAIN_POOL=full` | keep sub-threshold photos as training data, gate only eval. Slightly better on all metrics; use `filtered` (default) for like-for-like. | `filtered` |
| `QUICK=1` | 2 folds, tiny epochs. Smoke path only. | unset |

## Model names for `ONLY=`

Important ones:

| name | what |
|---|---|
| `logreg` | summary features + logistic regression — the champion (R@1 ~0.72) |
| `raw_voting` | no training, soft-chamfer baseline |
| `e2e_pretrained` | frozen 62-dim encoder + learned voting |
| `e2e_transformer` | learned encoder + voting, trained jointly |
| `set_transformer` | match-set → attention net |
| `deepsets_cons` | match-set → DeepSets, seed-ensembled |
| `axial_cnn` | match-set → axial CNN |
| `mlp_deep` | summary features + 3-layer MLP |

SSL rows (need caches built by `ssl_pretrain.py`, else auto-dropped): `ssl_simclr`,
`ssl_random`, `ssl_corr`, `ssl_corr_audit`, `ssl_corr_human`, `ssl_corr_finetune`,
`ssl_aug0.5`, `ssl_aug2.0`, `ssl_ep100`.

### Notes

- **e2e** = end-to-end: learn the spot embedding *and* the voting rule in one backprop pass.
  The other families keep the 62-dim spot embedding fixed and only learn the aggregation.
- `e2e_pretrained` and `e2e_transformer` share the same voting head; they differ only in the
  encoder. `e2e_pretrained` = `arch=frozen` → encoder is the identity, the hand-engineered
  62-dim is used as-is, only the head trains (fast, ~150s/fold). `e2e_transformer` =
  self-attention encoder re-embedding each spot in the context of the animal's other spots,
  trained jointly (slow, hours/fold). In the best run the transformer did **not** beat frozen
  (0.717 vs 0.730 R@1) — the encoder fine-tuning bought nothing at this data size.
- **Fast informative subset:** `ONLY=logreg,e2e_pretrained,set_transformer,deepsets_cons,raw_voting`
  — champion, its frozen-encoder cousin, the two best novelty models, the free baseline.
- **Safe to drop:** `mlp_deep` (dominated by logreg on every metric), `ssl_corr_finetune`
  (collapses to 0.5 every fold — broken). `axial_cnn` is slowest by far but has the best
  novelty AUROC (0.796, tied with `deepsets_cons`) — keep it *only* if you care about the
  known-vs-novel call, drop it if you only care about R@1.
- `set_transformer` and `deepsets_cons` are the same family (match-set → net). Both are slow
  (~40–60 min/fold). `deepsets_cons` beats `set_transformer` on novelty AUROC (0.796 vs
  0.715) and censusF, so `set_transformer` is a research row only — drop it for a deploy run.
- **Tuning logreg:** no real headroom. It is plain logistic regression; the optimizer knobs
  converge to the same solution and `mlp_deep` already tested "more capacity" and lost.
  Headroom is elsewhere: hard-negative mining (train vs confusable individuals, not random),
  feature / spot-embedding variants (`sweep_bakeoff.py`), more multi-photo individuals
  (scaling curve still rising), and `TRAIN_POOL=full`. Do that work in `sweep_all9.py` /
  `sweep_bakeoff.py`, not here — `train_all13` is the deploy run, not the search.

## Where the output goes

```
artifacts/spot_transformer/sweeps/all13_<timestamp>/
├── manifest.json          # [{name, kind, metrics, weights_path}]  <- the app reads this
├── RESULTS_all13.md        # the comparison table
├── logreg/weights.pt       # {state_dict, scaler, config}
├── set_transformer/weights.pt
└── ...                     # one dir per model; raw_voting has none
```

The final line on stdout is `wrote <path>/manifest.json`.

## Import into the app

### Option A — let the app run it

Trigger the training job; it registers every model and (if `auto_promote` is on)
promotes the best. Set the gate on the **server** process so the subprocess inherits it:

```bash
MIN_QUALITY=0.4 pixi run app:serve
curl -X POST localhost:8000/models/retrain      # -> {job_id}; poll GET /jobs/{job_id}
```

### Option B — import checkpoints from a run you did yourself

```bash
RUN=artifacts/spot_transformer/sweeps/all13_<timestamp>
python - "$RUN/manifest.json" <<'EOF'
import json, subprocess, sys
LET = [("a","r1"),("d","bal_acc"),("e","novelty_auroc"),("f","review_at_90")]
for m in json.load(open(sys.argv[1])):
    if not m["weights_path"]:            # skip raw_voting (no checkpoint)
        continue
    args = ["pixi","run","python","-m","app","import-model", m["name"],
            m["weights_path"], "--kind", m["kind"], "--notes", "all13 MIN_QUALITY=0.4"]
    for letter, key in LET:
        if m["metrics"].get(key) is not None:
            args += ["--metric", f"{letter}={m['metrics'][key]}"]
    subprocess.run(args, check=True)
EOF
```

`import-model` copies the checkpoint to `<data-dir>/models/<name>/` and adds a registry
row. Make one active:

```bash
pixi run python -m app import-model <name> <weights.pt> --make-active   # at import time
# or later:
curl -X POST localhost:8000/models/promote -d '{"name":"<name>"}'
```

## Note

The registry, Models page, metrics and promotion work — but the app's live matcher
(`Bridges.production`) does not load these `weights.pt` at match time yet
(`StubMatchingBridge`; see `app/bridges.py`). Importing catalogues and scores them; it
does not change match results until the inference bridge is wired.
