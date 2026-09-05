# deprecated/ — superseded spot_transformer tracks

Archived on 2026-07-18. These were the two investigation tracks that lost to the
summary-feature **logistic regression** champion and the current **set-aggregation / census**
track. Kept for reference and their last sweep outputs; not maintained.

The live pipeline is one level up: `sweep_set.py` (+ `aggregator_set.py`, `census.py`,
`aggregator.py`, `data.py`, `embeddings.py`). The raw-voting baseline and the logreg model live
in `../aggregator.py` and are still run as references inside `sweep_set.py`.

## What's here

**Image-level embedding track** — learn a per-image embedding with a set transformer.
Lost to raw-spot voting (R@1 ~0.35 vs ~0.56).
- `sweep.py` → `sweep_results/` · `train.py` · `eval.py` · `transformer.py` · `losses.py`

**Raw-spot track** — per-spot encoder + voting, contextual set transformer.
The learned contextual model did not beat raw-spot voting.
- `sweep_spot.py` → `sweep_results_spot/` · `train_spot.py` · `eval_spot.py`
  (also uses `transformer.py` / `losses.py` above)

The full narrative and per-scheme results are in `../results.md`.

## Re-running (caveat)

These modules resolve siblings with a run-as-script fallback (`import data as d`, etc.) that
expected them to sit next to `data.py`. After the move, `data.py` / `embeddings.py` are in the
parent dir, so the script-mode `import data` no longer resolves from here. To revive one, run it
from `pipeline/spot_transformer/` with this folder on `sys.path`, or fix the imports to the
`pipeline.spot_transformer.*` package paths.
