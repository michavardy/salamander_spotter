"""Full-scale training run: the ALL-9/13-model bake-off (see ``sweep_all9.py``), plus a
per-model final-fit checkpoint trained on 100% of the real images for deployment.

Runs sweep_all9's exact CV protocol unchanged (same ``MODELS``, same ``run_fold``, same
metrics — via its now-shared ``prepare()``/``run_fold``/``aggregate_rows``) so the printed
per-model-per-fold lines are directly comparable to history. Adds two things sweep_all9.py
doesn't do:

  1. ``PROGRESS <frac> <message>`` lines an external caller (the app's
     ``PipelineTrainingBridge``) parses to drive a progress bar — ``frac`` covers the CV
     folds 0.0-0.8, then the final-fit models 0.8-1.0.
  2. A **final-fit phase**: retrain each model that survived CV once more on ALL real
     images (no held-out fold) and save a deployable checkpoint, then write
     ``manifest.json`` — ``[{name, kind, metrics, weights_path}, ...]`` — which is what the
     app reads to register every model for inference selection.

Defaults ``SOURCE=sasa`` (this app scopes training to the sasa population) and the real
(non-QUICK) 5-fold protocol. Override via the same env vars ``sweep_all9.py`` reads.

    pixi run python pipeline/spot_transformer/sweeps/train_all13.py
    QUICK=1 pixi run python pipeline/spot_transformer/sweeps/train_all13.py   # smoke test
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SOURCE", "sasa")

import numpy as np
import torch

_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval", "sweeps"):
    _p = str(_ST / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# direct script execution never triggers sweeps/__init__.py, so bootstrap + call explicitly
_REPO_ROOT = Path(__file__).resolve().parents[3]  # sweeps -> spot_transformer -> pipeline -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.utils.encoding_utils import reconfigure_stream_to_utf8  # noqa: E402
from pipeline.utils.logger_utils import get_logger                    # noqa: E402

reconfigure_stream_to_utf8()
logger = get_logger(Path(__file__).stem)

import data as d                                              # noqa: E402
import sweep_all9 as a9                                       # noqa: E402
from aggregator import build_pairs, train_aggregator          # noqa: E402
from aggregator_e2e import build_e2e_pairs, train_e2e         # noqa: E402
from aggregator_set import build_record_pairs, train_set      # noqa: E402



def _progress(frac: float, msg: str) -> None:
    # plain print, not logger — PipelineTrainingBridge regexes this exact `PROGRESS <frac> <msg>`
    # format from raw subprocess stdout; a logger-formatted line would break that parse
    print(f"PROGRESS {frac:.4f} {msg}", flush=True)


def _sanitize(name: str) -> str:
    """Model names become registry ids — dots aren't safe there (``ssl_aug0.5``)."""
    return name.replace(".", "_")


def final_fit(sets, models: list[dict], outdir: Path) -> list[dict]:
    """Train each model once more on ALL real (non-synthetic) images and save a
    checkpoint. Returns one entry per model: ``{orig_name, name, kind, weights_path}``
    (no metrics yet — the caller fills those in from the CV aggregation)."""
    train_imgs = [i for i in range(len(sets)) if not sets[i].is_synth]
    entries = []
    n = len(models)
    for i, m in enumerate(models):
        name, fam = m["name"], m["family"]
        model_dir = outdir / _sanitize(name)
        weights_path: Path | None = None
        t0 = time.time()
        try:
            if fam == "raw":
                pass  # no trainable state
            elif fam == "feat":
                X, y, _, _ = build_pairs(sets, train_imgs, neg_per_query=a9.NEG_PER_QUERY, seed=a9.SEED)
                kw = {k: m[k] for k in ("hidden", "n_layers", "dropout", "weight_decay") if k in m}
                model, scaler = train_aggregator(X, y, seed=a9.SEED, **kw)
                model_dir.mkdir(parents=True, exist_ok=True)
                weights_path = model_dir / "weights.pt"
                torch.save({"state_dict": model.state_dict(), "scaler": scaler, "config": m}, weights_path)
            elif fam == "set":
                kw = {k: m[k] for k in ("arch", "h", "dropout", "weight_decay", "l1",
                                         "n_heads", "n_layers", "kernel") if k in m}
                recs_tr, y_tr, _, _ = build_record_pairs(
                    sets, train_imgs, neg_per_query=a9.NEG_PER_QUERY, seed=a9.SEED,
                    n_bands=m.get("n_bands"),
                )
                model, scaler, _ = train_set(recs_tr, y_tr, epochs=a9.EP_SET, seed=a9.SEED,
                                             beta=a9.BETA, verbose=False, **kw)
                model_dir.mkdir(parents=True, exist_ok=True)
                weights_path = model_dir / "weights.pt"
                torch.save({"state_dict": model.state_dict(), "scaler": scaler, "config": m}, weights_path)
            else:  # e2e
                kw = {k: m[k] for k in ("arch", "out_dim", "depth", "dropout", "n_heads",
                                         "feature_attr", "residual") if k in m}
                pairs_tr, y_tr, _, _ = build_e2e_pairs(sets, train_imgs, neg_per_query=a9.NEG_PER_QUERY, seed=a9.SEED)
                model = train_e2e(sets, pairs_tr, y_tr, epochs=a9.EP_E2E, seed=a9.SEED, verbose=False, **kw)
                model_dir.mkdir(parents=True, exist_ok=True)
                weights_path = model_dir / "weights.pt"
                torch.save({"state_dict": model.state_dict(), "config": m}, weights_path)
            logger.info(f"final-fit {name:<17} ok ({time.time() - t0:.0f}s)")
        except Exception as exc:                             # one model failing must not kill the run
            logger.error(f"final-fit {name:<17} FAILED: {type(exc).__name__}: {exc}")
            weights_path = None
        entries.append({
            "orig_name": name, "name": _sanitize(name), "kind": fam,
            "weights_path": str(weights_path) if weights_path else None,
        })
        _progress(0.8 + 0.2 * (i + 1) / n, f"final-fit {name} ({i + 1}/{n})")
    return entries


def metrics_for(row: tuple) -> dict:
    """Map one ``aggregate_rows`` row -> the app's registry metric names. This sweep is an
    open-set census evaluation, not closed retrieval@k, so r5/r10 are left null."""
    _, _, ir, _irs, ab, bb, cb, ba, rv, _cf = row
    finite = [v for v in (ab, bb, cb) if np.isfinite(v)]
    novelty_auroc = max(finite, key=lambda v: abs(v - 0.5)) if finite else None
    return {
        "r1": ir if np.isfinite(ir) else None,
        "r5": None,
        "r10": None,
        "bal_acc": ba if np.isfinite(ba) else None,
        "novelty_auroc": novelty_auroc,
        "review_at_90": rv if np.isfinite(rv) else None,
    }



def main() -> int:
    logger.info(f"train-all13: source={d.SOURCE} quick={a9.QUICK}")
    sets, folds, models = a9.prepare()
    n_folds = len(folds)
    logger.info(f"{len(models)} models x {n_folds} folds")

    per_fold: dict[str, list] = {m["name"]: [] for m in models}
    for fi, (tr, ev) in enumerate(folds):
        logger.info(f"=== fold {fi} " + "=" * 50)
        res = a9.run_fold(sets, tr, ev, models)
        for k, v in res.items():
            if v is not None:
                per_fold[k].append(v)
        _progress(0.8 * (fi + 1) / n_folds, f"fold {fi + 1} of {n_folds} complete")

    rows, _feat_votes = a9.aggregate_rows(models, per_fold)
    metrics_by_name = {r[0]: metrics_for(r) for r in rows}
    trained_models = [m for m in models if m["name"] in metrics_by_name]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "sweeps" / f"all13_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{len(trained_models)} models survived CV; final-fit on 100% real images ...")
    entries = final_fit(sets, trained_models, outdir)

    manifest = [
        {"name": e["name"], "kind": e["kind"], "metrics": metrics_by_name[e["orig_name"]],
         "weights_path": e["weights_path"]}
        for e in entries
    ]
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # NOTE: plain print, not logger — app/pipeline_bridge/training.py regexes this exact line
    # (`^wrote (.*[/\\]manifest\.json)$`) from raw subprocess stdout to locate the manifest.
    print(f"wrote {outdir / 'manifest.json'}")

    fam_name = {"feat": "summary→clf", "set": "match-set→net", "e2e": "encoder+voting",
                "raw": "none (baseline)"}
    wpath_by_name = {e["orig_name"]: e["weights_path"] for e in entries}
    md = ["# All-13 full training run — census F0.5 + final-fit checkpoints", "",
          f"- folds: {n_folds} · seed {a9.SEED} · quick={a9.QUICK} · source `{d.SOURCE}`",
          "- final-fit: each surviving model retrained once more on 100% of real images "
          "for deployment (checkpoint column below)", "",
          "| model | formulation | identR@1 | AUROC base | b′ | c′ | balAcc | review@90 "
          "| censusF | checkpoint |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, fam, ir, irs, ab, bb, cb, ba, rv, cf in rows:
        if n not in wpath_by_name:
            continue
        wp = wpath_by_name[n] or "—"
        md.append(f"| {n} | {fam_name[fam]} | **{ir:.3f} ± {irs:.3f}** | {ab:.3f} | {bb:.3f} "
                  f"| {cb:.3f} | {ba:.3f} | {rv:.0%} | {cf:.3f} | `{wp}` |")
    (outdir / "RESULTS_all13.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info(f"wrote {outdir / 'RESULTS_all13.md'}")

    _progress(1.0, "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
