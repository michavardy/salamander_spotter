"""Orchestration for the spot_embedding pipeline (pure logic, no argument parsing).

Phase 0 exposes two entry points used by the CLIs:

* :func:`prepare` — load the dataset, build + cache SpotSets and CV splits, run the leakage
  guard, and write a manifest. This is ``pixi run emb-prepare``.
* :func:`evaluate_matcher` — build a named matcher, run it through the harness on the splits,
  and write a report. This is ``pixi run emb-eval``.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ._common import dataset_name, prepared_dir, resolve_dataset, save_json, set_seed
from .data import (
    check_leakage,
    dataset_summary,
    load_spotsets,
    make_folds,
    reconcile_raw,
)
from .data.splits import save_splits
from .eval import evaluate, evaluate_per_fold, write_report
from .models import MATCHERS, build_matcher

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)


def prepare(dataset: str, *, k: int = 5, mode: str = "individual", seed: int = 0,
            rebuild: bool = False) -> dict:
    """Cache SpotSets + splits and write a manifest. Returns the manifest dict."""
    set_seed(seed)
    dataset_dir = resolve_dataset(dataset)
    name = dataset_name(dataset_dir)

    spotsets = load_spotsets(dataset, use_cache=True, rebuild=rebuild)
    summary = dataset_summary(spotsets)
    recon = reconcile_raw(dataset, spotsets)

    folds = make_folds(spotsets, k=k, mode=mode, seed=seed)
    problems = check_leakage(folds, spotsets)

    out_dir = prepared_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_splits(out_dir / "splits.json", folds,
                meta={"dataset": name, "k": k, "mode": mode, "seed": seed})

    manifest = {
        "dataset": name,
        "summary": summary,
        "reconcile": recon,
        "splits": {"k": k, "mode": mode, "seed": seed, "leakage_problems": problems},
    }
    save_json(out_dir / "manifest.json", manifest)

    logger.info(f"prepared '{name}': {summary['images']} images, {summary['labels']} labels, "
          f"{summary['empty_images']} zero-spot, {summary['total_spots']} spots")
    if recon["raw_only"]:
        logger.info(f"  note: {len(recon['raw_only'])} raw file(s) absent from DB: {recon['raw_only']}")
    logger.info(f"  splits: {k}-fold '{mode}' -> {out_dir / 'splits.json'}"
          + ("" if not problems else f"  [!! {len(problems)} leakage problem(s)]"))
    if problems:
        for p in problems:
            logger.error(f"    - {p}")
    return manifest


def _matcher_kwargs(model: str, dataset: str, *, dim: int, seed: int,
                    limit: int | None, orient: str) -> dict:
    """Per-model construction kwargs (each matcher ignores what it doesn't take)."""
    if model == "dummy":
        return {"dim": dim, "seed": seed}
    if model == "cnn":
        return {"dataset": dataset}
    if model == "spotdesc":
        return {"dataset": dataset, "mode": orient.upper()}
    if model == "gemini":
        return {"dataset": dataset, "limit": limit}
    return {}


# learned models are trained fresh per fold (Phase 3) rather than built once
LEARNED = {"set_transformer", "gnn", "hungarian", "st_cnn"}


def _build_allsets(dataset: str, spotsets: list):
    """Lazily build the SetFeatures map the hand-feature learned models train on."""
    from .train.features import build_set_features
    return build_set_features(spotsets, dataset)


def _build_crops(dataset: str, spotsets: list):
    """Lazily build the per-spot crop cache the st_cnn model trains on."""
    from .train.crops import build_spot_crops
    return build_spot_crops(dataset, spotsets)


def _learned_factory(model: str, data: dict, *, seed: int, epochs: int, kept: set | None = None):
    """A ``make_matcher(fold)`` that trains an encoder on the fold's train split and returns a matcher.

    ``data`` holds ``allsets`` (hand-feature sets) and/or ``crops`` (per-spot crops), as needed.
    ``kept`` (if given) restricts the training pool to quality-passing ids (bad images dropped).
    """
    from .models.learned import FoldEmbeddingMatcher, FoldHungarianMatcher
    from .train import (TrainConfig, embed_sets, embed_sets_cnn, embed_tokens,
                        train_encoder, train_encoder_cnn)

    def _in_train(sid: str, label: str, train_labels: set) -> bool:
        return label in train_labels and (kept is None or sid in kept)

    def make(fold):
        train_labels = set(fold.train_labels)
        eval_ids = list(fold.gallery_ids) + list(fold.query_closed_ids) + list(fold.query_open_ids)

        if model == "st_cnn":
            crops = data["crops"]
            cfg = TrainConfig(arch="st_cnn", epochs=epochs, seed=seed)
            train_sets = [sc for sc in crops.values() if _in_train(sc.salamander_id, sc.label, train_labels)]
            logger.info(f"    fold {fold.fold}: training st_cnn ({epochs} ep) on {len(train_sets)} photos…")
            net, stats = train_encoder_cnn(train_sets, cfg)
            eval_sets = [crops[i] for i in eval_ids if i in crops]
            ids, emb = embed_sets_cnn(net, eval_sets, stats)
            return FoldEmbeddingMatcher(dict(zip(ids, emb)), name=model)

        allsets = data["allsets"]
        arch = "gcn" if model == "gnn" else "set_transformer"
        cfg = TrainConfig(arch=arch, epochs=epochs, seed=seed)
        train_sets = [sf for sf in allsets.values() if _in_train(sf.salamander_id, sf.label, train_labels)]
        logger.info(f"    fold {fold.fold}: training {arch} ({epochs} ep) on {len(train_sets)} photos…")
        net, mean, std = train_encoder(train_sets, cfg)
        eval_sets = [allsets[i] for i in eval_ids if i in allsets]
        if model == "hungarian":
            return FoldHungarianMatcher(embed_tokens(net, cfg, eval_sets, mean, std))
        ids, emb = embed_sets(net, cfg, eval_sets, mean, std)
        return FoldEmbeddingMatcher(dict(zip(ids, emb)), name=model)

    return make


def _quality_filter(dataset: str, spotsets: list, *, min_spots: int, min_blur: float,
                    max_largest_frac: float):
    """Return (eval_spotsets, kept_ids|None, dropped). Off (kept=None) unless a threshold is set."""
    active = min_spots > 0 or min_blur > 0 or max_largest_frac < 1.0
    if not active:
        return [s for s in spotsets if not s.is_empty], None, []
    from .data import QualityConfig, filter_ids

    cfg = QualityConfig(min_spots=min_spots, min_blur=min_blur, max_largest_frac=max_largest_frac)
    kept, dropped = filter_ids(spotsets, dataset, cfg)
    eval_ss = [s for s in spotsets if s.salamander_id in kept]
    if dropped:
        logger.warning(f"quality filter: dropped {len(dropped)} image(s) from eval+training "
              f"(min_spots={min_spots}, min_blur={min_blur:g}, max_largest_frac={max_largest_frac:g})")
    return eval_ss, kept, dropped


def evaluate_model(dataset: str, model_spec: str, spotsets: list, folds: list, *, seed: int = 0,
                   dim: int = 128, orient: str = "a", limit: int | None = None, epochs: int = 60,
                   allsets: dict | None = None, crops: dict | None = None,
                   kept: set | None = None) -> dict:
    """Score one model (fixed or learned) on preloaded spotsets+folds. Returns a result dict."""
    base, _, sub = model_spec.partition(":")
    if base in LEARNED:
        data = {}
        if base == "st_cnn":
            data["crops"] = crops if crops is not None else _build_crops(dataset, spotsets)
        else:
            data["allsets"] = allsets if allsets is not None else _build_allsets(dataset, spotsets)
        factory = _learned_factory(base, data, seed=seed, epochs=epochs, kept=kept)
        return evaluate_per_fold(factory, spotsets, folds, name=base)
    if base not in MATCHERS:
        raise ValueError(f"unknown model {base!r}; available: {sorted(set(MATCHERS) | LEARNED)}")
    orient_ = sub or orient
    matcher = build_matcher(base, **_matcher_kwargs(base, dataset, dim=dim, seed=seed,
                                                    limit=limit, orient=orient_))
    result = evaluate(matcher, spotsets, folds)
    result["matcher"] = f"{base}_{orient_}" if base == "spotdesc" else base
    return result


def evaluate_matcher(dataset: str, model: str, *, k: int = 5, mode: str = "individual",
                     seed: int = 0, dim: int = 128, limit: int | None = None,
                     orient: str = "a", epochs: int = 60, min_spots: int = 0,
                     min_blur: float = 0.0, max_largest_frac: float = 1.0) -> dict:
    """Run one named matcher through the harness and write a report. Returns the result."""
    set_seed(seed)
    spotsets = load_spotsets(dataset, use_cache=True)
    eval_ss, kept, _ = _quality_filter(dataset, spotsets, min_spots=min_spots,
                                       min_blur=min_blur, max_largest_frac=max_largest_frac)
    folds = make_folds(eval_ss, k=k, mode=mode, seed=seed)

    result = evaluate_model(dataset, model, eval_ss, folds, seed=seed, dim=dim,
                            orient=orient, limit=limit, epochs=epochs, kept=kept)
    config = {"model": model, "k": k, "mode": mode, "seed": seed, "dim": dim,
              "limit": limit, "orient": orient, "epochs": epochs,
              "min_spots": min_spots, "min_blur": min_blur, "max_largest_frac": max_largest_frac}
    run_dir = write_report(result, dataset=dataset_name(resolve_dataset(dataset)), config=config)

    agg = result["aggregate"]
    logger.info(f"[{result['matcher']}] rank-1={agg['rank1']:.4f}  mAP={agg['mAP']:.4f}  "
          f"verify_auc={agg['verify_auc']:.4f}  openset_auroc={agg['openset_auroc']:.4f}")
    logger.info(f"  report -> {run_dir}")
    return result


def bakeoff(dataset: str, models: list[str], *, k: int = 5, mode: str = "individual",
            seed: int = 0, epochs: int = 60, min_spots: int = 0, min_blur: float = 0.0,
            max_largest_frac: float = 1.0) -> dict:
    """Run several matchers through the harness and write a side-by-side comparison.md.

    Shares one load of the spotsets + one set of folds (and one SetFeatures build for the
    learned models) across all matchers, so the comparison is strictly apples-to-apples.
    """
    from ._common import ARTIFACTS_ROOT

    set_seed(seed)
    spotsets = load_spotsets(dataset, use_cache=True)
    eval_ss, kept, _ = _quality_filter(dataset, spotsets, min_spots=min_spots,
                                       min_blur=min_blur, max_largest_frac=max_largest_frac)
    folds = make_folds(eval_ss, k=k, mode=mode, seed=seed)
    name = dataset_name(resolve_dataset(dataset))
    bases = {m.partition(":")[0] for m in models}
    allsets = _build_allsets(dataset, eval_ss) if (bases & LEARNED) - {"st_cnn"} else None
    crops = _build_crops(dataset, eval_ss) if "st_cnn" in bases else None

    rows: dict[str, dict] = {}
    for model in models:
        try:
            logger.info(f"[{model}] evaluating…")
            agg = evaluate_model(dataset, model, eval_ss, folds, seed=seed, epochs=epochs,
                                 allsets=allsets, crops=crops, kept=kept)["aggregate"]
            rows[model] = agg
            logger.info(f"[{model}] rank-1={agg['rank1']:.4f}  mAP={agg['mAP']:.4f}  "
                  f"verify_auc={agg['verify_auc']:.4f}  openset_auroc={agg['openset_auroc']:.4f}")
        except Exception as exc:  # a missing dep / API key shouldn't sink the whole table
            rows[model] = {"error": str(exc)}
            logger.warning(f"[{model}] SKIPPED — {exc}")

    out_dir = ARTIFACTS_ROOT / "bakeoff"
    out_dir.mkdir(parents=True, exist_ok=True)

    def cell(agg, key):
        v = agg.get(key)
        return "err" if "error" in agg else ("nan" if v is None or v != v else f"{v:.4f}")

    lines = [
        "# Bake-off comparison",
        "",
        f"- dataset: `{name}`  ·  {k}-fold `{mode}`  ·  seed {seed}",
        "- read top→bottom as a ladder: dummy = chance floor, oracle = perfect ceiling.",
        "",
        "| model | rank-1 | rank-5 | mAP | verify AUC | TPR@1%FPR | open-set AUROC |",
        "|-------|--------|--------|-----|------------|-----------|----------------|",
    ]
    for model in models:
        agg = rows[model]
        if "error" in agg:
            lines.append(f"| {model} | — | — | — | — | — | — |  _(skipped: {agg['error'][:60]})_")
        else:
            lines.append(
                f"| {model} | {cell(agg,'rank1')} | {cell(agg,'rank5')} | {cell(agg,'mAP')} | "
                f"{cell(agg,'verify_auc')} | {cell(agg,'verify_tpr@fpr')} | {cell(agg,'openset_auroc')} |"
            )
    lines.append("")
    (out_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"comparison -> {out_dir / 'comparison.md'}")
    return rows
