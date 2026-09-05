"""Embedding bakeoff — additive (concat) vs conjunctive (tensor), across every matcher family.

The question this exists to answer: does making a spot match require shape AND position
(``spot_embeddings_tensor``, the outer-product flow) beat letting one compensate for the other
(``spot_embeddings``, the original concatenation)? And does the answer depend on how much
learning sits on top -- none (raw voting), a little (logreg), a lot (MLP), or a re-embedding
transformer (e2e)?

Every (embedding x model x hyper-parameter) run is cross-validated, and EVERY EPOCH of every
trained run is logged to ``metrics.jsonl`` -- train loss, eval loss, R@1/5/10, MRR, pair AUROC,
and the abstention signal (margin AUROC). Training-free models log a single row at epoch -1.

Per-epoch retrieval is affordable because each fold's eval pairs are featurised ONCE up front;
an epoch's evaluation is then a forward pass plus an argsort over a cached matrix.

Matcher families, in order of how much they charge for pattern that does NOT match:

    raw / logreg / mlp    summary features over each query spot's BEST match — unmatched pattern
                          is simply absent from the score
    strict / strict_logreg  distinctiveness-weighted explained-vs-contradicted evidence: a spot the
                          other animal visibly lacks pushes the score DOWN (see run_strict_family)
    e2e_xf / e2e_frozen   a transformer that re-embeds the spots itself

    pixi run bakeoff --space quick            # smoke: 1 fold, small grid, few epochs
    pixi run bakeoff                          # the real run (see --space full)
    pixi run bakeoff --models raw,logreg      # subset
    pixi run bakeoff --models strict          # the penalty sweep: lam/gamma x 5 folds, no training
    pixi run bakeoff --embeddings tensor      # one flow only

Outputs, under ``artifacts/bakeoff/<dataset>/<run-id>/``:

    metrics.jsonl      one row per (run, fold, epoch) -- the curves
    summary.csv        one row per (embedding, model, config) -- fold-averaged finals
    RESULTS.md         the leaderboard
    ranked_pairs/      per-run top-1 predictions, for `pair-review-gen --from-bakeoff`
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

_ST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ST.parents[1]))                     # repo root -> `pipeline.*`
for _sub in ("core", "models", "eval", "sweeps"):
    p = str(_ST / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

from pipeline.spot_transformer.core import data as d                       # noqa: E402
from pipeline.spot_transformer.models.aggregator import (                  # noqa: E402
    attach_centroids, build_pairs, train_aggregator, _prob, _logit,
)

EMB_TABLES = {"concat": "spot_embeddings", "tensor": "spot_embeddings_tensor"}
OUT_ROOT = d.REPO_ROOT / "artifacts" / "bakeoff"


# ----------------------------------------------------------------------------- search space
def grids(space: str) -> dict[str, list[dict]]:
    """Hyper-parameter grid per model family. ``quick`` is a smoke test, ``full`` is the sweep."""
    if space == "quick":
        return {
            "raw":        [dict()],
            "logreg":     [dict(hidden=0, lr=0.05, weight_decay=1e-4, epochs=100)],
            "mlp":        [dict(hidden=64, n_layers=2, dropout=0.1, lr=0.01,
                                weight_decay=1e-4, epochs=100)],
            "strict":     [dict(lam=0.0, gamma=0.0, sigma_pos=0.12, match_thr=0.4),
                           dict(lam=1.0, gamma=0.5, sigma_pos=0.12, match_thr=0.4)],
            "strict_logreg": [dict(hidden=0, lr=0.05, weight_decay=1e-4, epochs=100,
                                   sigma_pos=0.12, match_thr=0.4)],
            "e2e_xf":     [dict(arch="transformer", depth=2, n_heads=2, dropout=0.1,
                                lr=1e-3, epochs=3, gate_lambda=0.5)],
            "e2e_frozen": [dict(arch="frozen", depth=2, n_heads=2, dropout=0.1,
                                lr=1e-3, epochs=3, gate_lambda=0.5)],
        }
    g: dict[str, list[dict]] = {"raw": [dict()]}
    # lam/gamma are pure arithmetic on cached match parts, so this whole grid costs the same as
    # its two distinct sigma_pos values. lam=0, gamma=0 is the no-penalty control: it is the same
    # matching with unmatched pattern ignored, which is what makes the penalty's effect readable.
    g["strict"] = [dict(lam=l, gamma=gm, sigma_pos=sp, match_thr=0.4)
                   for l in (0.0, 0.5, 1.0, 2.0, 4.0) for gm in (0.0, 0.5, 1.0)
                   for sp in (0.12, 0.25)]
    g["strict_logreg"] = [dict(hidden=0, lr=lr, weight_decay=wd, epochs=400,
                               sigma_pos=0.12, match_thr=0.4)
                          for lr in (0.02, 0.05) for wd in (1e-4, 1e-3)]
    g["logreg"] = [dict(hidden=0, lr=lr, weight_decay=wd, epochs=400)
                   for lr in (0.02, 0.05, 0.1) for wd in (1e-5, 1e-4, 1e-3)]
    g["mlp"] = [dict(hidden=h, n_layers=nl, dropout=dp, lr=lr, weight_decay=wd, epochs=400)
                for h in (32, 64, 128) for nl in (1, 2, 3) for dp in (0.1, 0.3)
                for lr in (0.005, 0.02) for wd in (1e-4, 1e-3)]
    g["e2e_xf"] = [dict(arch="transformer", depth=dep, n_heads=nh, dropout=dp, lr=lr,
                        epochs=25, gate_lambda=gl)
                   for dep in (2, 3, 4) for nh in (2, 4) for dp in (0.1, 0.3)
                   for lr in (5e-4, 1e-3) for gl in (0.0, 0.5)]
    g["e2e_frozen"] = [dict(arch="frozen", depth=dep, n_heads=2, dropout=0.1, lr=lr,
                            epochs=25, gate_lambda=0.5)
                       for dep in (2, 3) for lr in (5e-4, 1e-3)]
    return g


# ----------------------------------------------------------------------------- metrics
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def retrieval_metrics(scores, y, qids, clab, true_label_of) -> dict:
    """Rank candidates per query -> R@k, MRR, pair AUROC, and the margin/abstention signal.

    Only ONE candidate per query is correct (the query's own individual), so average precision
    collapses to 1/rank -- reported as MRR rather than mAP, which is what it actually is.
    """
    ranks, margins, correct = [], [], []
    for q in np.unique(qids):
        m = qids == q
        s = scores[m]
        order = np.argsort(-s)
        ranked = clab[m][order]
        hit = np.where(ranked == true_label_of[q])[0]
        if not len(hit):
            continue
        rank = int(hit[0]) + 1
        ranks.append(rank)
        ss = np.sort(s)[::-1]
        margins.append(float(ss[0] - ss[1]) if len(ss) > 1 else float(ss[0]))
        correct.append(int(rank == 1))
    if not ranks:
        return {}
    ranks = np.array(ranks); margins = np.array(margins); correct = np.array(correct)
    n_cand = float(np.median([int((qids == q).sum()) for q in np.unique(qids)]))
    return {
        "r_at_1": float((ranks <= 1).mean()),
        "r_at_5": float((ranks <= 5).mean()),
        "r_at_10": float((ranks <= 10).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "median_rank": float(np.median(ranks)),
        "pair_auroc": auroc(scores[y == 1], scores[y == 0]),
        "margin_auroc": auroc(margins[correct == 1], margins[correct == 0]),
        "mean_margin": float(margins.mean()),
        "n_queries": int(len(ranks)),
        "n_candidates": n_cand,
    }


# ----------------------------------------------------------------------------- fold runners
def _bce(scores01, y):
    p = np.clip(scores01, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _norm_rows(x):
    x = np.asarray(x, np.float64)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def top1_pairs(scores, qids, clab, sets, pool) -> list[dict]:
    """Each query's top-1 predicted individual, as a (query image -> gallery image) pair.

    The matcher ranks INDIVIDUALS, but the review app renders two photos, so the predicted
    individual is represented by whichever of its photos matches the query best (soft-chamfer).
    Rows mirror ``visualize_match_errors.rank_for`` so ``pair-review-gen`` can render them
    without knowing anything about this sweep.
    """
    by_label: dict[str, list[int]] = {}
    for i in pool:
        by_label.setdefault(sets[i].label, []).append(i)
    out = []
    for q in np.unique(qids):
        m = qids == q
        s = scores[m]
        j = int(np.argmax(s))
        pred = clab[m][j]
        cands = [g for g in by_label.get(pred, []) if g != q]
        if not cands:
            continue
        qe = _norm_rows(sets[q].spots)
        rep = max(cands, key=lambda g: float((qe @ _norm_rows(sets[g].spots).T).max(1).mean()))
        out.append({"query": sets[q].sid, "matched": sets[rep].sid,
                    "score": round(float(s[j]), 4),
                    "correct": bool(pred == sets[q].label),
                    "predicted_label": str(pred), "true_label": sets[q].label})
    return out


def run_aggregator_family(model_name, cfg, sets, tr, ev, *, full_gallery, seed, log, eval_every):
    """logreg / mlp / raw: all consume the 17 summary match-features.

    The eval fold's features are built ONCE, so per-epoch retrieval is a forward pass + argsort.
    """
    train_imgs = [i for i in tr if not sets[i].is_synth]
    gal = train_imgs if full_gallery else None
    Xev, yev, qev, cev = build_pairs(sets, ev, gallery_images=gal, neg_per_query=None)
    true_of = {q: sets[q].label for q in np.unique(qev)}

    pool = list(ev) + (gal or [])

    if model_name == "raw":                                  # training-free: soft-chamfer sum
        m = retrieval_metrics(Xev[:, 0], yev, qev, cev, true_of)
        log(epoch=-1, train_loss=None, eval_loss=None, **m)
        return m, top1_pairs(Xev[:, 0], qev, cev, sets, pool)

    Xtr, ytr, _, _ = build_pairs(sets, train_imgs, neg_per_query=cfg.get("neg_per_query", 30),
                                 seed=seed)
    last: dict = {}
    final_scores = None

    def on_epoch(ep, model, scaler, train_loss):
        nonlocal last, final_scores
        if ep % eval_every and ep != cfg["epochs"] - 1:
            log(epoch=ep, train_loss=train_loss)
            return
        p = _prob(model, scaler, Xev)
        final_scores = p
        last = retrieval_metrics(p, yev, qev, cev, true_of)
        log(epoch=ep, train_loss=train_loss, eval_loss=_bce(p, yev), **last)

    train_aggregator(Xtr, ytr, hidden=cfg.get("hidden", 0), n_layers=cfg.get("n_layers", 1),
                     dropout=cfg.get("dropout", 0.1), epochs=cfg["epochs"], lr=cfg["lr"],
                     weight_decay=cfg["weight_decay"], seed=seed, on_epoch=on_epoch)
    pairs = top1_pairs(final_scores, qev, cev, sets, pool) if final_scores is not None else []
    return last, pairs


# ------------------------------------------------------------------ strict / residual family
"""``strict`` and ``strict_logreg``: score what does NOT match, not only what does.

Every other family here summarises a pair by its matches (``qmax = S.max(1)`` and statistics of
it), so a candidate is never charged for the pattern it fails to account for. Review says that is
the missing half: unrelated animals do share plausible-looking spots, and what separates them is
the striking spot on one that the other visibly lacks. These two models score
``explained / (explained + lambda * contradicted)`` over distinctiveness-weighted spot mass, with a
one-to-one assignment, an occlusion gate, and a single-worst-spot veto -- see
``strict_match.residual_parts``.

``strict`` fixes ``lambda``/``gamma`` by hand (training-free); ``strict_logreg`` hands the same
parts to a logistic regression and lets the trade-off be fitted. Both are per photo-pair, folded to
one score per individual by taking its best photo (``strict_voter.fold_to_candidates``).
"""

_STRICT_CACHE: dict = {}
_STRICT_SETS = None            # the `sets` the cache belongs to — held so nothing else can be it


def prepare_strict(sets):
    """``(factor frame, position lookup)`` for this flow — the per-spot inputs the residual score
    needs. Computed once per ``sets`` (they are population statistics) and cached.

    Switching embedding flow means a new ``sets``, and everything cached — factors, positions,
    match parts — was computed from the old one, so the cache is dropped whole. Identity, not
    equality: two flows have the same images in the same order and differ only in the vectors.
    """
    global _STRICT_SETS
    if _STRICT_SETS is not sets:
        _STRICT_CACHE.clear()
        _STRICT_SETS = sets
    key = "prep"
    if key not in _STRICT_CACHE:
        import distinctiveness as dist                                       # noqa: PLC0415
        import compare_strict as cs                                          # noqa: PLC0415
        frame = dist.load_spot_factors(sets)
        try:
            frame = dist.attach_labels(frame, dist.load_interesting())
        except (FileNotFoundError, OSError):                                 # no human clicks yet
            frame["labeled"] = False
            frame["y"] = 0
        _STRICT_CACHE[key] = (frame, cs.build_pos_lookup(frame))
    return _STRICT_CACHE[key]


def strict_weight_lookup(sets, frame, train_imgs, seed):
    """``(sid, spot_id) -> distinctiveness`` fitted on the TRAIN individuals' human-clicked spots.

    Fitting per fold (as ``compare_strict`` does) keeps the eval individuals out of the weight model
    as well as out of the matcher. With no usable labels — no clicks, or none on this fold's train
    side — it falls back to ``strict_match.distinctiveness``, the training-free composite of the
    same six factors, so the family still runs on a dataset nobody has reviewed.
    """
    key = ("wl", tuple(train_imgs), seed)                # per fold, not per config in the grid
    if key in _STRICT_CACHE:
        return _STRICT_CACHE[key]
    train_labels = {sets[i].label for i in train_imgs}
    ind = np.array(["_".join(s.split("_")[:2]) for s in frame["sid"]])
    fit = frame["labeled"].to_numpy(bool) & np.isin(ind, list(train_labels))
    y = frame.loc[fit, "y"].to_numpy(float)
    if fit.sum() >= 30 and 0 < y.sum() < len(y):
        import distinctiveness as dist                                       # noqa: PLC0415
        m, sc = train_aggregator(frame.loc[fit, dist.FACTOR_NAMES].to_numpy(float), y,
                                 hidden=0, seed=seed)
        out = dist.weight_lookup(m, sc, frame)
    else:
        out = {(sid, int(spid)): float(wi)
               for sid, spid, wi in zip(frame["sid"], frame["spot_id"],
                                        frame["distinctiveness"].to_numpy(float))}
    _STRICT_CACHE[key] = out
    return out


def strict_parts(sets, images, wl, pos, *, gallery, cfg, tag):
    """Cached ``strict_voter.residual_pairs`` — the expensive half (one Hungarian assignment per
    photo pair). Keyed by the image list AND the matching knobs, so sweeping ``lam``/``gamma``
    re-uses one matching pass while a new ``sigma_pos`` correctly forces a fresh one."""
    key = ("parts", tag, cfg["match_thr"], cfg["sigma_pos"], cfg.get("neg_per_query"),
           cfg.get("seed", 0), tuple(images), tuple(gallery or ()))
    if key not in _STRICT_CACHE:
        import strict_voter as sv                                            # noqa: PLC0415
        _STRICT_CACHE[key] = sv.residual_pairs(
            sets, images, wl, pos, gallery_images=gallery,
            neg_per_query=cfg.get("neg_per_query"), seed=cfg.get("seed", 0),
            match_thr=cfg["match_thr"], sigma_pos=cfg["sigma_pos"])
    return _STRICT_CACHE[key]


def photo_pairs(scores, qids, clab, gids, sets) -> list[dict]:
    """Top-1 rows for ``pair-review-gen``. Unlike :func:`top1_pairs` the rendered gallery photo is
    the one the score actually won on, not a re-derived soft-chamfer stand-in."""
    out = []
    for q in np.unique(qids):
        m = qids == q
        j = int(np.argmax(scores[m]))
        pred = clab[m][j]
        out.append({"query": sets[q].sid, "matched": sets[int(gids[m][j])].sid,
                    "score": round(float(scores[m][j]), 4),
                    "correct": bool(pred == sets[q].label),
                    "predicted_label": str(pred), "true_label": sets[q].label})
    return out


def run_strict_family(model_name, cfg, sets, tr, ev, *, full_gallery, seed, log, eval_every):
    """strict (training-free lam/gamma) and strict_logreg (fitted over the same parts)."""
    import strict_match as sm                                                # noqa: PLC0415
    import strict_voter as sv                                                # noqa: PLC0415

    train_imgs = [i for i in tr if not sets[i].is_synth]
    gal = train_imgs if full_gallery else None
    frame, pos = prepare_strict(sets)
    wl = strict_weight_lookup(sets, frame, train_imgs, seed)
    # what "conspicuous" means on THIS dataset: the veto is relative to a top-decile spot, so it
    # stays comparable whether the weights are human-fitted probabilities (often compressed into a
    # narrow band) or the training-free composite (spread across [0, 1]).
    w_ref = float(np.percentile(np.fromiter(wl.values(), float), 90)) if wl else 1.0

    Pev, yev, qev, cev, gev = strict_parts(sets, ev, wl, pos, gallery=gal,
                                           cfg={**cfg, "neg_per_query": None}, tag="ev")
    if not len(Pev):
        return {}, []
    true_of = {q: sets[q].label for q in np.unique(qev)}

    def fold(s):
        """photo-pair scores -> per-(query, individual) rows + labels."""
        sc, q2, c2, g2 = sv.fold_to_candidates(s, qev, cev, gev)
        y2 = np.array([int(c == sets[q].label) for q, c in zip(q2, c2)], float)
        return sc, y2, q2, c2, g2

    if model_name == "strict":                                   # training-free
        sc, y2, q2, c2, g2 = fold(sm.combine_residual(Pev, lam=cfg["lam"], gamma=cfg["gamma"],
                                                      w_ref=w_ref))
        m = retrieval_metrics(sc, y2, q2, c2, true_of)
        log(epoch=-1, train_loss=None, eval_loss=None, **m)
        return m, photo_pairs(sc, q2, c2, g2, sets)

    Ptr, ytr, _, _, _ = strict_parts(sets, train_imgs, wl, pos, gallery=None,
                                     cfg={**cfg, "neg_per_query": cfg.get("neg_per_query", 30),
                                          "seed": seed}, tag="tr")
    Xtr = sv.residual_features(Ptr, w_ref=w_ref)
    Xev = sv.residual_features(Pev, w_ref=w_ref)
    last: dict = {}
    final = None

    def on_epoch(ep, model, scaler, train_loss):
        nonlocal last, final
        if ep % eval_every and ep != cfg["epochs"] - 1:
            log(epoch=ep, train_loss=train_loss)
            return
        p = _prob(model, scaler, Xev)
        final = p
        sc, y2, q2, c2, _ = fold(p)
        last = retrieval_metrics(sc, y2, q2, c2, true_of)
        log(epoch=ep, train_loss=train_loss, eval_loss=_bce(p, yev), **last)

    train_aggregator(Xtr, ytr, hidden=cfg.get("hidden", 0), n_layers=cfg.get("n_layers", 1),
                     dropout=cfg.get("dropout", 0.1), epochs=cfg["epochs"], lr=cfg["lr"],
                     weight_decay=cfg["weight_decay"], seed=seed, on_epoch=on_epoch)
    if final is None:
        return last, []
    sc, _, q2, c2, g2 = fold(final)
    return last, photo_pairs(sc, q2, c2, g2, sets)


def run_e2e_family(model_name, cfg, sets, tr, ev, *, full_gallery, seed, log, eval_every):
    """e2e transformer: re-embeds the spots itself, so it consumes spot SETS, not features.

    ``sets`` must already carry ``interesting`` (gate supervision) and ``pos`` (position gate);
    :func:`prepare_e2e` does that once per flow.
    """
    import aggregator_e2e_strict as e2s                                    # noqa: PLC0415
    from aggregator_e2e import build_e2e_pairs                             # noqa: PLC0415

    train_imgs = [i for i in tr if not sets[i].is_synth]
    gal_imgs = train_imgs if full_gallery else None
    pairs_tr, y_tr, _, _ = build_e2e_pairs(sets, train_imgs,
                                           neg_per_query=cfg.get("neg_per_query", 25), seed=seed)
    pairs_ev, y_ev, q_ev, c_ev = build_e2e_pairs(sets, ev, gallery_images=gal_imgs,
                                                 neg_per_query=None, seed=seed)
    true_of = {q: sets[q].label for q in np.unique(q_ev)}
    last: dict = {}
    final_scores = None

    def on_epoch(ep, model, train_loss):
        nonlocal last, final_scores
        if ep % eval_every and ep != cfg["epochs"] - 1:
            log(epoch=ep, train_loss=train_loss)
            return
        s = np.asarray(e2s.score_strict_e2e(model, sets, pairs_ev))
        final_scores = s
        last = retrieval_metrics(s, y_ev, q_ev, c_ev, true_of)
        last["gate_auroc"] = float(e2s.gate_auroc(model, sets, ev))
        log(epoch=ep, train_loss=train_loss,
            eval_loss=float(e2s.pair_loss(model, sets, pairs_ev, y_ev)), **last)

    e2s.train_strict_e2e(sets, pairs_tr, y_tr, arch=cfg["arch"], depth=cfg["depth"],
                         n_heads=cfg["n_heads"], dropout=cfg["dropout"], lr=cfg["lr"],
                         epochs=cfg["epochs"], gate_lambda=cfg["gate_lambda"], seed=seed,
                         cosine=True, on_epoch=on_epoch)
    pool = list(ev) + (gal_imgs or [])
    pairs = top1_pairs(final_scores, q_ev, c_ev, sets, pool) if final_scores is not None else []
    return last, pairs


def prepare_e2e(sets):
    """Attach the per-spot inputs the e2e model needs: human 'interesting' labels for the gate
    loss, and body-frame positions for its position gate. No-op cost if e2e is not being run."""
    import aggregator_e2e_strict as e2s                                    # noqa: PLC0415
    import distinctiveness as dist                                         # noqa: PLC0415
    import compare_strict as cs                                            # noqa: PLC0415

    e2s.attach_interesting(sets)
    frame = dist.attach_labels(dist.load_spot_factors(sets), dist.load_interesting())
    e2s.attach_positions(sets, cs.build_pos_lookup(frame))
    return sets


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", default="concat,tensor",
                    help="comma list of flows to compare (concat,tensor)")
    ap.add_argument("--models", default="raw,logreg,mlp,strict,strict_logreg,e2e_xf,e2e_frozen")
    ap.add_argument("--space", choices=["quick", "full"], default="full")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1,
                    help="log full retrieval metrics every N epochs (1 = every epoch)")
    ap.add_argument("--full-gallery", action="store_true",
                    help="rank against the whole population, not just the eval fold")
    ap.add_argument("--limit-configs", type=int, default=0,
                    help="cap configs per model family (0 = no cap)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    embeddings = [e.strip() for e in args.embeddings.split(",") if e.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    space = grids(args.space)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or (OUT_ROOT / d.dataset_name / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ranked_pairs").mkdir(exist_ok=True)
    jsonl = (out_dir / "metrics.jsonl").open("w", encoding="utf-8")

    logger.info(f"run    : {run_id}")
    logger.info(f"out    : {out_dir}")
    logger.info(f"flows  : {embeddings}    models: {models}    space: {args.space}")
    logger.info(f"gallery: {'FULL population' if args.full_gallery else 'eval fold only'}")

    summary: list[dict] = []
    best_pairs: dict[tuple[str, str], dict] = {}
    for emb_name in embeddings:
        table = EMB_TABLES[emb_name]
        try:
            sets = attach_centroids(d.get_image_sets(d.get_spot_embeddings(table=table)))
        except Exception as exc:
            logger.error(f"[{emb_name}] SKIPPED — cannot read '{table}': {exc}")
            logger.error(f"    build it first:  pixi run build-embeddings --combine {emb_name}")
            continue
        dim = sets[0].spots.shape[1]
        if any(m.startswith("e2e") for m in models):
            prepare_e2e(sets)
        folds = d.get_cv_folds(sets, k=args.folds, seed=args.seed)
        logger.info(f"=== flow '{emb_name}' ({table}, dim {dim}) — {len(sets)} image sets ===")

        for model_name in models:
            cfgs = space.get(model_name, [])
            if args.limit_configs:
                cfgs = cfgs[: args.limit_configs]
            for ci, cfg in enumerate(cfgs):
                t0 = time.time()
                per_fold = []
                cfg_pairs: list[dict] = []
                for fi, (tr, ev) in enumerate(folds):
                    def log(**row):
                        jsonl.write(json.dumps({
                            "run_id": run_id, "embedding": emb_name, "emb_dim": dim,
                            "model": model_name, "config_idx": ci, "config": cfg,
                            "fold": fi, "elapsed_s": round(time.time() - t0, 2), **row,
                        }) + "\n")
                        jsonl.flush()

                    runner = (run_e2e_family if model_name.startswith("e2e") else
                              run_strict_family if model_name.startswith("strict") else
                              run_aggregator_family)
                    try:
                        final, pairs = runner(model_name, cfg, sets, tr, ev,
                                              full_gallery=args.full_gallery, seed=args.seed,
                                              log=log, eval_every=max(1, args.eval_every))
                    except Exception as exc:
                        logger.error(f"  [{model_name} cfg{ci} fold{fi}] FAILED: "
                              f"{type(exc).__name__}: {exc}")
                        continue
                    if final:
                        per_fold.append(final)
                    cfg_pairs.extend(pairs)
                if not per_fold:
                    continue
                agg = {k: float(np.mean([f[k] for f in per_fold if k in f]))
                       for k in per_fold[0] if isinstance(per_fold[0][k], (int, float))}
                agg_std = float(np.std([f["r_at_1"] for f in per_fold]))
                row = {"embedding": emb_name, "emb_dim": dim, "model": model_name,
                       "config_idx": ci, "config": json.dumps(cfg), "folds": len(per_fold),
                       "r_at_1_std": round(agg_std, 4),
                       **{k: round(v, 4) for k, v in agg.items()},
                       "elapsed_s": round(time.time() - t0, 1)}
                summary.append(row)
                # Keep the review pairs of the best config per (flow, matcher) only -- reviewing
                # every config's pairs would be hundreds of files nobody opens.
                key = (emb_name, model_name)
                prev = best_pairs.get(key)
                if cfg_pairs and (prev is None or agg.get("r_at_1", 0) > prev["r_at_1"]):
                    best_pairs[key] = {"r_at_1": agg.get("r_at_1", 0.0), "config": cfg,
                                       "config_idx": ci, "pairs": cfg_pairs}
                logger.info(f"  {model_name:11s} cfg{ci:<3d} R@1 {agg.get('r_at_1', float('nan')):.3f}"
                      f"+-{agg_std:.3f}  R@5 {agg.get('r_at_5', float('nan')):.3f}  "
                      f"MRR {agg.get('mrr', float('nan')):.3f}  "
                      f"pairAUC {agg.get('pair_auroc', float('nan')):.3f}  "
                      f"({time.time() - t0:.0f}s)  {cfg}")
    jsonl.close()

    if not summary:
        logger.warning("nothing ran.")
        return 1

    summary.sort(key=lambda r: -r.get("r_at_1", 0))
    keys = list({k for r in summary for k in r})
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(summary)

    lines = [f"# Embedding bakeoff — `{d.dataset_name}` — run `{run_id}`", "",
             f"- gallery: **{'full population' if args.full_gallery else 'eval fold only'}** "
             f"(median candidates/query: {summary[0].get('n_candidates', '?')})",
             f"- folds: {args.folds} · space: `{args.space}` · per-epoch curves in `metrics.jsonl`",
             "",
             "| embedding | dim | model | R@1 | R@5 | MRR | pair AUROC | margin AUROC | config |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in summary[:40]:
        lines.append(f"| {r['embedding']} | {r['emb_dim']} | {r['model']} | "
                     f"**{r.get('r_at_1', float('nan')):.3f}** ± {r['r_at_1_std']:.3f} | "
                     f"{r.get('r_at_5', float('nan')):.3f} | {r.get('mrr', float('nan')):.3f} | "
                     f"{r.get('pair_auroc', float('nan')):.3f} | "
                     f"{r.get('margin_auroc', float('nan')):.3f} | `{r['config']}` |")
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- review pairs: best config per (flow, matcher), for `pair-review-gen --from-bakeoff` ---
    for (emb_name, model_name), rec in best_pairs.items():
        name = f"{emb_name}__{model_name}"
        (out_dir / "ranked_pairs" / f"{name}.json").write_text(json.dumps({
            "run_id": run_id, "dataset": d.dataset_name, "embedding": emb_name,
            "model": model_name, "config": rec["config"], "config_idx": rec["config_idx"],
            "r_at_1": rec["r_at_1"], "rows": rec["pairs"],
        }, indent=2), encoding="utf-8")

    logger.info(f"wrote {out_dir / 'summary.csv'}, {out_dir / 'RESULTS.md'}, "
          f"{out_dir / 'metrics.jsonl'}")
    logger.info(f"wrote {len(best_pairs)} ranked-pair file(s) -> {out_dir / 'ranked_pairs'}")
    best = summary[0]
    logger.info(f"best: {best['embedding']}/{best['model']} cfg{best['config_idx']} "
          f"R@1 {best.get('r_at_1', float('nan')):.3f}")
    logger.info(f"review them:  pixi run pair-review-gen --from-bakeoff {out_dir}\n"
          f"              pixi run pair-review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
