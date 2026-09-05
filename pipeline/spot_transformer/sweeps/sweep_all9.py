"""All nine requested models under ONE census-F0.5 protocol.

The three formulations that have been tried in this repo each learn a different thing, and until
now each had its own driver. This runs all of them against the *same* folds, the same open-set
census split and the same metrics, so the numbers are directly comparable:

  formulation 1 -- summary features (17-dim) -> a classifier          [aggregator.py]
  formulation 2 -- the raw match SET -> a permutation-invariant net   [aggregator_set.py]
  formulation 3 -- the spot embedding AND the voting rule, jointly    [aggregator_e2e.py]

The headline is **census F0.5**, not R@1: for an automated population census a false MATCH
collapses two animals into one profile and DEFLATES the count (mark-recapture cannot undo it),
while a false NEW only inflates. Precision is therefore weighted 2x recall. R@1 is reported as a
ranking diagnostic only. See ``census.py`` for the full argument.

    QUICK=1 pixi run python pipeline/spot_transformer/sweep_all9.py   # 2 folds, few epochs
    MIN_QUALITY=0.4 pixi run python pipeline/spot_transformer/sweep_all9.py
    ONLY=logreg,axial_cnn pixi run python pipeline/spot_transformer/sweep_all9.py
    SOURCE=sasa pixi run python pipeline/spot_transformer/sweep_all9.py   # sasa population only
    SOURCE=sasa SASA_TRAIN_ONLY=1 pixi run python .../sweep_all9.py       # + Haifa gone from training

Results -> ``artifacts/spot_transformer/sweeps/all9<quality_tag>/RESULTS_all9<source_tag>.md``.

SCOPE NOTE on ``e2e_pretrained``: "head on top of a pretrained network" is implemented as the
frozen hand-engineered 62-dim embedding with only the voting head trainable -- the direct
ablation against the two fine-tuned encoders. A true ImageNet-pretrained CNN over per-spot image
crops is a heavier cross-track build (``spot_embedding/train/crops.py`` already extracts the
crops) and is deliberately NOT what this row measures.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# core/ models/ eval/ sweeps/ -- the same bootstrap the rest of the package uses. Inserting only
# this file's own directory held while every module sat flat in spot_transformer/; after the
# refactor it left census/data/novelty/aggregator unresolvable, so the sweep could not be run as a
# script at all.
_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval", "sweeps"):
    _p = str(_ST / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

import census as cen                                          # noqa: E402
import data as d                                              # noqa: E402
import novelty as nov                                         # noqa: E402
from aggregator import (attach_centroids, build_pairs, crossval_aggregator,  # noqa: E402,F401
                        train_aggregator, _prob)
from aggregator_e2e import (build_e2e_openset_pairs, build_e2e_pairs,  # noqa: E402
                            score_e2e, train_e2e)

NOV_MARGIN_IDX = nov.NOVELTY_FEATURES.index("margin")
from aggregator_set import (build_record_pairs, score_pairs_set, train_set)  # noqa: E402
from sweep_set import BETA, per_query_top1  # noqa: E402

QUICK = bool(os.environ.get("QUICK"))
ONLY = {s.strip() for s in os.environ.get("ONLY", "").split(",") if s.strip()}
# Where the quality gate applies. 'filtered' (default, the historical behaviour) drops failing
# photos outright, which gates TRAINING too -- at MIN_QUALITY=0.4 that is ~3x fewer training
# images. 'full' keeps them as training data and gates only what is scored, which measured better
# on R@1, AUROC, census F0.5 and balanced accuracy alike (sweep_quality_control.py).
TRAIN_POOL = os.environ.get("TRAIN_POOL", "filtered")
# Which COLLECTION to score. ``all_sasa_norm`` is two merged populations -- the original sasa study
# (290 individuals, 87 eval-eligible) and the Haifa-KF field export (461, 237). Every number in
# results.md predates that merge and is a sasa-only figure, so ``SOURCE=sasa`` is the like-for-like
# comparison rather than an easier subset. The env var is read in ``data.py``; ``d.source_keep_mask``
# and ``d.source_tag()`` follow it. The gate always restricts the SCORED side (queries + gallery =
# the population the metric is about). ``SASA_TRAIN_ONLY=1`` additionally drops the other collection
# from the training pool (the literal "that data is gone" run) -- off by default because the models
# here are individual-agnostic and the extra photos may still teach something that transfers.
SASA_TRAIN_ONLY = bool(os.environ.get("SASA_TRAIN_ONLY"))
K_FOLDS = 2 if QUICK else 5
SEED = 0
NEG_PER_QUERY = 10 if QUICK else 60
N_BANDS = 16
NOVEL_FRAC = 0.35

# epochs per family (QUICK cuts them hard -- this is a smoke path, not a result)
EP_SET = 15 if QUICK else 250
EP_E2E = 3 if QUICK else 30


# ============================================================ the nine
# family: "feat" = 17-dim summary -> classifier | "set" = match records -> net
#         "e2e"  = spot encoder + voting trained jointly | "raw" = no training
MODELS = [
    # -- formulation 1: the voting rule over summary features
    dict(name="logreg",         family="feat", hidden=0),
    dict(name="mlp_deep",       family="feat", hidden=64, n_layers=3, dropout=0.2,
         weight_decay=1e-3),
    # -- reference: no training at all
    dict(name="raw_voting",     family="raw"),
    # -- formulation 2: the voting rule over the match set
    dict(name="axial_cnn",      family="set", arch="axialcnn", h=32, dropout=0.2,
         weight_decay=1e-3, n_layers=2, kernel=3, n_bands=N_BANDS),
    dict(name="set_transformer", family="set", arch="transformer", h=32, dropout=0.3,
         weight_decay=1e-2, n_layers=1, n_heads=2),
    # #6: DeepSets made CONSERVATIVE = low-variance, NOT low-capacity. The reference config's
    # failure was instability -- fold-3 top-1 collapsed to 0.152 while its R@5 held at 0.826,
    # i.e. seed/init variance, not a broken model. An earlier attempt here crushed capacity
    # (h=16, dropout .4, wd 5e-2, +l1) and the model collapsed outright (census F0.5 0.009,
    # NEGATIVE count bias = predicting match for everything). So: keep the reference capacity
    # that scored 0.493 and kill the variance with a seed ensemble whose mean probability is
    # the score -- the cheapest fix that treats the actual diagnosis.
    dict(name="deepsets_cons",  family="set", arch="deepsets", h=32, dropout=0.3,
         weight_decay=1e-2, ensemble=2 if QUICK else 5),
    # -- formulation 3: learn the embedding AND the voting rule
    # out_dim is left unset so it defaults to the input width (62), keeping the residual path
    # -- the encoder starts at ~identity, i.e. at what `e2e_pretrained` already achieves.
    dict(name="e2e_transformer", family="e2e", arch="transformer", depth=3,
         dropout=0.2, n_heads=2),
    dict(name="e2e_pretrained", family="e2e", arch="frozen", dropout=0.2),
    # --- self-supervised representation learning + its controls -------------------------------
    # The ablation that isolates what PRETRAINING buys. All three share the identical voting head
    # and differ only in the per-spot representation, so any gap is attributable to the encoder:
    #   ssl_simclr  SimCLR-pretrained on ~36k unlabeled crops  (needs no identity labels)
    #   ssl_random  the SAME architecture, never trained       (architecture-only control)
    #   e2e_pretrained (above) the hand-engineered 62-dim      (feature-engineering reference)
    dict(name="ssl_simclr",     family="e2e", arch="frozen", dropout=0.2,
         feature_attr="ssl_feat", ssl_tag="simclr"),
    dict(name="ssl_random",     family="e2e", arch="frozen", dropout=0.2,
         feature_attr="ssl_rand_feat", ssl_tag="random"),
    # The positive-mining ablation: identical encoder and objective, but positives are two
    # photographs of the SAME physical spot rather than two augmentations of one crop. Isolates
    # how much the choice of positive pair matters, independently of architecture.
    dict(name="ssl_corr",       family="e2e", arch="frozen", dropout=0.2,
         feature_attr="ssl_corr_feat", ssl_tag="corr"),
    # Same positives, chosen by MEASUREMENT instead of by argument. eval/mining_audit.py scores the
    # mining rule against the human verdicts and finds precision flat (~0.84) across every
    # SSL_MIN_SIM AND with the RANSAC filter on or off -- so neither knob is controlling label
    # quality, and RANSAC's unresolvable precision gain costs half the training pairs. This row
    # keeps the same ~16% noise rate and trains on 3-4x the data.
    # Built by: SSL_MODE=corr SSL_GEOM=0 SSL_MIN_SIM=0.30 SSL_TAG=corr_audit ssl_pretrain.py
    dict(name="ssl_corr_audit", family="e2e", arch="frozen", dropout=0.2,
         feature_attr="ssl_corr_audit_feat", ssl_tag="corr_audit"),
    # ...and the same again with the human-verified correspondences unioned in. The mined pairs
    # carry ~16% wrong positives that no knob removes; these few hundred carry ~none, and 415 of
    # them are pairs the miner never proposed at all. Tests whether a small clean signal is worth
    # more than a large noisy one at this scale.
    # Built by: SSL_MODE=corr SSL_GEOM=0 SSL_MIN_SIM=0.30 SSL_HUMAN=1 SSL_TAG=corr_human ...
    dict(name="ssl_corr_human", family="e2e", arch="frozen", dropout=0.2,
         feature_attr="ssl_corr_human_feat", ssl_tag="corr_human"),
    # COUPLING the representation to the match objective. The rows above freeze the SSL encoder,
    # so nothing ever tells it what MATCHING needs -- it only ever saw the pretext task. These add
    # a small adapter on top of the SSL features, trained end-to-end through the voting loss, and
    # the residual path starts it at identity so it begins exactly where the frozen row sits and
    # can only improve from there. This is the missing 'fine-tune' half of pretrain->fine-tune.
    dict(name="ssl_finetune",   family="e2e", arch="mlp", depth=2, dropout=0.2,
         feature_attr="ssl_feat", ssl_tag="simclr"),
    dict(name="ssl_corr_finetune", family="e2e", arch="mlp", depth=2, dropout=0.2,
         feature_attr="ssl_corr_feat", ssl_tag="corr"),
    # Augmentation-STRENGTH ablation. Contrastive learning becomes invariant to whatever the
    # augmentations vary, so strength is not a "more is better" dial but a choice about how much
    # of the signal to discard -- and it has to be judged downstream, since pretext loss is known
    # to diverge from downstream utility here. Built by: SSL_AUG=<x> SSL_TAG=aug<x> ssl_pretrain.py
    dict(name="ssl_aug0.5",     family="e2e", arch="frozen", dropout=0.2,
         feature_attr="ssl_aug05_feat", ssl_tag="aug0.5"),
    dict(name="ssl_aug2.0",     family="e2e", arch="frozen", dropout=0.2,
         feature_attr="ssl_aug20_feat", ssl_tag="aug2.0"),
    # longer cosine schedule (the converged 30-epoch run cannot improve by running longer at the
    # same schedule; stretching the schedule is a different experiment)
    dict(name="ssl_ep100",      family="e2e", arch="frozen", dropout=0.2,
         feature_attr="ssl_ep100_feat", ssl_tag="ep100"),
]

# DROPPED (kept here as history, not deleted):
#   e2e_mlp        -- collapsed to a constant score at 30 epochs (census F0.5 nan, R@1 at
#                     chance) after scoring 0.258 at 3 epochs: it diverges with longer training.
#   pretrained_cnn -- frozen ImageNet ResNet18 over spot mask crops. NOT a training failure:
#                     with no training at all its features give a same-vs-different identity gap
#                     of +0.002 (vs +0.024 for the hand-engineered 62-dim), i.e. ImageNet shape
#                     priors do not transfer to binary spot masks. Code kept in
#                     pretrained_cnn.py if it is ever worth revisiting with a trained CNN.


# ============================================================ per-fold evaluation
def _census_metrics(os_scores, os_q, os_c, os_true):
    """Open-set metrics, DECOMPOSED into the two decisions the task really contains.

    census_* is the blended fully-automated number (precision-weighted, TN-blind). The rest
    separate the tractable comparative task from the hard absolute one:
      ident_r1   -- of queries that ARE re-sights, did top-1 name the right animal? (matching)
      novel_rec  -- of queries that are genuinely new, how many were correctly enrolled as new
      os_auroc   -- threshold-FREE separability of known vs novel: is the signal there at all
      bal_acc    -- balanced accuracy of the new-vs-known call, so TN finally counts
    """
    top = per_query_top1(os_scores, os_q, os_c, os_true)
    sweep, best = cen.census_sweep(top["top1"], top["top1_correct"], top["is_known"], BETA)
    nov = cen.novelty_metrics(top["top1"], top["is_known"], best["thr"])

    known = np.asarray(top["is_known"]).astype(bool)
    corr = np.asarray(top["top1_correct"]).astype(bool)
    ident_r1 = float(corr[known].mean()) if known.any() else float("nan")

    return dict(census_f=best["f"], census_p=best["precision"], census_r=best["recall"],
                census_bias=best["count_bias"],
                ident_r1=ident_r1, novel_rec=nov["novel_recall"],
                known_rec=nov["known_recall"], bal_acc=nov["balanced_acc"],
                os_auroc=nov["openset_auroc"])


def _r1(scores, q, c, true):
    return float(np.mean(per_query_top1(scores, q, c, true)["top1_correct"]))


def _novelty_block(sc, oq, oc, os_true, sc_n, nq, nc, nov_true):
    """Compare the three ways of deciding known-vs-novel, plus the human-review budget.

    baseline -- the raw top-1 score (what the census threshold currently uses).
    b'       -- the single best RELATIVE statistic, used directly, no training.
    c'       -- a classifier trained on the train-side known/novel labels.

    All three are scored by open-set AUROC, which is threshold-free: it measures whether the
    score ORDERS known before novel, independent of where any cut is placed. That keeps the
    comparison about signal rather than calibration.
    """
    Xe, known_e, corr_e, _ = nov.build_novelty_table(sc, oq, oc, os_true)
    Xt, known_t, _, _ = nov.build_novelty_table(sc_n, nq, nc, nov_true)
    res: dict = {}

    if len(Xe) == 0 or len(np.unique(known_e)) < 2:
        return dict(auroc_base=float("nan"), auroc_bprime=float("nan"),
                    auroc_cprime=float("nan"), bprime_feat="-",
                    bal_acc_cal=float("nan"), review_at90=float("nan"))

    # baseline + b': every relative feature alone, best one wins
    per_feat = nov.single_feature_auroc(Xe, known_e)
    res["auroc_base"] = per_feat["top1"]
    rel = {k: v for k, v in per_feat.items() if k != "top1" and np.isfinite(v)}
    # a feature that ranks novel ABOVE known is just inverted, so score on distance from 0.5
    best_feat = max(rel, key=lambda k: abs(rel[k] - 0.5)) if rel else "-"
    res["bprime_feat"] = best_feat
    res["auroc_bprime"] = rel.get(best_feat, float("nan"))

    # c': trained on the train-side split only
    cprime = None
    if len(Xt) and len(np.unique(known_t)) == 2:
        nmodel, nscaler = nov.train_novelty(Xt, known_t, seed=SEED)
        cprime = nov.score_novelty(nmodel, nscaler, Xe)          # logits, not probs (ties)
        res["auroc_cprime"] = cen.auroc(cprime, known_e)
    else:
        res["auroc_cprime"] = float("nan")

    # Calibrate on whichever scorer actually won, not on c' by default -- c' is the newest, not
    # automatically the best, and preferring it blindly propagated its failure into balAcc and
    # the review budget when it collapsed.
    cands = [("base", Xe[:, 0], res["auroc_base"])]
    if best_feat != "-":
        cands.append(("b'", Xe[:, nov.NOVELTY_FEATURES.index(best_feat)], res["auroc_bprime"]))
    if cprime is not None:
        cands.append(("c'", cprime, res["auroc_cprime"]))
    # AUROC below 0.5 means the score ranks novel above known: still signal, just inverted
    winner = max(cands, key=lambda t: abs(t[2] - 0.5) if np.isfinite(t[2]) else -1)
    res["novelty_winner"] = winner[0]
    best_score = winner[1] if winner[2] >= 0.5 else -winner[1]

    # fix 1: calibrate the cut for BALANCED accuracy instead of F0.5
    sel = nov.select_threshold(best_score, known_e, criterion="balanced")
    res["bal_acc_cal"] = sel["balanced_acc"]
    res["known_rec_cal"] = sel["known_recall"]
    res["novel_rec_cal"] = sel["novel_recall"]

    # (a) human-in-the-loop: auto-decide the most confident queries, review the rest.
    # "correct" = the full decision was right (a caught re-sight with the right id, or a
    # correctly-enrolled new animal), so the budget reflects end-to-end usefulness.
    # Confidence = distance from the decision boundary: a query far above the cut (clearly a
    # re-sight) or far below it (clearly new) is safe to auto-decide; the uncertain middle is
    # what a human should look at.
    # A novel query is only handled correctly if it was actually REJECTED. Crediting every novel
    # query unconditionally (the previous `1.0`) handed free accuracy to exactly the queries the
    # confidence rule is most sure about -- a novel animal scoring far below the cut is maximally
    # confident, so those rows dominated the high-coverage end of the curve and made review@90
    # optimistic. A novel animal absorbed into an existing profile is the single worst census
    # error there is; it cannot count as a correct decision.
    matched = best_score >= sel["thr"]
    decision_ok = np.where(known_e > 0.5, matched & (corr_e > 0.5), ~matched).astype(float)
    rc = cen.risk_coverage(np.abs(best_score - sel["thr"]), decision_ok)
    ok = np.where(rc["accuracy"] >= 0.90)[0]
    res["review_at90"] = float(1.0 - rc["coverage"][ok[-1]]) if len(ok) else 1.0
    res["aurc"] = rc["aurc"]
    return res


def run_fold(sets, tr, ev, models):
    """Train + score every model on one fold. Returns {name: metrics}."""
    train_imgs = [i for i in tr if not sets[i].is_synth]      # real-only training pool
    gal, qry = cen.make_openset_split(sets, ev, NOVEL_FRAC, SEED)
    os_true = {q: sets[q].label for q in qry}
    # A SECOND open-set split, carved out of the TRAIN images, supplies the known/novel labels the
    # novelty classifier (c') learns from. It must come from train: fitting it on the eval split
    # would leak the very answer it is being scored on.
    gal_n, qry_n = cen.make_openset_split(sets, train_imgs, NOVEL_FRAC, SEED + 101)
    nov_true = {q: sets[q].label for q in qry_n}
    out = {}

    # ---- data views, built once per fold and shared by every model in that family ----
    need = {m["family"] for m in models}
    view: dict = {}
    if "feat" in need:
        view["feat_tr"] = build_pairs(sets, train_imgs, neg_per_query=NEG_PER_QUERY, seed=SEED)
        view["feat_os"] = cen.build_openset_features(sets, gal, qry)
        view["feat_nov"] = cen.build_openset_features(sets, gal_n, qry_n)
    if "set" in need or "raw" in need:
        # set_tr is the TRAINING record set; raw_voting does no training, so building it for a
        # raw-only run is pure waste -- and it is the most expensive view here (every train image
        # against every candidate, with RANSAC per pair). Skipping it makes raw_voting nearly
        # free to include as a training-free control.
        if "set" in need:
            view["set_tr"] = build_record_pairs(sets, train_imgs, neg_per_query=NEG_PER_QUERY,
                                                seed=SEED)
        view["set_os"] = cen.build_openset_records(sets, gal, qry)
        view["set_nov"] = cen.build_openset_records(sets, gal_n, qry_n)
    if any(m.get("n_bands") for m in models):
        view["band_tr"] = build_record_pairs(sets, train_imgs, neg_per_query=NEG_PER_QUERY,
                                             seed=SEED, n_bands=N_BANDS)
        view["band_os"] = cen.build_openset_records(sets, gal, qry, n_bands=N_BANDS)
        view["band_nov"] = cen.build_openset_records(sets, gal_n, qry_n, n_bands=N_BANDS)
    if "e2e" in need:
        view["e2e_tr"] = build_e2e_pairs(sets, train_imgs, neg_per_query=NEG_PER_QUERY, seed=SEED)
        view["e2e_os"] = build_e2e_openset_pairs(sets, gal, qry)
        view["e2e_nov"] = build_e2e_openset_pairs(sets, gal_n, qry_n)

    for m in models:
        t0, fam, name = time.time(), m["family"], m["name"]
        try:
            if fam == "raw":
                recs, oq, oc, _ = view["set_os"]
                sc = np.array([r[:, 0].sum() for r in recs])  # soft-chamfer sum, feature 0
                rn, nq, nc, _ = view["set_nov"]
                sc_n = np.array([r[:, 0].sum() for r in rn])

            elif fam == "feat":
                X, y, _, _ = view["feat_tr"]
                kw = {k: m[k] for k in ("hidden", "n_layers", "dropout", "weight_decay") if k in m}
                model, scaler = train_aggregator(X, y, seed=SEED, **kw)
                Xo, oq, oc = view["feat_os"]
                sc = _prob(model, scaler, Xo)
                Xn, nq, nc = view["feat_nov"]
                sc_n = _prob(model, scaler, Xn)

            elif fam == "set":
                key = "band" if m.get("n_bands") else "set"
                recs_tr, y_tr, _, _ = view[f"{key}_tr"]
                mk = {k: m[k] for k in ("arch", "h", "dropout", "weight_decay", "l1",
                                        "n_heads", "n_layers", "kernel") if k in m}
                n_seed = int(m.get("ensemble", 1))
                Ms, Ss = [], []
                for si in range(n_seed):
                    mo, sca, _ = train_set(recs_tr, y_tr, epochs=EP_SET, seed=SEED + si,
                                           beta=BETA, verbose=False, **mk)
                    Ms.append(mo); Ss.append(sca)
                M = Ms if n_seed > 1 else Ms[0]
                S = Ss if n_seed > 1 else Ss[0]
                recs_os, oq, oc, _ = view[f"{key}_os"]
                sc = score_pairs_set(M, S, recs_os, oq, oc)
                rn, nq, nc, _ = view[f"{key}_nov"]
                sc_n = score_pairs_set(M, S, rn, nq, nc)

            else:                                             # e2e
                pairs_tr, y_tr, _, _ = view["e2e_tr"]
                kw = {k: m[k] for k in ("arch", "out_dim", "depth", "dropout", "n_heads",
                                        "feature_attr", "residual") if k in m}
                fattr = m.get("feature_attr", "spots")
                model = train_e2e(sets, pairs_tr, y_tr, epochs=EP_E2E, seed=SEED,
                                  verbose=False, **kw)
                pairs_os, oq, oc = view["e2e_os"]
                sc = score_e2e(model, sets, pairs_os, feature_attr=fattr)
                pairs_n, nq, nc = view["e2e_nov"]
                sc_n = score_e2e(model, sets, pairs_n, feature_attr=fattr)

            met = _census_metrics(sc, oq, oc, os_true)
            met.update(_novelty_block(sc, oq, oc, os_true, sc_n, nq, nc, nov_true))
            met["r1"] = _r1(sc, oq, oc, os_true)
            met["secs"] = time.time() - t0
            out[name] = met
            logger.info(f"    {name:<17} identR@1 {met['ident_r1']:.3f}  |  novelty AUROC: "
                  f"base {met['auroc_base']:.3f}  b' {met['auroc_bprime']:.3f} "
                  f"({met['bprime_feat']})  c' {met['auroc_cprime']:.3f}  |  "
                  f"balAcc {met['bal_acc_cal']:.3f}  review@90% {met['review_at90']:.0%}  "
                  f"({met['secs']:.0f}s)")
        except Exception as exc:                              # one model failing must not kill the sweep
            logger.error(f"    {name:<17} FAILED: {type(exc).__name__}: {exc}")
            out[name] = None
    return out


# ============================================================ driver
def prepare() -> tuple[list, list, list[dict]]:
    """Data load -> quality/source gates -> SSL/CNN feature attachment -> CV folds.

    Shared by ``main()`` and ``train_all13.py`` so both run the identical bake-off
    protocol against the identical data. Pure setup — no training happens here.
    """
    sets = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))
    # Both branches are no-ops unless MIN_QUALITY / MAX_SPOTS_OUTSIDE is set.
    if TRAIN_POOL == "full":
        eval_mask = d.quality_keep_mask(sets)                 # gate the SCORED side only
    else:
        sets = d.apply_quality_filter(sets)                   # gate everything (historical)
        eval_mask = None

    # Source gate: restrict the SCORED side (queries + the census gallery they rank against) to one
    # collection. Computed AFTER apply_quality_filter so the mask stays aligned to ``sets``. All-True
    # unless SOURCE=sasa|kf, so this is a no-op for the default run.
    source_keep = d.source_keep_mask(sets)
    if not source_keep.all():
        eval_mask = source_keep if eval_mask is None else (eval_mask & source_keep)
    models = [m for m in MODELS if not ONLY or m["name"] in ONLY]

    # The pretrained-CNN row needs frozen ResNet18 features over per-spot crops. That is the one
    # dependency outside this track (torchvision weights + the spot_embedding crop cache), so it
    # is attached only when the row is actually selected, and its absence drops just that row
    # instead of failing the whole sweep.
    if any(m.get("feature_attr") == "cnn_feat" for m in models):
        try:
            from pretrained_cnn import attach_cnn_features
            logger.info(" attaching frozen ResNet18 spot features ...")
            sets = attach_cnn_features(sets)
        except Exception as exc:
            logger.warning(f" !! pretrained_cnn unavailable ({type(exc).__name__}: {exc}) — "
                  f"dropping that row")
            models = [m for m in models if m.get("feature_attr") != "cnn_feat"]

    # Self-supervised rows need their pretrained caches (built by ssl_pretrain.py). Each is
    # attached independently so a missing control drops only its own row.
    want: dict[str, str] = {}
    for m in models:
        if m.get("ssl_tag"):
            want[m["feature_attr"]] = m["ssl_tag"]
    for attr, tag in want.items():
        try:
            from ssl_pretrain import attach_ssl_features
            logger.info(f" attaching SSL spot features [{tag}] ...")
            sets = attach_ssl_features(sets, tag=tag, attr=attr)
        except Exception as exc:
            logger.warning(f" !! SSL cache '{tag}' unavailable ({type(exc).__name__}: {exc}) — dropping "
                  f"that row. Build it with: "
                  f"{ {'random': 'SSL_RANDOM_INIT=1 SSL_TAG=random ',
                       'corr': 'SSL_MODE=corr SSL_TAG=corr '}.get(tag, '') }"
                  f"pixi run python pipeline/spot_transformer/ssl_pretrain.py")
            models = [m for m in models if m.get("feature_attr") != attr]
    folds = d.get_cv_folds(sets, k=K_FOLDS, seed=SEED, eval_mask=eval_mask)
    if SASA_TRAIN_ONLY and not source_keep.all():
        # Drop the other collection from every fold's training pool too -- the "that data is gone"
        # run. eval_idx is already source-gated via eval_mask, so only the train side changes here.
        folds = [([i for i in tr if source_keep[i]], ev) for tr, ev in folds]
    d.assert_evaluable(folds, what=f"the source gate (SOURCE={d.SOURCE})")
    return sets, folds, models


def run_cv(sets, folds, models) -> dict[str, list]:
    """Run every fold, printing the same per-model lines ``main()`` always has."""
    per_fold: dict[str, list] = {m["name"]: [] for m in models}
    for fi, (tr, ev) in enumerate(folds):
        logger.info(f"=== fold {fi} " + "=" * 50)
        res = run_fold(sets, tr, ev, models)
        for k, v in res.items():
            if v is not None:
                per_fold[k].append(v)
    return per_fold


def aggregate_rows(models: list[dict], per_fold: dict[str, list]) -> tuple[list[tuple], dict[str, int]]:
    """Mean/std per model over folds -> the ``rows`` the console table and RESULTS.md share,
    plus which single relative novelty feature won b' most often (``feat_votes``)."""
    def agg(name, key):
        v = [r[key] for r in per_fold[name] if r is not None]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))

    rows = []
    for m in models:
        n = m["name"]
        if not per_fold[n]:
            continue
        ir_m, ir_s = agg(n, "ident_r1")
        rows.append((n, m["family"], ir_m, ir_s,
                     agg(n, "auroc_base")[0], agg(n, "auroc_bprime")[0],
                     agg(n, "auroc_cprime")[0], agg(n, "bal_acc_cal")[0],
                     agg(n, "review_at90")[0], agg(n, "census_f")[0]))
    # sort by identification R@1: the tractable half of the task, and the half that actually
    # discriminates between models. Novelty is reported alongside but barely separates them.
    rows.sort(key=lambda r: -(r[2] if np.isfinite(r[2]) else -1))

    # which relative feature won b', pooled over folds
    feat_votes: dict[str, int] = {}
    for m in models:
        for r in per_fold[m["name"]]:
            f = r.get("bprime_feat")
            if f and f != "-":
                feat_votes[f] = feat_votes.get(f, 0) + 1
    return rows, feat_votes


def main():
    # The console here is cp1255, which cannot encode characters like U+2032 PRIME. Without this
    # the summary table raises UnicodeEncodeError *after* every model has finished training --
    # i.e. it throws away a multi-hour run at the last print. Results files are written UTF-8
    # explicitly, so this only concerns stdout.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    sets, folds, models = prepare()

    logger.info("=" * 78)
    logger.info(f" ALL-9 SWEEP  ({len(models)} models x {K_FOLDS} folds)"
          f"{'   [QUICK]' if QUICK else ''}")
    logger.info(f" headline = census F0.5 (precision weighted 2x recall); R@1 = diagnostic only")
    logger.info(f" source = {d.SOURCE}"
          + ("  (Haifa also removed from training)" if SASA_TRAIN_ONLY else "")
          + ("" if d.SOURCE == "all" else "   -- results.md numbers predate the Haifa merge"))
    logger.info("=" * 78)

    per_fold = run_cv(sets, folds, models)
    rows, feat_votes = aggregate_rows(models, per_fold)

    logger.info("=" * 100)
    h_b, h_c = "b'", "c'"                                     # ASCII, not U+2032 (see main())
    logger.info(f" {'model':<17}{'fam':<6}{'identR@1':>14}{'AUROC base':>12}{h_b:>8}"
          f"{h_c:>8}{'balAcc':>9}{'review@90':>11}")
    logger.info("-" * 100)
    for n, fam, ir, irs, ab, bb, cb, ba, rv, cf in rows:
        logger.info(f" {n:<17}{fam:<6}{ir:>8.3f}±{irs:<5.3f}{ab:>12.3f}{bb:>8.3f}{cb:>8.3f}"
              f"{ba:>9.3f}{rv:>10.0%}")
    if feat_votes:
        top = sorted(feat_votes.items(), key=lambda t: -t[1])
        logger.info(f" {h_b} winning feature (folds x models): "
              + ", ".join(f"{k} x{v}" for k, v in top[:4]))

    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "sweeps" / f"all9{d.quality_tag()}"
    outdir.mkdir(parents=True, exist_ok=True)
    # A partial (ONLY=/QUICK) or differently-gated (TRAIN_POOL=) run must not overwrite the full
    # table -- none of them are comparable to it.
    # emb_tag() names the active EMB_TABLE. Without it, a representation sweep (baseline vs
    # +size vs +morph vs h20) writes every variant to the SAME RESULTS file and each run silently
    # destroys the previous one -- so the comparison the sweep exists to make would be impossible
    # to read afterwards.
    suffix = ("_quick" if QUICK else "") + ("_subset" if ONLY else "") \
        + ("_trainfull" if TRAIN_POOL == "full" else "") + d.emb_tag() \
        + d.source_tag() + ("_traingone" if (SASA_TRAIN_ONLY and d.SOURCE != "all") else "")
    fname = f"RESULTS_all9{suffix}.md"
    md = ["# All-9 comparison — census F0.5", "",
          ("> **QUICK / partial run — not a result.** Few folds and few epochs; use it to check "
           "the plumbing, not to rank models.") if (QUICK or ONLY) else "",
          f"- folds: {K_FOLDS} · seed {SEED} · neg/query {NEG_PER_QUERY}"
          f" · epochs set/e2e {EP_SET}/{EP_E2E}"
          f" · quality filter `{d.quality_tag() or 'none'}`"
          f" · train pool `{TRAIN_POOL}`"
          f" · source `{d.SOURCE}`"
          + ("  (Haifa-KF removed from training too)" if (SASA_TRAIN_ONLY and d.SOURCE != "all")
             else ""),
          ("> Scored on the **sasa** population only (the original study group). results.md's "
           "numbers are sasa-only figures from before the Haifa-KF merge, so this is the "
           "like-for-like comparison.") if d.SOURCE == "sasa" else "",
          "",
          "The task is two decisions of very different difficulty, scored separately:",
          "",
          "| column | question |",
          "|---|---|",
          "| `identR@1` | of queries that ARE re-sights, did top-1 name the right animal? (**tractable**) |",
          "| `AUROC base` | can the raw top-1 score tell known from novel? (0.5 = coin flip) |",
          "| `b′` | best single **relative** statistic used alone, no training |",
          "| `c′` | classifier trained on train-side known/novel labels |",
          "| `balAcc` | balanced accuracy after recalibrating the cut (fix 1) |",
          "| `review@90` | fraction of photos a human must check to reach 90% end-to-end accuracy |",
          "",
          "All three novelty scores are compared by **threshold-free AUROC**, so the comparison",
          "measures signal rather than where a cut happens to sit. Thresholds are chosen for",
          "**balanced accuracy**, not F0.5 — the latter rewards labelling everything 'new'.",
          "",
          "| model | formulation | identR@1 | AUROC base | b′ | c′ | balAcc | review@90 | censusF |",
          "|---|---|---|---|---|---|---|---|---|"]
    fam_name = {"feat": "summary→clf", "set": "match-set→net", "e2e": "encoder+voting",
                "raw": "none (baseline)"}
    for n, fam, ir, irs, ab, bb, cb, ba, rv, cf in rows:
        md.append(f"| {n} | {fam_name[fam]} | **{ir:.3f} ± {irs:.3f}** | {ab:.3f} | {bb:.3f} "
                  f"| {cb:.3f} | {ba:.3f} | {rv:.0%} | {cf:.3f} |")
    if feat_votes:
        top = sorted(feat_votes.items(), key=lambda t: -t[1])
        md += ["", "**b′ winning feature** (folds × models): "
               + ", ".join(f"`{k}` ×{v}" for k, v in top[:5])]
    (outdir / fname).write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info(f"wrote {outdir / fname}")


if __name__ == "__main__":
    main()
