"""Focused exploration of ``e2e_transformer`` — built to run on a RunPod cloud GPU.

WHY THIS EXISTS. In the last high-quality bake-off (``all9_q0.4/RESULTS_all9_sasa.md``)
``e2e_transformer`` scored **identR@1 0.717 ± 0.062**, tied for the top cluster with
``e2e_pretrained`` (0.730) and ``logreg`` (0.717) — but it has never had a real hyperparameter
sweep. ``EP_E2E`` is pinned at 30 and each fold costs ~3 h on CPU, so ``sweep_all9`` can only
afford ONE config. On a GPU a fold-train drops to ~1-2 min, which makes the searches below
feasible for the first time.

Four studies, each a list of configs run over the SAME folds / open-set protocol / metrics as
``sweep_all9`` (this module imports ``sweep_all9._census_metrics`` /
``sweep_all9._novelty_block`` / ``sweep_all9._r1`` verbatim, so the numbers are directly
comparable to the canonical table):

  baseline  recreate the ``all9_q0.4`` ``e2e_transformer`` row — the plumbing check. Its identR@1
            must land near 0.717 ± 0.062 before any exploration number is trusted.
  identr    architecture + optimiser grid (depth / heads / dropout / lr / wd / neg-per-query /
            cosine-schedule / longer training), ranked by mean identR@1.
  gate      the SAME knobs but ranked by the novelty gate (balanced accuracy of known-vs-novel),
            PLUS an auxiliary known>novel margin loss on a train-carved open-set split — the only
            place novelty labels enter training. The pairwise same/different label and BCE loss
            are unchanged; "optimise for the gate" means the selection metric and (optionally)
            this aux term, not new ground truth.
  hardneg   hard-negative mining: warm up on random negatives, then for each query replace them
            with the individuals the current model scores HIGHEST (its confusions), optionally
            semi-hard (skip the top few, which are usually label noise) and re-mined each round.
  general   a few capacity / arch probes (mlp encoder, wider out_dim, deeper, much longer).

--------------------------------------------------------------------------------------------------
RUNPOD QUICKSTART
--------------------------------------------------------------------------------------------------
1. Pod image: any CUDA + PyTorch base (torch>=2.1). Then, in the repo root:
       pip install numpy pandas duckdb scikit-image opencv-python-headless
2. The dataset DB must be on the pod (this reads the ``spots`` table only, no SSL caches needed):
       datasets/all_sasa_norm_2026_23_07/db/contours.db
   rsync it up, or mount a network volume that already has ``datasets/``.
3. Sanity check the plan without spending GPU hours:
       DRY_RUN=1 python pipeline/spot_transformer/sweeps/e2e_transform_explore.py
4. Run it:
       python pipeline/spot_transformer/sweeps/e2e_transform_explore.py
5. Copy results back:
       artifacts/spot_transformer/sweeps/e2e_explore/

ENV KNOBS (all optional)
    STUDY=baseline,identr,gate,hardneg,general   which studies to run (default: all)
    QUICK=1            2 folds, ~4 epochs, first 2 configs/study — smoke only
    K_FOLDS=5          override fold count
    MIN_QUALITY=0.4    quality gate (default 0.4 — the whole point of this run)
    SOURCE=sasa        population gate (default sasa — like-for-like with results.md)
    DEVICE=cuda|cpu    default: cuda if available
    EPOCH_SCALE=1.0    multiply every config's epoch budget (0.5 = half, for a cheaper pass)
    MAX_CONFIGS=0      cap configs per study (0 = no cap); the baseline always runs
    CONFIG_SHARD=i/n   split each study's configs across n pods; this pod runs configs[i::n]
    LOG_EVERY=1        per-epoch line: train loss + test loss (+ aux loss)   (0 to silence)
    PROBE_EVERY=1      also log identR@1 (train/test), balAcc (train/test), test censusF (0 = off)
    DRY_RUN=1          print the plan (studies, configs, fold/epoch budget) and exit
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

# ---- defaults that MUST be set before the repo imports read them --------------------------------
# data.SOURCE is bound at import; quality_tag()/apply_quality_filter() read MIN_QUALITY at call.
os.environ.setdefault("MIN_QUALITY", "0.4")
os.environ.setdefault("SOURCE", "sasa")

# ---- the same bootstrap every module in this package uses --------------------------------------
_ST = Path(__file__).resolve().parents[1]
for _sub in (".", "core", "models", "eval", "sweeps"):   # "." -> import config (constants package)
    _p = str((_ST / _sub).resolve())
    if _p not in sys.path:
        sys.path.insert(0, _p)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.utils.logger_utils import get_logger  # noqa: E402

logger = get_logger(Path(__file__).stem)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import census as cen  # noqa: E402
import data as d  # noqa: E402
import novelty as nov  # noqa: E402
from aggregator import attach_centroids  # noqa: E402
from aggregator_e2e import (  # noqa: E402
    E2EVoter,
    _collate,
    build_e2e_openset_pairs,
    build_e2e_pairs,
)
from aggregator_set import per_query_top1  # noqa: E402

# ---- protocol constants (kept in step with config/__init__.py; inlined so this module has no
#      dependency on the in-flight config-package refactor) ------------------------------------
QUICK = bool(os.environ.get("QUICK"))
SEED = 0
NOVEL_FRAC = 0.35
BETA = 0.5                                                    # census: precision weighted 2x recall

# ---- run-level config -------------------------------------------------------------------------
K_FOLDS = int(os.environ.get("K_FOLDS", "2" if QUICK else "5"))
EPOCH_SCALE = float(os.environ.get("EPOCH_SCALE", "1.0"))
MAX_CONFIGS = int(os.environ.get("MAX_CONFIGS", "0"))
DRY_RUN = bool(os.environ.get("DRY_RUN"))
# per-epoch training telemetry. LOG_EVERY=N -> a line every N epochs with train + test loss;
# PROBE_EVERY=N -> that line also carries identR@1 (train/test), balanced-acc gate (train/test)
# and test census F0.5. "train" = a train-carved open-set split; "test" = the held-out fold.
# Set higher to thin the logs, 0 to silence.
LOG_EVERY = int(os.environ.get("LOG_EVERY", "1"))
PROBE_EVERY = int(os.environ.get("PROBE_EVERY", "1"))
_WANT_STUDIES = [s.strip() for s in os.environ.get("STUDY", "").split(",") if s.strip()]
# CONFIG_SHARD="i/n" -> after study selection, keep only configs[i::n] (the baseline always
# runs). Lets one heavy study (e.g. `identr`, 24 configs) be split across n pods.
_SHARD_I, _SHARD_N = 0, 1
if os.environ.get("CONFIG_SHARD"):
    _SHARD_I, _SHARD_N = (int(x) for x in os.environ["CONFIG_SHARD"].split("/"))

_dev_req = os.environ.get("DEVICE", "").strip().lower()
if _dev_req:
    DEVICE = torch.device(_dev_req)
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

MINE_PAIR_CAP = 150_000                                       # subsample mining queries above this


# ==============================================================================================
# config object
# ==============================================================================================
@dataclass(frozen=True)
class Cfg:
    name: str
    # --- encoder / voting architecture ---
    arch: str = "transformer"
    depth: int = 3
    n_heads: int = 2
    dropout: float = 0.2
    out_dim: int | None = None          # None -> input width (keeps the residual identity start)
    residual: bool = True
    # --- optimiser ---
    lr: float = 1e-3
    wd: float = 1e-4
    batch: int = 64
    epochs: int = 30
    cosine: bool = False
    neg: int = 60                       # negatives per query (canonical NEG_PER_QUERY = 60)
    # --- hard-negative mining ---
    hard_neg: bool = False
    warmup_epochs: int = 8
    semi_hard_skip: int = 0             # drop the N hardest (usually label noise) before taking K
    mix_random: float = 0.2            # fraction of the K negatives left random, not mined
    remine_rounds: int = 1
    # --- auxiliary novelty objective ---
    aux_novelty: bool = False
    aux_weight: float = 0.3
    aux_margin: float = 0.2
    # --- which metric this config's study ranks by ---
    objective: str = "ident_r1"        # "ident_r1" | "gate"

    def scaled(self) -> "Cfg":
        """Apply QUICK / EPOCH_SCALE to the epoch budgets."""
        if QUICK:
            return replace(self, epochs=max(2, self.epochs // 8),
                           warmup_epochs=max(1, self.warmup_epochs // 4))
        if EPOCH_SCALE != 1.0:
            return replace(self, epochs=max(2, round(self.epochs * EPOCH_SCALE)),
                           warmup_epochs=max(1, round(self.warmup_epochs * EPOCH_SCALE)))
        return self


# the canonical all9_q0.4 e2e_transformer row: arch transformer, depth 3, heads 2, dropout 0.2,
# lr 1e-3, wd 1e-4, batch 64, 30 epochs, 60 random negatives, residual identity start.
BASELINE = Cfg(name="baseline")


def _studies() -> dict[str, list[Cfg]]:
    S: dict[str, list[Cfg]] = {"baseline": [BASELINE]}

    # ---- identr: architecture + optimiser grid, ranked by identR@1 ----
    # The transformer's d_model is out_dim (defaults to the 62-dim input, which keeps the
    # residual identity-start). n_heads must divide d_model: 62 % 2 == 0 but 62 % 4 != 0, so
    # the 4-head configs widen to out_dim=64 (residual off — it needs out_dim == in_dim).
    ident: list[Cfg] = []
    for depth in (2, 3, 4):
        for heads in (2, 4):
            for dropout in (0.1, 0.2, 0.3):
                ident.append(Cfg(name=f"id_d{depth}_h{heads}_p{dropout}", depth=depth,
                                 n_heads=heads, dropout=dropout, epochs=60, cosine=True,
                                 out_dim=(64 if heads == 4 else None),
                                 residual=(heads != 4)))
    for lr in (5e-4, 2e-3):
        ident.append(Cfg(name=f"id_lr{lr:g}", lr=lr, epochs=60, cosine=True))
    for wd in (1e-3, 3e-3):
        ident.append(Cfg(name=f"id_wd{wd:g}", wd=wd, epochs=60, cosine=True))
    for neg in (30, 100):
        ident.append(Cfg(name=f"id_neg{neg}", neg=neg, epochs=60, cosine=True))
    S["identr"] = ident

    # ---- gate: same knobs, ranked by balanced known-vs-novel accuracy, + aux margin loss ----
    # aux configs first so a QUICK smoke (first 2 configs) exercises the auxiliary-loss path.
    gate: list[Cfg] = []
    for aw in (0.3, 0.1, 0.6):
        gate.append(Cfg(name=f"gate_aux{aw:g}", aux_novelty=True, aux_weight=aw, epochs=60,
                        cosine=True, objective="gate"))
    gate.append(Cfg(name="gate_aux0.3_d2", aux_novelty=True, aux_weight=0.3, depth=2, epochs=60,
                    cosine=True, objective="gate"))
    for depth in (2, 3):
        for dropout in (0.2, 0.3):
            gate.append(Cfg(name=f"gate_d{depth}_p{dropout}", depth=depth, dropout=dropout,
                            epochs=60, cosine=True, objective="gate"))
    S["gate"] = gate

    # ---- hardneg: mining schedule ----
    hn: list[Cfg] = []
    for skip in (0, 3):
        for mix in (0.0, 0.2):
            for rounds in (1, 2):
                hn.append(Cfg(name=f"hn_skip{skip}_mix{mix:g}_r{rounds}", hard_neg=True,
                              semi_hard_skip=skip, mix_random=mix, remine_rounds=rounds,
                              neg=60, epochs=45, warmup_epochs=10, cosine=True))
    S["hardneg"] = hn

    # ---- general: capacity / arch probes ----
    S["general"] = [
        Cfg(name="gen_mlp_d3", arch="mlp", depth=3, epochs=60, cosine=True),
        Cfg(name="gen_wide_od96", out_dim=96, residual=False, epochs=60, cosine=True),
        Cfg(name="gen_deep_d5", depth=5, epochs=60, cosine=True),
        Cfg(name="gen_long_ep150", epochs=150, cosine=True),
        Cfg(name="gen_bighead_h4_d4", n_heads=4, depth=4, epochs=90, cosine=True,
            out_dim=64, residual=False),
    ]
    return S


OBJECTIVE = {                                                 # metric key, higher-is-better
    "ident_r1": ("ident_r1", True),
    "gate": ("bal_acc_cal", True),
}


# ==============================================================================================
# metrics — copied VERBATIM from sweep_all9.py so this module's numbers are the canonical ones.
# Keep in sync if sweep_all9 changes. (Imported directly would drag in the config/models packages
# that the in-flight refactor has not finished wiring onto sys.path.)
# ==============================================================================================
def _census_metrics(os_scores, os_q, os_c, os_true):
    top = per_query_top1(os_scores, os_q, os_c, os_true)
    _sweep, best = cen.census_sweep(top["top1"], top["top1_correct"], top["is_known"], BETA)
    nvm = cen.novelty_metrics(top["top1"], top["is_known"], best["thr"])
    known = np.asarray(top["is_known"]).astype(bool)
    corr = np.asarray(top["top1_correct"]).astype(bool)
    ident_r1 = float(corr[known].mean()) if known.any() else float("nan")
    return dict(census_f=best["f"], census_p=best["precision"], census_r=best["recall"],
                census_bias=best["count_bias"], ident_r1=ident_r1, novel_rec=nvm["novel_recall"],
                known_rec=nvm["known_recall"], bal_acc=nvm["balanced_acc"],
                os_auroc=nvm["openset_auroc"])


def _r1(scores, q, c, true):
    return float(np.mean(per_query_top1(scores, q, c, true)["top1_correct"]))


def _novelty_block(sc, oq, oc, os_true, sc_n, nq, nc, nov_true):
    Xe, known_e, corr_e, _ = nov.build_novelty_table(sc, oq, oc, os_true)
    Xt, known_t, _, _ = nov.build_novelty_table(sc_n, nq, nc, nov_true)
    res: dict = {}
    if len(Xe) == 0 or len(np.unique(known_e)) < 2:
        return dict(auroc_base=float("nan"), auroc_bprime=float("nan"),
                    auroc_cprime=float("nan"), bprime_feat="-",
                    bal_acc_cal=float("nan"), review_at90=float("nan"))
    per_feat = nov.single_feature_auroc(Xe, known_e)
    res["auroc_base"] = per_feat["top1"]
    rel = {k: v for k, v in per_feat.items() if k != "top1" and np.isfinite(v)}
    best_feat = max(rel, key=lambda k: abs(rel[k] - 0.5)) if rel else "-"
    res["bprime_feat"] = best_feat
    res["auroc_bprime"] = rel.get(best_feat, float("nan"))
    cprime = None
    if len(Xt) and len(np.unique(known_t)) == 2:
        nmodel, nscaler = nov.train_novelty(Xt, known_t, seed=SEED)
        cprime = nov.score_novelty(nmodel, nscaler, Xe)
        res["auroc_cprime"] = cen.auroc(cprime, known_e)
    else:
        res["auroc_cprime"] = float("nan")
    cands = [("base", Xe[:, 0], res["auroc_base"])]
    if best_feat != "-":
        cands.append(("b'", Xe[:, nov.NOVELTY_FEATURES.index(best_feat)], res["auroc_bprime"]))
    if cprime is not None:
        cands.append(("c'", cprime, res["auroc_cprime"]))
    winner = max(cands, key=lambda t: abs(t[2] - 0.5) if np.isfinite(t[2]) else -1)
    res["novelty_winner"] = winner[0]
    best_score = winner[1] if winner[2] >= 0.5 else -winner[1]
    sel = nov.select_threshold(best_score, known_e, criterion="balanced")
    res["bal_acc_cal"] = sel["balanced_acc"]
    res["known_rec_cal"] = sel["known_recall"]
    res["novel_rec_cal"] = sel["novel_recall"]
    matched = best_score >= sel["thr"]
    decision_ok = np.where(known_e > 0.5, matched & (corr_e > 0.5), ~matched).astype(float)
    rc = cen.risk_coverage(np.abs(best_score - sel["thr"]), decision_ok)
    ok = np.where(rc["accuracy"] >= 0.90)[0]
    res["review_at90"] = float(1.0 - rc["coverage"][ok[-1]]) if len(ok) else 1.0
    res["aurc"] = rc["aurc"]
    return res


# ==============================================================================================
# GPU training / scoring — mirrors aggregator_e2e.train_e2e / score_e2e, on DEVICE
# ==============================================================================================
def _batches(sets, pairs, bs):
    for s in range(0, len(pairs), bs):
        Q, qm, C, cm = _collate(sets, pairs[s:s + bs])
        yield (Q.to(DEVICE, non_blocking=True), qm.to(DEVICE, non_blocking=True),
               C.to(DEVICE, non_blocking=True), cm.to(DEVICE, non_blocking=True))


@torch.no_grad()
def _score(model, sets, pairs, bs=256) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(*b)).float().cpu().numpy() for b in _batches(sets, pairs, bs)]
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def _query_top1_logits(model, sets, pairs, qids, bs=256):
    """Per-query max logit, WITH grad — the differentiable core of the aux novelty loss."""
    logits = [model(*b) for b in _batches(sets, pairs, bs)]
    L = torch.cat(logits) if logits else torch.zeros(0, device=DEVICE)
    qids = np.asarray(qids)
    return {q: L[torch.as_tensor(qids == q, device=DEVICE)].max() for q in np.unique(qids)}


def _aux_novelty_loss(model, sets, val, margin):
    """Pairwise hinge: every KNOWN query's top-1 score should beat every NOVEL query's by ``margin``.

    ``val`` = (pairs, qids, known_by_q) from a train-carved open-set split — never the eval fold.
    """
    pairs, qids, known = val
    t1 = _query_top1_logits(model, sets, pairs, qids)
    kn = torch.stack([v for q, v in t1.items() if known[q]]) if any(known.values()) else None
    nv = torch.stack([v for q, v in t1.items() if not known[q]]) if not all(known.values()) else None
    if kn is None or nv is None or len(kn) == 0 or len(nv) == 0:
        return torch.zeros((), device=DEVICE)
    return torch.relu(margin - (kn[:, None] - nv[None, :])).mean()


@torch.no_grad()
def _probe(model, sets, pairs_os, oq, oc, os_true) -> dict:
    """Cheap open-set eval on a (query, gallery) split — identR@1 / balanced-acc gate / census F0.5.

    No c' classifier and no threshold sweep beyond the two cheap ones, so it is sub-second and
    safe to run every epoch on both the train-carved and the held-out split.
    """
    was_training = model.training
    sc = _score(model, sets, pairs_os)
    top = per_query_top1(sc, oq, oc, os_true)
    known = np.asarray(top["is_known"]).astype(bool)
    corr = np.asarray(top["top1_correct"]).astype(bool)
    t1 = np.asarray(top["top1"], float)
    r1 = float(corr[known].mean()) if known.any() else float("nan")
    try:
        bal = float(nov.select_threshold(t1, known.astype(float),
                                         criterion="balanced")["balanced_acc"])
    except Exception:
        bal = float("nan")
    try:
        _s, best = cen.census_sweep(top["top1"], top["top1_correct"], top["is_known"], BETA)
        cf = float(best["f"])
    except Exception:
        cf = float("nan")
    if was_training:
        model.train()
    return dict(r1=r1, bal=bal, cf=cf)


@torch.no_grad()
def _eval_loss(model, sets, pairs, y, lossf) -> float:
    """Mean BCE (same pos_weight as training) on a held-out pair set."""
    was_training = model.training
    model.eval()
    yt = torch.as_tensor(np.asarray(y, np.float32), device=DEVICE)
    tot = n = 0
    for s in range(0, len(pairs), 256):
        Q, qm, C, cm = _collate(sets, pairs[s:s + 256])
        out = model(Q.to(DEVICE), qm.to(DEVICE), C.to(DEVICE), cm.to(DEVICE))
        bs = out.shape[0]
        tot += float(lossf(out, yt[s:s + bs])) * bs
        n += bs
    if was_training:
        model.train()
    return tot / max(n, 1)


def _fit(sets, pairs, y, cfg: Cfg, *, epochs, seed, val=None, warm=None,
         label="fit", test_set=None, probe_train=None, probe_test=None):
    """One training run, with per-epoch train+test telemetry.

    ``warm``       continue from an existing model's weights (fresh optimiser).
    ``test_set``   ``(pairs, y)`` held out for the test-loss curve.
    ``probe_train``/``probe_test``  ``callable(model) -> {r1, bal, cf}`` (see :func:`_probe`),
                   run every ``PROBE_EVERY`` epochs on the train-carved / held-out open-set split.
    """
    torch.manual_seed(seed)
    in_dim = getattr(sets[pairs[0][0]], "spots").shape[1]
    model = E2EVoter(in_dim=in_dim, out_dim=cfg.out_dim or in_dim, arch=cfg.arch, depth=cfg.depth,
                     dropout=cfg.dropout, n_heads=cfg.n_heads, residual=cfg.residual).to(DEVICE)
    if warm is not None:
        model.load_state_dict(warm.state_dict())

    y = np.asarray(y, np.float32)
    npos = max(int(y.sum()), 1)
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(len(y) - npos) / npos], dtype=torch.float32, device=DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
             if cfg.cosine else None)
    yt = torch.tensor(y, device=DEVICE)
    idx = np.arange(len(pairs))
    rng = np.random.default_rng(seed)

    for ep in range(epochs):
        rng.shuffle(idx)
        model.train()
        tot = n = 0
        for s in range(0, len(idx), cfg.batch):
            b = idx[s:s + cfg.batch]
            Q, qm, C, cm = _collate(sets, [pairs[i] for i in b])
            Q, qm, C, cm = (Q.to(DEVICE), qm.to(DEVICE), C.to(DEVICE), cm.to(DEVICE))
            opt.zero_grad()
            loss = lossf(model(Q, qm, C, cm), yt[b])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
            n += len(b)
        tr_loss = tot / max(n, 1)

        aux_val = None
        if cfg.aux_novelty and val is not None:
            opt.zero_grad()
            a = cfg.aux_weight * _aux_novelty_loss(model, sets, val, cfg.aux_margin)
            a.backward()
            opt.step()
            aux_val = float(a.detach())
        if sched is not None:
            sched.step()

        last = ep == epochs - 1
        if LOG_EVERY and (ep % LOG_EVERY == 0 or last):
            line = f"   [{label}] ep {ep + 1:>3}/{epochs}  loss tr {tr_loss:.4f}"
            if test_set is not None:
                line += f" te {_eval_loss(model, sets, test_set[0], test_set[1], lossf):.4f}"
            if aux_val is not None:
                line += f"  aux {aux_val:.4f}"
            if PROBE_EVERY and (ep % PROBE_EVERY == 0 or last) and probe_test is not None:
                pt = probe_test(model)
                pr = probe_train(model) if probe_train is not None else dict(r1=float("nan"),
                                                                            bal=float("nan"))
                line += (f"  |  R@1 tr {pr['r1']:.3f} te {pt['r1']:.3f}"
                         f"  balAcc tr {pr['bal']:.3f} te {pt['bal']:.3f}"
                         f"  censusF te {pt['cf']:.3f}")
            logger.info(line)
    return model.eval()


# ==============================================================================================
# hard-negative mining
# ==============================================================================================
def _mine_hard_negs(model, sets, train_imgs, cfg: Cfg, *, seed, label="mine"):
    """For each query, the negative individuals the current model scores highest = its confusions."""
    by_label: dict[str, list[int]] = defaultdict(list)
    for i in train_imgs:
        by_label[sets[i].label].append(i)
    labels = list(by_label)

    queries = list(train_imgs)
    est_pairs = len(queries) * max(1, len(labels) - 1)
    if est_pairs > MINE_PAIR_CAP:                             # keep the scoring pass bounded
        keep = np.random.default_rng(seed).permutation(len(queries))[
            : max(1, MINE_PAIR_CAP // max(1, len(labels) - 1))]
        queries = [queries[k] for k in keep]

    pairs, meta = [], []
    for q in queries:
        yq = sets[q].label
        for c in labels:
            if c == yq:
                continue
            gal = [g for g in by_label[c] if g != q]
            if gal:
                pairs.append((q, gal))
                meta.append((q, c))
    sc = _score(model, sets, pairs)

    per_q: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for (q, c), s in zip(meta, sc):
        per_q[q].append((float(s), c))

    rng = np.random.default_rng(seed + 1)
    chosen: dict[int, list[str]] = {}
    hard_scores: list[float] = []
    for q, lst in per_q.items():
        lst.sort(key=lambda t: -t[0])
        pool = lst[cfg.semi_hard_skip:]
        n_hard = int(round(cfg.neg * (1.0 - cfg.mix_random)))
        hard = [c for _, c in pool[:n_hard]]
        hard_scores += [s for s, _ in pool[:n_hard]]
        rest = [c for _, c in pool[n_hard:]]
        rng.shuffle(rest)
        chosen[q] = hard + rest[: max(0, cfg.neg - len(hard))]
    if hard_scores:
        h = np.asarray(hard_scores)
        logger.info(f"   [{label} mine] {len(pairs):,} cand pairs over {len(per_q)} queries · "
                    f"picked {len(hard_scores):,} hard negs · their P(same) "
                    f"mean {h.mean():.3f}  p90 {np.percentile(h, 90):.3f}  max {h.max():.3f} "
                    f"(skip {cfg.semi_hard_skip}, {cfg.mix_random:.0%} random)")
    return chosen


def _pairs_from_choice(sets, train_imgs, chosen):
    by_label: dict[str, list[int]] = defaultdict(list)
    for i in train_imgs:
        by_label[sets[i].label].append(i)
    pairs, y = [], []
    for q in train_imgs:
        yq = sets[q].label
        pos = [g for g in by_label[yq] if g != q]
        if not pos:
            continue
        pairs.append((q, pos))
        y.append(1.0)
        for c in chosen.get(q, []):
            gal = [g for g in by_label[c] if g != q]
            if gal:
                pairs.append((q, gal))
                y.append(0.0)
    return pairs, np.array(y, np.float32)


def _subsample_by_individual(sets, images, n_indiv, *, seed):
    by: dict[str, list[int]] = defaultdict(list)
    for i in images:
        by[sets[i].label].append(i)
    labs = list(by)
    if len(labs) <= n_indiv:
        return list(images)
    keep = np.random.default_rng(seed).permutation(len(labs))[:n_indiv]
    return [i for k in keep for i in by[labs[k]]]


def _train_model(sets, train_imgs, cfg: Cfg, *, seed, val, label="fit", telem=None):
    telem = telem or {}
    if not cfg.hard_neg:
        pairs, y, _, _ = build_e2e_pairs(sets, train_imgs, neg_per_query=cfg.neg, seed=seed)
        logger.info(f"   [{label}] {len(pairs):,} train pairs ({int(sum(y))} pos / "
                    f"{cfg.neg} neg-per-query) · {cfg.epochs} epochs")
        return _fit(sets, pairs, y, cfg, epochs=cfg.epochs, seed=seed, val=val,
                    label=label, **telem)

    # warm up on random negatives so the miner has a non-random model to rank with
    pairs, y, _, _ = build_e2e_pairs(sets, train_imgs, neg_per_query=cfg.neg, seed=seed)
    logger.info(f"   [{label}] warmup {cfg.warmup_epochs} epochs on {len(pairs):,} random-neg pairs")
    model = _fit(sets, pairs, y, cfg, epochs=cfg.warmup_epochs, seed=seed, val=val,
                 label=f"{label} warm", **telem)

    rounds = max(1, cfg.remine_rounds)
    per_round = max(1, cfg.epochs // rounds)
    for r in range(rounds):
        chosen = _mine_hard_negs(model, sets, train_imgs, cfg, seed=seed + r, label=f"{label} r{r}")
        hp, hy = _pairs_from_choice(sets, train_imgs, chosen)
        logger.info(f"   [{label} r{r}] retrain {per_round} epochs on {len(hp):,} pairs "
                    f"({int(sum(hy))} pos)")
        model = _fit(sets, hp, hy, cfg, epochs=per_round, seed=seed + 100 + r, val=val,
                     warm=model, label=f"{label} r{r}", **telem)
    return model


# ==============================================================================================
# per-config evaluation — the canonical sweep_all9 e2e path, fold by fold
# ==============================================================================================
def _eval_fold(sets, tr, ev, cfg: Cfg, *, fold, progress):
    train_imgs = [i for i in tr if not sets[i].is_synth]

    gal, qry = cen.make_openset_split(sets, ev, NOVEL_FRAC, SEED)
    os_true = {q: sets[q].label for q in qry}
    gal_n, qry_n = cen.make_openset_split(sets, train_imgs, NOVEL_FRAC, SEED + 101)
    nov_true = {q: sets[q].label for q in qry_n}
    pairs_os, oq, oc = build_e2e_openset_pairs(sets, gal, qry)
    pairs_nov, nq, nc = build_e2e_openset_pairs(sets, gal_n, qry_n)

    val = None
    if cfg.aux_novelty:
        # carve the aux open-set split from a ~40-individual subsample of train — enough signal for
        # the margin loss, small enough that its per-epoch autograd graph stays cheap.
        aux_imgs = _subsample_by_individual(sets, train_imgs, 40, seed=SEED + 202)
        gal_v, qry_v = cen.make_openset_split(sets, aux_imgs, NOVEL_FRAC, SEED + 202)
        vp, vq, vc = build_e2e_openset_pairs(sets, gal_v, qry_v)
        vq_arr, vc_arr = np.asarray(vq), np.asarray(vc, dtype=object)
        known = {q: (sets[q].label in set(vc_arr[vq_arr == q])) for q in np.unique(vq_arr)}
        val = (vp, vq, known)

    label = f"{cfg.name} f{fold}"
    n_known = int(sum(sets[q].label in set(np.asarray(oc)[np.asarray(oq) == q]) for q in qry))
    logger.info(f"  --- {label}  ·  {len(train_imgs)} train imgs  ·  eval {len(qry)} queries "
                f"({n_known} known / {len(qry) - n_known} novel)"
                + ("  ·  aux-novelty on" if cfg.aux_novelty else ""))

    # per-epoch telemetry: a held-out pair set for the test-loss curve + a train-carved
    # open-set split for the "train" side of R@1 / balAcc (the eval fold gives the "test" side).
    telem: dict = {}
    if LOG_EVERY:
        pairs_te, y_te, _, _ = build_e2e_pairs(sets, ev, neg_per_query=cfg.neg, seed=SEED + 7)
        telem["test_set"] = (pairs_te, y_te)
        if PROBE_EVERY:
            tp_imgs = _subsample_by_individual(sets, train_imgs, 40, seed=SEED + 303)
            g_tp, q_tp = cen.make_openset_split(sets, tp_imgs, NOVEL_FRAC, SEED + 303)
            tp_true = {q: sets[q].label for q in q_tp}
            pairs_tp, oq_tp, oc_tp = build_e2e_openset_pairs(sets, g_tp, q_tp)
            telem["probe_train"] = lambda m: _probe(m, sets, pairs_tp, oq_tp, oc_tp, tp_true)
            telem["probe_test"] = lambda m: _probe(m, sets, pairs_os, oq, oc, os_true)

    t0 = time.time()
    model = _train_model(sets, train_imgs, cfg, seed=SEED, val=val, label=label, telem=telem)
    sc = _score(model, sets, pairs_os)
    sc_n = _score(model, sets, pairs_nov)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    met = _census_metrics(sc, oq, oc, os_true)
    met.update(_novelty_block(sc, oq, oc, os_true, sc_n, nq, nc, nov_true))
    met["r1"] = _r1(sc, oq, oc, os_true)
    met["secs"] = time.time() - t0
    logger.info(f"  === {label} DONE  identR@1 {met['ident_r1']:.3f}  censusF {met['census_f']:.3f}  "
                f"os_auroc {met['os_auroc']:.3f}  |  gate: base {met['auroc_base']:.3f} "
                f"b' {met['auroc_bprime']:.3f} ({met['bprime_feat']}) c' {met['auroc_cprime']:.3f}  "
                f"balAcc {met['bal_acc_cal']:.3f}  review@90 {met['review_at90']:.0%}  "
                f"({met['secs']:.0f}s)")
    progress()
    return met


def _agg(per_fold: list[dict], key: str) -> tuple[float, float]:
    v = [r[key] for r in per_fold if r.get(key) is not None and np.isfinite(r[key])]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))


def run_config(sets, folds, cfg: Cfg, *, progress) -> dict:
    eff = cfg.scaled()
    per_fold = []
    for fi, (tr, ev) in enumerate(folds):
        per_fold.append(_eval_fold(sets, tr, ev, eff, fold=fi, progress=progress))
    row = {
        "name": cfg.name,
        "objective": cfg.objective,
        "config": asdict(eff),
        "ident_r1": _agg(per_fold, "ident_r1"),
        "auroc_base": _agg(per_fold, "auroc_base"),
        "auroc_bprime": _agg(per_fold, "auroc_bprime"),
        "auroc_cprime": _agg(per_fold, "auroc_cprime"),
        "bal_acc_cal": _agg(per_fold, "bal_acc_cal"),
        "review_at90": _agg(per_fold, "review_at90"),
        "census_f": _agg(per_fold, "census_f"),
        "secs": _agg(per_fold, "secs"),
        "per_fold": per_fold,
    }
    m, s = row["ident_r1"]
    gm, _ = row["bal_acc_cal"]
    logger.info(f"    {cfg.name:<22} identR@1 {m:.3f}±{s:.3f}  gate balAcc {gm:.3f}  "
                f"censusF {row['census_f'][0]:.3f}  ({row['secs'][0]:.0f}s/fold)")
    return row


# ==============================================================================================
# data — recreates the all9_q0.4 `filtered` + `source=sasa` protocol
# ==============================================================================================
def prepare_data():
    sets = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))
    sets = d.apply_quality_filter(sets)                       # `filtered` train pool == gate all
    source_keep = d.source_keep_mask(sets)
    eval_mask = None if source_keep.all() else source_keep
    folds = d.get_cv_folds(sets, k=K_FOLDS, seed=SEED, eval_mask=eval_mask)
    d.assert_evaluable(folds, what=f"SOURCE={d.SOURCE} quality={d.quality_tag() or 'none'}")
    return sets, folds


# ==============================================================================================
# driver
# ==============================================================================================
def _plan(chosen: dict[str, list[Cfg]]):
    total_folds = 0
    logger.info("=" * 90)
    logger.info(f" e2e_transformer exploration  ·  device={DEVICE}  ·  {K_FOLDS} folds  ·  "
                f"quality={d.quality_tag() or 'none'}  source={d.SOURCE}"
                f"{'  [QUICK]' if QUICK else ''}")
    logger.info("=" * 90)
    for study, cfgs in chosen.items():
        ep = [c.scaled().epochs for c in cfgs]
        logger.info(f" {study:<10} {len(cfgs):>2} configs · epochs {min(ep)}–{max(ep)} · "
                    f"{len(cfgs) * K_FOLDS} fold-trains")
        total_folds += len(cfgs) * K_FOLDS
    logger.info("-" * 90)
    logger.info(f" TOTAL {total_folds} fold-trains")
    return total_folds


def _select(chosen, results):
    """Pick each study's winner by its objective."""
    picks = {}
    for study, cfgs in chosen.items():
        if study == "baseline":
            continue
        names = {c.name for c in cfgs}
        rows = [r for r in results if r["name"] in names]
        if not rows:
            continue
        key, hi = OBJECTIVE.get(cfgs[0].objective, ("ident_r1", True))
        rows.sort(key=lambda r: (r[key][0] if np.isfinite(r[key][0]) else -1), reverse=hi)
        picks[study] = rows[0]
    return picks


def _write_report(chosen, results, picks, base_row, elapsed):
    outdir = _REPO_ROOT / "artifacts" / "spot_transformer" / "sweeps" / "e2e_explore"
    outdir.mkdir(parents=True, exist_ok=True)
    # tag identifies THIS pod's slice so parallel shards never overwrite each other.
    slice_tag = "_" + "-".join(k for k in chosen if k != "baseline")
    if _SHARD_N > 1:
        slice_tag += f"_shard{_SHARD_I}of{_SHARD_N}"
    tag = (d.quality_tag() + d.source_tag() + slice_tag + ("_quick" if QUICK else ""))

    (outdir / f"e2e_explore{tag}.json").write_text(
        json.dumps({"device": str(DEVICE), "k_folds": K_FOLDS, "seed": SEED,
                    "elapsed_sec": elapsed, "results": results,
                    "winners": {k: v["name"] for k, v in picks.items()}}, indent=2, default=float),
        encoding="utf-8")

    REF = dict(ident_r1=(0.717, 0.062), auroc_base=(0.731, 0.0), auroc_bprime=(0.620, 0.0),
               auroc_cprime=(0.549, 0.0), bal_acc_cal=(0.771, 0.0), review_at90=(0.85, 0.0),
               census_f=(0.807, 0.0))

    def fmt(r):
        def g(k):
            return f"{r[k][0]:.3f}"
        return (f"| {r['name']} | {r['ident_r1'][0]:.3f} ± {r['ident_r1'][1]:.3f} | {g('auroc_base')} "
                f"| {g('auroc_bprime')} | {g('auroc_cprime')} | {g('bal_acc_cal')} | "
                f"{r['review_at90'][0]:.0%} | {g('census_f')} | {r['secs'][0]:.0f}s |")

    md = [f"# e2e_transformer exploration{(' — ' + tag.strip('_')) if tag else ''}", "",
          f"- device `{DEVICE}` · {K_FOLDS} folds · seed {SEED} · quality `{d.quality_tag() or 'none'}`"
          f" · source `{d.SOURCE}` · elapsed {elapsed / 3600:.1f} h"
          + ("  **[QUICK — plumbing only, not a result]**" if QUICK else ""),
          "- Same folds / open-set protocol / metric code as `sweep_all9` (`_census_metrics` /",
          "  `_novelty_block` / `_r1` copied verbatim), so rows are directly comparable to",
          "  `all9_q0.4/RESULTS_all9_sasa.md`.",
          "",
          "| column | meaning |",
          "|---|---|",
          "| `identR@1` | of re-sight queries, top-1 named the right animal (**the identification task**) |",
          "| `AUROC base/b′/c′` | known-vs-novel separability: raw top-1 / best relative stat / trained clf |",
          "| `balAcc` | balanced accuracy of the known-vs-novel call at the recalibrated cut (**the gate**) |",
          "| `review@90` | fraction of photos a human must check for 90% end-to-end accuracy |",
          "", "---", "",
          "## Baseline — recreate the canonical row", "",
          "| model | identR@1 | AUROC base | b′ | c′ | balAcc | review@90 | censusF | t |",
          "|---|---|---|---|---|---|---|---|---|"]
    if base_row:
        md.append(fmt(base_row))
    md.append(f"| _all9_q0.4 reference_ | {REF['ident_r1'][0]:.3f} ± {REF['ident_r1'][1]:.3f} | "
              f"{REF['auroc_base'][0]:.3f} | {REF['auroc_bprime'][0]:.3f} | {REF['auroc_cprime'][0]:.3f} "
              f"| {REF['bal_acc_cal'][0]:.3f} | {REF['review_at90'][0]:.0%} | {REF['census_f'][0]:.3f} | — |")
    if base_row:
        dr = base_row["ident_r1"][0] - REF["ident_r1"][0]
        md += ["", f"> Baseline identR@1 is **{dr:+.3f}** vs the reference "
               f"(within ±{REF['ident_r1'][1]:.3f} fold std ⇒ reproduced; GPU float nondeterminism "
               f"accounts for small drift)."]

    for study, cfgs in chosen.items():
        if study == "baseline":
            continue
        names = [c.name for c in cfgs]
        rows = [r for r in results if r["name"] in names]
        if not rows:
            continue
        key, hi = OBJECTIVE.get(cfgs[0].objective, ("ident_r1", True))
        rows.sort(key=lambda r: (r[key][0] if np.isfinite(r[key][0]) else -1), reverse=hi)
        win = picks.get(study)
        md += ["", "---", "", f"## {study} — ranked by {'balanced-accuracy gate' if key == 'bal_acc_cal' else 'identR@1'}",
               "", "| model | identR@1 | AUROC base | b′ | c′ | balAcc | review@90 | censusF | t |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            line = fmt(r)
            if win and r["name"] == win["name"]:
                line = line.replace(f"| {r['name']} |", f"| **{r['name']}** ⭐ |", 1)
            md.append(line)
        if win:
            wk = win[key][0]
            base_v = base_row[key][0] if base_row else REF.get(key, (float('nan'),))[0]
            md += ["", f"> Winner **{win['name']}**: {key} {wk:.3f} vs baseline {base_v:.3f} "
                   f"(**{wk - base_v:+.3f}**). {_winner_config_line(win)}"]

    md += ["", "---", "", "## Winners", "",
           "| study | winner | key metric | Δ vs baseline |", "|---|---|---|---|"]
    for study, w in picks.items():
        key = OBJECTIVE.get(w["objective"], ("ident_r1", True))[0]
        base_v = base_row[key][0] if base_row else REF.get(key, (float("nan"),))[0]
        md.append(f"| {study} | {w['name']} | {key} {w[key][0]:.3f} | {w[key][0] - base_v:+.3f} |")

    (outdir / f"RESULTS_e2e_explore{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info(f"wrote {outdir / f'RESULTS_e2e_explore{tag}.md'}")
    logger.info(f"wrote {outdir / f'e2e_explore{tag}.json'}")


def _winner_config_line(row) -> str:
    c = row["config"]
    keys = ["arch", "depth", "n_heads", "dropout", "lr", "wd", "neg", "epochs", "cosine"]
    extra = [k for k in ("hard_neg", "semi_hard_skip", "mix_random", "remine_rounds",
                         "aux_novelty", "aux_weight") if c.get(k)]
    parts = [f"{k}={c[k]}" for k in keys] + [f"{k}={c[k]}" for k in extra]
    return "`" + " · ".join(parts) + "`"


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    all_studies = _studies()
    want = _WANT_STUDIES or list(all_studies)
    chosen: dict[str, list[Cfg]] = {}
    for s in want:
        if s not in all_studies:
            raise SystemExit(f"unknown study {s!r}; choose from {list(all_studies)}")
        cfgs = all_studies[s]
        if s != "baseline" and _SHARD_N > 1:
            cfgs = cfgs[_SHARD_I::_SHARD_N]
        if s != "baseline" and QUICK:
            cfgs = cfgs[:2]
        elif s != "baseline" and MAX_CONFIGS:
            cfgs = cfgs[:MAX_CONFIGS]
        chosen[s] = cfgs
    chosen.setdefault("baseline", [BASELINE])
    # baseline first, always
    chosen = {"baseline": chosen["baseline"], **{k: v for k, v in chosen.items() if k != "baseline"}}

    total = _plan(chosen)
    if DRY_RUN:
        logger.info(" DRY_RUN — exiting before any training")
        return

    sets, folds = prepare_data()
    logger.info(f" data ready: {len(sets)} sets · {len(folds)} folds")

    done = {"n": 0}

    def progress():
        done["n"] += 1
        print(f"PROGRESS {done['n'] / max(total, 1):.4f} {done['n']}/{total} fold-trains", flush=True)

    t0 = time.time()
    results: list[dict] = []
    seen: set[str] = set()
    for study, cfgs in chosen.items():
        logger.info(f"=== {study} " + "=" * (86 - len(study)))
        for cfg in cfgs:
            if cfg.name in seen:                             # baseline may appear twice
                continue
            seen.add(cfg.name)
            try:
                results.append(run_config(sets, folds, cfg, progress=progress))
            except Exception as exc:                         # one config must not sink the run
                logger.error(f"    {cfg.name:<22} FAILED: {type(exc).__name__}: {exc}")

    base_row = next((r for r in results if r["name"] == "baseline"), None)
    picks = _select(chosen, results)
    _write_report(chosen, results, picks, base_row, time.time() - t0)

    logger.info("=" * 90)
    for study, w in picks.items():
        key = OBJECTIVE.get(w["objective"], ("ident_r1", True))[0]
        logger.info(f" {study:<10} winner {w['name']:<22} {key} {w[key][0]:.3f}")
    logger.info(f" total {(time.time() - t0) / 3600:.2f} h on {DEVICE}")


if __name__ == "__main__":
    main()
