"""Set-based learned aggregation (formulation 2).

Instead of hand-summarizing a query->candidate match into a fixed feature vector
(``aggregator.py``), feed the **variable-length set of per-spot match records** into a
permutation-invariant network that learns an *attention weight per match*, pools, and scores.
More expressive than the summary + logistic-regression; the attention directly implements
"down-weight bad matches, up-weight strong / mutual / geometrically-consistent ones."

One record per query spot: [sim1, sim2, sim3, margin, is_mutual, geom_inlier, log_nc].
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist

try:  # run as script (path[0] = this dir) OR imported as a package
    import data as d
    from aggregator import iter_pairs, attach_centroids
except ModuleNotFoundError:
    from pipeline.spot_transformer import data as d
    from pipeline.spot_transformer.aggregator import iter_pairs, attach_centroids

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

REC_NAMES = ["sim1", "sim2", "sim3", "margin", "is_mutual", "geom_inlier", "log_nc"]


def _geom_inlier_mask(Qa, Cb, mutual, iters=40, thr_frac=0.12, seed=0):
    """Per-query-spot RANSAC inlier mask: fit a similarity transform on the mutual matches,
    then flag every query spot whose best-candidate lands within threshold under it."""
    n, m = len(Qa), len(mutual)
    mask = np.zeros(n, bool)
    if m < 3:
        return mask
    Qm, Cm = Qa[mutual], Cb[mutual]
    ref = float(np.median(pdist(Cm)))
    if ref < 1e-9:
        return mask
    thr = thr_frac * ref
    rng = np.random.default_rng(seed)
    best = -1
    for _ in range(iters):
        a, b = rng.choice(m, 2, replace=False)
        dq, dc = Qm[b] - Qm[a], Cm[b] - Cm[a]
        nq = np.hypot(*dq)
        if nq < 1e-9:
            continue
        s = np.hypot(*dc) / nq
        ang = np.arctan2(dc[1], dc[0]) - np.arctan2(dq[1], dq[0])
        R = s * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        pred = Qa @ R.T + (Cm[a] - R @ Qm[a])
        inl = np.hypot(*(pred - Cb).T) < thr
        if inl.sum() > best:
            best, mask = inl.sum(), inl
    return mask


def match_records(q_emb, c_emb, q_xy=None, c_xy=None) -> np.ndarray:
    """Per-query-spot match records, ``(nq, 7)`` -> columns are REC_NAMES."""
    S = q_emb @ c_emb.T                                      # (nq, nc)
    nq, nc = S.shape
    order = np.argsort(-S, axis=1)
    ar = np.arange(nq)
    sim1 = S[ar, order[:, 0]]
    sim2 = S[ar, order[:, 1]] if nc > 1 else np.zeros(nq)
    sim3 = S[ar, order[:, 2]] if nc > 2 else np.zeros(nq)
    qbest = order[:, 0]
    is_mutual = (S.argmax(0)[qbest] == ar).astype(float)
    geom = np.zeros(nq)
    if q_xy is not None:
        # iter_pairs / iter_pairs_openset hand us aggregator.spot_geometry's (N, 3) form
        # (image x, y + body-frame axis_t). _geom_inlier_mask fits a pure image-plane similarity
        # transform and unpacks each row as (x, y) -- a 3rd column makes np.hypot(*dq) raise
        # "return arrays must be of ArrayType". aggregator.match_features uses axis_t itself; this
        # path does not, so keep only the pixel coords.
        q_xy = np.asarray(q_xy)[:, :2]
        Cb = np.asarray(c_xy)[:, :2][qbest]
        geom = _geom_inlier_mask(q_xy, Cb, np.where(is_mutual > 0)[0]).astype(float)
    return np.stack([sim1, sim2, sim3, sim1 - sim2, is_mutual, geom, np.full(nq, np.log1p(nc))], axis=1)


def _jitter(e, std, rng):
    """Gaussian feature-space noise on L2-normalized spot embeddings, re-normalized. Simulates
    descriptor measurement noise -> the aggregator sees perturbed match stats (train-time aug)."""
    if std <= 0:
        return e
    e = e + rng.normal(0.0, std, size=e.shape).astype(e.dtype)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


BAND_NAMES = [f"{s}_{n}" for s in ("mean", "max") for n in REC_NAMES] + ["count"]


def axial_bands(rec, axis_t, n_bands=16):
    """Match records -> a FIXED ``(n_bands, 15)`` grid ordered head->tail along the body axis.

    ``axis_t`` is each query spot's position on the centre line (0=head..1=tail), so band *j*
    covers the same fraction of the animal in every photo regardless of pose, rotation or scale.
    Each band holds the mean and max of its spots' records plus an occupancy count -- the
    summary an ordered convolution can slide over.

    This is what makes a CNN meaningful here: the raw match records are an unordered SET (which
    is why the other archs are permutation-invariant), whereas bands have a real, comparable
    axis, so "a run of strong matches along the midsection" becomes a learnable local pattern.
    Spots with no axis (NaN ``axis_t``, e.g. no usable anatomy) are dropped from the grid.
    """
    F = rec.shape[1]
    out = np.zeros((n_bands, 2 * F + 1), np.float32)
    t = np.asarray(axis_t, float)
    ok = np.isfinite(t)
    if not ok.any():
        return out
    idx = np.clip((t[ok] * n_bands).astype(int), 0, n_bands - 1)
    r = rec[ok]
    for j in range(n_bands):
        m = idx == j
        if not m.any():
            continue
        out[j, :F] = r[m].mean(0)
        out[j, F:2 * F] = r[m].max(0)
        out[j, -1] = m.sum()
    if out[:, -1].max() > 0:                                 # occupancy -> scale-free
        out[:, -1] /= out[:, -1].max()
    return out


def build_record_pairs(sets, images, *, neg_per_query=None, seed=0, use_geom=True, jitter=0.0,
                       n_bands=0):
    """Per-pair match records. ``n_bands>0`` returns axis-ordered band grids instead of the raw
    record set (for the axial CNN); every grid is the same length, so the padding/mask path
    downstream is a no-op and nothing else has to change."""
    recs, y, qids, clab = [], [], [], []
    rng = np.random.default_rng(seed + 1)                    # jitter stream (distinct from neg-sampling)
    for qe, ce, qx, cx, lab, q, c in iter_pairs(sets, images, neg_per_query=neg_per_query,
                                                seed=seed, use_geom=use_geom):
        r = match_records(_jitter(qe, jitter, rng), _jitter(ce, jitter, rng), qx, cx)
        if n_bands:
            at = getattr(sets[q], "axis_t", None)
            if at is None:
                raise ValueError("axial bands need ImageSet.axis_t — call attach_centroids(sets)")
            r = axial_bands(r, at, n_bands)
        recs.append(r)
        y.append(lab); qids.append(q); clab.append(c)
    return recs, np.array(y), np.array(qids), np.array(clab, dtype=object)


def pad_records(recs, N=None):
    N = N or max(len(r) for r in recs)
    P, F = len(recs), recs[0].shape[1]
    X = np.zeros((P, N, F), np.float32)
    mask = np.zeros((P, N), bool)
    for i, r in enumerate(recs):
        n = min(len(r), N)
        X[i, :n] = r[:n]; mask[i, :n] = True
    return X, mask


class _AttnPoolHead(nn.Module):
    """Shared readout: attention-pool + mean-pool + log(set size) -> logit.

    Both encoders below produce per-match features ``e (B, N, h)``; this turns that set into
    one P(same) logit. ``attn_pool`` learns to up-/down-weight matches, ``mean_pool`` is the
    breadth signal, ``logn`` lets the head normalize for candidate-set size (soft-chamfer's flaw)."""

    def __init__(self, h, dropout):
        super().__init__()
        self.attn = nn.Linear(h, 1)                          # learned per-match weight
        self.rho = nn.Sequential(nn.Linear(2 * h + 1, h), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(h, 1))

    def forward(self, e, mask):
        a = self.attn(e).squeeze(-1).masked_fill(~mask, float("-inf"))
        w = torch.softmax(a, dim=1).unsqueeze(-1)            # attention over matches
        attn_pool = (w * e).sum(1)                           # (B, h)
        denom = mask.sum(1, keepdim=True).clamp(min=1)
        mean_pool = (e * mask.unsqueeze(-1)).sum(1) / denom  # (B, h)
        logn = torch.log1p(mask.sum(1, keepdim=True).float())  # nq, the size signal
        return self.rho(torch.cat([attn_pool, mean_pool, logn], dim=1)).squeeze(-1)


class SetAggregator(nn.Module):
    """Deep-Sets: per-match MLP ``phi`` (matches don't see each other) -> attention pool -> P(same)."""

    def __init__(self, in_dim=7, h=64, dropout=0.1, **_):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout))
        self.head = _AttnPoolHead(h, dropout)

    def forward(self, X, mask):
        return self.head(self.phi(X), mask)


class SetTransformer(nn.Module):
    """Self-attention over the match set: each match is re-represented in the CONTEXT of the
    others (multi-head attention) before the same attention-pool readout. More expressive than
    Deep-Sets (matches interact), hence the heavier-regularization / more-data caveat."""

    def __init__(self, in_dim=7, h=64, dropout=0.1, n_heads=4, n_layers=1, **_):
        super().__init__()
        if h % n_heads:                                      # heads must divide width
            n_heads = max(1, h // (h // n_heads) if h // n_heads else 1)
        self.proj = nn.Linear(in_dim, h)
        layer = nn.TransformerEncoderLayer(d_model=h, nhead=n_heads, dim_feedforward=2 * h,
                                           dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = _AttnPoolHead(h, dropout)

    def forward(self, X, mask):
        e = self.proj(X)
        e = self.encoder(e, src_key_padding_mask=~mask)      # True == PAD -> ignored by attention
        e = torch.nan_to_num(e)                              # fully-padded rows -> 0 (guard)
        return self.head(e, mask)


class AxialCNN(nn.Module):
    """Conv1d along the body axis over the banded grid from :func:`axial_bands`.

    Unlike the other two encoders this one is deliberately NOT permutation-invariant -- the
    bands are ordered head->tail, and that order is the signal. Convolution gives translation
    equivariance along the body, so "three consecutive strong bands" is one learned kernel
    wherever it occurs, and stacked layers widen the receptive field to longer stretches.

    Global mean+max pooling over bands makes the output length-independent, then the shared
    attention head scores it exactly like the set models (so the readout is held constant and
    the encoder is the only thing that differs).
    """

    def __init__(self, in_dim=15, h=64, dropout=0.1, n_layers=2, kernel=3, **_):
        super().__init__()
        convs, prev = [], in_dim
        for _ in range(max(1, n_layers)):
            convs += [nn.Conv1d(prev, h, kernel_size=kernel, padding=kernel // 2),
                      nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.conv = nn.Sequential(*convs)
        self.head = _AttnPoolHead(h, dropout)

    def forward(self, X, mask):
        e = self.conv(X.transpose(1, 2)).transpose(1, 2)      # (B, N, h), N = bands
        return self.head(e, mask)


def build_agg_model(arch, in_dim, h, dropout, n_heads=4, n_layers=1, kernel=3):
    """Model factory: ``arch`` in {"deepsets", "transformer", "axialcnn"}."""
    cls = {"deepsets": SetAggregator, "transformer": SetTransformer, "axialcnn": AxialCNN}[arch]
    kw = dict(in_dim=in_dim, h=h, dropout=dropout, n_heads=n_heads, n_layers=n_layers)
    if arch == "axialcnn":
        kw["kernel"] = kernel
    return cls(**kw)


def _best_fbeta(scores, same, beta=0.5):
    """Threshold-free best F_beta of a match/no-match decision over scored pairs.

    Sweep the decision threshold over the scores; at each, precision = TP/(TP+FP),
    recall = TP/all-positives; return the *max* F_beta. beta<1 weights precision more
    (beta=0.5 -> precision counts 2x recall) -- the census priority: a false MATCH collapses
    two animals (deflates the count) and is costlier than a missed re-sight."""
    same = np.asarray(same).astype(bool)
    P = int(same.sum())
    if P == 0 or P == len(same):
        return float("nan")
    o = np.argsort(-np.asarray(scores))                      # high score first = predicted match
    tp = np.cumsum(same[o]); fp = np.cumsum(~same[o])
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / P
    b2 = beta * beta
    f = (1 + b2) * prec * rec / np.maximum(b2 * prec + rec, 1e-12)
    return float(f.max())


def _pairs_from_records(model, scaler, recs, qids, clab, true_by_q):
    """Score precomputed record pairs -> (scores, same) for cheap per-epoch monitoring."""
    scores = _score_set(model, scaler, recs)
    same = np.array([clab[i] == true_by_q[qids[i]] for i in range(len(qids))], float)
    return scores, same


def train_set(recs, y, *, arch="deepsets", h=64, epochs=250, lr=0.02, batch=256, seed=0,
              weight_decay=1e-4, l1=0.0, dropout=0.1, n_heads=4, n_layers=1, kernel=3, beta=0.5,
              patience=0, log_every=0, monitors=None, verbose=True):
    """Train a set aggregator over match records. Returns ``(model, (mu, sd), history)``.

    Regularization knobs (all guard against overfitting the ~few-hundred positive pairs):
    ``weight_decay`` = L2, ``l1`` = L1 penalty (sparsifies), ``dropout``. ``patience`` > 0 turns
    on early stopping on ``monitors["val"]`` pairwise F_beta (val is carved from TRAIN, never the
    test fold) and restores the best weights. Feature jitter is applied upstream at record-build.

    ``history`` records, per epoch: ``loss`` + for each monitor ``{name}_r1`` and ``{name}_f``
    (pairwise best-F_beta) -- the train/test learning curves. ``log_every`` > 0 also prints them.
    """
    torch.manual_seed(seed)
    X, mask = pad_records(recs)
    flat, real = X.reshape(-1, X.shape[-1]), mask.reshape(-1)
    mu, sd = flat[real].mean(0), flat[real].std(0) + 1e-8
    Xs = ((X - mu) / sd).astype(np.float32); Xs[~mask] = 0.0
    Xt, Mt, yt = torch.tensor(Xs), torch.tensor(mask), torch.tensor(y, dtype=torch.float32)
    model = build_agg_model(arch, X.shape[-1], h, dropout, n_heads=n_heads, n_layers=n_layers,
                            kernel=kernel)
    npos = max(int(y.sum()), 1)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y) - npos) / npos], dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    idx = np.arange(len(y)); rng = np.random.default_rng(seed)
    monitors = monitors or {}
    history, best = [], {"val_f": -1.0, "state": None, "epoch": -1, "wait": 0}
    for ep in range(epochs):
        rng.shuffle(idx)
        ep_loss = []
        for s in range(0, len(idx), batch):
            bi = idx[s:s + batch]
            opt.zero_grad()
            loss = lossf(model(Xt[bi], Mt[bi]), yt[bi])
            if l1 > 0:
                loss = loss + l1 * sum(p.abs().sum() for p in model.parameters())
            loss.backward(); opt.step()
            ep_loss.append(loss.item())

        model.eval()
        rec = {"epoch": ep, "loss": float(np.mean(ep_loss))}
        for nm, (mr, mq, mc, mt) in monitors.items():
            sc, sm = _pairs_from_records(model, (mu, sd), mr, mq, mc, mt)
            rec[f"{nm}_r1"] = _r1_from_records(model, (mu, sd), mr, mq, mc, mt)
            rec[f"{nm}_f"] = _best_fbeta(sc, sm, beta)
        model.train()
        history.append(rec)

        if patience and "val" in monitors:                  # early stop on val F_beta (no test leak)
            vf = rec.get("val_f", float("nan"))
            if np.isfinite(vf) and vf > best["val_f"] + 1e-4:
                best.update(val_f=vf, state={k: v.clone() for k, v in model.state_dict().items()},
                            epoch=ep, wait=0)
            else:
                best["wait"] += 1
        if log_every and (ep % log_every == 0 or ep == epochs - 1) and verbose:
            msg = f"      ep {ep:>3} | loss {rec['loss']:.4f}"
            for nm in monitors:
                msg += f" | {nm} R@1 {rec[f'{nm}_r1']:.3f} F{beta:g} {rec[f'{nm}_f']:.3f}"
            logger.info(msg)
        if patience and best["wait"] >= patience:
            if verbose:
                logger.info(f"      early stop @ ep {ep} (best val F @ ep {best['epoch']})")
            break
    if best["state"] is not None:                            # restore best-val weights
        model.load_state_dict(best["state"])
    return model, (mu, sd), history


def _score_set(model, scaler, recs):
    mu, sd = scaler
    X, mask = pad_records(recs)
    Xs = ((X - mu) / sd).astype(np.float32); Xs[~mask] = 0.0
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(Xs), torch.tensor(mask))).numpy()


def _r1_from_records(model, scaler, recs, qids, clab, true_by_q):
    """Voting R@1 over precomputed record pairs (for cheap per-epoch train/test monitoring)."""
    scores = _score_set(model, scaler, recs)
    hits = n = 0
    for q in np.unique(qids):
        m = qids == q
        hits += int(clab[m][np.argmax(scores[m])] == true_by_q[q]); n += 1
    return hits / n if n else float("nan")


def rank_eval_set(model, scaler, sets, eval_images, ks=(1, 5, 10), use_geom=True):
    recs, y, qids, clab = build_record_pairs(sets, eval_images, neg_per_query=None, use_geom=use_geom)
    scores = _score_set(model, scaler, recs)
    hits = {k: 0 for k in ks}; n = 0; margins = []; correct = []
    for q in np.unique(qids):
        m = qids == q
        ss = np.sort(scores[m])[::-1]
        ranked = clab[m][np.argsort(-scores[m])]
        rank = int(np.where(ranked == sets[q].label)[0][0]) + 1
        for k in ks:
            hits[k] += int(rank <= k)
        margins.append(float(ss[0] - ss[1]) if len(ss) > 1 else float(ss[0]))
        correct.append(int(rank == 1)); n += 1
    out = {f"recall@{k}": hits[k] / n for k in ks}
    return out, dict(margins=np.array(margins), correct=np.array(correct))


def score_pairs_set(model, scaler, recs, qids, clab):
    """Score prebuilt (query, candidate) record pairs -> per-pair prob. The substrate the census
    metrics threshold: pass ``model`` OR a list of models (ensemble -> mean prob)."""
    if isinstance(model, (list, tuple)):                     # multi-seed ensemble: average probs
        return np.mean([_score_set(m, sc, recs) for m, sc in zip(model, scaler)], axis=0)
    return _score_set(model, scaler, recs)


def per_query_top1(scores, qids, clab, true_by_q):
    """Collapse pair scores to a per-query decision row: top-1 candidate + its score, the
    runner-up score (margin), whether top-1 is the true individual, and whether the query's
    true individual is present in the candidate set at all (novel = absent -> must be rejected).

    Returns dict of arrays keyed by query id order: ``q, top1_label, top1, top2, margin,
    top1_correct, is_known``."""
    rows = {k: [] for k in ["q", "top1_label", "top1", "top2", "margin", "top1_correct", "is_known"]}
    for q in np.unique(qids):
        m = qids == q
        sc, cl = scores[m], clab[m]
        o = np.argsort(-sc)
        s1 = float(sc[o[0]]); s2 = float(sc[o[1]]) if len(o) > 1 else 0.0
        true = true_by_q.get(q)
        rows["q"].append(q); rows["top1_label"].append(cl[o[0]])
        rows["top1"].append(s1); rows["top2"].append(s2); rows["margin"].append(s1 - s2)
        rows["top1_correct"].append(int(cl[o[0]] == true))
        rows["is_known"].append(int(true in set(cl)))
    return {k: np.array(v, dtype=object if k in ("q", "top1_label") else float) for k, v in rows.items()}


def crossval_set(sets, *, k=5, seed=0, h=64, neg_per_query=30, use_geom=True, use_synth=False,
                 weight_decay=1e-4, l1=0.0, dropout=0.1, epochs=250, log_every=0):
    try:
        from aggregator import rank_eval                    # raw baseline via soft-chamfer feature
    except ModuleNotFoundError:
        from pipeline.spot_transformer.aggregator import rank_eval
    folds = d.get_cv_folds(sets, k=k, seed=seed)
    raws, learns, r5, mc = [], [], [], []
    logger.info(f" cfg: h={h} neg/q={neg_per_query} synth={use_synth} L2={weight_decay} L1={l1} "
          f"drop={dropout} epochs={epochs}")
    logger.info(f" {'fold':>4} | {'raw R@1':>7} | {'set R@1':>7} | {'R@5':>5} | {'dR@1':>6} | {'pairs':>6}")
    logger.info(" " + "-" * 54)
    for fi, (tr, ev) in enumerate(folds):
        train_imgs = tr if use_synth else [i for i in tr if not sets[i].is_synth]
        recs, y, _, _ = build_record_pairs(sets, train_imgs, neg_per_query=neg_per_query, seed=seed, use_geom=use_geom)

        monitors = None
        if log_every:                                        # per-epoch train/test R@1 monitoring
            real_train = [i for i in tr if not sets[i].is_synth]
            rbl: dict[str, list[int]] = {}
            for i in real_train:
                rbl.setdefault(sets[i].label, []).append(i)
            multi = [l for l, ii in rbl.items() if len(ii) >= 2]
            n_ev = len({sets[i].label for i in ev})
            chosen = np.random.default_rng(seed).choice(multi, size=min(n_ev, len(multi)), replace=False)
            td = [i for l in chosen for i in rbl[l]]          # gallery-matched train subset (like eval)
            r_td, _, q_td, c_td = build_record_pairs(sets, td, neg_per_query=None, seed=seed, use_geom=use_geom)
            r_ev, _, q_ev, c_ev = build_record_pairs(sets, ev, neg_per_query=None, seed=seed, use_geom=use_geom)
            monitors = {"train": (r_td, q_td, c_td, {q: sets[q].label for q in np.unique(q_td)}),
                        "test":  (r_ev, q_ev, c_ev, {q: sets[q].label for q in np.unique(q_ev)})}
            logger.info(f"  [fold {fi}] {len(y)} train pairs -- monitoring train/test R@1 every {log_every} epochs")

        model, scaler, _hist = train_set(recs, y, h=h, seed=seed, weight_decay=weight_decay,
                                          l1=l1, dropout=dropout, epochs=epochs, log_every=log_every, monitors=monitors)
        raw, _ = rank_eval(lambda X: X[:, 0], sets, ev, use_geom=use_geom)
        learned, det = rank_eval_set(model, scaler, sets, ev, use_geom=use_geom)
        raws.append(raw["recall@1"]); learns.append(learned["recall@1"]); r5.append(learned["recall@5"])
        mc += list(zip(det["margins"], det["correct"]))
        logger.info(f" {fi:>4} | {raw['recall@1']:>7.3f} | {learned['recall@1']:>7.3f} | "
              f"{learned['recall@5']:>5.3f} | {learned['recall@1']-raw['recall@1']:>+6.3f} | "
              f"{len(y):>6}")
    logger.info(" " + "-" * 54)
    logger.info(f" MEAN | {np.mean(raws):>7.3f} | {np.mean(learns):>7.3f} | {np.mean(r5):>5.3f} | "
          f"{np.mean(learns)-np.mean(raws):>+6.3f}")
    logger.info(f"      raw {np.mean(raws):.3f}+-{np.std(raws):.3f}   set {np.mean(learns):.3f}+-{np.std(learns):.3f}")
    mc = np.array(mc); o = np.argsort(mc[:, 0]); t = len(o) // 3
    logger.info(" confidence (top1-top2 margin -> accuracy):")
    for lab, sl in [("low  ", o[:t]), ("mid  ", o[t:2*t]), ("high ", o[2*t:])]:
        logger.info(f"   {lab}: {mc[sl, 1].mean():.3f}  (n={len(sl)})")
    return raws, learns


if __name__ == "__main__":
    sets = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))
    logger.info("=" * 54)
    logger.info(" 5-FOLD CV: set-attention aggregator (heavy reg + large training)")
    logger.info("=" * 54)
    # Heavy L2 + large training (synthetic images -> many more positive pairs).
    # To try L1 instead: set weight_decay=0.0, l1=1e-4.
    # To scale further: raise neg_per_query / epochs.
    crossval_set(sets, k=5, seed=0, h=32, neg_per_query=60, use_synth=True,
                 weight_decay=1e-2, l1=0.0, dropout=0.3, epochs=200, log_every=20)
