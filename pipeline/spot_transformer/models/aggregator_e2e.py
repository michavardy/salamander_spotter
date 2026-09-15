"""End-to-end: learn the spot embedding AND the voting rule together (formulation 3).

``aggregator.py`` and ``aggregator_set.py`` both hold the 62-dim spot embedding FIXED and learn
only how to aggregate matches. This learns both -- the encoder that maps a raw spot to the space
matches are computed in, trained by backprop **through a differentiable voting rule** against the
pairwise same/different label.

Why this is not the experiment ``results.md`` already killed. That verdict ("do not pursue learned
spot metric-learning") was about training the encoder with **SupCon-by-individual** and then
voting at inference: SupCon pulls one animal's *different* spots together, destroying exactly the
per-spot correspondence voting depends on -- the objective fought the inference rule. Here the
objective IS the inference rule, so that failure mode cannot occur. It is a genuinely untested
formulation, not a re-run.

The voting rule is a differentiable restatement of the hand-engineered features in
``aggregator.FEATURE_NAMES``: soft-chamfer via a temperature-softmax instead of a hard ``max``,
and soft threshold-counts via sigmoids instead of step functions, so gradients reach the encoder.

Three encoders, matching the three requested variants:

* ``mlp``        -- deep fully-connected, each spot encoded independently (context-free).
* ``transformer``-- deep self-attention over the image's spots, so a spot is encoded in the
                    CONTEXT of the animal's other spots.
* ``frozen``     -- the 62-dim embedding is used as-is and ONLY the voting head trains: the
                    "head on top of a pretrained network" ablation, isolating how much the
                    encoder fine-tuning actually buys.

SCOPE NOTE on ``frozen``: the "pretrained network" here is the existing hand-engineered 62-dim
representation (EFD shape + body-intrinsic position), which is computed once and reused -- not an
ImageNet CNN. A true pretrained-CNN encoder would run ResNet18 over per-spot image crops
(``spot_embedding/train/crops.py`` already builds them) and is a heavier, cross-track build; see
the note in the sweep driver.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:  # run as script (path[0] = this dir) OR imported as a package
    import data as d
except ModuleNotFoundError:
    from pipeline.spot_transformer import data as d

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)


# ============================================================ encoders
class SpotEncoder(nn.Module):
    """Raw spot features ``(B, N, in_dim)`` -> learned embedding ``(B, N, out_dim)``.

    ``frozen`` returns the input untouched (no parameters), so the voting head alone trains.
    """

    def __init__(self, in_dim=62, out_dim=64, arch="mlp", depth=3, dropout=0.1, n_heads=4,
                 residual=True):
        super().__init__()
        self.arch = arch
        if arch == "frozen":
            self.out_dim = in_dim
            self.net = None
            self.residual = False
            return
        # Residual only makes sense when the encoder preserves width. It matters a lot here: a
        # randomly-initialized encoder SCRAMBLES the hand-engineered embedding, and the first
        # smoke run showed exactly that (e2e_mlp 0.005 vs 0.332 for the same head on a frozen
        # encoder). Starting at ~identity means training begins where `frozen` already is and
        # learns a refinement, instead of having to rediscover the representation from noise.
        self.residual = bool(residual) and out_dim == in_dim
        self.out_dim = out_dim
        if arch == "mlp":
            blocks, prev = [], in_dim
            for _ in range(max(1, depth)):
                blocks += [nn.Linear(prev, out_dim), nn.GELU(), nn.Dropout(dropout)]
                prev = out_dim
            self.net = nn.Sequential(*blocks)
        elif arch == "transformer":
            self.proj = nn.Linear(in_dim, out_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=out_dim, nhead=n_heads, dim_feedforward=2 * out_dim,
                dropout=dropout, activation="gelu", batch_first=True)
            self.net = nn.TransformerEncoder(layer, num_layers=max(1, depth))
        else:
            raise ValueError(f"unknown encoder arch {arch!r}")

    def forward(self, X, mask):
        if self.arch == "frozen":
            e = X
        elif self.arch == "mlp":
            e = self.net(X)
            if self.residual:
                e = X + e
        else:
            h = self.proj(X)
            e = torch.nan_to_num(self.net(h, src_key_padding_mask=~mask))
            if self.residual:
                e = X + e
        e = e * mask.unsqueeze(-1)                            # zero the pads before normalizing
        return e / (e.norm(dim=-1, keepdim=True) + 1e-12)


# ============================================================ differentiable voting
VOTE_THRESHOLDS = (0.9, 0.8, 0.7, 0.6)


class SoftVote(nn.Module):
    """Differentiable soft-chamfer voting -> P(same individual).

    For each query spot we need "similarity to its best-matching candidate spot". A hard ``max``
    passes gradient to one element only; a temperature softmax spreads it over the near-ties,
    which is what lets the encoder learn *why* a match was close. Likewise the
    ``fraction of spots matching above t`` features -- the breadth signal the logreg champion
    leans on hardest -- become sigmoids so they are differentiable.
    """

    def __init__(self, tau=0.05, sharp=20.0, hidden=32, dropout=0.1):
        super().__init__()
        self.tau, self.sharp = tau, sharp
        n_feat = 4 + 2 * len(VOTE_THRESHOLDS)
        self.head = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, qe, qm, ce, cm):
        S = torch.bmm(qe, ce.transpose(1, 2))                 # (B, Nq, Nc) cosine
        S = S.masked_fill(~cm.unsqueeze(1), float("-inf"))    # ignore padded candidate spots
        w = torch.softmax(S / self.tau, dim=2)
        best = (w * S.nan_to_num(0.0)).sum(2)                 # (B, Nq) soft max-similarity
        best = best * qm                                      # zero the padded query spots

        nq = qm.sum(1).clamp(min=1)
        nc = cm.sum(1).clamp(min=1)
        feats = [best.sum(1), best.sum(1) / nq,
                 torch.log1p(nq), torch.log1p(nc)]
        for t in VOTE_THRESHOLDS:                             # soft "fraction/count above t"
            soft = torch.sigmoid((best - t) * self.sharp) * qm
            feats += [soft.sum(1) / nq, soft.sum(1)]
        return self.head(torch.stack(feats, dim=1)).squeeze(-1)


class E2EVoter(nn.Module):
    """Encoder + differentiable voting, trained jointly on the pairwise same/different label."""

    def __init__(self, in_dim=62, out_dim=64, arch="mlp", depth=3, dropout=0.1,
                 n_heads=4, tau=0.05, sharp=20.0, hidden=32, residual=True):
        super().__init__()
        self.encoder = SpotEncoder(in_dim, out_dim, arch, depth, dropout, n_heads,
                                   residual=residual)
        self.vote = SoftVote(tau=tau, sharp=sharp, hidden=hidden, dropout=dropout)

    def forward(self, Q, qm, C, cm):
        return self.vote(self.encoder(Q, qm), qm, self.encoder(C, cm), cm)


# ============================================================ pair plumbing
def build_e2e_pairs(sets, images, *, gallery_images=None, neg_per_query=None, seed=0):
    """(query image, candidate individual) index pairs -- gathered lazily, never materialized.

    Returns ``(pairs, y, qids, clab)`` where each pair is ``(q_index, [gallery indices])``.
    Storing indices rather than stacked spot tensors keeps memory flat: the same gallery image
    appears in thousands of pairs and is gathered on demand instead of copied.

    ``gallery_images`` adds DISTRACTOR individuals to the candidate list without making them
    queries -- the e2e mirror of ``aggregator.iter_pairs``, so both families can be scored
    against a population-sized gallery instead of just the eval fold.
    """
    rng = np.random.default_rng(seed)
    pool = list(dict.fromkeys(list(images) + list(gallery_images or [])))
    by_label: dict[str, list[int]] = {}
    for i in pool:
        by_label.setdefault(sets[i].label, []).append(i)
    all_labels = list(by_label)

    pairs, y, qids, clab = [], [], [], []
    for q in images:
        yq = sets[q].label
        if not any(g != q for g in by_label[yq]):             # no gallery positive -> unusable
            continue
        cand = [c for c in all_labels if any(g != q for g in by_label[c])]
        if neg_per_query is not None:
            negs = [c for c in cand if c != yq]
            rng.shuffle(negs)
            cand = [yq] + negs[:neg_per_query]
        for c in cand:
            gal = [g for g in by_label[c] if g != q]
            pairs.append((q, gal))
            y.append(int(c == yq)); qids.append(q); clab.append(c)
    return pairs, np.array(y, float), np.array(qids), np.array(clab, dtype=object)


def build_e2e_openset_pairs(sets, gallery_imgs, query_imgs):
    """Index pairs under the CENSUS protocol — candidates are GALLERY individuals only.

    This mirrors ``census.iter_pairs_openset`` exactly and must not be confused with calling
    :func:`build_e2e_pairs` on the query list: that builds leave-one-out pairs *within* the
    queries, so two photos of the same NOVEL animal can match each other and the animal stops
    being novel. Using it for the census eval scores the e2e models on an easier, different
    problem than every other family.
    """
    by_label: dict[str, list[int]] = {}
    for i in gallery_imgs:
        by_label.setdefault(sets[i].label, []).append(i)
    pairs, qids, clab = [], [], []
    for q in query_imgs:
        for c, gal_all in by_label.items():
            gal = [g for g in gal_all if g != q]
            if not gal:
                continue
            pairs.append((q, gal)); qids.append(q); clab.append(c)
    return pairs, np.array(qids), np.array(clab, dtype=object)


def _collate(sets, batch_pairs, max_spots=120, feature_attr="spots"):
    """Gather a batch of index pairs into padded ``(Q, qm, C, cm)`` tensors.

    ``feature_attr`` selects which per-spot representation to vote over: ``spots`` (the 62-dim
    hand-engineered embedding) or ``cnn_feat`` (frozen pretrained-CNN features).
    """
    def _f(i):
        return getattr(sets[i], feature_attr)

    qs = [_f(q)[:max_spots] for q, _ in batch_pairs]
    cs = [np.concatenate([_f(g) for g in gal])[:max_spots] for _, gal in batch_pairs]
    B, F = len(qs), qs[0].shape[1]
    Nq, Nc = max(len(a) for a in qs), max(len(a) for a in cs)
    Q = np.zeros((B, Nq, F), np.float32); qm = np.zeros((B, Nq), bool)
    C = np.zeros((B, Nc, F), np.float32); cm = np.zeros((B, Nc), bool)
    for i, (a, b) in enumerate(zip(qs, cs)):
        Q[i, :len(a)] = a; qm[i, :len(a)] = True
        C[i, :len(b)] = b; cm[i, :len(b)] = True
    return (torch.tensor(Q), torch.tensor(qm), torch.tensor(C), torch.tensor(cm))


# ============================================================ train / score
def train_e2e(sets, pairs, y, *, arch="mlp", out_dim=None, depth=3, dropout=0.1, n_heads=4,
              epochs=30, lr=1e-3, batch=64, weight_decay=1e-4, seed=0, residual=True,
              feature_attr="spots", verbose=True):
    """Train an :class:`E2EVoter`. Returns the model.

    ``out_dim`` defaults to the input width, which is what keeps the residual path (and hence
    the identity start) available -- pass an explicit width only to disable it deliberately.
    ``feature_attr`` picks the per-spot representation (``spots`` | ``cnn_feat``).
    """
    torch.manual_seed(seed)
    in_dim = getattr(sets[pairs[0][0]], feature_attr).shape[1]
    model = E2EVoter(in_dim=in_dim, out_dim=out_dim or in_dim, arch=arch, depth=depth,
                     dropout=dropout, n_heads=n_heads, residual=residual)
    npos = max(int(y.sum()), 1)
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(len(y) - npos) / npos], dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    idx = np.arange(len(pairs)); rng = np.random.default_rng(seed)
    yt = torch.tensor(y, dtype=torch.float32)

    for ep in range(epochs):
        rng.shuffle(idx)
        model.train(); tot = 0.0
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            Q, qm, C, cm = _collate(sets, [pairs[i] for i in b], feature_attr=feature_attr)
            opt.zero_grad()
            loss = lossf(model(Q, qm, C, cm), yt[b])
            loss.backward(); opt.step()
            tot += loss.detach().item() * len(b)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            logger.info(f"   e2e[{arch}] epoch {ep:3d}  loss {tot / len(idx):.4f}")
    return model


@torch.no_grad()
def score_e2e(model, sets, pairs, batch=128, feature_attr="spots"):
    """P(same individual) for each index pair."""
    model.eval()
    out = []
    for s in range(0, len(pairs), batch):
        Q, qm, C, cm = _collate(sets, pairs[s:s + batch], feature_attr=feature_attr)
        out.append(torch.sigmoid(model(Q, qm, C, cm)).numpy())
    return np.concatenate(out) if out else np.zeros(0)
