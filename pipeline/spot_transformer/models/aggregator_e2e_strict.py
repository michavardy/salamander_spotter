"""End-to-end STRICT e2e_transformer: a transformer over PRE-EMBEDDED spots that learns to match
under the review's rules — prefer special spots, default to no match, find the few good matches,
and penalize (a) unmatched special spots, (b) too few matches, (c) matches that are far apart.

Input is the fixed 62-dim per-spot embedding (EFD shape + body-intrinsic position); the transformer
re-encodes each spot IN CONTEXT of its image's other spots. The only training signal is the
matching label (same individual vs not). Concretely the score is a differentiable
``coverage x support`` vote:

* **prefer special spots** — the encoder emits a per-spot gate ``w_i in (0,1)`` and every term is
  weighted by it. The gate is supervised by the human interesting-spot clicks (auxiliary loss), so
  "special" = what the human tagged (odd-shaped + big), not whatever the matcher stumbles into.
* **penalize far-apart matches** — appearance cosine is multiplied by a Gaussian **position gate**
  in body-frame ``(axis_t, axis_offset/length)`` coords, so a spot only counts as matched if a
  candidate spot is both similar AND in the same body location.
* **penalize unmatched special spots** — ``unexplained_q`` = the gate-weighted mass of query spots
  whose best (position-gated) match is weak; a striking spot that matches nothing drags the score
  down and flags novelty.
* **find the few good matches / penalize too few** — ``support = 1 - exp(-n_good / tau)`` collapses
  the score when only 1-3 spots corroborate.
* **default to no match** — enforced at the precision-first census threshold; training uses
  class-balanced BCE so the score stays well-ordered (``pos_weight=1`` collapses the model to a
  constant "never match" on 8:1-negative data, so the prior belongs at the decision, not the loss).

Reuses ``aggregator_e2e``'s ``SpotEncoder`` and pair plumbing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval"):
    p = str(_ST / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

from aggregator_e2e import SpotEncoder                       # noqa: E402
import distinctiveness as dist                               # noqa: E402

VOTE_T = (0.9, 0.8, 0.7)
TAU_LOW = 0.6


# ============================================================ per-spot side inputs
def attach_interesting(sets, interesting=None):
    """``ImageSet.interesting`` (N,): 1 tagged special, 0 not, NaN if the image was never reviewed
    (so the gate loss ignores it). Keyed by DB ``spot_id``."""
    interesting = interesting or dist.load_interesting()
    for s in sets:
        if s.sid in interesting:
            tag = interesting[s.sid]
            s.interesting = np.array([1.0 if int(i) in tag else 0.0 for i in s.spot_ids], float)
        else:
            s.interesting = np.full(len(s.spot_ids), np.nan)
    return sets


def attach_positions(sets, pos_lookup):
    """``ImageSet.pos`` (N,2): body-frame ``(axis_t, axis_offset/length)`` per spot; NaN -> (0.5, 0)
    so a spot with no fitted axis neither matches nor blocks on position."""
    for s in sets:
        xy = np.array([pos_lookup.get((s.sid, int(i)), (np.nan, np.nan)) for i in s.spot_ids],
                      float)
        xy[~np.isfinite(xy[:, 0]), 0] = 0.5
        xy[~np.isfinite(xy[:, 1]), 1] = 0.0
        s.pos = xy.astype(np.float32)
    return sets


# ============================================================ differentiable strict vote
class StrictSoftVote(nn.Module):
    """Distinctiveness-gated, position-gated, differentiable coverage/support vote -> P(same)."""

    def __init__(self, tau=0.05, sharp=20.0, hidden=32, dropout=0.1, sigma_pos=0.12):
        super().__init__()
        self.tau, self.sharp, self.sigma_pos = tau, sharp, sigma_pos
        n_feat = 3 + len(VOTE_T) + 4   # wsc_mean,max,support | cov@t | unexpl_q,wcov_c,log_nq,log_nc
        self.head = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, qe, qm, ce, cm, wq, wc, xyq=None, xyc=None):
        S = torch.bmm(qe, ce.transpose(1, 2))                 # (B, Nq, Nc) cosine
        if xyq is not None:                                   # position gate: down-weight far pairs
            d2 = torch.cdist(xyq, xyc) ** 2                   # (B, Nq, Nc) body-frame sq-distance
            S = S * torch.exp(-d2 / (2.0 * self.sigma_pos ** 2))
        Sq = S.masked_fill(~cm.unsqueeze(1), float("-inf"))
        best = (torch.softmax(Sq / self.tau, dim=2) * Sq.nan_to_num(0.0)).sum(2) * qm   # (B, Nq)
        Sc = S.transpose(1, 2).masked_fill(~qm.unsqueeze(1), float("-inf"))             # (B, Nc, Nq)
        best_c = (torch.softmax(Sc / self.tau, dim=2) * Sc.nan_to_num(0.0)).sum(2) * cm  # (B, Nc)

        wq = wq * qm; wc = wc * cm
        Wq = wq.sum(1).clamp(min=1e-6); Wc = wc.sum(1).clamp(min=1e-6)
        relu = torch.relu(best)
        wsc_mean = (wq * relu).sum(1) / Wq
        max_sim = best.max(1).values
        n_good = (torch.sigmoid((best - 0.75) * self.sharp) * qm).sum(1)
        support = 1.0 - torch.exp(-n_good / 2.5)
        cov = [(wq * torch.sigmoid((best - t) * self.sharp)).sum(1) / Wq for t in VOTE_T]
        unexpl_q = (wq * torch.sigmoid((TAU_LOW - best) * self.sharp)).sum(1) / Wq
        wcov_c = (wc * torch.sigmoid((best_c - 0.8) * self.sharp)).sum(1) / Wc
        nq = qm.sum(1).clamp(min=1); nc = cm.sum(1).clamp(min=1)
        feats = torch.stack([wsc_mean, max_sim, support, *cov, unexpl_q, wcov_c,
                             torch.log1p(nq), torch.log1p(nc)], dim=1)
        return self.head(feats).squeeze(-1)


class StrictE2E(nn.Module):
    def __init__(self, in_dim=62, out_dim=None, arch="transformer", depth=3, dropout=0.1,
                 n_heads=2, tau=0.05, hidden=32, residual=True, sigma_pos=0.12):
        super().__init__()
        self.encoder = SpotEncoder(in_dim, out_dim or in_dim, arch, depth, dropout, n_heads,
                                   residual=residual)
        self.gate = nn.Linear(self.encoder.out_dim, 1)
        self.vote = StrictSoftVote(tau=tau, hidden=hidden, dropout=dropout, sigma_pos=sigma_pos)

    def _encode(self, X, m):
        e = self.encoder(X, m)
        return e, self.gate(e).squeeze(-1)

    def forward(self, Q, qm, C, cm, xyq=None, xyc=None, return_gate=False):
        eq, gq = self._encode(Q, qm)
        ec, gc = self._encode(C, cm)
        logit = self.vote(eq, qm, ec, cm, torch.sigmoid(gq), torch.sigmoid(gc), xyq, xyc)
        return (logit, gq) if return_gate else logit


# ============================================================ collate
def _collate(sets, batch_pairs, max_spots=120):
    def q_arrays(q):
        s = sets[q]
        return s.spots[:max_spots], s.interesting[:max_spots], s.pos[:max_spots]

    def c_arrays(gal):
        sp = np.concatenate([sets[g].spots for g in gal])[:max_spots]
        po = np.concatenate([sets[g].pos for g in gal])[:max_spots]
        return sp, po

    qs, qi, qp = zip(*(q_arrays(q) for q, _ in batch_pairs))
    cs, cp = zip(*(c_arrays(gal) for _, gal in batch_pairs))
    B, F = len(qs), qs[0].shape[1]
    Nq, Nc = max(len(a) for a in qs), max(len(a) for a in cs)
    Q = np.zeros((B, Nq, F), np.float32); qm = np.zeros((B, Nq), bool)
    C = np.zeros((B, Nc, F), np.float32); cm = np.zeros((B, Nc), bool)
    Yi = np.full((B, Nq), np.nan, np.float32)
    XYq = np.zeros((B, Nq, 2), np.float32); XYc = np.zeros((B, Nc, 2), np.float32)
    for i in range(B):
        a, b = qs[i], cs[i]
        Q[i, :len(a)] = a; qm[i, :len(a)] = True; Yi[i, :len(qi[i])] = qi[i]
        XYq[i, :len(qp[i])] = qp[i]
        C[i, :len(b)] = b; cm[i, :len(b)] = True; XYc[i, :len(cp[i])] = cp[i]
    return (torch.tensor(Q), torch.tensor(qm), torch.tensor(C), torch.tensor(cm),
            torch.tensor(Yi), torch.tensor(XYq), torch.tensor(XYc))


# ============================================================ train / score
def _batch_loss(model, sets, pairs, y_slice, match_loss, gate_loss, gate_lambda):
    Q, qm, C, cm, Yi, XYq, XYc = _collate(sets, pairs)
    logit, gq = model(Q, qm, C, cm, XYq, XYc, return_gate=True)
    loss = match_loss(logit, y_slice)
    lab = torch.isfinite(Yi) & qm
    if gate_lambda > 0 and lab.any():
        loss = loss + gate_lambda * gate_loss(gq[lab], Yi[lab])
    return loss


def train_strict_e2e(sets, pairs, y, *, arch="transformer", depth=3, dropout=0.1, n_heads=2,
                     epochs=30, lr=1e-3, batch=64, weight_decay=1e-4, seed=0, residual=True,
                     pos_weight=None, gate_lambda=0.5, sigma_pos=0.12, cosine=False,
                     on_epoch=None, verbose=False):
    """Multi-task train: BCE(match) + ``gate_lambda`` * BCE(gate, interesting). ``pos_weight=None``
    balances to the class ratio. ``cosine=True`` decays the LR from ``lr`` to ~0 over ``epochs``
    (a constant LR just drifts on long runs). ``on_epoch(ep, model, train_loss)`` is called after
    each epoch for live train/test logging."""
    torch.manual_seed(seed)
    in_dim = sets[pairs[0][0]].spots.shape[1]
    model = StrictE2E(in_dim=in_dim, arch=arch, depth=depth, dropout=dropout, n_heads=n_heads,
                      residual=residual, sigma_pos=sigma_pos)
    npos = max(float(np.sum(y)), 1.0)
    pw = (len(y) - npos) / npos if pos_weight is None else float(pos_weight)
    match_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], dtype=torch.float32))
    gate_loss = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cosine else None
    idx = np.arange(len(pairs)); rng = np.random.default_rng(seed)
    yt = torch.tensor(y, dtype=torch.float32)

    for ep in range(epochs):
        rng.shuffle(idx); model.train(); tot = 0.0
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            opt.zero_grad()
            loss = _batch_loss(model, sets, [pairs[i] for i in b], yt[b],
                               match_loss, gate_loss, gate_lambda)
            loss.backward(); opt.step()
            tot += loss.detach().item() * len(b)
        if sched is not None:
            sched.step()
        train_loss = tot / len(idx)
        if verbose:
            logger.info(f"   ep{ep:3d} train_loss {train_loss:.4f}")
        if on_epoch is not None:
            on_epoch(ep, model, train_loss)
    return model


@torch.no_grad()
def pair_loss(model, sets, pairs, y, *, pos_weight=1.0, gate_lambda=0.0, batch=128):
    """Mean match loss over labelled pairs (for a test-loss curve). Gate loss off by default."""
    model.eval()
    match_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32),
                                      reduction="sum")
    gate_loss = nn.BCEWithLogitsLoss(reduction="sum")
    yt = torch.tensor(y, dtype=torch.float32); tot = 0.0
    for s in range(0, len(pairs), batch):
        tot += _batch_loss(model, sets, pairs[s:s + batch], yt[s:s + batch],
                           match_loss, gate_loss, gate_lambda).item()
    return tot / max(len(pairs), 1)


@torch.no_grad()
def score_strict_e2e(model, sets, pairs, batch=128):
    model.eval(); out = []
    for s in range(0, len(pairs), batch):
        Q, qm, C, cm, _, XYq, XYc = _collate(sets, pairs[s:s + batch])
        out.append(torch.sigmoid(model(Q, qm, C, cm, XYq, XYc)).numpy())
    return np.concatenate(out) if out else np.zeros(0)


# ============================================================ checkpoint save / load
def save_checkpoint(path, model, config, **meta):
    """Serialize the best model + everything needed to rebuild it (``config`` = StrictE2E kwargs)
    and reload it for inference. ``meta`` (epoch, metrics, dataset, fold, quality) is stored too."""
    torch.save(dict(state_dict=model.state_dict(), config=dict(config), **meta), str(path))


def load_strict_e2e(path):
    """Rebuild a trained :class:`StrictE2E` from a checkpoint. Returns ``(model.eval(), ckpt_dict)``.

    Inference: attach the same per-spot inputs the model expects and score pairs, e.g.::

        model, ck = load_strict_e2e(path)
        # sets = attach_centroids(get_image_sets(get_spot_embeddings()))
        # attach_positions(sets, pos_lookup)      # the vote's position gate needs xy
        scores = score_strict_e2e(model, sets, pairs)   # P(same individual) per (query, candidate)
    """
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    model = StrictE2E(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def gate_auroc(model, sets, images):
    """Held-out check that the learned gate recovers the human interesting-spot labels (AUROC)."""
    from census import auroc
    model.eval(); g, y = [], []
    for i in images:
        s = sets[i]
        if not np.isfinite(s.interesting).any():
            continue
        X = torch.tensor(s.spots[None]); m = torch.ones(1, len(s.spots), dtype=torch.bool)
        xy = torch.tensor(s.pos[None])
        _, gi = model(X, m, X, m, xy, xy, return_gate=True)
        gi = gi.squeeze(0).numpy(); ok = np.isfinite(s.interesting)
        g.extend(gi[ok].tolist()); y.extend(s.interesting[ok].tolist())
    return float(auroc(np.array(g), np.array(y))) if len(set(y)) == 2 else float("nan")
