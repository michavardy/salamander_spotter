"""Training losses (Phase 3).

Supervised contrastive (SupCon, Khosla et al. 2020): pull together every embedding sharing a
label, push apart the rest. Positives come from **both** two augmented views of one photo (so
even the 202 singletons contribute a positive pair) and different real photos of the same
individual — exactly the §7 signal.
"""
from __future__ import annotations

import torch


def supcon_loss(emb: torch.Tensor, labels, temperature: float = 0.1) -> torch.Tensor:
    """SupCon over L2-normalised embeddings ``emb`` (M, D) with integer/str ``labels`` (M,)."""
    device = emb.device
    if not torch.is_tensor(labels):
        # map arbitrary labels to ints
        uniq = {l: i for i, l in enumerate(sorted(set(labels)))}
        labels = torch.tensor([uniq[l] for l in labels], device=device)
    labels = labels.view(-1)
    m = emb.size(0)

    sim = emb @ emb.t() / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()      # stability
    self_mask = torch.eye(m, dtype=torch.bool, device=device)
    pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask

    exp = torch.exp(sim).masked_fill(self_mask, 0.0)
    denom = exp.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_prob = sim - torch.log(denom)

    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return sim.sum() * 0.0
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
    return -mean_log_prob_pos.mean()
