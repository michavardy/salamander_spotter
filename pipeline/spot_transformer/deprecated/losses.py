from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1,
                groups: torch.Tensor | None = None) -> torch.Tensor:
    """Supervised Contrastive loss (Khosla et al., 2020), single-view.

    Pulls embeddings that share a label together and pushes different labels apart, directly
    on the cosine geometry the retrieval metric uses.

    Parameters
    ----------
    z : (B, d) embeddings. L2-normalized here defensively, so it is safe to pass either the
        raw or the pre-normalized output of the model.
    labels : (B,) integer identity ids (from ``SpotSetDataset.label_to_id``).
    temperature : softmax sharpness; lower = harder contrast. 0.05-0.2 is the usual range.
    groups : optional (B,) group id per element. When given, pairs in the SAME group are
        excluded from *both* the positives and the denominator. For the **spot-level** track,
        ``groups`` = image id: same-image spots share attention context, so they are trivial
        positives that teach nothing about cross-photo identity -- dropping them forces the
        model to learn cross-image invariance.

    Returns a scalar. Each anchor needs >=1 valid positive; anchors without one are skipped.

    The math, per anchor i with positives P(i) = {p : y_p = y_i, p != i, group_p != group_i}::

        L_i = -1/|P(i)| * sum_{p in P(i)} log( exp(s_ip/T) / sum_{a in D(i)} exp(s_ia/T) )

    where D(i) excludes i and same-group elements.
    """
    device = z.device
    B = z.shape[0]
    z = F.normalize(z, dim=-1)

    logits = (z @ z.T) / temperature                                   # (B, B) cosine / T
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()  # log-sum-exp stability

    exclude = torch.eye(B, dtype=torch.bool, device=device)            # self
    if groups is not None:
        g = groups.view(-1, 1)
        exclude = exclude | (g == g.T)                                 # + same-group (same-image)
    exp_logits = torch.exp(logits).masked_fill(exclude, 0.0)           # denom excludes self/same-group
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T) & ~exclude                         # same label, not excluded
    pos_counts = pos_mask.sum(dim=1)                                   # |P(i)|

    valid = pos_counts > 0                                             # anchors with a positive
    if valid.sum() == 0:
        return (z.sum() * 0.0)                                         # no positives in batch (rare) -> 0
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1)[valid] / pos_counts[valid]
    return -mean_log_prob_pos.mean()


if __name__ == "__main__":
    torch.manual_seed(0)

    def make_batch(P=4, K=2, d=16, structure="aligned"):
        """A (P*K, d) batch with labels [0,0,1,1,...]. `structure` controls difficulty."""
        labels = torch.arange(P).repeat_interleave(K)
        centers = F.normalize(torch.randn(P, d), dim=-1)              # one direction per identity
        if structure == "aligned":                                   # members hug their center
            z = centers[labels] + 0.05 * torch.randn(P * K, d)
        elif structure == "random":                                  # no identity structure
            z = torch.randn(P * K, d)
        elif structure == "adversarial":                             # positives far, negatives close
            poles = F.normalize(torch.randn(K, d), dim=-1)           # K spread directions
            within = torch.arange(P * K) % K                         # 0,1,0,1,... position-in-class
            z = poles[within] + 0.05 * torch.randn(P * K, d)         # kth member of EVERY class -> pole k
        return F.normalize(z, dim=-1), labels

    def sims(z, labels):
        """Mean cosine within-identity (positives) and across-identity (negatives)."""
        S = z @ z.T
        same = (labels[:, None] == labels[None, :]) & ~torch.eye(len(z), dtype=torch.bool)
        diff = labels[:, None] != labels[None, :]
        return S[same].mean().item(), S[diff].mean().item()

    # 1) the loss must RANK structure: aligned (easy) < random < adversarial (hard)
    logger.info("structure     loss     intra-cos  inter-cos")
    for name in ("aligned", "random", "adversarial"):
        z, y = make_batch(structure=name)
        intra, inter = sims(z, y)
        logger.info(f"{name:12s}  {supcon_loss(z, y):.4f}   {intra:+.3f}     {inter:+.3f}")

    z_a, y = make_batch(structure="aligned")
    z_r, _ = make_batch(structure="random")
    z_x, _ = make_batch(structure="adversarial")
    assert supcon_loss(z_a, y) < supcon_loss(z_r, y) < supcon_loss(z_x, y)
    logger.info("OK  loss(aligned) < loss(random) < loss(adversarial)")

    # 2) it is differentiable and actually separates identities when minimized.
    #    Optimize the RAW embeddings directly (no model) and watch the two cosines split.
    z, y = make_batch(structure="random")
    z = z.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.05)
    intra0, inter0 = sims(F.normalize(z.detach(), dim=-1), y)
    for _ in range(200):
        opt.zero_grad()
        loss = supcon_loss(z, y, temperature=0.1)
        loss.backward()
        opt.step()
    intra1, inter1 = sims(F.normalize(z.detach(), dim=-1), y)
    logger.info(f"optimize raw z for 200 steps -> loss {loss.item():.4f}")
    logger.info(f"  intra-cos {intra0:+.3f} -> {intra1:+.3f}   (same identity: should RISE)")
    logger.info(f"  inter-cos {inter0:+.3f} -> {inter1:+.3f}   (different identity: should FALL)")
    assert intra1 > intra0 and inter1 < inter0
    logger.info("OK  minimizing SupCon pulls same-label together and pushes others apart")
