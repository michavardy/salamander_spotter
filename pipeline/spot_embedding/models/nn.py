"""Learned spot-set aggregators (Phase 3) — torch modules producing a salamander embedding.

Two architectures over the same per-spot tokens (`[invariant shape ⊕ log-area ⊕ rel-position]`):

* :class:`SetTransformer` (4.1) — global self-attention over spots + pooling-by-multihead-
  attention (PMA). Also exposes per-token embeddings (for the Hungarian matcher, 4.4).
* :class:`GCN` (4.2) — message passing over the kNN spot graph + attention pool (relative
  geometry enters through the adjacency).

Both are deliberately tiny (few layers, d_model≈64): 444 images punish capacity, so we lean on
augmentation. Inputs are padded batches with a validity ``mask`` (True = real spot).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid tokens. x:(B,N,D) mask:(B,N) bool (True=valid)."""
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


class SetTransformer(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 embed_dim: int = 64, ff: int = 128, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=ff,
                                           dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.seed = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(d_model, embed_dim))
        self.token_head = nn.Linear(d_model, embed_dim)

    def forward(self, x, mask):
        # mask: (B,N) True=valid  ->  key_padding_mask wants True=pad
        pad = ~mask
        h = self.encoder(self.proj(x), src_key_padding_mask=pad)           # (B,N,d)
        q = self.seed.expand(h.size(0), -1, -1)                            # (B,1,d)
        pooled, _ = self.pool_attn(q, h, h, key_padding_mask=pad)          # (B,1,d)
        set_emb = F.normalize(self.head(pooled.squeeze(1)), dim=-1)        # (B,embed)
        tok = F.normalize(self.token_head(h), dim=-1)                      # (B,N,embed)
        return set_emb, tok


class SpotCNN(nn.Module):
    """Tiny CNN over a single spot's mask crop → a learned per-spot appearance embedding.

    Replaces the Phase-2 hand-features with pixels the network learns from (fed
    OpenCV-augmented crops at train time). Deliberately small for a 32² binary input.
    """

    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),   # 32→16
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),  # 16→8
            nn.Conv2d(32, out_dim, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):        # x: (M, 1, H, W)
        return self.net(x).flatten(1)   # (M, out_dim)


class SetTransformerCNN(nn.Module):
    """Set Transformer whose per-spot tokens come from a learned :class:`SpotCNN`.

    Input: crops ``(B,N,1,H,W)`` + extras ``(B,N,3)`` = [log-area, pos_x, pos_y] + mask ``(B,N)``.
    """

    def __init__(self, spot_dim: int = 32, extra_dim: int = 3, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, embed_dim: int = 64, ff: int = 128):
        super().__init__()
        self.spot_cnn = SpotCNN(spot_dim)
        self.core = SetTransformer(spot_dim + extra_dim, d_model=d_model, n_heads=n_heads,
                                   n_layers=n_layers, embed_dim=embed_dim, ff=ff)

    def forward(self, crops, extra, mask):
        B, N = crops.shape[:2]
        flat = crops.reshape(B * N, 1, crops.shape[-2], crops.shape[-1])
        spot_emb = self.spot_cnn(flat).reshape(B, N, -1)     # (B,N,spot_dim)
        tokens = torch.cat([spot_emb, extra], dim=-1)        # (B,N,spot_dim+extra)
        return self.core(tokens, mask)


class GCNLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.self_w = nn.Linear(d_in, d_out)
        self.neigh_w = nn.Linear(d_in, d_out)

    def forward(self, h, adj_norm):
        # adj_norm: (B,N,N) row-normalised kNN adjacency (no self-loops)
        return self.self_w(h) + self.neigh_w(torch.bmm(adj_norm, h))


class GCN(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 64, n_layers: int = 2, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList([GCNLayer(d_model, d_model) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.attn = nn.Linear(d_model, 1)
        self.head = nn.Linear(d_model, embed_dim)
        self.token_head = nn.Linear(d_model, embed_dim)

    def forward(self, x, mask, adj_norm):
        h = self.proj(x)
        for layer, norm in zip(self.layers, self.norms):
            h = norm(F.gelu(layer(h, adj_norm)) + h)
        # attention pooling over valid tokens
        score = self.attn(h).squeeze(-1)                                   # (B,N)
        score = score.masked_fill(~mask, float("-inf"))
        w = torch.softmax(score, dim=1).unsqueeze(-1)                      # (B,N,1)
        pooled = (h * w).sum(1)
        set_emb = F.normalize(self.head(pooled), dim=-1)
        tok = F.normalize(self.token_head(h), dim=-1)
        return set_emb, tok
