from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)


class SpotSetTransformer(nn.Module):
    """Set-transformer over one photo's spot tokens -> a single image embedding.

    A photo is a variable-size *set* of 62-dim spot embeddings (from ``spot_embeddings``).
    We lift each token to ``d_model``, prepend a learnable CLS token, run a few self-attention
    layers, and read the CLS row as the image fingerprint.

    Deliberately has **no positional encoding**: the tokens form a set (spot order is
    meaningless), and body position already lives *inside* each 62-dim token (the position
    block). Adding a sequence PE would inject false structure. The result is permutation-
    invariant across spots and, thanks to the key-padding mask, invariant to padding.

    Tuning notes
    ------------
    Data is small (~679 train images, 272 individuals) and sequences are short (~15-30
    spots), so overfitting — not underfitting — is the risk. Defaults are deliberately
    small. Sweep against the CV score (``data.get_cv_folds``), one knob at a time:

    - ``n_layers`` — the highest-leverage knob. One attention layer already gives full
      all-pairs spot interaction; depth only buys higher-order interactions + capacity.
      Try 1 vs 2 vs 3; do not exceed ~4.
    - ``n_heads`` — raise for more attention subspaces, but keep head_dim = d_model/n_heads
      >= ~16 (at d_model=128 that caps you at 8). Minor knob; 4 is a fine default.
    - **Add an FC layer at the end** — ``head`` is currently a single ``Linear``. Swapping it
      for a 2-layer MLP projection head (``d_model -> d_model -> out_dim`` with GELU) is the
      SimCLR/SupCon trick and can help retrieval. See the ``return_pre_head`` note below.
    - ``dim_feedforward`` — capacity of the per-token FFN inside each block (standard is
      4*d_model=512; 256 here is conservative). Raise if underfitting, lower if overfitting.
    - ``dropout`` — this is the *model's* internal dropout. Two more regularizers live on the
      **data** side (``data.SpotSetDataset``): ``dropout_p`` (drop whole spots) and
      ``jitter_std`` (Gaussian feature jitter, off by default). Tune all three together.
    - ``out_dim`` — the stored fingerprint size (what ``image_embeddings`` holds and cosine
      compares). 64 vs 128 vs 256; smaller = tighter/faster retrieval, larger = more capacity.
    - ``normalize`` — keep True: it makes the SupCon training geometry and the cosine
      retrieval metric identical. Only flip it off if the loss stops using cosine.
    - ``d_model`` — overall width; the other capacity dial alongside n_layers (try 96/128).

    Retrieval-head experiment: contrastive models often retrieve better from the
    *pre-projection* pooled vector than from the trained head (the head discards info). If a
    single ``head`` Linear plateaus, expose the CLS-readout vector via a ``return_pre_head``
    flag and store that as the fingerprint instead.
    """

    def __init__(
        self,
        in_dim: int = 62,          # spot embedding dim (matches spot_embeddings)
        d_model: int = 128,        # transformer width
        n_heads: int = 4,          # try 8, 16
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        out_dim: int = 128,        # final image-embedding dim
        normalize: bool = True,    # L2-normalize the output (SupCon wants unit vectors)
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.proj = nn.Linear(in_dim, d_model)                 # 62 -> d_model
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,      # our tensors are (B, N, D)
            norm_first=True,       # pre-LN: more stable when training from scratch
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False,   # N/A with norm_first; silences warning
        )
        self.norm = nn.LayerNorm(d_model)                      # on the pooled CLS
        self.head = nn.Linear(d_model, out_dim)                # image (CLS) projection  -> v1
        self.spot_head = nn.Linear(d_model, out_dim)           # per-spot projection     -> v2

    def forward(self, X: torch.Tensor, key_padding_mask: torch.Tensor,
                *, return_tokens: bool = False):
        """``X`` (B, N, in_dim), ``key_padding_mask`` (B, N) bool [True == PAD, from
        ``collate_sets``] -> image embeddings (B, out_dim).

        Projects the tokens, prepends CLS, **extends the mask by one un-masked column** for
        the CLS position, encodes, and reads the CLS row.

        With ``return_tokens=True`` also returns the **transformed per-spot embeddings**
        ``(B, N, d_model)`` — the encoder's output at the N spot positions (CLS dropped).
        These are each spot *contextualized by attention over the other spots in its image*;
        pad positions are meaningless (use ``~key_padding_mask`` to keep the real ones).
        Note they are d_model-dim (pre-head) and were NOT trained with a per-spot objective.
        """
        B = X.size(0)
        h = self.proj(X)                                       # (B, N, d_model)
        cls = self.cls.expand(B, -1, -1)                       # (B, 1, d_model)
        h = torch.cat([cls, h], dim=1)                         # (B, N+1, d_model)

        # CLS is never padding -> prepend a False column so the mask lines up with N+1.
        cls_col = torch.zeros(B, 1, dtype=torch.bool, device=X.device)
        mask = torch.cat([cls_col, key_padding_mask], dim=1)   # (B, N+1)

        h = self.encoder(h, src_key_padding_mask=mask)         # (B, N+1, d_model)
        z = self.norm(h[:, 0])                                 # CLS readout -> (B, d_model)
        z = self.head(z)                                       # (B, out_dim)
        if self.normalize:
            z = F.normalize(z, dim=-1)                         # unit vectors for cosine/SupCon
        if return_tokens:
            return z, h[:, 1:]                                 # (B, out_dim), (B, N, d_model)
        return z

    def forward_spots(self, X: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """Per-spot embeddings ``(B, N, out_dim)``, L2-normalized -- the spot-level (v2) head.

        Each spot is embedded *in the context of its image's set* (self-attention), then
        projected by ``spot_head``. Pad positions are meaningless (mask them out downstream
        with ``~key_padding_mask``). This is what the spot-level SupCon and voting eval use.
        """
        _, tokens = self.forward(X, key_padding_mask, return_tokens=True)   # (B, N, d_model)
        s = self.spot_head(tokens)                                          # (B, N, out_dim)
        if self.normalize:
            s = F.normalize(s, dim=-1)
        return s


class SpotMLP(nn.Module):
    """Context-FREE per-spot encoder: an MLP applied **independently** to each spot — no
    attention, no cross-spot mixing. So the same physical spot maps to the same embedding
    regardless of which photo it's in, preserving the per-spot correspondence that voting
    relies on.

    This is the control for "is it learning, or is it context?": it shares the spot-level
    SupCon objective and the ``forward_spots`` interface with ``SpotSetTransformer``, but
    strips the contextualization. If it beats raw-spot voting where the transformer couldn't,
    the transformer's *context* was the problem, not learning per se.
    """

    def __init__(self, in_dim: int = 62, hidden: int = 128, out_dim: int = 128,
                 dropout: float = 0.1, normalize: bool = True, **_ignored) -> None:
        super().__init__()
        self.normalize = normalize
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward_spots(self, X: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Per-spot embeddings ``(B, N, out_dim)``, L2-normalized. ``key_padding_mask`` is
        accepted for interface parity but unused — each spot is embedded independently, so
        padding rows are just meaningless outputs (dropped downstream via the mask)."""
        s = self.net(X)                                    # (B, N, out_dim)
        if self.normalize:
            s = F.normalize(s, dim=-1)
        return s


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    import data as d   # run as: pixi run python pipeline/spot_transformer/transformer.py

    # --- data: build one real PK batch (X, mask, labels) ---------------------------------
    sets = d.get_image_sets(d.get_spot_embeddings())
    train_idx = d.get_cv_folds(sets, k=5, seed=0)[0][0]
    ds = d.SpotSetDataset(sets, train_idx, train=True, dropout_p=0.2, seed=0)
    sampler = d.PKSampler(ds, P=4, K=2, num_batches=3, seed=0)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=d.collate_sets)
    X, mask, labels = next(iter(loader))

    # --- model: forward pass -------------------------------------------------------------
    model = SpotSetTransformer(in_dim=X.shape[-1], d_model=128, out_dim=128)
    n_params = sum(p.numel() for p in model.parameters())
    model.eval()                                               # disable dropout for deterministic checks
    with torch.no_grad():
        z = model(X, mask)

    logger.info(f"model params : {n_params:,}")
    logger.info(f"in           : X={tuple(X.shape)}  mask={tuple(mask.shape)}")
    logger.info(f"out          : z={tuple(z.shape)}  (expect ({X.shape[0]}, 128))")
    logger.info(f"output norms : {[round(v, 4) for v in z.norm(dim=-1).tolist()]}  (expect ~1.0)")

    # --- correctness 1: padding-invariance -----------------------------------------------
    # image 0 encoded ALONE (no padding) must equal image 0 inside the padded batch.
    # This is the proof the key_padding_mask actually excludes the pad rows.
    n0 = int((~mask[0]).sum())                                 # real spot count of image 0
    x_solo = X[0:1, :n0]                                       # its real spots, unpadded
    m_solo = torch.zeros(1, n0, dtype=torch.bool)             # nothing masked
    with torch.no_grad():
        z_solo = model(x_solo, m_solo)
    d_pad = (z_solo[0] - z[0]).abs().max().item()
    logger.info(f"padding-invariant     : {torch.allclose(z_solo[0], z[0], atol=1e-5)}  "
          f"max|delta|={d_pad:.2e}")

    # --- correctness 2: permutation (set) invariance -------------------------------------
    # shuffling the spot order must not change the image embedding.
    perm = torch.randperm(n0)
    with torch.no_grad():
        z_perm = model(x_solo[:, perm], m_solo)
    d_perm = (z_perm[0] - z_solo[0]).abs().max().item()
    logger.info(f"permutation-invariant : {torch.allclose(z_perm[0], z_solo[0], atol=1e-5)}  "
          f"max|delta|={d_perm:.2e}")

    breakpoint()   # live: X, mask, labels, z, z_solo, z_perm, model
