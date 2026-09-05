"""Per-fold trainer for the learned aggregators (Phase 3).

Trains a :class:`SetTransformer` or :class:`GCN` with supervised-contrastive loss on
augmentation-manufactured two-view positive pairs. One model is trained per CV fold on that
fold's ``train_labels`` only (no eval leakage), then used to embed the fold's gallery/query.

Small + fast by design (444 images): a few dozen epochs of a tiny model on CPU.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..augment import geometry as G
from ..augment.spot_crops import SpotCropAug, augment_crop
from ..models.nn import GCN, SetTransformer, SetTransformerCNN
from .features import FEAT_DIM, SetFeatures, augment, feature_standardizer, normalize_pos
from .losses import supcon_loss

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

IN_DIM = FEAT_DIM + 2   # standardised shape/area + normalised (x, y)


@dataclass
class TrainConfig:
    arch: str = "set_transformer"   # or "gcn"
    d_model: int = 64
    embed_dim: int = 64
    n_layers: int = 2
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    temperature: float = 0.1
    knn: int = 6
    seed: int = 0
    augment: bool = True   # off = capacity/memorization check (two identical views)


def _make_model(cfg: TrainConfig):
    if cfg.arch == "gcn":
        return GCN(IN_DIM, d_model=cfg.d_model, n_layers=cfg.n_layers, embed_dim=cfg.embed_dim)
    return SetTransformer(IN_DIM, d_model=cfg.d_model, n_layers=cfg.n_layers, embed_dim=cfg.embed_dim)


def _knn_adj(pos: np.ndarray, k: int) -> np.ndarray:
    """Row-normalised kNN adjacency (no self-loops) from normalised positions."""
    n = len(pos)
    A = np.zeros((n, n), np.float32)
    if n > 1:
        d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        kk = min(k, n - 1)
        nn_idx = np.argsort(d, axis=1)[:, :kk]
        for i in range(n):
            A[i, nn_idx[i]] = 1.0
        deg = A.sum(1, keepdims=True)
        A = A / np.clip(deg, 1.0, None)
    return A


def _collate(items, mean, std, knn):
    """items: list of (feat, pos, label). -> X, mask, adj, labels tensors."""
    maxn = max(len(f) for f, _, _ in items)
    B = len(items)
    X = np.zeros((B, maxn, IN_DIM), np.float32)
    mask = np.zeros((B, maxn), bool)
    adj = np.zeros((B, maxn, maxn), np.float32)
    labels = []
    for b, (feat, pos, lab) in enumerate(items):
        n = len(feat)
        fz = (feat - mean) / std
        pz = normalize_pos(pos)
        X[b, :n] = np.hstack([fz, pz])
        mask[b, :n] = True
        adj[b, :n, :n] = _knn_adj(pz, knn)
        labels.append(lab)
    return (torch.from_numpy(X), torch.from_numpy(mask),
            torch.from_numpy(adj), labels)


def train_encoder(sets: list[SetFeatures], cfg: TrainConfig, *, verbose: bool = True):
    """Train one encoder on ``sets`` (the fold's train pool). Returns (model, mean, std)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    mean, std = feature_standardizer(sets)
    model = _make_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    every = max(1, cfg.epochs // 6)

    for ep in range(cfg.epochs):
        order = rng.permutation(len(sets))
        losses = []
        for i in range(0, len(sets), cfg.batch_size):
            batch = [sets[j] for j in order[i : i + cfg.batch_size]]
            items = []
            for sf in batch:                                  # two views per set
                for _ in range(2):
                    f, p = augment(sf, rng) if cfg.augment else (sf.feat, sf.pos)
                    items.append((f, p, sf.label))
            X, mask, adj, labels = _collate(items, mean, std, cfg.knn)
            set_emb, _ = model(X, mask) if cfg.arch != "gcn" else model(X, mask, adj)
            loss = supcon_loss(set_emb, labels, cfg.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if verbose and (ep % every == 0 or ep == cfg.epochs - 1):
            logger.info(f"      epoch {ep + 1}/{cfg.epochs}  supcon_loss={np.mean(losses):.4f}")
    model.eval()
    return model, mean, std


@torch.no_grad()
def embed_sets(model, cfg: TrainConfig, sets: list[SetFeatures], mean, std):
    """Inference set embeddings (no augmentation). Returns (ids, (M, embed) array)."""
    ids, embs = [], []
    for i in range(0, len(sets), 128):
        chunk = sets[i : i + 128]
        items = [(s.feat, s.pos, s.label) for s in chunk]
        X, mask, adj, _ = _collate(items, mean, std, cfg.knn)
        set_emb, _ = model(X, mask) if cfg.arch != "gcn" else model(X, mask, adj)
        embs.append(set_emb.cpu().numpy())
        ids.extend(s.salamander_id for s in chunk)
    return ids, np.vstack(embs).astype(np.float64)


@torch.no_grad()
def embed_tokens(model, cfg: TrainConfig, sets: list[SetFeatures], mean, std):
    """Per-spot token embeddings (no augmentation), for the Hungarian matcher. id -> (Ni, embed)."""
    out: dict[str, np.ndarray] = {}
    for s in sets:
        X, mask, adj, _ = _collate([(s.feat, s.pos, s.label)], mean, std, cfg.knn)
        _, tok = model(X, mask) if cfg.arch != "gcn" else model(X, mask, adj)
        out[s.salamander_id] = tok[0, : len(s.feat)].cpu().numpy().astype(np.float64)
    return out


# --------------------------------------------------------------------------- #
# Learned per-spot CNN encoder path (st_cnn): tokens come from OpenCV-augmented crops
# --------------------------------------------------------------------------- #
def _augment_cnn(sc, rng: np.random.Generator, crop_aug: SpotCropAug):
    """One augmented view of a SpotCrops: dropout + position warp + per-crop OpenCV augmentation."""
    crops, pos, logarea = sc.crops, sc.pos.copy(), sc.logarea.copy()
    n = len(crops)
    if n > 3:                                             # spot dropout (missing-spot robustness)
        keep = rng.random(n) >= 0.2
        if keep.sum() < 3:
            keep[rng.choice(n, 3, replace=False)] = True
        crops, pos, logarea = crops[keep], pos[keep], logarea[keep]
    a = np.ones(len(pos), np.float64)                     # position geometry augmentation
    pos, _ = G.similarity(pos, a, rng)
    pos, _ = G.affine_jitter(pos, a, rng)
    pos, _ = G.elastic_warp(pos, a, rng)
    pos, _ = G.jitter(pos, a, rng)
    aug = np.stack([augment_crop(c, rng, crop_aug) for c in crops])   # per-spot OpenCV aug
    return aug, pos, logarea


def _collate_cnn(items, la_mean, la_std, size):
    """items: list of (crops uint8 (n,size,size), pos, logarea, label). -> CR, EX, mask, labels."""
    maxn = max(len(c) for c, _, _, _ in items)
    B = len(items)
    CR = np.zeros((B, maxn, 1, size, size), np.float32)
    EX = np.zeros((B, maxn, 3), np.float32)
    mask = np.zeros((B, maxn), bool)
    labels = []
    for b, (crops, pos, logarea, lab) in enumerate(items):
        n = len(crops)
        CR[b, :n, 0] = crops.astype(np.float32) / 255.0
        pz = normalize_pos(pos)
        EX[b, :n, 0] = (logarea - la_mean) / la_std
        EX[b, :n, 1:] = pz
        mask[b, :n] = True
        labels.append(lab)
    return (torch.from_numpy(CR), torch.from_numpy(EX), torch.from_numpy(mask), labels)


def train_encoder_cnn(crops_sets: list, cfg: TrainConfig, *, verbose: bool = True):
    """Train the SpotCNN + Set Transformer on cropped spots. Returns (model, (la_mean, la_std))."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    size = crops_sets[0].crops.shape[-1]
    la = np.concatenate([s.logarea for s in crops_sets if len(s.logarea)])
    la_mean, la_std = float(la.mean()), float(la.std() or 1.0)
    model = SetTransformerCNN(d_model=cfg.d_model, n_layers=cfg.n_layers, embed_dim=cfg.embed_dim)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    crop_aug = SpotCropAug()
    every = max(1, cfg.epochs // 6)

    for ep in range(cfg.epochs):
        order = rng.permutation(len(crops_sets))
        losses = []
        for i in range(0, len(crops_sets), cfg.batch_size):
            batch = [crops_sets[j] for j in order[i : i + cfg.batch_size]]
            items = []
            for sc in batch:
                for _ in range(2):
                    c, p, l = _augment_cnn(sc, rng, crop_aug) if cfg.augment else (sc.crops, sc.pos, sc.logarea)
                    items.append((c, p, l, sc.label))
            CR, EX, mask, labels = _collate_cnn(items, la_mean, la_std, size)
            set_emb, _ = model(CR, EX, mask)
            loss = supcon_loss(set_emb, labels, cfg.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        if verbose and (ep % every == 0 or ep == cfg.epochs - 1):
            logger.info(f"      epoch {ep + 1}/{cfg.epochs}  supcon_loss={np.mean(losses):.4f}")
    model.eval()
    return model, (la_mean, la_std)


@torch.no_grad()
def embed_sets_cnn(model, crops_sets: list, stats) -> tuple[list, np.ndarray]:
    la_mean, la_std = stats
    size = crops_sets[0].crops.shape[-1]
    ids, embs = [], []
    for i in range(0, len(crops_sets), 64):
        chunk = crops_sets[i : i + 64]
        items = [(s.crops, s.pos, s.logarea, s.label) for s in chunk]
        CR, EX, mask, _ = _collate_cnn(items, la_mean, la_std, size)
        set_emb, _ = model(CR, EX, mask)
        embs.append(set_emb.cpu().numpy())
        ids.extend(s.salamander_id for s in chunk)
    return ids, np.vstack(embs).astype(np.float64)
