"""Self-supervised contrastive pretraining on spot crops (SimCLR / NT-Xent).

MOTIVATION — this attacks the project's diagnosed bottleneck head-on. Identity supervision is
scarce (~1.4 real photos per individual, and only ~320 individuals with two), which is why every
learned representation tried so far has lost to hand-engineered features and why the novelty
classifier scored *below* chance. But the SPOTS are abundant: ~36k of them, and learning a spot
representation needs no identity labels at all. If a self-supervised encoder can match or beat the
62-dim hand-engineered embedding, label scarcity was the binding constraint; if it cannot, the
constraint lies in the representation itself. Either outcome is a result.

METHOD. Standard SimCLR: two independently augmented views of the same spot crop are positives;
every other crop in the batch is a negative; NT-Xent over cosine similarities. The projection head
is discarded after pretraining and the encoder's features become the per-spot embedding fed to the
existing voting pipeline, so the ONLY thing that changes versus the baseline is the representation.

WHY THE AUGMENTATIONS ARE WHAT THEY ARE. Contrastive learning learns invariance to whatever the
augmentations vary, so they must mimic how the SAME physical spot genuinely differs between two
photographs -- not arbitrary image noise:

    rotation / flip      the animal is photographed at any orientation
    scale / translation  distance to camera, and imprecise crop centring
    erode / dilate       the segmenter draws the boundary slightly differently each time
    cutout               partial occlusion by grass, leaf litter, or another body part
    edge blur            focus and resolution differences

Deliberately NOT varied: gross shape. Shape is the identity signal, so an augmentation that
distorted it would teach the encoder to discard the very thing being learned.

    pixi run python pipeline/spot_transformer/ssl_pretrain.py            # pretrain + cache
    QUICK=1 pixi run python pipeline/spot_transformer/ssl_pretrain.py    # smoke
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The flat-import bootstrap every other module in this package uses. The old two-branch try/except
# assumed `data` sat in THIS directory, which held while the file lived at
# spot_transformer/ssl_pretrain.py; after the refactor into core/ models/ eval/ it could no longer
# be run as a script at all -- `python .../models/ssl_pretrain.py` died on `import data`, so every
# SSL cache build failed at startup.
_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval"):
    _p = str(_ST / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

import data as d                                              # noqa: E402

# models/ -> spot_transformer/ -> pipeline/ -> repo root. This was parents[2] while the file lived
# at spot_transformer/ssl_pretrain.py; the move into models/ left it pointing at pipeline/, so every
# cached encoder under artifacts/ became unreachable and each SSL row would silently retrain from
# scratch (or fail) instead of reusing its cache.
REPO_ROOT = Path(__file__).resolve().parents[3]
CROP_SIZE = 32
EMB_DIM = 64                                                  # encoder output = spot embedding
PROJ_DIM = 32                                                 # projection head (discarded after)

# The hand-engineered baseline is [ EFD shape (37) | body-intrinsic position (25) ] -- see
# embeddings.get_spot_embeddings. A crop-based encoder can only ever learn the SHAPE half, so
# comparing its 64-d output against the full 62-d baseline compares shape against shape+position
# and is not a representation comparison at all. POS_SLICE re-attaches the same position block to
# the SSL embedding, making it shape_ssl+position vs shape_efd+position -- a controlled test of
# the one thing that actually differs.
POS_DIM = 25
POS_SLICE = slice(-POS_DIM, None)


# ============================================================ augmentations
def _augment(crop: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """One stochastic view of a ``(32, 32)`` uint8 spot mask. See the module docstring.

    ``strength`` scales the NUISANCE augmentations only (1.0 = the calibrated default). It is not
    a "more is better" knob: contrastive learning makes the encoder invariant to whatever varies,
    so pushing this up teaches it to ignore progressively more of the signal, and gross shape --
    the identity signal itself -- must never be distorted at any strength. Rotation and flips are
    held fixed because they are TRUE symmetries of the problem (a salamander is photographed at
    any orientation), so they are always fully applied. Select this on downstream identR@1, never
    on the pretext loss: the two are known to diverge here.
    """
    import cv2

    s = max(0.0, float(strength))
    img = crop.astype(np.float32) / 255.0
    h, w = img.shape

    # rotation + scale + translation, as a single affine so they compose cleanly
    ang = float(rng.uniform(0, 360))                       # true symmetry -> never scaled
    scale = float(rng.uniform(1 - 0.2 * s, 1 + 0.2 * s))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, max(0.1, scale))
    M[0, 2] += float(rng.uniform(-2 * s, 2 * s))
    M[1, 2] += float(rng.uniform(-2 * s, 2 * s))
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)

    if rng.random() < 0.5:                                 # true symmetry
        img = img[:, ::-1].copy()
    if rng.random() < 0.5:
        img = img[::-1, :].copy()

    # boundary jitter: the segmenter is not pixel-identical between photos
    if rng.random() < min(0.9, 0.5 * s):
        k = np.ones((3, 3), np.uint8)
        it = 1 if s <= 1.5 else 2
        img = (cv2.erode if rng.random() < 0.5 else cv2.dilate)(img, k, iterations=it)

    # occlusion
    if rng.random() < min(0.9, 0.3 * s):
        hi = max(5, int(10 * s))
        ch, cw = int(rng.integers(4, hi)), int(rng.integers(4, hi))
        y0, x0 = int(rng.integers(0, max(1, h - ch))), int(rng.integers(0, max(1, w - cw)))
        img[y0:y0 + ch, x0:x0 + cw] = 0.0

    if rng.random() < min(0.9, 0.3 * s):
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img.astype(np.float32)


class SpotPairDataset(torch.utils.data.Dataset):
    """Yields TWO independent augmented views of one crop — the SimCLR positive pair."""

    def __init__(self, crops: np.ndarray, seed: int = 0, strength: float = 1.0):
        self.crops = crops
        self.seed = seed
        self.strength = strength

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        c = self.crops[i]
        return (torch.from_numpy(_augment(c, rng, self.strength))[None],
                torch.from_numpy(_augment(c, rng, self.strength))[None])


# ============================================================ encoder
class SpotCNN(nn.Module):
    """Small CNN trained FROM SCRATCH on 32x32 binary masks.

    Deliberately not a ResNet: the ImageNet-pretrained baseline was measured on this exact data
    and its features carry essentially no identity signal (+0.002 same-vs-different gap against
    +0.024 for the hand-engineered embedding), so the domain gap is real and depth transferred
    from natural images is not the missing ingredient. ~100k params suits ~36k training crops.
    """

    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class SimCLR(nn.Module):
    """Encoder + projection head. Only the ENCODER is kept for downstream use."""

    def __init__(self, emb_dim=EMB_DIM, proj_dim=PROJ_DIM):
        super().__init__()
        self.encoder = SpotCNN(emb_dim)
        self.proj = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(),
                                  nn.Linear(emb_dim, proj_dim))

    def forward(self, x):
        return self.proj(self.encoder(x))


def nt_xent(z1, z2, temperature=0.5):
    """NT-Xent (SimCLR). 2N views; each view's positive is its counterpart, negatives are the
    other 2N-2. Self-similarities are masked to -inf so a view cannot select itself."""
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)         # (2N, d)
    sim = (z @ z.T) / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


# ============================================================ pretraining
def load_crops_indexed(dataset: str | None = None):
    """Every spot crop, keyed by ``(salamander_id, spot_id)``.

    Deliberately NOT ``spot_embedding.train.crops.build_spot_crops``: that sorts by descending
    area, caps at 40 per image, and discards ``spot_id`` — so its crops cannot be mapped back to
    the spots they came from. Both correspondence mining and position concatenation need exactly
    that mapping, so this reads the masks straight from the DB in ``spot_id`` order and keeps it.
    """
    import cv2
    import duckdb

    dataset = dataset or d.dataset_name
    # Decoding ~40k PNGs takes minutes, and every pretraining run / temperature sweep / ablation
    # would otherwise pay it again. Cache like build_spot_crops does.
    cache = (REPO_ROOT / "artifacts" / "spot_transformer" / "ssl" /
             f"{dataset}_crops_{CROP_SIZE}.pkl")
    if cache.is_file():
        blob = pickle.loads(cache.read_bytes())
        if blob.get("size") == CROP_SIZE:
            return blob["crops"], blob["keys"]

    sys.path.insert(0, str(REPO_ROOT / "pipeline"))
    from spot_embedding.encoders.spot_encoder import _normalize_crop        # noqa: PLC0415

    db = REPO_ROOT / "datasets" / dataset / "db" / "contours.db"
    con = duckdb.connect(str(db), read_only=True)
    try:
        rows = con.execute("SELECT salamander_id, spot_id, mask_png FROM spots "
                           "ORDER BY salamander_id, spot_id").fetchall()
    finally:
        con.close()

    crops, keys = [], []
    for sid, spot_id, png in rows:
        m = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE) if png else None
        crops.append(_normalize_crop(m, CROP_SIZE) if m is not None
                     else np.zeros((CROP_SIZE, CROP_SIZE), np.uint8))
        keys.append((sid, int(spot_id)))
    crops = np.asarray(crops, dtype=np.uint8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps({"size": CROP_SIZE, "crops": crops, "keys": keys}))
    return crops, keys


def mine_correspondences(sets, key_to_idx, *, min_sim=0.40, seed=0, report=True, use_geom=True):
    """REAL positive pairs: the same physical spot photographed twice.

    SimCLR's default positive is two augmentations of one crop, which only teaches invariance to
    augmentations someone guessed at. These pairs instead come from two genuinely different
    photographs of the same animal, so they carry the real nuisance variation — pose, lighting,
    and the segmenter drawing the boundary differently — that the downstream task must survive.
    Still label-free in the sense that matters: it bootstraps from the existing matcher, with no
    manual annotation.

    Precision matters more than recall here (a wrong pair teaches the encoder that two DIFFERENT
    spots are the same), so a pair must be a mutual nearest neighbour, clear ``min_sim``, AND be
    a RANSAC geometric inlier under the constellation transform.

    ``min_sim`` must be read against THIS data, not intuition: genuine same-individual spot
    matches average ~0.34 cosine here, so a "safe-looking" 0.7 keeps only freak near-duplicates
    (measured: 136 pairs, 0.6% of spots — untrainable). The geometric RANSAC check is the real
    precision filter; the similarity gate is only meant to remove obvious noise, so it is set
    near the genuine-match average and the yield is reported at several cutoffs so the choice is
    visible rather than assumed.

    ``use_geom=False`` drops the RANSAC filter — the ablation ``eval/mining_audit.py`` uses to check
    that the "real precision filter" earns that description against the human verdicts, rather than
    just shrinking the yield.
    """
    from aggregator import _norm                                             # noqa: PLC0415
    from aggregator_set import _geom_inlier_mask                             # noqa: PLC0415

    by_label: dict[str, list[int]] = {}
    for i, s in enumerate(sets):
        if not s.is_synth and len(s.spots) and s.centroids is not None:
            by_label.setdefault(s.label, []).append(i)

    grid = sorted({round(x, 2) for x in (0.30, 0.40, 0.45, 0.50, 0.60, 0.70, min_sim)})
    yield_at = {t: 0 for t in grid}
    pairs, checked = [], 0
    for idxs in by_label.values():
        if len(idxs) < 2:
            continue
        for ai in range(len(idxs)):
            for bi in range(ai + 1, len(idxs)):
                A, B = sets[idxs[ai]], sets[idxs[bi]]
                if len(A.spots) < 3 or len(B.spots) < 3:
                    continue
                checked += 1
                S = _norm(A.spots) @ _norm(B.spots).T
                abest = S.argmax(1)
                bbest = S.argmax(0)
                mut_all = [i for i in range(len(S)) if bbest[abest[i]] == i]
                if len(mut_all) < 3:
                    continue
                inl = (_geom_inlier_mask(np.asarray(A.centroids),
                                         np.asarray(B.centroids)[abest],
                                         np.asarray(mut_all, int), seed=seed)
                       if use_geom else np.ones(len(A.spots), bool))
                for i in mut_all:
                    if not inl[i]:
                        continue
                    sim = float(S[i, abest[i]])
                    for t in grid:                       # yield curve, for choosing the cutoff
                        if sim >= t:
                            yield_at[t] += 1
                    if sim < min_sim:
                        continue
                    ka = (A.sid, int(A.spot_ids[i]))
                    kb = (B.sid, int(B.spot_ids[abest[i]]))
                    if ka in key_to_idx and kb in key_to_idx:
                        pairs.append((key_to_idx[ka], key_to_idx[kb]))
    if report:
        logger.info(f"  mined {len(pairs):,} pairs from {checked:,} photo pairs "
              f"(min_sim={min_sim})")
        logger.info("  yield vs cutoff: "
              + "  ".join(f"{t:.2f}->{yield_at[t]:,}" for t in grid))
    return pairs


def human_correspondence_pairs(key_to_idx, *, dataset: str | None = None, report=True):
    """Positives a HUMAN confirmed: the accepted spot-to-spot correspondences from the review app.

    :func:`mine_correspondences` bootstraps from the matcher, so it inherits the matcher's errors —
    ``eval/mining_audit.py`` measures that at ~16% wrong pairs, and no cutoff or geometric filter
    moves it. These pairs carry no such error: someone looked at the two crops and said yes.

    They are few (hundreds, not thousands) and they are not a substitute for mining. They are worth
    adding because they are the only positives in the pile with a known error rate near zero, and
    because they cover pairs the miner *missed* — 415 of the adjudicated correspondences were drawn
    by hand precisely because the machine did not propose them.

    Caveat that must travel with them: only ~9% join two real photographs; the rest involve a
    Gemini-generated view, where "the same spot" holds by construction rather than by observation.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
    import review_labels as rl                                          # noqa: PLC0415

    df = rl.match_verdicts(dataset or d.dataset_name)
    if not len(df):
        return []
    acc = df[df["accepted"]]
    pairs, missing = [], 0
    for r in acc.itertuples(index=False):
        ka, kb = (r.sid_a, r.spot_a), (r.sid_b, r.spot_b)
        if ka in key_to_idx and kb in key_to_idx:
            pairs.append((key_to_idx[ka], key_to_idx[kb]))
        else:
            missing += 1
    if report:
        kinds = acc["pair_kind"].value_counts().to_dict()
        logger.info(f"  {len(pairs):,} human-verified pairs ({missing} unresolvable); kinds {kinds}")
    return pairs


class CorrespondencePairDataset(torch.utils.data.Dataset):
    """Positives are two crops of the SAME physical spot from DIFFERENT photos.

    Both are still lightly augmented: the real pair supplies genuine photographic variation, and
    augmentation on top keeps the encoder from latching onto per-photo artefacts.
    """

    def __init__(self, crops: np.ndarray, pairs: list[tuple[int, int]], seed: int = 0,
                 strength: float = 1.0):
        self.crops, self.pairs, self.seed = crops, pairs, seed
        self.strength = strength

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b = self.pairs[i]
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        return (torch.from_numpy(_augment(self.crops[a], rng, self.strength))[None],
                torch.from_numpy(_augment(self.crops[b], rng, self.strength))[None])


def pretrain(crops, *, pairs=None, epochs=30, batch=256, lr=1e-3, temperature=0.5, seed=0,
             strength=1.0, verbose=True):
    """SimCLR pretraining. Returns ``(model, history)``; history is the per-epoch loss curve
    (kept because a training curve is a required figure for the write-up).

    ``pairs`` (from :func:`mine_correspondences`) switches the positives from augmented views of
    one crop to two photographs of the same physical spot.
    """
    torch.manual_seed(seed)
    ds = (CorrespondencePairDataset(crops, pairs, seed=seed, strength=strength) if pairs
          else SpotPairDataset(crops, seed=seed, strength=strength))
    # drop_last is required (NT-Xent assumes a full batch of negatives), but with fewer samples
    # than one batch it silently yields ZERO batches: training would "run", change nothing, and
    # cache a randomly-initialised encoder under a trained-looking tag. Shrink the batch instead,
    # and refuse outright if even that cannot form a meaningful contrastive problem.
    if len(ds) < 2 * batch:
        batch = max(16, len(ds) // 4)
        logger.warning(f"   note: only {len(ds):,} samples -> batch {batch} "
              f"(fewer negatives per step, so the loss is NOT comparable across conditions)")
    if len(ds) < 64:
        raise ValueError(
            f"only {len(ds)} training samples — too few for contrastive pretraining. "
            f"Lower SSL_MIN_SIM to mine more correspondence pairs, or use SSL_MODE=augment.")
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True,
                                     num_workers=0)
    model = SimCLR()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    history = []
    for ep in range(epochs):
        model.train(); tot = n = 0
        for v1, v2 in dl:
            opt.zero_grad()
            loss = nt_xent(model(v1), model(v2), temperature)
            loss.backward(); opt.step()
            tot += loss.detach().item() * len(v1); n += len(v1)
        sched.step()
        history.append(tot / max(n, 1))
        if verbose:
            logger.info(f"   ssl epoch {ep:3d}  NT-Xent {history[-1]:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}")
    return model, history


@torch.no_grad()
def encode_crops(model, crops, batch=512):
    """Encoder features for every crop (projection head discarded, as SimCLR prescribes)."""
    model.eval()
    out = []
    for s in range(0, len(crops), batch):
        x = torch.from_numpy(crops[s:s + batch].astype(np.float32) / 255.0)[:, None]
        out.append(model.encoder(x).numpy())
    return np.concatenate(out) if out else np.zeros((0, EMB_DIM), np.float32)


def cache_path(dataset: str, tag: str = "simclr") -> Path:
    return (REPO_ROOT / "artifacts" / "spot_transformer" / "ssl" /
            f"{dataset}_{tag}_{EMB_DIM}d.pkl")


def attach_ssl_features(sets, dataset: str | None = None, tag: str = "simclr",
                        attr: str = "ssl_feat", with_position: bool = True):
    """Attach cached spot embeddings to ``ImageSet.<attr>``, aligned to each image's ``spot_ids``.

    ``tag``/``attr`` are separate so the pretrained encoder and its random-init control can be
    attached side by side and appear as two rows of ONE ablation table -- the comparison that
    isolates what pretraining buys from what the architecture buys.

    ``with_position`` appends the baseline's own body-intrinsic position block (``POS_SLICE``).
    On by default because without it the SSL row is shape-only and the baseline is
    shape+position, which makes any comparison between them meaningless.
    """
    dataset = dataset or d.dataset_name
    blob = pickle.loads(cache_path(dataset, tag).read_bytes())
    if "by_spot" not in blob:
        # Caches written before spot_id indexing keyed features by image, in DESCENDING-AREA
        # order and capped at 40 per image, so a feature cannot be matched to the spot it came
        # from. Silently mis-aligning them would corrupt every downstream number, so refuse.
        raise ValueError(
            f"SSL cache '{tag}' predates spot_id indexing and cannot be aligned to spots. "
            f"Rebuild it: pixi run python pipeline/spot_transformer/ssl_pretrain.py")
    feats = blob["by_spot"]                      # (salamander_id, spot_id) -> vector
    dim = blob["dim"] + (POS_DIM if with_position else 0)
    miss_img = miss_spot = 0
    for s in sets:
        rows = []
        for j, spot_id in enumerate(np.asarray(s.spot_ids).tolist()):
            v = feats.get((s.sid, int(spot_id)))
            if v is None:
                miss_spot += 1
                v = np.zeros(blob["dim"], np.float32)
            if with_position:
                v = np.concatenate([v, np.asarray(s.spots[j], np.float32)[POS_SLICE]])
            rows.append(v)
        arr = np.asarray(rows, np.float32) if rows else np.zeros((0, dim), np.float32)
        if not len(arr):
            miss_img += 1
        setattr(s, attr, arr)
    if miss_img or miss_spot:
        logger.warning(f"  note: '{tag}' — {miss_img} empty image(s), {miss_spot} spot(s) without a crop")
    return sets


def main():
    quick = bool(os.environ.get("QUICK"))
    epochs = 2 if quick else int(os.environ.get("SSL_EPOCHS", 30))
    tag = os.environ.get("SSL_TAG", "simclr")
    # A QUICK run trains 2 epochs. Writing that to the real tag silently replaces a 30-epoch
    # encoder with a smoke-test one, and NOTHING downstream can tell -- every later sweep just
    # reports worse numbers for reasons nobody can see. So a smoke run gets its own tag, always.
    if quick and not tag.endswith("_quick"):
        tag = f"{tag}_quick"
        logger.info(f"QUICK -> writing to tag '{tag}' so the real cache is not overwritten")
    random_init = bool(os.environ.get("SSL_RANDOM_INIT"))      # the no-pretraining ablation
    mode = os.environ.get("SSL_MODE", "augment")               # augment | corr
    temperature = float(os.environ.get("SSL_TEMP", 0.5))
    strength = float(os.environ.get("SSL_AUG", 1.0))       # nuisance-augmentation strength

    logger.info(f"loading spot crops for {d.dataset_name} ...")
    crops, keys = load_crops_indexed()
    key_to_idx = {k: i for i, k in enumerate(keys)}
    logger.info(f"  {len(crops):,} crops of {CROP_SIZE}x{CROP_SIZE}  (mode={mode}, T={temperature}, aug={strength})")

    pairs = None
    if mode == "corr":
        from aggregator import attach_centroids                              # noqa: PLC0415
        sets = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))
        # SSL_GEOM=0 drops the RANSAC filter. eval/mining_audit.py measured it against the human
        # verdicts: it changes precision by +0.010 (intervals overlap, so unresolvable at this
        # sample size) while keeping only ~51% of the pairs. Off is the better trade unless a
        # larger verdict set says otherwise.
        pairs = mine_correspondences(sets, key_to_idx,
                                     min_sim=float(os.environ.get("SSL_MIN_SIM", 0.40)),
                                     use_geom=os.environ.get("SSL_GEOM", "1") != "0")
        if os.environ.get("SSL_HUMAN") == "1":
            # Union, not replacement: the human set is small and covers different pairs (415 of
            # them are correspondences the miner never proposed). Duplicates are harmless to
            # NT-Xent but removed anyway so the reported pair count means what it says.
            human = human_correspondence_pairs(key_to_idx)
            before = len(pairs)
            pairs = list({tuple(sorted(p)) for p in pairs} | {tuple(sorted(p)) for p in human})
            logger.info(f"  mined {before:,} + human {len(human):,} -> {len(pairs):,} unique pairs")
        if len(pairs) < 64:
            # Falling back silently would cache an AUGMENTATION-trained encoder under the 'corr'
            # tag and misreport the ablation. Fail loudly instead — "correspondence mining is
            # infeasible at this data density" is a legitimate finding, a mislabelled row is not.
            raise SystemExit(
                f"only {len(pairs)} correspondence pairs — not enough to pretrain.\n"
                f"  Either lower the gate (SSL_MIN_SIM=0.35) or report this as the result: at "
                f"~1.4 real photos/individual there are too few genuine cross-photo spot "
                f"correspondences to learn from.")
        if quick:
            pairs = pairs[:2000]

    if random_init:
        # The control that isolates what PRETRAINING buys, as opposed to what the architecture
        # buys: identical encoder, never trained.
        logger.info("SSL_RANDOM_INIT set -> untrained encoder (ablation control)")
        torch.manual_seed(0)
        model, history = SimCLR(), []
    else:
        train_crops = crops[:2000] if (quick and pairs is None) else crops
        model, history = pretrain(train_crops, pairs=pairs, epochs=epochs,
                                  temperature=temperature, strength=strength)

    feats = encode_crops(model, crops)
    by_spot = {k: f.astype(np.float32) for k, f in zip(keys, feats)}

    out = cache_path(d.dataset_name, tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps({"by_spot": by_spot, "dim": EMB_DIM, "history": history,
                                  "epochs": epochs, "random_init": random_init,
                                  "mode": mode, "temperature": temperature, "aug": strength,
                                  "n_pairs": len(pairs) if pairs else 0}))
    logger.info(f"cached {len(by_spot):,} spot embeddings ({EMB_DIM}d) -> {out}")


if __name__ == "__main__":
    main()
