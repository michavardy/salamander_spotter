"""Frozen ImageNet-pretrained ResNet18 over per-spot mask crops — the true "pretrained network".

The other nine models all start from the hand-engineered 62-dim spot embedding (EFD shape +
body-intrinsic position). This one throws that away and asks a different question: does a network
pretrained on *natural images* carry shape priors that transfer to salamander spots?

The ResNet is **frozen**, so its features are a fixed function of the crop and can be computed
ONCE and cached; only a small projection head trains downstream. That is what makes this row
affordable — the alternative (forward the CNN every minibatch) would dominate the whole sweep.

Two honest caveats, both inherent to the comparison rather than to this implementation:

* **Domain gap.** ImageNet is RGB photographs of objects; a spot crop is a 32x32 binary mask.
  The pretrained filters were never trained on anything like this, so a weak result means "these
  particular features do not transfer", not "CNNs cannot work here".
* **Spot cap.** ``build_spot_crops`` keeps the largest 40 spots per photo (by area) and orders
  them by descending area, so this row sees a *capped, differently-ordered* spot set than the
  other nine. Voting is permutation-invariant so order is harmless, but the cap means images
  with >40 spots are compared on a subset.

    from pretrained_cnn import attach_cnn_features
    sets = attach_cnn_features(sets)          # adds ImageSet.cnn_feat, cached to disk
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

try:  # run as script (path[0] = this dir) OR imported as a package
    import data as d
except ModuleNotFoundError:
    from pipeline.spot_transformer import data as d

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

REPO_ROOT = Path(__file__).resolve().parents[2]
FEAT_DIM = 512                                                # resnet18 penultimate width
CROP_SIZE = 32                                                # what build_spot_crops emits
CNN_INPUT = 64                                                # upsampled so the last map is 2x2


class _Shim:
    """Minimal stand-in for a spot_embedding ``SpotSet`` — ``build_spot_crops`` only reads these."""

    def __init__(self, sid, label):
        self.salamander_id, self.label = sid, label


def _cache_path(dataset: str) -> Path:
    return (REPO_ROOT / "artifacts" / "spot_transformer" / "pretrained_cnn" /
            f"{dataset}_resnet18_{CROP_SIZE}.pkl")


def _load_resnet():
    """Frozen resnet18 truncated before the classifier -> (B, 512). Needs the pretrained weights
    (downloaded on first use and then cached by torchvision)."""
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net


def build_cnn_features(sets, dataset: str | None = None, *, cap: int = 40, batch: int = 256,
                       rebuild: bool = False) -> dict[str, np.ndarray]:
    """id -> ``(n_spots, 512)`` frozen ResNet18 features. Cached; the CNN runs at most once."""
    import torch

    dataset = dataset or d.dataset_name
    cache = _cache_path(dataset)
    if cache.is_file() and not rebuild:
        return pickle.loads(cache.read_bytes())

    sys.path.insert(0, str(REPO_ROOT / "pipeline"))
    from spot_embedding.train.crops import build_spot_crops                      # noqa: PLC0415

    shims = [_Shim(s.sid, s.label) for s in sets]
    crops = build_spot_crops(str(REPO_ROOT / "datasets" / dataset), shims,
                             size=CROP_SIZE, cap=cap)
    logger.info(f"  spot crops: {len(crops)} images")

    net = _load_resnet()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # one flat pass over every spot in the dataset, then split back per image
    ids = [k for k in crops if len(crops[k].crops)]
    counts = [len(crops[k].crops) for k in ids]
    flat = np.concatenate([crops[k].crops for k in ids]).astype(np.float32) / 255.0
    logger.info(f"  forwarding {len(flat):,} spot crops through frozen resnet18 ...")

    feats = []
    with torch.no_grad():
        for s in range(0, len(flat), batch):
            x = torch.tensor(flat[s:s + batch]).unsqueeze(1)                     # (B,1,32,32)
            x = torch.nn.functional.interpolate(x, size=CNN_INPUT, mode="bilinear",
                                                align_corners=False)
            x = (x.repeat(1, 3, 1, 1) - mean) / std                              # grey -> RGB
            feats.append(net(x).numpy())
    feats = np.concatenate(feats)

    out, at = {}, 0
    for k, n in zip(ids, counts):
        out[k] = feats[at:at + n].astype(np.float32)
        at += n
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    logger.info(f"  cached -> {cache}")
    return out


def attach_cnn_features(sets, dataset: str | None = None, **kw):
    """Attach ``ImageSet.cnn_feat``; images with no crops get an empty ``(0, 512)`` array."""
    feats = build_cnn_features(sets, dataset, **kw)
    miss = 0
    for s in sets:
        f = feats.get(s.sid)
        if f is None or not len(f):
            f = np.zeros((0, FEAT_DIM), np.float32)
            miss += 1
        s.cnn_feat = f
    if miss:
        logger.warning(f"  note: {miss} image(s) have no spot crops -> empty cnn_feat")
    return sets
