from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

try:  # run as script (path[0] = this dir) OR imported as a package
    from data import SpotSetDataset, collate_sets
    from eval import match_precision_at_thresholds, match_auc
except ModuleNotFoundError:
    from pipeline.spot_transformer.data import SpotSetDataset, collate_sets
    from pipeline.spot_transformer.eval import match_precision_at_thresholds, match_auc

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)


def _spot_meta(sets, indices):
    """Per-spot (individual label, image index, spot_id) in the flatten order the encoders use."""
    labels, img, sid = [], [], []
    for k, i in enumerate(indices):
        n = len(sets[i].spots)
        labels.append(np.full(n, sets[i].label))
        img.append(np.full(n, k))                       # image index within `indices`
        sid.append(np.asarray(sets[i].spot_ids))
    return np.concatenate(labels), np.concatenate(img), np.concatenate(sid)


def encode_spots_raw(sets, indices) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """BEFORE baseline: the raw 62-dim spot embeddings, L2-normalized. -> (Z, labels, img, spot_id)."""
    Z = np.concatenate([sets[i].spots for i in indices]).astype(np.float64)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    return (Z, *_spot_meta(sets, indices))


@torch.no_grad()
def encode_spots_model(model, sets, indices, *, batch_size=32, device="cpu"):
    """AFTER: trained per-spot fingerprints via ``model.forward_spots``, flattened to real spots.
    -> (Z (M, out_dim), labels (M,), img (M,), spot_id (M,)). Row order follows `indices`."""
    from torch.utils.data import DataLoader
    model.eval()
    ds = SpotSetDataset(sets, indices, train=False)             # deterministic: no aug
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_sets)  # SequentialSampler
    chunks = []
    for X, mask, _ in loader:
        s = model.forward_spots(X.to(device), mask.to(device)).cpu().numpy()  # (B, N, out)
        chunks.append(s[(~mask).numpy()])                       # keep real spots, in-order
    Z = np.concatenate(chunks).astype(np.float64)
    labels, img, sid = _spot_meta(sets, indices)
    assert len(Z) == len(labels), f"spot count mismatch {len(Z)} vs {len(labels)}"
    return Z, labels, img, sid


def spot_vote_retrieval(Z, labels, img, ks=(1, 5, 10)) -> dict:
    """Image identification by SPOT VOTING (leave-one-image-out).

    For each query image, gallery = every spot from OTHER images. A candidate individual ``c``
    scores ``sum over query spots of ( max cosine to any gallery spot of c )`` -- a soft
    chamfer match. Rank individuals by score; ``recall@k`` = fraction of query images whose
    true individual lands in the top k. This is the aggregation that turns weak per-spot
    signal into a strong image ID, and it is directly comparable to v1's image R@k.
    """
    Z = np.asarray(Z); labels = np.asarray(labels); img = np.asarray(img)
    cats = np.unique(labels)
    hits = {k: 0 for k in ks}
    n = 0
    for q in np.unique(img):
        qm = img == q
        true = labels[qm][0]
        G, gl = Z[~qm], labels[~qm]
        cand = cats[[ (gl == c).any() for c in cats ]]          # individuals present in gallery
        if true not in cand:
            continue                                            # query's individual not in gallery
        S = Z[qm] @ G.T                                         # (nq, ng) cosine
        scores = np.array([S[:, gl == c].max(1).sum() for c in cand])
        ranked = cand[np.argsort(-scores)]
        rank = int(np.where(ranked == true)[0][0]) + 1
        for k in ks:
            hits[k] += int(rank <= k)
        n += 1
    out = {f"recall@{k}": (hits[k] / n if n else float("nan")) for k in ks}
    out["n_queries"] = n
    return out


def spot_match_precision(Z, labels, img, cutoffs=(0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)):
    """Cross-image spot-pair matching precision at cosine cutoffs (same-image pairs excluded).
    Directly comparable to the raw ``naive_match_all`` table."""
    return match_precision_at_thresholds(Z, labels, cutoffs, groups=img)


def spot_match_auc(Z, labels, img) -> float:
    """Threshold-free cross-image spot-match AUC (same-image pairs excluded)."""
    return match_auc(Z, labels, groups=img)


if __name__ == "__main__":
    import data as d
    sets = d.get_image_sets(d.get_spot_embeddings())
    _, eval_idx = d.get_cv_folds(sets, k=5, seed=0)[0]

    Z, lab, img, sid = encode_spots_raw(sets, eval_idx)
    logger.info(f"eval spots: {len(Z)}  from {len(np.unique(img))} images / {len(np.unique(lab))} individuals")
    logger.info(f"raw-spot VOTING identification: {spot_vote_retrieval(Z, lab, img)}")
    logger.info(f"raw-spot cross-image match AUC: {spot_match_auc(Z, lab, img):.3f}")
    from eval import print_match_table
    print_match_table(spot_match_precision(Z, lab, img), "raw-spot cross-image match precision:")
    breakpoint()   # live: Z, lab, img, sid
