from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)


def retrieval_metrics(Z: np.ndarray, labels: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Leave-one-out retrieval quality of a set of image fingerprints.

    Every image is used as a **query** in turn; the **gallery** is every *other* image.
    A retrieval is "correct" when the gallery image is the **same individual** as the query.
    We rank the gallery by cosine similarity and score how high the correct ones land.

    Parameters
    ----------
    Z : (M, d) fingerprints, assumed L2-normalized (so ``Z @ Z.T`` is cosine similarity).
    labels : (M,) identity of each image (strings or ints — only equality is used).
    ks : cutoffs for Recall@k.

    Returns a dict:
      - ``recall@k`` : fraction of queries with >=1 same-individual image in the top k
        (the re-ID question: "is the right salamander among the top k suggestions?").
      - ``mAP`` : mean Average Precision — rewards ranking *all* same-individual images high,
        not just the first.
      - ``median_rank`` : median position (1-based) of the *first* correct match.
      - ``n_queries`` : queries that had at least one positive in the gallery.
    """
    Z = np.asarray(Z, dtype=np.float64)
    labels = np.asarray(labels)
    M = len(Z)

    S = Z @ Z.T                                    # (M, M) cosine similarity
    np.fill_diagonal(S, -np.inf)                   # a query can't retrieve itself
    order = np.argsort(-S, axis=1)[:, : M - 1]     # gallery per query, best first, self dropped
    ranked_labels = labels[order]                  # (M, M-1) label at each rank
    match = ranked_labels == labels[:, None]       # True where retrieved == query individual

    has_pos = match.any(axis=1)                    # queries with a findable partner (all, here)
    match = match[has_pos]
    Mq = len(match)

    out: dict = {f"recall@{k}": float(match[:, :k].any(axis=1).mean()) for k in ks}

    aps = []                                       # Average Precision per query
    for row in match:
        hits = np.flatnonzero(row)                 # ranks (0-based) of the correct matches
        precisions = (np.arange(len(hits)) + 1) / (hits + 1)   # precision@each hit
        aps.append(precisions.mean())
    out["mAP"] = float(np.mean(aps))

    first_hit = match.argmax(axis=1) + 1           # 1-based rank of the first correct match
    out["median_rank"] = float(np.median(first_hit))
    out["n_queries"] = int(Mq)
    return out


def match_precision_at_thresholds(
    Z: np.ndarray, labels: np.ndarray,
    cutoffs=(0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9),
    groups: np.ndarray | None = None,
) -> list[dict]:
    """All-to-all matching precision/recall at cosine cutoffs (image-level analogue of the
    spot-level ``naive_match_all`` analysis).

    Every unordered pair of fingerprints is a candidate match; ``is_match`` = same individual.
    For each cutoff, of the pairs scoring ``>= cutoff``:
      - ``precision`` = fraction that are truly the same individual (your "True" fraction),
      - ``recall``    = fraction of ALL same-individual pairs that clear the cutoff,
      - ``n_above``   = how many pairs survive (watch this — high cutoffs go sparse/noisy).

    ``groups`` (optional): exclude same-group pairs. For spots, pass the image id so only
    cross-image spot pairs count (mirrors ``naive_match_all``'s same-photo exclusion).
    """
    Z = np.asarray(Z, dtype=np.float64)
    labels = np.asarray(labels)
    S = Z @ Z.T
    iu = np.triu_indices(len(Z), k=1)                  # upper triangle: each pair once, no self
    if groups is not None:
        keep = np.asarray(groups)[iu[0]] != np.asarray(groups)[iu[1]]
        iu = (iu[0][keep], iu[1][keep])
    sim = S[iu]
    same = labels[iu[0]] == labels[iu[1]]
    total_true = int(same.sum())
    rows = []
    for c in cutoffs:
        above = sim >= c
        n = int(above.sum())
        n_true = int(same[above].sum())
        rows.append(dict(
            cutoff=float(c), n_above=n, n_true=n_true,
            precision=(n_true / n if n else float("nan")),
            recall=(n_true / total_true if total_true else float("nan")),
        ))
    return rows


def match_auc(Z: np.ndarray, labels: np.ndarray, groups: np.ndarray | None = None) -> float:
    """Threshold-FREE match quality: P(a random same-individual pair scores higher than a
    random different-individual pair) = ROC-AUC of same-vs-different over all pairs.

    Scale-invariant, so unlike precision-at-cutoff it IS comparable across embedding spaces
    (mean-pool vs trained). 0.5 = chance, 1.0 = perfect separation. ``groups`` (optional):
    exclude same-group pairs (pass image id for spots -> cross-image only).
    """
    from scipy.stats import rankdata
    Z = np.asarray(Z, dtype=np.float64)
    labels = np.asarray(labels)
    iu = np.triu_indices(len(Z), k=1)
    if groups is not None:
        keep = np.asarray(groups)[iu[0]] != np.asarray(groups)[iu[1]]
        iu = (iu[0][keep], iu[1][keep])
    sim = (Z @ Z.T)[iu]
    same = labels[iu[0]] == labels[iu[1]]
    n_pos, n_neg = int(same.sum()), int((~same).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(sim)                                   # average ranks (handles ties)
    return float((r[same].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def print_match_table(rows: list[dict], title: str = "") -> None:
    """Pretty-print a :func:`match_precision_at_thresholds` result."""
    if title:
        logger.info(title)
    logger.info("   cutoff  n_pairs  precision(True)  recall")
    for r in rows:
        p = "  nan " if r["n_above"] == 0 else f"{r['precision']:.3f}"
        logger.info(f"    {r['cutoff']:.2f}  {r['n_above']:>7}    {p:>8}       {r['recall']:.3f}")


def encode_baseline(sets, indices) -> tuple[np.ndarray, np.ndarray]:
    """The zero-training baseline: mean-pool each image's spots into one L2-normalized
    fingerprint. This is what the trained transformer must beat."""
    Z = np.stack([sets[i].spots.mean(0) for i in indices]).astype(np.float64)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    labels = np.array([sets[i].label for i in indices])
    return Z, labels


@torch.no_grad()
def encode_model(model, sets, indices, *, batch_size=64, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    """Run the transformer over a split (no dropout/jitter) -> (M, d) fingerprints + labels.
    Row order follows ``indices``. This is what ``train.py`` calls to score a checkpoint."""
    from torch.utils.data import DataLoader
    try:
        from data import SpotSetDataset, collate_sets
    except ModuleNotFoundError:
        from pipeline.spot_transformer.data import SpotSetDataset, collate_sets

    model.eval()
    ds = SpotSetDataset(sets, indices, train=False)          # deterministic: no aug
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_sets)  # SequentialSampler
    Z = [model(X.to(device), mask.to(device)).cpu().numpy() for X, mask, _ in loader]
    labels = np.array([sets[i].label for i in indices])
    return np.concatenate(Z), labels


if __name__ == "__main__":
    import data as d   # run as: pixi run python pipeline/spot_transformer/eval.py

    sets = d.get_image_sets(d.get_spot_embeddings())
    train_idx, eval_idx = d.get_cv_folds(sets, k=5, seed=0)[0]

    # fingerprints for the held-out eval images (mean-pool baseline; no model needed)
    Z, labels = encode_baseline(sets, eval_idx)
    sids = np.array([sets[i].sid for i in eval_idx])
    logger.info(f"eval fold: {len(eval_idx)} images, {len(set(labels))} individuals, dim={Z.shape[1]}")

    metrics = retrieval_metrics(Z, labels, ks=(1, 5, 10))
    logger.info("mean-pool baseline:")
    for k, v in metrics.items():
        logger.info(f"   {k:12s} = {v:.3f}" if isinstance(v, float) else f"   {k:12s} = {v}")

    # ---- worked example: watch ONE query retrieve its neighbours ----
    # Recall@1 asks: is rank-1 a MATCH?  Recall@5: is any of the top 5 a MATCH?
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    q = 0
    n_pos = int((labels == labels[q]).sum() - 1)          # how many correct answers exist
    order = np.argsort(-S[q])                             # gallery ranked best-first
    logger.info(f"query: {sids[q]}   individual={labels[q]}   ({n_pos} correct match(es) in gallery)")
    logger.info("   rank   cos     retrieved             individual")
    for r, j in enumerate(order[:8], 1):
        flag = "  <== SAME INDIVIDUAL" if labels[j] == labels[q] else ""
        logger.info(f"   {r:>3}   {S[q, j]:+.3f}   {sids[j]:<18}   {labels[j]:<10}{flag}")
    first = np.flatnonzero(labels[order] == labels[q])[0] + 1
    logger.info(f"   -> first correct match is at rank {first} "
          f"(so this query {'HITS' if first == 1 else 'misses'} recall@1, "
          f"{'HITS' if first <= 5 else 'misses'} recall@5)")

    breakpoint()   # live: Z, labels, sids, S, metrics, order
