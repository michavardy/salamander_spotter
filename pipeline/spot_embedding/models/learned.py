"""Learned matchers (Phase 3) — thin wrappers over a fold-trained encoder.

Built per fold by the runner: after training an encoder on a fold's train split, embeddings (or
per-spot token embeddings) for that fold's gallery/query are precomputed once, and these wrappers
just look them up — so the harness's repeated ``distance_matrix`` calls stay cheap.

* :class:`FoldEmbeddingMatcher` (4.1 Set Transformer / 4.2 GNN) — cosine distance between set
  embeddings.
* :class:`FoldHungarianMatcher` (4.4) — optimal bipartite assignment between two photos' per-spot
  token embeddings (unmatched spots pay a penalty), i.e. explicit set-matching.
"""
from __future__ import annotations

import numpy as np

from .base import EmbeddingMatcher, Matcher


class FoldEmbeddingMatcher(EmbeddingMatcher):
    def __init__(self, emb_by_id: dict[str, np.ndarray], name: str = "learned"):
        self.emb_by_id = emb_by_id
        self.name = name

    def embed(self, spotsets: list) -> np.ndarray:
        return np.vstack([self.emb_by_id[ss.salamander_id] for ss in spotsets])


class FoldHungarianMatcher(Matcher):
    name = "hungarian"
    produces_embedding = False

    def __init__(self, tokens_by_id: dict[str, np.ndarray], unmatched_penalty: float = 1.0):
        self.tokens_by_id = tokens_by_id
        self.unmatched_penalty = unmatched_penalty

    def _score(self, a_id: str, b_id: str) -> float:
        from scipy.optimize import linear_sum_assignment

        A = self.tokens_by_id[a_id]      # (na, d) L2-normalised
        B = self.tokens_by_id[b_id]      # (nb, d)
        if len(A) == 0 or len(B) == 0:
            return 0.0
        sim = A @ B.T                    # cosine similarity (na, nb)
        cost = 1.0 - sim                 # assignment cost
        ri, ci = linear_sum_assignment(cost)
        matched = sim[ri, ci].sum()
        # normalise by the larger set so unmatched spots (missing/extra) dilute the score
        return float(matched / max(len(A), len(B)))

    def distance_matrix(self, queries: list, gallery: list) -> np.ndarray:
        out = np.empty((len(queries), len(gallery)), dtype=np.float64)
        for i, q in enumerate(queries):
            for j, g in enumerate(gallery):
                out[i, j] = 1.0 - self._score(q.salamander_id, g.salamander_id)
        return out
