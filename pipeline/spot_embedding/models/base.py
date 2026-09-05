"""The ``Matcher`` interface every candidate implements.

The harness only needs one thing from a matcher: a ``(Q, G)`` **distance** matrix between
query and gallery spot-sets (smaller = more similar). Two flavours cover every candidate in
the plan:

* **embedding matchers** (Set Transformer, GNN, CNN) — implement :meth:`EmbeddingMatcher.embed`;
  the base class turns embeddings into cosine distances and supports O(1) ANN retrieval.
* **scoring / matching matchers** (Hungarian, classical) — override :meth:`distance_matrix`
  directly with a pairwise score.

Phase 0 ships one of each (dummy = embedding, oracle = scoring) to exercise both paths.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Matcher(ABC):
    name: str = "matcher"
    produces_embedding: bool = False

    @abstractmethod
    def distance_matrix(self, queries: list, gallery: list) -> np.ndarray:
        """Return a ``(len(queries), len(gallery))`` distance matrix (smaller = closer)."""
        raise NotImplementedError


class EmbeddingMatcher(Matcher):
    """Base for matchers that map a ``SpotSet`` to a vector; distance = cosine distance."""

    produces_embedding = True

    @abstractmethod
    def embed(self, spotsets: list) -> np.ndarray:
        """Return an ``(N, D)`` float array of L2-normalizable embeddings."""
        raise NotImplementedError

    @staticmethod
    def _l2norm(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    def distance_matrix(self, queries: list, gallery: list) -> np.ndarray:
        eq = self._l2norm(np.asarray(self.embed(queries), dtype=np.float64))
        eg = self._l2norm(np.asarray(self.embed(gallery), dtype=np.float64))
        sim = eq @ eg.T
        return np.clip(1.0 - sim, 0.0, 2.0)
