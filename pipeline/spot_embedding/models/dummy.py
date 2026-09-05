"""Phase-0 sanity matchers: Dummy (chance) and Oracle (perfect).

These validate the harness, not the science:

* :class:`DummyMatcher` — a random embedding seeded by the **image id** (not the label), so
  it is deterministic/reproducible yet uncorrelated with identity → chance performance. If
  the harness reports better-than-chance for this, something leaks.
* :class:`OracleMatcher` — distance 0 within a label, 1 across, by reading the ground-truth
  label. It *should* score ~1.0 everywhere; if it doesn't, the harness plumbing is wrong.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .base import EmbeddingMatcher, Matcher


def _seeded_vector(seed: int, key: str, dim: int) -> np.ndarray:
    """A deterministic standard-normal vector from (seed, key), independent of process hash."""
    h = hashlib.sha1(f"{seed}:{key}".encode()).digest()
    local_seed = int.from_bytes(h[:8], "little")
    return np.random.default_rng(local_seed).standard_normal(dim)


class DummyMatcher(EmbeddingMatcher):
    name = "dummy"

    def __init__(self, dim: int = 128, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def embed(self, spotsets: list) -> np.ndarray:
        return np.vstack(
            [_seeded_vector(self.seed, ss.salamander_id, self.dim) for ss in spotsets]
        )


class OracleMatcher(Matcher):
    name = "oracle"
    produces_embedding = False

    def distance_matrix(self, queries: list, gallery: list) -> np.ndarray:
        ql = np.array([q.label for q in queries])
        gl = np.array([g.label for g in gallery])
        return (ql[:, None] != gl[None, :]).astype(np.float64)
