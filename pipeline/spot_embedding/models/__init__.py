"""models — the Matcher interface and the Phase-0 sanity matchers.

Every candidate matcher (Set Transformer, GNN, CNN, Hungarian, classical) will implement
:class:`~spot_embedding.models.base.Matcher`. Phase 0 ships only the two sanity matchers
that validate the harness:

* :class:`DummyMatcher` — random, id-seeded embeddings → chance performance (leakage probe).
* :class:`OracleMatcher` — cheats by reading labels → perfect performance (plumbing probe).
"""
from .base import EmbeddingMatcher, Matcher
from .classical import ConstellationMatcher
from .dummy import DummyMatcher, OracleMatcher
from .spotdesc import SpotDescMatcher

# name -> class. cnn / gemini import their heavy deps lazily, so listing them here is safe
# even when torch / genai are absent; they only fail if you actually build them.
MATCHERS = {
    "dummy": DummyMatcher,
    "oracle": OracleMatcher,
    "classical": ConstellationMatcher,
    "spotdesc": SpotDescMatcher,
    "cnn": ("spot_embedding.models.cnn_baseline", "CNNBaselineMatcher"),
    "gemini": ("spot_embedding.models.gemini_ref", "GeminiMatcher"),
}


def build_matcher(name: str, **kwargs):
    """Instantiate a matcher by name, importing lazy (torch/genai) ones only on demand."""
    if name not in MATCHERS:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MATCHERS)}")
    entry = MATCHERS[name]
    if isinstance(entry, tuple):
        import importlib

        module, cls_name = entry
        entry = getattr(importlib.import_module(module), cls_name)
    return entry(**kwargs)


__all__ = [
    "Matcher",
    "EmbeddingMatcher",
    "DummyMatcher",
    "OracleMatcher",
    "ConstellationMatcher",
    "MATCHERS",
    "build_matcher",
]
