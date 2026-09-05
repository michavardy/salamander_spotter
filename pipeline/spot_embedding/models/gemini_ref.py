"""Gemini-as-matcher reference (1.3) — asks Gemini "same individual?" for a pairwise score.

The worst-case fallback and a difficulty reference, deliberately **outside** the automated
bake-off: it is billed, external, and non-deterministic, so it is never run by
``emb-selfcheck`` and refuses to run without an explicit ``--limit`` (a hard cap on API
calls). Responses are cached to ``prepared/<name>/gemini_cache.json`` so re-runs are free.

Run it opt-in, on a small subset only, e.g.::

    pixi run emb-eval --model gemini --limit 40   # caps folds to ~40 imgs -> bounded API calls

Requires GEMINI_API_KEY in the environment / .env (same key the label pipeline uses).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .._common import dataset_name, prepared_dir, raw_dir, resolve_dataset
from .base import Matcher

_PROMPT = (
    "These are two photographs of fire salamanders (Salamandra salamandra). Each individual "
    "has a unique, permanent pattern of yellow spots. Are these two photos the SAME individual? "
    "Answer with a single number from 0.0 (definitely different) to 1.0 (definitely the same) "
    "and nothing else."
)


class GeminiMatcher(Matcher):
    name = "gemini"
    produces_embedding = False

    def __init__(self, dataset: str, limit: int | None = None, model: str | None = None):
        if limit is None:
            raise ValueError(
                "GeminiMatcher is billed/external — pass --limit N to cap API calls "
                "(e.g. `pixi run emb-eval --model gemini --limit 40`)."
            )
        self.dataset = dataset
        self.limit = limit
        dataset_dir = resolve_dataset(dataset)
        self.name_ = dataset_name(dataset_dir)
        self.raw = raw_dir(dataset_dir)
        self.model = model or "gemini-2.5-flash"
        self._cache_path: Path = prepared_dir(self.name_) / "gemini_cache.json"
        self._cache: dict[str, float] = (
            json.loads(self._cache_path.read_text(encoding="utf-8"))
            if self._cache_path.is_file() else {}
        )
        self._client = None
        self._calls = 0

    # --- lazy client + image lookup -----------------------------------------
    def _ensure_client(self):
        if self._client is None:
            from google import genai  # noqa: PLC0415

            self._client = genai.Client()
        return self._client

    def _img_path(self, sid: str) -> Path:
        hits = list(self.raw.glob(f"{sid}.*"))
        if not hits:
            raise FileNotFoundError(f"no raw image for {sid!r} in {self.raw}")
        return hits[0]

    def _pair_score(self, sid_a: str, sid_b: str) -> float:
        key = "|".join(sorted((sid_a, sid_b)))
        if key in self._cache:
            return self._cache[key]
        if self._calls >= self.limit:
            return 0.0  # budget exhausted -> neutral; keeps the run bounded
        from google.genai import types  # noqa: PLC0415

        client = self._ensure_client()
        parts = [
            types.Part.from_bytes(data=self._img_path(sid_a).read_bytes(), mime_type="image/jpeg"),
            types.Part.from_bytes(data=self._img_path(sid_b).read_bytes(), mime_type="image/jpeg"),
            types.Part.from_text(text=_PROMPT),
        ]
        resp = client.models.generate_content(model=self.model, contents=parts)
        self._calls += 1
        try:
            score = float(resp.text.strip().split()[0])
        except (ValueError, AttributeError, IndexError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        self._cache[key] = score
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        return score

    def distance_matrix(self, queries: list, gallery: list) -> np.ndarray:
        out = np.ones((len(queries), len(gallery)), dtype=np.float64)
        for i, q in enumerate(queries):
            for j, g in enumerate(gallery):
                out[i, j] = 1.0 - self._pair_score(q.salamander_id, g.salamander_id)
        return out
