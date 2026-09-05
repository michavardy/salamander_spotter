"""Gemini whole-image augmentation (opt-in, offline) — the second augmentation tier.

Where the OpenCV tier ([spot_crops](spot_crops.py)) perturbs individual spot masks, this tier
asks Gemini to synthesise **new photorealistic views of the same individual** — different
lighting, background, angle, wetness — while holding the yellow-spot pattern fixed. Each
accepted view becomes an extra photo of that individual (a real positive pair), directly
attacking the data-scarcity that limits the learned models.

Deliberately **offline and guarded** (billed/external/non-deterministic, and identity
preservation is *not* guaranteed): it refuses to run without an explicit ``limit`` (hard cap on
API calls), skips already-generated files (cache), and names outputs so they slot into the
dataset's identity scheme. Generated views must then be **spot-extracted and self-consistency
filtered** before training — see the workflow in ``docs/running.md``.

Outputs: ``images/<out_dir>/<label>_g<k>.png`` — e.g. a view of ``aa_1_1`` (individual ``aa_1``)
is written ``aa_1_g0.png``, so ``derive_label`` maps it back to ``aa_1``.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from .._common import REPO_ROOT, derive_label, raw_dir, resolve_dataset

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

G_RE = re.compile(r"^(.+)_g(\d+)$")

_PROMPT = (
    "This is a photograph of a single fire salamander (Salamandra salamandra). Generate a new, "
    "photorealistic image of the SAME individual salamander. You MUST preserve its unique pattern "
    "of yellow spots EXACTLY — the same number of spots, the same spot shapes, and the same spot "
    "positions relative to each other. Do NOT add, remove, move, merge, or reshape any yellow "
    "spot. Change only the {variation}. Keep the dorsal (top-down) view of the whole animal."
)
_VARIATIONS = [
    "ambient lighting and time of day",
    "natural background surface the animal rests on",
    "camera angle very slightly",
    "wet-vs-dry sheen of the skin",
    "amount of surrounding leaf litter",
]


class GeminiViewGenerator:
    def __init__(self, dataset: str, *, limit: int, out_dir: str | None = None,
                 model: str | None = None):
        if limit is None:
            raise ValueError(
                "GeminiViewGenerator is billed/external — pass a limit to cap API calls "
                "(e.g. `pixi run emb-gen-augment --limit 20`; --limit 0 = no cap)."
            )
        self.dataset_dir = resolve_dataset(dataset)
        self.raw = raw_dir(self.dataset_dir)
        # limit <= 0 means "no cap" — an explicit opt-in to generate everything remaining
        self.limit = float("inf") if limit <= 0 else limit
        self.model = model  # resolved lazily from GEMINI_MODEL if None
        self.out = REPO_ROOT / "images" / (out_dir or f"synth_{self.dataset_dir.name}")
        self._client = None
        self._calls = 0

    # --- lazy client (reuses the spotter's env handling) --------------------
    def _ensure_client(self):
        if self._client is None:
            from generate_spot_labels._common import getenv, load_dotenv  # type: ignore
            from google import genai

            dotenv = load_dotenv()
            key = getenv("GEMINI_API_KEY", "", dotenv)
            if not key:
                raise RuntimeError("GEMINI_API_KEY not set (env or .env)")
            self.model = self.model or getenv("GEMINI_MODEL", "gemini-2.5-flash-image", dotenv)
            self._client = genai.Client(api_key=key)
        return self._client

    @staticmethod
    def _extract_image(resp) -> bytes:
        for cand in getattr(resp, "candidates", []) or []:
            for part in getattr(getattr(cand, "content", None), "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    return inline.data
        raise RuntimeError("Gemini response contained no image part")

    def _one_view(self, image_bytes: bytes, mime: str, variation: str) -> bytes:
        from google.genai import types

        client = self._ensure_client()
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            _PROMPT.format(variation=variation),
        ]
        config = types.GenerateContentConfig(response_modalities=["IMAGE"])
        resp = client.models.generate_content(model=self.model, contents=contents, config=config)
        self._calls += 1
        return self._extract_image(resp)

    # --- what the output dir already holds ----------------------------------
    def existing_views(self) -> Counter:
        """label -> how many ``_g`` views the output dir already has."""
        if not self.out.exists():
            return Counter()
        return Counter(m.group(1) for p in self.out.glob("*_g*.png")
                       if (m := G_RE.match(p.stem)))

    def next_free_index(self) -> dict[str, int]:
        """label -> the first unused ``_g`` index in the output dir (0 if it has none).

        Writing in place next to views that already exist means ``_g0`` is taken; continuing the
        numbering is what keeps a fresh view from being mistaken for one already generated.
        """
        top: dict[str, int] = {}
        if self.out.exists():
            for p in self.out.glob("*_g*.png"):
                if m := G_RE.match(p.stem):
                    top[m.group(1)] = max(top.get(m.group(1), -1), int(m.group(2)))
        return {lbl: i + 1 for lbl, i in top.items()}

    def generate(self, src_ids: list[str], *, n_per: int = 2,
                 start: dict[str, int] | None = None) -> list[Path]:
        """Generate up to ``n_per`` views for each source id (subject to the call budget).

        Returns the list of written PNG paths. Skips outputs that already exist (cache).
        ``start`` maps a label to the first ``_g`` index to write, for generating in place next to
        views that already exist; the default (all zeros) keeps the fresh-output-dir behaviour.
        """
        self.out.mkdir(parents=True, exist_ok=True)
        start = start or {}
        written: list[Path] = []
        for sid in src_ids:
            hits = list(self.raw.glob(f"{sid}.*"))
            if not hits:
                logger.warning(f"  skip {sid}: no raw image")
                continue
            src = hits[0]
            label = derive_label(sid)
            data = src.read_bytes()
            mime = "image/jpeg" if src.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            base = start.get(label, 0)
            for k in range(base, base + n_per):
                dst = self.out / f"{label}_g{k}.png"
                if dst.exists():
                    continue
                if self._calls >= self.limit:
                    logger.info(f"  budget of {self.limit} calls reached — stopping")
                    return written
                try:
                    dst.write_bytes(self._one_view(data, mime, _VARIATIONS[k % len(_VARIATIONS)]))
                    written.append(dst)
                    logger.info(f"  wrote {dst.relative_to(REPO_ROOT)}  (call {self._calls}/{self.limit})")
                except Exception as exc:
                    logger.error(f"  ERROR generating {label}_g{k}: {exc}")
        return written
