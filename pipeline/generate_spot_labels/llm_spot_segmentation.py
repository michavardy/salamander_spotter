#!/usr/bin/env python3
"""Stage 1 — LLM spot segmentation via Gemini image inpainting.

Sends each salamander image to Gemini (``gemini-2.5-flash-image`` by default) with
an inpainting prompt that recolours the yellow/orange spots to flat magenta
(#FF00FF) while leaving every other pixel untouched. Because the image model
re-renders at its own resolution, the output is resized back onto the original's
pixel grid (see :func:`conform_to_original`) so downstream contours stay in the
source coordinate space. The result is written as PNG (lossless — no JPEG
artefacts on the flat key colour) to::

    <input_dir>/purple/<stem>.png

Pure logic: no argument parsing here. Import :class:`GeminiSpotSegmenter` and call
:meth:`segment_file`, or :func:`segment_dir` for a whole folder. The CLI that configures
and calls it is ``scripts/dataset/extract_spot_labels.py segment`` (``pixi run extract-spot-labels
segment``). The network is only touched inside this module.
"""
from __future__ import annotations

import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

from ._common import (
    getenv,
    list_images,
    load_dotenv,
    purple_dir_for,
    resolve_input_dir,
)

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

# The inpainting instruction. Sent as the text part alongside the source image.
# Written to fight the two observed failure modes: (a) magenta flooding the whole
# body instead of the discrete spots, and (b) spots rounded into smooth blobs.
PROMPT = """\
System Instruction / Role: Act as a precise, pixel-level image inpainting engine. You
are NOT re-imagining the photo — you recolour a set of small, SEPARATE regions and copy
every other pixel through unchanged.

Input: The attached photo of a fire salamander (black skin with discrete yellow/orange
spots) is the absolute baseline template.

Task — recolour ONLY the yellow/orange spots to flat magenta (Hex #FF00FF):
1. Find each individual yellow/orange spot. The spots are separate islands of colour
   scattered over black skin; there may be a dozen or more.
2. Fill each spot with a completely flat, uniform #FF00FF — no gradients, highlights,
   specular reflections or shading. Solid single-colour shapes only.
3. Trace the exact, irregular outline of every spot: keep each bump, notch, spike and
   jagged edge. Do NOT round or smooth spots into clean ovals or blobs; the magenta
   boundary must follow the real spot edge pixel-for-pixel.
4. Every pixel that is NOT a yellow/orange spot stays identical: the grey pavement, the
   BLACK skin (including all black skin BETWEEN spots), shadows, and any occluding twigs
   or pine needles. Do not re-synthesize or re-imagine the background.

Hard constraints — this is exactly where the model usually fails, so do NOT:
- Do NOT paint a single continuous magenta blob or stripe down the body.
- If two spots are separated by ANY black skin, they MUST stay two separate magenta
  shapes with the black skin left untouched between them. Never bridge spots.
- Do NOT let magenta bleed past a spot's edge onto the surrounding black skin.
- Most of the body is black skin and MUST stay black. Do not colour the whole salamander."""

# Yellow/orange spot colour in the ORIGINAL photo, used by the bleed check (OpenCV HSV,
# H 0..179). Loose on purpose: the judge only needs to catch gross flooding, not extract.
ORIG_SPOT_HSV_LOW = (10, 60, 60)
ORIG_SPOT_HSV_HIGH = (40, 255, 255)

DEFAULT_MAX_BLEED = 0.35        # reject a draft if >this fraction of magenta lands off the spots
DEFAULT_MAX_ATTEMPTS = 3        # cheap-model re-draws to try to beat DEFAULT_MAX_BLEED
DEFAULT_ESCALATE_ATTEMPTS = 2   # extra draws on the pricier model if the cheap one never passed

_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def mime_for(path: Path) -> str:
    return _MIME_BY_EXT.get(path.suffix.lower(), "image/jpeg")


def conform_to_original(purple_bytes: bytes, original_path: Path) -> bytes:
    """Resize the model's output back onto the original image's pixel grid.

    ``gemini-2.5-flash-image`` re-renders at its own resolution / aspect bucket, so
    the returned image is a scaled (and slightly reframed) version of the input.
    We resize it back to the original's (width, height) and re-encode as PNG, so the
    spot centroids/contours extracted in stage 2 live in the *original* image's
    coordinate space. Returns the bytes unchanged if either image fails to decode.

    Resampling is nearest-neighbour, not linear: the spots are a flat key colour and
    we want their edges to stay crisp. Linear interpolation would blend magenta with
    the adjacent black skin into off-key intermediate hues that stage 2's HSV band then
    slices through arbitrarily — softening and eroding the very edges we want to keep.
    """
    import cv2
    import numpy as np

    orig = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    purple = cv2.imdecode(np.frombuffer(purple_bytes, np.uint8), cv2.IMREAD_COLOR)
    if orig is None or purple is None:
        return purple_bytes
    oh, ow = orig.shape[:2]
    ph, pw = purple.shape[:2]
    if (pw, ph) != (ow, oh):
        purple = cv2.resize(purple, (ow, oh), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", purple)
    return buf.tobytes() if ok else purple_bytes


def bleed_fraction(original_bgr, purple_bgr, tol_px: int = 9) -> float:
    """Fraction of magenta output pixels that do NOT sit on an originally spot-coloured
    pixel — the signature of the flood failure.

    ``~0.0`` means the magenta landed only where the photo was yellow/orange (a clean
    recolour); a high value means magenta bled onto black skin / flooded the body. The
    original spot mask is dilated by ``tol_px`` so anti-aliased spot edges still count as
    "on a spot". Returns ``1.0`` if nothing was painted (also a failure).
    """
    import cv2
    import numpy as np
    from .extract_spot_contours import magenta_mask  # reuse stage 2's magenta definition

    m_out = magenta_mask(purple_bgr) > 0
    hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    m_spot = cv2.inRange(hsv, np.array(ORIG_SPOT_HSV_LOW, np.uint8),
                         np.array(ORIG_SPOT_HSV_HIGH, np.uint8))
    if tol_px and tol_px >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol_px, tol_px))
        m_spot = cv2.dilate(m_spot, k)
    m_spot = m_spot > 0

    painted = int(m_out.sum())
    if painted == 0:
        return 1.0
    bled = int(np.logical_and(m_out, ~m_spot).sum())
    return bled / painted


def parse_model_list(spec: str | list[str] | None) -> list[str]:
    """Normalise a models spec (comma-separated string or list) to an ordered, unique list.

    Used for the escalation ladder: cheapest model first, e.g.
    ``"gemini-3.1-flash-image,gemini-3-pro-image"`` -> two rungs above the primary.
    """
    if not spec:
        return []
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    out: list[str] = []
    for it in items:
        m = it.strip()
        if m and m not in out:
            out.append(m)
    return out


class SegmentResult(NamedTuple):
    """Outcome of a judged segmentation: the chosen PNG plus why it was chosen."""
    png: bytes
    bleed: float      # bleed_fraction of the returned draft (nan if it couldn't be scored)
    passed: bool      # True if bleed <= max_bleed within the attempt budget
    attempts: int     # total Gemini draws across all model tiers
    model: str        # the model that produced the returned draft


class GeminiSpotSegmenter:
    """Thin wrapper over google-genai for the spot-recolouring inpaint call."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 5,
        base_delay: float = 2.0,
        temperature: float | None = None,
        escalate_models: str | list[str] | None = None,
        escalate_attempts: int = DEFAULT_ESCALATE_ATTEMPTS,
    ):
        dotenv = load_dotenv()
        self.api_key = api_key or getenv("GEMINI_API_KEY", "", dotenv)
        self.model = model or getenv("GEMINI_MODEL", "gemini-2.5-flash-image", dotenv)
        # Ordered ladder of pricier fallbacks (cheapest first), each tried in turn only
        # when every cheaper rung failed to beat the bleed threshold. Primary is dropped
        # if it reappears in the list.
        raw = escalate_models if escalate_models is not None else getenv("GEMINI_ESCALATE_MODELS", "", dotenv)
        self.escalate_models = [m for m in parse_model_list(raw) if m != self.model]
        self.escalate_attempts = escalate_attempts
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is empty (set it in salamander_spotter/.env)")
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.temperature = temperature
        from google import genai  # lazy import — only needed for the real call
        self._client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _extract_image(resp) -> bytes:
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    return inline.data
        raise RuntimeError("Gemini response contained no image part")

    def segment(self, image_bytes: bytes, mime_type: str = "image/jpeg",
                model: str | None = None, prompt: str | None = None) -> bytes:
        """Return PNG/JPEG bytes of the image with spots recoloured to magenta.

        ``model`` overrides the primary model for this call (used by the escalation
        ladder); ``prompt`` overrides the default inpaint instruction (used to re-issue a
        stronger, anti-shadow request on a re-roll); ``max_retries`` here is transport-level
        backoff for API errors, not the quality-driven re-draws that :meth:`segment_file` does.
        """
        from google.genai import types

        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt or PROMPT,
        ]
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            **({"temperature": self.temperature} if self.temperature is not None else {}),
        )
        model = model or self.model
        delay = self.base_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=model, contents=contents, config=config
                )
                return self._extract_image(resp)
            except Exception as exc:  # broad: SDK error taxonomy varies by version
                if attempt == self.max_retries:
                    raise
                msg = str(exc).lower()
                cool = delay * 4 if ("429" in msg or "quota" in msg or "rate" in msg) else delay
                time.sleep(cool + random.uniform(0, delay))  # backoff + jitter
                delay = min(delay * 2, 60.0)
        raise RuntimeError("unreachable")

    def _tiers(self, max_attempts: int) -> list[tuple[str, int]]:
        """Model escalation ladder: (model, attempts), cheapest first.

        The primary model gets ``max_attempts`` draws; each configured escalate model
        then gets ``escalate_attempts`` more, tried in order only after every cheaper
        rung is exhausted without passing.
        """
        tiers = [(self.model, max(1, max_attempts))]
        if self.escalate_attempts > 0:
            tiers.extend((m, self.escalate_attempts) for m in self.escalate_models)
        return tiers

    def segment_file(
        self,
        src: Path,
        keep_gemini_size: bool = False,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        max_bleed: float = DEFAULT_MAX_BLEED,
        prompt: str | None = None,
    ) -> SegmentResult:
        """Segment ``src`` with a judged, escalating retry loop; return the chosen draft.

        Each tier's model is drawn in turn (see :meth:`_tiers`); every draft is scored
        with :func:`bleed_fraction` against the original photo. The first draft at or
        under ``max_bleed`` is returned immediately (cheapest model that clears the bar
        wins), otherwise the best-scoring draft across all tiers is returned with
        ``passed=False`` so the caller can flag it for review.

        Pass ``keep_gemini_size=True`` to keep the model's raw output resolution. If the
        original can't be decoded to judge against, the first draft is accepted as-is.
        """
        import cv2
        import numpy as np

        original = cv2.imread(str(src), cv2.IMREAD_COLOR)
        src_bytes, mime = src.read_bytes(), mime_for(src)

        best: SegmentResult | None = None
        draws = 0
        for model, n_attempts in self._tiers(max_attempts):
            for _ in range(n_attempts):
                draws += 1
                raw = self.segment(src_bytes, mime, model=model, prompt=prompt)
                png = raw if keep_gemini_size else conform_to_original(raw, src)

                if original is None:  # nothing to judge against — take the first draft
                    return SegmentResult(png, float("nan"), True, draws, model)

                purple = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
                bleed = bleed_fraction(original, purple) if purple is not None else 1.0
                if bleed <= max_bleed:
                    return SegmentResult(png, bleed, True, draws, model)
                if best is None or bleed < best.bleed:
                    best = SegmentResult(png, bleed, False, draws, model)

        # Nothing passed: return the least-bad draft, but report the *total* draws spent
        # (best.attempts holds only the index at which it was found).
        return best._replace(attempts=draws)  # type: ignore[union-attr]  # loop ran >=1 time


def purple_path_for(purple_dir: Path, src: Path) -> Path:
    """Output PNG path for a source image (always .png, lossless key colour)."""
    return purple_dir / f"{src.stem}.png"


def segment_dir(
    input: str,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    model: str | None = None,
    keep_gemini_size: bool = False,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_bleed: float = DEFAULT_MAX_BLEED,
    temperature: float | None = None,
    escalate_models: str | list[str] | None = None,
    escalate_attempts: int = DEFAULT_ESCALATE_ATTEMPTS,
    workers: int = 1,
) -> int:
    """Stage 1 over one input dir: write ``purple/<stem>.png`` for each image.

    Judged + retried + escalating per :meth:`GeminiSpotSegmenter.segment_file`. Returns a
    process exit code (non-zero if any image errored).

    ``workers`` > 1 runs that many Gemini calls concurrently in a thread pool. Each image is
    an independent network call writing its own PNG (no shared state; the genai client is
    thread-safe and the per-call path already backs off on 429/quota), so this is safe. Order
    of the progress lines is then non-deterministic. ``workers=1`` keeps the serial path.
    """
    input_dir = resolve_input_dir(input)
    purple_dir = purple_dir_for(input_dir)
    purple_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if limit is not None:
        images = images[:limit]
    if not images:
        logger.error(f"error: no images found in {input_dir}")
        return 1

    # Pre-filter the work so skips don't occupy a worker (and the count is right up front).
    todo = [src for src in images
            if overwrite or not purple_path_for(purple_dir, src).exists()]
    skipped = len(images) - len(todo)

    segmenter = GeminiSpotSegmenter(model=model, temperature=temperature,
                                    escalate_models=escalate_models,
                                    escalate_attempts=escalate_attempts)
    total = len(todo)
    logger.info(f"processing images in dir {input_dir}")
    logger.info(f"  purple model: {segmenter.model}")
    logger.info(f"  to do: {total}   already have purple (skipped): {skipped}"
          + (f"   workers: {workers}" if workers > 1 else ""))
    failures = 0
    flagged = 0

    def _do(src: Path) -> "SegmentResult":
        res = segmenter.segment_file(
            src, keep_gemini_size=keep_gemini_size,
            max_attempts=max_attempts, max_bleed=max_bleed,
        )
        purple_path_for(purple_dir, src).write_bytes(res.png)
        return res

    def _report(i: int, src: Path, res, exc) -> None:
        nonlocal failures, flagged
        if exc is not None:
            failures += 1
            logger.error(f"[{i}/{total}] {src.name}  ERROR: {exc}")
            return
        if not res.passed:
            flagged += 1
        tag = "ok" if res.passed else "FLAGGED (bleed)"
        logger.info(f"[{i}/{total}] {src.name} -> "
              f"{purple_path_for(purple_dir, src).name} "
              f"[bleed={res.bleed:.2f}, {res.attempts} draw(s), {res.model}, {tag}]")

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_do, src): src for src in todo}
            for i, fut in enumerate(as_completed(futs), start=1):
                src = futs[fut]
                try:
                    _report(i, src, fut.result(), None)
                except Exception as exc:
                    _report(i, src, None, exc)
    else:
        for i, src in enumerate(todo, start=1):
            try:
                _report(i, src, _do(src), None)
            except Exception as exc:
                _report(i, src, None, exc)

    done = total - failures
    logger.info(f"done: {done}/{total} purple images in {purple_dir}"
          + (f", {flagged} flagged for bleed" if flagged else "")
          + (f" ({failures} failed)" if failures else ""))
    return 1 if failures else 0
