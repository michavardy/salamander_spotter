#!/usr/bin/env python3
"""Stage 1b, mask mode — ask Gemini to PAINT THE BODY, and derive the rest ourselves.

One call, one job: fill the salamander's trunk and tail with flat magenta, and drop a green dot
on the snout and a red one on the tail tip. The outlines, the centre line and the true tips are
then computed from that mask by :mod:`body_mask`, with no model involved.

Why this exists (the numbers that killed the old stage 1b)
----------------------------------------------------------
The two-outline method asked the model to draw the body's two flanks as separate curves and
took their mean. Over a 56-image run it was rejected by its own judge **73 %** of the time, and
climbing the model ladder did not rescue it::

    gemini-2.5-flash-image    93 % rejected
    gemini-3.1-flash-image    87 % rejected
    gemini-3-pro-image        89 % rejected     <- the expensive rung bought nothing

    follows_edges     failed 74 % of drafts
    goes_head_to_tail        40 %
    dots_at_tips             30 %
    opposite_sides           18 %

The rejected drafts say why. The model finds the body BOUNDARY accurately and then refuses to
decompose it: again and again it drew one closed loop around the whole animal in *both* colours.
Tracing a region's edge is something it can do; splitting that edge into a left half and a right
half and keeping them straight is not. Filling a region, on the other hand, is the very trick
stage 1 already leans on for the spots — so we ask for the fill and do the geometry in numpy.

What that buys
--------------
* **Three of the four failure modes cannot occur.** There are no edges to follow, no two lines
  to keep apart, and the tips come from the mask rather than the model's aim (:func:`snap_tip`
  moves a dot from the middle of the head to the actual snout).
* **The LLM judge is gone.** It existed because no pixel metric can see whether a *drawn* line
  bisects a body. Once the body is a region the bisection is COMPUTED — right by construction —
  and the only open question ("is this a plausible salamander mask?") is arithmetic. The gates
  in :func:`~.body_mask.check` answer it for nothing.
* **The retry feedback is free.** A rejected draft is still re-drawn with a written note about
  what was wrong, exactly as before — but the note now comes from the gates rather than from a
  billed judge call, so the correction loop costs one image call instead of two.

Per image: **1 draw + 0 judge calls** on the happy path, against the old 3 + 3.

Output (unchanged, so stage 2 needs no changes)::

    <input_dir>/anatomy/<stem>.png    QA render: mask tinted, outline + centre line drawn
    <input_dir>/anatomy/<stem>.json   {head, tail_tip, midline, left, right, source, ...}

Pure orchestration + the network call; the maths lives in :mod:`body_mask`.
"""
from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from . import body_mask as bm
from ._common import getenv, list_images, load_dotenv, resolve_input_dir
from .llm_anatomy import Anatomy, NoImageError, _is_permanent, anatomy_dir_for
from .llm_spot_segmentation import conform_to_original, mime_for, parse_model_list
from .runlog import ACCEPTED, REGENERATE, RunLog

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

# The primary model is deliberately NOT the cheap one, and not GEMINI_MODEL either. Measured
# over the first 108 drafts of a real run, first-shot pass rates were::
#
#     gemini-2.5-flash-image    31 %      <- of its failures: 10 echoed the photo back
#     gemini-3.1-flash-image    89 %         unpainted, 13 painted the wrong region,
#                                            10 painted well but left the dots off
#
# With 2.5-flash leading, the ladder was not rescuing a hard task — it was rescuing a bad first
# draft, at 2.2 draws per image. Leading with 3.1-flash costs more per call and less per IMAGE.
#
# Read from GEMINI_FILL_MODEL, not GEMINI_MODEL: stage 1 (paint the spots) and stage 1b (paint
# the body) are different jobs, and .env's GEMINI_MODEL — set for stage 1 — would otherwise
# silently drag this back down to the 31 % model.
DEFAULT_MODEL = "gemini-3.1-flash-image"
DEFAULT_MAX_ATTEMPTS = 2        # draws on the primary model before escalating
DEFAULT_ESCALATE_MODELS = "gemini-3-pro-image"
DEFAULT_ESCALATE_ATTEMPTS = 1   # draws per escalate rung
DEFAULT_IMAGE_RETRIES = 3       # re-asks when the model answers in prose instead of drawing

PROMPT = f"""\
System Instruction / Role: Act as a precise image annotation engine. You are NOT re-imagining
the photo — you copy every pixel through unchanged, and paint on top of it.

Input: The attached photo of a fire salamander is the absolute baseline template. Keep the same
framing, the same animal, the same background.

Task — make ALL THREE of these marks. Every one of the three is REQUIRED; an annotation that is
missing any of them is useless and will be rejected.

MARK 1 (REQUIRED) — a filled circle in pure green ({bm.HEAD_HEX}) on the animal's HEAD, at the
front end: the snout. Make it clearly visible, but smaller than the width of the tail.

MARK 2 (REQUIRED) — a filled circle in pure red ({bm.TAIL_HEX}) on the very TIP OF THE TAIL: the
rear-most point of the animal. Same size as the green one.

The green and red circles are what tell us which end of the animal is the head and which is the
tail. Draw them FIRST, before you paint anything, and do not leave them out. They do not have to
be perfectly placed — near the right end is enough — but they MUST BOTH BE THERE, they must sit
ON the animal, and each must be a solid blob of its colour, not an outline.

MARK 3 (REQUIRED) — PAINT THE BODY. Fill the salamander's TRUNK AND TAIL with flat, opaque
magenta ({bm.FILL_HEX}). Paint right over everything: the yellow spots, the black skin, the eyes
— all of it becomes solid magenta, with no texture and no shading showing through. Leave the two
circles from marks 1 and 2 visible on top of the magenta.

   - The painted shape must cover the animal COMPLETELY, from the tip of the snout to the tip
     of the tail, edge to edge across its width. No unpainted slivers of salamander.
   - The painted shape must stop EXACTLY at the animal's outline, where its body meets the
     ground. No magenta spilling onto the background.
   - DO NOT PAINT THE LEGS, FEET OR TOES. Stop at the base of each limb, where the limb meets
     the body, and let the painted edge run straight on along the trunk's own outline — as if
     the animal had no limbs. The head and the tail ARE painted; only the four limbs are not.

All three marks must be flat, fully saturated, unshaded colour. Use each colour ONLY for its own
mark. Do not add text, arrows, labels, outlines or shading.

Before you answer, check your image: is there a green circle on it? is there a red circle on it?
is the body solid magenta? If any of the three is missing, add it.

Output the annotated IMAGE. Do not reply with a text description instead of an image."""

RETRY_PREFIX = """\
Your PREVIOUS attempt at this annotation was REJECTED. The second image attached is your
rejected attempt; the first image is the clean original you must annotate again.

What was wrong with it:
{feedback}

Paint the original again, fixing exactly that. Do not repeat the same mistake. The instructions
are unchanged:

"""

# What each failed gate should tell the model to do differently. The old pipeline paid a judge
# to write this sentence; here it falls out of arithmetic that already ran.
ADVICE = {
    "mask_found": "You did not paint the body at all. The salamander's trunk and tail must be "
                  "filled with solid, flat magenta.",
    "area_sane": "The magenta region is the wrong size for the animal — you either painted only "
                 "a fragment of it, or the fill escaped onto the background. Paint the whole "
                 "body and nothing but the body.",
    "dots_found": "The green and/or red dot is missing, or is not on the animal. Put a small "
                  "green dot on the snout and a small red dot on the tip of the tail.",
    "axis_long": "The painted region is far too small to be the whole animal. Paint the entire "
                 "body, from the snout all the way to the tip of the tail.",
    "line_inside": "The painted region is broken — it is not one single connected body. Paint "
                   "the salamander as ONE unbroken shape from snout to tail tip, with no gaps "
                   "and no separate islands of magenta.",
}


def advice_for(body: bm.Body) -> str:
    """The failed gates -> the note that goes back into the prompt. Costs nothing."""
    notes = [ADVICE[k] for k in body.failures if k in ADVICE]
    return " ".join(notes) if notes else (body.reason or "The painted body was not usable.")


def to_anatomy(body: bm.Body) -> Anatomy:
    """A :class:`~.body_mask.Body` -> the :class:`~.llm_anatomy.Anatomy` stage 2 already eats.

    Same fields, so the ``body_axis`` table, the binning and ``load_anatomy`` are all untouched
    — ``left``/``right`` are simply the two halves of the mask's own contour now, and
    ``judged_ok`` is the verdict of the free gates rather than of a billed judge.
    """
    return Anatomy(head=body.head, tail_tip=body.tail_tip, midline=body.midline,
                   left=body.left, right=body.right,
                   source=body.source, judged_ok=body.passed if body.checks else None,
                   judge_feedback=body.reason)


class FillResult(NamedTuple):
    png: bytes                  # the QA render (derived geometry drawn on the ORIGINAL photo)
    body: bm.Body
    passed: bool
    attempts: int
    model: str
    mask_png: bytes = b""       # the body mask itself: 1-bit, in the ORIGINAL's pixel grid

    @property
    def anatomy(self) -> Anatomy:
        return to_anatomy(self.body)


class GeminiBodyPainter:
    """Thin wrapper over google-genai for the body-fill call."""

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 max_retries: int = 5, base_delay: float = 2.0,
                 temperature: float | None = None,
                 escalate_models: str | list[str] | None = None,
                 escalate_attempts: int = DEFAULT_ESCALATE_ATTEMPTS):
        dotenv = load_dotenv()
        self.api_key = api_key or getenv("GEMINI_API_KEY", "", dotenv)
        self.model = model or getenv("GEMINI_FILL_MODEL", DEFAULT_MODEL, dotenv)
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is empty (set it in salamander_spotter/.env)")
        raw = (escalate_models if escalate_models is not None
               else getenv("GEMINI_FILL_ESCALATE_MODELS", DEFAULT_ESCALATE_MODELS, dotenv))
        self.escalate_models = [m for m in parse_model_list(raw) if m != self.model]
        self.escalate_attempts = escalate_attempts
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.temperature = temperature
        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    def tiers(self, max_attempts: int) -> list[tuple[str, int]]:
        rungs = [(self.model, max(1, max_attempts))]
        if self.escalate_attempts > 0:
            rungs.extend((m, self.escalate_attempts) for m in self.escalate_models)
        return rungs

    def ladder_str(self, max_attempts: int) -> str:
        return " -> ".join(f"{m} x{n}" for m, n in self.tiers(max_attempts))

    @staticmethod
    def _parts(resp) -> bytes | None:
        for cand in getattr(resp, "candidates", []) or []:
            for part in getattr(getattr(cand, "content", None), "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    return inline.data
        return None

    def paint(self, image_bytes: bytes, mime_type: str, model: str,
              prompt: str, prior_png: bytes | None = None) -> bytes:
        """One call -> the painted image. Raises NoImageError if it answered in prose."""
        from google.genai import types

        contents: list = [types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
        if prior_png:                       # let it SEE what it got wrong, not just read about it
            contents.append(types.Part.from_bytes(data=prior_png, mime_type="image/png"))
        contents.append(prompt)
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            **({"temperature": self.temperature} if self.temperature is not None else {}),
        )
        delay = self.base_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=model, contents=contents, config=config)
            except Exception as exc:
                if _is_permanent(exc) or attempt == self.max_retries:
                    raise
                msg = str(exc).lower()
                cool = delay * 4 if ("429" in msg or "quota" in msg or "rate" in msg) else delay
                time.sleep(cool + random.uniform(0, delay))
                delay = min(delay * 2, 60.0)
                continue
            png = self._parts(resp)
            if png is None:
                raise NoImageError(f"{model} returned no image part (it replied with text)")
            return png
        raise RuntimeError("unreachable")

    def paint_file(self, src: Path, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                   min_axis_frac: float = bm.DEFAULT_MIN_AXIS_FRAC,
                   image_retries: int = DEFAULT_IMAGE_RETRIES,
                   log: RunLog | None = None) -> FillResult:
        """Paint ``src``, re-painting with the gates' feedback until the mask is good.

        The loop is the one the old stage 1b had — draw, grade, feed the complaint AND the
        rejected image back in, escalate if a rung keeps failing — with the judge call deleted
        from the middle of it. Grading is :func:`~.body_mask.check`, which is free, so a retry
        costs exactly one image call.

        If no draft ever clears every gate the best one is kept (most gates passed, then longest
        axis) and returned with ``passed=False``: its spots still bin, they are just marked
        ``judged_ok=false``, exactly as a judge-rejected draft was.
        """
        src_bytes, mime = src.read_bytes(), mime_for(src)
        original = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if original is None:
            raise RuntimeError(f"could not read {src}")

        best: FillResult | None = None
        best_score = -1
        feedback = ""
        prior_png: bytes | None = None
        draws = 0

        for model, n_draws in self.tiers(max_attempts):
            for _ in range(n_draws):
                prompt = (RETRY_PREFIX.format(feedback=feedback) + PROMPT) if feedback else PROMPT

                png = None
                last: Exception | None = None
                for tri in range(1, max(1, image_retries) + 1):
                    try:
                        raw = self.paint(src_bytes, mime, model, prompt, prior_png)
                        png = conform_to_original(raw, src)
                        break
                    except NoImageError as exc:
                        logger.warning(f"  {model}: no image returned (try {tri}/{image_retries}) — "
                              "re-asking")
                        last = exc
                    except Exception as exc:
                        if _is_permanent(exc):
                            logger.warning(f"  warning: {model!r} is not usable with this key "
                                  f"({str(exc)[:70]}) — skipping this rung")
                            last = exc
                            break
                        raise
                total = sum(n for _, n in self.tiers(max_attempts))
                if png is None:
                    draws += 1
                    if log:
                        log.gated(image=src.name, attempt=draws, max_attempts=total,
                                  draw_model=model, checks={}, result=REGENERATE,
                                  anatomy=Anatomy(),
                                  note=f"{model} never returned an image ({last}) — escalating")
                    break                       # this rung will not draw: climb the ladder

                draws += 1
                painted = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
                if painted is None:
                    if log:
                        log.gated(image=src.name, attempt=draws, max_attempts=total,
                                  draw_model=model, checks={}, result=REGENERATE,
                                  anatomy=Anatomy(), note="the model returned an undecodable image")
                    continue

                body, mask = bm.extract_body(painted, min_axis_frac=min_axis_frac)
                qa = bm.overlay(original, body, mask)
                result = FillResult(cv2.imencode(".png", qa)[1].tobytes(), body, body.passed,
                                    draws, model, cv2.imencode(".png", mask)[1].tobytes())

                if log:
                    log.gated(image=src.name, attempt=draws, max_attempts=total,
                              draw_model=model, checks=body.checks,
                              result=ACCEPTED if body.passed else REGENERATE,
                              anatomy=to_anatomy(body),
                              note="" if body.passed else body.reason)
                if body.passed:
                    return result

                score = sum(body.checks.values())
                if score > best_score or (score == best_score and best is not None
                                          and body.length_px > best.body.length_px):
                    best, best_score = result, score
                feedback = advice_for(body)     # free — no judge call
                prior_png = png

        if best is None:
            return FillResult(b"", bm.Body(), False, max(draws, 1), self.model)
        return best._replace(attempts=draws)


# --- disk ------------------------------------------------------------------
def paint_file_to_disk(painter: GeminiBodyPainter, src: Path, out_dir: Path,
                       max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                       min_axis_frac: float = bm.DEFAULT_MIN_AXIS_FRAC,
                       image_retries: int = DEFAULT_IMAGE_RETRIES,
                       log: RunLog | None = None) -> FillResult:
    """One image: paint (gated + re-painted), persist the QA png, the mask, and the geometry json.

    The MASK is kept, not just the geometry derived from it. It is the expensive artefact on this
    stage — the one thing a billed call actually produced — and everything else here is a pure
    function of it, so keeping it means any later change to the maths (a different smoothing
    window, more bands, a real medial axis) can be re-derived over the whole dataset for FREE,
    without re-painting 614 images. It is also a body mask per photo in the original's pixel
    grid, which is exactly the (image, mask) pair a segmentation model wants for training.
    """
    res = painter.paint_file(src, max_attempts=max_attempts, min_axis_frac=min_axis_frac,
                             image_retries=image_retries, log=log)
    if res.png:
        (out_dir / f"{src.stem}.png").write_bytes(res.png)
    if res.mask_png:
        (out_dir / f"{src.stem}_mask.png").write_bytes(res.mask_png)
    (out_dir / f"{src.stem}.json").write_text(
        json.dumps(res.body.as_dict(), indent=2), encoding="utf-8")
    if log:
        log.summary(image=src.name, attempts=res.attempts, accepted=res.passed,
                    anatomy=res.anatomy)

    if not res.passed:
        if res.body.ok:
            logger.warning(f"  GIVING UP after {res.attempts} draw(s): gates never all passed "
                  f"[{', '.join(res.body.failures)}]. Keeping the best mask ({res.model}) — its "
                  f"spots WILL be binned but marked judged_ok=false")
        else:
            logger.warning(f"  GIVING UP after {res.attempts} draw(s): no usable body mask "
                  f"— spots will be unbinned")
    return res


def paint_dir(input: str, *, overwrite: bool = False, limit: int | None = None,
              model: str | None = None, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
              min_axis_frac: float = bm.DEFAULT_MIN_AXIS_FRAC,
              temperature: float | None = None,
              escalate_models: str | list[str] | None = None,
              escalate_attempts: int = DEFAULT_ESCALATE_ATTEMPTS,
              image_retries: int = DEFAULT_IMAGE_RETRIES,
              workers: int = 1,
              run_log: RunLog | None = None) -> int:
    """Stage 1b (mask mode) over one input dir. Returns a process exit code.

    ``workers`` > 1 paints that many images concurrently in a thread pool. Each image is an
    independent Gemini call writing its own ``<stem>.{png,json,_mask.png}`` and appending to
    the shared (lock-guarded) run log, so this is safe; ``workers=1`` keeps the serial path.
    """
    input_dir = resolve_input_dir(input)
    out_dir = anatomy_dir_for(input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if limit is not None:
        images = images[:limit]
    if not images:
        logger.error(f"error: no images found in {input_dir}")
        return 1

    # Pre-filter skips so a worker is never spent on an image that already has anatomy.
    todo = [src for src in images
            if overwrite or not (out_dir / f"{src.stem}.json").is_file()]
    skipped = len(images) - len(todo)

    painter = GeminiBodyPainter(model=model, temperature=temperature,
                                escalate_models=escalate_models,
                                escalate_attempts=escalate_attempts)
    log = (run_log.child(task="anatomy", input=input_dir.name) if run_log
           else RunLog(out_dir / "pipeline_log.jsonl", task="anatomy", input=input_dir.name))
    logger.info(f"painting body masks in dir {input_dir}")
    logger.info(f"  paint ladder: {painter.ladder_str(max_attempts)}")
    logger.info(f"  judge       : NONE — the mask is graded by free geometric gates")
    logger.info(f"  to do: {len(todo)}   already have anatomy (skipped): {skipped}"
          + (f"   workers: {workers}" if workers > 1 else ""))
    logger.info(f"  log         : {log.path}")

    total, failures, flagged = len(todo), 0, 0
    rejected: list[str] = []

    def _do(src: Path):
        return paint_file_to_disk(painter, src, out_dir, max_attempts=max_attempts,
                                  min_axis_frac=min_axis_frac,
                                  image_retries=image_retries, log=log)

    def _report(i: int, src: Path, res, exc) -> None:
        nonlocal failures, flagged
        if exc is not None:
            failures += 1
            logger.error(f"[{i}/{total}] {src.name}  ERROR: {exc}")
            return
        if not res.passed:
            flagged += 1
            rejected.append(src.name)

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
            logger.info(f"processing image {i} / {total}: {src.name}")
            try:
                _report(i, src, _do(src), None)
            except Exception as exc:
                _report(i, src, None, exc)

    if rejected:
        csv_path = out_dir / "flagged_axis.csv"
        csv_path.write_text(
            "# images whose body mask never cleared every gate; redo with --rewrite\n"
            + "\n".join(rejected) + "\n", encoding="utf-8")
        logger.info(f"flagged {len(rejected)} image(s) -> {csv_path}")

    done = total - failures
    logger.info(f"done: {done}/{total} body masks in {out_dir}"
          + (f", {flagged} failed the gates" if flagged else "")
          + (f" ({failures} errored)" if failures else ""))
    logger.info(f"billed calls: {log.billed}   log -> {log.path}")
    return 1 if failures else 0
