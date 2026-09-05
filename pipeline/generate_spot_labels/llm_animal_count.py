#!/usr/bin/env python3
"""Stage 0 — HOW MANY SALAMANDERS ARE IN THIS FRAME?

Everything downstream assumes one animal per photo. Stage 1 paints "the salamander's" spots,
stage 1b paints "the salamander's" body, and the filename supplies ONE identity label for the
result. When a frame holds two animals that assumption breaks in two different ways, and it is
worth naming them separately because they do different damage:

**Overlapping.** The body fill spans both animals, so the mask comes back as a Y and stage 1b's
centre line tracks one animal while a whole second body hangs off it unaccounted for. Both
animals' spots are then extracted, binned against the wrong axis, and stored under a single
identity — so some spots carry the wrong animal's label and every metric computed against them
is quietly wrong. ``ca_14_5`` is the confirmed example.

**Separate.** ``solidify()`` keeps only the largest blob, so the body mask is a clean single
animal — but stage 1 paints every yellow spot in the frame, so the *other* animal's spots are
still extracted and land outside the mask, and the photo's single filename label is now
ambiguous about which animal it names.

Why a model and not a pixel metric
----------------------------------
Two heuristics over artefacts already on disk were measured against the corpus and both are too
weak to use as a filter:

* ``solidity`` × spot-count (the ``data_hygiene`` worklist) ranks ``ca_14_5`` #1 of 1265 — and
  the next eleven are all single animals with splayed legs. Solidity cannot tell a leg from a
  second salamander; both just make the mask non-convex.
* Depth-weighted **off-axis mass** (body-mask area beyond 1.35x the body's own half-width from
  the midline) is sharper — ``ca_14_5`` separates from the pack — but its next eleven are all
  single animals in a tight C-curl, where the two arms of the curl sit far from the midline for
  entirely innocent reasons.

Both are also structurally blind to the *separate* case, because the mask they read has already
had the second animal discarded. Counting animals is a perception task, which is the one thing
the model in this pipeline is reliably good at — the same reasoning that moved stage 1b from
"draw me two flank curves" to "paint the region".

The call
--------
A **vision+text** model (``gemini-3.5-flash`` by default — the judge's model, not the image
model), one billed call per image, no drawing. It answers with JSON:

    {"n_salamanders": 2, "n_heads_visible": 2, "n_tails_visible": 2,
     "overlapping": true, "confidence": "high",
     "reasoning": "two distinct heads; the upper animal's trunk crosses the lower one's back"}

``n_heads_visible`` / ``n_tails_visible`` are asked for on purpose. Counting *animals* invites
a snap judgement, and the failure this has to survive is a single salamander curled into a
horseshoe, which "looks like two" at a glance. Counting heads forces the answer to rest on
evidence that a curled single animal cannot fake, and it makes a wrong answer legible afterwards
(one head and two tails means the model saw the curl, not a second animal).

Images are downscaled to ``--max-dim`` (1024 px) before sending. Counting animals does not need
12 MP, and the corpus is 4032x3024.

Resumable and capped, like every billed stage here: an image that already has
``animals/<stem>.json`` is never paid for twice, ``--limit`` hard-caps the call count, and
``--dry-run`` prints the plan and the estimate without touching the network.

    pixi run extract-spot-labels count --input all_sasa_norm --dry-run
    pixi run extract-spot-labels count --input all_sasa_norm --limit 25
    pixi run extract-spot-labels count --input all_sasa_norm --workers 6
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from . import animal_screen
from ._common import getenv, list_images, load_dotenv, resolve_input_dir
from .llm_anatomy import _is_permanent
from .runlog import DONE, FAILED, SKIPPED, RunLog

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

DEFAULT_MODEL = "gemini-3.5-flash"      # vision+text, NOT an image model — this stage draws nothing
DEFAULT_MAX_DIM = 1024                  # px; a 12 MP photo is pure waste for counting animals
DEFAULT_MAX_RETRIES = 4
DEFAULT_BASE_DELAY = 2.0

CONFIDENCES = ("high", "medium", "low")


class FatalCountError(RuntimeError):
    """The counting model is misconfigured (bad model id, bad key) — retrying cannot help."""


PROMPT = """\
You are a careful zoological image annotator. Count the fire salamanders (Salamandra
salamandra: black skin, discrete yellow or orange spots) in the attached photo.

Count an animal if ANY part of it is visible, including one that is partly out of frame, partly
buried in leaf litter, or lying underneath another animal.

Count ONLY salamanders. These photos are field records, so they routinely contain things that
are not animals and must never be counted: a gloved hand, bare fingers, a pen, a ruler, a tape
measure, a coin, a bottle cap, a phone, a boot, twigs, roots, wet leaves, and the animal's own
shadow or its reflection in water.

THE MISTAKE TO AVOID: a single salamander curled into a C, an S or a horseshoe looks like two
animals at a glance, because the trunk and the tail lie side by side. Before answering "2",
find TWO SEPARATE HEADS. One head means one animal, however tangled the body looks. Two tails
with one head is a curled single animal, not a pair.

Work through it in this order:
1. Count the heads you can actually see (a snout with eyes).
2. Count the tail tips you can actually see.
3. Decide how many animals that implies, allowing that a head or a tail may be hidden.
4. If there is more than one, say whether their bodies TOUCH OR OVERLAP in the image, or
   whether they are fully separate with visible background between them.

Reply with ONLY this JSON object and nothing else:

{"n_salamanders": <integer>,
 "n_heads_visible": <integer>,
 "n_tails_visible": <integer>,
 "overlapping": <true|false>,
 "confidence": "<high|medium|low>",
 "reasoning": "<one sentence naming the evidence you counted>"}

Rules for the fields:
- "n_salamanders" is your best single answer. Use 0 if there is no salamander in the frame.
- "overlapping" must be false when "n_salamanders" is 0 or 1.
- "confidence" is "low" if a second animal is possible but you cannot confirm a second head.
"""


class AnimalCount(NamedTuple):
    """One image's answer. ``n < 0`` means the reply could not be parsed."""
    n_salamanders: int
    n_heads_visible: int
    n_tails_visible: int
    overlapping: bool
    confidence: str
    reasoning: str
    model: str = ""
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.n_salamanders >= 0

    @property
    def multi(self) -> bool:
        return self.n_salamanders > 1

    @property
    def usable(self) -> bool:
        """Exactly one salamander — the assumption the rest of the pipeline is built on."""
        return self.n_salamanders == 1

    def as_dict(self) -> dict:
        return {"n_salamanders": self.n_salamanders,
                "n_heads_visible": self.n_heads_visible,
                "n_tails_visible": self.n_tails_visible,
                "overlapping": self.overlapping,
                "confidence": self.confidence,
                "reasoning": self.reasoning,
                "model": self.model}

    @classmethod
    def from_dict(cls, d: dict) -> "AnimalCount":
        return cls(n_salamanders=int(d.get("n_salamanders", -1)),
                   n_heads_visible=int(d.get("n_heads_visible", -1)),
                   n_tails_visible=int(d.get("n_tails_visible", -1)),
                   overlapping=bool(d.get("overlapping", False)),
                   confidence=str(d.get("confidence", "")),
                   reasoning=str(d.get("reasoning", "")),
                   model=str(d.get("model", "")))

    @classmethod
    def unparsed(cls, why: str, model: str = "", raw: str = "") -> "AnimalCount":
        return cls(-1, -1, -1, False, "", why, model, raw)


def parse_count(text: str, model: str = "") -> AnimalCount:
    """The model's JSON reply -> an :class:`AnimalCount`.

    Tolerant of the wrapper the model sometimes puts around the object (a ```json fence, a
    sentence of preamble), because that is a formatting slip, not a wrong answer — the same
    forgiveness :func:`~.llm_anatomy.parse_verdict` extends to the judge.
    """
    if not text:
        return AnimalCount.unparsed("empty reply", model)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return AnimalCount.unparsed("no JSON object in the reply", model, text[:400])
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return AnimalCount.unparsed(f"undecodable JSON ({exc})", model, text[:400])

    def as_int(key: str) -> int:
        v = d.get(key)
        try:
            return int(v)
        except (TypeError, ValueError):
            return -1

    n = as_int("n_salamanders")
    if n < 0:
        return AnimalCount.unparsed("no usable n_salamanders field", model, text[:400])
    conf = str(d.get("confidence", "")).strip().lower()
    return AnimalCount(
        n_salamanders=n,
        n_heads_visible=as_int("n_heads_visible"),
        n_tails_visible=as_int("n_tails_visible"),
        # A model that answers 1 and then "overlapping": true has contradicted itself; the count
        # is the field being asked for, so it wins.
        overlapping=bool(d.get("overlapping", False)) and n > 1,
        confidence=conf if conf in CONFIDENCES else "",
        reasoning=str(d.get("reasoning", "")).strip(),
        model=model,
        raw=text[:400],
    )


def _downscale(path: Path, max_dim: int) -> bytes:
    """The photo, shrunk to ``max_dim`` on its long side, as PNG bytes.

    Re-encoded rather than passed through so the payload is predictable whatever the source
    format is; the counting task is unaffected by the resample.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not read {path}")
    h, w = img.shape[:2]
    s = min(1.0, float(max_dim) / float(max(h, w)))
    if s < 1.0:
        img = cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)),
                         interpolation=cv2.INTER_AREA)
    return cv2.imencode(".png", img)[1].tobytes()


class GeminiAnimalCounter:
    """Thin wrapper over google-genai for the one counting call."""

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 max_dim: int = DEFAULT_MAX_DIM,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 base_delay: float = DEFAULT_BASE_DELAY,
                 temperature: float | None = None):
        dotenv = load_dotenv()
        self.api_key = api_key or getenv("GEMINI_API_KEY", "", dotenv)
        self.model = model or getenv("GEMINI_COUNT_MODEL", DEFAULT_MODEL, dotenv)
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is empty (set it in salamander_spotter/.env)")
        self.max_dim = max_dim
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.temperature = temperature
        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    def preflight(self) -> None:
        """Prove the model answers BEFORE any billed image is sent.

        One tiny call. Without it a bad model id is only discovered image-by-image, and across a
        1200-image dir that is 1200 failures for one configuration mistake. A model appearing in
        ``models.list()`` is not proof your key may call it.
        """
        from google.genai import types
        probe = cv2.imencode(".png", np.zeros((16, 16, 3), np.uint8))[1].tobytes()
        try:
            self._client.models.generate_content(
                model=self.model,
                contents=[types.Part.from_bytes(data=probe, mime_type="image/png"),
                          'Reply with only this JSON: {"ok": true}'])
        except Exception as exc:
            if _is_permanent(exc):
                raise FatalCountError(
                    f"the counting model {self.model!r} is not usable with this API key:\n"
                    f"    {exc}\n"
                    f"  Pick one your key can actually call, e.g.:\n"
                    f"    --count-model gemini-3.5-flash        (recommended)\n"
                    f"    --count-model gemini-flash-latest\n"
                    f"  or set GEMINI_COUNT_MODEL in .env."
                ) from exc
            raise                                   # transient: let the caller see it

    def count(self, src: Path) -> AnimalCount:
        """Count the salamanders in one photo. Retries transient failures with backoff."""
        from google.genai import types

        payload = _downscale(src, self.max_dim)
        contents = [types.Part.from_bytes(data=payload, mime_type="image/png"), PROMPT]
        config = types.GenerateContentConfig(response_mime_type="application/json",
                                             **({"temperature": self.temperature}
                                                if self.temperature is not None else {}))
        delay = self.base_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config)
                return parse_count(getattr(resp, "text", "") or "", self.model)
            except Exception as exc:
                if _is_permanent(exc):
                    raise FatalCountError(
                        f"the counting model {self.model!r} stopped being usable mid-run:\n"
                        f"    {exc}\n  Re-run with --count-model <a model your key can call>."
                    ) from exc
                if attempt == self.max_retries:
                    return AnimalCount.unparsed(f"unreachable after {attempt} tries ({exc})",
                                                self.model)
                msg = str(exc).lower()
                cool = delay * 4 if ("429" in msg or "quota" in msg or "rate" in msg) else delay
                time.sleep(cool + random.uniform(0, delay))
                delay = min(delay * 2, 60.0)
        return AnimalCount.unparsed("retries exhausted", self.model)


# --- the two-layer decision -------------------------------------------------
DECIDED_ALGORITHM = "algorithm"
DECIDED_GEMINI = "gemini"


class Decision(NamedTuple):
    """One image's final answer, and which layer produced it.

    ``single_salamander`` is the field the rest of the pipeline reads. It is ``None`` only when
    the model was asked and its reply could not be parsed — an explicit "unknown" rather than a
    guess in either direction, because both guesses are harmful: ``True`` silently readmits a
    poisoned frame, ``False`` silently discards a good one.
    """
    stem: str
    single_salamander: bool | None
    decided_by: str
    screen: animal_screen.Screen | None = None
    count: AnimalCount | None = None

    @property
    def needs_review(self) -> bool:
        return self.single_salamander is not True

    def as_dict(self) -> dict:
        d: dict = {"single_salamander": self.single_salamander,
                   "decided_by": self.decided_by}
        if self.screen is not None:
            d["screen"] = self.screen.as_dict()
        if self.count is not None:
            d.update(self.count.as_dict())
        return d

    @classmethod
    def from_dict(cls, stem: str, d: dict) -> "Decision":
        single = d.get("single_salamander")
        scr = d.get("screen") or {}
        screen = (animal_screen.Screen(stem, scr.get("verdict", ""),
                                       float(scr.get("offaxis", -1)),
                                       float(scr.get("stray_spot_gap", -1)),
                                       scr.get("reason", ""),
                                       float(scr.get("spots_outside_frac", 0)))
                  if scr else None)
        count = AnimalCount.from_dict(d) if d.get("n_salamanders") is not None else None
        return cls(stem, single if single is None else bool(single),
                   str(d.get("decided_by", "")), screen, count)


def from_screen(screen: animal_screen.Screen) -> Decision:
    """The free layer cleared it. No call was made and none will be."""
    return Decision(screen.stem, True, DECIDED_ALGORITHM, screen, None)


def from_count(stem: str, count: AnimalCount,
               screen: animal_screen.Screen | None = None) -> Decision:
    """The model answered. Anything other than exactly one animal is not a single salamander."""
    return Decision(stem, (count.usable if count.ok else None), DECIDED_GEMINI, screen, count)


# --- disk -------------------------------------------------------------------
def animals_dir_for(input_dir: Path) -> Path:
    return input_dir / "animals"


def is_synthetic(stem: str) -> bool:
    """``aj_1_g0`` -> True. The ``_g<k>`` naming convention from ``emb-gen-augment``.

    A synthetic view is a re-rendering of ONE source photo of ONE animal, so its count is known
    without asking: it is whatever the source was. Counting them is a third of the corpus paid
    for twice, which is why they are skipped by default.
    """
    tail = stem.rsplit("_", 1)[-1]
    return len(tail) > 1 and tail[0] == "g" and tail[1:].isdigit()


def load_decision(animals_dir: Path, stem: str) -> Decision | None:
    """The stored answer for one image, or None if it was never decided."""
    p = animals_dir / f"{stem}.json"
    if not p.is_file():
        return None
    try:
        return Decision.from_dict(stem, json.loads(p.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return None


def write_decision(out_dir: Path, dec: Decision) -> None:
    (out_dir / f"{dec.stem}.json").write_text(
        json.dumps(dec.as_dict(), indent=2), encoding="utf-8")


def count_file_to_disk(counter: GeminiAnimalCounter, src: Path, out_dir: Path,
                       screen: animal_screen.Screen | None = None,
                       log: RunLog | None = None) -> Decision:
    """Layer 2 for one image: ask the model, persist ``<stem>.json``.

    Only raises :class:`FatalCountError` — a transient failure comes back as an unparsed
    :class:`AnimalCount`, which becomes ``single_salamander: null``.
    """
    res = counter.count(src)
    dec = from_count(src.stem, res, screen)
    write_decision(out_dir, dec)
    if log:
        log.write({"task": "count", "image": src.name, "draw_model": None,
                   "judge_model": res.model,
                   "judgment": {"single_salamander": dec.single_salamander,
                                "n_salamanders": res.n_salamanders,
                                "n_heads_visible": res.n_heads_visible,
                                "n_tails_visible": res.n_tails_visible,
                                "overlapping": res.overlapping,
                                "confidence": res.confidence},
                   "result": DONE if res.ok else FAILED,
                   "feedback": res.reasoning,
                   "billed_calls": 1})
    return dec


CSV_FIELDS = ("image", "single_salamander", "decided_by", "n_salamanders", "n_heads_visible",
              "n_tails_visible", "overlapping", "confidence", "screen_offaxis",
              "screen_gap", "reasoning", "model")


def write_summary(out_dir: Path, results: dict[str, Decision]) -> tuple[Path, Path]:
    """-> (the full CSV, the not-single CSV).

    Two files on purpose. ``animal_counts.csv`` is the record of every decision, including the
    ones the free layer settled for nothing; ``flagged_multi.csv`` is a single-column list in
    exactly the shape ``--rewrite`` eats, so the frames that need re-doing (or excluding) pipe
    straight back into the pipeline.
    """
    full = out_dir / "animal_counts.csv"
    with open(full, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_FIELDS)
        for name in sorted(results):
            d = results[name]
            c, s = d.count, d.screen
            w.writerow([name, d.single_salamander, d.decided_by,
                        c.n_salamanders if c else "", c.n_heads_visible if c else "",
                        c.n_tails_visible if c else "", c.overlapping if c else "",
                        c.confidence if c else "",
                        f"{s.offaxis:.5f}" if s else "", f"{s.gap:.5f}" if s else "",
                        (c.reasoning if c else (s.reason if s else "")),
                        c.model if c else ""])

    flagged = out_dir / "flagged_multi.csv"
    bad = sorted(n for n, d in results.items() if d.needs_review)
    flagged.write_text(
        "# frames that are NOT a confirmed single salamander (more than one, none, or the\n"
        "# model's reply was unparseable). CONFIRM BY EYE before excluding anything.\n"
        "# Feeds --rewrite, or a WHERE clause on the packaged dataset.\n"
        + ("\n".join(bad) + "\n" if bad else ""), encoding="utf-8")
    return full, flagged


def _report(results: dict[str, Decision], full: Path, flagged: Path) -> None:
    n = len(results)
    if not n:
        return
    by_alg = [d for d in results.values() if d.decided_by == DECIDED_ALGORITHM]
    by_llm = [d for d in results.values() if d.decided_by == DECIDED_GEMINI]
    single = [d for d in results.values() if d.single_salamander is True]
    multi = [(k, d) for k, d in results.items() if d.count is not None and d.count.multi]
    none_ = [(k, d) for k, d in results.items()
             if d.count is not None and d.count.ok and d.count.n_salamanders == 0]
    unknown = [k for k, d in results.items() if d.single_salamander is None]

    logger.info(f"{'=' * 78}\n decided {n} image(s)\n{'=' * 78}")
    logger.info(f"  single_salamander = True : {len(single):>5}  ({len(single) / n:.1%})")
    logger.info(f"      settled free by layer 1: {len(by_alg):>5}")
    logger.info(f"      confirmed by Gemini    : {len(single) - len(by_alg):>5}")
    logger.info(f"  single_salamander = False: {len(multi) + len(none_):>5}"
          f"   ({len(multi)} multi-animal, {len(none_)} with no salamander)")
    if unknown:
        logger.info(f"  single_salamander = null : {len(unknown):>5}   (unparseable reply — re-run "
              f"those with --overwrite)")
    logger.info(f"  billed calls made: {len(by_llm)} of {n} frames "
          f"({len(by_llm) / n:.1%}) — layer 1 saved {len(by_alg)}")

    if multi:
        overlap = [k for k, d in multi if d.count.overlapping]
        logger.info(f"  of the {len(multi)} multi-animal frames, {len(overlap)} OVERLAP "
              f"(fused body mask — the poisoned case) and {len(multi) - len(overlap)} are "
              f"separate")
        low = [k for k, d in multi if d.count.confidence == "low"]
        if low:
            logger.info(f"  {len(low)} of them {'is' if len(low) == 1 else 'are'} 'low' confidence "
                  f"— look at {'that' if len(low) == 1 else 'those'} first")
        logger.info("  flagged frames:")
        for name, d in sorted(multi)[:25]:
            c = d.count
            logger.info(f"    {name:<22}n={c.n_salamanders} heads={c.n_heads_visible} "
                  f"tails={c.n_tails_visible} "
                  f"{'overlapping' if c.overlapping else 'separate':<12}{c.confidence}")
        if len(multi) > 25:
            logger.info(f"    ... and {len(multi) - 25} more in {full.name}")

    logger.info(f"  wrote {full}\n        {flagged}")
    logger.info("  These are CANDIDATES, not a verdict — confirm by eye before excluding a frame.")


def count_dir(input: str, *, overwrite: bool = False, limit: int | None = None,
              model: str | None = None, max_dim: int = DEFAULT_MAX_DIM,
              temperature: float | None = None, workers: int = 1,
              include_synthetic: bool = False, screen: bool = True,
              offaxis_cutoff: float = animal_screen.DEFAULT_OFFAXIS_CUTOFF,
              gap_cutoff: float = animal_screen.DEFAULT_GAP_CUTOFF,
              refresh_screen: bool = False,
              dry_run: bool = False, run_log: RunLog | None = None) -> int:
    """Stage 0 over one input dir, as a two-layer cascade. Returns a process exit code.

    Layer 1 (:mod:`animal_screen`) is free and runs first; every frame it clears is written
    straight out with ``single_salamander: true`` and never reaches the network. Layer 2 asks
    Gemini, and only about what layer 1 deferred. ``--no-screen`` skips layer 1 and bills every
    frame, which is the honest thing to do when you want the model's opinion on all of them.

    ``workers`` > 1 sends that many layer-2 calls concurrently; each is an independent call
    writing its own JSON and appending to the shared (lock-guarded) run log.
    """
    input_dir = resolve_input_dir(input)
    out_dir = animals_dir_for(input_dir)

    images = list_images(input_dir)
    synth = 0
    if not include_synthetic:
        keep = [p for p in images if not is_synthetic(p.stem)]
        synth = len(images) - len(keep)
        images = keep
    # --limit caps the images CONSIDERED, so it must be applied after the synthetic filter --
    # otherwise `--limit 25` on a dir whose first 25 files are all synthetic does nothing.
    if limit is not None:
        images = images[:limit]
    if not images:
        logger.error(f"error: no images found in {input_dir}")
        return 1

    todo = [src for src in images
            if overwrite or not (out_dir / f"{src.stem}.json").is_file()]
    skipped = len(images) - len(todo)

    logger.info(f"counting salamanders per frame in {input_dir}")
    logger.info(f"  model : {model or getenv('GEMINI_COUNT_MODEL', DEFAULT_MODEL)}")
    logger.info(f"  images: {len(images)}   already decided (skipped): {skipped}"
          + (f"   synthetic _g views (skipped): {synth}" if synth else "")
          + (f"   workers: {workers}" if workers > 1 else ""))

    # --- layer 1: the free screen -------------------------------------------
    screens: dict[str, animal_screen.Screen] = {}
    if screen and todo:
        screens = animal_screen.screen_dir(
            input_dir, [s.stem for s in todo],
            offaxis_cutoff=offaxis_cutoff, gap_cutoff=gap_cutoff,
            refresh=refresh_screen, cache=out_dir / "screen.csv")
        animal_screen.report(screens, offaxis_cutoff=offaxis_cutoff,
                             gap_cutoff=gap_cutoff)

    cleared = [s for s in todo
               if screens.get(s.stem) is not None and screens[s.stem].single]
    cleared_set = set(cleared)
    ask = [s for s in todo if s not in cleared_set]

    logger.info(f"  BILLED: {len(ask)} call(s), one per deferred frame, sent at <= {max_dim}px"
          + (f"   ({len(cleared)} cleared free by layer 1)" if cleared else ""))

    if dry_run:
        logger.info("DRY RUN — nothing sent, nothing written. Re-run without --dry-run.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Decision] = {}
    for src in images:                              # re-read the skipped ones so the CSV is whole
        if src not in todo:
            prior = load_decision(out_dir, src.stem)
            if prior is not None:
                results[src.name] = prior

    # Layer 1's verdicts are written before a single call is made, so a run that dies during
    # layer 2 still leaves every free answer on disk.
    for src in cleared:
        dec = from_screen(screens[src.stem])
        write_decision(out_dir, dec)
        results[src.name] = dec
        if run_log:
            run_log.write({"task": "count", "image": src.name,
                           "judgment": {"single_salamander": True},
                           "result": SKIPPED, "feedback": screens[src.stem].reason,
                           "billed_calls": 0})

    total, failures = len(ask), 0
    counter = log = None
    if ask:
        counter = GeminiAnimalCounter(model=model, max_dim=max_dim, temperature=temperature)
        counter.preflight()
        log = (run_log.child(task="count", input=input_dir.name) if run_log
               else RunLog(out_dir / "pipeline_log.jsonl", task="count", input=input_dir.name))
        logger.info(f"  log   : {log.path}")

    def _do(src: Path) -> Decision:
        return count_file_to_disk(counter, src, out_dir, screens.get(src.stem), log=log)

    def _note(i: int, src: Path, dec: Decision | None, exc: Exception | None) -> None:
        nonlocal failures
        if exc is not None:
            failures += 1
            logger.error(f"[{i}/{total}] {src.name}  ERROR: {exc}")
            return
        results[src.name] = dec
        c = dec.count
        flag = "" if dec.single_salamander is True else "   <-- CHECK"
        logger.info(f"[{i}/{total}] {src.name:<24}single={dec.single_salamander!s:<6}"
              f"n={c.n_salamanders if c else '?'} "
              f"heads={c.n_heads_visible if c else '?'} tails={c.n_tails_visible if c else '?'} "
              f"{('overlapping' if c and c.overlapping else ''):<12}"
              f"{c.confidence if c else ''}{flag}")

    try:
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_do, src): src for src in ask}
                for i, fut in enumerate(as_completed(futs), start=1):
                    src = futs[fut]
                    try:
                        _note(i, src, fut.result(), None)
                    except FatalCountError:
                        raise
                    except Exception as exc:
                        _note(i, src, None, exc)
        else:
            for i, src in enumerate(ask, start=1):
                try:
                    _note(i, src, _do(src), None)
                except FatalCountError:
                    raise
                except Exception as exc:
                    _note(i, src, None, exc)
    finally:
        # Whatever was decided before the run stopped is still worth having on disk.
        if results:
            full, flagged = write_summary(out_dir, results)
            _report(results, full, flagged)

    if failures:
        logger.error(f"{failures} image(s) errored")
    if log:
        logger.info(f"billed calls: {log.billed}   log -> {log.path}")
    return 1 if failures else 0
