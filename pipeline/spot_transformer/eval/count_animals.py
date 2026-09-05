"""How many salamanders are actually in each photo? — ask the model that already labels them.

``ca_14_5`` is two animals overlapping. The body mask fused them, so its 49 "spots" belong to two
different individuals — which corrupts matching *and* the ground truth, since some of those spots
are filed under the wrong animal's identity.

The obvious next step was a geometric detector, and it failed completely. Calibrated on that one
confirmed case, ``solidity`` at the 0th percentile looked decisive; checked against the photographs,
two of its top three candidates (``af_2_1``, ``ri_1_2``) are single animals whose masks are non-
convex because the limbs are splayed or leaves cover part of the body. Six other candidate signals
were tested against one positive and two negatives and **none separated them** — ``ca_14_5`` sits at
unremarkable middle percentiles on every metric except solidity. The quality table describes a
single fused blob and cannot see inside it.

So: ask the vision model, which is already in this pipeline drawing the spots. One cheap call per
photo answers directly what six geometric proxies could not.

**This is a prevalence measurement first, a worklist second.** We have exactly one confirmed case.
Whether the right response is "exclude a handful of frames" or "build an instance-segmentation
front end" depends entirely on whether this is 1% or 20% of the dataset, and that number does not
exist yet. The script also cross-checks the counts against spot survival, because a fault that does
not predict worse matching is not worth acting on however common it is.

Resumable — every answer is cached, so a run that dies on a quota resumes for free and nothing is
re-billed.

    pixi run count-animals --dry-run          # ALWAYS first: the plan + billed-call estimate
    pixi run count-animals --limit 150        # a sample, enough to estimate prevalence
    pixi run count-animals                    # the whole dataset
    pixi run count-animals --report           # cross-check what is already cached, no calls
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval"):
    p = str(_ST / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, str(_ST.parents[0]))          # pipeline/ — for generate_spot_labels._common

import data as d                                     # noqa: E402
from aggregator import _norm                         # noqa: E402

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

IMAGE_DIR = "images/all_sasa_norm"
CACHE = d.REPO_ROOT / "artifacts" / "spot_transformer" / "multi_animal" / "counts.json"
DEFAULT_MODEL = "gemini-3.5-flash"                   # a counting question, not an image edit
MATCH_THR = 0.4

PROMPT = (
    "How many distinct fire salamanders (Salamandra salamandra) are visible in this photograph, "
    "including any that are only partly visible — an overlapping body, a tail, or a limb entering "
    "the frame counts as one animal.\n"
    "Ignore anything that is not a salamander: pens, coins, gloves, leaves, sticks, shadows.\n"
    'Reply with JSON only, no other text: {"count": <integer>, "note": "<up to 8 words>"}'
)


# ----------------------------------------------------------------------------- cache
def load_cache() -> dict:
    if CACHE.is_file():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    tmp.replace(CACHE)


# ----------------------------------------------------------------------------- the call
class Counter:
    """One cheap vision call per photo. Lazy client so ``--report`` and ``--dry-run`` need no key."""

    def __init__(self, model: str | None = None):
        from generate_spot_labels._common import getenv, load_dotenv     # noqa: PLC0415
        dot = load_dotenv()
        self.api_key = getenv("GEMINI_API_KEY", "", dot)
        self.model = model or getenv("GEMINI_COUNT_MODEL", DEFAULT_MODEL, dot)
        self._client = None
        if not self.api_key:
            raise SystemExit("GEMINI_API_KEY not set (.env) — needed for the counting calls.\n"
                             "  `--dry-run` and `--report` work without it.")

    @property
    def client(self):
        if self._client is None:
            from google import genai                                     # noqa: PLC0415
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def count(self, path: Path, retries: int = 3) -> dict:
        from google.genai import types                                   # noqa: PLC0415
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        parts = [types.Part.from_bytes(data=path.read_bytes(), mime_type=mime), PROMPT]
        last = ""
        for attempt in range(retries):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=parts)
                txt = (resp.text or "").strip()
                last = txt
                m = re.search(r'\{.*\}', txt, re.S)
                if m:
                    obj = json.loads(m.group(0))
                    return {"count": int(obj.get("count", -1)),
                            "note": str(obj.get("note", ""))[:80], "model": self.model}
                m = re.search(r'\d+', txt)                                # bare number fallback
                if m:
                    return {"count": int(m.group(0)), "note": "", "model": self.model}
            except Exception as exc:                                      # quota, transient, parse
                last = f"{type(exc).__name__}: {exc}"
                time.sleep(1.5 * (attempt + 1))
        # Recorded, never guessed: an unanswerable photo must not silently become "1".
        return {"count": -1, "note": f"FAILED {last[:60]}", "model": self.model}


# ----------------------------------------------------------------------------- cross-check
def survival_by_image() -> dict[str, float]:
    """``sid -> mean spot survival against its own siblings`` — what the counts are checked against.

    A photo containing two animals should match its own individual's other photos WORSE, because
    half its spots belong to somebody else. If the flagged photos match no worse, the fault is real
    but not worth spending effort on.
    """
    sets = [s for s in d.get_image_sets(d.get_spot_embeddings()) if not s.is_synth]
    by: dict[str, list] = {}
    for s in sets:
        if len(s.spots) >= 4:
            by.setdefault(s.label, []).append(s)
    out: dict[str, list[float]] = {}
    for lbl, ss in by.items():
        if len(ss) < 2:
            continue
        for A in ss:
            vals = []
            for B in ss:
                if A.sid == B.sid:
                    continue
                S = _norm(A.spots) @ _norm(B.spots).T
                ab, bb = S.argmax(1), S.argmax(0)
                n = sum(1 for i in range(len(S))
                        if bb[ab[i]] == i and float(S[i, ab[i]]) >= MATCH_THR)
                vals.append(n / len(A.spots))
            if vals:
                out[A.sid] = float(np.mean(vals))
    return out


def report(cache: dict) -> None:
    import duckdb
    con = duckdb.connect(str(d.DB_PATH), read_only=True)
    try:
        q = con.execute("SELECT salamander_id AS sid, n_spots, solidity, overall_quality "
                        "FROM image_quality").df().set_index("sid")
    finally:
        con.close()

    rows = [{"sid": k, **v} for k, v in cache.items() if v.get("count", -1) >= 0]
    if not rows:
        logger.warning(" nothing usable in the cache yet.")
        return
    df = pd.DataFrame(rows)
    for c in ("n_spots", "solidity", "overall_quality"):
        df[c] = df["sid"].map(q[c] if c in q.columns else {})

    n = len(df)
    multi = df[df["count"] >= 2]
    logger.info(f" counted {n} photos")
    logger.info(f"   {int((df['count'] == 1).sum()):>5}  one salamander      ({(df['count'] == 1).mean():.1%})")
    logger.info(f"   {len(multi):>5}  TWO OR MORE         ({len(multi) / n:.1%})   <- the prevalence number")
    logger.info(f"   {int((df['count'] == 0).sum()):>5}  none detected       ({(df['count'] == 0).mean():.1%})")
    n_fail = sum(1 for v in cache.values() if v.get("count", -1) < 0)
    if n_fail:
        logger.warning(f"   {n_fail:>5}  call failed (retry to fill these in)")

    if len(multi):
        logger.info(" multi-animal photos (up to 25):")
        logger.info(f" {'photo':>12}{'count':>7}{'spots':>7}{'solidity':>10}  note")
        for r in multi.sort_values("count", ascending=False).head(25).itertuples(index=False):
            sp = "" if not np.isfinite(r.n_spots or np.nan) else f"{int(r.n_spots)}"
            so = "" if not np.isfinite(r.solidity or np.nan) else f"{r.solidity:.3f}"
            logger.info(f" {r.sid:>12}{r.count:>7}{sp:>7}{so:>10}  {r.note}")

    # ---- does it predict worse matching? ----
    surv = survival_by_image()
    df["survival"] = df["sid"].map(surv)
    have = df.dropna(subset=["survival"])
    one = have[have["count"] == 1]["survival"]
    two = have[have["count"] >= 2]["survival"]
    logger.info(" does it matter? — spot survival against the photo's own siblings")
    if len(one) and len(two):
        logger.info(f"   one animal   : {one.mean():.3f}   (n={len(one)})")
        logger.info(f"   two or more  : {two.mean():.3f}   (n={len(two)})")
        logger.info(f"   difference   : {two.mean() - one.mean():+.3f}")
        if two.mean() < one.mean() - 0.03:
            logger.info("   => flagged photos DO match worse. Excluding them should help, and their")
            logger.info("      spots are also mislabelled in the ground truth.")
        else:
            logger.info("   => flagged photos do NOT match measurably worse. Real fault, low priority;")
            logger.info("      do not build an instance-segmentation stage on this evidence.")
    else:
        logger.warning(f"   not enough overlap yet (one={len(one)}, multi={len(two)}) — count more photos.")

    outdir = CACHE.parent
    df.to_csv(outdir / "animal_counts.csv", index=False)
    logger.info(f" wrote {outdir / 'animal_counts.csv'}")
    if len(multi):
        (outdir / "multi_animal.csv").write_text(
            "salamander_id\n" + "\n".join(sorted(multi["sid"])) + "\n", encoding="utf-8")
        logger.info(f" wrote {outdir / 'multi_animal.csv'}  ({len(multi)} photos to exclude or split)")


# ----------------------------------------------------------------------------- main
def main():
    import argparse
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="the plan + billed-call estimate, no calls")
    ap.add_argument("--report", action="store_true", help="analyse the cache, make no calls")
    ap.add_argument("--limit", type=int, default=0, help="count at most N new photos (0 = all)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--include-synth", action="store_true",
                    help="also count Gemini-generated views (they are single by construction)")
    args = ap.parse_args()

    cache = load_cache()
    imgs = [p for p in sorted((d.REPO_ROOT / IMAGE_DIR).glob("*"))
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not args.include_synth:
        imgs = [p for p in imgs if not p.stem.rsplit("_", 1)[-1].startswith("g")]
    todo = [p for p in imgs if p.stem not in cache or cache[p.stem].get("count", -1) < 0]
    if args.limit:
        todo = todo[:args.limit]

    logger.info(f"dataset {d.dataset_name}")
    logger.info(f" {len(imgs)} real photos · {len(cache)} already cached · {len(todo)} to call")

    if args.report:
        report(cache); return
    if args.dry_run:
        logger.info(" DRY RUN — no calls made.")
        logger.info(f"   would call the vision model {len(todo)} times ({DEFAULT_MODEL} unless --model)")
        logger.info(f"   one small image + ~60 tokens per call; resumable, so a partial run is never lost")
        logger.info(f"   cache -> {CACHE}")
        logger.info(" Recommended first pass:  pixi run count-animals --limit 150")
        logger.info(f" 150 photos is enough to estimate prevalence to about +/-3%, which is what decides")
        logger.info(f" whether this needs a detector or just an exclusion list.")
        if cache:
            report(cache)
        return

    counter = Counter(args.model)
    logger.info(f" model: {counter.model}")
    t0 = time.time()
    for i, p in enumerate(todo, 1):
        cache[p.stem] = counter.count(p)
        if i % 10 == 0 or i == len(todo):
            save_cache(cache)
            done = sum(1 for v in cache.values() if v.get("count", -1) >= 0)
            logger.info(f"  {i}/{len(todo)}  ({time.time() - t0:.0f}s)  cached={done}")
    save_cache(cache)
    report(cache)


if __name__ == "__main__":
    main()
