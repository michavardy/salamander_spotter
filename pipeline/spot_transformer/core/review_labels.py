"""The hand labels from the preprocessing web app, as one loader.

``scripts/tools/preprocessing_review.py`` writes everything a reviewer does into a single store::

    artifacts/preprocessing/<dataset>/review.json

and four different things in that file are supervision the modelling side wants:

* **interesting spots** — which spots "draw the eye" (``distinctiveness.py``, and the auxiliary
  gate loss in ``aggregator_e2e_strict``). ~3k positives over 477 images.
* **match verdicts** — accept/reject on individual spot-to-spot correspondences, split by whether
  the machine proposed it or the human drew it. This is the only ground truth in the repo for the
  matcher's own output, and it says the current matcher runs at ~80% precision / <=46% recall.
* **image decisions** — accept/reject per photo with a reason slug, plus ``train_ok``/``eval_ok``.
* **reprocess queue** — re-extraction requests (axis corrections and the like).

Everything went through ``interesting_spots.json`` before, which the app still exports for
backwards compatibility — but that export is a *union snapshot* taken whenever someone remembers to
run ``preprocess-export``, and it only ever carried the interesting-spot clicks. Reading the store
directly means (a) no export step between labelling and training, and (b) the other three label
kinds stop being invisible. The legacy file remains the fallback so a checkout without a review
store still trains.

Spot keys are DB ``spot_id`` values throughout (the UI resolves a click to a spot through the mask
table), so they align directly with ``ImageSet.spot_ids`` — no positional guessing anywhere.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

REPO_ROOT = Path(__file__).resolve().parents[3]

def images_folder_for(dataset: str) -> str:
    """``all_sasa_norm_2026_23_07`` -> ``all_sasa_norm``; mirrors ``preprocessing_review.folder_for``.

    Only strips a trailing ``_YYYY_DD_MM``, so a dataset named without a date is returned unchanged
    rather than losing its last three segments.
    """
    parts = dataset.rsplit("_", 3)
    return parts[0] if len(parts) == 4 and parts[1].isdigit() else dataset


def review_path(dataset: str) -> Path:
    return REPO_ROOT / "artifacts" / "preprocessing" / dataset / "review.json"


def legacy_path(dataset: str) -> Path:
    return (REPO_ROOT / "artifacts" / "interesting_spots" / images_folder_for(dataset)
            / "interesting_spots.json")


@lru_cache(maxsize=4)
def load_review(dataset: str) -> dict | None:
    """The raw review store, or ``None`` when this dataset was never reviewed.

    Cached: every regime loads it, it is ~1.3 MB of JSON, and it does not change mid-run.
    """
    p = review_path(dataset)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def review_summary(dataset: str) -> str:
    """One line naming what supervision is actually available — printed by every consumer so a run
    log always records which labels produced it."""
    rev = load_review(dataset)
    if rev is None:
        return f"review store: MISSING ({review_path(dataset)}) — falling back to legacy export"
    c = rev.get("counts", {})
    return (f"review store rev {rev.get('rev', '?')} @ {rev.get('updated_at', '?')}: "
            f"{c.get('interesting_spots', 0)} interesting spots over "
            f"{len(rev.get('interesting_spots') or {})} images · "
            f"{c.get('matches_accepted', 0)}/{c.get('matches_accepted', 0) + c.get('matches_rejected', 0)}"
            f" matches accepted · {c.get('images_accepted', 0)} accepted / "
            f"{c.get('images_rejected', 0)} rejected images")


# ----------------------------------------------------------------------------- interesting spots
def interesting_spots(dataset: str, *, allow_legacy: bool = True) -> dict[str, set[int]]:
    """``salamander_id -> {spot_id, ...}`` the human tagged as interesting.

    Prefers the review store (the live superset — it seeds itself from the legacy export on first
    run and only grows). Falls back to ``interesting_spots.json`` when there is no store, so this is
    a drop-in for the old ``distinctiveness.load_interesting``.
    """
    rev = load_review(dataset)
    if rev is not None and rev.get("interesting_spots"):
        return {sid: {int(i) for i in ids} for sid, ids in rev["interesting_spots"].items() if ids}
    if not allow_legacy:
        return {}
    p = legacy_path(dataset)
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {sid: {int(i) for i in ids} for sid, ids in raw.get("labels", {}).items() if ids}


# ----------------------------------------------------------------------------- match verdicts
MATCH_COLUMNS = ["match_id", "individual", "sid_a", "spot_a", "sid_b", "spot_b",
                 "accepted", "proposed_by", "method", "pair_kind"]


def _is_synth(sid: str) -> bool:
    return sid.split("_")[-1].startswith("g")


def _pair_kind(a: str, b: str) -> str:
    n = _is_synth(a) + _is_synth(b)
    return ("real-real", "real-synth", "synth-synth")[n]


def match_verdicts(dataset: str) -> pd.DataFrame:
    """One row per human-adjudicated spot correspondence, as ``MATCH_COLUMNS``.

    A match group can hold more than two members (the same physical spot seen in three photos), so
    groups are expanded to all member pairs — a 3-member group is three correspondences, which is
    what it asserts.

    ``accepted`` is the label. Read ``proposed_by`` before using it:

    * ``algorithm`` + accepted/rejected  — the machine proposed, the human judged. **This is the
      calibration set**: 342 accepted / 85 rejected, i.e. the matcher's own precision.
    * ``human`` + accepted               — the human drew a correspondence the machine missed.
      These are recall failures, not negatives, and they are NOT interchangeable with the accepted
      algorithm rows: they were selected *because* the machine did not find them, so training a
      scorer on them as positives fits the machine's blind spot rather than the task.

    ``pair_kind`` matters just as much: only ~9% of these are ``real-real``. The rest involve a
    Gemini-generated view, where "same spot" is guaranteed by construction rather than observed,
    so any conclusion drawn from them is a conclusion about the generator (see results.md #11).
    """
    rev = load_review(dataset)
    if rev is None:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    rows = []
    for ind, entry in (rev.get("individuals") or {}).items():
        for m in entry.get("matches") or []:
            members = m.get("members") or []
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    if a.get("image") is None or b.get("image") is None:
                        continue
                    rows.append({
                        "match_id": m.get("match_id"),
                        "individual": ind,
                        "sid_a": a["image"], "spot_a": int(a["spot"]),
                        "sid_b": b["image"], "spot_b": int(b["spot"]),
                        "accepted": bool(m.get("accepted")),
                        "proposed_by": m.get("proposed_by") or "unknown",
                        "method": m.get("method") or "unknown",
                        "pair_kind": _pair_kind(a["image"], b["image"]),
                    })
    return pd.DataFrame(rows, columns=MATCH_COLUMNS)


def verdict_edges(dataset: str, *, proposed_by: str | None = "algorithm",
                  pair_kind: str | None = None) -> pd.DataFrame:
    """:func:`match_verdicts` filtered to one supervision regime.

    Defaults to the algorithm-proposed rows, because those are the only ones with *both* labels —
    a set of human-drawn matches contains no negatives and cannot train or calibrate a threshold.
    """
    df = match_verdicts(dataset)
    if len(df) and proposed_by:
        df = df[df["proposed_by"] == proposed_by]
    if len(df) and pair_kind:
        df = df[df["pair_kind"] == pair_kind]
    return df.reset_index(drop=True)


def matcher_operating_point(dataset: str) -> dict:
    """What the verdicts say about the CURRENT matcher, with no model fitted.

    ``recall`` is an upper bound: a true correspondence the reviewer never got around to drawing
    counts here as a success for the machine, so the real number can only be lower.
    """
    df = match_verdicts(dataset)
    if not len(df):
        return {}
    alg = df[df["proposed_by"] == "algorithm"]
    hum = df[df["proposed_by"] == "human"]
    tp = int(alg["accepted"].sum())
    fp = int((~alg["accepted"]).sum())
    missed = int(hum["accepted"].sum())
    return dict(
        n_edges=len(df), n_algorithm=len(alg), n_human=len(hum),
        precision=tp / max(tp + fp, 1), recall_upper=tp / max(tp + missed, 1),
        tp=tp, fp=fp, missed=missed,
        n_individuals=int(df["individual"].nunique()),
        pair_kinds=df["pair_kind"].value_counts().to_dict(),
    )


# ----------------------------------------------------------------------------- pair verdicts
PAIR_COLUMNS = ["id", "query", "matched", "correct", "score", "sources", "verdict", "reasons",
                "disagrees_with_label", "comment"]


def pair_verdicts(dataset: str) -> pd.DataFrame:
    """One row per human-judged candidate PAIR, from ``scripts/tools/pair_review.py``.

    A different altitude of label from :func:`match_verdicts`, and deliberately so. That one is a
    verdict per spot↔spot *edge*, and results.md #44 measured it as unlearnable over the current
    representation (accepted edges cosine 0.645 ± 0.154, rejected 0.650 ± 0.167, AUROC 0.484). This
    one is a verdict per *pair of animals* — ``match`` / ``different`` / ``unsure`` — which is the
    question #44 says the reviewer is actually answering, and it is the calibration set the
    three-way ``match / new / abstain`` head needs (open threads #6).

    Rows with ``verdict`` null are pairs rendered but never judged; filter them out before treating
    the frame as labels. ``unsure`` is a real class (the abstain target), not a missing value.

    **``sources`` is not decoration — read it before comparing scores across rows.** ``pair_review
    generate`` samples by score BAND (top / mid / bottom) per method, so this frame is a stratified
    sample, not a random one. Any statistic over ``score`` that ignores the band is measuring the
    sampling design as much as the matcher: ``s[0]["band"]`` is the stratum each pair came from.

    ``disagrees_with_label`` is the useful side effect: True where the filenames and the reviewer
    conflict, i.e. a suspected identity error (open threads #4). Null where no verdict, or
    ``unsure`` — "nobody looked" is not "the human agrees".
    """
    p = REPO_ROOT / "artifacts" / "pair_review" / dataset / "review.json"
    if not p.is_file():
        return pd.DataFrame(columns=PAIR_COLUMNS)
    rows = json.loads(p.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)
    for c in PAIR_COLUMNS:                      # a store written before verdicts existed has none
        if c not in df.columns:
            df[c] = None
    return df[PAIR_COLUMNS]


def pair_verdict_summary(dataset: str) -> dict:
    """Counts of what the pair review holds — judged, by class, and the label conflicts."""
    df = pair_verdicts(dataset)
    judged = df[df["verdict"].notna()] if len(df) else df
    if not len(judged):
        return {}
    return dict(
        n_pairs=len(df), n_judged=len(judged),
        by_verdict=judged["verdict"].value_counts().to_dict(),
        n_conflicts=int(judged["disagrees_with_label"].fillna(False).astype(bool).sum()),
        reasons=pd.Series([r for rs in judged["reasons"] if rs for r in rs]
                          ).value_counts().to_dict(),
    )


# ----------------------------------------------------------------------------- image decisions
def image_decisions(dataset: str) -> pd.DataFrame:
    """One row per REVIEWED photo: ``sid, label, decision, reasons, train_ok, eval_ok`` + the
    ``quality_at_review`` metrics snapshot the UI showed at decision time.

    Only reviewed photos appear. An unreviewed photo is *unknown*, not *accepted* — the caller must
    decide what to do with the ~91% of the dataset that was never opened, and silently treating them
    as one class or the other is how a gate that looks like a quality filter becomes a sampling bias.
    """
    rev = load_review(dataset)
    if rev is None:
        return pd.DataFrame(columns=["sid", "label", "decision", "reasons", "train_ok", "eval_ok"])
    rows = []
    for ind, entry in (rev.get("individuals") or {}).items():
        for sid, img in (entry.get("images") or {}).items():
            dec = img.get("decision")
            if dec in (None, "unreviewed"):
                continue
            q = img.get("quality_at_review") or {}
            rows.append({
                "sid": sid, "label": ind, "decision": dec,
                "reasons": tuple(img.get("reasons") or ()),
                "train_ok": bool(img.get("train_ok")), "eval_ok": bool(img.get("eval_ok")),
                "is_synth": bool(img.get("is_synthetic")), "n_spots": img.get("n_spots"),
                **{k: q.get(k) for k in ("overall_quality", "blur_quality", "spot_extraction_quality",
                                         "body_extraction_quality", "lighting_quality",
                                         "spots_outside_frac")},
            })
    return pd.DataFrame(rows)


def accepted_images(dataset: str) -> set[str]:
    """``salamander_id`` of every photo a human accepted."""
    df = image_decisions(dataset)
    return set(df.loc[df["decision"] == "accept", "sid"]) if len(df) else set()


def rejected_images(dataset: str) -> set[str]:
    df = image_decisions(dataset)
    return set(df.loc[df["decision"] == "reject", "sid"]) if len(df) else set()


# ----------------------------------------------------------------------------- reprocess queue
def reprocess_queue(dataset: str) -> pd.DataFrame:
    """Outstanding re-extraction requests (``image, action, note, status``).

    Small but pointed: the axis corrections in here are bug reports against the component the
    simulator named as the largest remaining source of difficulty (results.md #36).
    """
    rev = load_review(dataset)
    if rev is None:
        return pd.DataFrame(columns=["image", "action", "note", "status"])
    return pd.DataFrame(rev.get("reprocess_queue") or [],
                        columns=["image", "action", "note", "status", "requested_at",
                                 "completed_at"])


# ----------------------------------------------------------------------------- inspection
def main():
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import data as d                                                   # noqa: PLC0415

    ds = d.dataset_name
    logger.info(f"dataset {ds}")
    logger.info(" " + review_summary(ds))

    ints = interesting_spots(ds)
    legacy = interesting_spots(ds, allow_legacy=True) if load_review(ds) is None else None
    logger.info(f"interesting spots: {sum(len(v) for v in ints.values())} positives over "
          f"{len(ints)} images")
    if legacy is None:
        p = legacy_path(ds)
        if p.is_file():
            old = json.loads(p.read_text(encoding="utf-8")).get("labels", {})
            logger.info(f"  legacy export has {sum(len(v) for v in old.values())} over {len(old)} images"
                  f"  -> {len(ints) - len(old):+d} images by reading the store directly")

    op = matcher_operating_point(ds)
    if op:
        logger.info(f"match verdicts: {op['n_edges']} correspondences over {op['n_individuals']} individuals")
        logger.info(f"  algorithm-proposed {op['n_algorithm']}  ->  precision {op['precision']:.3f}"
              f"  ({op['tp']} accepted / {op['fp']} rejected)")
        logger.info(f"  human-drawn        {op['n_human']}  ->  recall <= {op['recall_upper']:.3f}"
              f"  (upper bound: undrawn misses count as successes)")
        logger.info(f"  pair kinds: {op['pair_kinds']}")
        cal = verdict_edges(ds)
        logger.info(f"  calibration set (algorithm-proposed, both labels): {len(cal)} rows, "
              f"{int(cal['accepted'].sum())} pos / {int((~cal['accepted']).sum())} neg")

    pv = pair_verdict_summary(ds)
    if pv:
        logger.info(f"pair verdicts: {pv['n_judged']} judged of {pv['n_pairs']} rendered pairs")
        logger.info(f"  by verdict: {pv['by_verdict']}")
        logger.info(f"  reasons   : {pv['reasons'] or '(none given)'}")
        logger.info(f"  conflicts : {pv['n_conflicts']} pairs where the reviewer disagrees with the "
              f"filename label -> suspected identity errors")

    imgs = image_decisions(ds)
    if len(imgs):
        logger.info(f"image decisions: {len(imgs)} reviewed  "
              f"({int((imgs['decision'] == 'accept').sum())} accept / "
              f"{int((imgs['decision'] == 'reject').sum())} reject)")
        from collections import Counter
        logger.info(f"  reasons: {dict(Counter(r for rs in imgs['reasons'] for r in rs))}")

    q = reprocess_queue(ds)
    if len(q):
        logger.info(f"reprocess queue: {len(q)} requested")
        for r in q.itertuples(index=False):
            logger.info(f"  {r.image:<12} {r.action:<16} {r.note or ''}")


if __name__ == "__main__":
    main()
