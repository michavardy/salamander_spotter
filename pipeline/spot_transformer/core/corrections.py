"""Known-bad data, in one file the whole pipeline reads.

Three faults have been confirmed by hand and have nowhere to live: two animals filed under one
name, one animal filed under two names, and photos whose extraction is simply wrong. Each is a
fact about the DATA, so fixing it in a model, a sweep or a notebook fixes it in one place and
leaves it broken in the other twenty. This module puts all of them in
``datasets/<dataset>/corrections.json`` and applies them at the single point every consumer goes
through — :func:`data.get_image_sets`.

Two operations, deliberately different in kind:

**merge** — one animal, two names. ``{"keep": "ac_3", "alias": "ca_62"}`` makes every photo named
``ca_62_*`` carry the label ``ac_3``. This does not touch a filename or the database; it rewrites
the label at load. Until it is applied, the matcher is scored **wrong for being right**: finding
``ca_62`` when the query is ``ac_3`` counts as an error on every metric the project quotes.

**exclude** — the photo should not be used at all. Unlike ``MIN_QUALITY`` (which gates the SCORED
side only, because a blurry photo is a bad query but perfectly good training data — results.md
#14), an excluded photo is dropped from **training too**. That is the whole distinction: a bad
photograph is weak evidence, whereas a frame holding two animals has spots labelled with the wrong
animal's identity, so learning from it teaches the wrong thing.

Every entry carries a ``source`` and a ``note``, because an unexplained exclusion is
indistinguishable from cherry-picking. ``CORRECTIONS=0`` disables the whole file for an A/B.

    pixi run corrections                # what is in effect, and does it still match the data
    pixi run corrections --check        # exit 1 if any entry names something that does not exist
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

ENABLED = os.environ.get("CORRECTIONS", "1") != "0"
_cache: dict[str, dict] = {}


def corrections_path(dataset: str) -> Path:
    """``datasets/<dataset>/corrections.json`` — versioned with the data it describes."""
    import data as d                                                     # noqa: PLC0415
    return d.REPO_ROOT / "datasets" / dataset / "corrections.json"


def load(dataset: str) -> dict:
    """The raw file, or an empty skeleton when there is none (the no-corrections case is normal)."""
    if not ENABLED:
        return {"dataset": dataset, "merge": [], "exclude": [], "_disabled": True}
    if dataset in _cache:
        return _cache[dataset]
    p = corrections_path(dataset)
    if not p.is_file():
        out = {"dataset": dataset, "merge": [], "exclude": []}
    else:
        out = json.loads(p.read_text(encoding="utf-8"))
        out.setdefault("merge", [])
        out.setdefault("exclude", [])
    _cache[dataset] = out
    return out


def alias_map(dataset: str) -> dict[str, str]:
    """``{alias_label: keep_label}``, chains resolved so A->B->C all land on C.

    Chains are resolved rather than rejected because duplicates arrive pairwise: confirming
    ``a/b`` and later ``b/c`` is three photos of one animal, discovered in two sittings.
    """
    direct = {}
    for m in load(dataset)["merge"]:
        alias, keep = str(m["alias"]), str(m["keep"])
        if alias == keep:
            continue
        direct[alias] = keep
    out = {}
    for alias in direct:
        seen, cur = {alias}, direct[alias]
        while cur in direct and cur not in seen:
            seen.add(cur)
            cur = direct[cur]
        out[alias] = cur
    return out


def label_for(label: str, dataset: str) -> str:
    """The label a photo should carry once merges are applied."""
    return alias_map(dataset).get(label, label)


def excluded_sids(dataset: str) -> set[str]:
    """Photo ids to drop entirely. An entry naming an ``individual`` expands at load time."""
    return {str(e["sid"]) for e in load(dataset)["exclude"] if e.get("sid")}


def excluded_individuals(dataset: str) -> set[str]:
    return {str(e["individual"]) for e in load(dataset)["exclude"] if e.get("individual")}


def is_excluded(sid: str, label: str, dataset: str) -> bool:
    return sid in excluded_sids(dataset) or label in excluded_individuals(dataset)


def reasons(dataset: str) -> dict[str, int]:
    """``{reason: n}`` over the exclude list — what is being dropped, and for what."""
    out: dict[str, int] = {}
    for e in load(dataset)["exclude"]:
        out[str(e.get("reason", "unspecified"))] = out.get(str(e.get("reason", "unspecified")), 0) + 1
    return out


def describe(dataset: str) -> str:
    """One line for a run log, so no result is ever printed without saying what was corrected."""
    if not ENABLED:
        return " corrections DISABLED (CORRECTIONS=0)"
    am, ex = alias_map(dataset), load(dataset)["exclude"]
    if not am and not ex:
        return f" corrections: none on file for {dataset}"
    bits = []
    if am:
        bits.append(f"{len(am)} identities merged")
    if ex:
        rs = ", ".join(f"{k} {v}" for k, v in sorted(reasons(dataset).items()))
        bits.append(f"{len(ex)} photos excluded ({rs})")
    return " corrections: " + " · ".join(bits)


# ------------------------------------------------------------------------------------ validation
def validate(dataset: str) -> list[str]:
    """Problems with the file itself, as human-readable strings. Empty list = clean.

    Checks every name against the database, because the failure mode of a hand-edited file is a
    typo that silently does nothing — an alias for a label that does not exist merges no photos
    and raises no error.
    """
    import duckdb                                                        # noqa: PLC0415
    import data as d                                                     # noqa: PLC0415
    problems = []
    doc = load(dataset)

    con = duckdb.connect(str(d.DB_PATH), read_only=True)
    try:
        sids = {r[0] for r in con.execute("SELECT salamander_id FROM images").fetchall()}
    finally:
        con.close()
    labels = {"_".join(s.split("_")[:2]) for s in sids}

    seen_alias = {}
    for m in doc["merge"]:
        a, k = str(m.get("alias", "")), str(m.get("keep", ""))
        if not a or not k:
            problems.append(f"merge entry missing keep/alias: {m}")
            continue
        if a not in labels:
            problems.append(f"merge alias '{a}' matches no individual in the data")
        if k not in labels:
            problems.append(f"merge keep '{k}' matches no individual in the data")
        if a == k:
            problems.append(f"merge '{a}' -> itself does nothing")
        if a in seen_alias and seen_alias[a] != k:
            problems.append(f"'{a}' is merged into both '{seen_alias[a]}' and '{k}'")
        seen_alias[a] = k
        if not m.get("source"):
            problems.append(f"merge {a}->{k} has no 'source' (who confirmed it?)")

    for e in doc["exclude"]:
        sid, ind = e.get("sid"), e.get("individual")
        if not sid and not ind:
            problems.append(f"exclude entry names neither sid nor individual: {e}")
            continue
        if sid and sid not in sids:
            problems.append(f"exclude sid '{sid}' is not a photo in the data")
        if ind and ind not in labels:
            problems.append(f"exclude individual '{ind}' matches no individual in the data")
        if not e.get("reason"):
            problems.append(f"exclude {sid or ind} has no 'reason'")
    return problems


def main() -> None:
    import argparse
    import data as d                                                     # noqa: PLC0415

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="What data corrections are in effect")
    ap.add_argument("--check", action="store_true", help="exit 1 if the file has problems")
    args = ap.parse_args()

    ds = d.dataset_name
    p = corrections_path(ds)
    logger.info(f"dataset {ds}")
    logger.info(f"file    {p}{'' if p.is_file() else '   (does not exist — nothing is being corrected)'}")

    doc = load(ds)
    am = alias_map(ds)
    if am:
        logger.info(f" MERGES — {len(am)} names folded into another ({len(set(am.values()))} animals affected)")
        for m in doc["merge"]:
            logger.info(f"   {m['alias']:>10s} -> {m['keep']:<10s}  {m.get('source', '?')}"
                  f"   {m.get('note', '')}")
    else:
        logger.info(" MERGES — none")

    ex = doc["exclude"]
    if ex:
        logger.info(f" EXCLUSIONS — {len(ex)} photos dropped from training AND scoring")
        for k, v in sorted(reasons(ds).items()):
            logger.info(f"   {v:>4d}  {k}")
    else:
        logger.info(" EXCLUSIONS — none")

    problems = validate(ds)
    if problems:
        logger.warning(f" PROBLEMS — {len(problems)}")
        for s in problems:
            logger.warning(f"   ! {s}")
    else:
        logger.info(" PROBLEMS — none; every name in the file matches something in the data")

    if args.check and problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
