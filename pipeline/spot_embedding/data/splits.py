"""Leakage-safe k-fold CV splits for open-set re-identification.

The grouping unit is the identity **label** (``individual`` mode) or the source **code**
(``session`` mode — keeps all individuals of one shoot together so shared background can't
leak). Folds partition the groups; per fold:

* **train**  — labels reserved for training the model (later phases; no images enrolled).
* **eval**   — the held-out labels, split into gallery + queries:
    - a *multi-photo* individual contributes all-but-one photo to the **gallery** and one
      held-out photo as a **closed query** (its label IS in the gallery);
    - a *singleton* individual becomes a **novel / open query** (label NOT in the gallery);
    - *zero-spot* images are dropped from every role.

This gives every metric something to chew on: closed queries → rank-k / mAP, gallery×query
pairs → verification, closed-vs-open max-similarity → open-set detection. By construction
``train`` and ``eval`` label sets are disjoint, and open-query labels never appear in the
gallery.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .._common import is_synthetic, save_json
from .spot_store import SpotSet


@dataclass
class EvalFold:
    fold: int
    mode: str
    train_labels: list[str]
    gallery_ids: list[str]        # enrolled photos
    query_closed_ids: list[str]   # queries whose label IS in the gallery
    query_open_ids: list[str]     # novel queries whose label is NOT in the gallery


def make_folds(
    spotsets: list[SpotSet],
    *,
    k: int = 5,
    mode: str = "individual",
    seed: int = 0,
    hold_out: str = "last",
) -> list[EvalFold]:
    """Build ``k`` CV folds. ``mode`` is ``individual`` or ``session``.

    ``hold_out`` picks which photo of a multi-photo individual becomes the closed query
    (``last`` / ``first`` by sorted id — deterministic).
    """
    if mode not in ("individual", "session"):
        raise ValueError(f"mode must be 'individual' or 'session', got {mode!r}")
    if k < 2:
        raise ValueError("k must be >= 2")

    live = [ss for ss in spotsets if not ss.is_empty]

    # images per identity label, id-sorted for determinism
    ids_by_label: dict[str, list[str]] = {}
    label_group: dict[str, str] = {}   # label -> grouping key (label itself, or code)
    for ss in live:
        ids_by_label.setdefault(ss.label, []).append(ss.salamander_id)
        label_group[ss.label] = ss.label if mode == "individual" else ss.code
    for ids in ids_by_label.values():
        ids.sort()

    # partition the *groups* into k folds
    groups = sorted({label_group[lbl] for lbl in ids_by_label})
    random.Random(seed).shuffle(groups)
    fold_of_group = {g: (i % k) for i, g in enumerate(groups)}

    labels_sorted = sorted(ids_by_label)
    folds: list[EvalFold] = []
    for f in range(k):
        eval_labels = [lbl for lbl in labels_sorted if fold_of_group[label_group[lbl]] == f]
        train_labels = [lbl for lbl in labels_sorted if fold_of_group[label_group[lbl]] != f]

        gallery: list[str] = []
        q_closed: list[str] = []
        q_open: list[str] = []
        for lbl in eval_labels:
            # gallery/query are built from REAL photos only; synthetic views are training-only
            real = [i for i in ids_by_label[lbl] if not is_synthetic(i)]
            if not real:
                continue                       # individual with only synthetic views → unevaluable
            if len(real) >= 2:
                held = real[-1] if hold_out == "last" else real[0]
                gallery.extend(i for i in real if i != held)
                q_closed.append(held)
            else:
                q_open.append(real[0])         # singleton (by real photos) -> novel query

        folds.append(
            EvalFold(
                fold=f,
                mode=mode,
                train_labels=train_labels,
                gallery_ids=sorted(gallery),
                query_closed_ids=sorted(q_closed),
                query_open_ids=sorted(q_open),
            )
        )
    return folds


def check_leakage(folds: list[EvalFold], spotsets: list[SpotSet]) -> list[str]:
    """Return a list of leakage problems (empty list == clean). Raises nothing.

    Verifies, per fold: train ∩ eval labels are disjoint; gallery/query ids don't overlap;
    open-query labels are absent from the gallery; every closed query has a same-label
    gallery photo; no zero-spot image is used in any role.
    """
    label_of = {ss.salamander_id: ss.label for ss in spotsets}
    empty_ids = {ss.salamander_id for ss in spotsets if ss.is_empty}
    problems: list[str] = []

    for fd in folds:
        tag = f"fold {fd.fold}"
        gset, cset, oset = set(fd.gallery_ids), set(fd.query_closed_ids), set(fd.query_open_ids)

        # roles must not overlap
        for a_name, a, b_name, b in [
            ("gallery", gset, "closed", cset),
            ("gallery", gset, "open", oset),
            ("closed", cset, "open", oset),
        ]:
            if a & b:
                problems.append(f"{tag}: {a_name}∩{b_name} overlap: {sorted(a & b)[:3]}…")

        # train vs eval label disjointness
        eval_labels = {label_of[i] for i in (gset | cset | oset)}
        train_labels = set(fd.train_labels)
        if eval_labels & train_labels:
            problems.append(f"{tag}: train∩eval labels overlap: {sorted(eval_labels & train_labels)[:3]}…")

        # open queries must be novel (label not in gallery)
        gallery_labels = {label_of[i] for i in gset}
        open_labels = {label_of[i] for i in oset}
        if gallery_labels & open_labels:
            problems.append(f"{tag}: open-query label(s) present in gallery: {sorted(gallery_labels & open_labels)[:3]}…")

        # every closed query must have a same-label gallery photo
        for i in cset:
            if label_of[i] not in gallery_labels:
                problems.append(f"{tag}: closed query {i} has no gallery match")
                break

        # no empties anywhere
        used_empty = (gset | cset | oset) & empty_ids
        if used_empty:
            problems.append(f"{tag}: zero-spot image(s) used in a role: {sorted(used_empty)[:3]}…")

        # synthetic views must never appear in gallery/query (training-only)
        synth_used = {i for i in (gset | cset | oset) if is_synthetic(i)}
        if synth_used:
            problems.append(f"{tag}: synthetic id(s) in a role: {sorted(synth_used)[:3]}…")

    return problems


def folds_to_jsonable(folds: list[EvalFold]) -> list[dict]:
    return [asdict(fd) for fd in folds]


def save_splits(path: Path, folds: list[EvalFold], meta: dict | None = None) -> None:
    save_json(path, {"meta": meta or {}, "folds": folds_to_jsonable(folds)})
