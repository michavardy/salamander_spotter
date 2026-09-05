"""The decision engine (spec §9.1) — what happens to a sighting after matching,
and how a reviewer's verdict is applied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from ..db import Database, new_id
from ..ids import allocate_individual_code, display_id, next_instance
from ..settings_store import SettingsStore
from . import matching

VERDICT_CONFIRM = "confirm"
VERDICT_NEW = "new"
VERDICT_UNCERTAIN = "uncertain"
VERDICT_DISQUALIFY = "disqualify"


@dataclass
class Triage:
    outcome: str            # auto_confirmed | queued
    queued_as: str | None   # hand_correction | needs_decision | likely_new
    suggestion_individual_id: str | None
    suggestion_confidence: float | None


def triage(db: Database, settings: SettingsStore, *, image_id: str, match_id: str | None) -> Triage:
    """Apply the §9.1 table to a freshly-matched sighting."""
    thr = settings.decision_thresholds()
    img = db.query_one("SELECT * FROM images WHERE image_id = ?", [image_id])
    candidates = matching.load_candidates(db, match_id) if match_id else []
    top = candidates[0] if candidates else None

    if img and img["ladder_tier"] == "failed":
        _set_status(db, image_id, "in_review")
        return Triage("queued", "hand_correction", None, None)

    if top and top["calibrated_confidence"] >= thr.auto_approve:
        _apply_confirm(db, image_id, top["individual_id"], suggestion=top,
                       reviewer_id=None, was_override=False, auto=True)
        return Triage("auto_confirmed", None, top["individual_id"], top["calibrated_confidence"])

    _set_status(db, image_id, "in_review")
    if top and top["calibrated_confidence"] >= thr.match_threshold:
        return Triage("queued", "needs_decision", top["individual_id"], top["calibrated_confidence"])
    return Triage("queued", "likely_new", None, None)


@dataclass
class DecisionInput:
    image_id: str
    verdict: str
    chosen_individual_id: str | None = None
    reason_chips: list[str] | None = None
    note: str | None = None
    reviewer_id: str | None = None
    confirm_override: bool = False   # client acknowledged the override guard


@dataclass
class DecisionOutcome:
    decision_id: str
    verdict: str
    individual_id: str | None
    was_override: bool
    override_warning: str | None = None
    new_individual_id: str | None = None


def apply_decision(db: Database, settings: SettingsStore, inp: DecisionInput) -> DecisionOutcome:
    thr = settings.decision_thresholds()
    img = db.query_one("SELECT * FROM images WHERE image_id = ?", [inp.image_id])
    if not img:
        raise ValueError(f"unknown image {inp.image_id}")

    match = matching.latest_match_for(db, inp.image_id)
    candidates = matching.load_candidates(db, match["id"]) if match else []
    top = candidates[0] if candidates else None
    suggestion_id = top["individual_id"] if top and top["calibrated_confidence"] >= thr.match_threshold else None
    suggestion_conf = top["calibrated_confidence"] if top else None

    was_override = False
    override_warning = None
    if inp.verdict == VERDICT_CONFIRM and suggestion_id and inp.chosen_individual_id != suggestion_id:
        was_override = True
    if inp.verdict == VERDICT_NEW and suggestion_id is not None:
        was_override = True
    if was_override and thr.warn_on_override and not inp.confirm_override:
        chosen_conf = next(
            (c["calibrated_confidence"] for c in candidates if c["individual_id"] == inp.chosen_individual_id),
            0.0,
        )
        override_warning = (
            f"You picked {inp.chosen_individual_id or 'New'} ({chosen_conf:.2f}) over the "
            f"suggested {suggestion_id} ({suggestion_conf:.2f}). Sure?"
        )
        return DecisionOutcome(
            decision_id="", verdict=inp.verdict, individual_id=None,
            was_override=True, override_warning=override_warning,
        )

    decision_id = new_id("dec")
    new_individual_id = None
    resolved_individual = None

    with db.transaction():
        if inp.verdict == VERDICT_CONFIRM:
            if not inp.chosen_individual_id:
                raise ValueError("confirm needs chosen_individual_id")
            _apply_confirm(db, inp.image_id, inp.chosen_individual_id, suggestion=top,
                           reviewer_id=inp.reviewer_id, was_override=was_override, auto=False,
                           decision_id=decision_id)
            resolved_individual = inp.chosen_individual_id

        elif inp.verdict == VERDICT_NEW:
            new_individual_id = _enroll_new(db, inp.image_id, reviewer_id=inp.reviewer_id)
            resolved_individual = new_individual_id
            _record_decision(db, decision_id, inp, img, "new", None, new_individual_id,
                             top, was_override)

        elif inp.verdict == VERDICT_UNCERTAIN:
            _set_status(db, inp.image_id, "flagged_uncertain")
            _record_decision(db, decision_id, inp, img, "uncertain", None, None, top, was_override)
            db.log_activity(kind="review", summary=f"{inp.image_id} flagged uncertain",
                            actor=inp.reviewer_id, ref_type="image", ref_id=inp.image_id)

        elif inp.verdict == VERDICT_DISQUALIFY:
            if not inp.reason_chips:
                raise ValueError("disqualify requires a reason chip")
            _set_status(db, inp.image_id, "disqualified")
            _record_decision(db, decision_id, inp, img, "disqualify", None, None, top, was_override)
            db.log_activity(kind="review", summary=f"{inp.image_id} disqualified",
                            actor=inp.reviewer_id, ref_type="image", ref_id=inp.image_id)
        else:
            raise ValueError(f"unknown verdict {inp.verdict!r}")

        for chip in inp.reason_chips or []:
            db.insert("review_decision_reasons", {"decision_id": decision_id, "chip": chip})

    return DecisionOutcome(
        decision_id=decision_id, verdict=inp.verdict, individual_id=resolved_individual,
        was_override=was_override, new_individual_id=new_individual_id,
    )


# --------------------------------------------------------------------------- #
def _set_status(db: Database, image_id: str, status: str) -> None:
    db.execute("UPDATE images SET status = ?, rev = rev + 1 WHERE image_id = ?", [status, image_id])


def _apply_confirm(db, image_id, individual_id, *, suggestion, reviewer_id, was_override,
                   auto, decision_id=None):
    img = db.query_one("SELECT * FROM images WHERE image_id = ?", [image_id])
    if not db.query_one("SELECT 1 FROM individuals WHERE individual_id = ?", [individual_id]):
        raise ValueError(f"unknown individual {individual_id}")
    db.execute(
        "UPDATE images SET individual_id = ?, status = 'confirmed', rev = rev + 1 WHERE image_id = ?",
        [individual_id, image_id],
    )
    db.execute(
        "UPDATE individuals SET last_seen = coalesce(?, last_seen), "
        "reference_image_id = coalesce(reference_image_id, ?), rev = rev + 1 WHERE individual_id = ?",
        [img.get("photographed_at") if img else None, image_id, individual_id],
    )
    if img and img.get("contributor_id"):
        db.execute(
            "INSERT INTO individual_contributors (individual_id, contributor_id) VALUES (?, ?) "
            "ON CONFLICT DO NOTHING",
            [individual_id, img["contributor_id"]],
        )
    did = decision_id or new_id("dec")
    _record_decision(db, did, DecisionInput(image_id, "confirm", individual_id,
                                            reviewer_id=reviewer_id),
                     img or {}, "confirm", individual_id, None, suggestion, was_override)
    db.log_activity(
        kind="review",
        summary=f"{'Auto-confirmed' if auto else 'Confirmed'} {image_id} as {individual_id}",
        actor=reviewer_id, ref_type="image", ref_id=image_id,
    )


def _enroll_new(db: Database, image_id: str, *, reviewer_id: str | None) -> str:
    used = {r[0] for r in db.query("SELECT DISTINCT split_part(individual_id,'_',1) FROM individuals")}
    used.add("up")
    code = allocate_individual_code(used)
    individual_id = f"{code}_1"
    img = db.query_one("SELECT * FROM images WHERE image_id = ?", [image_id])
    db.insert(
        "individuals",
        {
            "individual_id": individual_id,
            "display_id": display_id(individual_id),
            "site_id": img["site_id"] if img else None,
            "status": "provisional",
            "first_seen": img.get("photographed_at") if img else None,
            "last_seen": img.get("photographed_at") if img else None,
            "reference_image_id": image_id,
        },
    )
    db.execute(
        "UPDATE images SET individual_id = ?, status = 'enrolled_new', rev = rev + 1 WHERE image_id = ?",
        [individual_id, image_id],
    )
    if img and img.get("contributor_id"):
        db.insert("individual_contributors",
                  {"individual_id": individual_id, "contributor_id": img["contributor_id"]})
    db.log_activity(kind="enrollment", summary=f"Enrolled new individual {individual_id} from {image_id}",
                    actor=reviewer_id, ref_type="individual", ref_id=individual_id)
    return individual_id


def _record_decision(db, decision_id, inp: "DecisionInput", img: dict, verdict, chosen, new_id_,
                     suggestion, was_override):
    db.insert(
        "review_decisions",
        {
            "id": decision_id,
            "image_id": inp.image_id,
            "batch_id": img.get("review_batch_id") if isinstance(img, dict) else None,
            "verdict": verdict,
            "chosen_individual_id": chosen,
            "new_individual_id": new_id_,
            "reviewer_id": inp.reviewer_id,
            "model_suggestion_individual_id": suggestion["individual_id"] if suggestion else None,
            "model_suggestion_confidence": suggestion["calibrated_confidence"] if suggestion else None,
            "was_override": was_override,
            "note": inp.note,
        },
    )
