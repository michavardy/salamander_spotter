"""Quality ladder -> tier (spec §7.2).

The tier is derived from the ``image_quality`` composites at extraction time and
shown throughout the UI. Gates are tunable in Settings; the defaults live here.
"""

from __future__ import annotations

from dataclasses import dataclass

TIER_AUTO_ACCEPT = "auto_accept"
TIER_NEEDS_A_LOOK = "needs_a_look"
TIER_HAND_CORRECTION = "hand_correction"
TIER_FAILED = "failed"

ALL_TIERS = (TIER_AUTO_ACCEPT, TIER_NEEDS_A_LOOK, TIER_HAND_CORRECTION, TIER_FAILED)


@dataclass(frozen=True)
class LadderConfig:
    auto_accept_quality: float = 0.65
    needs_a_look_floor: float = 0.40
    min_spots_auto_accept: int = 3  # Settings `min_spots_auto_accept` (spec D9)


DEFAULT_LADDER = LadderConfig()


def ladder_tier(
    *,
    overall_quality: float | None,
    has_axis: bool,
    n_spots: int | None,
    extraction_failed: bool = False,
    config: LadderConfig = DEFAULT_LADDER,
) -> str:
    """Map an extraction result onto ``{auto_accept, needs_a_look, hand_correction, failed}``.

    * failed extraction (no geometry at all) -> ``failed``
    * ``overall_quality >= 0.65`` AND a usable axis AND >= N spots -> ``auto_accept``
    * ``0.40 <= overall_quality < 0.65`` -> ``needs_a_look``
    * ``overall_quality < 0.40`` but geometry was returned -> ``hand_correction``
    * ``overall_quality >= 0.65`` but axis/spots insufficient -> ``needs_a_look``
    """
    n = n_spots or 0
    if extraction_failed or overall_quality is None:
        return TIER_FAILED

    if (
        overall_quality >= config.auto_accept_quality
        and has_axis
        and n >= config.min_spots_auto_accept
    ):
        return TIER_AUTO_ACCEPT

    if overall_quality < config.needs_a_look_floor:
        return TIER_HAND_CORRECTION

    return TIER_NEEDS_A_LOOK
