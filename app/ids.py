"""Identifier conventions — must match the pipeline (spec §6.1).

* **Image ID**   = raw file stem ``<code>_<individual>_<instance>`` e.g. ``aj_1_2``.
                   Synthetic views are ``<label>_g<k>`` e.g. ``aj_1_g0``.
* **Individual ID (label)** = image ID with the trailing ``_<instance>`` removed:
                   ``aj_1_2 -> aj_1``  /  ``aj_1_g0 -> aj_1``.
* **Display ID** = label upper-cased with the last ``_`` as ``-``: ``aj_1 -> AJ-1``.
                   Round-trips losslessly.
* **Provisional upload ID** = ``up_<shortid>`` — reserved, never allocated to an individual.
* **New individual code** = the next free two-letter combo not used by imported data,
                   expanding to three-or-more letters once ``zz`` is reached.
"""

from __future__ import annotations

import re
import secrets
import string

#: reserved id prefix for freshly-uploaded photos, before any re-ID decision
PROVISIONAL_PREFIX = "up"

_INSTANCE_RE = re.compile(r"^(?P<label>.+)_(?P<instance>\d+|g\d+)$")
_LABEL_RE = re.compile(r"^(?P<code>[a-z0-9]+)_(?P<num>\d+)$")


class IdError(ValueError):
    """Raised when an id does not follow the pipeline convention."""


def split_instance(image_id: str) -> tuple[str, str]:
    """``aj_1_2 -> ("aj_1", "2")``  /  ``aj_1_g0 -> ("aj_1", "g0")``."""
    m = _INSTANCE_RE.match(image_id)
    if not m:
        raise IdError(f"not an <label>_<instance> image id: {image_id!r}")
    return m.group("label"), m.group("instance")


def individual_id_of(image_id: str) -> str:
    """The identity label for a photo id (``ca_10_5 -> ca_10``)."""
    return split_instance(image_id)[0]


def is_synthetic_id(image_id: str) -> bool:
    """True for a ``<label>_g<k>`` synthetic view."""
    try:
        return split_instance(image_id)[1].startswith("g")
    except IdError:
        return False


def is_provisional_id(image_id: str) -> bool:
    return image_id.startswith(PROVISIONAL_PREFIX + "_")


def code_of(label_or_image_id: str) -> str:
    """The leading ``<code>`` token (``aj_1_2 -> aj``, ``amr_5 -> amr``)."""
    return label_or_image_id.split("_", 1)[0]


def display_id(individual_id: str) -> str:
    """``aj_1 -> AJ-1``. Raises :class:`IdError` for a non-label string."""
    m = _LABEL_RE.match(individual_id)
    if not m:
        raise IdError(f"not an <code>_<num> individual id: {individual_id!r}")
    return f"{m.group('code').upper()}-{m.group('num')}"


def individual_id_from_display(display: str) -> str:
    """``AJ-1 -> aj_1``. Inverse of :func:`display_id`."""
    if "-" not in display:
        raise IdError(f"not a display id: {display!r}")
    code, num = display.rsplit("-", 1)
    if not num.isdigit():
        raise IdError(f"not a display id: {display!r}")
    return f"{code.lower()}_{num}"


def new_provisional_id() -> str:
    """A fresh ``up_<shortid>`` for an uploaded photo (§6.1)."""
    return f"{PROVISIONAL_PREFIX}_{secrets.token_hex(4)}"


def _code_sequence():
    """``aa, ab, ..., zz, aaa, aab, ...`` — width is not fixed (§6.1)."""
    letters = string.ascii_lowercase
    width = 2
    while True:
        # odometer over `letters` of the current width
        idx = [0] * width
        while True:
            yield "".join(letters[i] for i in idx)
            pos = width - 1
            while pos >= 0:
                idx[pos] += 1
                if idx[pos] < len(letters):
                    break
                idx[pos] = 0
                pos -= 1
            if pos < 0:
                break
        width += 1


def allocate_individual_code(used_codes: set[str]) -> str:
    """Next free code not in ``used_codes`` (which must include every imported prefix
    and the reserved ``up`` prefix, so new ids never collide — §6.1)."""
    for code in _code_sequence():
        if code not in used_codes:
            return code
    raise AssertionError("unreachable: _code_sequence is infinite")


def next_instance(existing_instances: list[str]) -> str:
    """The next free numeric instance for an individual (``["1","3"] -> "4"``)."""
    nums = [int(i) for i in existing_instances if i.isdigit()]
    return str(max(nums, default=0) + 1)
