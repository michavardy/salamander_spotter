from __future__ import annotations

import pytest

from app import ids


@pytest.mark.parametrize(
    "image_id, label, instance",
    [
        ("aj_1_2", "aj_1", "2"),
        ("ca_10_5", "ca_10", "5"),
        ("aj_1_g0", "aj_1", "g0"),
        ("amr_5_1", "amr_5", "1"),
        ("up_7f3a2b9c_1", "up_7f3a2b9c", "1"),
    ],
)
def test_split_instance(image_id, label, instance):
    assert ids.split_instance(image_id) == (label, instance)
    assert ids.individual_id_of(image_id) == label


def test_split_instance_rejects_bad_id():
    with pytest.raises(ids.IdError):
        ids.split_instance("no-underscores")


def test_is_synthetic_id():
    assert ids.is_synthetic_id("aj_1_g0")
    assert ids.is_synthetic_id("aj_1_g12")
    assert not ids.is_synthetic_id("aj_1_2")
    assert not ids.is_synthetic_id("garbage")


def test_is_provisional_id():
    assert ids.is_provisional_id("up_7f3a2b9c")
    assert not ids.is_provisional_id("aj_1_2")


@pytest.mark.parametrize(
    "label, display",
    [("aj_1", "AJ-1"), ("ca_10", "CA-10"), ("en_1", "EN-1"), ("amr_5", "AMR-5")],
)
def test_display_id_roundtrip(label, display):
    assert ids.display_id(label) == display
    assert ids.individual_id_from_display(display) == label


def test_display_id_rejects_non_label():
    with pytest.raises(ids.IdError):
        ids.display_id("aj_1_2")


def test_code_of():
    assert ids.code_of("aj_1_2") == "aj"
    assert ids.code_of("amr_5") == "amr"


def test_new_provisional_id_shape():
    pid = ids.new_provisional_id()
    assert ids.is_provisional_id(pid)
    assert len(pid.split("_")[1]) == 8


class TestCodeAllocator:
    def test_first_free_two_letter(self):
        assert ids.allocate_individual_code(set()) == "aa"
        assert ids.allocate_individual_code({"aa"}) == "ab"
        assert ids.allocate_individual_code({"aa", "ab", "ac"}) == "ad"

    def test_skips_imported_prefixes(self):
        used = {"aa", "aj", "ca", "en", "up"}
        code = ids.allocate_individual_code(used)
        assert code not in used
        assert code == "ab"

    def test_expands_past_two_letters(self):
        import string

        two = {a + b for a in string.ascii_lowercase for b in string.ascii_lowercase}
        assert ids.allocate_individual_code(two) == "aaa"


def test_next_instance():
    assert ids.next_instance([]) == "1"
    assert ids.next_instance(["1", "2"]) == "3"
    assert ids.next_instance(["1", "3"]) == "4"
    assert ids.next_instance(["1", "g0", "2"]) == "3"
