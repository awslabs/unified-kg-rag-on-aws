# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A name key is its own key: ``entity_key``, ``compact_name_key`` and
``link_key`` of a key give the key back.

A key that keys otherwise when keyed again names one entity two ways: a
stored key compared with a key of it (an incremental merge, a query link)
misses. Regressions: the trailing "." a removed legal form left ("Müller &
Co. KG" gave "müller & co."), a middle dot a dropped quote mark or space put
between two letters, and an "&" a dropped space joined to the word before.

And names with one ``entity_key`` (one entity id) have one matching key of
every kind: the discriminator tokens split "한·빛" at the middle dot its key
joins, so whether the fuzzy matcher linked a spelling depended on which one
the merged entity was displayed under. Names with one ``compact_name_key``
(spacing and legal-form variants: one entity) have one discriminator token
set and one head character, so the fuzzy matcher's guards never keep two
spellings of one entity apart ("제3조" and "제 3 조" had {제3조} and
{제, 3, 조}).
"""

from __future__ import annotations

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from unified_kg_rag.domain.ingestion.names import (
    compact_name_key,
    discriminator_tokens,
    head_character,
)
from unified_kg_rag.domain.retrieval.linking import link_key
from unified_kg_rag.shared.utils import entity_key

pytestmark = pytest.mark.property

# The characters and words each fold reads: dense letters, Latin and Cyrillic
# words, digits, spaces, dots, dashes and the minus sign, middle dots and
# Korean's "ㆍ", quote marks and apostrophe stand-ins, invisible format
# characters, Arabic and Hebrew marks, and legal forms of every kind.
_PARTS = st.sampled_from(
    [
        *"가나다アイ漢字ab1 .,-–−_·・ㆍ「」《》\"'’´ʼ′ʻ",
        "\u00ad",
        "\u200b",
        "\u200d",
        "\u200f",
        "\ufeff",
        "ـ",
        "ً",
        "ׂ",
        "。",
        "Sá",
        "Cổ",
        "Sapa",
        "kg",
        "500",
        "04",
        "Inc",
        " Ltd",
        " Co.",
        "ООО ",
        "Công ty ",
        "Cong ty ",
        "주식회사",
        "(주)",
        " GmbH",
        " & Co. KG",
        " e.V.",
        " S.A.B. de C.V.",
        " จำกัด (มหาชน)",
    ]
)
_NAMES = st.one_of(
    st.text(max_size=12),
    st.lists(_PARTS, min_size=1, max_size=8).map("".join),
)


@given(name=_NAMES)
@example(name="漢・》漢《")
@example(name="・·イ・」주식회사")
@example(name="a. ;")
@example(name="가·・漢")
@example(name="ß́")
def test_entity_key_is_its_own_key(name: str) -> None:
    key = entity_key(name)
    assert entity_key(key) == key


@given(name=_NAMES)
@example(name="Müller & Co. KG")
@example(name="가 & Co. KG")
@example(name="字–・イ GmbH")
@example(name="字漢 -1")
@example(name="บริษัท ปตท. จำกัด (มหาชน)")
def test_compact_name_key_is_its_own_key(name: str) -> None:
    key = compact_name_key(name)
    assert compact_name_key(key) == key
    assert link_key(link_key(name)) == link_key(name)


@given(name=_NAMES)
@example(name="한·빛")
@example(name="中・日")
@example(name="״빛(주)「")
def test_names_with_one_entity_key_have_one_matching_key(name: str) -> None:
    # Each key is a function of the entity key: so two names with one
    # entity key have the same keys.
    key = entity_key(name)
    for matching_key in (compact_name_key, discriminator_tokens, head_character):
        assert matching_key(name) == matching_key(key), matching_key.__name__


@given(name=_NAMES, spaced=st.lists(st.sampled_from([" ", "  ", ""]), max_size=12))
@example(name="제3조", spaced=[" ", " "])
@example(name="第3条", spaced=[" ", " "])
@example(name="2공장", spaced=[" "])
@example(name="3号館", spaced=[" "])
@example(name="1차 협력사", spaced=[" ", "", " "])
@example(name="(주)가나다 2공장", spaced=["", "", "", "", "", " "])
def test_names_with_one_compact_key_have_one_guard_key(
    name: str, spaced: list[str]
) -> None:
    # The name respaced between its characters, and the name's compact key:
    # where they share a compact key they share the guards' keys too.
    respaced = "".join(
        c + (spaced[i] if i < len(spaced) else "") for i, c in enumerate(name)
    )
    key = compact_name_key(name)
    for other in (respaced, key):
        if compact_name_key(other) == key:
            assert discriminator_tokens(other) == discriminator_tokens(name), other
            assert head_character(other) == head_character(name), other
