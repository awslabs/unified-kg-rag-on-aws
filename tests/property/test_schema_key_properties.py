# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invariants of ``schema_key``, the one comparison of schema labels.

Over arbitrary text: the key is its own key (a label written as its key
names the same type), and over labels of letters and digits the key does
not depend on surrounding whitespace, on the spelling of the separators
between words, or on letter case where the letters case-map both ways.
"""

from __future__ import annotations

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from unified_kg_rag.domain.models.schema import schema_key

pytestmark = pytest.mark.property

_WORD = st.text(
    alphabet=st.characters(categories=("Lu", "Ll", "Nd")), min_size=1, max_size=8
)
_SEPARATOR = st.text(alphabet=" _\t　 ", min_size=1, max_size=3)


@given(label=st.text())
# Case folding that leaves a decomposed sequence: "ß" + combining acute
# folds to "ss" + U+0301, which NFKC composes.
@example(label="ß́")
@example(label="ẞ́")
@example(label="Ϊ́")
def test_schema_key_is_idempotent(label: str) -> None:
    key = schema_key(label)
    assert schema_key(key) == key


@given(
    words=st.lists(_WORD, min_size=1, max_size=4),
    separators=st.lists(_SEPARATOR, min_size=3, max_size=3),
    padding=st.tuples(_SEPARATOR, _SEPARATOR).filter(
        lambda pad: "_" not in "".join(pad)
    ),
)
def test_schema_key_ignores_padding_and_separator_spelling(
    words: list[str], separators: list[str], padding: tuple[str, str]
) -> None:
    joined = "_".join(words)
    respelled = words[0] + "".join(
        separator + word
        for separator, word in zip(separators[: len(words) - 1], words[1:], strict=True)
    )
    assert schema_key(f"{padding[0]}{respelled}{padding[1]}") == schema_key(joined)


@given(label=st.text(alphabet=st.characters(categories=("Lu", "Ll", "Nd"))))
def test_schema_key_ignores_case(label: str) -> None:
    # Only where upper- and lowercasing round-trip (not "ı", "ſ" or "ǅ").
    if label.upper().lower() != label.lower():
        return
    assert schema_key(label.upper()) == schema_key(label.lower())
