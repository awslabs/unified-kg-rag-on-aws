# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fuzzy entity resolution on Hangul, Han and Kana names.

Regressions: character shingles merged CJK names that differ only in their
final character, which in these scripts is the head noun ("...연구원" an
institute vs "...연구소" a lab; "...전자" vs "...전기"; "...商事" vs
"...物産"; the given names 김철수 vs 김철민). Meanwhile spacing variants
("가나다상사" / "가나다 상사") and legal-form designators ("(주)가나다" /
"가나다", where "주" even counted as an identifier) never merged. All names
are synthetic.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.ingestion.base_resolver import (
    FuzzyMatcher,
    discriminator_tokens,
    strip_legal_forms,
)
from unified_kg_rag.domain.models import ResolutionMethod
from unified_kg_rag.shared.utils import entity_key

pytestmark = pytest.mark.unit

METHODS = [ResolutionMethod.MINHASH, ResolutionMethod.SEQUENCE_MATCHER]


def _matches(method: ResolutionMethod, query: str, candidate: str) -> bool:
    matcher = FuzzyMatcher(
        [candidate], resolution_method=method, similarity_threshold=0.5
    )
    return candidate in {name for name, _ in matcher.find_all_matches(query)}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("한국가나다연구원", "한국가나다연구소"),
        ("가나다라전자", "가나다라전기"),
        ("青空海運商事", "青空海運物産"),
        ("김철수", "김철민"),
        ("そらいろ銀行", "そらいろ証券"),
    ],
)
def test_different_head_character_never_fuzzy_merges(method, a: str, b: str) -> None:
    assert not _matches(method, a, b)
    assert not _matches(method, b, a)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("가나다상사", "가나다 상사"),
        ("(주)가나다", "가나다"),
        ("㈜가나다", "가나다"),
        ("가나다(주)", "가나다"),
        ("주식회사 가나다", "가나다"),
        ("가나다 주식회사", "가나다"),
        ("(유)가나다", "가나다"),
        ("유한회사 가나다", "가나다"),
        ("株式会社そらいろ", "そらいろ"),
        ("㈱そらいろ", "そらいろ"),
        ("华青有限公司", "华青"),
        ("한국 가나다 연구원", "한국가나다연구원"),
    ],
)
def test_spacing_and_legal_form_variants_merge(method, a: str, b: str) -> None:
    assert _matches(method, a, b)
    assert _matches(method, b, a)


@pytest.mark.parametrize("method", METHODS)
def test_designator_alone_is_not_stripped(method) -> None:
    # A name that is only a designator keeps it, so it never matches "".
    assert strip_legal_forms("주식회사") == "주식회사"
    assert strip_legal_forms("Inc.") == "inc."
    assert not _matches(method, "(주)", "(유)")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("(주)가나다", set()),
        ("㈜가나다", set()),
        ("가나다 (유)", set()),
        ("(株)そらいろ", set()),
        ("(주)가나다 2공장", {"2공장"}),
        ("Vendor A Inc.", {"a"}),
        ("Acme Co", set()),
    ],
)
def test_legal_form_is_not_a_discriminator(name: str, expected: set[str]) -> None:
    assert discriminator_tokens(name) == frozenset(expected)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Acme Inc", "acme"),
        ("Acme, Inc.", "acme"),
        ("Acme Ltd", "acme"),
        ("Acme LLC", "acme"),
        ("Acme Co.", "acme"),
        ("Acme Corp", "acme"),
        ("Taco", "taco"),  # only a whole trailing word is a designator
        ("Acme Corporation", "acme corporation"),
        ("Co Holdings", "co holdings"),  # Latin forms are trailing only
    ],
)
def test_strip_legal_forms_latin(name: str, expected: str) -> None:
    assert strip_legal_forms(entity_key(name)) == expected


def test_entity_key_is_unchanged() -> None:
    # Ids hash entity_key; the matching rules must not change it.
    assert entity_key("(주)가나다") == "(주)가나다"
    assert entity_key("가나다 상사") == "가나다 상사"


@pytest.mark.parametrize("method", METHODS)
def test_english_head_words_unaffected(method) -> None:
    # The head-character guard applies only to dense-script names.
    assert _matches(method, "Acme Corporation", "Acme Corporations")
