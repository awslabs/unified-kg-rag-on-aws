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
)
from unified_kg_rag.domain.languages.legal_forms import strip_legal_forms
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
        ("(주)가나다 2공장", {"2"}),
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
        ("Acme Corporation", "acme"),
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


# Spacing variants of a dense-script name with a number share a compact key
# (one entity) but split into other whitespace words: their discriminator
# tokens differed, so the default (non-fuzzy) resolution kept them apart.
_SPACED_NUMBERED = [
    ("제3조", "제 3 조"),
    ("2공장", "2 공장"),
    ("제1항", "제 1 항"),
    ("1차 협력사", "1 차 협력사"),
    ("第3条", "第 3 条"),
    ("3号館", "3 号館"),
    ("第2章", "第 2 章"),
    ("第3条", "第3 条"),
]


@pytest.mark.parametrize(("a", "b"), _SPACED_NUMBERED)
def test_spacing_variants_with_a_number_are_one_entity(a: str, b: str) -> None:
    from unified_kg_rag.domain.ingestion.graph_resolver import GraphResolver
    from unified_kg_rag.domain.models import Config, Entity

    resolver = GraphResolver(Config(), max_workers=1, use_process_pool=False)
    entities = [
        Entity(id=f"e{i}", name=n, type="clause", text_unit_ids=[f"t{i}"])
        for i, n in enumerate((a, b))
    ]
    resolved, _, _ = resolver.entity_resolver.resolve(entities)
    assert len(resolved) == 1
    for method in METHODS:
        matcher = FuzzyMatcher(
            [b], resolution_method=method, similarity_threshold=0.9, fuzzy=False
        )
        assert [name for name, _ in matcher.find_all_matches(a)] == [b]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("a", "b"),
    [
        # An identifier glued inside a dense word: a Han numeral, a Latin
        # letter, digits.
        ("東京第一工場", "東京第二工場"),
        ("东京第一工厂", "东京第二工厂"),
        ("가나다아파트 A동", "가나다아파트 B동"),
        ("プロジェクトA号", "プロジェクトB号"),
        ("가나다아파트 101동", "가나다아파트 102동"),
        ("서울공장 1라인", "서울공장 2라인"),
    ],
)
def test_an_identifier_inside_a_dense_word_never_fuzzy_links(
    method: ResolutionMethod, a: str, b: str
) -> None:
    matcher = FuzzyMatcher(
        [b], resolution_method=method, similarity_threshold=0.5, fuzzy=True
    )
    assert matcher.find_all_matches(a) == []
