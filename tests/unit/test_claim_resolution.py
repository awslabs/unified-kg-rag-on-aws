# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for claim->entity resolution (AWS-free, pure task function).

Regression cover for the over-dropping bug: a claim is anchored to its SUBJECT
entity; the OBJECT may be a literal value (date/amount/status), so it must not be
required to resolve to an entity.
"""

from __future__ import annotations

import pytest

from aws_graphrag.domain.ingestion.base_resolver import FuzzyMatcher
from aws_graphrag.domain.ingestion.claim_resolver import resolve_single_claim_task
from aws_graphrag.domain.models import Claim

pytestmark = pytest.mark.unit


def _claim(subject: str, obj: str) -> Claim:
    return Claim(
        id="c1",
        subject_id="",
        subject_name=subject,
        object_id="",
        object_name=obj,
        type="PERFORMANCE",
        description="...",
    )


def _maps(entities: dict[str, str]):
    # entities: normalized_name -> id ; also build name maps + matcher.
    name_to_id = dict(entities)
    name_to_original = {n: n for n in entities}
    matcher = FuzzyMatcher(candidates=list(entities))
    return name_to_id, name_to_original, matcher


def test_object_literal_value_is_kept_with_no_object_id() -> None:
    # Subject 'alice' is an entity; object '$100k revenue' is a literal value.
    name_to_id, name_to_orig, matcher = _maps({"alice": "e-alice"})
    resolved = resolve_single_claim_task(
        _claim("alice", "$100k revenue"), name_to_id, name_to_orig, matcher
    )
    assert resolved is not None  # not dropped
    assert resolved.subject_id == "e-alice"
    assert resolved.object_id is None  # literal value, no entity
    assert resolved.object_name == "$100k revenue"  # original text preserved


def test_both_entities_resolve_to_ids() -> None:
    name_to_id, name_to_orig, matcher = _maps({"alice": "e-alice", "acme": "e-acme"})
    resolved = resolve_single_claim_task(
        _claim("alice", "acme"), name_to_id, name_to_orig, matcher
    )
    assert resolved is not None
    assert resolved.subject_id == "e-alice"
    assert resolved.object_id == "e-acme"


def test_unresolvable_subject_drops_claim() -> None:
    # No anchor entity for the subject -> claim cannot attach to the graph.
    name_to_id, name_to_orig, matcher = _maps({"acme": "e-acme"})
    resolved = resolve_single_claim_task(
        _claim("nonexistent-person-xyz", "acme"), name_to_id, name_to_orig, matcher
    )
    assert resolved is None


class TestAbbreviationLocaleGuard:
    """Acronym/caps abbreviation generation is a Latin/cased-script notion;
    non-Latin (CJK, etc.) input must not yield garbage acronyms."""

    def test_latin_multiword_yields_acronym(self) -> None:
        abbrevs = FuzzyMatcher._generate_abbreviations("Amazon Web Services")
        assert "AWS" in abbrevs

    def test_latin_caps_extracted(self) -> None:
        abbrevs = FuzzyMatcher._generate_abbreviations("eXtensible Markup Language")
        # caps-extraction picks up embedded capitals
        assert any(a.isupper() and len(a) > 1 for a in abbrevs)

    def test_cjk_input_yields_no_abbreviations(self) -> None:
        # Korean multi-"word" name: no ASCII letters -> no acronym noise.
        assert FuzzyMatcher._generate_abbreviations("아마존 웹 서비스") == set()
        assert FuzzyMatcher._generate_abbreviations("한국어 엔티티 이름") == set()

    def test_mixed_script_still_uses_latin_letters(self) -> None:
        # Mixed input has ASCII letters -> guard lets it through.
        abbrevs = FuzzyMatcher._generate_abbreviations("AWS 클라우드")
        assert "AWS" in abbrevs
