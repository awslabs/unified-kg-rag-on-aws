# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for claim->entity resolution (AWS-free, pure functions).

A claim is anchored to its SUBJECT entity; the OBJECT may be a literal value
(date/amount/status), so it must not be required to resolve to an entity.
Fuzzy matches go through the guarded ``find_all_matches``, so an identifier
that differs ("PO 1003" vs "PO 1002") never rewrites a claim onto another
entity.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.ingestion.base_resolver import FuzzyMatcher
from unified_kg_rag.domain.ingestion.claim_resolver import resolve_single_claim
from unified_kg_rag.domain.models import Claim, ResolutionMethod
from unified_kg_rag.shared.utils import entity_key

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


def _resolve(claim: Claim, names: list[str], method: str = ResolutionMethod.MINHASH):
    names_by_key = {entity_key(n): n for n in names}
    ids_by_key = {entity_key(n): f"e-{n}" for n in names}
    matcher = FuzzyMatcher(names, resolution_method=method, similarity_threshold=0.6)
    return resolve_single_claim(claim, names_by_key, ids_by_key, matcher)


def test_object_literal_value_is_kept_with_no_object_id() -> None:
    resolved = _resolve(_claim("alice", "$100k revenue"), ["alice"])
    assert resolved is not None  # not dropped
    assert resolved.subject_id == "e-alice"
    assert resolved.object_id is None  # literal value, no entity
    assert resolved.object_name == "$100k revenue"  # original text preserved


def test_both_entities_resolve_to_ids() -> None:
    resolved = _resolve(_claim("alice", "acme"), ["alice", "acme"])
    assert resolved is not None
    assert resolved.subject_id == "e-alice"
    assert resolved.object_id == "e-acme"


def test_unresolvable_subject_drops_claim() -> None:
    resolved = _resolve(_claim("nonexistent-person-xyz", "acme"), ["acme"])
    assert resolved is None


def test_spelling_variant_resolves_through_compact_key() -> None:
    resolved = _resolve(_claim("(주)가나다상사", "acme"), ["가나다 상사", "acme"])
    assert resolved is not None
    assert resolved.subject_name == "가나다 상사"


@pytest.mark.parametrize(
    "method", [ResolutionMethod.MINHASH, ResolutionMethod.SEQUENCE_MATCHER]
)
def test_differing_identifiers_are_not_rewritten_onto_another_entity(method) -> None:
    # "PO 1003 -> Vendor B" must not become "PO 1002 -> Vendor A": the
    # shared prefix dominates the similarity, the identifier tells them apart.
    resolved = _resolve(
        _claim("Purchase Order 1003", "Vendor B"),
        ["Purchase Order 1002", "Vendor A"],
        method,
    )
    assert resolved is None  # no subject entity


@pytest.mark.parametrize(
    "method", [ResolutionMethod.MINHASH, ResolutionMethod.SEQUENCE_MATCHER]
)
def test_literal_value_is_not_matched_to_a_similar_entity(method) -> None:
    resolved = _resolve(
        _claim("Purchase Order 1002", "USD 1,500"),
        ["Purchase Order 1002", "USD 1,000"],
        method,
    )
    assert resolved is not None
    assert resolved.object_id is None
    assert resolved.object_name == "USD 1,500"


def test_equal_scores_break_ties_on_the_name() -> None:
    # Both candidates are compact-key variants of the reference (score 1.0).
    names = ["Gamma Lab Inc", "Gamma Lab Ltd"]
    resolved = _resolve(_claim("Gamma Lab", "x"), names)
    reversed_order = _resolve(_claim("Gamma Lab", "x"), names[::-1])
    assert resolved is not None and reversed_order is not None
    assert resolved.subject_name == reversed_order.subject_name == "Gamma Lab Inc"
