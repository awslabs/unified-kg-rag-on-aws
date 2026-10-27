# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for claim->entity resolution (AWS-free, pure task function).

Regression cover for the over-dropping bug: a claim is anchored to its SUBJECT
entity; the OBJECT may be a literal value (date/amount/status), so it must not be
required to resolve to an entity. Only an entity stated in the claim's text
units is a candidate.
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
        text_unit_ids=["u1"],
    )


def _resolve(
    claim: Claim,
    names: list[str],
    method: str = ResolutionMethod.MINHASH,
    elsewhere: tuple[str, ...] = (),
):
    """``claim`` resolved against ``names``, each stated in the claim's unit
    but those ``elsewhere`` (stated in another document's unit only)."""
    names_by_key = {entity_key(n): n for n in names}
    ids_by_key = {entity_key(n): f"e-{n}" for n in names}
    matcher = FuzzyMatcher(names, resolution_method=method, similarity_threshold=0.6)
    units = {n: {"u9" if n in elsewhere else "u1"} for n in names}
    return resolve_single_claim(claim, names_by_key, ids_by_key, matcher, units)


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
    # No anchor entity for the subject -> claim cannot attach to the graph.
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


def test_only_an_entity_stated_in_the_claims_units_is_a_candidate() -> None:
    # "Acme Inc" matches "Acme" and "Acme Corp" equally (legal forms). The
    # claim's unit states "Acme Corp"; "Acme" comes from another document,
    # so an incremental run deriving the claim from its unit alone sees only
    # "Acme Corp": a full build must not choose "Acme" either.
    names = ["Acme", "Acme Corp", "Buyer"]
    resolved = _resolve(_claim("Acme Inc", "Buyer"), names, elsewhere=("Acme",))
    assert resolved is not None
    assert resolved.subject_name == "Acme Corp"


def test_a_subject_stated_only_in_another_document_drops_the_claim() -> None:
    # The same name, but the claim's unit did not list it: its entity
    # extraction did not state the entity there.
    assert (
        _resolve(_claim("Acme", "Buyer"), ["Acme", "Buyer"], elsewhere=("Acme",))
        is None
    )
    assert (
        _resolve(_claim("Acme Inc", "Buyer"), ["Acme", "Buyer"], elsewhere=("Acme",))
        is None
    )


def test_an_object_stated_only_in_another_document_stays_a_literal() -> None:
    resolved = _resolve(
        _claim("Buyer", "Acme Inc"), ["Acme", "Buyer"], elsewhere=("Acme",)
    )
    assert resolved is not None
    assert (resolved.object_id, resolved.object_name) == (None, "Acme Inc")


def test_the_claim_resolver_resolves_against_the_claims_units() -> None:
    # Through ClaimResolver: it hands each entity's text units over.
    from unified_kg_rag.domain.ingestion.claim_resolver import ClaimResolver
    from unified_kg_rag.domain.models import Config, Entity

    entities = [
        Entity(id="e-acme", name="Acme", text_unit_ids=["u1"]),
        Entity(id="e-corp", name="Acme Corp", text_unit_ids=["u2"]),
        Entity(id="e-buyer", name="Buyer", text_unit_ids=["u2"]),
    ]
    claim = _claim("Acme Inc", "Buyer").model_copy(update={"text_unit_ids": ["u2"]})

    (resolved,), _ = ClaimResolver(
        Config(), max_workers=1, use_process_pool=False
    ).resolve([claim], entities)

    assert resolved.subject_id == "e-corp"
