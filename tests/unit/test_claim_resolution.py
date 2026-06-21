# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
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
