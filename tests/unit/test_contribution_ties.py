# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tie between the statements of an item resolves the same whatever their
order: the order of the text units differs between corpora and namespaces
of the same text, so the first statement's value made a full build depend
on them."""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.ingestion.contributions import (
    merge_claim_contributions,
    merge_entity_contributions,
)
from unified_kg_rag.domain.models import Claim, Entity

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("order", [("PERSON", "LOCATION"), ("LOCATION", "PERSON")])
def test_a_tied_entity_type_is_the_smallest(order: tuple[str, str]) -> None:
    entities = [
        Entity(id="e", name="Acme", type=t, text_unit_ids=[f"t{n}"])
        for n, t in enumerate(order)
    ]

    (merged,) = merge_entity_contributions(entities)

    assert merged.type == "location"


def test_the_most_frequent_type_still_wins() -> None:
    entities = [
        Entity(id="e", name="Acme", type=t, text_unit_ids=[f"t{n}"])
        for n, t in enumerate(["PERSON", "PERSON", "LOCATION"])
    ]

    (merged,) = merge_entity_contributions(entities)

    assert merged.type == "person"


@pytest.mark.parametrize("order", [("TRUE", "SUSPECTED"), ("SUSPECTED", "TRUE")])
def test_a_tied_claim_status_is_the_smallest(order: tuple[str, str]) -> None:
    claims = [
        Claim(
            id="c",
            subject_id="e",
            subject_name="Acme",
            object_name="100",
            type="PAYS",
            status=status,
            text_unit_ids=[f"t{n}"],
        )
        for n, status in enumerate(order)
    ]

    (merged,) = merge_claim_contributions(claims)

    assert merged.status == "suspected"
