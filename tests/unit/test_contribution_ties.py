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
    merge_relationship_contributions,
)
from unified_kg_rag.domain.models import Claim, Entity, Relationship

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


@pytest.mark.parametrize(
    "names", [("Acme", "ACME"), ("ACME", "Acme"), ("ACME", "Acme", "Acme")]
)
def test_a_name_is_its_most_frequent_spelling_the_smallest_on_a_tie(
    names: tuple[str, ...],
) -> None:
    entities = [
        Entity(id="e", name=name, text_unit_ids=[f"t{n}"])
        for n, name in enumerate(names)
    ]

    (merged,) = merge_entity_contributions(entities)

    assert merged.name == ("Acme" if len(names) == 3 else "ACME")


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


def _claim_statements(order: tuple[int, ...]) -> list[Claim]:
    # Two spellings of one literal object and claim type, tied.
    stated = [("PAYS", "USD 100", "Acme pays 100."), ("Pays", "usd 100", "A fee.")]
    return [
        Claim(
            id="c",
            subject_id="e",
            subject_name="Acme",
            object_name=stated[n][1],
            type=stated[n][0],
            description=stated[n][2],
            source_text=stated[n][2],
            text_unit_ids=[f"t{n}"],
        )
        for n in order
    ]


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_merged_claim_statements_do_not_depend_on_their_order(
    order: tuple[int, ...],
) -> None:
    # The object name and the descriptions' order were the first statement's.
    (merged,) = merge_claim_contributions(_claim_statements(order))

    assert merged.object_name == "USD 100"
    assert merged.description == "A fee.\nAcme pays 100."
    assert merged.source_text == "A fee.\nAcme pays 100."


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_resolved_claims_do_not_depend_on_their_order(order: tuple[int, ...]) -> None:
    # Claim resolution kept the first claim's type and object name and its
    # description first.
    from unified_kg_rag.domain.ingestion.claim_resolver import ClaimResolver

    resolver = object.__new__(ClaimResolver)
    merged = resolver._merge_claims(_claim_statements(order))

    assert (merged.type, merged.object_name) == ("PAYS", "USD 100")
    assert merged.description == "A fee.\nAcme pays 100."
    assert merged.source_text == "A fee.\nAcme pays 100."


@pytest.mark.parametrize("order", [(0, 1, 1), (1, 0, 1), (1, 1, 0)])
def test_the_most_frequent_claim_object_name_wins(order: tuple[int, ...]) -> None:
    # "usd 100" is stated twice, "USD 100" (the smaller) once: frequency
    # comes before the tie-break, in the merge and in claim resolution.
    from unified_kg_rag.domain.ingestion.claim_resolver import ClaimResolver

    statements = _claim_statements(order)

    (merged,) = merge_claim_contributions(_claim_statements(order))
    resolved = object.__new__(ClaimResolver)._merge_claims(statements)

    assert merged.object_name == resolved.object_name == "usd 100"


@pytest.mark.parametrize("name", ["{Seller}", "{0}", "{}", "Map{K,V}", "100%"])
def test_statements_of_a_name_with_braces_merge(name: str) -> None:
    # The merge's debug message named the item inside a str.format template,
    # so a brace in a name raised (KeyError / IndexError) whatever the log
    # level, and the unit failed on every run.
    entities = [
        Entity(id="e", name=name, type="ROLE", text_unit_ids=[f"t{n}"])
        for n in range(2)
    ]
    relationships = [
        Relationship(
            id="r",
            source_id="e",
            target_id="f",
            source_name=name,
            target_name="Buyer",
            type=name,
            text_unit_ids=[f"t{n}"],
        )
        for n in range(2)
    ]
    claims = [
        Claim(
            id="c",
            subject_id="",
            subject_name=name,
            object_id="",
            object_name=name,
            type="LEGAL",
            text_unit_ids=[f"t{n}"],
        )
        for n in range(2)
    ]

    assert [e.name for e in merge_entity_contributions(entities)] == [name]
    assert len(merge_relationship_contributions(relationships)) == 1
    assert len(merge_claim_contributions(claims)) == 1


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("empty", ["", None])
def test_an_empty_value_casts_no_vote(
    order: tuple[int, ...], empty: str | None
) -> None:
    # A missing or empty element (an empty <type/>) is no vote: "" sorts
    # first, so it won every tie it took part in.
    entities = [
        Entity(id="e", name=name, type=type_, text_unit_ids=[f"t{n}"])
        for n, (name, type_) in enumerate([("Acme", "ORGANIZATION"), ("", empty)])
    ]
    claims = [
        Claim(
            id="c",
            subject_id="e",
            subject_name="Acme",
            object_name=object_name,
            type=type_,
            status=status,
            description=description,
            text_unit_ids=[f"t{n}"],
        )
        for n, (object_name, type_, status, description) in enumerate(
            [("Buyer", "PAYS", "TRUE", "Acme pays."), ("", "", empty, empty)]
        )
    ]

    (entity,) = merge_entity_contributions([entities[n] for n in order])
    (claim,) = merge_claim_contributions([claims[n] for n in order])

    assert (entity.name, entity.type) == ("Acme", "organization")
    assert (claim.object_name, claim.type, claim.status, claim.description) == (
        "Buyer",
        "pays",
        "true",
        "Acme pays.",
    )


def test_a_value_no_statement_gives_stays_empty() -> None:
    entities = [
        Entity(id="e", name="Acme", type=t, text_unit_ids=[f"t{n}"])
        for n, t in enumerate(["", None])
    ]

    (merged,) = merge_entity_contributions(entities)

    assert merged.type == ""
