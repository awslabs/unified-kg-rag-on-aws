# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stored items' content derived from their text units' current records.

What a stored entity, relationship or claim says must follow the text units
it cites now: a removed unit's description lines, type votes and claim text
go with it, a name follows the units' support, and an item a record is
missing for keeps its stored content.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.derivation import derive
from unified_kg_rag.domain.ingestion.graph_resolver import ENTITY_KEYS_ATTRIBUTE
from unified_kg_rag.domain.ingestion.relationship_weights import (
    apply_text_unit_weights,
)
from unified_kg_rag.domain.models import (
    Claim,
    Config,
    Entity,
    Relationship,
    TextUnitExtraction,
)
from unified_kg_rag.shared.utils import entity_key

pytestmark = pytest.mark.unit


def _id(name: str) -> str:
    return BaseProcessor._generate_entity_id(name)


def _entity(name: str, unit: str, type_: str = "ORG") -> Entity:
    return Entity(
        id=_id(name),
        name=name,
        type=type_,
        description=f"{name} in {unit}",
        text_unit_ids=[unit],
    )


def _edge(source: str, target: str, unit: str) -> Relationship:
    edge = Relationship(
        id=BaseProcessor._generate_relationship_id(source, target, "SUPPLIES"),
        source_id=_id(source),
        target_id=_id(target),
        source_name=source,
        target_name=target,
        type="SUPPLIES",
        description=f"{source} supplies {target} in {unit}",
    )
    apply_text_unit_weights(edge, {unit: 1.0})
    return edge


def _record(unit: str, *entities: Entity, edges=()) -> TextUnitExtraction:
    return TextUnitExtraction(
        text_unit_id=unit, entities=list(entities), relationships=list(edges)
    )


def _stored(name: str, units: list[str], **kw) -> Entity:
    return Entity(
        id=kw.pop("id", _id(name)),
        name=name,
        type=kw.pop("type", "org"),
        description=kw.pop("description", "stale text"),
        text_unit_ids=units,
        **kw,
    )


def test_a_removed_units_description_and_type_vote_are_gone() -> None:
    records = {
        "u2": _record("u2", _entity("Acme", "u2", "PERSON")),
        "u3": _record("u3", _entity("Acme", "u3", "PERSON")),
    }
    stored = _stored("Acme", ["u2", "u3"], description="Acme in u1\nAcme in u2")

    (derived,) = derive(Config(), [stored], [], [], records).entities

    assert derived.description.split("\n") == ["Acme in u2", "Acme in u3"]
    assert derived.type == "person"
    assert (derived.id, derived.text_unit_ids) == (stored.id, ["u2", "u3"])


def test_the_name_follows_the_support_and_the_id_stays() -> None:
    # "Acme" was indexed first; two units now say "Acme Inc".
    records = {
        "u2": _record("u2", _entity("Acme Inc", "u2")),
        "u3": _record("u3", _entity("Acme Inc", "u3")),
        "u4": _record("u4", _entity("Acme", "u4")),
    }
    stored = _stored(
        "Acme",
        ["u2", "u3", "u4"],
        attributes={ENTITY_KEYS_ATTRIBUTE: ["acme", "acme inc"]},
    )

    (derived,) = derive(Config(), [stored], [], [], records).entities

    assert (derived.id, derived.name) == (_id("Acme"), "Acme Inc")
    assert set(derived.attributes[ENTITY_KEYS_ATTRIBUTE]) == {
        entity_key("Acme"),
        entity_key("Acme Inc"),
    }


def test_edges_and_claims_take_the_live_units_text() -> None:
    records = {
        "u2": _record(
            "u2",
            _entity("Acme", "u2"),
            _entity("Buyer", "u2"),
            edges=[_edge("Acme", "Buyer", "u2")],
        )
    }
    claim = Claim(
        id="c",
        subject_id="",
        subject_name="Acme",
        object_name="100",
        type="PAYMENT",
        description="Acme pays 100 (u2)",
        source_text="pays 100",
        text_unit_ids=["u2"],
    )
    stored_edge = _edge("Acme", "Buyer", "u2").model_copy(
        update={"description": "Acme supplies Buyer in u1\nAcme supplies Buyer in u2"}
    )
    stored_claim = Claim(
        id="",
        subject_id=_id("Acme"),
        subject_name="Acme",
        object_name="100",
        type="payment",
        description="Acme pays 120 (u1)\nAcme pays 100 (u2)",
        source_text="pays 120\npays 100",
        text_unit_ids=["u2"],
    )
    from unified_kg_rag.domain.ingestion.claim_resolver import resolved_claim_id

    stored_claim.id = resolved_claim_id(stored_claim)
    entities = [_stored("Acme", ["u2"]), _stored("Buyer", ["u2"])]

    derived = derive(
        Config(),
        entities,
        [stored_edge],
        [stored_claim],
        records,
        {"u2": [claim]},
    )

    (edge,) = derived.relationships
    assert edge.description == "Acme supplies Buyer in u2"
    (derived_claim,) = derived.claims
    assert (derived_claim.description, derived_claim.source_text) == (
        "Acme pays 100 (u2)",
        "pays 100",
    )


def test_an_item_whose_record_is_missing_keeps_its_content(caplog) -> None:
    stored = _stored("Acme", ["u1", "u2"])
    records = {"u2": _record("u2", _entity("Acme", "u2"))}

    derived = derive(Config(), [stored], [], [], records)

    assert derived.entities[0].description == "stale text"
    assert derived.underived == [stored.id]
    assert "without an extraction record" in caplog.text
