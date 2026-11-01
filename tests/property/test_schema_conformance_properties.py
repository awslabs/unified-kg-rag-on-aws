# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invariants of the extraction-schema check of one text unit.

Over random statements (declared, respelled, undeclared and empty types;
listed and unlisted endpoints): strict keeps a subset of what align keeps,
align keeps everything, the check is idempotent and does not depend on the
order of the statements, and a kept relationship never names an entity
strict rejected.
"""

from __future__ import annotations

from functools import cache

import pytest
from hypothesis import given
from hypothesis import strategies as st

from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.schema_conformance import SchemaConformance
from unified_kg_rag.domain.models import Entity, ExtractionSchema, Relationship
from unified_kg_rag.shared.utils import entity_key

pytestmark = pytest.mark.property

_ITEMS = {
    "entity_types": ["ORGANIZATION", "PERSON", "PRODUCT"],
    "relationship_types": ["SUPPLIES", "WORKS_FOR", "RELATED_TO"],
    "relationship_patterns": [
        "ORGANIZATION -SUPPLIES-> PRODUCT",
        "PERSON -WORKS_FOR-> ORGANIZATION",
        "ORGANIZATION -SUPPLIES-> ORGANIZATION",
    ],
}
_NAMES = ["Vendor", "Buyer", "Depot", "Widget", "Ghost"]
_ENTITY_TYPES = ["ORGANIZATION", "person", "Product", "vendor", ""]
_RELATIONSHIP_TYPES = ["SUPPLIES", "works for", "RELATED_TO", "KNOWS"]


@cache
def _conformance(mode: str) -> SchemaConformance:
    return SchemaConformance(
        ExtractionSchema.from_items(
            _ITEMS["entity_types"],
            _ITEMS["relationship_types"],
            _ITEMS["relationship_patterns"],
            mode=mode,  # type: ignore[arg-type]
        )
    )


def _entity(name: str, type_: str) -> Entity:
    return Entity(
        id=BaseProcessor._generate_entity_id(name),
        name=name,
        type=_conformance("strict").entity_type(type_),
    )


def _relationship(source: str, target: str, type_: str) -> Relationship:
    type_ = _conformance("strict").relationship_type(type_)
    return Relationship(
        id=BaseProcessor._generate_relationship_id(source, target, type_),
        source_id=BaseProcessor._generate_entity_id(source),
        target_id=BaseProcessor._generate_entity_id(target),
        source_name=source,
        target_name=target,
        type=type_,
    )


_entities = st.lists(
    st.builds(_entity, st.sampled_from(_NAMES[:-1]), st.sampled_from(_ENTITY_TYPES)),
    max_size=6,
)
_relationships = st.lists(
    st.builds(
        _relationship,
        st.sampled_from(_NAMES),
        st.sampled_from(_NAMES),
        st.sampled_from(_RELATIONSHIP_TYPES),
    ).filter(lambda r: r.source_id != r.target_id),
    max_size=6,
)


def _ids(items) -> list[str]:
    return [item.id for item in items]


@given(entities=_entities, relationships=_relationships)
def test_strict_keeps_a_subset_of_align(entities, relationships) -> None:
    align = _conformance("align").conform(entities, relationships)
    strict = _conformance("strict").conform(entities, relationships)
    assert _ids(align.entities) == _ids(entities)
    assert _ids(align.relationships) == _ids(relationships)
    assert align.off_schema_keys == set()
    assert set(_ids(strict.entities)) <= set(_ids(align.entities))
    assert set(_ids(strict.relationships)) <= set(_ids(align.relationships))
    # Counted alike in both modes; strict drops what it counts.
    assert strict.entities_off_schema == strict.entities_dropped_off_schema
    assert align.entities_off_schema == strict.entities_off_schema


@given(entities=_entities, relationships=_relationships)
def test_strict_drops_every_off_schema_statement(entities, relationships) -> None:
    conformance = _conformance("strict")
    result = conformance.conform(entities, relationships)
    assert not any(conformance.entity_off_schema(e.type) for e in result.entities)
    assert not any(
        conformance.relationship_off_schema(r.type) for r in result.relationships
    )
    for rel in result.relationships:
        keys = {entity_key(rel.source_name), entity_key(rel.target_name)}
        assert not keys & result.off_schema_keys
    # A rejected key has no statement left.
    assert not result.off_schema_keys & {entity_key(e.name) for e in result.entities}


@pytest.mark.parametrize("mode", ["off", "align", "strict"])
@given(entities=_entities, relationships=_relationships, data=st.data())
def test_conform_is_idempotent_and_order_independent(
    mode: str, entities, relationships, data
) -> None:
    conformance = _conformance(mode)
    once = conformance.conform(entities, relationships)
    twice = conformance.conform(once.entities, once.relationships)
    assert _ids(twice.entities) == _ids(once.entities)
    assert _ids(twice.relationships) == _ids(once.relationships)
    shuffled = conformance.conform(
        data.draw(st.permutations(entities)), data.draw(st.permutations(relationships))
    )
    assert sorted(_ids(shuffled.entities)) == sorted(_ids(once.entities))
    assert sorted(_ids(shuffled.relationships)) == sorted(_ids(once.relationships))
    assert shuffled.off_schema_keys == once.off_schema_keys
