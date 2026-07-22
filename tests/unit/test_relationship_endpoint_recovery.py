# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for relationship endpoint recovery (AWS-free).

Regression cover for the relationship-drop bug (same class as the claim bug):
a relationship must NOT be discarded just because its endpoint entity was not
listed in the same chunk's entity block — endpoint ids are name-derived and the
entity usually exists globally or can be materialized as a stub.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.models import Config, Entity, Relationship, TextUnit

pytestmark = pytest.mark.unit


def _text_unit() -> TextUnit:
    return TextUnit(id="tu1", short_id="tu1", text="Alice founded Acme.")


def test_relationship_kept_when_endpoint_not_in_local_map() -> None:
    proc = BaseProcessor(config=Config())
    # The local entity map is EMPTY (LLM listed the relation but not the entities).
    rel = proc.parse_relationship_data(
        {"source": "Alice", "target": "Acme", "type": "FOUNDED"},
        _text_unit(),
        entity_name_to_id={},
    )
    assert rel is not None  # not dropped
    # Endpoint ids derived deterministically from the names.
    assert rel.source_id == BaseProcessor._generate_entity_id("alice")
    assert rel.target_id == BaseProcessor._generate_entity_id("acme")


def test_local_map_id_is_preferred_and_matches_derived() -> None:
    proc = BaseProcessor(config=Config())
    derived = BaseProcessor._generate_entity_id("alice")
    rel = proc.parse_relationship_data(
        {"source": "Alice", "target": "Acme", "type": "FOUNDED"},
        _text_unit(),
        entity_name_to_id={"alice": derived},
    )
    assert rel is not None
    assert rel.source_id == derived


def test_still_dropped_when_names_or_type_missing() -> None:
    proc = BaseProcessor(config=Config())
    assert (
        proc.parse_relationship_data(
            {"source": "Alice", "target": "", "type": "FOUNDED"},
            _text_unit(),
            entity_name_to_id={},
        )
        is None
    )


def test_materialize_endpoints_creates_stub_entities() -> None:
    from unified_kg_rag.adapters.ingestion.graph_extractor import GraphExtractor

    extractor = GraphExtractor.__new__(GraphExtractor)
    # Only 'alice' was extracted; 'acme' is referenced only by the relationship.
    entities = [Entity(id=BaseProcessor._generate_entity_id("alice"), name="alice")]
    acme_id = BaseProcessor._generate_entity_id("acme")
    rels = [
        Relationship(
            id="r1",
            source_id=entities[0].id,
            source_name="alice",
            target_id=acme_id,
            target_name="acme",
            type="FOUNDED",
            text_unit_ids=["tu1"],
        )
    ]
    result = extractor._materialize_relationship_endpoints(entities, rels)
    ids = {e.id for e in result}
    assert acme_id in ids  # stub created for the relationship-only endpoint
    assert len(result) == 2
