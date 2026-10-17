# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stub endpoint entity is written under its relationship's index suffix.

A relationship can name an endpoint no entity of the batch has; the merge of
the text units' records turns it into a stub entity. The stub must carry the relationship's ``index``
(and ``filters``) attribute, or it lands in the default index while the edge is
written under the relationship's suffix, where Neptune finds no endpoint vertex
and the edge write fails on every retry.
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.stores import FakeGraphStore
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.contributions import merge_contributions
from unified_kg_rag.domain.ingestion.graph_resolver import GraphResolver
from unified_kg_rag.domain.ingestion.relationship_endpoints import (
    cite_relationship_endpoints,
)
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    Relationship,
    TextUnitExtraction,
)
from unified_kg_rag.ports.indexer import BaseIndexer

pytestmark = pytest.mark.unit

_PLACEMENT = {"index": "tenantx", "filters": {"region": "north"}}
VENDOR = BaseProcessor._generate_entity_id("Vendor")
DEPOT = BaseProcessor._generate_entity_id("Depot")


def _graph() -> tuple[list[Entity], list[Relationship]]:
    vendor = Entity(
        id=VENDOR,
        name="Vendor",
        type="ORG",
        text_unit_ids=["t1"],
        attributes=dict(_PLACEMENT),
    )
    # Depot is named only inside the relationship.
    edge = Relationship(
        id=BaseProcessor._generate_relationship_id("Vendor", "Depot", "SHIPS_TO"),
        source_id=VENDOR,
        source_name="Vendor",
        target_id=DEPOT,
        target_name="Depot",
        type="SHIPS_TO",
        text_unit_ids=["t1"],
        attributes=dict(_PLACEMENT),
    )
    return [vendor], [edge]


def test_stub_inherits_the_relationship_placement() -> None:
    entities, relationships = _graph()

    result, stubs = cite_relationship_endpoints(entities, relationships)

    assert stubs == 1
    stub = next(e for e in result if e.id == DEPOT)
    assert stub.attributes == _PLACEMENT
    assert BaseIndexer.get_suffix(stub) == "tenantx"


def test_stub_without_placement_keeps_no_attributes() -> None:
    entities, relationships = _graph()
    relationships[0].attributes = {"evidence": "x"}

    result, _ = cite_relationship_endpoints(entities, relationships)

    stub = next(e for e in result if e.id == DEPOT)
    assert not stub.attributes
    assert BaseIndexer.get_suffix(stub) == "default"


def test_resolved_stub_and_edge_share_a_suffix_and_the_edge_is_written() -> None:
    # Resolution starts from the merged records, whose merge adds the stub.
    vendor, edge = _graph()
    entities, relationships = merge_contributions(
        [TextUnitExtraction(text_unit_id="t1", entities=vendor, relationships=edge)]
    )

    result, _ = GraphResolver(Config(), use_process_pool=False).resolve_graph(
        entities, relationships
    )

    suffixes = {BaseIndexer.get_suffix(e) for e in result["entities"]}
    assert suffixes == {"tenantx"}
    assert {BaseIndexer.get_suffix(r) for r in result["relationships"]} == {"tenantx"}

    store = FakeGraphStore()
    assert store.upsert_entities(result["entities"]).failed_items == 0
    written = store.upsert_relationships(result["relationships"])
    assert written.failed_items == 0
    assert written.successful_items == 1
    assert {e.id for e in store.read_entities([VENDOR, DEPOT], "tenantx")} == {
        VENDOR,
        DEPOT,
    }
