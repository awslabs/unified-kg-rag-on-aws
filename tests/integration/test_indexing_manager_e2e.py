# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end IndexingManager tests over in-memory fake stores (AWS-free).

Exercises the full-rebuild and delta (upsert) paths plus delete-by-id across the
graph + vector stores, asserting idempotency and that claims are indexed.
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.models import (
    Claim,
    Config,
    Entity,
    Relationship,
    TextUnit,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def manager():
    config = Config()
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    mgr = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    return mgr, graph, vector


def _artifacts():
    text_units = [TextUnit(id="t1", text="alice works at acme")]
    entities = [
        Entity(id="e-alice", name="Alice", text_unit_ids=["t1"]),
        Entity(id="e-acme", name="Acme", text_unit_ids=["t1"]),
    ]
    relationships = [
        Relationship(id="r1", source_id="e-alice", target_id="e-acme", weight=1.0)
    ]
    claims = [
        Claim(
            id="c1",
            subject_id="e-alice",
            subject_name="Alice",
            object_id=None,
            object_name="$100k",
            type="PERFORMANCE",
            description="Alice generated $100k",
        )
    ]
    return text_units, entities, relationships, claims


def test_index_all_data_writes_to_both_stores(manager) -> None:
    mgr, graph, vector = manager
    text_units, entities, relationships, claims = _artifacts()

    mgr.index_all_data(
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        claims=claims,
    )

    # Vector store holds text units, entities, relationships, and claims.
    assert vector.ids("text_units") == {"t1"}
    assert vector.ids("entities") == {"e-alice", "e-acme"}
    assert vector.ids("claims") == {"c1"}
    # Graph store holds entities + relationships.
    assert graph.ids("entities") == {"e-alice", "e-acme"}
    assert graph.ids("relationships") == {"r1"}


def test_delta_upsert_is_idempotent(manager) -> None:
    mgr, graph, vector = manager
    text_units, entities, relationships, claims = _artifacts()

    for _ in range(3):  # re-run the same delta thrice
        mgr.index_delta(
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            claims=claims,
        )

    # No duplication: id-keyed upsert collapses repeats.
    assert vector.ids("entities") == {"e-alice", "e-acme"}
    assert vector.ids("claims") == {"c1"}
    assert graph.ids("relationships") == {"r1"}


def test_delete_documents_removes_by_id_from_both_stores(manager) -> None:
    mgr, graph, vector = manager
    text_units, entities, relationships, claims = _artifacts()
    mgr.index_all_data(
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        claims=claims,
    )

    # Delete entity e-acme's id across stores.
    mgr.delete_documents({"default": ["e-acme"]})

    assert "e-acme" not in vector.ids("entities")
    assert "e-acme" not in graph.ids("entities")
    # e-alice survives.
    assert "e-alice" in vector.ids("entities")

    # Deletion is fanned out per OpenSearch index prefix (text_units, entities,
    # relationships, claims, community_reports) — verify the routing contract.
    deleted_prefixes = {p for p, _ in vector.delete_calls}
    os_cfg = mgr.config.indexing.opensearch
    assert deleted_prefixes == {
        os_cfg.text_units_index_prefix,
        os_cfg.entities_index_prefix,
        os_cfg.relationships_index_prefix,
        os_cfg.claims_index_prefix,
        os_cfg.community_reports_index_prefix,
    }
