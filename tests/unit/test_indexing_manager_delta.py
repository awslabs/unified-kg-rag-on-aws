# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for IndexingManager delta routing (M2).

These verify that the incremental path dispatches to the indexers' idempotent
``upsert_*`` / ``delete_by_id`` methods (not the full-rebuild ``index_*``), with
the indexers themselves mocked so no AWS clients are constructed.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.models import Entity, Relationship
from unified_kg_rag.ports.indexer import IndexingStats

pytestmark = pytest.mark.unit


@pytest.fixture
def manager(mocker):
    # Patch the indexer classes before IndexingManager.__init__ instantiates them.
    os_indexer = mocker.MagicMock()
    neptune_indexer = mocker.MagicMock()
    os_indexer.upsert_entities.return_value = IndexingStats()
    os_indexer.upsert_text_units.return_value = IndexingStats()
    os_indexer.upsert_relationships.return_value = IndexingStats()
    os_indexer.index_community_reports.return_value = IndexingStats()
    os_indexer.delete_by_id.return_value = IndexingStats()
    os_indexer.opensearch_config.text_units_index_prefix = "text-units"
    os_indexer.opensearch_config.entities_index_prefix = "entities"
    os_indexer.opensearch_config.relationships_index_prefix = "relationships"
    os_indexer.opensearch_config.claims_index_prefix = "claims"
    os_indexer.opensearch_config.community_reports_index_prefix = "community-reports"
    neptune_indexer.upsert_entities.return_value = IndexingStats()
    neptune_indexer.upsert_relationships.return_value = IndexingStats()
    neptune_indexer.index_communities.return_value = IndexingStats()
    neptune_indexer.delete_by_id.return_value = IndexingStats()
    # No orphaned incident edges by default; orphan-cleanup test overrides this.
    neptune_indexer.find_incident_relationship_ids.return_value = []

    mocker.patch(
        "unified_kg_rag.application.storage.indexing_manager.OpenSearchIndexer",
        return_value=os_indexer,
    )
    mocker.patch(
        "unified_kg_rag.application.storage.indexing_manager.NeptuneIndexer",
        return_value=neptune_indexer,
    )
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager
    from unified_kg_rag.domain.models import Config

    mgr = IndexingManager(config=Config())
    return mgr, os_indexer, neptune_indexer


def test_index_delta_routes_to_upserts(manager) -> None:
    mgr, os_indexer, neptune_indexer = manager
    entities = [Entity(id="e1", name="Alice")]
    relationships = [Relationship(id="r1", source_id="e1", target_id="e2")]

    mgr.index_delta(entities=entities, relationships=relationships)

    os_indexer.upsert_entities.assert_called_once_with(entities)
    neptune_indexer.upsert_entities.assert_called_once_with(entities)
    neptune_indexer.upsert_relationships.assert_called_once_with(relationships)
    # Relationship vectors (LightRAG global) are upserted on a delta run too.
    os_indexer.upsert_relationships.assert_called_once_with(relationships)
    # Full-rebuild entity path must NOT be used on a delta run.
    os_indexer.index_entities.assert_not_called()
    neptune_indexer.index_entities.assert_not_called()
    neptune_indexer.index_relationships.assert_not_called()


def test_delete_documents_routes_to_delete_by_id(manager) -> None:
    mgr, os_indexer, neptune_indexer = manager

    mgr.delete_documents({"default": ["e1", "r1", "t1"]})

    # Suffix is threaded through so the Neptune drop scopes to this suffix's
    # labels (cross-tenant safety: content-hash ids can collide across suffixes).
    neptune_indexer.delete_by_id.assert_called_once_with(
        ["e1", "r1", "t1"], suffix="default"
    )
    # OpenSearch delete is called per vector index prefix: text-units +
    # entities + relationships + claims + community-reports (none of these
    # artifacts — including LightRAG relationship vectors and claim vectors —
    # may be orphaned on document deletion).
    assert os_indexer.delete_by_id.call_count == 5


def test_delete_documents_skips_empty(manager) -> None:
    mgr, os_indexer, neptune_indexer = manager
    mgr.delete_documents({"default": []})
    neptune_indexer.delete_by_id.assert_not_called()
    os_indexer.delete_by_id.assert_not_called()


def test_delete_documents_cascades_orphan_relationship_ids(manager) -> None:
    # Orphan-edge cleanup: a relationship pointing at a deleted entity is owned
    # (in lineage) by a surviving document, so it is NOT in the exclusive id-set.
    # delete_documents must query Neptune for incident edge ids and fold them
    # into the OpenSearch relationship-index deletion so no dangling relationship
    # document survives.
    mgr, os_indexer, neptune_indexer = manager
    neptune_indexer.find_incident_relationship_ids.return_value = ["r_orphan"]

    mgr.delete_documents({"default": ["e_target"]})

    # Incident edges are queried for the deleted entities, scoped to the suffix.
    neptune_indexer.find_incident_relationship_ids.assert_called_once_with(
        ["e_target"], suffix="default"
    )
    # The relationships-index deletion includes BOTH the exclusive id and the
    # orphaned incident edge; other indexes get only the exclusive id.
    calls = {c.args[1]: c.args[0] for c in os_indexer.delete_by_id.call_args_list}
    assert set(calls["relationships"]) == {"e_target", "r_orphan"}
    assert calls["entities"] == ["e_target"]
    assert calls["text-units"] == ["e_target"]
