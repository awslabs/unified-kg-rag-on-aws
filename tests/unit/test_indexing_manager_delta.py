# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for IndexingManager delta routing (M2).

These verify that the incremental path dispatches to the indexers' idempotent
``upsert_*`` / ``delete_by_id`` methods (not the full-rebuild ``index_*``), with
the indexers themselves mocked so no AWS clients are constructed.
"""

from __future__ import annotations

import pytest

from aws_graphrag.domain.models import Entity, Relationship
from aws_graphrag.ports.indexer import IndexingStats

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
    neptune_indexer.upsert_entities.return_value = IndexingStats()
    neptune_indexer.upsert_relationships.return_value = IndexingStats()
    neptune_indexer.index_communities.return_value = IndexingStats()
    neptune_indexer.delete_by_id.return_value = IndexingStats()

    mocker.patch(
        "aws_graphrag.application.storage.indexing_manager.OpenSearchIndexer",
        return_value=os_indexer,
    )
    mocker.patch(
        "aws_graphrag.application.storage.indexing_manager.NeptuneIndexer",
        return_value=neptune_indexer,
    )
    from aws_graphrag.application.storage.indexing_manager import IndexingManager
    from aws_graphrag.domain.models import Config

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

    neptune_indexer.delete_by_id.assert_called_once_with(["e1", "r1", "t1"])
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
