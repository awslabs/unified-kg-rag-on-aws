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

    # Inject the mocked indexers via the port-based DI seam (no module-level
    # patching needed — the manager constructs concrete adapters only when none
    # are supplied).
    os_indexer.delete_document_artifacts.side_effect = (
        lambda ids, suffix, extra_relationship_ids=None: {
            f"opensearch_delete_{prefix}_{suffix}": IndexingStats()
            for prefix in (
                "text-units",
                "entities",
                "relationships",
                "claims",
                "community-reports",
            )
        }
    )
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager
    from unified_kg_rag.domain.models import Config

    mgr = IndexingManager(
        config=Config(), vector_indexer=os_indexer, graph_indexer=neptune_indexer
    )
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
    # The manager delegates the per-index vector fan-out to the vector port's
    # cohesive delete_document_artifacts (the prefix layout is the adapter's
    # concern, not the manager's), threading the suffix and no orphan edges.
    os_indexer.delete_document_artifacts.assert_called_once_with(
        ["e1", "r1", "t1"], "default", extra_relationship_ids=[]
    )


def test_delete_documents_skips_empty(manager) -> None:
    mgr, os_indexer, neptune_indexer = manager
    mgr.delete_documents({"default": []})
    neptune_indexer.delete_by_id.assert_not_called()
    os_indexer.delete_document_artifacts.assert_not_called()


def test_delete_documents_cascades_orphan_relationship_ids(manager) -> None:
    # Orphan-edge cleanup: a relationship pointing at a deleted entity is owned
    # (in lineage) by a surviving document, so it is NOT in the exclusive id-set.
    # delete_documents must query Neptune for incident edge ids and pass them as
    # extra_relationship_ids so the vector backend can fold them into the
    # relationship-index deletion (no dangling relationship document survives).
    mgr, os_indexer, neptune_indexer = manager
    neptune_indexer.find_incident_relationship_ids.return_value = ["r_orphan"]

    mgr.delete_documents({"default": ["e_target"]})

    # Incident edges are queried for the deleted entities, scoped to the suffix.
    neptune_indexer.find_incident_relationship_ids.assert_called_once_with(
        ["e_target"], suffix="default"
    )
    # The orphaned incident edge is handed to the vector port as an extra
    # relationship id to delete.
    os_indexer.delete_document_artifacts.assert_called_once_with(
        ["e_target"], "default", extra_relationship_ids=["r_orphan"]
    )


def test_cross_run_merge_threads_entity_id_remap_to_relationships(manager) -> None:
    # When the cross-run entity merge collapses two ids onto one survivor, the
    # delta relationship endpoints must be remapped so edges point at the
    # surviving entity id (the manager previously discarded the remap, leaving
    # relationships dangling on a now-nonexistent id).
    mgr, os_indexer, neptune_indexer = manager

    # Existing entity 'e_canon' and a delta entity 'e_dup' that normalizes to the
    # same name -> merge_entities collapses e_dup -> e_canon and returns a remap.
    survivor = Entity(id="e_canon", name="acme corp", text_unit_ids=["t1"])
    dup = Entity(id="e_dup", name="acme corp", text_unit_ids=["t2"])
    neptune_indexer.read_entities.return_value = [survivor]
    # No existing relationship rows read back — the remap path must still apply.
    neptune_indexer.read_relationships.return_value = []

    delta_rel = Relationship(
        id="r1", source_id="e_dup", target_id="e_other", type="WORKS_AT"
    )
    merged_entities, merged_rels = mgr._merge_with_existing_graph([dup], [delta_rel])

    # The relationship endpoint that referenced the merged-away id is remapped.
    assert merged_rels is not None and len(merged_rels) == 1
    assert merged_rels[0].source_id == "e_canon"
    assert merged_rels[0].target_id == "e_other"
