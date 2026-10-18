# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Community drift across incremental commits (in-memory fakes, AWS-free).

A delta run clusters only its own documents, so an entity an earlier run
already clustered joins a second level-0 community. The cross-run merge
unions the entity's stored level-0 community ids with the delta's, and the
commit reports how many of the delta's clustered entities overlap.
"""

from __future__ import annotations

import logging

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.community_drift import with_level0_community_ids
from unified_kg_rag.domain.models import (
    Community,
    CommunityReport,
    Config,
    DocumentLineage,
    Entity,
    TextUnit,
)

pytestmark = pytest.mark.integration


def _community(cid: str, members: list[str]) -> Community:
    return Community(
        id=cid, name=cid, level="0", parent="", children=[], entity_ids=members
    )


def _commit(incremental: IncrementalIndexer, doc_id: str, community: Community):
    entities = with_level0_community_ids(
        [
            Entity(id=e, name=e, text_unit_ids=[f"t-{doc_id}"])
            for e in community.entity_ids
        ],
        [community],
    )
    incremental.commit(
        lineages=[
            DocumentLineage(
                doc_id=doc_id,
            )
        ],
        fingerprints={doc_id: f"hash-{doc_id}"},
        entities=entities,
        communities=[community],
    )
    return incremental.community_drift


def test_delta_runs_report_cumulative_overlap(caplog) -> None:
    config = Config()
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    incremental = IncrementalIndexer(FakeDocStatusStore(), manager)

    first = _commit(incremental, "d1", _community("c1", ["acme", "f1"]))
    assert first is not None and first.overlapping_entities == 0

    with caplog.at_level(logging.WARNING):
        second = _commit(incremental, "d2", _community("c2", ["acme", "f2"]))
    assert second is not None
    assert (second.overlapping_entities, second.stale_communities) == (1, 1)
    assert second.overlap_share == 0.5
    assert second.rebuild_recommended
    assert "indexing.reset: true" in caplog.text

    third = _commit(incremental, "d3", _community("c3", ["acme", "f3"]))
    # acme now sits in three level-0 communities: two earlier ones are stale.
    assert third is not None and third.stale_communities == 2
    (acme,) = graph.read_entities(["acme"])
    assert acme.community_ids == ["c1", "c2", "c3"]


def test_a_deletion_only_run_reports_the_reports_citing_removed_text() -> None:
    # Regression: a run that only removed text units appended no community, so
    # it reported no drift, while the report kept for its other text unit may
    # still state the removed document's text.
    config = Config()
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    vector.upsert_text_units(
        [TextUnit(id="t-kept", text="kept"), TextUnit(id="t-gone", text="gone")]
    )
    vector.upsert_community_reports(
        [
            CommunityReport(
                id="report-1",
                community_id="c1",
                name="c1",
                title="Both",
                summary="quotes the removed text",
                text_unit_ids=["t-kept", "t-gone"],
            )
        ]
    )
    incremental = IncrementalIndexer(FakeDocStatusStore(), manager)

    manager.remove_text_units("default", ["t-gone"])
    incremental.commit(lineages=[], fingerprints={})

    drift = incremental.community_drift
    assert drift is not None
    assert (drift.outdated_reports, drift.rebuild_recommended) == (1, True)
    assert vector.ids("community_reports") == {"report-1"}
