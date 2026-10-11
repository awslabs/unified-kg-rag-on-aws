# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entity rank = degree, relationship rank = combined degree (MS GraphRAG)."""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.graph_rank import with_graph_ranks
from unified_kg_rag.domain.ingestion.graph_resolver import GraphResolver
from unified_kg_rag.domain.models import (
    Config,
    DocumentLineage,
    Entity,
    Relationship,
)

pytestmark = pytest.mark.unit


def _rel(rid: str, source: str, target: str, rel_type: str = "REL") -> Relationship:
    return Relationship(id=rid, source_id=source, target_id=target, type=rel_type)


def test_rank_is_degree_and_combined_degree() -> None:
    entities = [Entity(id=i, name=i) for i in ("a", "b", "c", "d")]
    relationships = [
        _rel("r1", "a", "b"),
        # Another relationship to the same neighbour counts it once.
        _rel("r2", "b", "a", "PAYS"),
        _rel("r3", "a", "c"),
        _rel("r4", "c", "c"),
    ]
    ranked_entities, ranked_relationships = with_graph_ranks(entities, relationships)

    assert {e.id: e.rank for e in ranked_entities} == {"a": 2, "b": 1, "c": 1, "d": 0}
    assert {r.id: r.rank for r in ranked_relationships} == {
        "r1": 3,
        "r2": 3,
        "r3": 3,
        "r4": 2,
    }
    # The inputs are left as they were.
    assert entities[0].rank == 1


def test_stored_neighbors_join_the_delta_ones() -> None:
    (entity,), (rel,) = with_graph_ranks(
        [Entity(id="a", name="a")],
        [_rel("r1", "a", "b")],
        stored_neighbors={"a": ["b", "c", "a"], "b": ["a"]},
    )
    assert entity.rank == 2
    assert rel.rank == 3


def test_full_build_resolution_sets_ranks() -> None:
    resolver = GraphResolver(Config(), max_workers=1, use_process_pool=False)
    result, _ = resolver.resolve_graph(
        [Entity(id=i, name=n) for i, n in (("a", "Vendor"), ("b", "Buyer"))],
        [_rel("r1", "a", "b")],
    )
    assert [e.rank for e in result["entities"]] == [1, 1]
    assert [r.rank for r in result["relationships"]] == [2]


def test_incremental_merge_ranks_over_the_stored_graph() -> None:
    config = Config()
    graph = FakeGraphStore()
    manager = IndexingManager(
        config=config,
        vector_indexer=FakeVectorStore(opensearch_config=config.indexing.opensearch),
        graph_indexer=graph,
    )
    incremental = IncrementalIndexer(FakeDocStatusStore(), manager)
    names = {"a": "Vendor", "b": "Buyer", "c": "Carrier"}

    def commit(doc_id: str, relationship: Relationship) -> None:
        ids = [relationship.source_id, relationship.target_id]
        incremental.commit(
            lineages=[DocumentLineage(doc_id=doc_id, entity_ids=ids)],
            fingerprints={doc_id: f"h-{doc_id}"},
            entities=[Entity(id=i, name=names[i], text_unit_ids=[doc_id]) for i in ids],
            relationships=[relationship],
        )

    commit("d1", _rel("r1", "a", "b"))
    commit("d2", _rel("r2", "a", "c"))

    ranks = {e.id: e.rank for e in graph.read_entities(["a", "b", "c"])}
    assert ranks == {"a": 2, "b": 1, "c": 1}
    (r2,) = graph.read_relationships(["r2"])
    assert r2.rank == 3
