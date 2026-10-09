# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-run merge: a delta commit unions with existing graph state (AWS-free).

When ``indexing.cross_run_merge`` is on, IncrementalIndexer.commit reads the
existing entities/relationships back from the graph store and merges them with
the delta (description / text_unit_ids union, frequency recompute) instead of
overwriting, and records the surviving ids in the registry lineage. Validated
against the in-memory fakes (which support read-back).
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.models import (
    Config,
    DocumentLineage,
    Entity,
    Relationship,
)

pytestmark = pytest.mark.integration


def _harness(*, cross_run_merge: bool, fuzzy: bool = False):
    config = Config()
    config.indexing.cross_run_merge = cross_run_merge
    config.indexing.cross_run_fuzzy_merge = fuzzy
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    store = FakeDocStatusStore()
    return IncrementalIndexer(store, manager), graph, store


def _commit_entity(incremental: IncrementalIndexer, doc_id: str, entity: Entity):
    incremental.commit(
        lineages=[DocumentLineage(doc_id=doc_id, entity_ids=[entity.id])],
        fingerprints={doc_id: f"hash-{doc_id}"},
        entities=[entity],
    )


def test_cross_run_merge_unions_descriptions_and_text_units() -> None:
    incremental, graph, _ = _harness(cross_run_merge=True)

    # Run 1: entity 'vendor' seen in text unit t1 with description A.
    _commit_entity(
        incremental,
        "doc-a",
        Entity(id="e1", name="vendor", description="A", text_unit_ids=["t1"]),
    )
    # Run 2: same entity (by name) seen in t2 with description B.
    _commit_entity(
        incremental,
        "doc-b",
        Entity(id="e1", name="vendor", description="B", text_unit_ids=["t2"]),
    )

    (stored,) = graph.read_entities(["e1"])
    # Descriptions unioned (not overwritten) and text units accumulated.
    assert stored.description == "A\nB"
    assert set(stored.text_unit_ids or []) == {"t1", "t2"}


def test_without_flag_overwrites() -> None:
    incremental, graph, _ = _harness(cross_run_merge=False)

    _commit_entity(
        incremental,
        "doc-a",
        Entity(id="e1", name="vendor", description="A", text_unit_ids=["t1"]),
    )
    _commit_entity(
        incremental,
        "doc-b",
        Entity(id="e1", name="vendor", description="B", text_unit_ids=["t2"]),
    )

    (stored,) = graph.read_entities(["e1"])
    # Overwrite semantics: only the latest delta's values remain.
    assert stored.description == "B"
    assert set(stored.text_unit_ids or []) == {"t2"}


def test_lineage_records_the_ids_the_merge_kept() -> None:
    # A fuzzy merge folds the delta entity "e-new" into the stored "e-old", and
    # the delta edge into the stored edge with the same (source, target, type).
    # The registry must point at the ids actually written, or deleting the doc
    # later would miss them.
    incremental, graph, store = _harness(cross_run_merge=True, fuzzy=True)
    incremental.commit(
        lineages=[
            DocumentLineage(
                doc_id="doc-old",
                entity_ids=["e-old", "e-buyer"],
                relationship_ids=["r-old"],
            )
        ],
        fingerprints={"doc-old": "h-old"},
        entities=[
            Entity(id="e-old", name="Acme Corporation", text_unit_ids=["t-old"]),
            Entity(id="e-buyer", name="Buyer", text_unit_ids=["t-old"]),
        ],
        relationships=[
            Relationship(
                id="r-old",
                source_id="e-old",
                target_id="e-buyer",
                type="SUPPLIES",
                text_unit_ids=["t-old"],
            )
        ],
    )

    incremental.commit(
        lineages=[
            DocumentLineage(
                doc_id="doc-new",
                entity_ids=["e-buyer", "e-new"],
                relationship_ids=["r-new"],
            )
        ],
        fingerprints={"doc-new": "h-new"},
        entities=[
            Entity(id="e-new", name="Acme Corporatio", text_unit_ids=["t-new"]),
            Entity(id="e-buyer", name="Buyer", text_unit_ids=["t-new"]),
        ],
        relationships=[
            Relationship(
                id="r-new",
                source_id="e-new",
                target_id="e-buyer",
                type="SUPPLIES",
                text_unit_ids=["t-new"],
            )
        ],
    )

    assert graph.ids("entities") == {"e-old", "e-buyer"}
    assert graph.ids("relationships") == {"r-old"}
    record = store.get("doc-new")
    assert record is not None
    assert record.entity_ids == ["e-buyer", "e-old"]
    assert record.relationship_ids == ["r-old"]


def test_delta_entity_merges_into_a_corrected_entity_with_its_id() -> None:
    # A gleaning ENTITY_CORRECTION renames an entity in place and keeps the id
    # derived from the old name. A later doc naming the old form gets that same
    # id; the merge must fold it into the stored entity, not keep both under
    # one id (the second write would drop the first one's text units).
    incremental, graph, _ = _harness(cross_run_merge=True)
    _commit_entity(
        incremental,
        "doc-a",
        Entity(id="e1", name="Acme Corporation", description="A", text_unit_ids=["t1"]),
    )
    _commit_entity(
        incremental,
        "doc-b",
        Entity(id="e1", name="Acme Corp", description="B", text_unit_ids=["t2"]),
    )

    (stored,) = graph.read_entities(["e1"])
    assert stored.name == "Acme Corporation"
    assert set(stored.text_unit_ids or []) == {"t1", "t2"}
    assert stored.description == "A\nB"


def test_delta_relationship_merges_into_a_corrected_edge_with_its_id() -> None:
    # A RELATIONSHIP_CORRECTION changes type/direction in place and keeps the
    # id. A later doc extracting the uncorrected edge gets that id; the merge
    # must return one edge for it, carrying both docs' text units.
    incremental, _, _ = _harness(cross_run_merge=True)
    incremental.commit(
        lineages=[DocumentLineage(doc_id="doc-a", relationship_ids=["r1"])],
        fingerprints={"doc-a": "h-a"},
        entities=[
            Entity(id="e-v", name="Vendor", text_unit_ids=["t1"]),
            Entity(id="e-b", name="Buyer", text_unit_ids=["t1"]),
        ],
        relationships=[
            Relationship(
                id="r1",
                source_id="e-b",
                target_id="e-v",
                type="PAYS",
                text_unit_ids=["t1"],
            )
        ],
    )
    delta = [
        Relationship(
            id="r1",
            source_id="e-v",
            target_id="e-b",
            type="SUPPLIES",
            text_unit_ids=["t2"],
        )
    ]

    merged = incremental.indexing_manager.merge_with_existing_graph(None, delta)

    assert merged.relationships is not None
    (edge,) = merged.relationships
    assert (edge.id, edge.source_id, edge.target_id) == ("r1", "e-b", "e-v")
    assert edge.type == "PAYS"
    assert set(edge.text_unit_ids or []) == {"t1", "t2"}
