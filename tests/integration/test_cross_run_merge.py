# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-run merge: a delta commit unions with existing graph state (AWS-free).

IncrementalIndexer.commit reads the existing entities/relationships back
from the graph store and merges them with the delta (description /
text_unit_ids union, frequency recompute) instead of overwriting, and
records the surviving ids in the registry lineage. Validated against the
in-memory fakes (which support read-back).
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    build_document_lineage,
)
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.claim_resolver import resolved_claim_id
from unified_kg_rag.domain.ingestion.delta_detector import detect_delta
from unified_kg_rag.domain.models import (
    Claim,
    Community,
    Config,
    Document,
    DocumentDelta,
    DocumentLineage,
    Entity,
    Relationship,
    TextUnit,
)

pytestmark = pytest.mark.integration


def _harness(*, fuzzy: bool = False):
    config = Config()
    config.processing.fuzzy_entity_resolution = fuzzy
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    store = FakeDocStatusStore()
    return IncrementalIndexer(store, manager), graph, store


def _commit_entity(incremental: IncrementalIndexer, doc_id: str, entity: Entity):
    incremental.commit(
        lineages=[DocumentLineage(doc_id=doc_id)],
        fingerprints={doc_id: f"hash-{doc_id}"},
        entities=[entity],
    )


def test_cross_run_merge_unions_descriptions_and_text_units() -> None:
    incremental, graph, _ = _harness()

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


def test_a_fuzzy_merge_writes_only_the_kept_ids() -> None:
    # A fuzzy merge folds the delta entity "e-new" into the stored "e-old", and
    # the delta edge into the stored edge with the same (source, target, type):
    # nothing is written under the delta's ids.
    incremental, graph, store = _harness(fuzzy=True)
    incremental.commit(
        lineages=[
            DocumentLineage(
                doc_id="doc-old",
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


def test_delta_entity_merges_into_a_corrected_entity_with_its_id() -> None:
    # A gleaning ENTITY_CORRECTION renames an entity in place and keeps the id
    # derived from the old name. A later doc naming the old form gets that same
    # id; the merge must fold it into the stored entity, not keep both under
    # one id (the second write would drop the first one's text units).
    incremental, graph, _ = _harness()
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
    # An earlier version kept the id when a RELATIONSHIP_CORRECTION changed an
    # edge's type/direction (corrections now re-key it). A later doc
    # extracting the uncorrected edge gets that stored id; the merge must
    # return one edge for it, carrying both docs' text units.
    incremental, _, _ = _harness()
    incremental.commit(
        lineages=[DocumentLineage(doc_id="doc-a")],
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


def test_failed_read_back_keeps_a_deleted_docs_registry_row(mocker) -> None:
    # Deleting doc-b strips its text unit from the entity it shares with doc-a.
    # If reading that entity back fails, the strip did not happen: the removal
    # must count as failed so doc-b's row (its lineage) is kept for a retry.
    incremental, graph, store = _harness()
    for doc_id, unit in (("doc-a", "t1"), ("doc-b", "t2")):
        incremental.commit(
            lineages=[DocumentLineage(doc_id=doc_id, text_unit_ids=[unit])],
            fingerprints={doc_id: f"h-{doc_id}"},
            entities=[Entity(id="e1", name="Vendor", text_unit_ids=[unit])],
        )
    mocker.patch.object(
        graph, "read_entities", side_effect=RuntimeError("connection reset")
    )

    removed = incremental.remove_changed_and_deleted(DocumentDelta(deleted=["doc-b"]))

    assert removed is False
    assert store.get("doc-b") is not None


def test_failed_read_back_fails_the_commit_before_any_write(mocker) -> None:
    # Overwriting on a failed read would drop the stored entity's lineage;
    # failing first leaves the doc unrecorded, so the next run retries it.
    incremental, graph, store = _harness()
    _commit_entity(incremental, "doc-a", Entity(id="e1", name="Vendor"))
    mocker.patch.object(
        graph, "read_entities", side_effect=RuntimeError("connection reset")
    )

    with pytest.raises(RuntimeError):
        _commit_entity(incremental, "doc-b", Entity(id="e1", name="Vendor"))

    assert store.get("doc-b") is None


def test_a_remapped_entity_rewrites_its_edges_only_for_its_new_rank() -> None:
    # "Acme Inc" merges into the stored hub "Acme": its incident edges are
    # read to find a match; the delta's edge (new here) is written, and the
    # hub's stored edges only for the rank its new edge gives them.
    incremental, graph, _ = _harness()
    spokes = [f"e-{i}" for i in range(3)]
    incremental.commit(
        lineages=[DocumentLineage(doc_id="doc-a")],
        fingerprints={"doc-a": "h-a"},
        entities=[
            Entity(id="e-hub", name="Acme", text_unit_ids=["t1"]),
            *(Entity(id=s, name=s, text_unit_ids=["t1"]) for s in spokes),
        ],
        relationships=[
            Relationship(id=f"r-{s}", source_id="e-hub", target_id=s, type="SUPPLIES")
            for s in spokes
        ],
    )
    merged = incremental.indexing_manager.merge_with_existing_graph(
        [
            Entity(id="e-new", name="Acme Inc", text_unit_ids=["t2"]),
            Entity(id="e-bank", name="Bank", text_unit_ids=["t2"]),
        ],
        [Relationship(id="r-bank", source_id="e-new", target_id="e-bank", type="PAYS")],
    )

    edge, *spoke_edges = merged.relationships or []
    assert (edge.id, edge.source_id) == ("r-bank", "e-hub")
    # The hub's rank still counts its stored edges, whose own rank follows.
    ranks = {e.id: e.rank for e in merged.entities or []}
    assert ranks == {"e-hub": 4, "e-bank": 1}
    assert {(r.id, r.rank) for r in spoke_edges} == {(f"r-{s}", 5) for s in spokes}
    assert graph.ids("relationships") == {f"r-{s}" for s in spokes}


def test_communities_and_claims_follow_the_merge_remap() -> None:
    # "Acme Inc" merges into the stored "Acme" (another id), and its edge into
    # the stored edge: the delta's community and claim must reference the
    # kept ids, not ids that are never written.
    incremental, graph, _ = _harness()
    vector = incremental.indexing_manager.opensearch_indexer
    incremental.commit(
        lineages=[DocumentLineage(doc_id="doc-a")],
        fingerprints={"doc-a": "h-a"},
        entities=[
            Entity(id="e-old", name="Acme", text_unit_ids=["t1"]),
            Entity(id="e-buyer", name="Buyer", text_unit_ids=["t1"]),
        ],
        relationships=[
            Relationship(
                id="r-old", source_id="e-old", target_id="e-buyer", type="SUPPLIES"
            )
        ],
    )
    incremental.commit(
        lineages=[DocumentLineage(doc_id="doc-b")],
        fingerprints={"doc-b": "h-b"},
        entities=[
            Entity(id="e-new", name="Acme Inc", text_unit_ids=["t2"]),
            Entity(id="e-buyer", name="Buyer", text_unit_ids=["t2"]),
        ],
        relationships=[
            Relationship(
                id="r-new", source_id="e-new", target_id="e-buyer", type="SUPPLIES"
            )
        ],
        communities=[
            Community(
                id="c1",
                name="Community 1",
                level="0",
                parent="",
                children=[],
                entity_ids=["e-new", "e-buyer", "e-old"],
                relationship_ids=["r-new"],
            )
        ],
        claims=[
            Claim(
                id="cl1",
                subject_id="e-new",
                subject_name="Acme Inc",
                object_id="e-buyer",
                object_name="Buyer",
                type="SUPPLIES",
            )
        ],
    )

    community = graph.data["communities"][("default", "c1")]
    assert community.entity_ids == ["e-old", "e-buyer"]
    assert community.relationship_ids == ["r-old"]
    # Re-keyed to the id the full build gives a claim about e-old.
    (claim,) = vector.data["claims"].values()
    assert (claim.subject_id, claim.object_id) == ("e-old", "e-buyer")
    assert claim.id == resolved_claim_id(claim)


def test_an_entity_an_unchanged_document_states_survives_its_removal() -> None:
    # With the removed indexing.cross_run_merge: false, run 2 overwrote the
    # shared entity with b's text units only, and removing b in run 3 then
    # deleted it although a still stated it.
    config = Config.model_validate({"indexing": {"cross_run_merge": False}})
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    incremental = IncrementalIndexer(
        FakeDocStatusStore(),
        IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph),
    )

    def doc(path: str, text: str) -> Document:
        return Document(
            page_content=text, document_id=path, file_name=path[1:], file_path=path
        )

    def run(docs: list[Document], units: list[TextUnit], entities: list[Entity]):
        delta, fingerprints = detect_delta(docs, incremental.doc_status)
        incremental.remove_changed_and_deleted(delta)
        cited = {unit.document_ids[0] for unit in units}
        incremental.commit(
            lineages=build_document_lineage(
                documents=[d for d in docs if d.document_id in cited],
                text_units=units,
            ),
            fingerprints=fingerprints,
            text_units=units,
            entities=entities,
            relationships=[],
        )

    a, b = doc("/a.txt", "x"), doc("/b.txt", "y")
    ta = TextUnit(id="tu-a", text="x", document_ids=[a.document_id])
    tb = TextUnit(id="tu-b", text="y", document_ids=[b.document_id])
    run([a, b], [ta, tb], [Entity(id="e-s", name="S", text_unit_ids=["tu-a", "tu-b"])])
    b2 = doc("/b.txt", "y2")
    tb2 = TextUnit(id="tu-b2", text="y2", document_ids=[b2.document_id])
    run([a, b2], [tb2], [Entity(id="e-s", name="S", text_unit_ids=["tu-b2"])])
    delta, _ = detect_delta([a], incremental.doc_status)
    incremental.remove_changed_and_deleted(delta)

    assert "e-s" in graph.ids("entities")
