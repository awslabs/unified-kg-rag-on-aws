# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One run that changes one document and deletes another (AWS-free).

Artifacts only the changed and the deleted document reference must go: the
removal is planned over both sets at once. Planning each set separately treats
the other as a survivor and leaves such artifacts behind with no text units.
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
from unified_kg_rag.domain.ingestion.delta_detector import compute_doc_id
from unified_kg_rag.domain.models import (
    Config,
    Document,
    Entity,
    Relationship,
    TextUnit,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def harness():
    config = Config()
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    store = FakeDocStatusStore()
    return IncrementalIndexer(store, manager), store, graph, vector


def _doc(path: str, text: str) -> Document:
    return Document(
        page_content=text,
        document_id=path,
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
        file_type="txt",
        total_pages=1,
    )


def _meets(doc: Document, unit_id: str, other: str):
    """Artifacts of a document reading "Vendor meets <other>."."""
    unit = TextUnit(id=unit_id, text=doc.page_content, document_ids=[doc.document_id])
    other_id = f"e-{other.lower()}"
    entities = [
        Entity(id="e-vendor", name="Vendor", text_unit_ids=[unit_id]),
        Entity(id=other_id, name=other, text_unit_ids=[unit_id]),
    ]
    edge = Relationship(
        id=f"r-vendor-{other.lower()}",
        source_id="e-vendor",
        target_id=other_id,
        type="MEETS",
        weight=1.0,
        text_unit_ids=[unit_id],
    )
    return [unit], entities, [edge]


def _commit(inc, docs, text_units, entities, relationships) -> None:
    _, fingerprints = inc.plan(docs)
    lineages = build_document_lineage(
        documents=docs,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        communities=[],
        claims=[],
    )
    inc.commit(
        lineages=lineages,
        fingerprints=fingerprints,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
    )


def _merge(*artifacts):
    units, entities, edges = [], {}, []
    for unit_list, entity_list, edge_list in artifacts:
        units += unit_list
        edges += edge_list
        for entity in entity_list:
            if entity.id in entities:
                entities[entity.id].text_unit_ids += entity.text_unit_ids
            else:
                entities[entity.id] = entity
    return units, list(entities.values()), edges


def test_artifacts_shared_only_by_a_changed_and_a_deleted_doc_are_removed(
    harness,
) -> None:
    inc, store, graph, vector = harness
    a, b = _doc("/a.txt", "Vendor meets Depot."), _doc("/b.txt", "Vendor meets Depot.")
    first_a, first_b = _meets(a, "tu-a1", "Depot"), _meets(b, "tu-b1", "Depot")
    units, entities, edges = _merge(first_a, first_b)
    edges = [
        edges[0].model_copy(update={"weight": 2.0, "text_unit_ids": ["tu-a1", "tu-b1"]})
    ]
    _commit(inc, [a, b], units, entities, edges)
    assert graph.ids("entities") == {"e-vendor", "e-depot"}

    # Next run: a.txt now says "Vendor meets Carrier.", b.txt is gone.
    a2 = _doc("/a.txt", "Vendor meets Carrier.")
    delta, _ = inc.plan([a2])
    assert delta.changed == [compute_doc_id("/a.txt")]
    assert delta.deleted == [compute_doc_id("/b.txt")]
    assert inc.remove_changed_and_deleted(delta)
    _commit(inc, [a2], *_meets(a2, "tu-a2", "Carrier"))

    assert graph.ids("entities") == {"e-vendor", "e-carrier"}
    assert graph.ids("relationships") == {"r-vendor-carrier"}
    assert vector.ids("entities") == {"e-vendor", "e-carrier"}
    assert vector.ids("relationships") == {"r-vendor-carrier"}
    assert vector.ids("text_units") == {"tu-a2"}
    (vendor,) = graph.read_entities(["e-vendor"])
    assert vendor.text_unit_ids == ["tu-a2"]
    (edge,) = graph.read_relationships(["r-vendor-carrier"])
    assert edge.weight == 1.0
    assert {r.doc_id for r in store.list_all()} == {compute_doc_id("/a.txt")}


def test_a_survivor_keeps_its_text_units_on_shared_artifacts(harness) -> None:
    inc, store, graph, _ = harness
    a, b, c = (_doc(f"/{n}.txt", "Vendor meets Depot.") for n in "abc")
    units, entities, _ = _merge(
        _meets(a, "tu-a1", "Depot"),
        _meets(b, "tu-b1", "Depot"),
        _meets(c, "tu-c1", "Depot"),
    )
    edge = Relationship(
        id="r-vendor-depot",
        source_id="e-vendor",
        target_id="e-depot",
        type="MEETS",
        weight=3.0,
        text_unit_ids=["tu-a1", "tu-b1", "tu-c1"],
    )
    _commit(inc, [a, b, c], units, entities, [edge])

    a2 = _doc("/a.txt", "Vendor meets Carrier.")
    delta, _ = inc.plan([a2, c])
    assert inc.remove_changed_and_deleted(delta)
    _commit(inc, [a2], *_meets(a2, "tu-a2", "Carrier"))

    # c.txt still says "Vendor meets Depot.": Depot and the edge stay, citing
    # only c's text unit; Vendor cites c's unit and a's new one.
    assert graph.ids("entities") == {"e-vendor", "e-depot", "e-carrier"}
    (depot,) = graph.read_entities(["e-depot"])
    assert depot.text_unit_ids == ["tu-c1"]
    (vendor,) = graph.read_entities(["e-vendor"])
    assert set(vendor.text_unit_ids) == {"tu-c1", "tu-a2"}
    (kept_edge,) = graph.read_relationships(["r-vendor-depot"])
    assert kept_edge.text_unit_ids == ["tu-c1"]
    assert kept_edge.weight == 1.0
    assert compute_doc_id("/b.txt") not in {r.doc_id for r in store.list_all()}
