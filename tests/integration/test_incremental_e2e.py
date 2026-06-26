# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end incremental indexing over fake stores + fake registry (AWS-free).

Drives IncrementalIndexer (the production orchestrator) against the real
IndexingManager wired to in-memory fake graph/vector stores and the fake
doc-status registry, proving the add -> change -> delete cycle reaches the
stores: deltas upsert, deletions propagate, and shared artifacts survive.
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
from unified_kg_rag.domain.models import Config, Document, Entity, TextUnit

pytestmark = pytest.mark.integration


@pytest.fixture
def harness(mocker):
    config = Config()
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    mocker.patch(
        "unified_kg_rag.application.storage.indexing_manager.OpenSearchIndexer",
        return_value=vector,
    )
    mocker.patch(
        "unified_kg_rag.application.storage.indexing_manager.NeptuneIndexer",
        return_value=graph,
    )
    manager = IndexingManager(config=config)
    store = FakeDocStatusStore()
    inc = IncrementalIndexer(store, manager)
    return inc, store, graph, vector


def _doc(path: str, text: str) -> Document:
    return Document(
        page_content=text,
        document_id=path,  # use path as the per-run id for deterministic lineage
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
        file_type="txt",
        total_pages=1,
    )


def _artifacts_for(doc: Document, entity_ids: list[str]):
    tu = TextUnit(
        id=f"tu-{doc.document_id}", text="...", document_ids=[doc.document_id]
    )
    entities = [Entity(id=e, name=e, text_unit_ids=[tu.id]) for e in entity_ids]
    return [tu], entities


def _commit(inc, docs, all_text_units, all_entities, all_relationships=None):
    delta, fingerprints = inc.plan(docs)
    lineages = build_document_lineage(
        documents=docs,
        text_units=all_text_units,
        entities=all_entities,
        relationships=all_relationships or [],
        communities=[],
        claims=[],
    )
    inc.commit(
        lineages=lineages,
        fingerprints=fingerprints,
        text_units=all_text_units,
        entities=all_entities,
        relationships=all_relationships or [],
    )
    return delta


def test_add_then_delete_cycle_reaches_stores(harness) -> None:
    inc, store, graph, vector = harness

    a, b = _doc("/a.txt", "Alice at Acme"), _doc("/b.txt", "Bob at Beta")
    tu_a, ent_a = _artifacts_for(a, ["e-alice"])
    tu_b, ent_b = _artifacts_for(b, ["e-bob"])

    # Initial run: both docs.
    _commit(inc, [a, b], tu_a + tu_b, ent_a + ent_b)
    assert vector.ids("entities") == {"e-alice", "e-bob"}
    assert graph.ids("entities") == {"e-alice", "e-bob"}
    assert {r.doc_id for r in store.list_all()} == {
        compute_doc_id("/a.txt"),
        compute_doc_id("/b.txt"),
    }

    # Second run: /b.txt deleted from the corpus -> its artifacts pruned.
    delta = inc.plan([a])[0]
    assert delta.deleted == [compute_doc_id("/b.txt")]
    inc.remove_deleted(delta)

    assert "e-bob" not in vector.ids("entities")
    assert "e-bob" not in graph.ids("entities")
    assert "e-alice" in vector.ids("entities")  # survivor untouched
    assert {r.doc_id for r in store.list_all()} == {compute_doc_id("/a.txt")}


def test_shared_entity_survives_deletion(harness) -> None:
    inc, store, graph, vector = harness

    a, b = _doc("/a.txt", "x"), _doc("/b.txt", "y")
    tu_a = TextUnit(id="tu-a", text="...", document_ids=[a.document_id])
    tu_b = TextUnit(id="tu-b", text="...", document_ids=[b.document_id])
    # Both documents reference the SAME shared entity (different text units).
    shared = Entity(id="e-shared", name="Shared", text_unit_ids=["tu-a", "tu-b"])

    _commit(inc, [a, b], [tu_a, tu_b], [shared])
    assert vector.ids("entities") == {"e-shared"}

    # Delete /b.txt: e-shared is still referenced by /a.txt -> must NOT be removed.
    delta = inc.plan([a])[0]
    inc.remove_deleted(delta)
    assert "e-shared" in vector.ids("entities")
    assert "e-shared" in graph.ids("entities")
