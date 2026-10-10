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
from unified_kg_rag.domain.models import (
    Config,
    DocStatusRecord,
    Document,
    DocumentLineage,
    Entity,
    Relationship,
    TextUnit,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def harness():
    config = Config()
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
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


def _shared_vendor_corpus(inc):
    """doc a and doc b both mention Vendor and its edge to Buyer."""
    a, b = _doc("/a.txt", "x"), _doc("/b.txt", "y")
    tu_a = TextUnit(id="tu-a", text="...", document_ids=[a.document_id])
    tu_b = TextUnit(id="tu-b", text="...", document_ids=[b.document_id])
    vendor = Entity(
        id="e-vendor",
        name="Vendor",
        description="Vendor supplies parts.",
        text_unit_ids=["tu-a", "tu-b"],
    )
    buyer = Entity(id="e-buyer", name="Buyer", text_unit_ids=["tu-a", "tu-b"])
    supplies = Relationship(
        id="r-supplies",
        source_id="e-vendor",
        target_id="e-buyer",
        type="SUPPLIES",
        weight=2.0,
        text_unit_ids=["tu-a", "tu-b"],
    )
    _commit(inc, [a, b], [tu_a, tu_b], [vendor, buyer], [supplies])
    return a, b


def test_deleting_a_doc_strips_its_text_units_from_shared_artifacts(harness) -> None:
    inc, _, graph, vector = harness
    a, _ = _shared_vendor_corpus(inc)

    inc.remove_deleted(inc.plan([a])[0])

    (vendor,) = graph.read_entities(["e-vendor"])
    assert vendor.text_unit_ids == ["tu-a"]
    # Neptune does not store frequency; the vector index does.
    assert vector.data["entities"]["e-vendor"].frequency == 1
    # The description cannot be re-derived without an LLM call and is kept.
    assert vendor.description == "Vendor supplies parts."
    (edge,) = graph.read_relationships(["r-supplies"])
    assert edge.text_unit_ids == ["tu-a"]
    # tu-b's share of the 2.0 is gone.
    assert edge.weight == 1.0
    assert vector.data["entities"]["e-vendor"].text_unit_ids == ["tu-a"]
    assert vector.data["relationships"]["r-supplies"].text_unit_ids == ["tu-a"]


def test_changed_doc_replaces_its_text_units_on_shared_artifacts(harness) -> None:
    inc, _, graph, vector = harness
    a, _ = _shared_vendor_corpus(inc)
    edited = _doc("/b.txt", "y, edited")
    tu_b2 = TextUnit(id="tu-b2", text="...", document_ids=[edited.document_id])

    delta = inc.plan([a, edited])[0]
    assert inc.prune_changed(delta)
    _commit(
        inc,
        [edited],
        [tu_b2],
        [Entity(id="e-vendor", name="Vendor", text_unit_ids=["tu-b2"])],
    )

    (vendor,) = graph.read_entities(["e-vendor"])
    assert vendor.text_unit_ids == ["tu-a", "tu-b2"]
    assert vector.data["entities"]["e-vendor"].frequency == 2


def test_exclusive_deletion_is_scoped_per_suffix(harness) -> None:
    # "Vendor" has the same id in both tenants. Deleting tenant-a's only
    # document must remove tenant-a's Vendor even though tenant-b's document
    # still references the same id, and must leave tenant-b's Vendor alone.
    inc, store, graph, vector = harness
    for suffix in ("tenant-a", "tenant-b"):
        inc.commit(
            lineages=[
                DocumentLineage(
                    doc_id=f"doc-{suffix}",
                    suffix=suffix,
                    entity_ids=["e-vendor"],
                    text_unit_ids=[f"tu-{suffix}"],
                )
            ],
            fingerprints={f"doc-{suffix}": "h"},
            entities=[
                Entity(
                    id="e-vendor",
                    name="Vendor",
                    text_unit_ids=[f"tu-{suffix}"],
                    attributes={"index": suffix},
                )
            ],
        )

    assert inc.remove_obsolete_artifacts(["doc-tenant-a"])
    inc.doc_status.delete("doc-tenant-a")

    assert graph.read_entities(["e-vendor"], suffix="tenant-a") == []
    (kept,) = graph.read_entities(["e-vendor"], suffix="tenant-b")
    assert kept.text_unit_ids == ["tu-tenant-b"]
    assert {suffix for _, suffix in vector.delete_calls} == {"tenant-a"}
    assert [r.doc_id for r in store.list_all()] == ["doc-tenant-b"]


@pytest.mark.parametrize(
    ("survivor_scope", "kept"),
    [
        # Another additional_suffix: same item suffix, another namespace.
        ("default-y|/corpus", False),
        # The same namespace, another source scope.
        ("default-x|/other", True),
        # Written before scopes existed: its namespace is unknown, so it keeps
        # the id rather than risk deleting what it references.
        (None, True),
    ],
)
def test_survivors_retain_artifacts_only_in_their_namespace(
    harness, survivor_scope, kept
) -> None:
    inc, store, _, _ = harness
    store.put(
        DocStatusRecord(
            doc_id="doc-x",
            content_hash="h",
            scope="default-x|/corpus",
            entity_ids=["e-vendor"],
            text_unit_ids=["tu-x"],
        )
    )
    store.put(
        DocStatusRecord(
            doc_id="doc-survivor",
            content_hash="h",
            scope=survivor_scope,
            entity_ids=["e-vendor"],
            text_unit_ids=["tu-survivor"],
        )
    )

    (removal,) = inc._plan_removal(["doc-x"]).values()

    assert ("e-vendor" in removal.exclusive_ids) is not kept
    assert removal.shared_entity_ids == (["e-vendor"] if kept else [])
    assert "tu-x" in removal.exclusive_ids


@pytest.mark.parametrize(
    ("survivor_namespace", "file_path", "visible", "kept"),
    [
        # Written before scopes existed under the legacy key of another
        # namespace a scope in the registry names.
        ("default-y", "a.txt", True, False),
        # Under the legacy key of a namespace no scope names: its namespace
        # is unknown, so it retains.
        ("default-y", "a.txt", False, True),
        # Under the removed record's namespace's legacy key.
        ("default-x", "a.txt", False, True),
        # Under the run's namespace's legacy key: not the removed record's.
        ("default-run", "a.txt", False, False),
        # No file_path (initial release): namespace unknown, so it retains.
        ("default-y", None, True, True),
    ],
)
def test_a_scopeless_survivor_retains_only_in_its_legacy_namespace(
    harness, survivor_namespace, file_path, visible, kept
) -> None:
    inc, store, _, _ = harness
    inc.scope = "default-run|/corpus"
    store.put(
        DocStatusRecord(
            doc_id="doc-x",
            content_hash="h",
            scope="default-x|/corpus",
            entity_ids=["e-vendor"],
            text_unit_ids=["tu-x"],
        )
    )
    if visible:
        # A record of the survivor's namespace that a run since the upgrade
        # wrote with a scope (it lists nothing the removal touches).
        store.put(
            DocStatusRecord(doc_id="doc-y", content_hash="h", scope="default-y|/new")
        )
    store.put(
        DocStatusRecord(
            doc_id=compute_doc_id("a.txt", survivor_namespace),
            content_hash="h",
            file_path=file_path,
            entity_ids=["e-vendor"],
            text_unit_ids=["tu-survivor"],
        )
    )
    (removal,) = inc._plan_removal(["doc-x"]).values()

    assert ("e-vendor" in removal.exclusive_ids) is not kept
    assert removal.shared_entity_ids == (["e-vendor"] if kept else [])


def test_a_scopeless_survivor_under_a_custom_doc_id_retains(harness) -> None:
    inc, store, _, _ = harness
    inc.scope = "default-x|/corpus"
    store.put(
        DocStatusRecord(
            doc_id="doc-x",
            content_hash="h",
            scope="default-x|/corpus",
            entity_ids=["e-vendor"],
            text_unit_ids=["tu-x"],
        )
    )
    # Committed through the direct API under the caller's own doc id: no
    # namespace's legacy key, so its namespace is unknown.
    store.put(
        DocStatusRecord(
            doc_id="contract-42",
            content_hash="h",
            file_path="contracts/42.txt",
            entity_ids=["e-vendor"],
            text_unit_ids=["tu-42"],
        )
    )

    (removal,) = inc._plan_removal(["doc-x"]).values()

    assert removal.shared_entity_ids == ["e-vendor"]
    assert "e-vendor" not in removal.exclusive_ids


def test_a_direct_api_record_retains_for_a_scoped_run_of_its_namespace(
    mocker,
) -> None:
    """Direct API without a scope, then a scoped run in the same stores.

    With ``indexing.additional_suffix`` set, the indexers write under
    ``default-x`` while a scope-less direct-API run keys its records by the
    bare suffix (``compute_doc_id(path, "default")``). A later scoped run in
    ``default-x`` that adds and deletes a document sharing an entity must
    not delete it.
    """
    config = Config()
    config.indexing.additional_suffix = "x"
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    store = FakeDocStatusStore()

    direct = IncrementalIndexer(store, manager)
    a = _doc("a.txt", "Vendor ships.")
    tu_a, _ = _artifacts_for(a, [])
    shared = Entity(id="e-shared", name="Shared", text_unit_ids=[tu_a[0].id])
    _commit(direct, [a], tu_a, [shared])
    (record,) = store.list_all()
    assert record.scope is None
    assert record.doc_id == compute_doc_id("a.txt", "default")

    scoped = IncrementalIndexer(store, manager, scope="default-x|/corpus")
    c = _doc("c.txt", "Vendor bills.")
    c.metadata["registry_namespace"] = "default-x"
    c.metadata["registry_source"] = "/corpus"
    tu_c, _ = _artifacts_for(c, [])
    shared_c = Entity(id="e-shared", name="Shared", text_unit_ids=[tu_c[0].id])
    _commit(scoped, [c], tu_c, [shared_c])
    delta, _ = scoped.plan([])
    assert delta.deleted == [compute_doc_id("c.txt", "default-x", "/corpus")]
    assert scoped.remove_deleted(delta)

    assert "e-shared" in graph.ids("entities")
    assert "e-shared" in vector.ids("entities")


def test_a_removed_scopeless_record_of_no_known_namespace_is_retained_for(
    harness,
) -> None:
    inc, store, _, _ = harness
    # Removed: no scope, legacy key of a namespace no candidate names.
    store.put(
        DocStatusRecord(
            doc_id=compute_doc_id("a.txt", "default-y"),
            content_hash="h",
            file_path="a.txt",
            entity_ids=["e-vendor"],
            text_unit_ids=["tu-a"],
        )
    )
    store.put(
        DocStatusRecord(
            doc_id="doc-y",
            content_hash="h",
            scope="default-z|/corpus",
            entity_ids=["e-vendor"],
            text_unit_ids=["tu-y"],
        )
    )

    (removal,) = inc._plan_removal([compute_doc_id("a.txt", "default-y")]).values()

    assert removal.shared_entity_ids == ["e-vendor"]
    assert "tu-a" in removal.exclusive_ids
