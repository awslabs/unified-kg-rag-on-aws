# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end incremental-indexing tests using in-memory fakes (M2).

Exercises the full add -> change -> delete cycle through the
:class:`IncrementalIndexer` against the fake doc-status store and a recording
fake indexing manager, with no AWS clients involved. Lineage is attributed
per-document so deletion removes only a document's *exclusive* artifacts.
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.domain.ingestion.delta_detector import compute_doc_id
from unified_kg_rag.domain.models import Document, DocumentLineage, Entity
from unified_kg_rag.ingestion import IncrementalIndexer
from unified_kg_rag.ports.indexer import IndexingStats

pytestmark = pytest.mark.integration


class FakeIndexingManager:
    """Records delta/delete calls instead of touching real stores."""

    def __init__(self) -> None:
        self.delta_calls: list[dict] = []
        self.delete_calls: list[dict] = []

    def index_delta(self, **kwargs) -> dict[str, IndexingStats]:
        self.delta_calls.append(kwargs)
        return {}

    def delete_documents(self, ids_by_suffix) -> dict[str, IndexingStats]:
        self.delete_calls.append(ids_by_suffix)
        return {}


def _doc(path: str, text: str) -> Document:
    return Document(
        page_content=text,
        document_id="x",
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
        file_type="txt",
        total_pages=1,
    )


def _lineage(path: str, entity_ids: list[str]) -> DocumentLineage:
    return DocumentLineage(doc_id=compute_doc_id(path), entity_ids=entity_ids)


@pytest.fixture
def indexer() -> tuple[IncrementalIndexer, FakeDocStatusStore, FakeIndexingManager]:
    store = FakeDocStatusStore()
    manager = FakeIndexingManager()
    return IncrementalIndexer(store, manager), store, manager  # type: ignore[arg-type]


def test_first_run_marks_all_new_and_records_registry(indexer) -> None:
    inc, store, manager = indexer
    docs = [_doc("/a.txt", "A"), _doc("/b.txt", "B")]

    delta, fingerprints = inc.plan(docs)
    assert set(delta.new) == set(fingerprints)
    assert not delta.deleted

    to_process = inc.documents_to_process(docs, delta)
    assert len(to_process) == 2

    inc.commit(
        lineages=[_lineage("/a.txt", ["ea"]), _lineage("/b.txt", ["eb"])],
        fingerprints=fingerprints,
        entities=[Entity(id="ea", name="A"), Entity(id="eb", name="B")],
    )
    assert len(manager.delta_calls) == 1
    assert len(store.list_all()) == 2


def test_second_run_unchanged_is_noop_delta(indexer) -> None:
    inc, store, manager = indexer
    docs = [_doc("/a.txt", "A")]
    _, fps = inc.plan(docs)
    inc.commit([_lineage("/a.txt", ["e1"])], fps, entities=[Entity(id="e1", name="A")])

    delta2, _ = inc.plan(docs)
    assert delta2.is_empty
    assert inc.documents_to_process(docs, delta2) == []


def test_changed_document_is_reprocessed(indexer) -> None:
    inc, store, manager = indexer
    docs = [_doc("/a.txt", "original")]
    _, fps = inc.plan(docs)
    inc.commit([_lineage("/a.txt", ["e1"])], fps, entities=[Entity(id="e1", name="A")])

    edited = [_doc("/a.txt", "EDITED")]
    delta, _ = inc.plan(edited)
    assert delta.changed == [compute_doc_id("/a.txt")]
    assert len(inc.documents_to_process(edited, delta)) == 1


def test_changed_document_prunes_stale_artifacts(indexer) -> None:
    inc, store, manager = indexer
    # First run: doc a produced entities e1, e2.
    _, fps = inc.plan([_doc("/a.txt", "v1")])
    inc.commit(
        [_lineage("/a.txt", ["e1", "e2"])],
        fps,
        entities=[Entity(id="e1", name="A"), Entity(id="e2", name="B")],
    )

    # Edit: re-extraction will produce only e1; e2 must be pruned.
    edited = [_doc("/a.txt", "v2")]
    delta, _ = inc.plan(edited)
    inc.prune_changed(delta)

    assert manager.delete_calls
    pruned = manager.delete_calls[0]["default"]
    assert "e1" in pruned and "e2" in pruned  # all old artifacts of the changed doc


def test_changed_document_does_not_prune_entity_shared_with_survivor(indexer) -> None:
    # A changed doc and a surviving doc both reference 'shared'; pruning the
    # changed doc's stale artifacts must NOT drop the shared entity (still
    # referenced by the survivor), or the survivor's graph loses a reference.
    inc, store, manager = indexer
    a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
    _, fps = inc.plan([a, b])
    inc.commit(
        [_lineage("/a.txt", ["shared", "only_a"])],
        fps,
        entities=[Entity(id="shared", name="S"), Entity(id="only_a", name="A")],
    )
    inc.commit(
        [_lineage("/b.txt", ["shared"])], fps, entities=[Entity(id="shared", name="S")]
    )

    # Edit /a.txt -> prune its stale artifacts before re-extraction.
    edited = [_doc("/a.txt", "EDITED"), b]
    delta, _ = inc.plan(edited)
    inc.prune_changed(delta)

    pruned = [i for call in manager.delete_calls for ids in call.values() for i in ids]
    assert "only_a" in pruned  # exclusive to the changed doc -> pruned
    assert "shared" not in pruned  # still referenced by surviving /b.txt -> kept


def test_deleted_document_removes_exclusive_artifacts(indexer) -> None:
    inc, store, manager = indexer
    a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
    _, fps = inc.plan([a, b])
    inc.commit([_lineage("/a.txt", ["ea"])], fps, entities=[Entity(id="ea", name="A")])
    inc.commit([_lineage("/b.txt", ["eb"])], fps, entities=[Entity(id="eb", name="B")])

    delta, _ = inc.plan([a])
    assert delta.deleted == [compute_doc_id("/b.txt")]

    inc.remove_deleted(delta)
    assert {r.doc_id for r in store.list_all()} == {compute_doc_id("/a.txt")}
    assert manager.delete_calls
    deleted_ids = manager.delete_calls[0]["default"]
    assert "eb" in deleted_ids
    assert "ea" not in deleted_ids


def test_deleted_document_removes_claim_artifacts(indexer) -> None:
    # Claims are first-class indexed artifacts; their lineage must be tracked
    # and removed on deletion like entities, or they leak as orphans.
    inc, store, manager = indexer
    a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
    _, fps = inc.plan([a, b])
    inc.commit(
        [DocumentLineage(doc_id=compute_doc_id("/a.txt"), claim_ids=["ca"])],
        fps,
        entities=[Entity(id="ea", name="A")],
    )
    inc.commit(
        [DocumentLineage(doc_id=compute_doc_id("/b.txt"), claim_ids=["cb"])],
        fps,
        entities=[Entity(id="eb", name="B")],
    )

    # The registry records claim ids per document.
    rec_b = store.get(compute_doc_id("/b.txt"))
    assert rec_b is not None and rec_b.claim_ids == ["cb"]

    delta, _ = inc.plan([a])  # delete b
    inc.remove_deleted(delta)
    deleted_ids = manager.delete_calls[0]["default"]
    assert "cb" in deleted_ids
    assert "ca" not in deleted_ids


def test_shared_artifacts_are_not_deleted(indexer) -> None:
    inc, store, manager = indexer
    a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
    _, fps = inc.plan([a, b])
    # Both docs reference the SAME shared entity id.
    inc.commit(
        [_lineage("/a.txt", ["shared"])], fps, entities=[Entity(id="shared", name="S")]
    )
    inc.commit(
        [_lineage("/b.txt", ["shared"])], fps, entities=[Entity(id="shared", name="S")]
    )

    delta, _ = inc.plan([a])  # delete b
    inc.remove_deleted(delta)
    # Shared entity is still referenced by a -> must never be deleted. The
    # assertion must hold whether or not a delete batch was emitted: collect all
    # ids ever passed to delete_documents and require "shared" is absent.
    all_deleted_ids = [
        artifact_id
        for call in manager.delete_calls
        for ids in call.values()
        for artifact_id in ids
    ]
    assert "shared" not in all_deleted_ids
    # b's registry record is still removed even though its only artifact survives.
    assert {r.doc_id for r in store.list_all()} == {compute_doc_id("/a.txt")}
