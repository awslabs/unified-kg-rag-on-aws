# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for cross-run delta detection (M2 incremental indexing)."""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.domain.ingestion.delta_detector import (
    assign_document_identity,
    assign_registry_source,
    compute_content_hash,
    compute_doc_id,
    detect_delta,
    document_doc_id,
    filter_documents_to_process,
    fingerprint_documents,
    legacy_doc_id,
    registry_scope,
    scope_namespace,
)
from unified_kg_rag.domain.models import DocStatus, DocStatusRecord, Document
from unified_kg_rag.ports import DocStatusPort

pytestmark = pytest.mark.unit


def _doc(file_path: str, text: str, doc_id: str = "x") -> Document:
    return Document(
        page_content=text,
        document_id=doc_id,
        file_name=file_path.rsplit("/", 1)[-1],
        file_path=file_path,
        file_type="txt",
        total_pages=1,
    )


class TestComputeDocId:
    def test_is_stable(self) -> None:
        assert compute_doc_id("/a/b/c.txt") == compute_doc_id("/a/b/c.txt")

    def test_normalizes_backslashes(self) -> None:
        assert compute_doc_id("a\\b\\c.txt") == compute_doc_id("a/b/c.txt")

    def test_differs_by_path(self) -> None:
        assert compute_doc_id("/a/x.txt") != compute_doc_id("/a/y.txt")


class TestContentHash:
    def test_changes_with_content(self) -> None:
        h1 = compute_content_hash(_doc("/a.txt", "hello"))
        h2 = compute_content_hash(_doc("/a.txt", "world"))
        assert h1 != h2

    def test_stable_for_same_content(self) -> None:
        h1 = compute_content_hash(_doc("/a.txt", "same"))
        h2 = compute_content_hash(_doc("/a.txt", "same"))
        assert h1 == h2


class TestFingerprintDocuments:
    def test_maps_doc_id_to_content_hash(self) -> None:
        docs = [_doc("/a.txt", "A"), _doc("/b.txt", "B")]
        fps = fingerprint_documents(docs)
        assert set(fps) == {compute_doc_id("/a.txt"), compute_doc_id("/b.txt")}

    def test_same_path_last_wins(self) -> None:
        docs = [_doc("/a.txt", "old"), _doc("/a.txt", "new")]
        fps = fingerprint_documents(docs)
        assert len(fps) == 1
        assert fps[compute_doc_id("/a.txt")] == compute_content_hash(
            _doc("/a.txt", "new")
        )


class TestDetectDelta:
    def test_first_run_all_new(self) -> None:
        store = FakeDocStatusStore()
        docs = [_doc("/a.txt", "A"), _doc("/b.txt", "B")]
        delta, fps = detect_delta(docs, store)
        assert set(delta.new) == set(fps)
        assert not delta.changed and not delta.deleted

    def test_detects_changed_and_deleted(self) -> None:
        store = FakeDocStatusStore()
        # Seed registry: a.txt unchanged, b.txt will change, c.txt will be deleted.
        a_id = compute_doc_id("/a.txt")
        b_id = compute_doc_id("/b.txt")
        c_id = compute_doc_id("/c.txt")
        store.put(
            DocStatusRecord(
                doc_id=a_id, content_hash=compute_content_hash(_doc("/a.txt", "A"))
            )
        )
        store.put(DocStatusRecord(doc_id=b_id, content_hash="stale"))
        store.put(DocStatusRecord(doc_id=c_id, content_hash="whatever"))

        docs = [_doc("/a.txt", "A"), _doc("/b.txt", "B-new")]
        delta, _ = detect_delta(docs, store)

        assert delta.unchanged == [a_id]
        assert delta.changed == [b_id]
        assert delta.deleted == [c_id]
        assert delta.new == []


class _CountingStore(FakeDocStatusStore):
    def __init__(self) -> None:
        super().__init__()
        self.gets = 0
        self.batches: list[int] = []

    def get(self, doc_id: str) -> DocStatusRecord | None:
        self.gets += 1
        return super().get(doc_id)

    def get_many(self, doc_ids):
        ids = list(doc_ids)
        self.batches.append(len(ids))
        return super().get_many(ids)


class TestLegacyLookups:
    SCOPE = "default|/corpus"

    def _docs(self, count: int) -> list[Document]:
        docs = [_doc(f"/corpus/f{i}.txt", f"text {i}") for i in range(count)]
        for document in docs:
            assign_document_identity(document, "/corpus")
            assign_registry_source(document, "/corpus")
        return docs

    def test_absent_legacy_keys_are_not_read(self) -> None:
        store = _CountingStore()
        docs = self._docs(250)
        legacy = {document_doc_id(d): legacy_doc_id(d) for d in docs}

        delta, _ = detect_delta(
            docs, store, scope=self.SCOPE, legacy_doc_ids=legacy, max_failures=3
        )

        assert len(delta.new) == 250
        # The diff already shows that no legacy key is stored: nothing is read.
        assert store.gets == 0
        assert store.batches == []

    def test_a_legacy_record_is_adopted_through_the_batch(self) -> None:
        store = _CountingStore()
        docs = self._docs(3)
        legacy = {document_doc_id(d): legacy_doc_id(d) for d in docs}
        first = docs[0]
        store.put(
            DocStatusRecord(
                doc_id=legacy_doc_id(first),
                content_hash=compute_content_hash(first),
                status="processed",
                scope=self.SCOPE,
                file_path="f0.txt",
                entity_ids=["e1"],
            )
        )

        delta, _ = detect_delta(docs, store, scope=self.SCOPE, legacy_doc_ids=legacy)

        assert delta.unchanged == [document_doc_id(first)]
        assert len(delta.new) == 2 and delta.deleted == []
        assert store.gets == 0
        assert store.batches == [1]
        assert store.get(legacy_doc_id(first)) is None
        assert store.get(document_doc_id(first)).entity_ids == ["e1"]


class TestFilterDocumentsToProcess:
    def test_keeps_only_new_and_changed(self) -> None:
        store = FakeDocStatusStore()
        a_id = compute_doc_id("/a.txt")
        store.put(
            DocStatusRecord(
                doc_id=a_id, content_hash=compute_content_hash(_doc("/a.txt", "A"))
            )
        )
        docs = [_doc("/a.txt", "A"), _doc("/b.txt", "B")]  # a unchanged, b new
        delta, _ = detect_delta(docs, store)

        kept = filter_documents_to_process(docs, delta)
        assert [d.file_path for d in kept] == ["/b.txt"]


class _PortSubclassStore(DocStatusPort):
    """Explicitly subclasses the port, so it inherits the default get_many."""

    def __init__(self) -> None:
        self.inner = FakeDocStatusStore()
        self.gets: list[str] = []

    def get(self, doc_id: str) -> DocStatusRecord | None:
        self.gets.append(doc_id)
        return self.inner.get(doc_id)

    def put(self, record: DocStatusRecord) -> None:
        self.inner.put(record)

    def delete(self, doc_id: str) -> None:
        self.inner.delete(doc_id)

    def list_all(self) -> list[DocStatusRecord]:
        return self.inner.list_all()

    def diff(self, incoming: dict[str, str], scope: str | None = None):
        return self.inner.diff(incoming, scope=scope)


def test_port_default_get_many_loops_get_and_skips_unknown_ids() -> None:
    store = _PortSubclassStore()
    store.put(DocStatusRecord(doc_id="a", content_hash="h"))

    assert list(store.get_many(["a", "b", "a"])) == ["a"]
    assert store.gets == ["a", "b"]


def test_scope_namespace_reads_the_namespace_of_a_registry_scope() -> None:
    assert scope_namespace(registry_scope("default-x", "/corpus|a")) == "default-x"
    assert scope_namespace(registry_scope("tenant", "s3://bucket/p/")) == "tenant"
    assert scope_namespace(None) is None
    assert scope_namespace("no-separator") is None


def _failed_corpus(store: FakeDocStatusStore, count: int) -> list[Document]:
    """``count`` documents, every one recorded FAILED 3 times with its content
    except the first, which was edited."""
    documents = [_doc(f"/d{i}.txt", f"text {i}") for i in range(count)]
    for i, document in enumerate(documents):
        store.put(
            DocStatusRecord(
                doc_id=document_doc_id(document),
                content_hash="edited" if i == 0 else compute_content_hash(document),
                status=DocStatus.FAILED,
                failure_count=3,
            )
        )
    return documents


def test_exhausted_retries_are_read_in_one_batch() -> None:
    store = _CountingStore()
    documents = _failed_corpus(store, 300)

    delta, _ = detect_delta(documents, store, max_failures=3)

    assert delta.changed == [document_doc_id(documents[0])]
    assert len(delta.unchanged) == 299
    assert store.gets == 0
    assert store.batches == [300]


def test_exhausted_retries_fall_back_to_get_without_get_many() -> None:
    class _Legacy(FakeDocStatusStore):
        get_many = None  # type: ignore[assignment]

    store = _Legacy()
    documents = _failed_corpus(store, 3)

    delta, _ = detect_delta(documents, store, max_failures=3)

    assert delta.changed == [document_doc_id(documents[0])]
    assert len(delta.unchanged) == 2
