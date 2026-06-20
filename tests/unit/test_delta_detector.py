# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for cross-run delta detection (M2 incremental indexing)."""

from __future__ import annotations

import pytest

from aws_graphrag.domain.ingestion.delta_detector import (
    compute_content_hash,
    compute_doc_id,
    detect_delta,
    filter_documents_to_process,
    fingerprint_documents,
)
from aws_graphrag.domain.models import DocStatusRecord, Document
from tests.fixtures.fakes.doc_status import FakeDocStatusStore

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
