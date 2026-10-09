# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partial write failures under the tolerated rate are retried, not lost.

An incremental commit records every document PROCESSED when each artifact
type's failure rate stays within ``indexing.max_failure_rate``. A document
whose entity or relationship write was dropped was then classified unchanged
on every later run, so the missing artifact was never rewritten. The
indexers now report the ids of the items that failed, and the commit records
the documents owning them FAILED. AWS-free: in-memory registry and stub
backends.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
from unified_kg_rag.domain.ingestion.delta_detector import detect_delta
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    DocStatusRecord,
    Document,
    DocumentLineage,
)
from unified_kg_rag.ports.indexer import IndexingStats

pytestmark = pytest.mark.unit


class _Manager:
    def __init__(self) -> None:
        self.config = Config()


def _lineages() -> list[DocumentLineage]:
    return [
        DocumentLineage(
            doc_id="doc-a",
            text_unit_ids=["t-a"],
            entity_ids=["e-vendor", "e-shared"],
            relationship_ids=["r-a"],
        ),
        DocumentLineage(
            doc_id="doc-b",
            text_unit_ids=["t-b"],
            entity_ids=["e-buyer", "e-shared"],
        ),
        DocumentLineage(doc_id="doc-c", text_unit_ids=["t-c"], entity_ids=["e-x"]),
    ]


_FINGERPRINTS = {"doc-a": "hash-a", "doc-b": "hash-b", "doc-c": "hash-c"}


def _stats(total: int, failed_ids: list[str], unattributed: int = 0) -> IndexingStats:
    stats = IndexingStats(total_items=total)
    stats.add_success(total - len(failed_ids) - unattributed)
    if failed_ids:
        stats.add_error("write rejected", len(failed_ids), ids=failed_ids)
    if unattributed:
        stats.add_error("write failed", unattributed)
    return stats


def _record(store: FakeDocStatusStore, results: dict[str, IndexingStats]) -> bool:
    indexer = IncrementalIndexer(store, _Manager())  # type: ignore[arg-type]
    return indexer.record(_lineages(), _FINGERPRINTS, results)


def _statuses(store: FakeDocStatusStore) -> dict[str, DocStatus]:
    return {record.doc_id: record.status for record in store.list_all()}


def test_documents_owning_a_failed_write_are_recorded_failed() -> None:
    store = FakeDocStatusStore()
    recorded = _record(
        store,
        {
            "opensearch_entities": _stats(10, []),
            "neptune_relationships": _stats(10, ["r-a"]),
        },
    )

    assert recorded is True
    assert _statuses(store) == {
        "doc-a": DocStatus.FAILED,
        "doc-b": DocStatus.PROCESSED,
        "doc-c": DocStatus.PROCESSED,
    }
    failed = store.get("doc-a")
    assert failed is not None
    assert failed.failure_count == 1
    assert "artifacts" in (failed.error_info or "")
    # The lineage is kept so the retry first prunes what this run wrote.
    assert failed.relationship_ids == ["r-a"]


def test_a_failed_shared_artifact_fails_every_owner() -> None:
    store = FakeDocStatusStore()
    _record(store, {"opensearch_entities": _stats(10, ["e-shared"])})

    assert _statuses(store) == {
        "doc-a": DocStatus.FAILED,
        "doc-b": DocStatus.FAILED,
        "doc-c": DocStatus.PROCESSED,
    }


def test_a_failure_without_an_id_fails_the_whole_commit() -> None:
    store = FakeDocStatusStore()
    _record(store, {"neptune_entities": _stats(10, [], unattributed=1)})

    assert set(_statuses(store).values()) == {DocStatus.FAILED}


def test_clean_writes_record_every_document_processed() -> None:
    store = FakeDocStatusStore()
    _record(store, {"opensearch_entities": _stats(10, [])})

    assert set(_statuses(store).values()) == {DocStatus.PROCESSED}


def test_failure_rate_gate_still_records_nothing() -> None:
    store = FakeDocStatusStore()
    recorded = _record(store, {"neptune_relationships": _stats(2, ["r-a"])})

    assert recorded is False
    assert store.list_all() == []


def _document(path: str) -> Document:
    return Document(
        page_content=f"Synthetic text of {path}.",
        document_id=f"run-{path}",
        file_name=path,
        file_path=path,
        file_type="txt",
        total_pages=1,
    )


def test_repeated_write_failures_honour_max_document_failures() -> None:
    # The write-failure count accumulates like an extraction failure, so a
    # document whose artifact is rejected every run stops being retried.
    store = FakeDocStatusStore()
    document = _document("a.txt")
    _, fingerprints = detect_delta([document], store)
    (doc_id,) = fingerprints
    lineage = DocumentLineage(doc_id=doc_id, entity_ids=["e-1"])
    indexer = IncrementalIndexer(store, _Manager())  # type: ignore[arg-type]
    results = {"opensearch_entities": _stats(10, ["e-1"])}

    for expected_count in (1, 2, 3):
        indexer.record([lineage], fingerprints, results)
        record = store.get(doc_id)
        assert record is not None
        assert record.status is DocStatus.FAILED
        assert record.failure_count == expected_count

    delta, _ = detect_delta([document], store, max_failures=3)
    assert delta.unchanged == [doc_id]
    assert delta.changed == []

    store.put(
        DocStatusRecord(
            doc_id=doc_id,
            content_hash=fingerprints[doc_id],
            status=DocStatus.FAILED,
            failure_count=2,
        )
    )
    delta, _ = detect_delta([document], store, max_failures=3)
    assert delta.changed == [doc_id]


# --- the indexers report the ids of failed writes -----------------------------


class _BulkClient:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response

    def bulk_index(self, index_name, documents, **kwargs):  # noqa: ANN001, ARG002
        if self.response is None:
            raise RuntimeError("bulk failed")
        return self.response


def _opensearch_indexer(client: _BulkClient) -> OpenSearchIndexer:
    inst = OpenSearchIndexer.__new__(OpenSearchIndexer)
    inst.config = Config()
    inst.opensearch_config = inst.config.indexing.opensearch
    inst.opensearch_client = client  # type: ignore[assignment]
    return inst


def test_opensearch_bulk_item_failures_carry_their_ids() -> None:
    response = {
        "errors": True,
        "items": [{"index": {"_id": "e-2", "status": 429, "error": "rejected"}}],
    }
    indexer = _opensearch_indexer(_BulkClient(response))

    stats = indexer._perform_indexing("idx", [{"id": "e-1"}, {"id": "e-2"}])

    assert (stats.successful_items, stats.failed_items) == (1, 1)
    assert stats.failed_ids == ["e-2"]


def test_opensearch_failed_bulk_request_fails_every_id() -> None:
    indexer = _opensearch_indexer(_BulkClient(None))

    stats = indexer._perform_indexing("idx", [{"id": "e-1"}, {"id": "e-2"}])

    assert stats.failed_ids == ["e-1", "e-2"]


def test_failed_ids_survive_merging_stats() -> None:
    total = IndexingStats()
    total.merge(_stats(3, ["e-1"]))
    total.merge(_stats(3, ["e-2"]))
    assert total.failed_ids == ["e-1", "e-2"]
    assert total.failed_items == 2
