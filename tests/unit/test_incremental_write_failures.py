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
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    artifact_owners,
)
from unified_kg_rag.domain.ingestion.delta_detector import detect_delta
from unified_kg_rag.domain.models import (
    Community,
    Config,
    DocStatus,
    DocStatusRecord,
    Document,
    DocumentLineage,
    Entity,
    Relationship,
)
from unified_kg_rag.ports.indexer import IndexingStats

pytestmark = pytest.mark.unit


class _Manager:
    def __init__(self) -> None:
        self.config = Config()


def _lineages() -> list[DocumentLineage]:
    return [
        DocumentLineage(doc_id="doc-a", text_unit_ids=["t-a"]),
        DocumentLineage(doc_id="doc-b", text_unit_ids=["t-b"]),
        DocumentLineage(doc_id="doc-c", text_unit_ids=["t-c"]),
    ]


# The commit's artifacts, owned through the text units they cite.
_ARTIFACTS = [
    Entity(id="e-vendor", name="Vendor", text_unit_ids=["t-a"]),
    Entity(id="e-shared", name="Shared", text_unit_ids=["t-a", "t-b"]),
    Entity(id="e-buyer", name="Buyer", text_unit_ids=["t-b"]),
    Entity(id="e-x", name="X", text_unit_ids=["t-c"]),
    Relationship(
        id="r-a", source_id="e-vendor", target_id="e-shared", text_unit_ids=["t-a"]
    ),
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
    lineages = _lineages()
    return indexer.record(
        lineages, _FINGERPRINTS, results, owners=artifact_owners(lineages, _ARTIFACTS)
    )


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
    # A store's failure is not the document's: it is retried, not counted.
    assert failed.failure_count == 0
    assert "artifacts" in (failed.error_info or "")
    # The text units are kept so the retry first prunes what this run wrote.
    assert failed.text_unit_ids == ["t-a"]


def test_the_commit_lists_the_documents_it_recorded_failed() -> None:
    # failed_documents (the JSON result) and the summary count these: every
    # document the commit records FAILED (a failed write's owner, an
    # extraction failure), only those, the counted failures included.
    store = FakeDocStatusStore()
    indexer = IncrementalIndexer(store, _Manager())  # type: ignore[arg-type]
    lineages = _lineages()
    indexer.record(
        lineages,
        _FINGERPRINTS,
        {"neptune_relationships": _stats(10, ["r-a"])},
        ["doc-c"],
        artifact_owners(lineages, _ARTIFACTS),
        counted_failed_doc_ids=["doc-c"],
    )

    assert sorted(indexer.recorded_failed_doc_ids) == ["doc-a", "doc-c"]
    assert sorted((r.doc_id, r.failure_count) for r in store.list_all()) == [
        ("doc-a", 0),
        ("doc-b", 0),
        ("doc-c", 1),
    ]


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
    )


def test_repeated_write_failures_are_retried_without_counting() -> None:
    # A failed write is the store's, not the document's (an outage or a
    # throttled bulk item after the retries): counted, three runs of an
    # outage excluded the document. It is retried every run instead.
    store = FakeDocStatusStore()
    document = _document("a.txt")
    _, fingerprints = detect_delta([document], store)
    (doc_id,) = fingerprints
    lineage = DocumentLineage(doc_id=doc_id, text_unit_ids=["t-1"])
    indexer = IncrementalIndexer(store, _Manager())  # type: ignore[arg-type]
    results = {"opensearch_entities": _stats(10, ["e-1"])}
    owners = artifact_owners(
        [lineage], [Entity(id="e-1", name="E", text_unit_ids=["t-1"])]
    )

    for _ in range(3):
        indexer.record([lineage], fingerprints, results, owners=owners)
        record = store.get(doc_id)
        assert record is not None
        assert record.status is DocStatus.FAILED
        assert record.failure_count == 0

    delta, _ = detect_delta([document], store, max_failures=3)
    assert delta.changed == [doc_id]

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


def test_opensearch_bulk_item_failures_name_the_first_reason() -> None:
    # Only the count was reported: the reason the store rejected the items
    # (a mapping conflict, a vector of the wrong dimension) was dropped. It
    # is named, without the rejected value the reason previews.
    reason = (
        "failed to parse field [embedding] of type [knn_vector] in document "
        "with id 'e-2'. Preview of field's value: '[0.1, 0.2]'"
    )
    error = {"type": "mapper_parsing_exception", "reason": reason}
    response = {
        "errors": True,
        "items": [{"index": {"_id": "e-2", "status": 400, "error": error}}],
    }
    indexer = _opensearch_indexer(_BulkClient(response))

    (message,) = indexer._perform_indexing("idx", [{"id": "e-2"}]).errors

    assert message == (
        "Bulk API errors: 1 (first: mapper_parsing_exception: failed to parse "
        "field [embedding] of type [knn_vector] in document with id 'e-2'.)"
    )
    error["reason"] = "x" * 1000
    (message,) = indexer._perform_indexing("idx", [{"id": "e-2"}]).errors
    assert len(message) < 350


def test_opensearch_bulk_item_failures_name_the_cause_without_corpus_text() -> None:
    # The cause was dropped: a vector of the wrong dimension read as a parse
    # failure only, while its caused_by says "Vector dimension mismatch".
    # An immense-term rejection quoted the term's bytes, and a cause the
    # value it could not read: neither reaches the message.
    def message(error: dict) -> str:
        response = {
            "errors": True,
            "items": [{"index": {"_id": "e-2", "status": 400, "error": error}}],
        }
        indexer = _opensearch_indexer(_BulkClient(response))
        (text,) = indexer._perform_indexing("idx", [{"id": "e-2"}]).errors
        return text

    dimension = message(
        {
            "type": "mapper_parsing_exception",
            "reason": "failed to parse field [embedding] of type [knn_vector]",
            "caused_by": {
                "type": "illegal_argument_exception",
                "reason": "Vector dimension mismatch. Expected: 1024, Given: 8",
            },
        }
    )
    assert dimension.endswith(
        "(first: mapper_parsing_exception: failed to parse field [embedding] of "
        "type [knn_vector] (caused by illegal_argument_exception: Vector "
        "dimension mismatch. Expected: 1024, Given: 8))"
    ), dimension
    immense = message(
        {
            "type": "illegal_argument_exception",
            "reason": 'Document contains at least one immense term in field="text" '
            "(whose UTF8 encoding is longer than the max length 32766), all of "
            "which were skipped.  Please correct the analyzer to not produce "
            "such terms.  The prefix of the first immense term is: '[83, 101, "
            "99, 114, 101, 116]...', original message: bytes can be at most "
            "32766 in length; got 40000",
            "caused_by": {
                "type": "max_bytes_length_exceeded_exception",
                "reason": "bytes can be at most 32766 in length; got 40000",
            },
        }
    )
    assert "[83, 101" not in immense and "such terms." in immense
    assert "max_bytes_length_exceeded_exception" in immense
    value = message(
        {
            "type": "mapper_parsing_exception",
            "reason": "failed to parse field [rank] of type [long]",
            "caused_by": {
                "type": "number_format_exception",
                "reason": 'For input string: "Secret Corp"',
            },
        }
    )
    assert "Secret" not in value and 'For input string: "..."' in value


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


# --- failed ids are counted with multiplicity ---------------------------------


def _community(comm_id: str, entity_ids: list[str]) -> Community:
    return Community(
        id=comm_id,
        name=comm_id,
        level="0",
        parent="",
        children=[],
        entity_ids=entity_ids,
    )


def test_repeated_ids_alone_are_fully_attributed() -> None:
    stats = IndexingStats(total_items=40)
    stats.add_success(37)
    for _ in range(3):
        stats.add_error("edge batch failed", ids=["e-shared"])
    store = FakeDocStatusStore()

    _record(store, {"neptune_communities": stats})

    assert _statuses(store) == {
        "doc-a": DocStatus.FAILED,
        "doc-b": DocStatus.FAILED,
        "doc-c": DocStatus.PROCESSED,
    }


def test_mixed_attributed_and_unattributed_failures_fail_the_whole_commit() -> None:
    stats = IndexingStats(total_items=40)
    stats.add_success(37)
    stats.add_error("edge batch failed", ids=["e-shared"])
    stats.add_error("edge batch failed", ids=["e-shared"])
    stats.add_error("batch failed")
    assert stats.unattributed_failures == 1
    store = FakeDocStatusStore()

    _record(store, {"neptune_communities": stats})

    assert set(_statuses(store).values()) == {DocStatus.FAILED}


def test_partially_attributed_add_error_counts_the_rest_unattributed() -> None:
    stats = IndexingStats()
    stats.add_error("bulk failed", 3, ids=["e-1"])
    assert stats.unattributed_failures == 2


def test_directly_built_failure_counts_stay_unattributed() -> None:
    stats = IndexingStats(total_items=4, successful_items=3, failed_items=1)
    assert stats.unattributed_failures == 1


def test_add_error_rejects_more_ids_than_failed_items() -> None:
    with pytest.raises(ValueError, match="one id per failed item"):
        IndexingStats().add_error("bad", 1, ids=["e-1", "e-2"])
