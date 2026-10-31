# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-memory fake implementing ``DocStatusPort`` for fast, AWS-free tests.

Structurally conforms to ``unified_kg_rag.ports.DocStatusPort``. The diff
logic here is the reference behaviour the production adapters must match;
all are exercised by the same conformance suite. Records are kept by index
namespace (``doc_id_namespace``), like the DynamoDB partitions.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from unified_kg_rag.domain.ingestion.delta_detector import scope_namespace
from unified_kg_rag.domain.ingestion.document_identity import doc_id_namespace
from unified_kg_rag.domain.models import (
    DocStatusRecord,
    DocumentDelta,
    ExtractionSettings,
    StoredDocument,
)
from unified_kg_rag.shared import DataProcessingError


class FakeDocStatusStore:
    """Dict-backed document-status registry.

    ``max_record_ids`` emulates a backend item limit (DynamoDB's 400 KB): a
    record listing more text units does not fit (:meth:`record_fits`) and
    :meth:`put` rejects it. ``None`` means no limit. ``namespace_reads``
    counts :meth:`diff` and :meth:`list_namespace` reads by namespace (``None``
    for a read of every namespace).
    """

    def __init__(self, max_record_ids: int | None = None) -> None:
        self._records: dict[str, DocStatusRecord] = {}
        self.max_record_ids = max_record_ids
        self.namespace_reads: list[str | None] = []
        # namespace -> (owner, expires_at); ``clock`` is replaceable in tests.
        self.leases: dict[str, tuple[str, float]] = {}
        self.extraction_settings: dict[str, tuple[str, ExtractionSettings]] = {}
        self.clock = time.time

    def try_acquire_lease(self, namespace: str, owner: str, seconds: float) -> bool:
        now = self.clock()
        held = self.leases.get(namespace)
        if held is not None and held[0] != owner and held[1] > now:
            return False
        self.leases[namespace] = (owner, now + seconds)
        return True

    def lease_holder(self, namespace: str) -> tuple[str, float] | None:
        return self.leases.get(namespace)

    def release_lease(self, namespace: str, owner: str) -> None:
        if self.leases.get(namespace, ("", 0.0))[0] == owner:
            del self.leases[namespace]

    def get_extraction_settings(
        self, scope: str
    ) -> tuple[str, ExtractionSettings] | None:
        return self.extraction_settings.get(scope)

    def put_extraction_settings(
        self, scope: str, fingerprint: str, settings: ExtractionSettings
    ) -> None:
        self.extraction_settings[scope] = (fingerprint, settings)

    def get(self, doc_id: str) -> DocStatusRecord | None:
        return self._records.get(doc_id)

    def get_many(self, doc_ids: Iterable[str]) -> dict[str, DocStatusRecord]:
        return {
            doc_id: self._records[doc_id]
            for doc_id in doc_ids
            if doc_id in self._records
        }

    def put(self, record: DocStatusRecord) -> None:
        if not self.record_fits(record):
            raise DataProcessingError(
                f"record {record.doc_id} is over the fake's item limit"
            )
        self._records[record.doc_id] = record

    def put_many(self, records: Iterable[DocStatusRecord]) -> None:
        records = list(records)
        # Like DynamoDB: an oversized record fails the call before any write.
        for record in records:
            if not self.record_fits(record):
                raise DataProcessingError(
                    f"record {record.doc_id} is over the fake's item limit"
                )
        for record in records:
            self.put(record)

    def record_fits(self, record: DocStatusRecord) -> bool:
        return (
            self.max_record_ids is None
            or len(record.text_unit_ids) <= self.max_record_ids
        )

    def delete(self, doc_id: str) -> None:
        self._records.pop(doc_id, None)

    def list_namespace(self, namespace: str) -> list[DocStatusRecord]:
        self.namespace_reads.append(namespace)
        return [
            record
            for doc_id, record in self._records.items()
            if doc_id_namespace(doc_id) == namespace
        ]

    def list_all(self) -> list[DocStatusRecord]:
        return list(self._records.values())

    def diff(
        self,
        incoming: dict[str, str],
        scope: str | None = None,
        extraction_fingerprint: str | None = None,
    ) -> DocumentDelta:
        namespace = scope_namespace(scope) if scope is not None else None
        self.namespace_reads.append(namespace)
        stored = {
            doc_id: record
            for doc_id, record in self._records.items()
            if namespace is None or doc_id_namespace(doc_id) == namespace
        }
        return DocumentDelta.classify(
            incoming,
            {
                doc_id: StoredDocument(
                    record.content_hash,
                    record.scope,
                    record.status.value,
                    record.extraction_fingerprint,
                )
                for doc_id, record in stored.items()
            },
            scope,
            extraction_fingerprint,
        )
