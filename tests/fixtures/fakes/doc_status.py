# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-memory fake implementing ``DocStatusPort`` for fast, AWS-free tests.

Structurally conforms to ``unified_kg_rag.ports.DocStatusPort``. The diff
logic here is the reference behaviour the production DynamoDB adapter (M2) must
match; both are exercised by the same test suite.
"""

from __future__ import annotations

from collections.abc import Iterable

from unified_kg_rag.domain.models import (
    DocStatus,
    DocStatusRecord,
    DocumentDelta,
    DocumentLineage,
)
from unified_kg_rag.shared import DataProcessingError

_LINEAGE_FIELDS = (
    "entity_ids",
    "relationship_ids",
    "text_unit_ids",
    "community_ids",
    "claim_ids",
    "community_report_ids",
)


class FakeDocStatusStore:
    """Dict-backed document-status registry.

    ``max_record_ids`` emulates a backend item limit (DynamoDB's 400 KB): a
    record listing more artifact ids does not fit (:meth:`record_fits`) and
    :meth:`put` rejects it. ``None`` means no limit.
    """

    def __init__(self, max_record_ids: int | None = None) -> None:
        self._records: dict[str, DocStatusRecord] = {}
        self.max_record_ids = max_record_ids
        # Lineage overflow by doc_id: {lineage field: ids}.
        self.overflow: dict[str, dict[str, set[str]]] = {}

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
        if self.max_record_ids is None:
            return True
        size = sum(len(getattr(record, name)) for name in _LINEAGE_FIELDS)
        return size <= self.max_record_ids

    def add_lineage_overflow(self, lineages: Iterable[DocumentLineage]) -> None:
        for lineage in lineages:
            stored = self.overflow.setdefault(lineage.doc_id, {})
            for name in _LINEAGE_FIELDS:
                ids = getattr(lineage, name)
                if ids:
                    stored.setdefault(name, set()).update(ids)

    def get_lineage_overflow(
        self, doc_ids: Iterable[str]
    ) -> dict[str, DocumentLineage]:
        return {
            doc_id: DocumentLineage(
                doc_id=doc_id,
                **{
                    name: sorted(self.overflow[doc_id].get(name, ()))
                    for name in _LINEAGE_FIELDS
                },
            )
            for doc_id in dict.fromkeys(doc_ids)
            if doc_id in self.overflow
        }

    def delete_lineage_overflow(self, doc_ids: Iterable[str]) -> None:
        for doc_id in doc_ids:
            self.overflow.pop(doc_id, None)

    def delete(self, doc_id: str) -> None:
        self._records.pop(doc_id, None)

    def list_all(self) -> list[DocStatusRecord]:
        return list(self._records.values())

    def diff(self, incoming: dict[str, str], scope: str | None = None) -> DocumentDelta:
        delta = DocumentDelta()
        for doc_id, content_hash in incoming.items():
            existing = self._records.get(doc_id)
            if existing is None:
                delta.new.append(doc_id)
            elif (
                existing.content_hash != content_hash
                or existing.status is DocStatus.FAILED
            ):
                delta.changed.append(doc_id)
            else:
                delta.unchanged.append(doc_id)
        incoming_ids = set(incoming)
        delta.deleted = [
            doc_id
            for doc_id, record in self._records.items()
            if doc_id not in incoming_ids and (scope is None or record.scope == scope)
        ]
        delta.stored_scopes = sorted(
            {r.scope for r in self._records.values() if r.scope is not None}
        )
        delta.orphan_overflow = sorted(
            doc_id
            for doc_id in self.overflow
            if doc_id not in self._records
            or self._records[doc_id].status is not DocStatus.PENDING
        )
        return delta
