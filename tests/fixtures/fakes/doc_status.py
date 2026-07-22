# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-memory fake implementing ``DocStatusPort`` for fast, AWS-free tests.

Structurally conforms to ``unified_kg_rag.ports.DocStatusPort``. The diff
logic here is the reference behaviour the production DynamoDB adapter (M2) must
match; both are exercised by the same test suite.
"""

from __future__ import annotations

from unified_kg_rag.domain.models import DocStatusRecord, DocumentDelta


class FakeDocStatusStore:
    """Dict-backed document-status registry."""

    def __init__(self) -> None:
        self._records: dict[str, DocStatusRecord] = {}

    def get(self, doc_id: str) -> DocStatusRecord | None:
        return self._records.get(doc_id)

    def put(self, record: DocStatusRecord) -> None:
        self._records[record.doc_id] = record

    def delete(self, doc_id: str) -> None:
        self._records.pop(doc_id, None)

    def list_all(self) -> list[DocStatusRecord]:
        return list(self._records.values())

    def diff(self, incoming: dict[str, str]) -> DocumentDelta:
        delta = DocumentDelta()
        for doc_id, content_hash in incoming.items():
            existing = self._records.get(doc_id)
            if existing is None:
                delta.new.append(doc_id)
            elif existing.content_hash != content_hash:
                delta.changed.append(doc_id)
            else:
                delta.unchanged.append(doc_id)
        incoming_ids = set(incoming)
        delta.deleted = [
            doc_id for doc_id in self._records if doc_id not in incoming_ids
        ]
        return delta
