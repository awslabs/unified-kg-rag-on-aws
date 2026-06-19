# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""In-memory fake implementing ``DocStatusPort`` for fast, AWS-free tests.

Structurally conforms to ``aws_graphrag.core.ports.DocStatusPort``. The diff
logic here is the reference behaviour the production DynamoDB adapter (M2) must
match; both are exercised by the same test suite.
"""

from __future__ import annotations

from aws_graphrag.models import DocStatusRecord, DocumentDelta


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
