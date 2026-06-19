# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Document-status port — the persistence boundary for incremental indexing.

The domain needs to know, across runs, which documents have been seen, their
content hash, processing status, and which graph artifacts (entities,
relationships, text units, communities) they produced — so a re-run can compute
a delta (new / changed / deleted) and merge instead of re-indexing everything.

The production adapter (M2) will be DynamoDB (``aws_graphrag.aws.dynamodb``);
today only the in-memory fake (``tests/fixtures/fakes``) exists. Both conform to
this Protocol structurally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aws_graphrag.models.document import DocStatusRecord, DocumentDelta


@runtime_checkable
class DocStatusPort(Protocol):
    """Persistent registry of per-document processing state and lineage."""

    def get(self, doc_id: str) -> DocStatusRecord | None:
        """Return the stored record for ``doc_id``, or ``None`` if unknown."""
        ...

    def put(self, record: DocStatusRecord) -> None:
        """Insert or overwrite the record for ``record.doc_id``."""
        ...

    def delete(self, doc_id: str) -> None:
        """Remove the record for ``doc_id`` (no-op if absent)."""
        ...

    def list_all(self) -> list[DocStatusRecord]:
        """Return every stored record (used to diff against the new corpus)."""
        ...

    def diff(self, incoming: dict[str, str]) -> DocumentDelta:
        """Classify ``{doc_id: content_hash}`` against stored state.

        Returns the new / changed / unchanged / deleted partition driving an
        incremental run.
        """
        ...
