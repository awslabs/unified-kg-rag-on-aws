# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Document-status port — the persistence boundary for incremental indexing.

The domain needs to know, across runs, which documents have been seen, their
content hash, processing status, and their text units — so a re-run can
compute a delta (new / changed / deleted), find what a changed or deleted
document contributed (every artifact cites the text units it was stated in)
and merge instead of re-indexing everything. Records are kept per index
namespace, the prefix of their ``doc_id`` (``doc_id_namespace``): a run reads
only its own namespace's records. A run also holds its namespace's write
lease (:meth:`DocStatusPort.try_acquire_lease`), so two runs never write one
namespace at once.

Adapters: DynamoDB (``unified_kg_rag.adapters.doc_status.dynamodb``) and SQLite for
local mode (``unified_kg_rag.adapters.doc_status.sqlite``); tests use the
in-memory fake (``tests/fixtures/fakes``). All conform to this Protocol
structurally and pass one conformance suite
(``tests/integration/test_doc_status_conformance.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from unified_kg_rag.domain.models.document import (
        DocStatusRecord,
        DocumentDelta,
    )


@runtime_checkable
class DocStatusPort(Protocol):
    """Persistent registry of per-document processing state, by namespace."""

    def get(self, doc_id: str) -> DocStatusRecord | None:
        """Return the stored record for ``doc_id``, or ``None`` if unknown."""
        ...

    def get_many(self, doc_ids: Iterable[str]) -> dict[str, DocStatusRecord]:
        """Return ``{doc_id: record}`` for the ``doc_ids`` that are stored.

        Unknown ids are left out. The default looks each id up with
        :meth:`get`; an adapter whose backend reads many keys per request
        (DynamoDB ``BatchGetItem``) overrides it, so looking up a whole corpus
        is not one round trip per document.
        """
        records: dict[str, DocStatusRecord] = {}
        for doc_id in dict.fromkeys(doc_ids):
            record = self.get(doc_id)
            if record is not None:
                records[doc_id] = record
        return records

    def put(self, record: DocStatusRecord) -> None:
        """Insert or overwrite the record for ``record.doc_id``."""
        ...

    def put_many(self, records: Iterable[DocStatusRecord]) -> None:
        """Insert or overwrite every record; the last one per ``doc_id`` wins.

        Not atomic: an interruption can leave some records written. The
        default writes each record with :meth:`put`; an adapter whose backend
        writes many items per request (DynamoDB ``BatchWriteItem``) overrides
        it, so an incremental run's write-ahead and commit records are not
        one round trip per document.
        """
        for record in {record.doc_id: record for record in records}.values():
            self.put(record)

    def record_fits(self, record: DocStatusRecord) -> bool:
        """Whether :meth:`put` can store ``record`` (the backend's item limit).

        An incremental run checks its write-ahead records (a document's
        stored and planned text units) before writing any. The default (no
        limit) accepts every record.
        """
        return True

    def delete(self, doc_id: str) -> None:
        """Remove the record for ``doc_id`` (no-op if absent)."""
        ...

    def try_acquire_lease(self, namespace: str, owner: str, seconds: float) -> bool:
        """Take or renew the write lease of index ``namespace`` for ``owner``.

        Succeeds when no lease is held, the held one expired, or ``owner``
        holds it; the lease then expires ``seconds`` from now. Atomic (a
        conditional write), so two runs never both hold it. Returns whether
        ``owner`` holds the lease.
        """
        ...

    def lease_holder(self, namespace: str) -> tuple[str, float] | None:
        """``(owner, expires_at)`` of ``namespace``'s lease (epoch seconds;
        possibly expired), or ``None`` when none was taken."""
        ...

    def release_lease(self, namespace: str, owner: str) -> None:
        """Release ``namespace``'s lease if ``owner`` holds it (no-op else)."""
        ...

    def list_namespace(self, namespace: str) -> list[DocStatusRecord]:
        """Return every record of index ``namespace`` (its registry partition)."""
        ...

    def list_all(self) -> list[DocStatusRecord]:
        """Return every stored record, of every namespace (administration and
        tests; a run reads its namespace with :meth:`list_namespace` or
        :meth:`diff`)."""
        ...

    def diff(
        self,
        incoming: dict[str, str],
        scope: str | None = None,
        extraction_fingerprint: str | None = None,
    ) -> DocumentDelta:
        """Classify ``{doc_id: content_hash}`` against stored state.

        Returns the new / changed / unchanged / deleted partition driving an
        incremental run. With ``scope``, only the records of the scope's
        namespace (``scope_namespace``) are read, and only those of that scope
        are candidates for ``deleted`` (a record without a scope never is), so
        one tenant's or one corpus's run cannot delete another's documents.
        Callers key documents per scope (the source scope is part of the
        ``doc_id``), so an incoming id never matches another scope's record.
        ``scope=None`` reads every namespace. A stored ``FAILED``
        record is ``changed`` even when its hash matches, so it is retried.

        With ``extraction_fingerprint`` (the run's extraction settings, see
        ``DocStatusRecord.extraction_fingerprint``), a stored record whose
        fingerprint is set and differs is ``changed`` too, and also listed in
        ``DocumentDelta.settings_changed`` when its hash and status alone
        would read unchanged. A record without a fingerprint is compared by
        hash and status only.

        An adapter should also report the distinct scopes of the records it
        read in ``DocumentDelta.stored_scopes`` (the scan already reads them);
        the pipeline uses them only to warn about other corpora sharing the
        run's namespace, and leaving it empty just silences that warning.
        """
        ...
