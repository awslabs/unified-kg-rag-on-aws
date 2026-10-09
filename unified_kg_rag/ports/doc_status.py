# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Document-status port — the persistence boundary for incremental indexing.

The domain needs to know, across runs, which documents have been seen, their
content hash, processing status, and which graph artifacts (entities,
relationships, text units, communities) they produced — so a re-run can compute
a delta (new / changed / deleted) and merge instead of re-indexing everything.

The production adapter (M2) will be DynamoDB (``unified_kg_rag.adapters.aws.dynamodb``);
today only the in-memory fake (``tests/fixtures/fakes``) exists. Both conform to
this Protocol structurally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from unified_kg_rag.domain.models.document import (
        DocStatusRecord,
        DocumentDelta,
        DocumentLineage,
    )


@runtime_checkable
class DocStatusPort(Protocol):
    """Persistent registry of per-document processing state and lineage."""

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

        An incremental run's write-ahead record lists a document's stored and
        planned artifact ids, which can outgrow a limit the stored and the
        committed record each fit; the run then moves the planned ids into
        lineage overflow (:meth:`add_lineage_overflow`). The default (no
        limit) never needs overflow.
        """
        return True

    def add_lineage_overflow(self, lineages: Iterable[DocumentLineage]) -> None:
        """Add each lineage's artifact ids to its ``doc_id``'s overflow.

        Lineage overflow holds the artifact ids of a write-ahead record that
        do not fit in the record itself (see :meth:`record_fits`). It is never
        part of :meth:`get`, :meth:`get_many`, :meth:`list_all` or
        :meth:`diff`; the incremental run reads it with
        :meth:`get_lineage_overflow` and deletes it with
        :meth:`delete_lineage_overflow`. Only called when :meth:`record_fits`
        rejected a record, so a store without an item limit needs no
        implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.record_fits rejected a write-ahead record "
            "but the store implements no lineage overflow"
        )

    def get_lineage_overflow(
        self, doc_ids: Iterable[str]
    ) -> dict[str, DocumentLineage]:
        """Return ``{doc_id: overflow}`` for the ``doc_ids`` that have any."""
        return {}

    def delete_lineage_overflow(self, doc_ids: Iterable[str]) -> None:
        """Remove the overflow of every ``doc_id`` (no-op where there is none)."""

    def delete(self, doc_id: str) -> None:
        """Remove the record for ``doc_id`` (no-op if absent)."""
        ...

    def list_all(self) -> list[DocStatusRecord]:
        """Return every stored record (used to diff against the new corpus)."""
        ...

    def diff(self, incoming: dict[str, str], scope: str | None = None) -> DocumentDelta:
        """Classify ``{doc_id: content_hash}`` against stored state.

        Returns the new / changed / unchanged / deleted partition driving an
        incremental run. With ``scope``, only stored records of that scope are
        candidates for ``deleted`` (a record without a scope never is), so one
        tenant's or one corpus's run cannot delete another's documents.
        Callers key documents per scope (the source scope is part of the
        ``doc_id``), so an incoming id never matches another scope's record.
        ``scope=None`` considers every stored record. A stored ``FAILED``
        record is ``changed`` even when its hash matches, so it is retried.

        An adapter should also report the distinct scopes of the records it
        read in ``DocumentDelta.stored_scopes`` (the scan already reads them);
        the pipeline uses them only to warn about other corpora sharing the
        run's namespace, and leaving it empty just silences that warning.
        """
        ...
