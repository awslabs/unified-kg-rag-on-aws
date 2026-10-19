# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-run delta detection for incremental indexing.

Given the freshly loaded corpus and a :class:`DocStatusPort`, computes a stable
``doc_id`` and ``content_hash`` per document and classifies the corpus into
new / changed / unchanged / deleted / failed (:class:`DocumentDelta`).

``doc_id`` is derived from the document's index namespace, its corpus source
scope and its path relative to the corpus root (not the parser-assigned
``document_id``, which changes with the content) so the same file maps to the
same registry entry on every run, wherever the corpus is staged, and two
corpora written to one namespace never share an entry. ``content_hash`` is
computed over the aggregated document text so a content edit is detected even
when the path is unchanged.

A document whose content is unchanged is still re-extracted when the settings
that shaped its artifacts changed: the run passes its extraction fingerprint
(models, prompts, entity types, chunking...) to the diff, which compares it
with the one each record stored when the document was last extracted (a
record without one is compared by content only).

Deletion is scoped: a run only classifies registry documents of its own scope
(index namespace + corpus source, see :func:`registry_scope`) as deleted, and
never a corpus file that merely failed to parse this run.

A document whose processing keeps killing the run (its record counts the
runs that died, see ``domain.ingestion.attempts``) is isolated once the count
reaches ``max_attempts``: deferred while the run bisects the documents it
died with, then quarantined.

A corpus whose source scope changed (a local directory that moved) leaves its
records under the old scope. They are never taken over on a guess: whether a
directory exists on this host says nothing about a corpus another host
indexes into the same registry. A run that is told the old scope is gone
(``indexing.retire_source_scopes``) classifies its records as deleted
(:func:`retire_source_scopes`), and :func:`other_local_scopes` names the other
local source directories of the run's namespace so the run can warn.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from unified_kg_rag.domain.ingestion.attempts import (
    plan_attempts,
    release_quarantined,
)
from unified_kg_rag.domain.models import (
    Constants,
    DocStatus,
    DocStatusRecord,
    Document,
    DocumentDelta,
    settings_changed,
)
from unified_kg_rag.shared import ConfigurationError, get_logger
from unified_kg_rag.shared.utils.document_identity import (
    REGISTRY_NAMESPACE_KEY,
    REGISTRY_SOURCE_KEY,
    RELATIVE_PATH_KEY,
    compute_doc_id,
    compute_document_id,
    compute_text_hash,
    is_local_path_scope,
    normalize_source_path,
    normalize_source_scope,
    registry_namespace,
    relative_source_path,
)

__all__ = [
    "assign_document_identity",
    "assign_registry_source",
    "compute_content_hash",
    "compute_doc_id",
    "detect_delta",
    "document_doc_id",
    "filter_documents_to_process",
    "fingerprint_documents",
    "other_local_scopes",
    "registry_scope",
    "RetiredScope",
    "retire_source_scopes",
    "retired_scope_groups",
    "scope_namespace",
]

if TYPE_CHECKING:
    # Type only: the port module imports the domain models.
    from unified_kg_rag.ports import DocStatusPort

logger = get_logger(__name__)


def compute_content_hash(document: Document) -> str:
    """Compute a content hash for change detection over the document text."""
    return compute_text_hash(document.page_content)


def _document_suffix(document: Document) -> str | None:
    """The ``index`` metadata value the document's artifacts inherit."""
    value = document.metadata.get(Constants.INDEX.value)
    if isinstance(value, str) and value:
        return value
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return None


def assign_document_identity(
    document: Document,
    source_root: str | Path,
    additional_suffix: str | None = None,
    store_tag: str | None = None,
) -> None:
    """Re-derive ``document``'s identity against the corpus root, in place.

    Records the path relative to ``source_root`` and the registry namespace in
    the metadata, and sets ``document_id`` from that path plus the full-content
    hash, so the id does not depend on where the corpus was synced or checked
    out and two files with the same name in different folders never share an
    id (or text-unit ids, which derive from it).

    A document whose ``file_path`` lies outside the root (a pre-parsed JSON
    document records its original path) keeps the relative path the loader
    recorded for the file it was read from.
    """
    relative = (
        relative_source_path(document.file_path, source_root)
        or document.metadata.get(RELATIVE_PATH_KEY)
        or normalize_source_path(document.file_path)
    )
    document.metadata[RELATIVE_PATH_KEY] = relative
    document.metadata[REGISTRY_NAMESPACE_KEY] = registry_namespace(
        _document_suffix(document), additional_suffix, store_tag
    )
    document.document_id = compute_document_id(relative, compute_content_hash(document))


def assign_registry_source(document: Document, source_scope: str) -> None:
    """Record the corpus source scope ``document`` was read from, in place.

    Part of the registry key (see :func:`document_doc_id`), so two corpora
    written to one index namespace keep separate records even for files with
    the same relative path.
    """
    document.metadata[REGISTRY_SOURCE_KEY] = source_scope
    # Its text units' ids derive from ``document_id``: keyed by the registry
    # key, a text unit belongs to one record even when two corpora of one
    # namespace hold the same file, so removing one never drops the other's.
    document.document_id = compute_document_id(
        document_doc_id(document), compute_content_hash(document)
    )


def _identity(document: Document) -> tuple[str, str]:
    relative = document.metadata.get(RELATIVE_PATH_KEY) or document.file_path
    namespace = document.metadata.get(REGISTRY_NAMESPACE_KEY) or registry_namespace(
        _document_suffix(document)
    )
    return relative, namespace


def document_doc_id(document: Document) -> str:
    """The registry key of ``document`` (see :func:`compute_doc_id`).

    Uses the identity :func:`assign_document_identity` and
    :func:`assign_registry_source` recorded; a document that never went
    through them (built directly by a caller) is keyed by its ``file_path`` as
    given, the namespace of its ``index`` metadata and no source scope.
    """
    relative, namespace = _identity(document)
    return compute_doc_id(
        relative, namespace, document.metadata.get(REGISTRY_SOURCE_KEY)
    )


def registry_scope(namespace: str, source_scope: str) -> str:
    """The scope a run diffs in: its index namespace and its corpus source.

    Two runs share a scope only when they write the same index namespace from
    the same corpus source, so one tenant's run, or a run over a subfolder,
    never classifies another scope's documents as deleted.
    """
    return f"{namespace}|{source_scope}"


def scope_namespace(scope: str | None) -> str | None:
    """The index namespace of a :func:`registry_scope`.

    This is how a record's namespace is read: ``DocStatusRecord.suffix`` is
    the item suffix the indexers are called with (``index_value``), which
    every ``indexing.additional_suffix`` shares, so it cannot tell two
    namespaces in one registry table apart. ``None`` for a record written
    before scopes existed, whose namespace is unknown.
    """
    if scope is None:
        return None
    namespace, separator, _ = scope.partition("|")
    return namespace if separator else None


def fingerprint_documents(documents: list[Document]) -> dict[str, str]:
    """Map each document to ``{doc_id: content_hash}`` for diffing.

    When two documents resolve to the same ``doc_id`` (same source path), the
    last one wins, matching load-order precedence.
    """
    fingerprints: dict[str, str] = {}
    for document in documents:
        fingerprints[document_doc_id(document)] = compute_content_hash(document)
    return fingerprints


def detect_delta(
    documents: list[Document],
    doc_status: DocStatusPort,
    scope: str | None = None,
    failed_doc_ids: Iterable[str] = (),
    max_failures: int | None = None,
    retired_scopes: Iterable[str | RetiredScope] = (),
    extraction_fingerprint: str | None = None,
    max_attempts: int | None = None,
) -> tuple[DocumentDelta, dict[str, str]]:
    """Classify ``documents`` against the persisted registry.

    Args:
        documents: The documents read successfully this run.
        doc_status: The registry.
        scope: The run's :func:`registry_scope`. Only stored documents of this
            scope can be classified deleted. ``None`` diffs against the whole
            registry (the direct API without a scope).
        failed_doc_ids: Registry ids of corpus files that failed to parse or
            load this run. They are reported as ``failed`` and never deleted.
        max_failures: A document whose unchanged content was recorded FAILED
            this many consecutive times is classified unchanged instead of
            changed, so a deterministic failure is not re-extracted every run.
            ``None`` retries without limit.
        retired_scopes: Registry scopes (:func:`registry_scope`), or
            :class:`RetiredScope` groups of them, every record of which is
            classified deleted this run (see :func:`retire_source_scopes`). Needs a ``scope``, which it must
            not contain.
        extraction_fingerprint: The run's extraction settings fingerprint
            (``DocStatusRecord.extraction_fingerprint``). A document whose
            record stored another one is classified changed, so it is
            re-extracted with the current settings; a record without one is
            compared by content only. ``None`` compares by content only.
        max_attempts: A document whose processing of this content killed
            the run this many consecutive runs (``attempts.settled_attempts``)
            is a suspect: classified ``quarantined`` instead of changed once
            it died alone, else the run processes half of the suspects and
            defers the other new and changed documents
            (``attempts.plan_attempts``). ``None`` retries without limit.

    Returns:
        The :class:`DocumentDelta` plus the ``{doc_id: content_hash}``
        fingerprint map (so callers can persist new/changed records without
        recomputing hashes).

    Raises:
        ConfigurationError: ``retired_scopes`` holds the run's own ``scope``,
            or is given without one.
    """
    retired = _groups(retired_scopes)
    if retired and (scope is None or any(scope in g.scopes for g in retired)):
        raise ConfigurationError(
            f"Cannot retire the run's own registry scope ({scope!r}): every "
            "document of the run would be deleted"
        )
    fingerprints = fingerprint_documents(documents)
    delta = doc_status.diff(
        fingerprints, scope=scope, extraction_fingerprint=extraction_fingerprint
    )
    if retired:
        retire_source_scopes(doc_status, delta, retired, own_scope=scope)
    if max_attempts is not None and delta.quarantined_unchanged:
        release_quarantined(
            delta, doc_status.get_many(delta.quarantined_unchanged), max_attempts
        )
    if (max_failures is not None or max_attempts is not None) and delta.changed:
        stored = doc_status.get_many(delta.changed)
        if max_attempts is not None:
            plan_attempts(delta, stored, fingerprints, max_attempts)
        if max_failures is not None:
            _stop_retrying_exhausted(
                delta, stored, fingerprints, max_failures, extraction_fingerprint
            )
    failed = set(failed_doc_ids) - set(fingerprints)
    if failed:
        delta.deleted = [doc_id for doc_id in delta.deleted if doc_id not in failed]
        delta.failed = sorted(failed)
        logger.warning(
            "%d corpus files failed to parse or load; their indexed content is "
            "kept (not treated as deleted) until a run reads them",
            len(failed),
        )
    logger.info(
        "Delta detected: %d new, %d changed, %d unchanged, %d deleted, %d failed, "
        "%d quarantined, %d deferred",
        len(delta.new),
        len(delta.changed),
        len(delta.unchanged),
        len(delta.deleted),
        len(delta.failed),
        len(delta.quarantined),
        len(delta.deferred),
    )
    if delta.settings_changed:
        logger.warning(
            "Re-extracting %d documents with unchanged content because the "
            "extraction settings (models, prompts, entity types, chunking...) "
            "changed since they were indexed",
            len(delta.settings_changed),
        )
    return delta, fingerprints


class RetiredScope(NamedTuple):
    """A configured retired scope and the registry scopes it may mean.

    ``ambiguous``: the scopes are different readings of the value (a
    ``source_scope`` name and a directory), so at most one of them may hold
    records; otherwise they are spellings of one scope, all retired.
    """

    value: str
    scopes: tuple[str, ...]
    ambiguous: bool = False


def _as_given(value: str) -> tuple[list[str], bool]:
    return [value], False


def retired_scope_groups(
    retired: Iterable[str],
    source_scope: str,
    namespace: str,
    readings: Callable[[str], tuple[list[str], bool]] = _as_given,
) -> list[RetiredScope]:
    """The registry scopes to retire for each of the ``retired`` values.

    ``readings`` maps a value to the source scopes it may mean and whether
    they are different readings (default: the value as given). Each source
    scope is normalized the way the registry stores it (see
    :func:`normalize_source_scope`) and joined with the run's ``namespace``
    (:func:`registry_scope`), so retiring only ever reaches records of the
    namespace the run writes.

    Raises:
        ConfigurationError: A retired source scope is empty, or is the run's
            own ``source_scope`` (every document of the run would be deleted;
            for an ambiguous value, when every reading is).
    """
    own = normalize_source_scope(source_scope)
    groups: list[RetiredScope] = []
    for value in retired:
        spellings, ambiguous = readings(value)
        normalized = [normalize_source_scope(s) for s in spellings]
        if not value.strip() or not all(normalized):
            raise ConfigurationError(
                "indexing.retire_source_scopes / --retire-source-scope: a "
                "source scope is empty"
            )
        kept = [n for n in normalized if n != own] if ambiguous else normalized
        if not kept or own in kept:
            raise ConfigurationError(
                "indexing.retire_source_scopes / --retire-source-scope: "
                f"{value!r} is this run's own source scope. Retiring it would "
                "delete every document of this corpus; retire the scope the "
                "corpus was indexed under before it moved."
            )
        scopes = tuple(dict.fromkeys(registry_scope(namespace, n) for n in kept))
        groups.append(
            RetiredScope(value.strip(), scopes, ambiguous and len(scopes) > 1)
        )
    return groups


def _groups(retired_scopes: Iterable[str | RetiredScope]) -> list[RetiredScope]:
    return list(
        dict.fromkeys(
            g if isinstance(g, RetiredScope) else RetiredScope(g, (g,))
            for g in retired_scopes
        )
    )


def retire_source_scopes(
    doc_status: DocStatusPort,
    delta: DocumentDelta,
    retired_scopes: Iterable[str | RetiredScope],
    own_scope: str | None = None,
) -> int:
    """Classify every record of ``retired_scopes`` as deleted in ``delta``.

    The run then prunes their exclusive artifacts in its one removal pass
    (artifacts a surviving document shares are kept, stripped of their text
    units) and deletes their records once the removal succeeded, exactly like
    files removed from its own corpus. Costs one registry diff (a projected
    scan) per retired scope.

    A retired scope also matches the stored scopes of its namespace whose
    source scope normalizes to the same value (see
    :func:`normalize_source_scope`): a configured
    ``document_parsing.source_scope`` is stored as given, so a local path
    with a trailing slash would otherwise never match its normalized retire
    value. The stored scopes come from ``delta.stored_scopes``, and
    ``own_scope`` (the run's scope) is never retired.

    A :class:`RetiredScope` retires the records of all its scopes, or,
    when ``ambiguous``, of the one reading that has records.

    Returns:
        The number of records added to ``delta.deleted``.

    Raises:
        ConfigurationError: More than one reading of an ambiguous value has
            records (retiring both could delete another corpus).
    """
    deleted = set(delta.deleted)
    added = 0
    for group in _groups(retired_scopes):
        by_scope = {
            scope: [
                doc_id
                for variant in _stored_spellings(scope, delta.stored_scopes)
                if variant != own_scope
                for doc_id in doc_status.diff({}, scope=variant).deleted
            ]
            for scope in group.scopes
        }
        matched = [scope for scope, ids in by_scope.items() if ids]
        if group.ambiguous and len(matched) > 1:
            raise ConfigurationError(
                f"--retire-source-scope / indexing.retire_source_scopes "
                f"{group.value!r} is ambiguous: the registry holds records "
                f"under each of {', '.join(repr(s) for s in matched)}. Give "
                "the old directory as './<path>' or an absolute path."
            )
        doc_ids = list(
            dict.fromkeys(
                doc_id
                for scope in matched
                for doc_id in by_scope[scope]
                if doc_id not in deleted
            )
        )
        deleted.update(doc_ids)
        if not doc_ids:
            logger.warning(
                "Retired scope %r has no doc-status records (already retired, "
                "or not spelled the way the registry stores it)",
                group.value,
            )
            continue
        logger.warning(
            "Retiring %d doc-status records of scope %r: their exclusive "
            "artifacts are removed by this run",
            len(doc_ids),
            group.value,
        )
        delta.deleted.extend(doc_ids)
        added += len(doc_ids)
    return added


def _stored_spellings(retired: str, stored_scopes: Iterable[str]) -> list[str]:
    """``retired`` and every stored scope of its namespace whose source scope
    normalizes to the same value."""
    namespace = scope_namespace(retired)
    if namespace is None:
        return [retired]
    source = normalize_source_scope(retired[len(namespace) + 1 :])
    return list(
        dict.fromkeys(
            [
                retired,
                *(
                    stored
                    for stored in stored_scopes
                    if scope_namespace(stored) == namespace
                    and normalize_source_scope(stored[len(namespace) + 1 :]) == source
                ),
            ]
        )
    )


def other_local_scopes(
    delta: DocumentDelta, scope: str, namespace: str, retired: Iterable[str] = ()
) -> list[str]:
    """Other local source directories with records in the run's namespace.

    Read from ``delta.stored_scopes``, without a registry call. Each is either
    another local corpus written to the same index namespace (possibly on
    another host) or this corpus's location before a move; the two cannot be
    told apart from here, so the caller only reports them.

    Args:
        scope: The run's :func:`registry_scope`, left out.
        namespace: The run's index namespace.
        retired: Registry scopes retired this run, left out with every
            stored spelling of them (see :func:`retire_source_scopes`).

    Returns:
        The source scopes, sorted.
    """
    prefix = f"{namespace}|"
    skip = {
        scope,
        *(
            spelling
            for value in retired
            for spelling in _stored_spellings(value, delta.stored_scopes)
        ),
    }
    return sorted(
        stored[len(prefix) :]
        for stored in delta.stored_scopes
        if stored.startswith(prefix)
        and stored not in skip
        and is_local_path_scope(stored[len(prefix) :])
    )


def _stop_retrying_exhausted(
    delta: DocumentDelta,
    stored: Mapping[str, DocStatusRecord],
    fingerprints: dict[str, str],
    max_failures: int,
    extraction_fingerprint: str | None = None,
) -> None:
    """Move changed documents that used up their retries to ``unchanged``.

    A document whose extraction settings changed since it failed is retried
    (its earlier failures were with other settings). ``stored`` holds the
    changed documents' records, read in one batch.
    """
    exhausted: list[str] = []
    labels: list[str] = []
    for doc_id in delta.changed:
        record = stored.get(doc_id)
        if (
            record is not None
            and record.status is DocStatus.FAILED
            and record.content_hash == fingerprints[doc_id]
            and record.failure_count >= max_failures
            and not settings_changed(
                record.extraction_fingerprint, extraction_fingerprint
            )
        ):
            exhausted.append(doc_id)
            labels.append(record.file_path or doc_id)
    if not exhausted:
        return
    delta.changed = [doc_id for doc_id in delta.changed if doc_id not in exhausted]
    delta.unchanged.extend(exhausted)
    logger.warning(
        "Not retrying %d documents that failed %d consecutive runs with unchanged "
        "content (kept FAILED; edit the file or raise "
        "indexing.max_document_failures to retry): %s",
        len(exhausted),
        max_failures,
        ", ".join(labels),
    )


def filter_documents_to_process(
    documents: list[Document], delta: DocumentDelta
) -> list[Document]:
    """Return only the documents that need (re)indexing this run (new + changed)."""
    to_process = set(delta.to_process)
    return [
        document for document in documents if document_doc_id(document) in to_process
    ]
