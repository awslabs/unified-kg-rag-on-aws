# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _read_text_autodetect(path: Path) -> str:
    """Read a file as text, falling back to encoding detection.

    Heterogeneous corpora often contain non-UTF-8 files (e.g. cp949, latin-1).
    Attempt UTF-8 first (the common case), then fall back to charset-normalizer
    detection so such files are recovered rather than dropped. We do not silently
    mojibake: the detected encoding is logged at debug level.

    ``get_logger`` is imported lazily (function-local) to avoid a module-load
    circular import: ``shared`` pulls in ``domain.models`` at import time.
    """
    from unified_kg_rag.shared.logging import get_logger

    logger = get_logger(__name__)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        from charset_normalizer import from_bytes

        raw = path.read_bytes()
        best = from_bytes(raw).best()
        if best is None:
            # Last resort: decode as latin-1, which never raises, so a file is
            # never lost outright (better degraded text than a dropped document).
            logger.debug(
                "Encoding detection failed for '%s'; falling back to latin-1", path
            )
            return raw.decode("latin-1")
        logger.debug("Detected encoding '%s' for '%s'", best.encoding, path)
        return str(best)


class DocStatus(str, Enum):
    """Processing lifecycle of a document in the incremental-indexing registry.

    An incremental run records a document PENDING (the write-ahead record)
    before it touches the stores, then PROCESSED once its artifacts are
    written, or FAILED. A document whose indexing was started and never
    finished ``indexing.max_document_attempts`` consecutive runs (the run died
    each time) is QUARANTINED: it is skipped until its content or the
    extraction settings change.
    """

    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"
    QUARANTINED = "quarantined"


# Content hash of a write-ahead record (status PENDING): an incremental run
# writes one for every document it is about to (re)index before it touches the
# stores, and replaces it once the document's artifacts are written. No
# document hashes to it, so a document whose run was interrupted is classified
# changed on the next run (re-extracted, even if its content went back to the
# last indexed version) or deleted if it is gone.
PENDING_CONTENT_HASH = "pending"


class ExtractionSettings(BaseModel):
    """The settings that shaped what a document produced, field by field.

    Built by ``shared.utils.cache_keys.extraction_settings``: ``values`` maps
    a configuration path to the value in effect (long values by digest). A
    document's registry record holds the :attr:`fingerprint`; the record
    itself is kept once per index namespace, so a later run can tell which
    settings changed and treat a record that differs only by fields at their
    defaults as unchanged (``cache_keys.changed_settings``).
    """

    version: int = Field(description="Format version of the record")
    values: dict[str, Any] = Field(
        default_factory=dict, description="Configuration path -> value in effect"
    )

    @property
    def fingerprint(self) -> str:
        """Short digest of the record (``DocStatusRecord.extraction_fingerprint``)."""
        payload = json.dumps([self.version, self.values], sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class DocStatusRecord(BaseModel):
    """Per-document state persisted across indexing runs (DocStatusPort).

    Stores the content hash for change detection and the document's text
    units, by which a delta run finds (and removes or strips) everything the
    document contributed to the stores.
    """

    doc_id: str = Field(
        description="Stable document identifier: ``<index namespace>#<hash of the "
        "namespace, the corpus source scope and the path relative to the corpus "
        "root>``. The namespace prefix is the registry partition the record is "
        "kept in (see ``doc_id_namespace``)"
    )
    content_hash: str = Field(description="Content hash used for change detection")
    status: DocStatus = Field(
        default=DocStatus.PENDING, description="Current processing status"
    )
    suffix: str = Field(
        default="default",
        description="Item suffix (index_value) the document's artifacts were "
        "written under, as passed to the indexers. Every "
        "indexing.additional_suffix shares it, so it does not identify the "
        "index namespace: that is the prefix of ``scope`` (scope_namespace)",
    )
    scope: str | None = Field(
        default=None,
        description="Registry scope (index namespace + corpus source) of the run "
        "that recorded the document. A run only treats documents of its own scope "
        "as deleted; None (a record the direct API wrote without a scope) is "
        "never deleted automatically",
    )
    file_path: str | None = Field(
        default=None, description="Source file path relative to the corpus root"
    )
    text_unit_ids: list[str] = Field(
        default_factory=list,
        description="Text-unit ids of this document. Every other artifact is "
        "found in the stores by the text units it cites, so removing the "
        "document removes what it alone states and strips the rest",
    )
    error_info: str | None = Field(
        default=None, description="Error detail if status is FAILED"
    )
    failure_count: int = Field(
        default=0,
        ge=0,
        description="Consecutive runs that recorded this content FAILED; 0 once "
        "processed",
    )
    attempt_count: int = Field(
        default=0,
        ge=0,
        description="Consecutive runs that started indexing this document "
        "(wrote its PENDING record) and did not finish it: an interruption "
        "counts, a FAILED or PROCESSED commit resets it to 0",
    )
    attempted_hash: str | None = Field(
        default=None,
        description="Content hash of the version the attempts indexed (a "
        "PENDING record's ``content_hash`` is ``PENDING_CONTENT_HASH``): only "
        "attempts at the same content count toward quarantine",
    )
    extraction_fingerprint: str | None = Field(
        default=None,
        description="Fingerprint of the settings that determined the document's "
        "artifacts (parsing through graph resolution: models, prompts, entity "
        "types, chunking...) when it was last extracted. A diff given the run's "
        "fingerprint classifies a document whose stored fingerprint differs as "
        "changed. None (a record written without one, e.g. by the direct API) "
        "is never re-extracted for a settings change",
    )
    created_at: str | None = Field(
        default=None, description="ISO timestamp when first registered"
    )
    updated_at: str | None = Field(
        default=None, description="ISO timestamp of the last status change"
    )


class DocumentLineage(BaseModel):
    """The text units one document produced in an incremental commit.

    A text unit belongs to one document (its id derives from the document's
    registry key), and every graph and vector artifact cites the text units
    it was stated in, so the text units are all a later run needs to remove
    what the document contributed.
    """

    doc_id: str = Field(description="Stable document id these artifacts belong to")
    suffix: str = Field(
        default="default",
        description="Index/label suffix the artifacts were written under",
    )
    file_path: str | None = Field(
        default=None, description="Source file path relative to the corpus root"
    )
    text_unit_ids: list[str] = Field(default_factory=list)

    # The per-artifact id lists of earlier releases are gone: passing one is
    # an error, not silently ignored.
    model_config = ConfigDict(extra="forbid")


class StoredDocument(NamedTuple):
    """What the delta rule reads of a stored registry record."""

    content_hash: str
    scope: str | None
    failed: bool
    extraction_fingerprint: str | None


def settings_changed(stored: str | None, run: str | None) -> bool:
    """Whether a record extracted with fingerprint ``stored`` was extracted
    with other settings than the run's ``run``; unknown on either side (a
    record without a fingerprint, a run comparing by content) is not."""
    return run is not None and stored not in (None, run)


class DocumentDelta(BaseModel):
    """Partition of an incoming corpus relative to the persisted registry.

    Produced by ``DocStatusPort.diff``: ``changed`` documents exist with a
    different content hash, ``deleted`` were present before but are absent now.
    ``is_empty`` lets an update run short-circuit when nothing changed.
    """

    new: list[str] = Field(
        default_factory=list, description="Doc ids not previously seen"
    )
    changed: list[str] = Field(
        default_factory=list, description="Doc ids whose content hash changed"
    )
    unchanged: list[str] = Field(
        default_factory=list, description="Doc ids with an identical content hash"
    )
    deleted: list[str] = Field(
        default_factory=list,
        description="Doc ids of this run's scope that are no longer in the corpus",
    )
    failed: list[str] = Field(
        default_factory=list,
        description="Doc ids of corpus files that failed to parse or load this "
        "run. They are neither processed nor deleted: their indexed content, if "
        "any, is kept until a run reads them successfully",
    )
    stored_scopes: list[str] = Field(
        default_factory=list,
        description="Distinct scopes of the stored records the diff read "
        "(records without a scope left out), sorted. Informational: it never "
        "changes the partition. A store that does not report it leaves it "
        "empty",
    )
    quarantined: list[str] = Field(
        default_factory=list,
        description="Doc ids whose indexing this content was started and never "
        "finished ``indexing.max_document_attempts`` consecutive runs: not "
        "processed; the run removes what the attempts wrote and records them "
        "QUARANTINED",
    )
    settings_changed: list[str] = Field(
        default_factory=list,
        description="The ``changed`` doc ids whose content and status alone "
        "would read unchanged: their stored extraction fingerprint differs "
        "from the run's, so they are re-extracted with the current settings",
    )

    @classmethod
    def classify(
        cls,
        incoming: Mapping[str, str],
        stored: Mapping[str, StoredDocument],
        scope: str | None = None,
        extraction_fingerprint: str | None = None,
    ) -> "DocumentDelta":
        """The delta rule every ``DocStatusPort.diff`` applies.

        ``incoming`` is ``{doc_id: content_hash}``, ``stored`` the records the
        store read for the diff. A document is ``new`` without a record,
        ``changed`` when its hash differs or its record is FAILED (so it is
        retried), and also when only its extraction settings changed
        (``settings_changed`` too); else ``unchanged``. With ``scope``, only
        stored records of that scope can be ``deleted``.
        """
        delta = cls()
        for doc_id, content_hash in incoming.items():
            record = stored.get(doc_id)
            if record is None:
                delta.new.append(doc_id)
            elif record.content_hash != content_hash or record.failed:
                delta.changed.append(doc_id)
            elif settings_changed(
                record.extraction_fingerprint, extraction_fingerprint
            ):
                delta.changed.append(doc_id)
                delta.settings_changed.append(doc_id)
            else:
                delta.unchanged.append(doc_id)
        delta.deleted = [
            doc_id
            for doc_id, record in stored.items()
            if doc_id not in incoming and (scope is None or record.scope == scope)
        ]
        delta.stored_scopes = sorted(
            {r.scope for r in stored.values() if r.scope is not None}
        )
        return delta

    @property
    def is_empty(self) -> bool:
        """True when no documents were added, changed, or removed."""
        return not (self.new or self.changed or self.deleted or self.quarantined)

    @property
    def to_process(self) -> list[str]:
        """Doc ids requiring (re)indexing this run: new + changed."""
        return self.new + self.changed


class Document(BaseModel):
    """A parsed source document — a pure domain model (no LangChain base).

    LangChain loader output is converted into this model at the adapter/shared
    boundary (``shared.utils.document_converter``). ``page_content`` keeps the
    field name LangChain's Document used and is the document's only copy of
    its text.
    """

    page_content: str = Field(description="Full plain-text content of the document")
    document_id: str = Field(description="Unique identifier for the document")
    file_name: str = Field(description="Name of the source file")
    file_path: str = Field(description="Path to the source file")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the document"
    )
    error_info: str | None = Field(
        default=None, description="Error information if document processing failed"
    )

    @field_validator("document_id", "file_name", mode="before")
    @classmethod
    def validate_non_empty_string(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Field cannot be empty or just whitespace")
        return v.strip()

    @field_validator("file_path", mode="before")
    @classmethod
    def validate_file_path(cls, v: Any) -> str:
        return str(v)

    @classmethod
    def from_json_file(cls, input_path: str | Path) -> "Document":
        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(f"JSON file not found at '{path}'")

        data = json.loads(_read_text_autodetect(path))
        data.setdefault("page_content", "")
        return cls.model_validate(data)

    def to_json_file(self, output_path: str | Path | None = None, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        json_data = self.model_dump_json(**kwargs)

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_data, encoding="utf-8")

        return json_data

    @property
    def is_error(self) -> bool:
        return self.error_info is not None

    def __str__(self) -> str:
        status = "ERROR" if self.is_error else "OK"
        return f"Document(id={self.document_id}, file='{self.file_name}', chars={len(self.page_content)}, status={status})"
