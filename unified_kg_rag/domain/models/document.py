# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


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
    written, or FAILED.
    """

    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"


# Content hash of a write-ahead record (status PENDING): an incremental run
# writes one for every document it is about to (re)index before it touches the
# stores, and replaces it once the document's artifacts are written. No
# document hashes to it, so a document whose run was interrupted is classified
# changed on the next run (re-extracted, even if its content went back to the
# last indexed version) or deleted if it is gone.
PENDING_CONTENT_HASH = "pending"


class DocStatusRecord(BaseModel):
    """Per-document state persisted across indexing runs (DocStatusPort).

    Stores the content hash for change detection and the ids of every graph
    artifact the document produced, so a delta run can merge or remove exactly
    the affected entities/relationships/text-units/communities.
    """

    doc_id: str = Field(
        description="Stable document identifier: hash of the index namespace, the "
        "corpus source scope and the path relative to the corpus root"
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
        "as deleted; None marks a record written before scopes existed, which is "
        "never deleted automatically",
    )
    file_path: str | None = Field(
        default=None, description="Source file path relative to the corpus root"
    )
    entity_ids: list[str] = Field(
        default_factory=list, description="Entity ids produced by this document"
    )
    relationship_ids: list[str] = Field(
        default_factory=list, description="Relationship ids produced by this document"
    )
    text_unit_ids: list[str] = Field(
        default_factory=list, description="Text-unit ids produced by this document"
    )
    community_ids: list[str] = Field(
        default_factory=list, description="Community ids this document contributed to"
    )
    claim_ids: list[str] = Field(
        default_factory=list, description="Claim ids produced by this document"
    )
    community_report_ids: list[str] = Field(
        default_factory=list,
        description="Community-report ids this document contributed to",
    )
    error_info: str | None = Field(
        default=None, description="Error detail if status is FAILED"
    )
    failure_count: int = Field(
        default=0,
        ge=0,
        description="Consecutive runs that recorded this content FAILED; 0 once "
        "processed. A record written before the field existed reads as 0",
    )
    extraction_fingerprint: str | None = Field(
        default=None,
        description="Fingerprint of the settings that determined the document's "
        "artifacts (parsing through graph resolution: models, prompts, entity "
        "types, chunking...) when it was last extracted. A diff given the run's "
        "fingerprint classifies a document whose stored fingerprint differs as "
        "changed. None marks a record written before the field existed: it is "
        "never re-extracted for a settings change (a reset re-extracts it)",
    )
    created_at: str | None = Field(
        default=None, description="ISO timestamp when first registered"
    )
    updated_at: str | None = Field(
        default=None, description="ISO timestamp of the last status change"
    )


class DocumentLineage(BaseModel):
    """Per-document artifact attribution for an incremental commit.

    Lets the orchestrator record exactly which entities/relationships/text-units/
    communities a single document produced (and under which index suffix), so a
    later change/delete can remove that document's *exclusive* artifacts without
    touching ones shared with surviving documents.
    """

    doc_id: str = Field(description="Stable document id these artifacts belong to")
    suffix: str = Field(
        default="default",
        description="Index/label suffix the artifacts were written under",
    )
    file_path: str | None = Field(
        default=None, description="Source file path relative to the corpus root"
    )
    entity_ids: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)
    text_unit_ids: list[str] = Field(default_factory=list)
    community_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    community_report_ids: list[str] = Field(default_factory=list)


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
    settings_changed: list[str] = Field(
        default_factory=list,
        description="The ``changed`` doc ids whose content and status alone "
        "would read unchanged: their stored extraction fingerprint differs "
        "from the run's, so they are re-extracted with the current settings",
    )
    orphan_overflow: list[str] = Field(
        default_factory=list,
        description="Owner doc ids of write-ahead lineage overflow the diff "
        "read next to a record that is not PENDING, or next to no record, "
        "sorted: overflow left by a commit or removal interrupted before "
        "deleting it, which no later run reads. The incremental write-ahead "
        "deletes it for the owners of its namespace. A store that does not "
        "report it leaves it empty",
    )

    @property
    def is_empty(self) -> bool:
        """True when no documents were added, changed, or removed."""
        return not (self.new or self.changed or self.deleted)

    @property
    def to_process(self) -> list[str]:
        """Doc ids requiring (re)indexing this run: new + changed."""
        return self.new + self.changed


class Document(BaseModel):
    """A parsed source document — a pure domain model (no LangChain base).

    LangChain loader output is converted into this model at the adapter/shared
    boundary (``shared.utils.document_converter``). ``page_content`` keeps the
    field name LangChain's Document used and is the document's only copy of
    its text. JSON written by earlier versions still loads: their extra keys
    (LangChain's ``id``/``type``, ``file_type``, ``total_pages``, ``pages``
    and ``content``) are ignored, except that ``content.text`` is read as
    the text (see :meth:`from_json_file`).
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
        # Earlier versions kept the text authoritative under ``content.text``
        # (their ``page_content`` could be stale).
        legacy_text = (data.get("content") or {}).get("text")
        if isinstance(legacy_text, str):
            data["page_content"] = legacy_text
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
