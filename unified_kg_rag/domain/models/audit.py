# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-query audit record (``GraphRAGChain`` writes one per query on the
``unified_kg_rag.audit`` logger; docs/reference/telemetry.md#audit-record).

It says who asked, when, against which index build and settings, what the
answer rests on, which models it called at what cost, and how it ended. It
holds ids, counts and digests only, never the question's, the answer's or a
source's text: the question and the answer are a SHA-256 digest and a
length, so a stored answer can be checked against its record without the
record keeping either.
"""

from datetime import datetime
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from .usage import LLMUsage

# The version of the record's fields: raised when one is renamed, removed or
# changes meaning (a new optional field keeps it).
AUDIT_RECORD_VERSION: Final = 1


class AuditOutcome(StrEnum):
    """How a query ended (``QueryAuditRecord.outcome``)."""

    ANSWERED = "answered"
    ABSTAINED = "abstained"
    BLOCKED = "blocked"
    # Answered by the error answer of processing.ignore_errors.
    ERROR = "error"
    # Raised: the service's failure (a timeout included).
    FAILED = "failed"
    # Raised or answered with a caller's error (ErrorClass.is_caller_error).
    REFUSED = "refused"
    # Its events' reader stopped before the end (a stream closed early).
    STOPPED = "stopped"


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AuditText(_Record):
    """A text the record stands in for: its length and SHA-256 digest."""

    length: int = Field(ge=0, description="Characters in the text")
    sha256: str = Field(description="Hex SHA-256 of the text's UTF-8 bytes")


class AuditBuild(_Record):
    """The index build the query read: its namespace's build record as the
    query's admission read it (kept for up to the build-record cache's
    lifetime, methodology_check.HELD_TTL_SECONDS)."""

    pipeline_id: str = Field(description="The ingestion run that wrote it")
    mode: str = Field(description="'full' or 'incremental'")
    built_at: datetime | None = Field(
        default=None, description="When that run finished indexing (UTC)"
    )
    package_version: str | None = Field(
        default=None, description="The package version that ran it"
    )
    extraction_fingerprint: str | None = Field(
        default=None, description="Fingerprint of its extraction settings"
    )
    full_build_pipeline_id: str | None = Field(
        default=None,
        description="Of an incremental build, the full build it applies on",
    )


class AuditEvidence(_Record):
    """What the answer rests on, by id; each list holds at most
    ``shared.observability.MAX_RECORDED_IDS`` ids, as a retrieval span does
    (``truncated`` then says so)."""

    sources: int = Field(
        ge=0,
        description="Sources the answer model was given (a search's results)",
    )
    source_ids: list[str] = Field(
        default_factory=list, description="Their stored ids, in rank order"
    )
    cited_source_ids: list[str] = Field(
        default_factory=list,
        description="The ids of the context sections the answer cites ([n])",
    )
    cited_text_unit_ids: list[str] = Field(
        default_factory=list, description="The text units those sections rest on"
    )
    cited_document_ids: list[str] = Field(
        default_factory=list, description="The documents those sections rest on"
    )
    truncated: bool = Field(
        default=False, description="Whether a list was cut at its cap"
    )


class QueryAuditRecord(_Record):
    """One query's audit record (module docstring)."""

    record_version: Literal[1] = Field(
        default=AUDIT_RECORD_VERSION, description="The version of these fields"
    )
    query_id: str = Field(description="The query's id (its log lines carry it)")
    trace_id: str | None = Field(
        default=None, description="Its trace id, with tracing on"
    )
    finished_at: datetime = Field(description="When the query ended (UTC)")
    latency_ms: float = Field(ge=0, description="From its start to its end")
    mode: Literal["rag", "search"] = Field(
        description="An answer (rag) or a retrieval-only search"
    )
    namespace: str | None = Field(
        default=None, description="The index namespace it read"
    )
    user_id: str | None = Field(
        default=None, description="RAGInput.user_id: who asked, when known"
    )
    conversation_id: str | None = Field(
        default=None, description="RAGInput.conversation_id"
    )
    requested_strategy: str | None = Field(
        default=None, description="The search strategy asked for"
    )
    strategy: str | None = Field(
        default=None, description="The strategy that ran, when one did"
    )
    outcome: AuditOutcome = Field(description="How it ended")
    error_class: str | None = Field(
        default=None, description="A failure's ErrorClass value"
    )
    degraded: bool = Field(default=False, description="Answered without a step")
    query_limited: bool = Field(
        default=False, description="Its query limits cut it short"
    )
    package_version: str = Field(description="The package version that answered")
    config_fingerprint: str = Field(
        description=(
            "SHA-256 of every setting that can change an answer "
            "(cache_keys.answer_settings_fingerprint; the run-eval "
            "manifest's retrieval_config_sha256)"
        )
    )
    build: AuditBuild | None = Field(
        default=None,
        description="The index build it read (None: no build record read)",
    )
    question: AuditText | None = Field(
        default=None, description="The question asked, as a digest"
    )
    answer: AuditText | None = Field(
        default=None, description="The answer returned, as a digest"
    )
    evidence: AuditEvidence | None = Field(
        default=None, description="What it rests on (None: it returned none)"
    )
    models: list[str] = Field(
        default_factory=list, description="The model ids it sent requests to"
    )
    usage: LLMUsage | None = Field(
        default=None, description="Its model usage (RAGOutput.metadata['usage'])"
    )
    cost_usd: float | None = Field(
        default=None,
        description="Its USD cost by models.prices (None: a model is unpriced)",
    )
    unpriced_models: list[str] = Field(
        default_factory=list, description="The models models.prices cannot price"
    )
