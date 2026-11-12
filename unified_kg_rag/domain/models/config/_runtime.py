# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime sections: memory, caches, logging and observability."""

import re
from enum import Enum

from pydantic import (
    Field,
    field_validator,
)

from ._common import ConfigSection


class MemoryConfig(ConfigSection):
    max_conversations: int = Field(
        default=100,
        ge=1,
        description=(
            "Conversations held in conversation memory; a new one beyond it "
            "drops the one idle longest."
        ),
    )
    max_messages_per_conversation: int = Field(
        default=20,
        ge=1,
        description=("Messages kept per conversation; older ones are dropped."),
    )
    max_conversation_age_hours: int = Field(
        default=168,
        ge=1,
        description="Hours a conversation may stay idle (since its last message); "
        "an older one is dropped on the next memory lookup and starts over",
    )
    max_history_tokens: int = Field(
        default=4000,
        ge=1,
        description=(
            "Tokens (estimated) of a caller's RAGInput.history the follow-up "
            "rewrite reads: of its last max_messages_per_conversation turns, "
            "each answer cut to 1,000 characters, the oldest are dropped until "
            "the rest fit. Conversation memory is not bounded by it. "
            "Unmeasured."
        ),
    )


class LLMResponseCacheConfig(ConfigSection):
    """The ingestion LLM response cache (``adapters.llm.response_cache``)."""

    enabled: bool = Field(
        default=True,
        description="Keep the responses of the ingestion model calls with the "
        "run's stage cache, so a re-run with the same pipeline id (after a "
        "crash, a stop or a timeout) does not call the model again for a "
        "prompt that already answered. Query and evaluation calls are never "
        "cached",
    )
    flush_every: int = Field(
        default=100,
        ge=1,
        description="Responses per cache segment file; a segment is written "
        "(and uploaded with --s3-sync) when it is full and before each stage "
        "checkpoint, so a killed task loses at most this many responses",
    )


class CacheConfig(ConfigSection):
    ttl_seconds: int | None = Field(
        default=86400,
        ge=1,
        description=(
            "Time-to-live of the local stage-cache entries, in seconds; null ="
            " never expire. Not applied with --s3-sync, where the bucket "
            "lifecycle expires the checkpoints."
        ),
    )
    llm_responses: LLMResponseCacheConfig = Field(
        default_factory=LLMResponseCacheConfig,
        description="Ingestion LLM response cache",
    )


class LoggingConfig(ConfigSection):
    level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description=(
            "Log level: DEBUG, INFO, WARNING, ERROR or CRITICAL. DEBUG logs "
            "contain user and corpus text."
        ),
    )
    log_format: str = Field(
        default="structured",
        pattern="^(structured|plain)$",
        description=(
            "structured (JSON records, for CloudWatch and other log stores) or"
            " plain."
        ),
    )
    log_to_file: bool = Field(
        default=False,
        description=(
            "Also write the CLIs' logs to log_file_path (dated per day; a "
            "relative path resolves against the working directory, which "
            "must then be writable). Off by default: the console (or the "
            "container's log driver) carries every record. The package "
            "configures no handler when imported as a library."
        ),
    )
    log_file_path: str = Field(
        default="logs/log.txt",
        min_length=1,
        max_length=255,
        description="Log file path (CLIs only); relative to the working directory",
    )
    library_levels: dict[str, str] = Field(
        default_factory=lambda: {
            "langchain_aws": "WARNING",
            "botocore": "WARNING",
            "urllib3": "WARNING",
            # At DEBUG: numba (of the unstructured extra) logs every JIT
            # compile and httpcore every connection step (httpx).
            "numba": "WARNING",
            "httpcore": "WARNING",
            # At WARNING: every failed OpenSearch request, the 404s of an
            # alias not created yet included (the client's errors are logged
            # by adapters.aws.opensearch). At INFO: every model server
            # request (httpx: Ollama and OpenAI-compatible servers).
            "opensearch": "ERROR",
            "httpx": "WARNING",
            # The OTLP exporter warns on every retry to an unreachable collector.
            "opentelemetry": "ERROR",
        },
        description=(
            "Per-logger levels for chatty third-party libraries (logger name "
            "-> level). In a config file the key replaces the whole map; an "
            "environment variable sets one entry over the defaults."
        ),
    )


class ObservabilityBackend(str, Enum):
    """Where trace spans go when ``observability.enabled`` is true.

    ``none`` installs no exporter: spans go to the process's global
    OpenTelemetry tracer provider, which a host application or
    ``opentelemetry-instrument`` configures. ``otlp`` installs an OTLP
    exporter configured by the standard ``OTEL_EXPORTER_OTLP_*`` environment
    variables. ``langfuse`` and ``phoenix`` install the same OTLP exporter,
    pointed at that vendor's OTLP endpoint (from ``LANGFUSE_*`` /
    ``PHOENIX_*`` variables when the ``OTEL_*`` ones are unset), and add the
    vendor's attribute names (Langfuse observation types, OpenInference span
    kinds and token counts). No vendor SDK is needed.
    """

    NONE = "none"
    OTLP = "otlp"
    LANGFUSE = "langfuse"
    PHOENIX = "phoenix"


class ObservabilityConfig(ConfigSection):
    """OpenTelemetry tracing of ingestion and query runs (off by default).

    Needs the ``otel`` extra for every backend. The
    exporter endpoint and headers come from environment variables
    (``OTEL_EXPORTER_OTLP_*``, or ``LANGFUSE_*`` / ``PHOENIX_*`` for those
    backends), never from this file, so no credential lands in a config file.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Emit OpenTelemetry spans for ingestion, queries and evaluation "
            "(needs the otel extra). OTEL_SDK_DISABLED=true turns it off "
            "regardless."
        ),
    )
    backend: ObservabilityBackend = Field(
        default=ObservabilityBackend.NONE,
        description=(
            "none: spans go to the process's global tracer provider, which a "
            "host application or opentelemetry-instrument sets up. otlp: the "
            "package installs an OTLP exporter configured by the "
            "OTEL_EXPORTER_OTLP_* variables. langfuse / phoenix: the same "
            "exporter pointed at Langfuse or Arize Phoenix, with that vendor's"
            " attribute names added."
        ),
    )
    capture_content: bool = Field(
        default=False,
        description=(
            "Also record prompt and completion text on model spans, and the "
            "query and answer on the kg_rag.query span (after "
            "redact_patterns). Off by default: spans carry counts, ids and "
            "token usage only. Leave it off for confidential corpora."
        ),
    )
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of root traces sampled (backends otlp, langfuse, "
        "phoenix; with 'none' the host's provider samples); child spans follow "
        "their parent's decision",
    )
    service_name: str = Field(
        default="unified-kg-rag",
        min_length=1,
        description=(
            "OpenTelemetry service.name, and the Phoenix project; "
            "OTEL_SERVICE_NAME (and PHOENIX_PROJECT_NAME) win when set."
        ),
    )
    redact_patterns: list[str] = Field(
        default_factory=list,
        description="Regular expressions replaced with '[REDACTED]' in captured "
        "content and, for the backends this package installs (otlp, langfuse, "
        "phoenix), in every exported string attribute, event attribute and "
        "status message",
    )
    export_timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        le=60.0,
        description="Upper bound on one span-export request (backends otlp, "
        "langfuse, phoenix). OTEL_EXPORTER_OTLP_TRACES_TIMEOUT / "
        "OTEL_EXPORTER_OTLP_TIMEOUT win when set",
    )

    @field_validator("redact_patterns")
    @classmethod
    def _patterns_compile(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(
                    f"Invalid observability.redact_patterns entry {pattern!r}: {e}"
                ) from e
        return patterns
