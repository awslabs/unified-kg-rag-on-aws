# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterator


def causes(exc: BaseException) -> Iterator[BaseException]:
    """``exc`` and the exceptions it was raised from, innermost last."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


class GraphRAGException(Exception):
    pass


class AWSServiceError(GraphRAGException):
    pass


class CacheSyncError(AWSServiceError):
    """A pipeline cache sync with remote storage (S3) failed or was partial.

    Raised instead of returning a partial result so the run that lost its
    checkpoint exits non-zero, not the next run (another task, or an
    ``--enabled-stages`` window) with a misleading "missing stage" error.
    """


class ConfigurationError(GraphRAGException, ValueError):
    """The configuration (or a CLI option merged into it) is invalid for the run.

    Raised before any store is written, so a contradictory setting fails the
    run instead of being acted on.
    """


class SchemaSourceError(ConfigurationError):
    """The extraction schema source cannot be loaded (an unknown kind, a
    missing or invalid file) or holds a schema the config refuses."""


class SeedGraphError(ConfigurationError):
    """A seed graph cannot be imported: its source cannot be read, or what
    it states breaks a rule of the import (a dangling edge, a duplicate node
    id, a name collision, a cap)."""


class DataProcessingError(GraphRAGException):
    pass


class DocStatusRegistryError(AWSServiceError):
    """The incremental-indexing doc-status registry could not be read or written.

    Raised instead of falling back to a full rebuild: without the registry an
    incremental run cannot tell new, changed and deleted documents apart, and a
    full rebuild of only this run's documents replaces the live index content.
    """


class EvaluationException(GraphRAGException):
    pass


class GuardrailError(AWSServiceError):
    """A content guardrail could not screen a text (API error, bad reply).

    The query fails closed: it is not answered unguarded.
    """


class GraphError(GraphRAGException):
    pass


class ModelError(GraphRAGException):
    pass


class EmbeddingModelError(ModelError):
    pass


class IndexNotFoundError(GraphRAGException):
    """No store index exists for the queried suffix (not ingested, or a typo)."""


class DocumentNotFoundError(GraphRAGException, LookupError):
    """``DataIngestionPipeline.delete_documents`` named a document the
    doc-status registry does not hold for the run's index namespace and
    source scope. Raised before anything is written."""


class InvalidFilterError(GraphRAGException, ValueError):
    """A caller search filter names a key no target index or label declares."""


class MethodologyNotIndexedError(ConfigurationError):
    """A query's search strategy reads a methodology (``graphrag``,
    ``lightrag``, ...) the queried index namespace does not hold: it was not
    in ``indexing.methodologies`` when the namespace was last written. Raised
    before any model call; ``run-rag`` exits 2."""


class ToolArgumentError(GraphRAGException, ValueError):
    """An argument outside what the serving surface allows: an MCP tool
    call's bound, a namespace or a strategy the server does not serve, a
    missing or conflicting argument, or a strategy outside a chain's
    ``allowed_strategies``. Raised before any model call."""


class LanguageModelError(ModelError):
    pass


class LLMOutputTruncatedError(LanguageModelError):
    """A model response stopped at its output-token limit, cut mid-answer, or
    an extraction whose every entity lacks a name (cut off the same way). Not
    retried: the same input meets the same limit."""


class LanguageModelTimeoutError(LanguageModelError, TimeoutError):
    """Model calls gave up after their call timeout (``BatchProcessor``) without
    an answer (also a builtin ``TimeoutError``)."""


class PipelineExecutionError(GraphRAGException):
    pass


class PipelineInterrupted(KeyboardInterrupt):
    """The run was asked to stop by a signal (SIGTERM from ECS StopTask or a
    Step Functions timeout) after its finished stages were checkpointed.

    A ``KeyboardInterrupt``, like Ctrl-C, so the ``except Exception`` handlers
    that turn a stage error into a failed stage (and ``continue_on_error``)
    do not swallow it. ``signum`` gives the CLI its exit code (128 + signum).
    """

    def __init__(self, signum: int) -> None:
        super().__init__(f"interrupted by signal {signum}")
        self.signum = signum


class NamespaceLeaseError(PipelineExecutionError):
    """Another run holds the index namespace's write lease, or this run lost
    it: two runs writing one namespace at once would corrupt the registry
    and the stores, so the run stops before (or instead of) writing."""


class ConfirmationDeclinedError(PipelineExecutionError):
    """A question the run asks before it writes (a reset that drops another
    source scope's content) was answered no, or no one could answer it: the
    run stops there, before any write, as a cancellation rather than a
    failed stage."""


class PipelineResumeError(GraphRAGException):
    pass


class PipelineStageError(PipelineExecutionError):
    pass


class PipelineStateError(GraphRAGException):
    pass


class QueryTimeoutError(GraphRAGException, TimeoutError):
    """A query ran past its deadline (``search.query_timeout_seconds``), or a
    streamed answer sent no first chunk in time; the query's calls in flight
    were cancelled. Also a builtin ``TimeoutError``. ``timeout_seconds`` is
    the deadline that expired."""

    def __init__(self, message: str, *, timeout_seconds: float) -> None:
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class RerankModelError(ModelError):
    pass


class StoreConnectionError(AWSServiceError, ConnectionError):
    """A store (OpenSearch, Neptune) is unusable: unreachable, lost, overloaded.

    Raised by the store clients when a request could not be delivered or
    answered after the client's own retries: refused, DNS or TLS failure, a
    connection the server closed, or OpenSearch still answering HTTP
    429/502/503/504. A ``ConnectionError``, so retrieval reports it rather than
    reading it as "no results" (``adapters.store_errors``).
    """


class StoreTimeoutError(StoreConnectionError):
    """A store request got no answer in time (after the client's retries)."""


class StoreConfigurationError(AWSServiceError, ConfigurationError):
    """A store refuses this client the same way on every attempt.

    No endpoint, credentials or auth method to connect with, or a request the
    store rejects (``StoreRequestRejectedError``) or denies
    (``StoreAccessDeniedError``). ``reason`` is the server's reason, if it gave
    one.
    """

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason


class StoreAccessDeniedError(StoreConfigurationError):
    """The store denied the caller (HTTP 401/403): credentials or permissions."""


class StoreRequestRejectedError(StoreConfigurationError):
    """The store rejected a request as invalid (HTTP 400), as it will on retry."""
