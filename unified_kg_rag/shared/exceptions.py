# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
class GraphRAGException(Exception):
    pass


class AWSServiceError(GraphRAGException):
    pass


class CacheSyncError(AWSServiceError):
    """A pipeline cache sync with remote storage (S3) failed or was partial.

    Raised instead of returning a partial result so a phased run (one Step
    Functions task per phase) exits non-zero at the phase that lost its
    checkpoint, not later with a misleading "missing stage" error.
    """


class ConfigurationError(GraphRAGException, ValueError):
    """The configuration (or a CLI option merged into it) is invalid for the run.

    Raised before any store is written, so a contradictory setting fails the
    run instead of being acted on.
    """


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


class InvalidFilterError(GraphRAGException, ValueError):
    """A caller search filter names a key no target index or label declares."""


class LanguageModelError(ModelError):
    pass


class LLMOutputTruncatedError(LanguageModelError):
    """A model response stopped at its output-token limit, cut mid-answer."""


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


class PipelineResumeError(GraphRAGException):
    pass


class PipelineStageError(PipelineExecutionError):
    pass


class PipelineStateError(GraphRAGException):
    pass


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
