# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Collection, Iterator
from enum import StrEnum
from typing import Any, Literal, Self, TypeVar

_E = TypeVar("_E", bound=BaseException)


def causes(exc: BaseException) -> Iterator[BaseException]:
    """``exc`` and the exceptions it was raised from, innermost last."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def own_text(exc: BaseException) -> str | None:
    """The package's own message of ``exc`` (its ``own_text``), composed of
    ids, counts, settings, paths and log-safe error names (``error_name``),
    never a model's, a provider's or a document's text: ``True`` marks its
    whole message (:func:`error_text`), a string the part of it that is
    (the rest, a store's error it quotes, is DEBUG's); else ``None``. That
    text is shown and logged whatever the error's class: an unexpected
    failure's text otherwise is not (:func:`shows_error_text`)."""
    own = getattr(exc, "own_text", False)
    if own is True:
        return error_text(exc)
    return own if isinstance(own, str) and own else None


def has_own_text(exc: BaseException) -> bool:
    """Whether ``exc`` holds the package's own message (:func:`own_text`)."""
    return own_text(exc) is not None


def with_own_text(exc: _E) -> _E:
    """``exc``, marked as holding the package's own message
    (:func:`has_own_text`)."""
    exc.own_text = True  # type: ignore[attr-defined]
    return exc


class GraphRAGException(Exception):
    # The package's own message (:func:`own_text`): ``True``, all of it; a
    # string, the part of it that is; ``False``, none.
    own_text: bool | str = False

    @classmethod
    def caused_by(cls, context: str, cause: BaseException, **kwargs: Any) -> Self:
        """``cls(f"{context}: {cause}")``, to raise ``from cause``, whose own
        text (:func:`own_text`) is ``context`` and the cause's ``error_name``
        (its type, code and HTTP status): ``context`` is the package's (an
        operation, a URI, ids), the cause's message a store's, a provider's
        or a library's, which may quote a document, a prompt or an ARN. A
        surface and an INFO+ record show the own text, so an unexpected
        failure still says what failed; ``log_error`` logs the rest at
        DEBUG."""
        from .utils.common import error_name

        error = cls(f"{context}: {cause}", **kwargs)
        error.own_text = f"{context}: {error_name(cause)}"
        return error

    def described(self, namespaces: Collection[str] | None = None) -> str:
        """The message a surface shows (:func:`error_text`): the message and
        what the error holds as data for its reader (the namespaces of
        :class:`IndexNotFoundError`), naming only ``namespaces`` of them
        when given."""
        return str(self)


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

    ``indexing_hint``, what an indexing run can do about it, ends the message;
    :attr:`reason` is the message without it (what a reader reports). The
    message names the error it wraps by ``error_name``, never its text
    (:func:`has_own_text`).
    """

    own_text = True

    def __init__(self, message: str = "", *, indexing_hint: str = "") -> None:
        super().__init__(f"{message} {indexing_hint}" if indexing_hint else message)
        self.reason = message


class DocStatusFormatError(ConfigurationError):
    """The doc-status registry's table holds an earlier release's format.

    Like :class:`DocStatusRegistryError`, ``indexing_hint`` ends the message
    and :attr:`reason` is the message without it.
    """

    def __init__(self, message: str = "", *, indexing_hint: str = "") -> None:
        super().__init__(f"{message} {indexing_hint}" if indexing_hint else message)
        self.reason = message


class EvaluationException(GraphRAGException):
    pass


class GuardrailError(AWSServiceError):
    """A content guardrail could not screen a text (API error, bad reply).

    The query fails closed: it is not answered unguarded.
    """


class GraphError(GraphRAGException):
    pass


class GraphQueryError(GraphError):
    """A structured graph query (search strategy ``structured``) that did
    not produce a result; the strategy falls back."""


class InvalidGraphQueryError(GraphQueryError, ValueError):
    """A graph query plan its validation refuses. ``field`` names the plan
    field (``nodes[1].types``), ``reason`` what is wrong with it: the message
    the planner's repair call reads."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


class GraphQueryCompileError(GraphQueryError):
    """A compiled graph query the static checker refused. A bug in the
    compiler, never a model error: the query is not sent."""


class GraphQueryExecutionError(GraphQueryError):
    """The graph store failed one structured query without being unusable
    (a query it refused, or its server-side time limit): that query falls
    back, the store stays in use. ``timed_out`` says it ran out of time."""

    def __init__(self, message: str, *, timed_out: bool = False) -> None:
        super().__init__(message)
        self.timed_out = timed_out


class ModelError(GraphRAGException):
    """A model call that failed or a model that cannot be built: a
    language, embedding or rerank model."""


class EmbeddingModelError(ModelError):
    """An embedding model that cannot be built (an unsupported id or
    dimension) or a query embedding that failed after its retries."""


class IndexNotFoundError(GraphRAGException):
    """No store index exists for the queried suffix (not ingested, or a typo).

    ``ingested`` lists the suffixes the store holds an index for (``None``:
    it could not tell). They are data, not part of the message, which
    reaches logs, error answers and callers who may not read the other
    namespaces: a surface adds them (:meth:`described`), the MCP server
    only those it serves."""

    def __init__(self, message: str, *, ingested: list[str] | None = None) -> None:
        super().__init__(message)
        self.ingested = ingested

    def described(self, namespaces: Collection[str] | None = None) -> str:
        if self.ingested is None:
            return str(self)
        if namespaces is None:
            held = self.ingested
            what = "Suffixes ingested on this store"
        else:
            held = [name for name in self.ingested if name in namespaces]
            what = "Namespaces served here that hold an index"
        if not held:
            nothing = (
                "No suffix was ingested on this store"
                if namespaces is None
                else "No namespace served here holds an index"
            )
            return f"{self} {nothing}."
        return f"{self} {what}: {', '.join(held)}."


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


class InvalidArgumentError(GraphRAGException, ValueError):
    """An argument outside what the caller may pass: an invalid query input
    (an empty query, an unknown key or strategy), an MCP tool call's bound,
    a namespace or a strategy the server does not serve, a missing or
    conflicting argument, a strategy outside a chain's
    ``allowed_strategies``, or an output file that exists. Raised before any
    model call (error class ``invalid_argument``)."""


# The name before 0.3.0 (MCP's), kept: the same class.
ToolArgumentError = InvalidArgumentError


class LanguageModelError(ModelError):
    """A language model that cannot be built (an unsupported setting for it)
    or an answer it gave that cannot be used."""


class ModelOutputError(LanguageModelError, ValueError):
    """Model output that holds no object of the shape a prompt asked for
    (``parse_model_object``). A ``ValueError``, as ``json.JSONDecodeError``
    is, so a caller that caught that still does."""


class ModelOutputCutOffError(ModelOutputError):
    """Model output whose last JSON object is unclosed: cut off, so not
    repaired (a fixer could only invent the rest)."""


class LLMOutputTruncatedError(LanguageModelError):
    """A model response stopped at its output-token limit, cut mid-answer, or
    an extraction whose every entity lacks a name (cut off the same way). Not
    retried: the same input meets the same limit."""


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
    failed stage. Its message is the question (:func:`has_own_text`)."""

    own_text = True


class PipelineResumeError(GraphRAGException):
    pass


class PipelineStageError(PipelineExecutionError):
    """A stage failed; ``failed_files``: the files its items failed on;
    ``own_text``: its message is the package's own (:func:`own_text`; a
    string: the part of it that is), not one that quotes a store's or a
    model's error."""

    def __init__(
        self,
        *args: object,
        failed_files: list[str] | None = None,
        own_text: bool | str = False,
    ) -> None:
        super().__init__(*args)
        self.failed_files = list(failed_files or ())
        self.own_text = own_text


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
    """A rerank model that cannot be built or a rerank call that failed. A
    query raises it only with ``search.reranking.required``; otherwise it
    answers from the fused order."""


class ModelUnreachableError(ModelError, ConnectionError):
    """A model server could not be reached (connection refused, no route, a
    name that does not resolve, no connection in time): no model call can
    succeed until it runs. The message names the provider kind, its
    ``base_url`` and how to start it. Also a builtin ``ConnectionError``, so
    retrieval fails the query rather than answering without the model."""


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


class ErrorClass(StrEnum):
    """What a failure means to the caller, the same on every surface, which
    names it by its value (the stable key: a value is never renamed): the
    CLIs' JSON ``error_class`` and exit codes (``cli.conventions.EXIT_CODES``),
    the MCP server's tool errors (``[value] message``), the console's
    ``error_class``; docs/reference/exit-codes-and-json.md. ``GraphRAGChain``
    names the class of a failure it absorbed in ``metadata["error_class"]``."""

    INVALID_FILTER = "invalid_filter"
    INVALID_ARGUMENT = "invalid_argument"
    NOT_INDEXED = "not_indexed"
    INDEX_NOT_FOUND = "index_not_found"
    CONFIGURATION = "configuration"
    TIMEOUT = "timeout"
    STORE_UNREACHABLE = "store_unreachable"
    MODEL_UNREACHABLE = "model_unreachable"
    FAILURE = "failure"

    @property
    def is_caller_error(self) -> bool:
        """Whether the caller's request is at fault (an argument, a filter,
        a namespace without an index, a strategy it does not hold), not the
        service: the query metrics do not count it in ``query_failed``."""
        return self in _CALLER_ERRORS


_CALLER_ERRORS = frozenset(
    {
        ErrorClass.INVALID_FILTER,
        ErrorClass.INVALID_ARGUMENT,
        ErrorClass.NOT_INDEXED,
        ErrorClass.INDEX_NOT_FOUND,
    }
)


# Most specific first: MethodologyNotIndexedError is a ConfigurationError,
# a ModelUnreachableError and a StoreConnectionError ConnectionErrors.
_CLASSES: tuple[tuple[type[BaseException], ErrorClass], ...] = (
    (InvalidFilterError, ErrorClass.INVALID_FILTER),
    (InvalidArgumentError, ErrorClass.INVALID_ARGUMENT),
    (DocumentNotFoundError, ErrorClass.INVALID_ARGUMENT),
    (MethodologyNotIndexedError, ErrorClass.NOT_INDEXED),
    (IndexNotFoundError, ErrorClass.INDEX_NOT_FOUND),
    (QueryTimeoutError, ErrorClass.TIMEOUT),
    (ModelUnreachableError, ErrorClass.MODEL_UNREACHABLE),
    (StoreConnectionError, ErrorClass.STORE_UNREACHABLE),
    (ConfigurationError, ErrorClass.CONFIGURATION),
)


def classify(exc: BaseException) -> ErrorClass:
    """The :class:`ErrorClass` of ``exc``; ``FAILURE`` for any other."""
    for kind, error_class in _CLASSES:
        if isinstance(exc, kind):
            return error_class
    return ErrorClass.FAILURE


# The result metadata key naming the class of a failure the chain absorbed.
ERROR_CLASS_KEY = "error_class"


def error_class_of(value: object) -> ErrorClass:
    """The :class:`ErrorClass` a result's ``metadata["error_class"]`` names
    (``FAILURE`` for none or an unknown one)."""
    try:
        return ErrorClass(str(value))
    except ValueError:
        return ErrorClass.FAILURE


# What a surface says of a failure whose text it does not show its audience
# (shown_error_text): a provider's text (an unexpected failure's) reaches no
# surface, a store or model endpoint no MCP caller. The server logs the error
# under the ids the surface names (``log_error``): an outage's text, an
# unexpected failure's type and where it happened (its text at DEBUG).
INTERNAL_ERROR_MESSAGE = (
    "The call failed with an unexpected error (a store or model error); the "
    "log has the details."
)

# Who reads a surface's error text: ``caller``, a client the server does not
# trust (an MCP agent), or ``operator``, the console's user, who configured
# the stores and models and reads the server log.
ErrorAudience = Literal["caller", "operator"]


def shows_error_text(error_class: ErrorClass, audience: ErrorAudience) -> bool:
    """Whether a surface shows ``audience`` the text of an error of
    ``error_class``. A caller sees the text of what it can act on (a
    refusal: ``ErrorClass.is_caller_error``) and of the deadline, never an
    endpoint; an operator sees all but an unexpected failure's (``failure``),
    whose text is a provider's or a library's (it may quote a prompt, an ARN
    or corpus text)."""
    if audience == "caller":
        return error_class.is_caller_error or error_class is ErrorClass.TIMEOUT
    return error_class is not ErrorClass.FAILURE


def shown_error_text(
    error: BaseException | ErrorClass,
    text: str | None = None,
    *,
    audience: ErrorAudience,
    namespaces: Collection[str] | None = None,
) -> str:
    """The text a surface shows ``audience`` for ``error`` (an exception, or
    the class of one a result names, as the error answer of
    ``processing.ignore_errors``): ``text`` (by default
    :func:`error_text` of the exception) when :func:`shows_error_text`,
    else :data:`INTERNAL_ERROR_MESSAGE` (an operator also sees a failure
    whose message is the package's own: :func:`has_own_text`). The one rule
    every surface (the MCP tools, the console's API, its Ask events and
    history, the CLIs' output and JSON status, a stage's recorded error)
    applies."""
    error_class = error if isinstance(error, ErrorClass) else classify(error)
    if not shows_error_text(error_class, audience):
        own = (
            own_text(error)
            if audience == "operator" and isinstance(error, BaseException)
            else None
        )
        return INTERNAL_ERROR_MESSAGE if own is None else own
    if text is None:
        assert isinstance(error, BaseException), "text is required with a class"
        text = error_text(error, namespaces)
    return text


def operator_error_text(error: BaseException) -> str:
    """:func:`shown_error_text` for the operator, a masked failure named by
    what it is (``error_name``: its type, code and HTTP status) before the
    generic message: what the CLIs print and a stage records, whose reader
    has no query id to look the log up by."""
    from .utils.common import error_name

    text = shown_error_text(error, audience="operator")
    return f"{error_name(error)}: {text}" if text == INTERNAL_ERROR_MESSAGE else text


def named_failure_text(error_class: ErrorClass, name: str, shown: str) -> str:
    """How a saved result (an evaluation report, a failed metric's reason)
    names a failure: its class, its type (``name``) and the text a surface
    shows the operator of it (``shown``), left out for an unexpected failure
    (:data:`INTERNAL_ERROR_MESSAGE`): its text may quote the prompt, a
    model's reply or a document, and results are shown and uploaded (the
    log has it at DEBUG)."""
    if shown == INTERNAL_ERROR_MESSAGE:
        return f"[{error_class.value}] {name}"
    return f"[{error_class.value}] {name}: {shown}"


def result_error_text(error: BaseException) -> str:
    """:func:`named_failure_text` of ``error``, named by ``error_name`` (its
    type, code and HTTP status): what a saved result records of an error.
    One holding the package's own message (:func:`own_text`, which names
    the error it wraps: ``caused_by``) is named by its type alone."""
    from .utils.common import error_name

    name = type(error).__name__ if has_own_text(error) else error_name(error)
    return named_failure_text(
        classify(error), name, shown_error_text(error, audience="operator")
    )


# The public name of :func:`classify` (``unified_kg_rag.classify_error``).
classify_error = classify

# The ids a query failure carries (:func:`tag_query_failure`): the query's
# log lines' ``query_id`` and its trace's ``trace_id``.
QUERY_FAILURE_IDS = ("query_id", "trace_id")


def tag_query_failure(
    exc: BaseException, query_id: str | None, trace_id: str | None = None
) -> None:
    """Attach the ids of the query ``exc`` failed (``exc.query_id``,
    ``exc.trace_id``), so a surface reporting it ties it to the query's log
    lines and trace (:func:`query_failure_ids`). An id already attached (by
    an inner query) is kept; an exception that takes no attribute keeps
    none."""
    for name, value in zip(QUERY_FAILURE_IDS, (query_id, trace_id), strict=True):
        if value and not getattr(exc, name, None):
            try:
                setattr(exc, name, value)
            except (AttributeError, TypeError):
                return


def error_text(exc: BaseException, namespaces: Collection[str] | None = None) -> str:
    """The message a surface shows for ``exc``: with what a package error
    holds for its reader (:meth:`GraphRAGException.described`; only the
    ``namespaces`` given of the ones it names)."""
    if isinstance(exc, GraphRAGException):
        return exc.described(namespaces)
    return str(exc)


def waiter_copy(exc: BaseException) -> BaseException:
    """A copy of ``exc``, the failure of one call several waiters share,
    for one of them: the same type, ``args`` and attributes (what
    :func:`classify` and the surfaces read), without the ids another
    waiter's query attached (:func:`tag_query_failure`), so each query names
    its own. ``exc`` itself if it cannot be copied."""
    try:
        clone = type(exc).__new__(type(exc), *exc.args)
        clone.args = exc.args
        clone.__dict__.update(
            (k, v) for k, v in exc.__dict__.items() if k not in QUERY_FAILURE_IDS
        )
    except Exception:  # noqa: BLE001 - a type that builds otherwise
        return exc
    return clone


def query_failure_ids(exc: BaseException | str | None) -> dict[str, str]:
    """The ids :func:`tag_query_failure` attached to ``exc`` (none: ``{}``)."""
    if not isinstance(exc, BaseException):
        return {}
    ids = {name: getattr(exc, name, None) for name in QUERY_FAILURE_IDS}
    return {name: value for name, value in ids.items() if isinstance(value, str)}
