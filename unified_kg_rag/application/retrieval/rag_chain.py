# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import concurrent.futures
import contextlib
import contextvars
import functools
import logging
import threading
import time
import unicodedata
import uuid
import weakref
from collections import OrderedDict
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Iterator,
    Mapping,
    MutableMapping,
)
from contextlib import AbstractContextManager, aclosing, nullcontext
from dataclasses import dataclass, field
from typing import Any, Literal, Self, TypeVar, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import (
    BaseOutputParser,
    StrOutputParser,
)
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.runnables.config import merge_configs
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from structlog.contextvars import (
    bind_contextvars,
    get_contextvars,
    reset_contextvars,
)

from unified_kg_rag.adapters.llm.chain_factory import (
    STRICT_STREAM_TRUNCATION,
    FixingJsonOutputParser,
    create_json_output_parser,
    setup_chain,
)
from unified_kg_rag.adapters.llm.reachability import (
    is_unreachable_error,
    named_model_error,
)
from unified_kg_rag.adapters.llm.retry import retry_deadline
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
    current_ledger,
    degradation_scope,
    query_scope,
    record_degradation,
)
from unified_kg_rag.adapters.retrieval.memory_manager import (
    HISTORY_ANSWER_CHARS,
    ConversationKey,
    MemoryManager,
    history_buffer,
)
from unified_kg_rag.adapters.retrieval.strategy_registry import (
    QueryInput,
    RunRecord,
    get_strategy_spec,
    query_inputs,
    run_records,
    wrapped_strategy,
)
from unified_kg_rag.adapters.retrieval.token_manager import (
    EMPTY_CONTEXT_PLACEHOLDER,
    ContextSection,
    OptimizedContext,
    TokenManager,
)

# Importing the package executes each strategy module's @register_strategy
# decorator, populating the strategy registry used by _get_strategy_instance.
from unified_kg_rag.adapters.search_strategies import (  # noqa: F401
    DriftSearchStrategy,
    GlobalSearchStrategy,
    IterativeSearchStrategy,
    LightRAGSearchStrategy,
    LocalSearchStrategy,
    SimpleSearchStrategy,
)
from unified_kg_rag.application.bootstrap import (
    build_providers,
    build_query_guard,
)
from unified_kg_rag.application.observability import configure_observability
from unified_kg_rag.application.retrieval import (
    citations,
    filter_validation,
    methodology_check,
    query_processing,
)

# The extraction cache's scope lives with the extraction key; re-exported
# for the evaluation runners.
from unified_kg_rag.application.retrieval.query_processing import (
    _EXTRACTION_SCOPE,
)
from unified_kg_rag.application.retrieval.query_processing import (
    extraction_cache_scope as extraction_cache_scope,
)
from unified_kg_rag.application.retrieval.text_units import fetch_text_units
from unified_kg_rag.domain.ingestion.cache_keys import index_language
from unified_kg_rag.domain.ingestion.methodologies import NamespaceHoldings
from unified_kg_rag.domain.ingestion.report_references import REPORT_FINDINGS_FIELD
from unified_kg_rag.domain.models import (
    GENERAL_FALLBACKS,
    Citation,
    Config,
    LanguageCode,
    Methodology,
    ModelRole,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
    SectionType,
    Source,
    validate_safe_name,
)
from unified_kg_rag.domain.prompts import (
    AnswerGenerationPrompt,
    BasePrompt,
    ComprehensiveAnswerPrompt,
    QueryRewritePrompt,
    StrategySelectionPrompt,
)
from unified_kg_rag.domain.prompts.retrieval import (
    ABSTENTION_INSTRUCTIONS,
    CITATION_INSTRUCTIONS,
    DECLINE_INSTRUCTIONS,
    HEDGE_INSTRUCTIONS,
    QUERY_RESULT_INSTRUCTIONS,
    describe_routable_strategies,
)
from unified_kg_rag.domain.retrieval import source_fields
from unified_kg_rag.domain.retrieval.abstention import (
    ABSTENTION_REASON_KEY,
    CONTEXT_SUFFICIENT_KEY,
    NO_CONTEXT_REASON_KEY,
    REASON_INSUFFICIENT_CONTEXT,
    REASON_MARKER,
    REASON_NO_CONTEXT,
    MarkerStreamFilter,
    split_marker,
)
from unified_kg_rag.domain.retrieval.degradation import (
    DegradationReport,
    DegradationStep,
)
from unified_kg_rag.domain.retrieval.filters import (
    check_filter_values,
    typed_filter_values,
)
from unified_kg_rag.domain.retrieval.index_prefixes import HELD_ARTIFACTS_KEY
from unified_kg_rag.domain.retrieval.no_context import (
    NO_CONTEXT_ANSWERS,
    is_no_context_answer,
    no_context_answer,
)
from unified_kg_rag.domain.retrieval.source_fields import seed_provenance
from unified_kg_rag.domain.retrieval.store_names import (
    namespace_suffix,
    resolve_namespace,
)
from unified_kg_rag.ports.guardrail import (
    ContentGuardPort,
    GroundingContext,
    GuardAction,
    GuardSource,
    GuardVerdict,
)
from unified_kg_rag.ports.model_factory import LLMFactoryPort
from unified_kg_rag.shared import (
    ConfigurationError,
    QueryTimeoutError,
    StoreConnectionError,
    get_logger,
    observability,
)
from unified_kg_rag.shared.exceptions import (
    ERROR_CLASS_KEY,
    ErrorClass,
    InvalidArgumentError,
    MethodologyNotIndexedError,
    classify,
    error_class_of,
    tag_query_failure,
)
from unified_kg_rag.shared.genai import ERROR_TYPE, GEN_AI_CONVERSATION_ID
from unified_kg_rag.shared.logging import log_failure
from unified_kg_rag.shared.metrics import (
    QUERY_ABSTAINED_METRIC,
    QUERY_DEGRADED_METRIC,
    QUERY_FAILED_METRIC,
    QUERY_LATENCY_METRIC,
    QUERY_METRICS_NAMESPACE,
    QUERY_THROTTLED_METRIC,
)
from unified_kg_rag.shared.usage import UsageMeter, metering, usage_scope
from unified_kg_rag.shared.utils import (
    configure_event_loop,
    error_digest,
    named_choice,
    strip_embedding_fields,
    strip_invisible,
    text_digest,
)
from unified_kg_rag.shared.utils.concurrency import (
    cancel_on_exit,
    discard,
    gather_bounded,
    settle,
)

logger = get_logger(__name__)

# The input key of the error a query's admission met checking its namespace.
_MISSING_NAMESPACE = "_missing_namespace"


def _names(strategies: Collection[SearchStrategy]) -> str:
    """``strategies`` as a message lists them, in declaration order."""
    return ", ".join(s.value for s in SearchStrategy if s in strategies)


# The fixed reply when retrieval produced no context is no_context_answer's,
# in the query's language (NO_CONTEXT_ANSWER, the English one, is kept for
# importers); the output metadata then carries ``abstained=True``
# (``abstention_reason: no_context``) so evaluation can tell it from a real
# answer. Also the answer of a declined answer that was only the [NO_ANSWER]
# token (search.abstention.strip_marker).
NO_CONTEXT_ANSWER = NO_CONTEXT_ANSWERS[LanguageCode.EN]

# What a reported source keeps of each of a report's findings.
_FINDING_IDS = ("entity_ids", "relationship_ids")

# The characters one joined guardrail check of the model-written results
# reads at most (their texts joined by blank lines; a longer text is checked
# alone). ApplyGuardrail counts a text unit per 1,000 characters and
# allows 25 text units a second by default; the AWS Machine Learning Blog
# post on long-context inputs suggests chunks of 12 units, checked one after
# another, so a joined check never nears the quota on its own.
_JOINED_CHECK_CHARS = 12_000

# The model-written query texts a caller sees (query processing's rewrite,
# translation, entities and LightRAG keywords, and the search query built
# from them), screened as a synthesized result is: (text fields, list fields).
_PROCESSED_QUERY_MODEL_TEXT = (
    ("standalone_query", "translated_query", "final_query"),
    ("entities", "hl_keywords", "ll_keywords"),
)
_SEARCH_QUERY_MODEL_TEXT = (("query",), ("entity_focus", "hl_keywords", "ll_keywords"))
_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _query_model_texts(
    model: BaseModel, fields: tuple[tuple[str, ...], ...]
) -> list[str]:
    """The non-empty strings of ``model``'s ``fields`` (text, then list)."""
    text_fields, list_fields = fields
    found = [getattr(model, name) for name in text_fields]
    found.extend(v for name in list_fields for v in getattr(model, name))
    return [v for v in found if isinstance(v, str) and v]


def _replace_query_texts(
    model: _ModelT,
    fields: tuple[tuple[str, ...], ...],
    screened: dict[str, str | None],
    *,
    refused: str | None,
) -> _ModelT:
    """``model`` with its ``fields`` screened: a masked text replaced, a
    refused list item left out and a refused text field set to ``refused``
    (the original query for a required one, else ``None``)."""
    text_fields, list_fields = fields
    update: dict[str, Any] = {}
    for name in text_fields:
        value = getattr(model, name)
        if isinstance(value, str) and value in screened:
            kept = screened[value]
            update[name] = (
                kept
                if kept is not None
                else (refused if model.model_fields[name].is_required() else None)
            )
    for name in list_fields:
        values = getattr(model, name)
        update[name] = [
            kept
            for v in values
            if (kept := screened.get(v, v) if isinstance(v, str) else v) is not None
        ]
    return model.model_copy(update=update)


def _string_leaves(value: Any) -> list[str]:
    """Every non-empty string inside ``value`` (dicts and lists walked)."""
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, dict):
        return [s for item in value.values() for s in _string_leaves(item)]
    if isinstance(value, list | tuple):
        return [s for item in value for s in _string_leaves(item)]
    return []


def _replace_leaves(value: Any, screened: dict[str, str | None]) -> Any:
    """``value`` with each string replaced by its screened text (a refused
    one emptied, or left out of its list)."""
    if isinstance(value, str):
        text = screened.get(value, value)
        return "" if text is None else text
    if isinstance(value, dict):
        return {k: _replace_leaves(v, screened) for k, v in value.items()}
    if isinstance(value, list):
        return [
            _replace_leaves(v, screened)
            for v in value
            if not (isinstance(v, str) and screened.get(v, v) is None)
        ]
    return value


def _replace_texts(content: str, screened: dict[str, str | None]) -> str:
    """``content`` with each screened text that changed replaced by its
    screened form (a refused one by ``[removed]``), longest first, so a text
    inside a longer one is not cut first."""
    changed = {t: r for t, r in screened.items() if r != t}
    for text in sorted(changed, key=len, reverse=True):
        content = content.replace(text, changed[text] or "[removed]")
    return content


def _texts_at(value: Any, path: list[str]) -> list[str]:
    """The strings under the dotted ``path`` of ``value`` (``RunRecord``:
    a list on the way is walked item by item)."""
    if not path:
        return _string_leaves(value)
    if isinstance(value, list):
        return [text for item in value for text in _texts_at(item, path)]
    if isinstance(value, dict):
        return _texts_at(value.get(path[0]), path[1:])
    return []


def _replace_at(value: Any, path: list[str], screened: dict[str, str | None]) -> Any:
    """``value`` with the strings under ``path`` screened (``_replace_leaves``)."""
    if not path:
        return _replace_leaves(value, screened)
    if isinstance(value, list):
        return [_replace_at(item, path, screened) for item in value]
    if isinstance(value, dict) and path[0] in value:
        return {**value, path[0]: _replace_at(value[path[0]], path[1:], screened)}
    return value


def _record_texts(record: dict[str, Any], spec: RunRecord) -> list[str]:
    """The model-written strings of a strategy's run ``record``
    (``spec.model_text``), each once."""
    return list(
        dict.fromkeys(
            text
            for path in spec.model_text
            for text in _texts_at(record, path.split("."))
        )
    )


def _screened_record(
    record: dict[str, Any], spec: RunRecord, screened: dict[str, str | None]
) -> dict[str, Any]:
    """``record`` with its model-written strings replaced by their screened
    text; a refused one is left out of its list, or emptied."""
    for path in spec.model_text:
        record = _replace_at(record, path.split("."), screened)
    return record


# How an ``auto`` query chose its strategy, in the output metadata:
# ``decided_by`` is ``router`` (the router's reply named it),
# ``only_candidate`` (one strategy applies the query's filters on the
# indexes held: no router call) or ``fallback`` (the router failed or named
# none; ``_fallback_strategy``), and ``candidates`` the strategies it chose
# among. Absent for a query that named its strategy.
ROUTING_KEY = "routing"

DEFAULT_ERROR_MESSAGE: str = (
    "I apologize, but an error occurred while processing your request. Please try again in a moment."
)


class _RouteAnswer(BaseModel):
    """The router's answer (``StrategySelectionPrompt``)."""

    strategy: str


# Every strategy's name: a bare-name router answer is read without JSON.
_STRATEGY_NAMES = frozenset(strategy.value for strategy in SearchStrategy)


class ProcessedQuery(BaseModel):
    """How the chain rewrote one query before searching it.

    ``final_query`` is what the strategy searched: the original, rewritten
    as a standalone question from the conversation history and translated
    when those steps ran. ``entities`` and the LightRAG keywords are the
    query-side extractions the resolved strategy read (empty when it read
    none or query processing was disabled).
    """

    original_query: str = Field(
        description="The original user query before any processing"
    )
    standalone_query: str | None = Field(
        default=None,
        description="The follow-up rewritten as a standalone question from the "
        "conversation history (memory queries with history only)",
    )
    translated_query: str | None = Field(
        default=None,
        description="The query translated to target language if translation was performed",
    )
    final_query: str = Field(description="The final processed query used for search")
    entities: list[str] = Field(
        default_factory=list, description="List of entities extracted from the query"
    )
    hl_keywords: list[str] = Field(
        default_factory=list,
        description="High-level keywords (LightRAG modes only)",
    )
    ll_keywords: list[str] = Field(
        default_factory=list,
        description="Low-level keywords (LightRAG modes only)",
    )


class ConversationTurn(BaseModel):
    """One earlier message of a conversation a caller keeps itself
    (``RAGInput.history``; an unknown key is refused, as on ``RAGInput``)."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"] = Field(
        description="Who wrote it: the user, or the assistant (an earlier answer)"
    )
    content: str = Field(description="The message text")


class RAGInput(BaseModel):
    """One query to :class:`GraphRAGChain` (a dict with these keys also works).

    ``suffix`` selects the index namespace to search: it must match the
    ``processing.document_parsing.index_value`` the corpus was ingested with.
    ``None`` reads the chain config's ``index_value``, else ``default``
    (``resolve_namespace``), the namespace ingestion with that config wrote.
    Set ``conversation_id`` and ``use_memory`` to continue a conversation;
    with ``use_memory`` and no id, a new id is generated and returned on
    :class:`RAGOutput`. A stateless caller passes
    the conversation itself as ``history`` instead (docs/design.md §7.2).
    An unknown key is refused, not ignored (MCP's ``strategy`` is
    ``search_strategy`` here).
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        description="The user's search query (leading and trailing whitespace "
        "removed; an empty one is refused)"
    )
    suffix: str | None = Field(
        default=None,
        description="Suffix for multi-tenant or versioned indices; None reads "
        "the config's processing.document_parsing.index_value, else 'default'",
    )
    enable_thinking: bool = Field(
        default=False,
        description="Enable thinking mode for language model reasoning and step-by-step problem solving",
    )
    search_strategy: SearchStrategy = Field(
        default=SearchStrategy.AUTO,
        description=(
            "The search strategy: GraphRAG (auto, drift, global, local, simple), "
            "LightRAG dual-level keyword (mix, hybrid, naive), HippoRAG "
            "personalized PageRank (ppr), RAPTOR summary tree (tree), "
            "bounded follow-up rounds over local or mix (iterative) or a "
            "checked read-only graph query (structured)"
        ),
    )
    search_type: SearchType = Field(
        default=SearchType.HYBRID,
        description="The type of search to perform (hybrid, lexical, vector)",
    )
    top_k: int = Field(
        default=10, ge=1, description="Maximum number of results to retrieve"
    )
    retrieval_multiplier: int = Field(
        default=1,
        ge=1,
        description="Multiplier for retrieval operations to increase search depth",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Retrieval context budget in tokens for this query, capped "
        "at search.token_manager.max_context_tokens (the answer's output cap is "
        "models.default_max_output_tokens)",
    )
    conversation_id: str | None = Field(
        default=None, description="Unique identifier for the conversation session"
    )
    user_id: str | None = Field(
        default=None,
        description="Optional end-user id, recorded only on trace spans "
        "(user.id) when observability is enabled; pass an opaque id, not an "
        "e-mail address or name",
    )
    use_memory: bool = Field(
        default=False, description="Whether to use conversation memory"
    )
    history: list[ConversationTurn] | None = Field(
        default=None,
        description="The conversation so far, oldest turn first, for a caller "
        "that keeps it itself: the follow-up rewrite reads it where it reads "
        "memory otherwise. Bounded to memory.max_messages_per_conversation "
        "turns and memory.max_history_tokens, and screened by the input "
        "guardrail as the query is. Not with use_memory",
    )
    enable_query_processing: bool = Field(
        default=True,
        description="Whether to enable query processing (translation, entity "
        "and keyword extraction). The follow-up rewrite is not part of it: it "
        "runs whenever there is conversation history (use_memory's or the "
        "caller's history).",
    )
    target_language: str | None = Field(
        default=None, description="Target language for translation"
    )
    filters: dict[str, Any] | None = Field(
        default=None, description="Additional filters to apply to the search"
    )

    @field_validator("suffix")
    @classmethod
    def _safe_suffix(cls, suffix: str | None) -> str | None:
        # It names store indexes: a wildcard or comma would widen the target.
        return None if suffix is None else validate_safe_name(suffix, "suffix")

    @field_validator("query")
    @classmethod
    def _compose_query(cls, query: str) -> str:
        # Indexed text is NFC (see convert_langchain_to_document); a query typed
        # or pasted as decomposed Hangul (NFD) must compose to match it.
        query = strip_invisible(unicodedata.normalize("NFC", query))
        if not query:
            # Refused before any model call: an empty question (or one of
            # invisible characters alone, "\u200b") was routed, extracted
            # from and searched for.
            raise ValueError("The query is empty.")
        return query

    @model_validator(mode="after")
    def _one_conversation_source(self) -> "RAGInput":
        if self.history is not None and self.use_memory:
            raise ValueError(
                "Pass the conversation as history or keep it in memory "
                "(use_memory), not both."
            )
        return self

    @classmethod
    def checked(cls, **fields: Any) -> "RAGInput":
        """A query from a surface's arguments. One it refuses (an empty
        query, a value out of range) raises ``InvalidArgumentError``, an
        ``invalid_argument`` (``ErrorClass``), naming each field and why,
        rather than a ``ValidationError``.

        Raises:
            InvalidArgumentError: A field the query cannot take.
        """
        try:
            return cls(**fields)
        except ValidationError as e:
            raise InvalidArgumentError(validation_message(e)) from None


def validation_message(error: ValidationError) -> str:
    """``error`` as one line: each refused field and why
    (``query: The query is empty.``)."""
    return "; ".join(
        f"{'.'.join(str(p) for p in item.get('loc', ())) or 'input'}: "
        f"{str(item.get('msg', '')).removeprefix('Value error, ')}"
        for item in error.errors()
    )


class RAGOutput(BaseModel):
    """The answer of a RAG-mode ``invoke``/``ainvoke``.

    ``sources`` lists the documents and graph elements the answer's context
    was built from; ``search_results`` holds the raw retrieval results.
    ``metadata["error"]`` is ``True`` on the degraded answer returned for a
    failed query when ``processing.ignore_errors`` is set, with the error's
    class name in ``metadata["error_type"]``.
    ``metadata["usage"]`` counts the query's model calls (``llm_calls``,
    ``input_tokens``, ``output_tokens``, ``cache_hits``, and the same
    ``by_role`` and ``by_model``; ``domain.models.LLMUsage``). With a guardrail
    configured, ``metadata["guardrail"]`` lists its interventions (source,
    action, policies), and ``metadata["blocked"]`` is ``True`` when it
    blocked the query (the answer is then its blocked message and nothing
    was retrieved). A blocked answer is replaced by the blocked message and
    returns no ``sources`` or ``search_results.results``; a masked answer
    keeps its sources as retrieved (masking covers the answer only).
    ``metadata["abstained"]`` is ``True`` on a declined answer, with
    ``metadata["abstention_reason"]``: ``no_context`` (the fixed reply to an
    empty context; ``no_reports`` when it is global search's on a namespace
    that holds no community report), ``marker`` (the answer started with
    ``[NO_ANSWER]``, ``search.abstention.mode: mark``) or
    ``insufficient_context`` (the strategy flagged its context insufficient under
    ``search.abstention.on_insufficient: abstain``); a declined answer keeps
    the sources the model was given.
    ``citations`` (``search.citations``) lists each numbered section of the
    answer context, ``[n]`` in the answer, with the documents, pages and text
    units it rests on and whether the answer cites it; a number the answer
    uses that the context did not show is listed in
    ``metadata["unknown_citation_markers"]``. A blocked answer has none, and
    so has every answer of a strategy whose citations are off
    (``search.citations.strategies``: ``global`` and ``drift`` by default;
    ``auto`` follows its route, ``iterative`` its base): its context is not
    numbered and its answer carries no ``[n]`` markers.
    An ``auto`` query's ``metadata["routing"]`` says how its strategy was
    chosen (``ROUTING_KEY``).
    """

    answer: str = Field(description="The generated answer to the user's query")
    sources: list[Source] = Field(
        description="The sources the answer model was given: the text it saw "
        "and the stored item's metadata with its provenance"
    )
    search_results: SearchResult = Field(
        description="Detailed search results from the retrieval process"
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="The numbered sections of the answer context ([n]) and "
        "what each rests on (search.citations); empty when the strategy's "
        "citations are off (global and drift by default)",
    )
    conversation_id: str | None = Field(
        description="The conversation session identifier"
    )
    processed_query: ProcessedQuery = Field(
        description="Information about how the query was processed"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the RAG process"
    )


class SearchOutput(BaseModel):
    """The result of a retrieval-only ``search``/``asearch``: the fused,
    ranked results without an answer.

    ``metadata`` carries the same keys as :attr:`RAGOutput.metadata`
    (``processing_time``, ``error``, ``blocked``, ``guardrail``) plus the
    retrieval's own (``search_strategy``, ``total_results``, ``rerank``, ...).
    """

    search_results: SearchResult = Field(
        description="The fused, ranked retrieval results"
    )
    processed_query: ProcessedQuery = Field(
        description="Information about how the query was processed"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the search"
    )


def _strategy_dimension(value: Any, default: str = "auto") -> str:
    """The ``Strategy`` value of a query's metrics: ``value``'s
    ``SearchStrategy`` (``default`` when it names none), or ``invalid`` for
    a value that is no strategy (a refused input's), so a caller's string
    never opens a new billed series."""
    if value is None or value == "":
        return default
    try:
        return SearchStrategy(getattr(value, "value", value)).value
    except (TypeError, ValueError):
        return "invalid"


@dataclass
class _StreamOutcome:
    """What a streamed query's metrics and failure read (``astream``): its
    id, its trace's, the input once admitted, the class of the failure its
    error answer absorbed (``processing.ignore_errors``), its usage meter
    and its answer's metadata (degraded steps, abstention), as ``ainvoke``'s
    ``RAGOutput`` carries them."""

    query_id: str
    trace_id: str | None = None
    rag_input: RAGInput | None = None
    error_class: ErrorClass | None = None
    # The strategy that ran (auto's route), once retrieval did.
    ran: Any = None
    # The query's model calls and requests, once retrieval started.
    meter: UsageMeter | None = None
    # The streamed answer's metadata, once it was formatted.
    metadata: dict[str, Any] = field(default_factory=dict)


class GraphRAGChain(Runnable[RAGInput, RAGOutput]):
    """Query a knowledge graph built by the ingestion pipeline.

    One chain serves many queries, including concurrent ones:

    - ``invoke``/``ainvoke`` return a :class:`RAGOutput` with the generated
      answer and its sources; ``stream``/``astream`` yield the answer text.
    - ``search``/``asearch`` run the same retrieval without generating an
      answer and return a :class:`SearchOutput`.

    Lifecycle: the chain opens Neptune and OpenSearch connections on first use
    and keeps them for later queries. Call :meth:`close` (sync code) or
    ``await`` :meth:`aclose` (async code) when done; otherwise the connections
    stay open until garbage collection. A closed chain reconnects if used
    again.

    Threads and event loops: the async methods run on the caller's event
    loop, with retrievers built once per loop. The sync methods (``invoke``,
    ``batch``, ``stream``, ``search``) run on one event loop the chain owns on
    a daemon thread, so they also work from a thread that already runs an
    event loop; they must not be called from a coroutine running on that
    chain-owned loop (for example a callback), which raises ``RuntimeError``.

    Backends are injectable: ``providers`` (or the ``model_factory``
    shorthand) for models, ``retriever_builders`` for the graph and document
    stores, ``memory_manager`` for conversation memory, and ``guard`` for the
    screening of queries and answers (default: the Bedrock guardrail of
    ``aws.bedrock.guardrail`` when an identifier is set).
    """

    def __init__(
        self,
        config: Config,
        *,
        model_factory: LLMFactoryPort | None = None,
        providers: Providers | None = None,
        memory_manager: MemoryManager | None = None,
        retriever_builders: (
            dict[RetrieverRole, Callable[[], BaseGraphRAGRetriever]] | None
        ) = None,
        guard: ContentGuardPort | None = None,
        allowed_strategies: Collection[SearchStrategy] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        # The strategies a query may run, when a serving surface limits them
        # (the MCP server's mcp.strategies): auto routes only among them and
        # another is refused when the strategy resolves (_resolve_strategy).
        self.allowed_strategies: frozenset[SearchStrategy] | None = (
            frozenset(allowed_strategies) if allowed_strategies is not None else None
        )
        try:
            config.aws.bedrock.guardrail.check_contextual_grounding()
        except ValueError as e:
            raise ConfigurationError(f"aws.bedrock.guardrail: {e}") from e
        # Before any model is built, so the factory attaches the tracing
        # callback (no-op unless observability.enabled).
        configure_observability(config.observability)
        if providers is not None and (
            model_factory is not None or retriever_builders is not None
        ):
            raise ValueError(
                "Pass model_factory and retriever_builders either to GraphRAGChain "
                "or on the Providers bundle (llm_factory=, retriever_builders=), "
                "not both."
            )
        # Composition root (hexagonal): ONE provider bundle — session plus
        # LLM/embedding/rerank/token-counter factories — built here (or
        # injected) and passed explicitly to every component the chain builds
        # (strategies, scorer, token manager, memory, retrievers). An injected
        # factory, e.g. a local Ollama-backed LLMFactoryPort, therefore
        # reaches all of them; anything not injected is what config.providers
        # selects (Bedrock by default).
        self.providers = providers or build_providers(
            config,
            llm_factory=model_factory,
            retriever_builders=retriever_builders,
        )
        self.ignore_errors = self.config.processing.ignore_errors
        # Screens the raw query before anything reads it and the final answer
        # after generation; the model calls in between run unguarded (see
        # GuardrailConfig).
        self.guard: ContentGuardPort | None = guard or build_query_guard(
            config, self.providers
        )
        guardrail = self.config.aws.bedrock.guardrail
        self._screen_inputs = self.guard is not None and guardrail.check_input
        self._screen_outputs = self.guard is not None and guardrail.check_output
        # The answer's check also carries what the answer model was given.
        self._ground_outputs = (
            self._screen_outputs and guardrail.contextual_grounding.enabled
        )
        # Conversation memory belongs to the chain unless a MemoryManager is
        # injected (to share histories between chains, e.g. one per request).
        # Within it a conversation is keyed by namespace, user and id.
        self.memory_manager: MemoryManager = memory_manager or MemoryManager(config)
        self.token_manager = TokenManager(self.config, providers=self.providers)
        self.factory: LLMFactoryPort = self.providers.llm_factory
        # Retrievers and strategy instances are built once per event loop and
        # reused across queries: retrievers hold loop-bound async clients, and a
        # strategy holds its retrievers. Entries are keyed by the loop object
        # (not its id, which a new loop can reuse), so queries on different
        # loops at the same time each keep their own clients; a loop's entries
        # are released once that loop is closed, and the rest on chain close.
        # Strategies keep no per-query state on the instance (see
        # BaseSearchStrategy), so concurrent queries on one loop share one.
        # The lock guards both dicts: the sync API's loop thread and the
        # caller's threads mutate them concurrently.
        self._retriever_cache: dict[
            tuple[RetrieverRole, asyncio.AbstractEventLoop | None],
            BaseGraphRAGRetriever,
        ] = {}
        self._strategy_cache: dict[
            tuple[SearchStrategy, asyncio.AbstractEventLoop | None],
            BaseSearchStrategy,
        ] = {}
        self._cache_lock = threading.RLock()
        # What each queried suffix's index namespace holds, with when it was
        # read (methodology_check; guarded by the cache lock).
        self._held_artifacts: dict[str, tuple[float, NamespaceHoldings]] = {}
        # The namespaces a query found ingested (_arequire_namespace).
        self._present_namespaces: set[str] = set()
        # The schema annotation fields the configured extraction schema
        # writes (filter_validation._schema_annotation_fields), loaded on the
        # first filter on one of them; None until then, and for good once a
        # load failed.
        self._schema_fields: frozenset[str] | None = None
        self._schema_fields_unreadable = False
        # Prompt chains (prompt | llm | parser) by prompt, model, parser type
        # and model kwargs; not loop-bound, so shared by every loop.
        self._prompt_chains: dict[tuple[Any, ...], Runnable[Any, Any]] = {}
        # Query entity and keyword extractions by question, language, model
        # and prompt (search.query_extraction_cache_size): repeats and the
        # arms of a comparison reuse them.
        self._extractions = _ExtractionCache(
            self.config.search.query_extraction_cache_size
        )
        # The sync entry points (invoke/batch/stream) run on this one
        # long-lived loop instead of a fresh loop per call, so their
        # loop-bound retrievers are built once and reused.
        self._loop_runner = _LoopRunner(self.config.processing.io_workers)
        # This chain's own retriever closes running on daemon threads: chain
        # close waits for these only, never for another chain's.
        self._background_closes = _BackgroundCloses()
        # The speculative extractions queries left running (keep_in_background).
        self._background_extractions = _BackgroundExtractions()
        # Safety net for a chain dropped without close()/aclose(): release the
        # cached retrievers' sockets, then stop the loop. The finalizer holds
        # the caches and runner, never the chain itself, so it does not keep
        # the chain alive. Collection timing is up to the GC, so callers must
        # still close the chain when done.
        weakref.finalize(
            self,
            _release_cached_resources,
            self._retriever_cache,
            self._strategy_cache,
            self._cache_lock,
            self._loop_runner,
            self._background_closes,
        )
        self.chain = self._build_chain()

    def _build_chain(self) -> Runnable[Any, Any]:
        # Memory is loaded first: a follow-up is rewritten as a standalone
        # question from the history, and routing, query processing and
        # retrieval all work on that question.
        base_chain: Runnable[Any, Any] = (
            RunnableLambda(self._load_memory_step)
            | RunnablePassthrough.assign(standalone_query=self._rewrite_query_step)
            | RunnableLambda(self._route_and_process_step)
            | RunnablePassthrough.assign(search_results=self._search_step)
        )

        context_step: Runnable[Any, Any] = RunnablePassthrough.assign(
            optimized_context=self._context_optimization_step
        ) | RunnablePassthrough.assign(context=self._context_building_step)
        # Streaming runs everything up to (and including) context building as
        # one non-streamed pass, then streams only the answer LLM. Reusing the
        # same step objects keeps stream and invoke behaviour in lockstep.
        self._stream_context_chain: Runnable[Any, Any] = base_chain | context_step

        answer_step: Runnable[Any, Any] = RunnablePassthrough.assign(
            answer=self._answer_generation_step
        )
        # Detects a declined answer (search.abstention) before the output
        # guardrail and the reported output see it.
        answer_step = answer_step | RunnableLambda(self._abstention_step)
        if self._ground_outputs:
            # Keeps the context the answer was generated from for the
            # answer's grounding check; wired only when that check is on.
            answer_step = answer_step | RunnableLambda(self._capture_grounding_step)
        self._answer_step = answer_step
        rag_branch = (
            context_step
            | RunnableLambda(self._answer_with_citations_step)
            | RunnableLambda(self._format_output_step)
        )

        self._search_chain: Runnable[Any, Any] = _Ledgered(
            base_chain | RunnableLambda(self._format_search_output_step)
        )
        return _Ledgered(base_chain | rag_branch)

    async def _route_and_process_step(
        self, state: dict[str, Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """Resolve the strategy (auto routing), then process the query.

        One step, so the extractions an auto query starts alongside its
        router call (``search.auto_speculative_extraction``) never outlive
        their query's deadline. One the resolved strategy does not read is
        never waited for: the step hands it to the query entry point
        (``_LEFTOVER_EXTRACTIONS``), which returns without waiting for it
        and leaves it running in the background (``_BackgroundExtractions``),
        so its result still fills the extraction cache for a later query.
        Its request is sent and billed either way; its call counts in the
        query's usage from its start (``shared.usage``). It is cancelled
        when the query fails, at the query's deadline, or when the chain
        closes.
        """
        processing = state.get("enable_query_processing", True)
        prefetch: dict[str, asyncio.Future[Any]] | None = (
            {}
            if processing and self.config.search.auto_speculative_extraction
            else None
        )
        try:
            state = await self._resolve_strategy(state, config, prefetch=prefetch)
            state["processed_query"] = (
                await self._process_query_step(state, config, prefetched=prefetch)
                if processing
                else self._simple_query(state)
            )
        except BaseException:
            if prefetch:
                await discard(list(prefetch.values()))
            raise
        if prefetch:
            leftovers = _LEFTOVER_EXTRACTIONS.get()
            if leftovers is None:
                # No entry point keeps them: wait here.
                await settle(list(prefetch.values()), None)
            else:
                leftovers.extend(prefetch.values())
        return state

    def _simple_query(self, inputs: dict[str, Any]) -> ProcessedQuery:
        """The query as is (``enable_query_processing`` off): no LLM call."""
        return ProcessedQuery(
            original_query=inputs["query"],
            standalone_query=inputs.get("standalone_query"),
            final_query=self._question(inputs),
        )

    async def _resolve_strategy(
        self,
        state: dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        prefetch: dict[str, asyncio.Future[Any]] | None = None,
    ) -> dict[str, Any]:
        """Set ``resolved_strategy``: the requested one, or auto's route.

        Given ``prefetch``, an auto query that calls the router first starts
        the query processing calls the routable strategies may read (see
        :meth:`_start_query_inputs`) into it, to run alongside the router.
        The caller owns the tasks (awaits or discards them).
        """
        strategy = state.get("search_strategy", SearchStrategy.AUTO)

        if strategy != SearchStrategy.AUTO:
            self._check_allowed(strategy)
            state["resolved_strategy"] = strategy
            return state

        # Before routing, as an interactive session does: the router picks
        # only among the strategies that apply every filter key, so whether
        # a key is refused never depends on the route a question takes.
        routable = self._allowed_routes(
            filter_validation.auto_candidates(
                self,
                state.get("filters"),
                # What auto routes among; each candidate's filter keys are
                # checked against these indexes.
                state.get("held_artifacts"),
            ),
            state.get("suffix"),
        )
        candidates = [s.value for s in routable]
        if len(routable) == 1:
            # Nothing to choose between: no router call.
            state["resolved_strategy"] = routable[0]
            state[ROUTING_KEY] = {
                "decided_by": "only_candidate",
                "candidates": candidates,
            }
            return state
        if prefetch is not None:
            prefetch.update(
                self._start_query_inputs(
                    state,
                    # A wrapper reads what it wraps (iterative: its base).
                    entities=any(
                        QueryInput.ENTITIES in query_inputs(s, self.config)
                        for s in routable
                    ),
                    keywords=any(
                        QueryInput.DUAL_KEYWORDS in query_inputs(s, self.config)
                        for s in routable
                    ),
                    config=config,
                )
            )
        try:
            router = self._get_chain_for_prompt(
                StrategySelectionPrompt,
                StrOutputParser(),
                output_ceiling=self.config.search.output_ceilings.strategy_selection,
            )
            query = self._question(state)
            selected_strategy_str = await router.ainvoke(
                {
                    "query": query,
                    "strategies": ", ".join(s.value for s in routable),
                    "strategy_descriptions": describe_routable_strategies(routable),
                },
                config,
            )
            strategy, named = await self._routed_strategy(
                str(selected_strategy_str), routable
            )
            decided_by = "router" if named else "fallback"
        except Exception as e:
            if not self.ignore_errors or is_unreachable_error(e):
                raise
            strategy = self._fallback_strategy(routable)
            decided_by = "fallback"
            record_degradation(
                DegradationStep.ROUTING, f"routed to '{strategy.value}'", e
            )
            log_failure(
                logger,
                logging.WARNING,
                "Strategy auto-selection failed, using '%s' as fallback: %s",
                strategy.value,
                error=e,
            )

        state["resolved_strategy"] = strategy
        state[ROUTING_KEY] = {"decided_by": decided_by, "candidates": candidates}
        return state

    def _check_allowed(self, strategy: SearchStrategy) -> None:
        """Refuse ``strategy`` outside ``allowed_strategies``, before any
        model call of the query."""
        allowed = getattr(self, "allowed_strategies", None)
        if allowed is not None and strategy not in allowed:
            raise InvalidArgumentError(
                f"Search strategy '{strategy.value}' is not allowed here; "
                f"choose one of {_names(allowed)}."
            )

    def _allowed_routes(
        self, routable: tuple[SearchStrategy, ...], suffix: str | None
    ) -> tuple[SearchStrategy, ...]:
        """Of ``routable`` (auto's candidates, the routable fallback
        included), those in ``allowed_strategies``: the router picks only
        among them and the routing fallback takes the first, so auto never
        runs another. Raises ``MethodologyNotIndexedError``, before the
        router call, when none is left."""
        allowed = getattr(self, "allowed_strategies", None)
        if allowed is None:
            return routable
        kept = tuple(s for s in routable if s in allowed)
        if kept:
            return kept
        namespace = resolve_namespace(self.config, suffix)
        raise MethodologyNotIndexedError(
            f"auto may run only {_names(allowed)}, and none of them serves "
            f"the index namespace '{namespace}' (with these filters auto "
            f"could route to {_names(routable)}); pick one it serves, or "
            "ingest the methodology an allowed strategy reads."
        )

    async def _routed_strategy(
        self, raw: str, routable: tuple[SearchStrategy, ...]
    ) -> tuple[SearchStrategy, bool]:
        """The strategy the router's answer ``raw`` names, and whether it
        named a routable one (else the fallback, ``_fallback_strategy``).

        The answer is the strategy name alone (``StrategySelectionPrompt``:
        a JSON answer made the router write about three times the output
        tokens, docs/concepts/defaults-and-evidence.md), read from a line
        that is the name alone or a short label and the name
        (``named_choice``: a reasoning block, markdown, punctuation and
        case aside, so "**Local**", "Strategy: local" or a preamble line
        before the name cost no repair call); any other answer is read as
        ``{"strategy": "<name>"}``, as every JSON answer is (one repair when
        it does not parse). Prose is not guessed from, though a strategy
        name is one of its words ("a simple lookup", "not global").
        """
        name = named_choice(raw, _STRATEGY_NAMES) or ""
        if not name:
            try:
                answer = await self._json_parser(
                    StrategySelectionPrompt, schema=_RouteAnswer
                ).aparse(raw)
                name = str(answer.strategy).strip().casefold()
            except OutputParserException as e:
                name = ""
                record_degradation(
                    DegradationStep.ROUTING, "router answer unreadable", e
                )
        by_value = {strategy.value: strategy for strategy in routable}
        if (strategy := by_value.get(name)) is not None:
            return strategy, True
        fallback = GraphRAGChain._fallback_strategy(routable)
        logger.warning(
            "Router returned no routable strategy (%s); defaulting to '%s'",
            text_digest(raw),
            fallback.value,
        )
        logger.debug("Unrecognized router output: '%s'", raw)
        return fallback, False

    @staticmethod
    def _fallback_strategy(routable: tuple[SearchStrategy, ...]) -> SearchStrategy:
        """The strategy AUTO uses when routing fails: the first routable
        general-purpose strategy (``GENERAL_FALLBACKS``: local, then mix),
        else the first routable one, so a configuration that excludes them
        (their stores may not be provisioned) is never routed to them."""
        return next(
            (s for s in GENERAL_FALLBACKS if s in routable),
            routable[0] if routable else GENERAL_FALLBACKS[0],
        )

    def _get_chain_for_prompt(
        self,
        prompt_class: type[BasePrompt],
        parser: BaseOutputParser[Any],
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        role = prompt_class.model_role
        if role is None:
            raise ValueError(
                f"No model role declared for prompt {prompt_class.__name__}; "
                "set its model_role."
            )
        model_id = self.config.model_id(role)
        # Built once per chain: setup_chain builds the model and its boto
        # clients, so building per query paid that (and fresh connection
        # pools) on every request. Parsers are stateless; one of each type.
        key = (prompt_class, model_id, type(parser), tuple(sorted(kwargs.items())))
        with self._cache_lock:
            if (cached := self._prompt_chains.get(key)) is None:
                cached = self._prompt_chains[key] = setup_chain(
                    factory=self.factory,
                    model_id=model_id,
                    prompt_class=prompt_class,
                    parser=parser,
                    custom_prompts=self.config.custom_prompts,
                    **kwargs,
                )
            return cached

    def _json_parser(
        self,
        prompt_class: type[BasePrompt],
        *,
        answer_keys: tuple[str, ...] = (),
        schema: type[BaseModel] | None = None,
        instructions: str = "",
    ) -> FixingJsonOutputParser:
        """The JSON reader of ``prompt_class``'s answers
        (``create_json_output_parser``: every family's JSON, then one repair
        by ``models.roles.fixing``), built once per chain. Its fixing model
        is built on the first answer that does not parse."""
        key = ("json", prompt_class, answer_keys, schema, instructions)
        cache = getattr(self, "_prompt_chains", None)
        with getattr(self, "_cache_lock", None) or threading.RLock():
            if cache is not None and (cached := cache.get(key)) is not None:
                return cast(FixingJsonOutputParser, cached)
            parser = create_json_output_parser(
                getattr(self, "factory", None),  # type: ignore[arg-type]
                self.config.model_id(ModelRole.FIXING),
                answer_keys=answer_keys,
                schema=schema,
                instructions=instructions,
            )
            if cache is not None:
                cache[key] = cast(Runnable[Any, Any], parser)
            return parser

    def _start_query_inputs(
        self,
        inputs: dict[str, Any],
        *,
        entities: bool,
        keywords: bool,
        translation: bool = True,
        config: RunnableConfig | None,
    ) -> dict[str, asyncio.Future[Any]]:
        """``query_processing.start_query_inputs``."""
        return query_processing.start_query_inputs(
            self,
            inputs,
            entities=entities,
            keywords=keywords,
            translation=translation,
            config=config,
        )

    async def _translate_query(
        self, query: str, target_language: str, config: RunnableConfig | None
    ) -> str | None:
        """``query_processing.translate_query``."""
        return await query_processing.translate_query(
            self, query, target_language, config
        )

    async def _process_query_step(
        self,
        inputs: dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        prefetched: dict[str, asyncio.Future[Any]] | None = None,
    ) -> ProcessedQuery:
        """Translate the query and extract what the resolved strategy reads
        (``query_processing.process_query``); under ``ignore_errors`` a
        failure leaves the question as is.

        A call already started for this query (``prefetched``, auto's
        speculative extractions) is awaited instead of started again; it is
        taken out of ``prefetched``.
        """
        original_query = inputs.get("query", "")
        standalone_query = inputs.get("standalone_query")
        question = standalone_query or original_query
        prefetched = {} if prefetched is None else prefetched

        try:
            return ProcessedQuery(
                original_query=original_query,
                standalone_query=standalone_query,
                **await query_processing.process_query(
                    self, inputs, question, config, prefetched
                ),
            )
        except Exception as e:
            if not self.ignore_errors or is_unreachable_error(e):
                raise
            log_failure(logger, logging.WARNING, "Query processing failed: %s", error=e)
            record_degradation(
                DegradationStep.QUERY_PROCESSING, "searched the question as is", e
            )
            return ProcessedQuery(
                original_query=original_query,
                standalone_query=standalone_query,
                final_query=question,
            )

    async def _extract_query_entities(
        self,
        query: str,
        target_language: str,
        config: RunnableConfig | None,
        *,
        namespace: str | None = None,
    ) -> list[str]:
        """``query_processing.extract_query_entities``."""
        return await query_processing.extract_query_entities(
            self, query, target_language, config, namespace=namespace
        )

    def _extraction_key(
        self,
        prompt_class: type[BasePrompt],
        query: str,
        target_language: str,
        *,
        namespace: str | None = None,
    ) -> tuple[str, ...]:
        """``query_processing.extraction_key``."""
        return query_processing.extraction_key(
            self, prompt_class, query, target_language, namespace=namespace
        )

    def _cached_extraction(self, key: tuple[str, ...]) -> Any:
        """``query_processing.cached_extraction``."""
        return query_processing.cached_extraction(self, key)

    def _cache_extraction(self, key: tuple[str, ...], value: Any) -> None:
        """``query_processing.cache_extraction``."""
        query_processing.cache_extraction(self, key, value)

    async def _extract_dual_keywords(
        self,
        query: str,
        target_language: str,
        config: RunnableConfig | None = None,
        *,
        namespace: str | None = None,
    ) -> tuple[list[str], list[str]]:
        """``query_processing.extract_dual_keywords``."""
        return await query_processing.extract_dual_keywords(
            self, query, target_language, config, namespace=namespace
        )

    async def _load_memory_step(self, state: dict[str, Any]) -> dict[str, Any]:
        turns = state.get("history")
        if isinstance(turns, list):
            # The caller's own conversation (RAGInput.history), bounded and
            # screened at the query's edge (_admit_history).
            state["history"] = history_buffer(
                [
                    (
                        HumanMessage(content=t["content"])
                        if t["role"] == "user"
                        else AIMessage(content=t["content"])
                    )
                    for t in turns
                ],
                max_answer_chars=HISTORY_ANSWER_CHARS,
            )
            return state
        if not state.get("use_memory") or not (cid := state.get("conversation_id")):
            state["history"] = ""
            return state

        state["history"] = await self.memory_manager.load_history(
            _conversation_key(state.get("suffix"), state.get("user_id"), cid)
        )
        return state

    async def _rewrite_query_step(
        self, state: dict[str, Any], config: RunnableConfig | None = None
    ) -> str | None:
        """The follow-up as a standalone question, or None without history.

        A follow-up such as "what about its price?" retrieves nothing useful on
        its own. One LLM call resolves its references from the conversation
        history, and routing, query processing, retrieval and the answer then
        all read the standalone question.
        """
        history = state.get("history")
        if not history:
            return None
        try:
            rewriter = self._get_chain_for_prompt(QueryRewritePrompt, StrOutputParser())
            rewritten = await rewriter.ainvoke(
                {"query": state["query"], "conversation_history": history}, config
            )
        except Exception as e:
            if not self.ignore_errors or is_unreachable_error(e):
                raise
            log_failure(
                logger,
                logging.WARNING,
                "Follow-up rewrite failed; using the query as is: %s",
                error=e,
            )
            record_degradation(
                DegradationStep.FOLLOW_UP_REWRITE, "searched the follow-up as is", e
            )
            return None
        return await query_processing.read_rewrite(self, str(rewritten))

    @staticmethod
    def _question(state: dict[str, Any]) -> str:
        """The question to answer: the standalone rewrite, else the query."""
        return str(state.get("standalone_query") or state.get("query", ""))

    async def _search_step(
        self, state: dict[str, Any], config: RunnableConfig | None = None
    ) -> SearchResult:
        # A single GraphRAGChain is reused for concurrent invocations (the
        # evaluation path runs its queries concurrently). Keep the resolved
        # strategy in a LOCAL — storing it on `self` lets a concurrent
        # `_search_step` overwrite it between assignment and `await`, executing
        # one query against another query's strategy (silent cross-contamination).
        # The instance itself is cached per loop and holds no per-query state.
        strategy_instance = self._get_strategy_instance(state["resolved_strategy"])
        # A backstop: every entry point (ainvoke/asearch/astream, auto's
        # routing) has checked these keys before any model call already; this
        # catches a caller driving the composed runnable directly.
        if state.get("filters"):
            filter_validation.check_strategy_filter_keys(
                self, state["filters"], state["resolved_strategy"]
            )
        processed: ProcessedQuery = state["processed_query"]
        # Ordered dedupe (not set()) so the joined entity query, and hence its
        # embedding, is reproducible across processes. Only a strategy that
        # reads an entity focus gets one. Earlier turns reach it through the
        # standalone rewrite, which names what a follow-up refers to.
        entity_focus = (
            list(dict.fromkeys(processed.entities))
            if query_processing.needs_query_entities(state, self.config)
            else []
        )

        resolved_strategy: SearchStrategy = state["resolved_strategy"]
        # What the namespace holds (read when the query was admitted): a
        # strategy reads only those of its optional indexes it holds.
        held = state.get("held_artifacts")
        if held is None:
            held = await methodology_check.namespace_artifacts(
                self, state.get("suffix")
            )
        search_query = SearchQuery(
            query=processed.final_query,
            search_type=state.get("search_type", SearchType.HYBRID),
            top_k=state.get("top_k", 10),
            retrieval_multiplier=state.get("retrieval_multiplier", 1),
            suffix=state.get("suffix"),
            max_tokens=state.get("max_tokens"),
            entity_focus=entity_focus or [],
            hl_keywords=processed.hl_keywords,
            ll_keywords=processed.ll_keywords,
            filters=state.get("filters"),
            # One strategy class may serve several modes (LightRAG mix/hybrid/
            # naive), so it reads the resolved mode from here, and the
            # language its own searches are written in (iterative's).
            metadata={
                "search_strategy": resolved_strategy.value,
                "query_language": query_processing.query_language(self, state),
                HELD_ARTIFACTS_KEY: sorted(a.value for a in held),
            },
        )
        with observability.span(
            "kg_rag.strategy",
            strategy=resolved_strategy.value,
            search_type=search_query.search_type,
            top_k=search_query.top_k,
            filtered=bool(search_query.filters),
        ) as span:
            # One memo of store lookups per query (attribute probes), never
            # kept past it (the query's, when the query opened one).
            with query_scope():
                ledger = current_ledger()
                result = await strategy_instance.asearch(search_query, config=config)
                degraded = DegradationReport.of(result.metadata).degraded_error
                if degraded and ledger is not None and not ledger.entries:
                    # A strategy that reports its own (not through the ledger).
                    ledger.record(DegradationStep.RETRIEVAL, "strategy", degraded)
            # A failure ignore_errors absorbed still degraded the answer.
            if degraded:
                span.set_attribute(ERROR_TYPE, degraded)
            if observability.is_enabled():
                span.set_attributes(
                    {
                        "kg_rag.results": len(result.results),
                        "kg_rag.total_results": result.total_results,
                    }
                )
                if rerank := result.metadata.get("rerank"):
                    span.set_attribute("kg_rag.rerank_status", rerank["status"])
            return result

    @staticmethod
    def check_filter_values(filters: dict[str, Any] | None) -> None:
        """Raise ``InvalidFilterError`` if a value in ``filters`` is one a
        search engine would refuse or misread (``None``, NaN, a nested list,
        an empty list or range...), as every query checks before any model
        call. Needs no chain: a caller validating a whole query file first
        (``run-eval``) uses it before building one. A string a boolean or
        numeric field cannot take is refused too (``typed_filter_values``)."""
        check_filter_values(typed_filter_values(filters))

    @staticmethod
    def typed_filter_values(filters: dict[str, Any] | None) -> dict[str, Any] | None:
        """``filters`` with the strings of boolean and numeric fields
        converted to their type (``in_schema: "false"`` to ``False``), as
        every query converts them before any model call, so each store
        matches them alike. For a caller holding string values (a command
        line); raises ``InvalidFilterError`` on a string the field's type
        cannot take."""
        return typed_filter_values(filters)

    async def acheck_methodology(
        self, strategy: SearchStrategy, suffix: str | None = None
    ) -> tuple[Methodology, ...]:
        """The methodologies the index namespace of ``suffix`` holds (its
        build record, else the configured ``indexing.methodologies``), as
        every query reads them before any model call. Raises
        ``IndexNotFoundError`` if nothing was ingested under ``suffix`` (it
        holds no methodology: the config's are not described as its), and
        ``MethodologyNotIndexedError`` if ``strategy`` reads another one
        (``auto`` routes only among the held ones). For callers that check a
        strategy once before many queries (an interactive session)."""
        await self._arequire_namespace(suffix)
        return await methodology_check.check_strategy_methodology(
            self, strategy, suffix
        )

    async def aread_build_record(self, suffix: str | None) -> dict[str, Any] | None:
        """The build record ingestion wrote on the index namespace of
        ``suffix`` (``resolve_namespace``; ``VectorIndexer.record_build``:
        what it built, under which settings), read through the document
        retriever the answers come from. ``None`` when it records none (built before build records, or a
        backend without index metadata); a store error raises."""
        return await self._get_retriever(RetrieverRole.DOCUMENT).aread_build_record(
            resolve_namespace(self.config, suffix)
        )

    def document_retriever(self) -> BaseGraphRAGRetriever:
        """The retriever bound to the document role (an injected
        ``retriever_builders`` backend included): the indexes the answers are
        retrieved from, for a caller that reads them directly, as
        ``run-eval`` resolves text-unit ids to their files. The chain owns
        it: ``close``/``aclose`` release it."""
        return self._get_retriever(RetrieverRole.DOCUMENT)

    def check_filter_keys(
        self, filters: dict[str, Any] | None, strategy: SearchStrategy
    ) -> None:
        """Raise ``InvalidFilterError`` if ``filters`` names a key no index
        ``strategy`` reads declares, as each of its queries would (for
        ``auto``: if no routable strategy applies every key, since ``auto``
        routes only among those that do), or a value is malformed
        (``check_filter_values``). For callers that keep filters across
        queries (an interactive session)."""
        filter_validation.check_filter_keys(self, filters, strategy)

    def _get_strategy_instance(
        self, strategy_type: SearchStrategy
    ) -> BaseSearchStrategy:
        cache_key = (strategy_type, self._cache_loop())
        with self._cache_lock:
            if (cached := self._strategy_cache.get(cache_key)) is not None:
                return cached

            spec = get_strategy_spec(strategy_type)

            # Inject retrievers keyed by abstract role ("graph"/"document"), so
            # the strategy never names a concrete backend.
            retrievers = {
                role.value: self._get_retriever(role) for role in spec.required_roles
            }

            strategy = spec.strategy_class(
                config=self.config, retrievers=retrievers, providers=self.providers
            )
            self._strategy_cache[cache_key] = strategy
            return strategy

    def _get_retriever(self, role: RetrieverRole) -> BaseGraphRAGRetriever:
        cache_key = (role, self._cache_loop())
        with self._cache_lock:
            if (cached := self._retriever_cache.get(cache_key)) is not None:
                return cached
            # Injected retriever builders (the backend seam) come first, so a
            # custom store is wired without subclassing or editing any
            # strategy; the rest are the stores config.providers selects.
            retriever = self.providers.build_retriever(role)
            self._retriever_cache[cache_key] = retriever
            return retriever

    def _cache_loop(self) -> asyncio.AbstractEventLoop | None:
        """Return the cache key for the running loop (``None`` outside one).

        Entries bound to loops that have since closed are released first (they
        can no longer serve a query, and their clients would otherwise leak
        until chain close). Entries of loops that are still open are left
        alone: another thread may have a query in flight on them.
        """
        try:
            current: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        with self._cache_lock:
            dead = {
                loop
                for _, loop in (*self._retriever_cache, *self._strategy_cache)
                if loop is not None and loop.is_closed()
            }
            evicted = self._take_cached_retrievers(dead) if dead else {}
        for retrievers in evicted.values():
            # The loop is closed, so this is their synchronous close.
            self._release_retrievers(
                retrievers, None, wait=False, background=self._background_closes
            )
        return current

    def _take_cached_retrievers(
        self, loops: Collection[asyncio.AbstractEventLoop | None] | None = None
    ) -> dict[asyncio.AbstractEventLoop | None, list[BaseGraphRAGRetriever]]:
        """Remove cached entries (of ``loops``, or all) grouped by their loop."""
        return _take_cached(
            getattr(self, "_retriever_cache", {}),
            getattr(self, "_strategy_cache", {}),
            getattr(self, "_cache_lock", None) or threading.RLock(),
            loops,
        )

    @staticmethod
    def _release_retrievers(
        retrievers: list[BaseGraphRAGRetriever],
        loop: asyncio.AbstractEventLoop | None,
        *,
        wait: bool,
        background: "_BackgroundCloses | None" = None,
    ) -> None:
        """Close retrievers from sync code (best-effort, never raises).

        Retrievers bound to a loop that is still running on another thread are
        closed there with their awaited ``aclose``; otherwise (no loop, or the
        loop is gone) their synchronous ``close`` releases what it can. Called
        with a loop running in this thread, the synchronous closes run on a
        worker thread: a client's blocking close may drive its own event loop
        (gremlinpython's transport calls ``run_until_complete``), which raises
        inside a running loop and would leave the connection open.

        ``wait=False`` (eviction on the query path) returns without waiting
        for the closes: they run on a daemon thread, so a slow close does not
        stall the query; chain close waits for them. ``background`` tracks
        those closes (the owning chain's tracker; a private one if omitted).
        """
        if not retrievers:
            return
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if (
            loop is not None
            and loop is not current
            and loop.is_running()
            and not loop.is_closed()
        ):
            future = asyncio.run_coroutine_threadsafe(
                _aclose_retrievers(retrievers), loop
            )
            if wait:
                try:
                    future.result(timeout=_TEARDOWN_TIMEOUT_SECONDS)
                except Exception as e:  # noqa: BLE001 - teardown must never raise
                    logger.debug("Error closing retrievers on their loop: %s", e)
            return
        if current is None and wait:
            _close_retrievers(retrievers)
            return
        worker = (background or _BackgroundCloses()).start(retrievers)
        if wait:
            worker.join(timeout=_TEARDOWN_TIMEOUT_SECONDS)

    async def aclose(self) -> None:
        """Close every cached retriever's backing client (best-effort).

        Each retriever build opens a Neptune websocket + thread pool and/or an
        OpenSearch (a)sync HTTP pool that otherwise survive until GC. Call this
        when the chain is done (e.g. from the CLI ``finally``) so a process that
        finishes a query releases its sockets; the GC finalizer only does so
        whenever the garbage collector reaches a dropped chain. Retrievers are
        closed on the
        loop they are bound to, then the chain's sync-API loop is stopped.
        Never raises.
        """
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        for loop, retrievers in self._take_cached_retrievers().items():
            if loop is None or loop is current:
                await _aclose_retrievers(retrievers)
            elif loop.is_running() and not loop.is_closed():
                try:
                    await asyncio.wait_for(
                        asyncio.wrap_future(
                            asyncio.run_coroutine_threadsafe(
                                _aclose_retrievers(retrievers), loop
                            )
                        ),
                        timeout=_TEARDOWN_TIMEOUT_SECONDS,
                    )
                except Exception as e:  # noqa: BLE001 - teardown must never raise
                    logger.debug("Error closing retrievers on their loop: %s", e)
            else:
                # Their loop is gone: the blocking closes run off this loop.
                await asyncio.to_thread(_close_retrievers, retrievers)
        try:
            background = getattr(self, "_background_closes", None)
            if background is not None:
                await asyncio.to_thread(background.join)
        except Exception as e:  # noqa: BLE001 - teardown must never raise
            logger.debug("Error waiting for background retriever closes: %s", e)
        finally:
            extractions = getattr(self, "_background_extractions", None)
            if extractions is not None:
                await extractions.acancel()
            runner = getattr(self, "_loop_runner", None)
            if runner is not None:
                await asyncio.to_thread(runner.stop)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def close(self) -> None:
        """Synchronous teardown of cached retrievers and the sync-API loop
        (``with GraphRAGChain(...) as chain:`` calls it; ``async with``
        calls :meth:`aclose`)."""
        extractions = getattr(self, "_background_extractions", None)
        if extractions is not None:
            extractions.cancel()
        _release_cached_resources(
            getattr(self, "_retriever_cache", {}),
            getattr(self, "_strategy_cache", {}),
            getattr(self, "_cache_lock", None) or threading.RLock(),
            getattr(self, "_loop_runner", None),
            getattr(self, "_background_closes", None),
        )

    def _optimize_context(self, state: dict[str, Any]) -> OptimizedContext:
        query: ProcessedQuery = state["processed_query"]
        search_results: SearchResult = state["search_results"]
        strategy = state.get("resolved_strategy")
        return self.token_manager.optimize_context(
            retrieval_results=search_results.results,
            query=query.final_query,
            max_tokens=state.get("max_tokens"),
            spare_budget_first=(
                get_strategy_spec(
                    wrapped_strategy(strategy, self.config)
                ).strategy_class.spare_context_budget_first
                if strategy is not None
                else ()
            ),
            # search.citations, per strategy: auto's route, iterative's base.
            number_sections=citations.citations_enabled(self.config, strategy),
        )

    def _context_optimization_step(self, state: dict[str, Any]) -> OptimizedContext:
        """Select the budgeted sections once and keep them in chain state.

        Both the context string and the reported ``sources`` are built from this
        selection, so sources reflect exactly what the answer model was given.
        """
        try:
            optimized = self._optimize_context(state)
        except Exception as e:
            if not self.ignore_errors:
                raise
            log_failure(
                logger,
                logging.WARNING,
                "Context optimization failed: %s",
                error=e,
                traceback=True,
            )
            record_degradation(
                DegradationStep.CONTEXT_OPTIMIZATION, "answered without context", e
            )
            return OptimizedContext(
                sections=[],
                total_tokens=0,
                sections_included=0,
                sections_excluded=len(state["search_results"].results),
            )
        if not optimized.sections and any(
            r.content for r in state["search_results"].results
        ):
            # Retrieved evidence no section of which fits the budget: the
            # decline that follows is the budget's, not "nothing found".
            record_degradation(DegradationStep.CONTEXT_OPTIMIZATION, "context_budget")
        return optimized

    def _context_building_step(self, state: dict[str, Any]) -> str:
        optimized: OptimizedContext | None = state.get("optimized_context")
        if optimized is None:
            optimized = self._optimize_context(state)
        return self.token_manager.build_context_string(optimized)

    def _answer_generation_step(self, state: dict[str, Any]) -> Runnable[Any, Any]:
        # Never ask the LLM to answer from an empty/placeholder context: retrieval
        # produced nothing, so generating anyway invites a confident hallucination
        # with no supporting sources. Short-circuit to an explicit "cannot answer".
        if self._context_is_empty(state):
            return RunnableLambda(lambda s: self._no_context_answer(s))

        enable_thinking = state.get("enable_thinking", False)
        # Only passed when set, so the default answer chain (and its cache
        # key) is that of earlier releases.
        instructions = self._answer_instructions(state)
        answer_chain = self._get_chain_for_prompt(
            self._answer_prompt(state.get("resolved_strategy")),
            StrOutputParser(),
            enable_thinking=enable_thinking,
            **({"system_suffix": instructions} if instructions else {}),
        )
        # The answer reads the standalone question: the conversation history
        # reaches it only through that rewrite.
        answer: Runnable[Any, Any] = (
            RunnableLambda(
                lambda s: {"query": self._question(s), "context": s.get("context")}
            )
            | answer_chain
        )
        sink = _ANSWER_CHUNKS.get()
        if sink is None or self._screen_outputs or self._reads_marker(state):
            # Nothing is shown before the output guardrail screened the
            # answer, nor before the [NO_ANSWER] token is ruled in or out.
            return answer
        return _streamed_to(answer, sink)

    @staticmethod
    def _context_insufficient(state: dict[str, Any]) -> bool:
        """Whether the strategy flagged its context insufficient
        (``SearchResult.metadata["context_sufficient"] is False``)."""
        results = state.get("search_results")
        metadata = getattr(results, "metadata", None) or {}
        return metadata.get(CONTEXT_SUFFICIENT_KEY) is False

    def _declines_insufficient(self, state: dict[str, Any]) -> bool:
        """Whether the answer must decline: the context is flagged
        insufficient under ``search.abstention.on_insufficient: abstain``."""
        return (
            self.config.search.abstention.on_insufficient == "abstain"
            and self._context_insufficient(state)
        )

    def _answer_instructions(self, state: dict[str, Any]) -> str:
        """The text appended to the answer's system prompt: the citation
        instructions of a numbered context (``search.citations``), the graph
        query result's instruction when the context holds one (the
        structured strategy), then the ``search.abstention`` text ("" with
        all off)."""
        parts = []
        optimized: OptimizedContext | None = state.get("optimized_context")
        if optimized is not None and any(
            s.marker is not None for s in optimized.sections
        ):
            parts.append(CITATION_INSTRUCTIONS)
        if self._declines_insufficient(state):
            return "".join([*parts, DECLINE_INSTRUCTIONS])
        if self._holds_query_result(state):
            parts.append(QUERY_RESULT_INSTRUCTIONS)
        if self.config.search.abstention.mode == "mark":
            parts.append(ABSTENTION_INSTRUCTIONS)
        if self._context_insufficient(state):
            parts.append(HEDGE_INSTRUCTIONS)
        return "".join(parts)

    @staticmethod
    def _holds_query_result(state: dict[str, Any]) -> bool:
        """Whether the answer's context holds a graph query result."""
        optimized: OptimizedContext | None = state.get("optimized_context")
        if optimized is not None:
            return any(
                s.section_type is SectionType.QUERY_RESULT for s in optimized.sections
            )
        results = getattr(state.get("search_results"), "results", None) or []
        return any(r.retriever_type == SectionType.QUERY_RESULT.value for r in results)

    def _reads_marker(self, state: dict[str, Any]) -> bool:
        """Whether this answer is checked for the [NO_ANSWER] token: the
        model was given a context and asked to use it."""
        return not self._context_is_empty(state) and (
            self.config.search.abstention.mode == "mark"
            or self._declines_insufficient(state)
        )

    @staticmethod
    def _context_is_empty(state: dict[str, Any]) -> bool:
        context = str(state.get("context") or "").strip()
        return not context or context == EMPTY_CONTEXT_PLACEHOLDER

    @staticmethod
    def _no_context_reason(state: dict[str, Any]) -> str:
        """Why the context is empty: the strategy's reason
        (``NO_CONTEXT_REASON_KEY``, ``no_reports``), else ``no_context``."""
        results = state.get("search_results")
        metadata = results.metadata if isinstance(results, SearchResult) else {}
        reason = metadata.get(NO_CONTEXT_REASON_KEY)
        return reason if isinstance(reason, str) and reason else REASON_NO_CONTEXT

    def _no_context_answer(self, state: dict[str, Any]) -> str:
        """The fixed decline, in the language of the question (the index
        language when the question does not tell), as the answer prompt
        writes in the query's language."""
        return no_context_answer(self._question(state), index_language(self.config))

    def _abstention_step(self, state: dict[str, Any]) -> dict[str, Any]:
        """Set ``state["abstention_reason"]`` (``None`` for an answer) and
        strip the [NO_ANSWER] token per ``search.abstention.strip_marker``."""
        if self._context_is_empty(state):
            state["abstention_reason"] = self._no_context_reason(state)
            return state
        marked = False
        if self._reads_marker(state):
            marked, rest = split_marker(str(state.get("answer") or ""))
            if marked and self.config.search.abstention.strip_marker:
                state["answer"] = rest or self._no_context_answer(state)
        if self._declines_insufficient(state):
            # Declined by policy, whatever the model wrote: an answer it gave
            # instead of the decline is not shown.
            if not marked:
                state["answer"] = self._no_context_answer(state)
            state["abstention_reason"] = REASON_INSUFFICIENT_CONTEXT
        else:
            state["abstention_reason"] = REASON_MARKER if marked else None
        return state

    def _answer_prompt(self, strategy: SearchStrategy | None) -> type[BasePrompt]:
        """The answer prompt of ``strategy`` (``search.answer_prompt_style``):
        the concise default, or the comprehensive one."""
        style = (
            self.config.search.answer_prompt_style.get(strategy)
            if strategy is not None
            else None
        )
        if style == "comprehensive":
            return ComprehensiveAnswerPrompt
        return AnswerGenerationPrompt

    def _grounding_context(self, state: dict[str, Any]) -> GroundingContext | None:
        """The question and context the answer model was given, or ``None``
        when it was given no context (the fixed no-context answer)."""
        if self._context_is_empty(state):
            return None
        context = str(state.get("context") or "")
        optimized: OptimizedContext | None = state.get("optimized_context")
        return GroundingContext(
            query=self._question(state),
            source=context,
            ranked_sources=(
                self.token_manager.ranked_section_lines(optimized)
                if optimized is not None
                else []
            ),
        )

    def _capture_grounding_step(self, state: dict[str, Any]) -> dict[str, Any]:
        """Hand the answer's grounding context to the invoking ``_ainvoke``."""
        capture = _GROUNDING_CAPTURE.get()
        if capture is not None:
            capture.append(self._grounding_context(state))
        return state

    @staticmethod
    def _section_sources(
        state: dict[str, Any], optimized: OptimizedContext
    ) -> list[Source]:
        """The reported source of each section of ``optimized``, the answer
        context (``_section_source``), in section order: built once a query
        and kept in ``state``, as both its citations and its sources read
        them (each built them all, the hybrid scorer's widest contexts
        twice)."""
        kept = state.get(_SECTION_SOURCES)
        if kept is not None and kept[0] is optimized:
            entries: list[Source] = kept[1]
            return entries
        results: list[RetrievalResult] = state["search_results"].results
        entries = [
            GraphRAGChain._section_source(section, results)
            for section in optimized.sections
        ]
        state[_SECTION_SOURCES] = (optimized, entries)
        return entries

    @staticmethod
    def _context_citations(state: dict[str, Any]) -> list[Citation]:
        """One citation per numbered section of the answer context, in
        marker order (documents of graph sections not yet resolved)."""
        optimized: OptimizedContext | None = state.get("optimized_context")
        if optimized is None or GraphRAGChain._context_is_empty(state):
            return []
        entries = GraphRAGChain._section_sources(state, optimized)
        numbered = sorted(
            (
                (section.marker, entry)
                for section, entry in zip(optimized.sections, entries, strict=True)
                if section.marker is not None
            ),
            key=lambda pair: pair[0],
        )
        return [
            citations.citation_from_entry(marker, entry) for marker, entry in numbered
        ]

    async def _answer_with_citations_step(
        self, state: dict[str, Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """The answer (``_answer_step``) with the lineage read of its
        context's citations (``_citations_step``) running alongside: the
        read needs only the context, so it no longer waits for the answer.
        A failed or cancelled answer cancels the read."""
        read_state = dict(state)
        lineage = asyncio.ensure_future(self._citations_step(read_state))
        try:
            answered: dict[str, Any] = await self._answer_step.ainvoke(state, config)
        except BaseException:
            lineage.cancel()
            await asyncio.gather(lineage, return_exceptions=True)
            raise
        await lineage
        answered["citations"] = read_state["citations"]
        if _SECTION_SOURCES in read_state:
            answered[_SECTION_SOURCES] = read_state[_SECTION_SOURCES]
        return answered

    async def _citations_step(self, state: dict[str, Any]) -> dict[str, Any]:
        """Set ``state["citations"]``: the context's citations, the
        documents of graph sections read from their lineage
        (``citations.aresolve_documents``). A failed read leaves those
        without documents, recorded as unresolved (``citations_unresolved``);
        the answer stands."""
        cited = self._context_citations(state)
        state["citations"] = cited
        if not cited:
            return state
        try:
            await citations.aresolve_documents(
                cited,
                functools.partial(self._fetch_text_units, state),
                self.config.search.citations.lineage_units,
            )
        except QueryTimeoutError:
            raise
        except Exception as e:  # noqa: BLE001 - citations never fail an answer
            log_failure(
                logger,
                logging.WARNING,
                "Could not read the text units of the answer's graph "
                "citations; they name no document: %s",
                error=e,
                traceback=True,
            )
            record_degradation(
                DegradationStep.CITATION_LINEAGE,
                "graph citations name no document",
                e,
                unresolved=True,
            )
        return state

    async def _fetch_text_units(
        self, state: dict[str, Any], unit_ids: list[str]
    ) -> list[dict[str, Any]]:
        """The stored text units of ``unit_ids`` (id, documents, attributes)
        in the query's namespace that pass the caller's filters the
        text-units index declares: a unit they exclude is never named."""
        fetched = await fetch_text_units(
            self._get_retriever(RetrieverRole.DOCUMENT),
            self.config,
            state.get("suffix"),
            unit_ids,
            filters=state.get("filters"),
        )
        return list(fetched.units.values())

    @staticmethod
    def _format_output_step(state: dict[str, Any]) -> RAGOutput:
        sr: SearchResult = state["search_results"]
        sr.search_strategy = state["resolved_strategy"].value
        # `content` carries the retrieved TEXT. Omitting it made every vector-retriever
        # result serialise to {"source": <uuid>, "score": <f>} only (its metadata holds
        # no description/name), so `retrieved_contexts` opened with ~20 text-free
        # entries: any consumer that truncates to top-k saw no text at all. That
        # silently zeroed offline context-recall scoring AND starved the RAGAS context
        # metrics for the highest-scoring results. Retrieval and generation are
        # unaffected — this is what gets REPORTED, not what gets retrieved.
        #
        # Sources are what the answer model actually SAW: the token budgeter's
        # selection, not every search result (sections cut for budget were never
        # in the prompt). Without a selection in state (direct callers), fall
        # back to reporting every result. Synthesized results (e.g. the global
        # map-reduce summary) are LLM output the model may read, not retrieved
        # evidence, so they are never reported as sources on either path.
        optimized: OptimizedContext | None = state.get("optimized_context")
        if optimized is None:
            sources = [
                GraphRAGChain._source_entry(r, r.content, truncated=False)
                for r in sr.results
                if not GraphRAGChain._is_synthesized(r.metadata)
            ]
        elif GraphRAGChain._context_is_empty(state):
            # The answer step short-circuited: the model saw no context at all.
            sources = []
        else:
            sources = GraphRAGChain._sources_from_sections(
                optimized.sections,
                sr.results,
                entries=GraphRAGChain._section_sources(state, optimized),
            )

        metadata = {
            "search_strategy": sr.search_strategy,
            "processing_time": time.time() - state["start_time"],
            "total_results": len(sr.results),
            **sr.metadata,
            **({ROUTING_KEY: state[ROUTING_KEY]} if ROUTING_KEY in state else {}),
        }
        GraphRAGChain._report_degradation(state, metadata)
        reason = state.get("abstention_reason")
        if reason is None and GraphRAGChain._context_is_empty(state):
            # The answer step returned the fixed no-context reply.
            reason = GraphRAGChain._no_context_reason(state)
        if reason is not None:
            # A declined answer keeps its sources: the model saw them.
            metadata["abstained"] = True
            metadata[ABSTENTION_REASON_KEY] = reason
        if optimized is not None:
            metadata["context_sections_included"] = optimized.sections_included
            metadata["context_sections_excluded"] = optimized.sections_excluded
        cited = state.get("citations")
        if cited is None:
            # A streamed answer (and a direct caller): no lineage read.
            cited = GraphRAGChain._context_citations(state)
        cited, unknown = citations.mark_cited(cited, str(state.get("answer") or ""))
        if unknown and optimized is not None and optimized.numbered:
            # Only an answer asked to cite has markers that can be wrong.
            metadata["unknown_citation_markers"] = unknown

        return RAGOutput(
            answer=state["answer"],
            sources=sources,
            search_results=sr,
            citations=cited,
            conversation_id=state.get("conversation_id"),
            processed_query=state["processed_query"],
            metadata=metadata,
        )

    @staticmethod
    def _report_degradation(state: dict[str, Any], metadata: dict[str, Any]) -> None:
        """Set the query's degraded steps on ``metadata`` (its ledger's
        ``degraded``, ``degraded_error`` and ``citations_unresolved``,
        ``DegradationLedger.metadata``), over the strategy's own. The
        composed runnable opens a ledger when no query did
        (:class:`_Ledgered`)."""
        if (ledger := current_ledger()) is not None:
            metadata.update(ledger.metadata())

    @staticmethod
    def _is_synthesized(metadata: dict[str, Any] | None) -> bool:
        return bool((metadata or {}).get("synthesized"))

    @staticmethod
    def _source_entry(
        result: RetrievalResult, content: str, *, truncated: bool
    ) -> Source:
        """One reported source: the text the model saw plus its provenance.

        Embedding vectors are dropped (any retriever, not only OpenSearch, may
        carry them in metadata). Provenance keys are added without overwriting
        retriever-supplied metadata of the same name. A statement a seed
        graph makes adds ``provenance`` (:func:`seed_provenance`); it has no
        chunk, so its ``chunk_id`` stays ``None``. A source without seed
        evidence gets no ``provenance`` key.
        """
        metadata = strip_embedding_fields(result.metadata or {})
        findings = metadata.get(REPORT_FINDINGS_FIELD)
        if isinstance(findings, list):
            # A report's findings: their text is its content already; the
            # source keeps the records each cites.
            metadata[REPORT_FINDINGS_FIELD] = [
                {key: finding[key] for key in _FINDING_IDS if key in finding}
                for finding in findings
                if isinstance(finding, dict)
            ]
        chunk_id = result.chunk_id
        if chunk_id is None and result.retriever_type == SectionType.TEXT.value:
            chunk_id = metadata.get("id") or result.source
        provenance: dict[str, Any] = {
            "source_id": result.source,
            "document_ids": source_fields.document_ids(metadata),
            "chunk_id": chunk_id,
            "section_type": result.retriever_type,
            "score": result.score,
        }
        if (seeded := seed_provenance(metadata)) is not None:
            provenance["provenance"] = seeded
        for key, value in provenance.items():
            metadata.setdefault(key, value)
        metadata["truncated"] = truncated
        return Source(
            content=content,
            source=result.source,
            score=result.score,
            metadata=metadata,
        )

    @staticmethod
    def _sources_from_sections(
        sections: list[ContextSection],
        results: list[RetrievalResult],
        *,
        entries: list[Source] | None = None,
    ) -> list[Source]:
        """Map the budgeter's selected sections back to their results.

        Reported in retrieval-rank order (the selection itself is grouped by
        section type). A section without a usable ``result_index`` (a custom
        token manager) is reported from the section alone. Synthesized results
        are skipped (see ``_format_output_step``). ``entries``: each
        section's source, already built (``_section_sources``).
        """
        if entries is None:
            entries = [GraphRAGChain._section_source(s, results) for s in sections]

        def _rank(pair: tuple[ContextSection, Source]) -> int:
            index = pair[0].result_index
            return index if index is not None else len(results)

        sources: list[Source] = []
        for section, entry in sorted(zip(sections, entries, strict=True), key=_rank):
            if GraphRAGChain._is_synthesized(
                GraphRAGChain._section_result(section, results).metadata
            ):
                continue
            sources.append(entry)
        return sources

    @staticmethod
    def _section_result(
        section: ContextSection, results: list[RetrievalResult]
    ) -> RetrievalResult:
        """The result ``section`` was cut from (rebuilt from the section
        itself without a usable ``result_index``)."""
        index = section.result_index
        if index is not None and 0 <= index < len(results):
            return results[index]
        return RetrievalResult(
            content=section.content,
            score=section.priority,
            source=section.source_id,
            retriever_type=section.section_type,
            metadata=section.metadata,
        )

    @staticmethod
    def _section_source(
        section: ContextSection, results: list[RetrievalResult]
    ) -> Source:
        """The reported source of ``section``: the text the model saw plus
        its result's provenance."""
        return GraphRAGChain._source_entry(
            GraphRAGChain._section_result(section, results),
            section.content,
            truncated=bool(section.metadata.get("truncated")),
        )

    @staticmethod
    def _format_search_output_step(state: dict[str, Any]) -> SearchOutput:
        sr: SearchResult = state["search_results"]
        sr.search_strategy = state["resolved_strategy"].value

        metadata = {
            "search_strategy": sr.search_strategy,
            "processing_time": time.time() - state["start_time"],
            "total_results": len(sr.results),
            **sr.metadata,
            **({ROUTING_KEY: state[ROUTING_KEY]} if ROUTING_KEY in state else {}),
        }
        GraphRAGChain._report_degradation(state, metadata)

        return SearchOutput(
            search_results=sr,
            processed_query=state["processed_query"],
            metadata=metadata,
        )

    def invoke(
        self,
        input: RAGInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> RAGOutput:
        """Answer one query with a :class:`RAGOutput`; the sync form of
        :meth:`ainvoke`.

        Runs on the chain's own event loop (see the class docstring).
        """
        # Runs on the chain's long-lived loop (not asyncio.run per call), so it
        # also works from a thread that already runs an event loop and reuses
        # the loop-bound retrievers across calls (batch() fans out to here).
        result: RAGOutput = self._loop_runner.run(self.ainvoke(input, config, **kwargs))
        return result

    async def ainvoke(
        self,
        input: RAGInput,
        config: RunnableConfig | None = None,
        *,
        on_answer_chunk: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> RAGOutput:
        """Answer one query.

        Args:
            input: A :class:`RAGInput` or a dict of its fields.
            config: LangChain run config; its callbacks, tags and metadata
                reach every nested LLM call.
            on_answer_chunk: Called with each text chunk of the answer as
                the answer model streams it, for a client that shows the
                answer as it is written. A preview: the returned
                ``answer`` is authoritative (a decline or a failure
                replaces what was streamed). Nothing is streamed when the
                output guardrail screens answers or the answer is read for
                the ``[NO_ANSWER]`` token (``search.abstention``), nor for
                the fixed reply to an empty context.

        Returns:
            A :class:`RAGOutput`: the answer, its sources and the search
            results.

        Raises:
            Exception: Any query failure, unless ``processing.ignore_errors``
                is set, in which case a degraded result with
                ``metadata["error"] = True`` is returned instead.
        """
        token = _ANSWER_CHUNKS.set(on_answer_chunk)
        try:
            result = await self._query(input, config, search=False)
        finally:
            _ANSWER_CHUNKS.reset(token)
        assert isinstance(result, RAGOutput)
        return result

    def search(
        self, input: RAGInput, config: RunnableConfig | None = None
    ) -> SearchOutput:
        """Retrieve for one query without answering; the sync form of
        :meth:`asearch`."""
        result: SearchOutput = self._loop_runner.run(self.asearch(input, config))
        return result

    async def asearch(
        self, input: RAGInput, config: RunnableConfig | None = None
    ) -> SearchOutput:
        """Retrieve for one query without generating an answer.

        Runs the same steps as :meth:`ainvoke` up to retrieval (the input
        guardrail, follow-up rewrite, routing, query processing, search,
        fusion and rerank) and returns the ranked results. Nothing is written
        to conversation memory. Errors behave as in :meth:`ainvoke`.
        """
        result = await self._query(input, config, search=True)
        assert isinstance(result, SearchOutput)
        return result

    async def _query(
        self,
        input: RAGInput | dict[str, Any],
        config: RunnableConfig | None,
        *,
        search: bool,
    ) -> RAGOutput | SearchOutput:
        # Fixed when the query starts: its admission (the build-record read
        # of the methodology check, a store round trip) counts too, in the
        # deadline and the latency.
        deadline = _QueryDeadline(self.config.search.query_timeout_seconds)
        started = time.monotonic()
        query_id = self._query_id()
        # The admission's log lines carry the query's id too.
        admission_tokens = bind_contextvars(query_id=query_id)
        try:
            rag_input, input_dict = await self._admit_query(input, deadline)
        except Exception as e:
            tag_query_failure(e, query_id)
            if not search:
                self._record_query_metrics(started, input, error=e)
            raise
        finally:
            reset_contextvars(**admission_tokens)
        log_context = self._log_context(rag_input, query_id)
        log_tokens = bind_contextvars(**log_context)
        ids: tuple[str, str] | None = None
        meter: UsageMeter | None = None
        try:
            # The root span of a query trace: the strategy, retrieval, fusion,
            # rerank, context build and model calls nest under it, each
            # tagged with the namespace (the tenant boundary).
            with (
                self._identity(rag_input),
                observability.bind_attributes(namespace=rag_input.suffix),
                observability.span(
                    "kg_rag.query", self._query_span_attributes(rag_input, search)
                ) as span,
                usage_scope() as meter,
                # The query's store-lookup memo and degradation ledger.
                query_scope(),
            ):
                ids = observability.current_trace_ids()
                assert meter is not None
                leftovers: list[asyncio.Future[Any]] = []
                token = _LEFTOVER_EXTRACTIONS.set(leftovers)
                try:
                    result = await self._ainvoke(
                        rag_input, input_dict, config, span, search, deadline
                    )
                except BaseException:
                    await discard(leftovers)
                    raise
                finally:
                    _LEFTOVER_EXTRACTIONS.reset(token)
                # Not waited for: counted from their start below, at an
                # estimate until they answer.
                self._background_extractions.keep(leftovers, self._time_left(started))
                # The query's model calls and tokens (shared.usage), and the
                # ids that tie it to its log lines and trace.
                result.metadata["usage"] = meter.snapshot()
                result.metadata["query_id"] = log_context["query_id"]
                result.metadata["trace_id"] = ids[0] if ids else None
                span.set_attributes(
                    {
                        **self._query_outcome_attributes(result),
                        **_captured_content(
                            input=_screened_query(rag_input, result.metadata),
                            output=(
                                result.answer if isinstance(result, RAGOutput) else None
                            ),
                        ),
                    }
                )
        except Exception as e:
            tag_query_failure(e, query_id, ids[0] if ids else None)
            if not search:
                self._record_query_metrics(
                    started,
                    rag_input,
                    error=e,
                    usage=meter.snapshot() if meter is not None else None,
                )
            raise
        finally:
            reset_contextvars(**log_tokens)
        if not search:
            self._record_query_metrics(
                started, rag_input, result=result, usage=result.metadata.get("usage")
            )
        return result

    @staticmethod
    def _query_id() -> str:
        """A query's id: one bound by the caller (as the console's or an MCP
        call's, :func:`new_query_id`), else a new one."""
        return str(get_contextvars().get("query_id") or new_query_id())

    def record_unrun_query(
        self, query: Mapping[str, Any], error: BaseException, *, started: float
    ) -> None:
        """Publish the query metrics of an answer a host gave up on before
        or while the chain ran it, so no query metric was recorded (an MCP
        ``kg_answer`` that timed out waiting for a call slot, or at the
        backstop past the chain's deadline): one failed query, as
        :meth:`ainvoke` records its own. ``query``: the fields known
        (``search_strategy``, ``suffix``); ``started``: the
        ``time.monotonic()`` the call began."""
        self._record_query_metrics(started, dict(query), error=error)

    @staticmethod
    def _log_context(rag_input: RAGInput, query_id: str) -> dict[str, Any]:
        """What every log line of a query carries: its id, its conversation
        and its namespace (``rag_input.suffix``, resolved)."""
        return {
            "query_id": query_id,
            "conversation_id": rag_input.conversation_id,
            "namespace": rag_input.suffix,
        }

    def _record_query_metrics(
        self,
        started: float,
        query: RAGInput | dict[str, Any],
        *,
        result: RAGOutput | SearchOutput | None = None,
        error: BaseException | None = None,
        error_class: ErrorClass | None = None,
        usage: Mapping[str, Any] | None = None,
        ran: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Publish an answered query's latency and outcome through the
        providers' metrics sink (``Providers.metrics_sink``: ``--metrics-sink
        cloudwatch``, or a host's), per strategy that ran (the requested one
        when none did): ``query_failed`` (raised, its admission included, or
        the error answer of ``processing.ignore_errors``; not a caller's
        error, ``ErrorClass.is_caller_error``, whose class the
        ``error_class`` property names as any failure's), ``query_degraded``
        (answered without a step) and ``query_abstained``, 0 or 1 each, and
        ``query_throttled_requests`` (the requests Bedrock throttled, when
        the usage is known). Answers only, so a retrieval-only search does
        not lower the answer latency. ``query``: the admitted input, or the
        one the admission refused; ``ran`` and ``metadata``: the strategy
        that ran and the answer's metadata, when no ``result`` carries them
        (a streamed answer's). The ``Strategy`` dimension is a
        ``SearchStrategy`` value, or ``invalid`` for a refused input that
        names none (one series, not one per string)."""
        if result is not None:
            metadata = result.metadata
            ran = result.search_results.search_strategy
        metadata = metadata or {}
        if error is not None:
            error_class = classify(error)
        elif metadata.get("error"):
            error_class = error_class_of(metadata.get(ERROR_CLASS_KEY))
        # Any input, a refused one too: never an error here in its place.
        fields = query if isinstance(query, dict) else getattr(query, "__dict__", {})
        strategy = _strategy_dimension(fields.get("search_strategy"))
        if ran is not None and error_class is None:
            strategy = _strategy_dimension(ran, strategy)
        namespace = fields.get("suffix")
        with contextlib.suppress(Exception):
            # Resolved as the admission resolves it (a refused input's too).
            namespace = resolve_namespace(self.config, namespace)
        metrics = {
            QUERY_LATENCY_METRIC: (time.monotonic() - started) * 1000,
            QUERY_FAILED_METRIC: float(
                error_class is not None and not error_class.is_caller_error
            ),
            QUERY_DEGRADED_METRIC: float(bool(DegradationReport.of(metadata).degraded)),
            QUERY_ABSTAINED_METRIC: float(bool(metadata.get("abstained"))),
        }
        if usage is not None:
            metrics[QUERY_THROTTLED_METRIC] = float(
                usage.get("throttled_requests") or 0
            )
        # Log properties, not dimensions (QUERY_METRICS_DIMENSION_KEYS): the
        # namespace by the key the ingestion records use.
        properties = {"Strategy": strategy}
        if namespace:
            properties["namespace"] = str(namespace)
        if error_class is not None:
            properties["error_class"] = error_class.value
        try:
            self.providers.metrics_sink.emit(
                QUERY_METRICS_NAMESPACE, metrics, properties
            )
        except Exception as e:  # noqa: BLE001 - metrics never fail a query
            logger.warning("Query metrics emission failed: %s", error_digest(e))

    def _time_left(self, started: float) -> float | None:
        """Seconds left of the query deadline (``search.query_timeout_seconds``)
        of a query that started at ``time.monotonic()`` ``started``."""
        seconds = self.config.search.query_timeout_seconds
        if seconds is None:
            return None
        return max(0.0, seconds - (time.monotonic() - started))

    @staticmethod
    def _identity(rag_input: RAGInput) -> AbstractContextManager[None]:
        """Tag the query's spans with ``session.id`` (the conversation id) and
        ``user.id``."""
        return observability.bind_identity(
            session_id=rag_input.conversation_id, user_id=rag_input.user_id
        )

    @staticmethod
    def _identity_metadata(rag_input: RAGInput) -> dict[str, str]:
        """Run-config metadata naming the query's session and user."""
        identity = {
            "session_id": rag_input.conversation_id,
            "user_id": rag_input.user_id,
        }
        return {key: value for key, value in identity.items() if value}

    def _query_span_attributes(
        self, rag_input: RAGInput, search: bool
    ) -> dict[str, Any]:
        """Query-span attributes: the request's settings. The query text is
        recorded once screened (``_screened_query``), never before."""
        return {
            GEN_AI_CONVERSATION_ID: rag_input.conversation_id,
            "kg_rag.mode": "search" if search else "rag",
            "kg_rag.requested_strategy": rag_input.search_strategy,
            "kg_rag.top_k": rag_input.top_k,
            "kg_rag.use_memory": rag_input.use_memory,
        }

    @staticmethod
    def _query_outcome_attributes(
        output: RAGOutput | SearchOutput,
    ) -> dict[str, Any]:
        metadata = output.metadata
        attributes: dict[str, Any] = {
            "kg_rag.error": bool(metadata.get("error")),
            "kg_rag.abstained": bool(metadata.get("abstained")),
            "kg_rag.blocked": bool(metadata.get("blocked")),
            # A partial answer and an absorbed error's class, for trace lists
            # to filter on (the strategy span names the step).
            "kg_rag.degraded": bool(DegradationReport.of(metadata).degraded),
            "kg_rag.error_class": metadata.get(ERROR_CLASS_KEY),
        }
        if isinstance(output, RAGOutput):
            attributes["kg_rag.strategy"] = output.search_results.search_strategy
            attributes["kg_rag.sources"] = len(output.sources)
        return attributes

    async def _ainvoke(
        self,
        rag_input: RAGInput,
        input_dict: dict[str, Any],
        config: RunnableConfig | None,
        span: observability.SpanHandle,
        search: bool,
        deadline: "_QueryDeadline",
    ) -> RAGOutput | SearchOutput:
        guard_records: list[dict[str, Any]] = []
        capture: list[GroundingContext | None] = []
        capture_token = _GROUNDING_CAPTURE.set(
            capture if self._ground_outputs else None
        )
        try:
            async with deadline:
                # Before the first model call (the input guardrail's).
                self._raise_missing_namespace(input_dict)
                blocking = await self._screen_input(
                    rag_input, input_dict, guard_records
                )
                if blocking is None:
                    await self._admit_history(rag_input, input_dict, guard_records)
                else:
                    # Not written to memory: a refused turn is no history.
                    return self._short_circuit_output(
                        rag_input,
                        input_dict,
                        search=search,
                        answer=blocking.text,
                        strategy="blocked",
                        metadata={"blocked": True, "guardrail": guard_records},
                    )
                runnable = self._search_chain if search else self.chain
                output: RAGOutput | SearchOutput = await runnable.ainvoke(
                    input_dict, config
                )
                answer_blocked = False
                plan_texts: dict[str, str | None] = {}
                verdict, evidence_records = await self._screen_outputs_of(
                    output, guard_records, plan_texts, capture[-1] if capture else None
                )
                if isinstance(output, RAGOutput):
                    assert verdict is not None
                    screened = verdict.text != output.answer
                    output.answer = verdict.text
                    if verdict.grounding is not None:
                        output.metadata["grounding"] = verdict.grounding.model_dump()
                    answer_blocked = verdict.action is GuardAction.BLOCKED
                    if answer_blocked:
                        # A refused answer keeps none of the evidence behind it.
                        output.sources = []
                        output.citations = []
                        output.metadata.pop("unknown_citation_markers", None)
                        output.search_results = output.search_results.model_copy(
                            update={"results": [], "total_results": 0}
                        )
                        # Nor the model's reading of the question.
                        self._clear_query_texts(output)
                        self._clear_evidence_metadata(output)
                    elif screened:
                        self._mark_citations(output)
                if not answer_blocked:
                    # A refused answer keeps none of what they screened.
                    guard_records.extend(evidence_records)
                if isinstance(output, RAGOutput):
                    self._refresh_screened_copies(output, plan_texts)
                if guard_records:
                    output.metadata["guardrail"] = guard_records
                if isinstance(output, RAGOutput) and not answer_blocked:
                    # A refused answer is no history either.
                    await self._save_memory(output, rag_input)
                return output
        except Exception as raised:
            # A model server that cannot be reached, named (provider kind,
            # base_url), and a model client's own error as a
            # LanguageModelError, rather than as its transport's error.
            e = named_model_error(raised, self.config)
            if not self.ignore_errors:
                if e is raised:
                    raise
                raise e from raised

            log_failure(
                logger,
                logging.ERROR,
                "RAG chain execution failed for query %s: %s",
                text_digest(rag_input.query),
                error=e,
            )
            # The degraded answer is still a failed query: mark its span so.
            span.record_exception(e)
            span.set_error(type(e).__name__)
            span.set_attribute(ERROR_TYPE, type(e).__name__)
            return self._short_circuit_output(
                rag_input,
                input_dict,
                search=search,
                answer=DEFAULT_ERROR_MESSAGE,
                strategy="error",
                # The error's type and class (ErrorClass), for a caller that
                # tells a refused query (a strategy or filter to change, the
                # deadline, a server to start) apart.
                metadata={
                    "error": True,
                    "error_type": type(e).__name__,
                    ERROR_CLASS_KEY: classify(e).value,
                },
                search_metadata={"error": str(e)},
            )
        finally:
            _GROUNDING_CAPTURE.reset(capture_token)

    @staticmethod
    def _refresh_screened_copies(
        output: RAGOutput, plan_texts: dict[str, str | None]
    ) -> None:
        """Give ``output`` the screened model text of its search results
        where it holds copies made before the screening: its metadata's
        run records (``RunRecord``; ``_format_output_step`` copies the
        search metadata) and the reported query-result sources and their
        citations, whose text, metadata and title render the structured plan
        (``plan_texts``: each screened text and what the guardrail made of
        it)."""
        for spec in run_records():
            if spec.key in output.metadata:
                output.metadata[spec.key] = output.search_results.metadata.get(spec.key)
        if not any(text != screened for text, screened in plan_texts.items()):
            return
        # A query result's citation is titled by its plan description.
        output.citations = [
            (
                c.model_copy(update={"title": _replace_texts(c.title, plan_texts)})
                if c.title and c.section_type == SectionType.QUERY_RESULT.value
                else c
            )
            for c in output.citations
        ]
        output.sources = [
            (
                {
                    **source,
                    "content": _replace_texts(source.get("content") or "", plan_texts),
                    "metadata": _replace_leaves(
                        source.get("metadata") or {}, plan_texts
                    ),
                }
                if (source.get("metadata") or {}).get("section_type")
                == SectionType.QUERY_RESULT.value
                else source
            )
            for source in output.sources
        ]

    @staticmethod
    def _mark_citations(output: RAGOutput) -> None:
        """Set ``output``'s ``cited`` flags from its (screened) answer."""
        output.citations, unknown = citations.mark_cited(
            output.citations, output.answer
        )
        reported = output.metadata.pop("unknown_citation_markers", None)
        # Reported for an answer asked to cite (it has citations or had
        # unknown markers), as _format_output_step reports them.
        if unknown and (output.citations or reported is not None):
            output.metadata["unknown_citation_markers"] = unknown

    def _short_circuit_output(
        self,
        rag_input: RAGInput,
        input_dict: dict[str, Any],
        *,
        search: bool,
        answer: str,
        strategy: str,
        metadata: dict[str, Any],
        search_metadata: dict[str, Any] | None = None,
    ) -> RAGOutput | SearchOutput:
        """The result of a query that ended before retrieval (error, block)."""
        processing_time = time.time() - input_dict["start_time"]
        search_result = SearchResult(
            query=SearchQuery(query=rag_input.query),
            results=[],
            total_results=0,
            search_strategy=strategy,
            processing_time=processing_time,
            metadata=search_metadata or {},
        )
        processed_query = ProcessedQuery(
            original_query=rag_input.query, final_query=rag_input.query
        )
        metadata = {**metadata, "processing_time": processing_time}
        if search:
            return SearchOutput(
                search_results=search_result,
                processed_query=processed_query,
                metadata=metadata,
            )
        return RAGOutput(
            answer=answer,
            sources=[],
            search_results=search_result,
            conversation_id=rag_input.conversation_id,
            processed_query=processed_query,
            metadata=metadata,
        )

    async def _screen_input(
        self,
        rag_input: RAGInput,
        input_dict: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> GuardVerdict | None:
        """Screen the raw query; the blocking verdict, or ``None`` to go on.

        A masked query (e.g. PII anonymized) replaces the query in
        ``rag_input`` and ``input_dict``, so retrieval, the model calls and
        conversation memory only ever see the masked text.
        """
        if not self._screen_inputs:
            return None
        assert self.guard is not None
        verdict = await self.guard.acheck(rag_input.query, GuardSource.INPUT)
        if not verdict.intervened:
            return None
        records.append(_guard_record(GuardSource.INPUT, verdict))
        if verdict.action is GuardAction.BLOCKED:
            return verdict
        rag_input.query = input_dict["query"] = verdict.text
        return None

    def _bounded_history(self, turns: list[ConversationTurn]) -> list[ConversationTurn]:
        """The turns of ``turns`` the rewrite reads: the last
        ``memory.max_messages_per_conversation``, then the most recent ones
        whose text (answers cut as memory cuts them) fits
        ``memory.max_history_tokens``."""
        memory = self.config.memory
        kept = turns[-memory.max_messages_per_conversation :]
        budget = memory.max_history_tokens
        messages: list[BaseMessage] = []
        bounded: list[ConversationTurn] = []
        for turn in reversed(kept):
            message: BaseMessage = (
                HumanMessage(content=turn.content)
                if turn.role == "user"
                else AIMessage(content=turn.content)
            )
            text = history_buffer([message, *messages], HISTORY_ANSWER_CHARS)
            if estimate_token_count(text) > budget:
                break
            messages.insert(0, message)
            bounded.insert(0, turn)
        return bounded

    async def _admit_history(
        self,
        rag_input: RAGInput,
        input_dict: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> None:
        """Bound the caller's ``history`` and screen it as the query is.

        One input check reads every bounded turn; only when it intervenes is
        each checked on its own: a masked turn is replaced, a blocked one
        dropped (recorded), so a client that replays a refused turn is not
        refused for good."""
        if rag_input.history is None:
            return
        turns = self._bounded_history(rag_input.history)
        if self._screen_inputs and turns:
            assert self.guard is not None
            verdict = await self.guard.acheck(
                "\n\n".join(t.content for t in turns), GuardSource.INPUT
            )
            if verdict.intervened:
                screened: list[ConversationTurn] = []
                for turn in turns:
                    one = await self.guard.acheck(turn.content, GuardSource.INPUT)
                    if not one.intervened:
                        screened.append(turn)
                        continue
                    records.append(_guard_record(GuardSource.INPUT, one))
                    if one.action is not GuardAction.BLOCKED:
                        screened.append(turn.model_copy(update={"content": one.text}))
                turns = screened
        rag_input.history = turns
        input_dict["history"] = [t.model_dump() for t in turns]

    async def _screen_outputs_of(
        self,
        output: RAGOutput | SearchOutput,
        records: list[dict[str, Any]],
        plan_texts: dict[str, str | None],
        grounding: GroundingContext | None,
    ) -> tuple[GuardVerdict | None, list[dict[str, Any]]]:
        """Screen ``output``'s model-written text at once: the answer
        (:meth:`_screen_output`, its verdict returned and recorded on
        ``records``; ``None`` for a search), the synthesized search results
        (:meth:`_screen_synthesized`) and the query texts
        (:meth:`_query_text_screening`), which are applied to ``output``.
        The last two do not depend on the answer's verdict, so they no longer
        wait for it: one guardrail round trip after the answer, not three.
        Their records are returned apart, for the caller to keep unless the
        answer was refused (its refusal drops what they screened)."""
        evidence: list[dict[str, Any]] = []
        queried: list[dict[str, Any]] = []

        async def answer() -> GuardVerdict | None:
            if not isinstance(output, RAGOutput):
                return None
            return await self._screen_output(
                output.answer, records, grounding=grounding
            )

        tasks = (
            asyncio.ensure_future(answer()),
            asyncio.ensure_future(
                self._screen_synthesized(
                    output.search_results, evidence, plan_texts=plan_texts
                )
            ),
            asyncio.ensure_future(self._query_text_screening(output, queried)),
        )
        async with cancel_on_exit(tasks):
            verdict, synthesized, screened = await asyncio.gather(*tasks)
        output.search_results = synthesized
        self._apply_query_texts(output, screened)
        return verdict, evidence + queried

    async def _screen_output(
        self,
        answer: str,
        records: list[dict[str, Any]],
        *,
        grounding: GroundingContext | None = None,
    ) -> GuardVerdict:
        """The verdict on the answer; its text is what the user may see:
        unchanged, masked or refused.

        The answer is screened here; the sources and the retrieved search
        results are the caller's own indexed corpus text, returned as
        retrieved (a blocked answer drops them), while the model-written
        search results are screened apart (:meth:`_screen_synthesized`).
        The fixed no-context answer is not model output and is not screened. With the contextual grounding check on,
        ``grounding`` (the question and context the answer was generated
        from) goes with it.
        """
        if not self._screen_outputs or not answer or is_no_context_answer(answer):
            return GuardVerdict(action=GuardAction.NONE, text=answer)
        assert self.guard is not None
        if self._ground_outputs:
            verdict = await self.guard.acheck(
                answer, GuardSource.OUTPUT, grounding=grounding
            )
        else:
            verdict = await self.guard.acheck(answer, GuardSource.OUTPUT)
        if verdict.intervened:
            records.append(_guard_record(GuardSource.OUTPUT, verdict))
        return verdict

    async def _screen_synthesized(
        self,
        search_results: SearchResult,
        records: list[dict[str, Any]],
        *,
        plan_texts: dict[str, str | None] | None = None,
    ) -> SearchResult:
        """``search_results`` with each model-written result (``synthesized``:
        DRIFT's intermediate answers, global search's key points or summary)
        screened as an answer is: masked, or dropped when refused.

        They are returned with the search results (``ainvoke`` and
        ``asearch``), so with ``check_output`` they must not reach the caller
        unscreened. Retrieved results are corpus text and pass as they are.
        They are checked joined, in chunks of at most
        ``_JOINED_CHECK_CHARS`` characters (``_check_chunks``), one chunk
        after another; only the
        results of a chunk whose check intervened are checked each on its
        own, ``search.max_concurrency`` at a time (a check per key point,
        one after another, added a guardrail round trip per point to every
        global query). A joined check that intervened is recorded with
        ``scope: joined``, so an intervention on the combination no single
        result draws is reported; its text is never used. ``plan_texts``
        receives the structured plan's texts and what the guardrail made of
        each (``_screen_structured_metadata``).
        """
        if not self._screen_outputs:
            return search_results
        search_results = await self._screen_model_metadata(
            search_results, records, plan_texts=plan_texts
        )
        written = [
            result.content
            for result in search_results.results
            if self._is_synthesized(result.metadata) and result.content
        ]
        if not written:
            return search_results
        assert self.guard is not None
        guard = self.guard
        concurrency = self.config.search.max_concurrency
        chunks = _check_chunks(written, _JOINED_CHECK_CHARS)
        # One chunk after another: together they may exceed the quota.
        joined = await gather_bounded(
            1,
            (
                guard.acheck("\n\n".join(written[i] for i in chunk), GuardSource.OUTPUT)
                for chunk in chunks
            ),
        )
        verdicts: dict[int, GuardVerdict] = {}
        again: list[int] = []
        for chunk, verdict in zip(chunks, joined, strict=True):
            if not verdict.intervened:
                continue
            if len(chunk) == 1:
                # A chunk of one result: its check is that result's own.
                verdicts[chunk[0]] = verdict
                continue
            records.append(
                {**_guard_record(GuardSource.OUTPUT, verdict), "scope": "joined"}
            )
            again.extend(chunk)
        if not verdicts and not again:
            return search_results
        verdicts.update(
            zip(
                again,
                await gather_bounded(
                    concurrency,
                    (guard.acheck(written[i], GuardSource.OUTPUT) for i in again),
                ),
                strict=True,
            )
        )
        kept: list[RetrievalResult] = []
        position = -1
        for result in search_results.results:
            if not self._is_synthesized(result.metadata) or not result.content:
                kept.append(result)
                continue
            position += 1
            own = verdicts.get(position)
            if own is None or not own.intervened:
                kept.append(result)
                continue
            records.append(_guard_record(GuardSource.OUTPUT, own))
            if own.action is not GuardAction.BLOCKED:
                kept.append(result.model_copy(update={"content": own.text}))
        return search_results.model_copy(
            update={"results": kept, "total_results": len(kept)}
        )

    async def _screen_texts(
        self, texts: list[str], records: list[dict[str, Any]]
    ) -> dict[str, str | None] | None:
        """Each of the short model-written ``texts`` and what the output
        guardrail made of it (``None``: refused), or ``None`` when nothing
        needs replacing. One check reads them all; only when it intervenes
        is each checked on its own (and each intervention recorded)."""
        if not texts:
            return None
        assert self.guard is not None
        verdict = await self.guard.acheck("\n".join(texts), GuardSource.OUTPUT)
        if not verdict.intervened:
            return None
        screened: dict[str, str | None] = {}
        for text in dict.fromkeys(texts):
            one = await self.guard.acheck(text, GuardSource.OUTPUT)
            if one.intervened:
                records.append(_guard_record(GuardSource.OUTPUT, one))
            screened[text] = (
                None
                if one.intervened and one.action is GuardAction.BLOCKED
                else (one.text if one.intervened else text)
            )
        return screened

    @staticmethod
    def _clear_evidence_metadata(output: RAGOutput) -> None:
        """Drop the evidence a strategy's metadata holds (a refused answer):
        every run record (``RunRecord``: the structured plan, its linked
        entities and rows, iterative search's follow-up queries and
        verdicts), in the answer's and its search results' metadata."""
        keys = {spec.key for spec in run_records()}
        for key in keys:
            output.metadata.pop(key, None)
        output.search_results = output.search_results.model_copy(
            update={
                "metadata": {
                    k: v
                    for k, v in output.search_results.metadata.items()
                    if k not in keys
                }
            }
        )

    @staticmethod
    def _clear_query_texts(output: RAGOutput | SearchOutput) -> None:
        """Drop ``output``'s model-written query texts (a refused answer):
        the query reads as the user's own, with no extractions."""
        original = output.processed_query.original_query
        output.processed_query = ProcessedQuery(
            original_query=original, final_query=original
        )
        output.search_results = output.search_results.model_copy(
            update={
                "query": output.search_results.query.model_copy(
                    update={
                        "query": original,
                        "entity_focus": [],
                        "hl_keywords": [],
                        "ll_keywords": [],
                    }
                )
            }
        )

    async def _query_text_screening(
        self, output: RAGOutput | SearchOutput, records: list[dict[str, Any]]
    ) -> dict[str, str | None] | None:
        """What the output guardrail makes of ``output``'s model-written
        query texts (the rewritten and translated query, the entities and
        LightRAG keywords, and the search query built from them;
        ``_PROCESSED_QUERY_MODEL_TEXT``), screened as a synthesized result
        is (``_screen_texts``; ``None``: nothing to replace). The original
        query is the user's own, screened as input."""
        if not self._screen_outputs:
            return None
        processed, query = output.processed_query, output.search_results.query
        original = processed.original_query
        texts = [
            text
            for text in dict.fromkeys(
                [
                    *_query_model_texts(processed, _PROCESSED_QUERY_MODEL_TEXT),
                    *_query_model_texts(query, _SEARCH_QUERY_MODEL_TEXT),
                ]
            )
            if text != original
        ]
        return await self._screen_texts(texts, records)

    @staticmethod
    def _apply_query_texts(
        output: RAGOutput | SearchOutput, screened: dict[str, str | None] | None
    ) -> None:
        """``output``'s query texts as screened (``_query_text_screening``):
        masked, or left out when refused (a refused searched query reads as
        the original)."""
        if screened is None:
            return
        processed, query = output.processed_query, output.search_results.query
        original = processed.original_query
        output.processed_query = _replace_query_texts(
            processed, _PROCESSED_QUERY_MODEL_TEXT, screened, refused=original
        )
        output.search_results = output.search_results.model_copy(
            update={
                "query": _replace_query_texts(
                    query, _SEARCH_QUERY_MODEL_TEXT, screened, refused=original
                )
            }
        )

    async def _screen_model_metadata(
        self,
        search_results: SearchResult,
        records: list[dict[str, Any]],
        *,
        plan_texts: dict[str, str | None] | None = None,
    ) -> SearchResult:
        """``search_results`` with the model-written text of every run
        record its metadata holds (``RunRecord.model_text``: iterative's
        follow-up queries and bridge entities, the structured plan) screened
        as a synthesized result is, in the record and in the query results
        that render it: masked, or removed when refused. One check per
        record reads its texts; only when it intervenes is each checked on
        its own, and ``plan_texts`` receives each text's screened form
        (``None``: refused)."""
        for spec in run_records():
            record = search_results.metadata.get(spec.key)
            if not isinstance(record, dict):
                continue
            screened = await self._screen_texts(_record_texts(record, spec), records)
            if screened is None:
                continue
            if plan_texts is not None:
                plan_texts.update(screened)
            search_results = search_results.model_copy(
                update={
                    "results": [
                        (
                            result.model_copy(
                                update={
                                    "content": _replace_texts(result.content, screened),
                                    "metadata": _replace_leaves(
                                        result.metadata, screened
                                    ),
                                }
                            )
                            if result.retriever_type == SectionType.QUERY_RESULT.value
                            else result
                        )
                        for result in search_results.results
                    ],
                    "metadata": {
                        **search_results.metadata,
                        spec.key: _screened_record(record, spec, screened),
                    },
                }
            )
        return search_results

    async def _admit_query(
        self, inputs: RAGInput | dict[str, Any], deadline: "_QueryDeadline"
    ) -> tuple[RAGInput, dict[str, Any]]:
        """``inputs`` checked before any model call (``_prepare_invoke``),
        with its namespace resolved once (``resolve_namespace``: a query
        without a suffix reads the config's ``index_value``), the methodology
        and the filter keys checked against what the namespace holds, within
        the query's ``deadline``.

        A deadline that passes during the checks is the query's
        ``QueryTimeoutError``, raised before its first model call as a
        namespace check a store failed is (``_raise_missing_namespace``):
        under ``ignore_errors`` the error answer, as a timeout later in the
        query gives. A refused strategy or filter still raises: a usage
        error, not a failed query."""
        rag_input, input_dict = self._prepare_invoke(inputs)
        rag_input.suffix = input_dict["suffix"] = resolve_namespace(
            self.config, rag_input.suffix
        )
        try:
            async with deadline.scope():
                await self._check_holdings(rag_input, input_dict)
        except QueryTimeoutError as e:
            input_dict["held_artifacts"] = frozenset()
            input_dict[_MISSING_NAMESPACE] = e
        return rag_input, input_dict

    async def _check_holdings(
        self, rag_input: RAGInput, input_dict: dict[str, Any]
    ) -> None:
        """The namespace, methodology and filter-key checks of
        ``_admit_query``; the namespace's holdings go to ``input_dict``."""
        try:
            await self._arequire_namespace(rag_input.suffix)
        except Exception as e:  # noqa: BLE001 - raised by the query below
            # Nothing ingested under it (IndexNotFoundError: it holds
            # nothing, not what the config builds, which a methodology
            # refusal would name), or a store that cannot tell: the query
            # raises it (_raise_missing_namespace; under ignore_errors, its
            # error answer) before any model call.
            input_dict["held_artifacts"] = frozenset()
            input_dict[_MISSING_NAMESPACE] = e
            return
        # What auto routes among and the strategy reads: one read of the
        # build record a query (a failed one is not kept, nor paid twice).
        holdings = await methodology_check.check_strategy_holdings(
            self, rag_input.search_strategy, rag_input.suffix
        )
        input_dict["held_artifacts"] = holdings.artifacts
        # Filter keys after the methodology check, against the indexes the
        # namespace holds: a key only an index it lacks declares would be
        # dropped everywhere.
        filter_validation.check_requested_filter_keys(
            self, rag_input, input_dict["held_artifacts"]
        )

    @staticmethod
    def _raise_missing_namespace(input_dict: dict[str, Any]) -> None:
        """Raise the error the admission's namespace check met
        (``_admit_query``: nothing ingested under the query's suffix, a
        store that could not be reached, or the query's deadline)."""
        if (missing := input_dict.pop(_MISSING_NAMESPACE, None)) is not None:
            raise missing

    async def _arequire_namespace(self, suffix: str | None) -> None:
        """Raise ``IndexNotFoundError`` when nothing was ingested under
        ``suffix`` (``arequire_namespace`` of the document retriever), before
        the query's first model call (under ``ignore_errors`` its error
        answer); a namespace found once is not checked again by this
        chain."""
        suffix = resolve_namespace(self.config, suffix)
        if suffix in self._present_namespaces:
            return
        retriever = self._get_retriever(RetrieverRole.DOCUMENT)
        # A retriever built outside the base class may not tell.
        if (require := getattr(retriever, "arequire_namespace", None)) is not None:
            await require(suffix)
        with self._cache_lock:
            self._present_namespaces.add(suffix)

    @staticmethod
    def _prepare_invoke(
        inputs: RAGInput | dict[str, Any],
    ) -> tuple[RAGInput, dict[str, Any]]:
        # A copy: the generated conversation id and a masked query must not
        # change the caller's object.
        rag_input = (
            RAGInput.checked(**inputs)
            if isinstance(inputs, dict)
            else inputs.model_copy()
        )

        # A string for a boolean or numeric field becomes that type: the
        # graph stores match a value as given, OpenSearch converts it.
        # Malformed filter values fail now, before any model call.
        rag_input.filters = typed_filter_values(rag_input.filters)
        check_filter_values(rag_input.filters)
        if rag_input.use_memory and not rag_input.conversation_id:
            rag_input.conversation_id = str(uuid.uuid4())

        input_dict = rag_input.model_dump()
        input_dict["start_time"] = time.time()
        return rag_input, input_dict

    async def _save_memory(self, output: RAGOutput, rag_input: RAGInput) -> None:
        if not output.conversation_id:
            return

        try:
            query = (
                output.processed_query.original_query if output.processed_query else ""
            )
            key = _conversation_key(
                rag_input.suffix, rag_input.user_id, output.conversation_id
            )
            await self.memory_manager.add_turn(key, query, output.answer)
        except Exception as e:  # noqa: BLE001 - memory never fails an answer
            log_failure(
                logger,
                logging.ERROR,
                "Failed to save conversation memory for conversation '%s': %s",
                output.conversation_id,
                error=e,
                traceback=True,
            )

    # Streaming yields answer TEXT chunks (str), not RAGOutput: callers of a
    # streamed answer want tokens as they arrive. Sources/metadata stay on the
    # non-streaming invoke/ainvoke path.
    def stream(  # type: ignore[override]
        self,
        input: RAGInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Synchronously stream answer chunks.

        Yields the answer as ``str`` chunks only, never a :class:`RAGOutput`:
        to get the sources and metadata as well, call :meth:`invoke` or
        :meth:`ainvoke` instead.

        Drives :meth:`astream` on the chain's long-lived event loop (the one
        ``invoke`` uses), pulling one chunk at a time. This works both from
        plain sync code and from a thread that already runs an event loop
        (where ``asyncio.run`` would raise), while still delivering chunks as
        they are produced rather than after the whole answer is generated.
        """
        yield from _iterate_async_generator(
            self.astream(input, config, **kwargs), self._loop_runner
        )

    async def astream(  # type: ignore[override]
        self,
        input: RAGInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Asynchronously stream answer chunks.

        Yields the answer as ``str`` chunks only, never a :class:`RAGOutput`:
        to get the sources and metadata as well, call :meth:`ainvoke` instead.
        With ``processing.ignore_errors`` set, a failure yields the standard
        error message as a final chunk instead of raising.

        Retrieval, query processing and context building run once (the same
        steps ``ainvoke`` uses); only the answer-generation LLM is streamed.
        An empty context yields the standard no-data answer once without
        calling the LLM. On success the full answer is written to conversation
        memory exactly as ``ainvoke`` would. With a guardrail, a blocked query
        yields the blocked message once, and with ``check_output`` the answer
        is generated in full, screened, and yielded as one chunk.

        A streamed answer publishes its query metrics as ``ainvoke``'s do
        (latency to its end, ``query_failed`` on an exception), the query
        span and its log lines name its namespace and query id, and an
        exception it raises carries them (``tag_query_failure``). A stream
        the caller stops reading early (``aclose``) is no failure; a
        cancelled one, as a cancelled ``ainvoke``, records no metrics.
        """
        started = time.monotonic()
        # The speculative extractions its route did not read (see ainvoke).
        leftovers: list[asyncio.Future[Any]] = []
        outcome = _StreamOutcome(query_id=self._query_id())
        stopped: BaseException | None = None
        try:
            async with aclosing(
                self._astream(input, config, leftovers, outcome, **kwargs)
            ) as chunks:
                async for chunk in chunks:
                    yield chunk
        except BaseException as e:
            stopped = e
            if isinstance(e, Exception):
                tag_query_failure(e, outcome.query_id, outcome.trace_id)
            await discard(leftovers)
            raise
        finally:
            if stopped is None or isinstance(stopped, Exception | GeneratorExit):
                self._record_query_metrics(
                    started,
                    outcome.rag_input or input,
                    error=stopped if isinstance(stopped, Exception) else None,
                    error_class=outcome.error_class,
                    usage=outcome.meter.snapshot() if outcome.meter else None,
                    ran=outcome.ran,
                    metadata=outcome.metadata,
                )
        self._background_extractions.keep(leftovers, self._time_left(started))

    async def _astream(
        self,
        input: RAGInput,
        config: RunnableConfig | None,
        leftovers: list[asyncio.Future[Any]],
        outcome: _StreamOutcome,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """The answer chunks of :meth:`astream`; the speculative extractions
        the route did not read go to ``leftovers``, and what its metrics
        read to ``outcome``."""
        search_config = self.config.search
        deadline = _QueryDeadline(
            search_config.query_timeout_seconds,
            first_chunk_seconds=search_config.model_timeout_seconds,
        )
        # Admitted within the deadline (_query), its log lines under the
        # query's id.
        admission_tokens = bind_contextvars(query_id=outcome.query_id)
        try:
            rag_input, input_dict = await self._admit_query(input, deadline)
        finally:
            reset_contextvars(**admission_tokens)
        outcome.rag_input = rag_input
        chunks: list[str] = []
        delivered = False
        # A refused answer, like a refused query, writes no memory.
        answer_blocked = False
        guard_records: list[dict[str, Any]] = []
        try:
            # Bound for retrieval and context building only: they run before
            # the first yield, in one context. Each later step of the generator
            # may run in another (stream() resumes it in a new task), where the
            # binding would not hold and its reset would fail.
            tokens = bind_contextvars(**self._log_context(rag_input, outcome.query_id))
            leftovers_token = _LEFTOVER_EXTRACTIONS.set(leftovers)
            try:
                # Only the steps before the first yield are traced here, for the
                # same reason; the streamed answer call gets its own model span.
                with (
                    self._identity(rag_input),
                    observability.bind_attributes(namespace=rag_input.suffix),
                    observability.span(
                        "kg_rag.query",
                        self._query_span_attributes(rag_input, search=False),
                    ) as span,
                    # Its usage and degradation ledger, as _query's; the
                    # answer's later steps meter into it again (metering).
                    usage_scope() as outcome.meter,
                    query_scope(),
                ):
                    ledger = current_ledger()
                    ids = observability.current_trace_ids()
                    outcome.trace_id = ids[0] if ids else None
                    async with deadline.scope():
                        self._raise_missing_namespace(input_dict)
                        blocking = await self._screen_input(
                            rag_input, input_dict, guard_records
                        )
                        if blocking is None:
                            await self._admit_history(
                                rag_input, input_dict, guard_records
                            )
                            span.set_attributes(
                                _captured_content(input=rag_input.query)
                            )
                            state: dict[str, Any] = (
                                await self._stream_context_chain.ainvoke(
                                    input_dict, config
                                )
                            )
                            outcome.ran = state.get("resolved_strategy")
            finally:
                _LEFTOVER_EXTRACTIONS.reset(leftovers_token)
                reset_contextvars(**tokens)
            if blocking is not None:
                yield blocking.text
                return
            meter = outcome.meter
            assert meter is not None
            answer_runnable = self._answer_generation_step(state)
            # The answer's model span starts after the query span has ended,
            # in a trace of its own; its run metadata carries the session and
            # user so the tracing callback can still group it.
            answer_config = merge_configs(
                config,
                {
                    "metadata": {
                        **self._identity_metadata(rag_input),
                        "namespace": rag_input.suffix,
                    }
                },
            )
            # The first chunk (text or not: a thinking model streams its
            # reasoning first) must come within the model timeout of the call,
            # so a stalled stream fails early; every chunk within the query's
            # deadline. One deadline scope per await: the generator may resume
            # in another task after each yield.
            started: float | None = time.monotonic()
            stream = aiter(answer_runnable.astream(state, answer_config))
            # Holds back the start of the answer until the [NO_ANSWER] token
            # is ruled in or out (search.abstention); chunks stay raw.
            marker = (
                MarkerStreamFilter(strip=self.config.search.abstention.strip_marker)
                if self._reads_marker(state)
                else None
            )
            # Under on_insufficient: abstain only a decline is shown; an
            # answer the model wrote instead becomes the fixed reply.
            declining = marker is not None and self._declines_insufficient(state)
            while True:
                with metering(meter):
                    async with deadline.scope(started):
                        try:
                            chunk = await anext(stream)
                        except StopAsyncIteration:
                            break
                started = None
                text = str(chunk)
                if not text:
                    continue
                chunks.append(text)
                if not self._screen_outputs:
                    shown = marker.feed(text) if marker is not None else text
                    if declining and marker is not None and not marker.marked:
                        shown = ""
                    if shown:
                        delivered = True
                        yield shown
            state["answer"] = "".join(chunks)
            self._abstention_step(state)
            if self._screen_outputs:
                # Nothing was delivered yet: the user sees only screened text.
                with metering(meter):
                    async with deadline.scope():
                        verdict = await self._screen_output(
                            state["answer"],
                            guard_records,
                            grounding=(
                                self._grounding_context(state)
                                if self._ground_outputs
                                else None
                            ),
                        )
                state["answer"] = verdict.text
                answer_blocked = verdict.action is GuardAction.BLOCKED
                if verdict.text:
                    delivered = True
                    yield verdict.text
            elif marker is not None:
                rest = marker.finish()
                if declining and not marker.marked:
                    rest = ""
                if not delivered and not rest and state["answer"]:
                    # Only the token: the stripped answer's fixed reply.
                    rest = state["answer"]
                if rest:
                    delivered = True
                    yield rest
        except Exception as e:
            if not self.ignore_errors:
                logger.error(
                    "RAG stream failed for query %s: %s",
                    text_digest(rag_input.query),
                    error_digest(e),
                )
                if (named := named_model_error(e, self.config)) is not e:
                    raise named from e
                raise
            log_failure(
                logger,
                logging.ERROR,
                "RAG stream failed for query %s after %s chunk(s): %s",
                text_digest(rag_input.query),
                len(chunks),
                error=e,
            )
            # Counted as ainvoke's error answer is.
            outcome.error_class = classify(e)
            # Mirror ainvoke's degraded answer. If part of the answer was
            # already delivered, append the notice so the truncation is visible.
            yield f"\n\n{DEFAULT_ERROR_MESSAGE}" if delivered else DEFAULT_ERROR_MESSAGE
            return

        output = self._format_output_step(state)
        # The query's degraded steps, as _query's ledger reports them; its
        # degradation and abstention go to the query metrics.
        if ledger is not None:
            output.metadata.update(ledger.metadata())
        outcome.metadata = output.metadata
        if answer_blocked:
            return
        with metering(meter):
            async with deadline.scope():
                await self._save_memory(output, rag_input)


# The state key of the answer context's section sources, with the context
# they were built from (``GraphRAGChain._section_sources``).
_SECTION_SOURCES = "_section_sources"


class _InFlight(MutableMapping[tuple[str, ...], "asyncio.Future[Any]"]):
    """The query extractions in flight (thread-safe), keyed within the
    current ``extraction_cache_scope``: the ``memo`` of
    :func:`~unified_kg_rag.shared.utils.concurrency.single_flight`. A call
    in flight on another event loop is none of this loop's."""

    def __init__(self, lock: threading.Lock) -> None:
        self._futures: dict[tuple[str, ...], asyncio.Future[Any]] = {}
        self._lock = lock

    def __getitem__(self, key: tuple[str, ...]) -> asyncio.Future[Any]:
        with self._lock:
            found = self._futures[(_EXTRACTION_SCOPE.get(), *key)]
        if found.get_loop() is not asyncio.get_running_loop():
            raise KeyError(key)
        return found

    def __setitem__(self, key: tuple[str, ...], future: asyncio.Future[Any]) -> None:
        with self._lock:
            self._futures[(_EXTRACTION_SCOPE.get(), *key)] = future

    def __delitem__(self, key: tuple[str, ...]) -> None:
        with self._lock:
            del self._futures[(_EXTRACTION_SCOPE.get(), *key)]

    def __iter__(self) -> Iterator[tuple[str, ...]]:
        with self._lock:
            return iter(list(self._futures))

    def __len__(self) -> int:
        return len(self._futures)


class _ExtractionCache:
    """A bounded, thread-safe LRU of query extractions (immutable values),
    keyed within the current ``extraction_cache_scope``; a size of 0 keeps
    nothing.

    It also tracks the extractions in flight (``inflight``; ``None`` with a
    size of 0), so two queries of one scope that miss together, the
    interleaved arms of a run-eval repeat, make one call: the first pays
    for it, the others wait for its value and count a cache hit."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._entries: OrderedDict[tuple[str, ...], Any] = OrderedDict()
        self._lock = threading.Lock()
        self.inflight = _InFlight(self._lock) if size > 0 else None

    def get(self, key: tuple[str, ...]) -> Any:
        key = (_EXTRACTION_SCOPE.get(), *key)
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, key: tuple[str, ...], value: Any) -> None:
        if self._size <= 0:
            return
        key = (_EXTRACTION_SCOPE.get(), *key)
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._size:
                self._entries.popitem(last=False)


class _BackgroundExtractions:
    """The speculative extractions queries left running (thread-safe).

    A query does not wait for an extraction its route did not read: it
    keeps running, uncancelled, so its answer still fills the extraction
    cache, and its call counts in the query's usage from its start. Each is
    cancelled at its query's deadline (``search.query_timeout_seconds``), or
    by :meth:`cancel` / :meth:`acancel` when the chain closes, so none
    outlives either.
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Future[Any]] = set()
        self._lock = threading.Lock()

    def keep(self, tasks: list[asyncio.Future[Any]], within: float | None) -> None:
        """Keep ``tasks`` (on the running loop) until they end, cancelling
        them after ``within`` seconds (``None``: no limit)."""
        running = [task for task in tasks if not task.done()]
        for task in tasks:
            if task.done():
                _retrieve(task)
        if not running:
            return
        loop = asyncio.get_running_loop()
        for task in running:
            timer: asyncio.TimerHandle | None = None
            if within is not None and within <= 0:
                task.cancel()
            elif within is not None:
                timer = loop.call_later(within, task.cancel)
            with self._lock:
                self._tasks.add(task)
            task.add_done_callback(functools.partial(self._ended, timer=timer))

    def _ended(
        self, task: asyncio.Future[Any], *, timer: asyncio.TimerHandle | None
    ) -> None:
        if timer is not None:
            timer.cancel()
        with self._lock:
            self._tasks.discard(task)
        _retrieve(task)

    def _take(self) -> list[asyncio.Future[Any]]:
        with self._lock:
            tasks, self._tasks = list(self._tasks), set()
        return tasks

    def cancel(self) -> None:
        """Cancel every kept task, on its own loop (from any thread)."""
        for task in self._take():
            loop = task.get_loop()
            if loop.is_closed():
                continue
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                task.cancel()
            else:
                try:
                    loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:  # closed in between
                    pass

    async def acancel(self) -> None:
        """:meth:`cancel`, then wait for the kept tasks of the running loop."""
        loop = asyncio.get_running_loop()
        own = [task for task in self._snapshot() if task.get_loop() is loop]
        self.cancel()
        if own:
            await asyncio.gather(*own, return_exceptions=True)

    def _snapshot(self) -> list[asyncio.Future[Any]]:
        with self._lock:
            return list(self._tasks)


def _retrieve(task: asyncio.Future[Any]) -> None:
    """Read how ``task`` ended, so a failure is not logged as unretrieved
    (its caller is gone; ``shared.usage`` counted its call)."""
    if not task.cancelled():
        task.exception()


# The speculative extractions (``auto_speculative_extraction``) an auto query
# started that its route did not read, handed from ``_route_and_process_step``
# to the query entry point, which keeps them running in the background
# (``_BackgroundExtractions``; ``None`` outside one: the step awaits them).
_LEFTOVER_EXTRACTIONS: contextvars.ContextVar[list[asyncio.Future[Any]] | None] = (
    contextvars.ContextVar("kg_rag_leftover_extractions", default=None)
)

# The grounding context of the answer an ``_ainvoke`` call generates, handed
# from the chain's capture step (``None`` outside a grounded call).
_GROUNDING_CAPTURE: contextvars.ContextVar[list[GroundingContext | None] | None] = (
    contextvars.ContextVar("kg_rag_grounding_capture", default=None)
)

# The ``on_answer_chunk`` callback of the ``ainvoke`` call running in this
# context (``None`` outside one): the answer step streams to it.
_ANSWER_CHUNKS: contextvars.ContextVar[Callable[[str], None] | None] = (
    contextvars.ContextVar("kg_rag_answer_chunks", default=None)
)


class _Ledgered(Runnable[Any, Any]):
    """The composed runnable ``inner`` with a degradation ledger of its own
    when no query opened one (a caller driving ``GraphRAGChain.chain``
    directly), so its steps record what they answered without where its
    output reports it; within a query it records into the query's."""

    def __init__(self, inner: Runnable[Any, Any]) -> None:
        self.inner = inner

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        with self._ledger():
            return self.inner.invoke(input, config, **kwargs)

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        with self._ledger():
            return await self.inner.ainvoke(input, config, **kwargs)

    @staticmethod
    def _ledger() -> AbstractContextManager[Any]:
        return nullcontext() if current_ledger() is not None else degradation_scope()


def _streamed_to(
    answer: Runnable[Any, Any], sink: Callable[[str], None]
) -> Runnable[Any, Any]:
    """``answer`` run as a stream whose text chunks go to ``sink`` as they
    arrive; returns the whole text, as the unstreamed chain would."""

    async def run(state: dict[str, Any], config: RunnableConfig) -> str:
        parts: list[str] = []
        # A response cut at the output limit fails as it would unstreamed.
        token = STRICT_STREAM_TRUNCATION.set(True)
        try:
            async for chunk in answer.astream(state, config):
                text = str(chunk)
                if text:
                    parts.append(text)
                    sink(text)
        finally:
            STRICT_STREAM_TRUNCATION.reset(token)
        return "".join(parts)

    return RunnableLambda(run)


def _conversation_key(
    suffix: str | None, user_id: str | None, conversation_id: str
) -> ConversationKey:
    """The memory key of a query's conversation: one tenant's (index suffix)
    or user's conversation id never reaches another's history."""
    return ConversationKey(namespace_suffix(suffix), user_id, conversation_id)


def _screened_query(rag_input: RAGInput, metadata: dict[str, Any]) -> str | None:
    """The query as screened (masked, if the guardrail masked it), or
    ``None`` when the guardrail blocked it: a blocked query is not recorded."""
    input_blocked = any(
        record["source"] == GuardSource.INPUT.value
        and record["action"] == GuardAction.BLOCKED.value
        for record in metadata.get("guardrail") or []
    )
    return None if input_blocked else rag_input.query


def _check_chunks(texts: list[str], limit: int) -> list[list[int]]:
    """The positions of ``texts`` in consecutive chunks whose texts, joined
    by blank lines, hold at most ``limit`` characters; a longer text is a
    chunk of its own."""
    chunks: list[list[int]] = []
    used = 0
    for position, text in enumerate(texts):
        if chunks and used + 2 + len(text) <= limit:
            chunks[-1].append(position)
            used += 2 + len(text)
        else:
            chunks.append([position])
            used = len(text)
    return chunks


def _guard_record(source: GuardSource, verdict: GuardVerdict) -> dict[str, Any]:
    """One guardrail intervention as reported in ``RAGOutput.metadata``.

    An answer refused by the contextual grounding check (the guard says so
    in ``GroundingCheck.blocked``) carries the reason
    ``contextual_grounding``, so it reads apart from a content-policy
    refusal."""
    record: dict[str, Any] = {
        "source": source.value,
        "action": verdict.action.value,
        "policies": verdict.policies,
    }
    if verdict.grounding is not None and verdict.grounding.blocked:
        record["reason"] = "contextual_grounding"
    return record


def _captured_content(
    input: str | None = None, output: str | None = None
) -> dict[str, str]:
    """``input.value`` / ``output.value`` for the query span, redacted, when
    ``observability.capture_content`` is on; empty otherwise.

    Langfuse and Phoenix both read these names, and take a trace's input and
    output in their trace lists from its root span.
    """
    policy = observability.content_policy()
    if not policy.capture_content:
        return {}
    values = {"input.value": input, "output.value": output}
    return {key: policy.redact(text) for key, text in values.items() if text}


_T = TypeVar("_T")


# Upper bound on waiting for retrievers to close on another thread's loop.
_TEARDOWN_TIMEOUT_SECONDS = 30.0


def _close_retrievers(retrievers: list[BaseGraphRAGRetriever]) -> None:
    """Call each retriever's synchronous ``close`` (best-effort, never raises)."""
    for retriever in retrievers:
        close = getattr(retriever, "close", None)
        if close is None:
            continue
        try:
            close()
        except Exception as e:  # noqa: BLE001 - teardown must never raise
            logger.debug("Error closing retriever %r: %s", retriever, e)


class _BackgroundCloses:
    """One chain's retriever closes running on daemon threads.

    Per chain rather than process-wide, so closing (or collecting) one chain
    waits only for its own closes, never for another chain's slow ones.
    """

    def __init__(self) -> None:
        self._workers: set[threading.Thread] = set()
        self._lock = threading.Lock()

    def start(self, retrievers: list[BaseGraphRAGRetriever]) -> threading.Thread:
        """Start the synchronous closes of ``retrievers`` on a daemon thread."""

        def run() -> None:
            try:
                _close_retrievers(retrievers)
            finally:
                with self._lock:
                    self._workers.discard(threading.current_thread())

        worker = threading.Thread(
            target=run, name="kg-rag-retriever-close", daemon=True
        )
        # Registered and started under one lock hold, so join() never sees a
        # worker that is not yet started (joining one raises RuntimeError).
        with self._lock:
            self._workers.add(worker)
            try:
                worker.start()
            except BaseException:
                self._workers.discard(worker)
                raise
        return worker

    def join(self, timeout: float = _TEARDOWN_TIMEOUT_SECONDS) -> None:
        """Wait, up to ``timeout`` in all, for these closes to finish."""
        deadline = time.monotonic() + timeout
        with self._lock:
            workers = list(self._workers)
        for worker in workers:
            if worker is threading.current_thread():
                continue
            worker.join(timeout=max(0.0, deadline - time.monotonic()))


async def _aclose_retrievers(retrievers: list[BaseGraphRAGRetriever]) -> None:
    for retriever in retrievers:
        aclose = getattr(retriever, "aclose", None)
        if aclose is None:
            continue
        try:
            await aclose()
        except Exception as e:  # noqa: BLE001 - teardown must never raise
            logger.debug("Error closing retriever %r: %s", retriever, e)


def _take_cached(
    retriever_cache: dict[Any, BaseGraphRAGRetriever],
    strategy_cache: dict[Any, BaseSearchStrategy],
    lock: threading.RLock,
    loops: Collection[asyncio.AbstractEventLoop | None] | None = None,
) -> dict[asyncio.AbstractEventLoop | None, list[BaseGraphRAGRetriever]]:
    """Remove cached entries (of ``loops``, or all) grouped by their loop."""
    taken: dict[asyncio.AbstractEventLoop | None, list[BaseGraphRAGRetriever]] = {}
    with lock:
        for key in list(retriever_cache):
            if loops is None or key[1] in loops:
                taken.setdefault(key[1], []).append(retriever_cache.pop(key))
        # Strategies hold the retrievers, so they go with them.
        for skey in list(strategy_cache):
            if loops is None or skey[1] in loops:
                del strategy_cache[skey]
    return taken


def _release_cached_resources(
    retriever_cache: dict[Any, BaseGraphRAGRetriever],
    strategy_cache: dict[Any, BaseSearchStrategy],
    lock: threading.RLock,
    runner: "_LoopRunner | None",
    background: _BackgroundCloses | None = None,
) -> None:
    """Close every cached retriever, then stop the sync-API loop (never raises).

    Takes the chain's state rather than the chain, so it also serves as the
    chain's GC finalizer.
    """
    try:
        for loop, retrievers in _take_cached(
            retriever_cache, strategy_cache, lock
        ).items():
            GraphRAGChain._release_retrievers(
                retrievers, loop, wait=True, background=background
            )
        if background is not None:
            background.join()
    except Exception as e:  # noqa: BLE001 - teardown must never raise
        logger.debug("Error releasing the chain's retrievers: %s", e)
    finally:
        # Whatever happened above, in-flight calls are cancelled and the loop
        # stopped (stop() itself never raises).
        if runner is not None:
            runner.stop()


# An absolute deadline (``time.monotonic()``) a caller set for the queries it
# runs (``caller_deadline``): a query's own deadline is never later.
_CALLER_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "kg_rag_caller_deadline", default=None
)


def new_query_id() -> str:
    """A new query id (the ``query_id`` of a query's log lines, metrics and
    errors); a host that binds one before calling the chain (as
    ``query_id`` in the structlog context) has the chain use it."""
    return uuid.uuid4().hex[:12]


@contextlib.contextmanager
def caller_deadline(at: float | None) -> Iterator[None]:
    """Queries started in this context end by ``at`` (``time.monotonic()``),
    if it comes before their own ``search.query_timeout_seconds``.

    A surface that bounds a whole call (the MCP server: the wait for a call
    slot and the query under one deadline) passes its deadline here, so the
    chain's own timeout fires first and the query is recorded as a timed-out
    one (metrics, ids, the span's ``error_class``) instead of being cancelled
    from outside unrecorded.
    """
    token = _CALLER_DEADLINE.set(at)
    try:
        yield
    finally:
        _CALLER_DEADLINE.reset(token)


class _QueryDeadline:
    """The deadline of one query (``search.query_timeout_seconds``), fixed
    when the query starts; ``None`` seconds is no deadline. A caller's
    earlier deadline (:func:`caller_deadline`) wins.

    ``async with`` it bounds the awaits inside to the time left: past it the
    task is cancelled (and, through ``gather_or_cancel``, the calls it fans
    out) and :class:`QueryTimeoutError` is raised. No transient retry starts
    past it either, on worker threads too (``retry_deadline``). A streamed
    answer resumes in a new task per chunk, so it enters one scope per await
    (:meth:`scope`), never across a ``yield``.
    """

    def __init__(
        self, seconds: float | None, first_chunk_seconds: float | None = None
    ) -> None:
        self.seconds = seconds
        self.setting = "search.query_timeout_seconds"
        self.first_chunk_seconds = first_chunk_seconds
        now = time.monotonic()
        self.at = None if seconds is None else now + seconds
        caller = _CALLER_DEADLINE.get()
        if caller is not None and (self.at is None or caller < self.at):
            # The query's budget is the time its caller left it (after a
            # wait for a call slot), not the setting.
            self.at = caller
            self.seconds = max(0.0, caller - now)
            self.setting = "the caller's deadline"
        self._scope: _DeadlineScope | None = None

    def scope(self, first_chunk_from: float | None = None) -> "_DeadlineScope":
        """A scope bounded by the deadline and, given ``first_chunk_from``
        (the ``time.monotonic()`` a stream's model call started), by
        ``first_chunk_seconds`` after it: the wait for its first chunk."""
        first_chunk_at = None
        if first_chunk_from is not None and self.first_chunk_seconds is not None:
            first_chunk_at = first_chunk_from + self.first_chunk_seconds
        return _DeadlineScope(self, first_chunk_at)

    async def __aenter__(self) -> None:
        self._scope = self.scope()
        await self._scope.__aenter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        scope, self._scope = self._scope, None
        if scope is not None:
            await scope.__aexit__(exc_type, exc, tb)


class _DeadlineScope:
    """One ``async with`` window of a :class:`_QueryDeadline`."""

    def __init__(self, deadline: _QueryDeadline, first_chunk_at: float | None) -> None:
        self._deadline = deadline
        self._first_chunk_at = first_chunk_at
        self._timeout: asyncio.Timeout | None = None
        self._retry_scope: AbstractContextManager[None] | None = None
        self._first_chunk = False

    async def __aenter__(self) -> None:
        at = self._deadline.at
        if at is None:
            return
        now = time.monotonic()
        if self._first_chunk_at is not None and self._first_chunk_at < at:
            at, self._first_chunk = self._first_chunk_at, True
        self._retry_scope = retry_deadline(at)
        self._retry_scope.__enter__()
        self._timeout = asyncio.timeout(max(0.0, at - now))
        await self._timeout.__aenter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        timeout, self._timeout = self._timeout, None
        retry_scope, self._retry_scope = self._retry_scope, None
        if retry_scope is not None:
            retry_scope.__exit__(None, None, None)
        if timeout is None:
            return
        try:
            await timeout.__aexit__(exc_type, exc, tb)
        except TimeoutError as e:
            if not timeout.expired():
                raise
            raise self._expired() from e
        if isinstance(exc, Exception) and timeout.expired():
            # The cancellation came back as another error (a layer that
            # turned it into its own): still the deadline's doing.
            raise self._expired() from exc

    def _expired(self) -> QueryTimeoutError:
        if self._first_chunk:
            limit = self._deadline.first_chunk_seconds
            setting = "search.model_timeout_seconds"
            what = "The answer stream sent no first chunk"
        else:
            limit = self._deadline.seconds
            setting = self._deadline.setting
            what = "The query did not finish"
        assert limit is not None
        message = f"{what} within {round(limit, 2):g} s"
        logger.warning("%s (%s); cancelled its calls in flight", message, setting)
        return QueryTimeoutError(
            f"{message} ({setting}); its calls in flight were cancelled",
            timeout_seconds=limit,
        )


class _LoopStoppedError(StoreConnectionError):
    """A sync call was in flight when its chain's loop was stopped."""


class _LoopRunner:
    """One event loop on a daemon thread, started on first use.

    Backs the chain's synchronous entry points: each call submits its
    coroutine to this loop and blocks for the result, so every sync call
    shares one loop (and the clients bound to it) and none needs a loop of
    its own in the calling thread.
    """

    def __init__(self, io_workers: int | None = None) -> None:
        self._io_workers = io_workers
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Calls in flight, and those stop() cancelled (so run() can tell a
        # close from a cancellation inside the call).
        self._pending: set[concurrent.futures.Future[Any]] = set()
        self._stopped: set[concurrent.futures.Future[Any]] = set()

    def _ensure_loop_locked(self) -> asyncio.AbstractEventLoop:
        """The running loop, started first if needed (caller holds the lock)."""
        if self._loop is None:
            loop = asyncio.new_event_loop()
            if self._io_workers is not None:
                configure_event_loop(loop, self._io_workers)
            thread = threading.Thread(
                target=_run_loop, args=(loop,), name="graphrag-chain-loop", daemon=True
            )
            thread.start()
            self._loop, self._thread = loop, thread
        return self._loop

    def run(self, awaitable: Awaitable[_T]) -> _T:
        """Run ``awaitable`` on the loop and block for its result.

        Raises:
            StoreConnectionError: :meth:`stop` closed the loop while the call
                was in flight (the chain was closed under it).
        """

        async def _await() -> _T:
            return await awaitable

        # One lock hold from finding (or starting) the loop to registering the
        # call: stop() cannot land in between, so a call is never submitted
        # to a loop that has been stopped (where it would never finish).
        with self._lock:
            loop = self._ensure_loop_locked()
            if threading.current_thread() is self._thread:
                raise RuntimeError(
                    "A synchronous GraphRAGChain method was called from the "
                    "chain's own event loop; await the async variant instead."
                )
            future = asyncio.run_coroutine_threadsafe(_await(), loop)
            self._pending.add(future)
        try:
            return future.result()
        except concurrent.futures.CancelledError:
            if future in self._stopped:
                raise _LoopStoppedError(
                    "GraphRAGChain was closed while this call was in flight"
                ) from None
            raise
        finally:
            with self._lock:
                self._pending.discard(future)
                self._stopped.discard(future)

    def stop(self) -> None:
        """Stop and close the loop (idempotent; restarted on next use).

        Calls still in flight are cancelled first: their callers raise
        :class:`StoreConnectionError` instead of waiting forever on a loop
        that no longer runs.
        """
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
            pending, self._pending = self._pending, set()
            self._stopped |= pending
        if loop is None or thread is None:
            return
        # Wakes the waiting callers and cancels their tasks on the loop.
        for future in pending:
            future.cancel()
        if thread is threading.current_thread():
            # Stopping from inside the loop: _run_loop cancels the remaining
            # tasks and closes the loop once run_forever returns.
            loop.call_soon_threadsafe(loop.stop)
            return
        try:
            asyncio.run_coroutine_threadsafe(_cancel_other_tasks(), loop).result(
                timeout=_TEARDOWN_TIMEOUT_SECONDS
            )
        except Exception as e:  # noqa: BLE001 - teardown must never raise
            logger.debug("Error cancelling the chain loop's tasks: %s", e)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=_TEARDOWN_TIMEOUT_SECONDS)


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    """The loop thread: run ``loop`` until stopped, then end what is left on
    it and close it, as ``asyncio.run`` does. A stop from inside the loop
    returns before the tasks it cancelled have unwound; closing with them
    pending would destroy them mid-flight."""
    try:
        loop.run_forever()
    finally:
        try:
            loop.run_until_complete(_cancel_other_tasks())
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception as e:  # noqa: BLE001 - teardown must never raise
            logger.debug("Error ending the chain loop's tasks: %s", e)
        finally:
            loop.close()


async def _cancel_other_tasks() -> None:
    """Cancel every other task on the running loop and wait for them to end."""
    current = asyncio.current_task()
    tasks = [task for task in asyncio.all_tasks() if task is not current]
    for task in tasks:
        # Not twice: a second cancel would interrupt a task's own cleanup.
        if not task.cancelling():
            task.cancel()
    if tasks:
        await asyncio.wait(tasks, timeout=_TEARDOWN_TIMEOUT_SECONDS)


def _iterate_async_generator(
    agen: AsyncIterator[_T], runner: _LoopRunner | None = None
) -> Iterator[_T]:
    """Consume an async iterator from sync code, one item at a time.

    Each step runs on ``runner``'s loop (a private one when omitted), so it
    never collides with an event loop already running in the caller's thread.
    Closing the returned iterator early (``break``) closes the async iterator
    on that loop.
    """
    owned = runner is None
    loop_runner = runner if runner is not None else _LoopRunner()
    stopped = False
    try:
        while True:
            try:
                item = loop_runner.run(agen.__anext__())
            except StopAsyncIteration:
                return
            except _LoopStoppedError:
                # The close cancelled the step, which ended the iterator; an
                # aclose would only restart the stopped loop.
                stopped = True
                raise
            yield item
    finally:
        try:
            aclose = getattr(agen, "aclose", None)
            if aclose is not None and not stopped:
                loop_runner.run(aclose())
        finally:
            if owned:
                loop_runner.stop()
