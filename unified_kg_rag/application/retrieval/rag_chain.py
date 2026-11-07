# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import concurrent.futures
import contextvars
import functools
import json
import re
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
)
from contextlib import AbstractContextManager, aclosing, contextmanager
from typing import Any, Literal, TypeVar

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
from pydantic import BaseModel, Field, field_validator, model_validator
from structlog.contextvars import bind_contextvars, reset_contextvars

from unified_kg_rag.adapters.llm.chain_factory import (
    STRICT_STREAM_TRUNCATION,
    setup_chain,
)
from unified_kg_rag.adapters.llm.retry import retry_deadline
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
    query_scope,
    scope_filters,
)
from unified_kg_rag.adapters.retrieval.memory_manager import (
    HISTORY_ANSWER_CHARS,
    ConversationKey,
    MemoryManager,
    history_buffer,
)
from unified_kg_rag.adapters.retrieval.strategy_registry import (
    QueryInput,
    get_strategy_spec,
    query_inputs,
    wrapped_strategy,
)
from unified_kg_rag.adapters.retrieval.token_manager import (
    EMPTY_CONTEXT_PLACEHOLDER,
    ContextSection,
    OptimizedContext,
    SectionType,
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
from unified_kg_rag.adapters.storage.filter_schema import (
    ATTRIBUTE_KEY_PREFIX,
    check_filter_values,
    typed_filter_values,
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
from unified_kg_rag.domain.ingestion.cache_keys import index_language
from unified_kg_rag.domain.ingestion.seed_graph import (
    DOCUMENT_SOURCE_KIND,
    SEED_FILTER_KEY,
    SEED_SOURCE_KIND,
    SOURCE_KINDS_FIELD,
)
from unified_kg_rag.domain.models import (
    Citation,
    Config,
    Constants,
    IndexArtifact,
    LanguageCode,
    Methodology,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
)
from unified_kg_rag.domain.models.config import SearchConfig
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
from unified_kg_rag.domain.retrieval.abstention import (
    ABSTENTION_REASON_KEY,
    CONTEXT_SUFFICIENT_KEY,
    REASON_INSUFFICIENT_CONTEXT,
    REASON_MARKER,
    REASON_NO_CONTEXT,
    MarkerStreamFilter,
    split_marker,
)
from unified_kg_rag.domain.retrieval.index_prefixes import HELD_ARTIFACTS_KEY
from unified_kg_rag.domain.retrieval.no_context import (
    NO_CONTEXT_ANSWERS,
    is_no_context_answer,
    no_context_answer,
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
    MethodologyNotIndexedError,
    ToolArgumentError,
)
from unified_kg_rag.shared.genai import ERROR_TYPE, GEN_AI_CONVERSATION_ID
from unified_kg_rag.shared.usage import usage_scope
from unified_kg_rag.shared.utils import (
    configure_event_loop,
    error_digest,
    strip_embedding_fields,
    text_digest,
)
from unified_kg_rag.shared.utils.concurrency import discard, settle

logger = get_logger(__name__)


def _names(strategies: Collection[SearchStrategy]) -> str:
    """``strategies`` as a message lists them, in declaration order."""
    return ", ".join(s.value for s in SearchStrategy if s in strategies)


def _listed(value: Any) -> list[str]:
    """A stored list value as a list of strings: a store may return a
    one-element list as its value (Gremlin), and a graph edge holds a list
    as its JSON string."""
    if isinstance(value, str) and value.startswith("["):
        try:
            value = json.loads(value)
        except ValueError:
            return [value]
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple | set):
        return [str(v) for v in value if v is not None and v != ""]
    return []


def seed_provenance(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Where a result's statement comes from, when a seed graph states it.

    ``kind`` is ``seed_graph`` for an item only seed graphs state and
    ``document+seed_graph`` for one documents state as well; ``seed_graphs``
    names the graphs (their ``seed_graph`` document filter, where the store
    keeps the filters) and ``iri`` the entity's IRI. ``None`` for an item
    without seed evidence (no ``source_kinds``): its source is reported as
    before.
    """
    kinds = _listed(metadata.get(SOURCE_KINDS_FIELD))
    if SEED_SOURCE_KIND not in kinds:
        return None
    attributes = metadata.get("attributes")
    filters = (
        attributes.get(Constants.FILTERS.value)
        if isinstance(attributes, dict)
        else None
    )
    names = filters.get(SEED_FILTER_KEY) if isinstance(filters, dict) else None
    if names is None:
        names = metadata.get(f"{ATTRIBUTE_KEY_PREFIX}{SEED_FILTER_KEY}")
    iri = metadata.get("iri")
    return {
        "kind": (
            f"{DOCUMENT_SOURCE_KIND}+{SEED_SOURCE_KIND}"
            if DOCUMENT_SOURCE_KIND in kinds
            else SEED_SOURCE_KIND
        ),
        "seed_graphs": sorted(set(_listed(names))),
        "iri": iri if isinstance(iri, str) and iri else None,
    }


# The fixed reply when retrieval produced no context is no_context_answer's,
# in the query's language (NO_CONTEXT_ANSWER, the English one, is kept for
# importers); the output metadata then carries ``abstained=True``
# (``abstention_reason: no_context``) so evaluation can tell it from a real
# answer. Also the answer of a declined answer that was only the [NO_ANSWER]
# token (search.abstention.strip_marker).
NO_CONTEXT_ANSWER = NO_CONTEXT_ANSWERS[LanguageCode.EN]

# The search metadata of iterative search and its model-written text: the
# follow-up queries searched (``searched``), and each verdict's bridge
# entities and follow-up queries.
_ITERATIVE_KEY = "iterative"
_ITERATIVE_MODEL_TEXT: dict[str, tuple[str, ...]] = {
    "searched": (),
    "verdicts": ("bridge_entities", "next_queries"),
}


def _model_texts(iterative: dict[str, Any]) -> list[tuple[str, str]]:
    """The model-written strings of ``iterative`` metadata, as (field, text)."""
    found: list[tuple[str, str]] = []
    for key, inner in _ITERATIVE_MODEL_TEXT.items():
        value = iterative.get(key)
        if not isinstance(value, list):
            continue
        if not inner:
            found.extend((key, v) for v in value if isinstance(v, str) and v)
            continue
        for record in value:
            if not isinstance(record, dict):
                continue
            for field_name in inner:
                item = record.get(field_name)
                items = item if isinstance(item, list) else [item]
                found.extend((field_name, v) for v in items if isinstance(v, str) and v)
    return found


# The search metadata of the structured strategy, whose plan (model output:
# its names, values, entities and keywords) and plan description are
# returned with the search results and written into its query result.
_STRUCTURED_KEY = "structured"
_STRUCTURED_MODEL_TEXT = ("plan", "plan_description")


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


def _replace_model_texts(
    iterative: dict[str, Any], screened: dict[str, str | None]
) -> dict[str, Any]:
    """``iterative`` with each model-written string replaced by its screened
    text; a refused one (``None``) is left out of its list, or emptied."""

    def one(value: Any) -> Any:
        if isinstance(value, list):
            return [screened.get(v, v) for v in value if screened.get(v, v) is not None]
        if isinstance(value, str):
            text = screened.get(value, value)
            return "" if text is None else text
        return value

    out = dict(iterative)
    for key, inner in _ITERATIVE_MODEL_TEXT.items():
        value = iterative.get(key)
        if not isinstance(value, list):
            continue
        if not inner:
            out[key] = one(value)
            continue
        out[key] = [
            (
                {**r, **{f: one(r[f]) for f in inner if f in r}}
                if isinstance(r, dict)
                else r
            )
            for r in value
        ]
    return out


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
    (``RAGInput.history``)."""

    role: Literal["user", "assistant"] = Field(
        description="Who wrote it: the user, or the assistant (an earlier answer)"
    )
    content: str = Field(description="The message text")


class RAGInput(BaseModel):
    """One query to :class:`GraphRAGChain` (a dict with these keys also works).

    ``suffix`` selects the index namespace to search: it must match the
    ``processing.document_parsing.index_value`` the corpus was ingested with
    (both default to ``default``). Set ``conversation_id`` and ``use_memory``
    to continue a conversation; with ``use_memory`` and no id, a new id is
    generated and returned on :class:`RAGOutput`. A stateless caller passes
    the conversation itself as ``history`` instead (docs/design.md §7.2).
    """

    query: str = Field(description="The user's search query")
    suffix: str | None = Field(
        default=None, description="Suffix for multi-tenant or versioned indices"
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

    @field_validator("query")
    @classmethod
    def _compose_query(cls, query: str) -> str:
        # Indexed text is NFC (see convert_langchain_to_document); a query typed
        # or pasted as decomposed Hangul (NFD) must compose to match it.
        return unicodedata.normalize("NFC", query)

    @model_validator(mode="after")
    def _one_conversation_source(self) -> "RAGInput":
        if self.history is not None and self.use_memory:
            raise ValueError(
                "Pass the conversation as history or keep it in memory "
                "(use_memory), not both."
            )
        return self


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
    empty context), ``marker`` (the answer started with ``[NO_ANSWER]``,
    ``search.abstention.mode: mark``) or ``insufficient_context`` (the
    strategy flagged its context insufficient under
    ``search.abstention.on_insufficient: abstain``); a declined answer keeps
    the sources the model was given.
    ``citations`` (``search.citations``) lists each numbered section of the
    answer context, ``[n]`` in the answer, with the documents, pages and text
    units it rests on and whether the answer cites it; a number the answer
    uses that the context did not show is listed in
    ``metadata["unknown_citation_markers"]``. A blocked answer has none.
    An ``auto`` query's ``metadata["routing"]`` says how its strategy was
    chosen (``ROUTING_KEY``).
    """

    answer: str = Field(description="The generated answer to the user's query")
    sources: list[dict[str, Any]] = Field(
        description="List of source documents used to generate the answer"
    )
    search_results: SearchResult = Field(
        description="Detailed search results from the retrieval process"
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="The numbered sections of the answer context ([n]) and "
        "what each rests on (search.citations)",
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
        self._held_artifacts: dict[str, tuple[float, frozenset[IndexArtifact]]] = {}
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

        self._search_chain: Runnable[Any, Any] = base_chain | RunnableLambda(
            self._format_search_output_step
        )
        return base_chain | rag_branch

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
            strategy = self._parse_routed_strategy(selected_strategy_str, routable)
            decided_by = (
                "router"
                if self._names_strategy(selected_strategy_str, strategy)
                else "fallback"
            )
        except Exception as e:
            if not self.ignore_errors:
                raise
            strategy = self._fallback_strategy(routable)
            decided_by = "fallback"
            logger.warning(
                "Strategy auto-selection failed, using '%s' as fallback: %s",
                strategy.value,
                error_digest(e),
            )
            logger.debug("Strategy auto-selection error: %s", e)

        state["resolved_strategy"] = strategy
        state[ROUTING_KEY] = {"decided_by": decided_by, "candidates": candidates}
        return state

    def _check_allowed(self, strategy: SearchStrategy) -> None:
        """Refuse ``strategy`` outside ``allowed_strategies``, before any
        model call of the query."""
        allowed = getattr(self, "allowed_strategies", None)
        if allowed is not None and strategy not in allowed:
            raise ToolArgumentError(
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
        namespace = (
            suffix
            or self.config.processing.document_parsing.index_value
            or Constants.DEFAULT_SUFFIX.value
        )
        raise MethodologyNotIndexedError(
            f"auto may run only {_names(allowed)}, and none of them serves "
            f"the index namespace '{namespace}' (with these filters auto "
            f"could route to {_names(routable)}); pick one it serves, or "
            "ingest the methodology an allowed strategy reads."
        )

    @staticmethod
    def _names_strategy(raw: str, strategy: SearchStrategy) -> bool:
        """Whether the router's reply ``raw`` named ``strategy`` (rather than
        ``_parse_routed_strategy`` falling back to it)."""
        return strategy.value in re.findall(r"[a-z]+", (raw or "").lower())

    @staticmethod
    def _parse_routed_strategy(
        raw: str, routable: tuple[SearchStrategy, ...] | None = None
    ) -> SearchStrategy:
        """Map a router LLM response to a routable strategy, tolerating extra text.

        The router is asked for a bare word, but LLMs add punctuation/prose
        ("Local search.", "I'd use local"). The response is split into word
        tokens and the FIRST token naming a routable strategy wins, so the
        result depends on what the model wrote, not on the order strategies
        are listed in (a substring scan picked whichever name it checked
        first). Defaults to :meth:`_fallback_strategy`.
        """
        allowed = routable or tuple(SearchConfig().auto_routable_strategies)
        by_value = {strat.value: strat for strat in allowed}
        for token in re.findall(r"[a-z]+", (raw or "").lower()):
            if token in by_value:
                return by_value[token]
        fallback = GraphRAGChain._fallback_strategy(allowed)
        logger.warning(
            "Router returned unrecognized strategy (%s); defaulting to '%s'",
            text_digest(raw),
            fallback.value,
        )
        logger.debug("Unrecognized router output: '%s'", raw)
        return fallback

    @staticmethod
    def _fallback_strategy(routable: tuple[SearchStrategy, ...]) -> SearchStrategy:
        """The strategy AUTO uses when routing fails: LOCAL, a general-purpose
        graph strategy and not the costliest one, when it is routable; else
        the first routable strategy, so a configuration that excludes LOCAL
        (its stores may not be provisioned) is never routed to it."""
        if not routable or SearchStrategy.LOCAL in routable:
            return SearchStrategy.LOCAL
        return routable[0]

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
    ) -> str:
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
            if not self.ignore_errors:
                raise
            logger.warning("Query processing failed: %s", error_digest(e))
            logger.debug("Query processing error: %s", e)
            return ProcessedQuery(
                original_query=original_query,
                standalone_query=standalone_query,
                final_query=question,
            )

    @staticmethod
    def _looks_like_translation_refusal(candidate: str, original: str) -> bool:
        """``query_processing.looks_like_translation_refusal``."""
        return query_processing.looks_like_translation_refusal(candidate, original)

    async def _extract_query_entities(
        self, query: str, target_language: str, config: RunnableConfig | None
    ) -> list[str]:
        """``query_processing.extract_query_entities``."""
        return await query_processing.extract_query_entities(
            self, query, target_language, config
        )

    def _extraction_key(
        self, prompt_class: type[BasePrompt], query: str, target_language: str
    ) -> tuple[str, ...]:
        """``query_processing.extraction_key``."""
        return query_processing.extraction_key(
            self, prompt_class, query, target_language
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
    ) -> tuple[list[str], list[str]]:
        """``query_processing.extract_dual_keywords``."""
        return await query_processing.extract_dual_keywords(
            self, query, target_language, config
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
            if not self.ignore_errors:
                raise
            logger.warning(
                "Follow-up rewrite failed; using the query as is: %s", error_digest(e)
            )
            logger.debug("Follow-up rewrite error: %s", e)
            return None
        return self._usable_rewrite(str(rewritten))

    @staticmethod
    def _usable_rewrite(rewritten: str) -> str | None:
        """``query_processing.usable_rewrite``."""
        return query_processing.usable_rewrite(rewritten)

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
        # What the namespace holds (cached since the methodology check): a
        # strategy reads only those of its optional indexes it holds.
        held = await methodology_check.namespace_artifacts(self, state.get("suffix"))
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
            # kept past it.
            with query_scope():
                result = await strategy_instance.asearch(search_query, config=config)
            # A failure ignore_errors absorbed still degraded the answer.
            if degraded := result.metadata.get("degraded_error"):
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
        ``MethodologyNotIndexedError`` if ``strategy`` reads another one
        (``auto`` routes only among the held ones). For callers that check a
        strategy once before many queries (an interactive session)."""
        return await methodology_check.check_strategy_methodology(
            self, strategy, suffix
        )

    async def aread_build_record(self, suffix: str | None) -> dict[str, Any] | None:
        """The build record ingestion wrote on the index namespace of
        ``suffix`` (``VectorIndexer.record_build``: what it built, under which
        settings), read through the document retriever the answers come
        from. ``None`` when it records none (built before build records, or a
        backend without index metadata); a store error raises."""
        return await self._get_retriever(RetrieverRole.DOCUMENT).aread_build_record(
            suffix
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

    def close(self) -> None:
        """Synchronous teardown of cached retrievers and the sync-API loop."""
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
            number_sections=self.config.search.citations.enabled,
        )

    def _context_optimization_step(self, state: dict[str, Any]) -> OptimizedContext:
        """Select the budgeted sections once and keep them in chain state.

        Both the context string and the reported ``sources`` are built from this
        selection, so sources reflect exactly what the answer model was given.
        """
        try:
            return self._optimize_context(state)
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.warning("Context optimization failed: %s", error_digest(e))
            logger.debug("Context optimization error: %s", e, exc_info=True)
            return OptimizedContext(
                sections=[],
                total_tokens=0,
                sections_included=0,
                sections_excluded=len(state["search_results"].results),
            )

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

    def _no_context_answer(self, state: dict[str, Any]) -> str:
        """The fixed decline, in the language of the question (the index
        language when the question does not tell), as the answer prompt
        writes in the query's language."""
        return no_context_answer(self._question(state), index_language(self.config))

    def _abstention_step(self, state: dict[str, Any]) -> dict[str, Any]:
        """Set ``state["abstention_reason"]`` (``None`` for an answer) and
        strip the [NO_ANSWER] token per ``search.abstention.strip_marker``."""
        if self._context_is_empty(state):
            state["abstention_reason"] = REASON_NO_CONTEXT
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
    def _context_citations(state: dict[str, Any]) -> list[Citation]:
        """One citation per numbered section of the answer context, in
        marker order (documents of graph sections not yet resolved)."""
        optimized: OptimizedContext | None = state.get("optimized_context")
        if optimized is None or GraphRAGChain._context_is_empty(state):
            return []
        results: list[RetrievalResult] = state["search_results"].results
        cited: list[Citation] = []
        for section in sorted(
            (s for s in optimized.sections if s.marker is not None),
            key=lambda s: s.marker or 0,
        ):
            entry = GraphRAGChain._section_source(section, results)
            assert section.marker is not None
            cited.append(citations.citation_from_entry(section.marker, entry))
        return cited

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
        if read_state.get("citations_unresolved"):
            answered["citations_unresolved"] = True
        return answered

    async def _citations_step(self, state: dict[str, Any]) -> dict[str, Any]:
        """Set ``state["citations"]``: the context's citations, the
        documents of graph sections read from their lineage
        (``citations.aresolve_documents``). A failed read leaves those
        without documents (``citations_unresolved``); the answer stands."""
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
            logger.warning(
                "Could not read the text units of the answer's graph "
                "citations; they name no document: %s",
                error_digest(e),
            )
            logger.debug("Citation lineage read error", exc_info=True)
            state["citations_unresolved"] = True
        return state

    async def _fetch_text_units(
        self, state: dict[str, Any], unit_ids: list[str]
    ) -> list[dict[str, Any]]:
        """The stored text units of ``unit_ids`` (id, documents, attributes)
        in the query's namespace that pass the caller's filters the
        text-units index declares: a unit they exclude is never named."""
        retriever = self._get_retriever(RetrieverRole.DOCUMENT)
        prefix = self.config.indexing.opensearch.text_units_index_prefix
        suffix = state.get("suffix")
        declared = retriever.filter_fields_for([prefix])
        filters = {
            key: value
            for key, value in (state.get("filters") or {}).items()
            if declared is None or declared.declares(key)
        }
        limits = await retriever.aindex_limits(prefix, suffix)
        size = max(1, limits.id_batch_size)
        found: list[dict[str, Any]] = []
        for start in range(0, len(unit_ids), size):
            batch = unit_ids[start : start + size]
            hits = await retriever.aretrieve(
                SearchQuery(
                    query="",
                    search_type=SearchType.LEXICAL,
                    top_k=len(batch),
                    index_prefixes=[prefix],
                    suffix=suffix,
                    filters=scope_filters(filters, id=batch),
                    source_fields=["id", "document_ids", "attributes"],
                )
            )
            found.extend(hit.metadata or {} for hit in hits)
        return found

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
                optimized.sections, sr.results
            )

        metadata = {
            "search_strategy": sr.search_strategy,
            "processing_time": time.time() - state["start_time"],
            "total_results": len(sr.results),
            **sr.metadata,
            **({ROUTING_KEY: state[ROUTING_KEY]} if ROUTING_KEY in state else {}),
        }
        reason = state.get("abstention_reason")
        if reason is None and GraphRAGChain._context_is_empty(state):
            # The answer step returned the fixed no-context reply.
            reason = REASON_NO_CONTEXT
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
        if state.get("citations_unresolved"):
            metadata["citations_unresolved"] = True

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
    def _is_synthesized(metadata: dict[str, Any] | None) -> bool:
        return bool((metadata or {}).get("synthesized"))

    @staticmethod
    def _source_entry(
        result: RetrievalResult, content: str, *, truncated: bool
    ) -> dict[str, Any]:
        """One reported source: the text the model saw plus its provenance.

        Embedding vectors are dropped (any retriever, not only OpenSearch, may
        carry them in metadata). Provenance keys are added without overwriting
        retriever-supplied metadata of the same name. A statement a seed
        graph makes adds ``provenance`` (:func:`seed_provenance`); it has no
        chunk, so its ``chunk_id`` stays ``None``. A source without seed
        evidence gets no ``provenance`` key.
        """
        metadata = strip_embedding_fields(result.metadata or {})
        raw_doc_ids = metadata.get("document_ids") or metadata.get("document_id")
        document_ids = (
            [str(d) for d in raw_doc_ids]
            if isinstance(raw_doc_ids, list | tuple | set)
            else [str(raw_doc_ids)] if raw_doc_ids else []
        )
        chunk_id = result.chunk_id
        if chunk_id is None and result.retriever_type == SectionType.TEXT.value:
            chunk_id = metadata.get("id") or result.source
        provenance: dict[str, Any] = {
            "source_id": result.source,
            "document_ids": document_ids,
            "chunk_id": chunk_id,
            "section_type": result.retriever_type,
            "score": result.score,
        }
        if (seeded := seed_provenance(metadata)) is not None:
            provenance["provenance"] = seeded
        for key, value in provenance.items():
            metadata.setdefault(key, value)
        metadata["truncated"] = truncated
        return {
            "content": content,
            "source": result.source,
            "score": result.score,
            "metadata": metadata,
        }

    @staticmethod
    def _sources_from_sections(
        sections: list[ContextSection], results: list[RetrievalResult]
    ) -> list[dict[str, Any]]:
        """Map the budgeter's selected sections back to their results.

        Reported in retrieval-rank order (the selection itself is grouped by
        section type). A section without a usable ``result_index`` (a custom
        token manager) is reported from the section alone. Synthesized results
        are skipped (see ``_format_output_step``).
        """

        def _rank(section: ContextSection) -> int:
            index = section.result_index
            return index if index is not None else len(results)

        sources: list[dict[str, Any]] = []
        for section in sorted(sections, key=_rank):
            if GraphRAGChain._is_synthesized(
                GraphRAGChain._section_result(section, results).metadata
            ):
                continue
            sources.append(GraphRAGChain._section_source(section, results))
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
            retriever_type=section.section_type.value,
            metadata=section.metadata,
        )

    @staticmethod
    def _section_source(
        section: ContextSection, results: list[RetrievalResult]
    ) -> dict[str, Any]:
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
        rag_input, input_dict = self._prepare_invoke(input)
        await self.acheck_methodology(rag_input.search_strategy, rag_input.suffix)
        # What auto routes among (cached since the check).
        input_dict["held_artifacts"] = await methodology_check.namespace_artifacts(
            self, rag_input.suffix
        )
        # Filter keys after the methodology check, against the indexes the
        # namespace holds: a key only an index it lacks declares would be
        # dropped everywhere.
        filter_validation.check_requested_filter_keys(
            self, rag_input, input_dict["held_artifacts"]
        )
        # The root span of a query trace: the strategy, retrieval, fusion,
        # rerank, context build and model calls nest under it.
        with (
            self._identity(rag_input),
            observability.span(
                "kg_rag.query", self._query_span_attributes(rag_input, search)
            ) as span,
            usage_scope() as meter,
        ):
            started = time.monotonic()
            leftovers: list[asyncio.Future[Any]] = []
            token = _LEFTOVER_EXTRACTIONS.set(leftovers)
            try:
                result = await self._ainvoke(
                    rag_input, input_dict, config, span, search
                )
            except BaseException:
                await discard(leftovers)
                raise
            finally:
                _LEFTOVER_EXTRACTIONS.reset(token)
            # Not waited for: counted from their start below, at an
            # estimate until they answer.
            self._background_extractions.keep(leftovers, self._time_left(started))
            # The query's model calls and tokens (shared.usage).
            result.metadata["usage"] = meter.snapshot()
            span.set_attributes(
                {
                    **self._query_outcome_attributes(result),
                    **_captured_content(
                        input=_screened_query(rag_input, result.metadata),
                        output=result.answer if isinstance(result, RAGOutput) else None,
                    ),
                }
            )
            return result

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
    ) -> RAGOutput | SearchOutput:
        tokens = bind_contextvars(
            query_id=uuid.uuid4().hex[:12], conversation_id=rag_input.conversation_id
        )

        guard_records: list[dict[str, Any]] = []
        capture: list[GroundingContext | None] = []
        capture_token = _GROUNDING_CAPTURE.set(
            capture if self._ground_outputs else None
        )
        try:
            async with _QueryDeadline(self.config.search.query_timeout_seconds):
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
                if isinstance(output, RAGOutput):
                    verdict = await self._screen_output(
                        output.answer,
                        guard_records,
                        grounding=capture[-1] if capture else None,
                    )
                    screened = verdict.text != output.answer
                    output.answer = verdict.text
                    if verdict.grounding is not None:
                        output.metadata["grounding"] = verdict.grounding.model_dump()
                    if verdict.action is GuardAction.BLOCKED:
                        # A refused answer keeps none of the evidence behind it.
                        output.sources = []
                        output.citations = []
                        output.metadata.pop("unknown_citation_markers", None)
                        output.search_results = output.search_results.model_copy(
                            update={"results": [], "total_results": 0}
                        )
                    elif screened:
                        self._mark_citations(output)
                output.search_results = await self._screen_synthesized(
                    output.search_results, guard_records
                )
                if isinstance(output, RAGOutput) and _ITERATIVE_KEY in output.metadata:
                    # The answer's copy of the search metadata, screened too.
                    output.metadata[_ITERATIVE_KEY] = (
                        output.search_results.metadata.get(_ITERATIVE_KEY)
                    )
                if guard_records:
                    output.metadata["guardrail"] = guard_records
                if isinstance(output, RAGOutput):
                    await self._save_memory(output, rag_input)
                return output
        except Exception as e:
            if not self.ignore_errors:
                raise

            logger.error(
                "RAG chain execution failed for query %s: %s",
                text_digest(rag_input.query),
                error_digest(e),
            )
            logger.debug("RAG chain execution error: %s", e)
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
                # The error's type, for a caller that tells a refused query
                # (a strategy or filter to change, the deadline) apart.
                metadata={"error": True, "error_type": type(e).__name__},
                search_metadata={"error": str(e)},
            )
        finally:
            _GROUNDING_CAPTURE.reset(capture_token)
            reset_contextvars(**tokens)

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
        self, search_results: SearchResult, records: list[dict[str, Any]]
    ) -> SearchResult:
        """``search_results`` with each model-written result (``synthesized``:
        DRIFT's intermediate answers, global search's key points or summary)
        screened as an answer is: masked, or dropped when refused.

        They are returned with the search results (``ainvoke`` and
        ``asearch``), so with ``check_output`` they must not reach the caller
        unscreened. Retrieved results are corpus text and pass as they are.
        """
        if not self._screen_outputs:
            return search_results
        search_results = await self._screen_model_metadata(search_results, records)
        if not any(self._is_synthesized(r.metadata) for r in search_results.results):
            return search_results
        assert self.guard is not None
        kept: list[RetrievalResult] = []
        for result in search_results.results:
            if not self._is_synthesized(result.metadata) or not result.content:
                kept.append(result)
                continue
            verdict = await self.guard.acheck(result.content, GuardSource.OUTPUT)
            if not verdict.intervened:
                kept.append(result)
                continue
            records.append(_guard_record(GuardSource.OUTPUT, verdict))
            if verdict.action is not GuardAction.BLOCKED:
                kept.append(result.model_copy(update={"content": verdict.text}))
        return search_results.model_copy(
            update={"results": kept, "total_results": len(kept)}
        )

    async def _screen_model_metadata(
        self, search_results: SearchResult, records: list[dict[str, Any]]
    ) -> SearchResult:
        """``search_results`` with the model-written text of its metadata
        (``iterative``'s follow-up queries, bridge entities and step
        questions, ``_ITERATIVE_MODEL_TEXT``) screened as a synthesized
        result is: masked, or left out when refused. One check reads them
        all; only when it intervenes is each checked on its own."""
        search_results = await self._screen_structured_metadata(search_results, records)
        iterative = search_results.metadata.get(_ITERATIVE_KEY)
        if not isinstance(iterative, dict):
            return search_results
        texts = [text for _, text in _model_texts(iterative)]
        if not texts:
            return search_results
        assert self.guard is not None
        verdict = await self.guard.acheck("\n".join(texts), GuardSource.OUTPUT)
        if not verdict.intervened:
            return search_results
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
        return search_results.model_copy(
            update={
                "metadata": {
                    **search_results.metadata,
                    _ITERATIVE_KEY: _replace_model_texts(iterative, screened),
                }
            }
        )

    async def _screen_structured_metadata(
        self, search_results: SearchResult, records: list[dict[str, Any]]
    ) -> SearchResult:
        """``search_results`` with the structured strategy's model-written
        plan text screened as a synthesized result is, in its metadata and
        in the query result that renders it: masked, or removed when
        refused. One check reads them all; only when it intervenes is each
        checked on its own."""
        structured = search_results.metadata.get(_STRUCTURED_KEY)
        if not isinstance(structured, dict):
            return search_results
        texts = list(
            dict.fromkeys(
                text
                for key in _STRUCTURED_MODEL_TEXT
                for text in _string_leaves(structured.get(key))
            )
        )
        if not texts:
            return search_results
        assert self.guard is not None
        verdict = await self.guard.acheck("\n".join(texts), GuardSource.OUTPUT)
        if not verdict.intervened:
            return search_results
        screened: dict[str, str | None] = {}
        for text in texts:
            one = await self.guard.acheck(text, GuardSource.OUTPUT)
            if one.intervened:
                records.append(_guard_record(GuardSource.OUTPUT, one))
            screened[text] = (
                None
                if one.intervened and one.action is GuardAction.BLOCKED
                else (one.text if one.intervened else text)
            )
        changed = {t: r for t, r in screened.items() if r != t}
        results = []
        for result in search_results.results:
            if result.retriever_type != SectionType.QUERY_RESULT.value:
                results.append(result)
                continue
            content = result.content
            # Longest first, so a text inside a longer one is not cut first.
            for text in sorted(changed, key=len, reverse=True):
                content = content.replace(text, changed[text] or "[removed]")
            results.append(
                result.model_copy(
                    update={
                        "content": content,
                        "metadata": _replace_leaves(result.metadata, screened),
                    }
                )
            )
        return search_results.model_copy(
            update={
                "results": results,
                "metadata": {
                    **search_results.metadata,
                    _STRUCTURED_KEY: {
                        **structured,
                        **{
                            key: _replace_leaves(structured.get(key), screened)
                            for key in _STRUCTURED_MODEL_TEXT
                        },
                    },
                },
            }
        )

    @staticmethod
    def _prepare_invoke(
        inputs: RAGInput | dict[str, Any],
    ) -> tuple[RAGInput, dict[str, Any]]:
        # A copy: the generated conversation id and a masked query must not
        # change the caller's object.
        rag_input = (
            RAGInput(**inputs) if isinstance(inputs, dict) else inputs.model_copy()
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
            logger.error(
                "Failed to save conversation memory for conversation '%s': %s",
                output.conversation_id,
                error_digest(e),
            )
            logger.debug("Conversation memory error: %s", e, exc_info=True)

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
        """
        started = time.monotonic()
        # The speculative extractions its route did not read (see ainvoke).
        leftovers: list[asyncio.Future[Any]] = []
        try:
            async with aclosing(
                self._astream(input, config, leftovers, **kwargs)
            ) as chunks:
                async for chunk in chunks:
                    yield chunk
        except BaseException:
            await discard(leftovers)
            raise
        self._background_extractions.keep(leftovers, self._time_left(started))

    async def _astream(
        self,
        input: RAGInput,
        config: RunnableConfig | None,
        leftovers: list[asyncio.Future[Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """The answer chunks of :meth:`astream`; the speculative extractions
        the route did not read go to ``leftovers``."""
        rag_input, input_dict = self._prepare_invoke(input)
        await self.acheck_methodology(rag_input.search_strategy, rag_input.suffix)
        # What auto routes among (cached since the check).
        input_dict["held_artifacts"] = await methodology_check.namespace_artifacts(
            self, rag_input.suffix
        )
        # Filter keys after the methodology check, against the indexes the
        # namespace holds: a key only an index it lacks declares would be
        # dropped everywhere.
        filter_validation.check_requested_filter_keys(
            self, rag_input, input_dict["held_artifacts"]
        )
        search_config = self.config.search
        deadline = _QueryDeadline(
            search_config.query_timeout_seconds,
            first_chunk_seconds=search_config.model_timeout_seconds,
        )
        chunks: list[str] = []
        delivered = False
        guard_records: list[dict[str, Any]] = []
        try:
            # Bound for retrieval and context building only: they run before
            # the first yield, in one context. Each later step of the generator
            # may run in another (stream() resumes it in a new task), where the
            # binding would not hold and its reset would fail.
            tokens = bind_contextvars(
                query_id=uuid.uuid4().hex[:12],
                conversation_id=rag_input.conversation_id,
            )
            leftovers_token = _LEFTOVER_EXTRACTIONS.set(leftovers)
            try:
                # Only the steps before the first yield are traced here, for the
                # same reason; the streamed answer call gets its own model span.
                with (
                    self._identity(rag_input),
                    observability.span(
                        "kg_rag.query",
                        self._query_span_attributes(rag_input, search=False),
                    ) as span,
                ):
                    async with deadline.scope():
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
            finally:
                _LEFTOVER_EXTRACTIONS.reset(leftovers_token)
                reset_contextvars(**tokens)
            if blocking is not None:
                yield blocking.text
                return
            answer_runnable = self._answer_generation_step(state)
            # The answer's model span starts after the query span has ended,
            # in a trace of its own; its run metadata carries the session and
            # user so the tracing callback can still group it.
            answer_config = merge_configs(
                config, {"metadata": self._identity_metadata(rag_input)}
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
                raise
            logger.error(
                "RAG stream failed for query %s after %s chunk(s): %s",
                text_digest(rag_input.query),
                len(chunks),
                error_digest(e),
            )
            logger.debug("RAG stream error: %s", e)
            # Mirror ainvoke's degraded answer. If part of the answer was
            # already delivered, append the notice so the truncation is visible.
            yield f"\n\n{DEFAULT_ERROR_MESSAGE}" if delivered else DEFAULT_ERROR_MESSAGE
            return

        async with deadline.scope():
            await self._save_memory(self._format_output_step(state), rag_input)


# The query extraction cache's scope (``extraction_cache_scope``): part of
# every entry's key, so the queries of one scope share no extraction with
# those of another ("" outside one).
_EXTRACTION_SCOPE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "kg_rag_extraction_scope", default=""
)


@contextmanager
def extraction_cache_scope(scope: str) -> Iterator[None]:
    """Key the query extractions cached inside the block by ``scope`` too.

    ``run-eval --repeats`` answers each repeat in a scope of its own, so a
    repeat extracts the query's entities and keywords again instead of
    reusing the first repeat's (the repeats then sample the extraction's
    run-to-run variance), while the arms of one repeat still share them.
    Tasks started inside the block inherit it.
    """
    token = _EXTRACTION_SCOPE.set(scope)
    try:
        yield
    finally:
        _EXTRACTION_SCOPE.reset(token)


class _ExtractionCache:
    """A bounded, thread-safe LRU of query extractions (immutable values),
    keyed within the current ``extraction_cache_scope``; a size of 0 keeps
    nothing."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._entries: OrderedDict[tuple[str, ...], Any] = OrderedDict()
        self._lock = threading.Lock()

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
    return ConversationKey(
        suffix or Constants.DEFAULT_SUFFIX.value, user_id, conversation_id
    )


def _screened_query(rag_input: RAGInput, metadata: dict[str, Any]) -> str | None:
    """The query as screened (masked, if the guardrail masked it), or
    ``None`` when the guardrail blocked it: a blocked query is not recorded."""
    input_blocked = any(
        record["source"] == GuardSource.INPUT.value
        and record["action"] == GuardAction.BLOCKED.value
        for record in metadata.get("guardrail") or []
    )
    return None if input_blocked else rag_input.query


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


class _QueryDeadline:
    """The deadline of one query (``search.query_timeout_seconds``), fixed
    when the query starts; ``None`` seconds is no deadline.

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
        self.first_chunk_seconds = first_chunk_seconds
        self.at = None if seconds is None else time.monotonic() + seconds
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
            setting = "search.query_timeout_seconds"
            what = "The query did not finish"
        assert limit is not None
        message = f"{what} within {limit:g} s"
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
