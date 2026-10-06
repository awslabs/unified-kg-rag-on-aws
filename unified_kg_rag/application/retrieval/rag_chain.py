# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import re
import threading
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from enum import Enum
from typing import Any, ClassVar, TypeVar

import boto3
from langchain_core.output_parsers import (
    BaseOutputParser,
    CommaSeparatedListOutputParser,
    StrOutputParser,
)
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)
from pydantic import BaseModel, Field
from structlog.contextvars import bind_contextvars, reset_contextvars

from unified_kg_rag.adapters.aws import NeptuneClient, OpenSearchClient
from unified_kg_rag.adapters.aws.chain_factory import setup_chain
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
)
from unified_kg_rag.adapters.retrieval.memory_manager import (
    MemoryManager,
    get_memory_manager,
)
from unified_kg_rag.adapters.retrieval.token_manager import (
    EMPTY_CONTEXT_PLACEHOLDER,
    ContextSection,
    OptimizedContext,
    SectionType,
    TokenManager,
)
from unified_kg_rag.adapters.retrievers import NeptuneRetriever, OpenSearchRetriever

# Importing the package executes each strategy module's @register_strategy
# decorator, populating the strategy registry used by _get_strategy_instance.
from unified_kg_rag.adapters.search_strategies import (  # noqa: F401
    DriftSearchStrategy,
    GlobalSearchStrategy,
    LightRAGSearchStrategy,
    LocalSearchStrategy,
    SimpleSearchStrategy,
)
from unified_kg_rag.adapters.storage.filter_schema import union_filter_fields
from unified_kg_rag.domain.models import (
    Config,
    MessageRole,
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
    ContextBuildingPrompt,
    EntityExtractionPrompt,
    KeywordsExtractionPrompt,
    StrategySelectionPrompt,
    TranslationPrompt,
)
from unified_kg_rag.domain.retrieval.strategy_registry import (
    QueryInput,
    get_strategy_spec,
)
from unified_kg_rag.ports.model_factory import LLMFactoryPort
from unified_kg_rag.shared import InvalidFilterError, get_logger
from unified_kg_rag.shared.utils import configure_event_loop, strip_embedding_fields

logger = get_logger(__name__)

# Fixed reply when retrieval produced no context; the output metadata then
# carries ``abstained=True`` so evaluation can tell it from a real answer.
NO_CONTEXT_ANSWER = (
    "I could not find relevant information in the available data to answer "
    "this question."
)

DEFAULT_ERROR_MESSAGE: str = (
    "I apologize, but an error occurred while processing your request. Please try again in a moment."
)


class ChainMode(str, Enum):
    RAG = "rag"
    SEARCH = "search"


class ProcessedQuery(BaseModel):
    original_query: str = Field(
        description="The original user query before any processing"
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


class RAGInput(BaseModel):
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
            "The search strategy: GraphRAG (auto, drift, global, local, simple) "
            "or LightRAG dual-level keyword (mix, hybrid, naive)"
        ),
    )
    search_type: SearchType = Field(
        default=SearchType.HYBRID,
        description="The type of search to perform (hybrid, lexical, vector)",
    )
    top_k: int = Field(default=10, description="Maximum number of results to retrieve")
    retrieval_multiplier: int = Field(
        default=1,
        description="Multiplier for retrieval operations to increase search depth",
    )
    max_tokens: int | None = Field(
        default=None, description="Maximum number of tokens for the generated answer"
    )
    conversation_id: str | None = Field(
        default=None, description="Unique identifier for the conversation session"
    )
    use_memory: bool = Field(
        default=False, description="Whether to use conversation memory"
    )
    enable_query_processing: bool = Field(
        default=True,
        description="Whether to enable query processing (translation, entity extraction)",
    )
    target_language: str | None = Field(
        default=None, description="Target language for translation"
    )
    filters: dict[str, Any] | None = Field(
        default=None, description="Additional filters to apply to the search"
    )


class RAGOutput(BaseModel):
    answer: str = Field(description="The generated answer to the user's query")
    sources: list[dict[str, Any]] = Field(
        description="List of source documents used to generate the answer"
    )
    search_results: SearchResult = Field(
        description="Detailed search results from the retrieval process"
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


class GraphRAGChain(Runnable[RAGInput, RAGOutput | dict[str, Any]]):
    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        mode: ChainMode = ChainMode.RAG,
        *,
        model_factory: LLMFactoryPort | None = None,
        providers: Providers | None = None,
        memory_manager: MemoryManager | None = None,
        retriever_builders: (
            dict[RetrieverRole, Callable[[], BaseGraphRAGRetriever]] | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = config
        if providers is not None and model_factory is not None:
            raise ValueError(
                "Pass either model_factory or providers, not both; set the LLM "
                "factory on the Providers bundle instead."
            )
        # Composition root (hexagonal): ONE provider bundle — session plus
        # LLM/embedding/rerank/token-counter factories — built here (or
        # injected) and passed explicitly to every component the chain builds
        # (strategies, scorer, token manager, memory, retrievers). An injected
        # factory, e.g. a local Ollama-backed LLMFactoryPort, therefore
        # reaches all of them; anything not injected defaults to Bedrock.
        self.providers = providers or Providers(
            config, boto_session, llm_factory=model_factory
        )
        self.boto_session = self.providers.boto_session
        self.mode = mode
        self.ignore_errors = self.config.processing.ignore_errors
        # Conversation memory is process-wide by default, so history survives
        # across chain instances (e.g. a chain per request); the first chain's
        # config and providers create it. Inject a MemoryManager to isolate a
        # chain's conversations.
        self.memory_manager: MemoryManager = memory_manager or get_memory_manager(
            config, self.providers
        )
        self.token_manager = TokenManager(self.config, providers=self.providers)
        self.factory: LLMFactoryPort = self.providers.llm_factory
        # Backend seam: inject custom retriever builders keyed by abstract role
        # ("graph"/"document") to swap Neptune/OpenSearch for another store
        # without subclassing. Unspecified roles fall back to the AWS defaults.
        self._retriever_builders_override = retriever_builders or {}
        # Retrievers and strategy instances are built once per event loop and
        # reused across queries: retrievers hold loop-bound async clients, and a
        # strategy holds its retrievers, so both caches are dropped together
        # when the loop changes. Strategies keep no per-query state on the
        # instance (see BaseSearchStrategy), so concurrent queries on one loop
        # share an instance.
        self._retriever_cache: dict[
            tuple[RetrieverRole, int | None], BaseGraphRAGRetriever
        ] = {}
        self._strategy_cache: dict[
            tuple[SearchStrategy, int | None], BaseSearchStrategy
        ] = {}
        # The loop the cached retrievers are bound to. Held by reference, not
        # id: a dead loop's id can be reused by a new loop.
        self._cached_loop: asyncio.AbstractEventLoop | None = None
        # The sync entry points (invoke/batch/stream) run on this one
        # long-lived loop instead of a fresh loop per call, so their
        # loop-bound retrievers are built once and reused.
        self._loop_runner = _LoopRunner(self.config.processing.io_workers)
        weakref.finalize(self, self._loop_runner.stop)
        self.chain = self._build_chain()

    def _build_chain(self) -> Runnable:
        base_chain: Runnable = (
            RunnableLambda(self._resolve_strategy)
            | RunnablePassthrough.assign(
                processed_query=self._query_processing_branch()
            )
            | RunnableLambda(self._load_memory_step)
            | RunnablePassthrough.assign(search_results=self._search_step)
        )

        context_step: Runnable = RunnablePassthrough.assign(
            optimized_context=self._context_optimization_step
        ) | RunnablePassthrough.assign(context=self._context_building_step)
        # Streaming runs everything up to (and including) context building as
        # one non-streamed pass, then streams only the answer LLM. Reusing the
        # same step objects keeps stream and invoke behaviour in lockstep.
        self._stream_context_chain: Runnable = base_chain | context_step

        rag_branch = (
            context_step
            | RunnablePassthrough.assign(answer=self._answer_generation_step)
            | RunnableLambda(self._format_output_step)
        )

        search_branch = RunnableLambda(self._format_search_output_step)

        return base_chain | RunnableBranch(
            (lambda _: self.mode == ChainMode.RAG, rag_branch),
            search_branch,
        )

    async def _resolve_strategy(self, state: dict[str, Any]) -> dict[str, Any]:
        strategy = state.get("search_strategy", SearchStrategy.AUTO)

        if strategy != SearchStrategy.AUTO:
            state["resolved_strategy"] = strategy
            return state

        try:
            router = self._get_chain_for_prompt(
                StrategySelectionPrompt, StrOutputParser()
            )
            query = state.get("query", "")
            routable = tuple(self.config.search.auto_routable_strategies)
            selected_strategy_str = await router.ainvoke(
                {
                    "query": query,
                    "strategies": ", ".join(s.value for s in routable),
                }
            )
            strategy = self._parse_routed_strategy(selected_strategy_str, routable)
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.warning(
                "Strategy auto-selection failed, using LOCAL as fallback: %s", e
            )
            strategy = SearchStrategy.LOCAL

        state["resolved_strategy"] = strategy
        return state

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
        first). Defaults to LOCAL, a general-purpose graph strategy, not the
        costliest one.
        """
        allowed = routable or tuple(SearchConfig().auto_routable_strategies)
        by_value = {strat.value: strat for strat in allowed}
        for token in re.findall(r"[a-z]+", (raw or "").lower()):
            if token in by_value:
                return by_value[token]
        logger.warning(
            "Router returned unrecognized strategy '%s'; defaulting to LOCAL", raw
        )
        return SearchStrategy.LOCAL

    def _get_chain_for_prompt(
        self,
        prompt_class: type[BasePrompt],
        parser: BaseOutputParser,
        **kwargs: Any,
    ) -> Runnable:
        model_id_map: dict[type[BasePrompt], str] = {
            EntityExtractionPrompt: self.config.search.entity_extraction_model_id,
            KeywordsExtractionPrompt: self.config.search.entity_extraction_model_id,
            TranslationPrompt: self.config.search.translation_model_id,
            StrategySelectionPrompt: self.config.search.strategy_selection_model_id,
            ContextBuildingPrompt: self.config.search.context_building_model_id,
            AnswerGenerationPrompt: self.config.search.answer_generation_model_id,
        }
        model_id = model_id_map.get(prompt_class)
        if model_id is None:
            raise ValueError(
                f"No model id configured for prompt {prompt_class.__name__}; "
                f"add it to _get_chain_for_prompt's model_id_map."
            )
        return setup_chain(
            factory=self.factory,
            model_id=model_id,
            prompt_class=prompt_class,
            parser=parser,
            custom_prompts=self.config.custom_prompts,
            **kwargs,
        )

    def _query_processing_branch(self) -> Runnable:
        def _simple_query(inputs: dict[str, Any]) -> ProcessedQuery:
            query = inputs["query"]
            return ProcessedQuery(original_query=query, final_query=query)

        return RunnableBranch(
            (
                lambda x: (
                    x.get("enable_query_processing", True)
                    if isinstance(x, dict)
                    else True
                ),
                self._process_query_step,
            ),
            RunnableLambda(_simple_query),
        )

    async def _process_query_step(self, inputs: dict[str, Any]) -> ProcessedQuery:
        original_query = inputs.get("query", "")

        try:
            requested_language = inputs.get("target_language")
            target_language = (
                requested_language or self.config.processing.translation.target_language
            )

            # Skip query translation when it would be a no-op: no explicit target
            # was requested AND the corpus is same-language (mirrors the ingestion
            # side's TranslationConfig.is_noop skip). This avoids paying an LLM
            # call per query for e.g. an English-only or Japanese-only corpus, and
            # — more importantly — avoids the failure mode where the translator
            # LLM returns a meta-response ("I notice the text you...") for a query
            # already in the target language, which would then be used verbatim as
            # the search query.
            skip_translation = (
                requested_language is None
                and self.config.processing.translation.is_noop
            )

            # The strategy is resolved (AUTO routing included) before this step,
            # so only pay for the LLM extractions the resolved strategy reads.
            needs_entities = self._needs_query_entities(inputs)
            needs_keywords = self._needs_dual_keywords(inputs)

            tasks: dict[str, Any] = {}
            if needs_entities:
                entity_extractor = self._get_chain_for_prompt(
                    EntityExtractionPrompt, CommaSeparatedListOutputParser()
                )
                tasks["entities"] = entity_extractor.ainvoke(
                    {"query": original_query, "target_language": target_language}
                )
            # Keyword extraction runs on the final (possibly translated) query.
            # When translation is skipped that is the original query, so it can
            # run concurrently with entity extraction instead of after it.
            if needs_keywords and skip_translation:
                tasks["keywords"] = self._extract_dual_keywords(
                    original_query, target_language
                )
            if not skip_translation:
                translator = self._get_chain_for_prompt(
                    TranslationPrompt, StrOutputParser()
                )
                tasks["translation"] = translator.ainvoke(
                    {"query": original_query, "target_language": target_language}
                )

            results = (
                await asyncio.gather(*tasks.values(), return_exceptions=True)
                if tasks
                else []
            )
            results_map = dict(zip(tasks.keys(), results, strict=True))
            translated_query = results_map.get("translation")
            if not isinstance(translated_query, str):
                if isinstance(translated_query, Exception):
                    if self.ignore_errors:
                        logger.warning("Query translation failed: %s", translated_query)
                    else:
                        raise translated_query
                translated_query = None
            elif self._looks_like_translation_refusal(translated_query, original_query):
                # The translator returned prose about the request instead of a
                # translation (common when the query is already in the target
                # language). Fall back to the original query rather than searching
                # for the LLM's meta-response.
                logger.warning(
                    "Query translation looks like non-translation LLM output; "
                    "falling back to the original query."
                )
                translated_query = None

            entity_data = results_map.get("entities", [])
            if not isinstance(entity_data, list):
                if isinstance(entity_data, Exception):
                    if self.ignore_errors:
                        logger.warning("Entity extraction failed: %s", entity_data)
                    else:
                        raise entity_data
                entity_data = []

            final_query = translated_query or original_query
            hl_keywords: list[str] = []
            ll_keywords: list[str] = []
            if "keywords" in results_map:
                keyword_data = results_map["keywords"]
                if isinstance(keyword_data, BaseException):
                    # _extract_dual_keywords already degrades to ([], []) under
                    # ignore_errors, so an exception here must propagate.
                    raise keyword_data
                hl_keywords, ll_keywords = keyword_data
            elif needs_keywords:
                hl_keywords, ll_keywords = await self._extract_dual_keywords(
                    final_query, target_language
                )

            return ProcessedQuery(
                original_query=original_query,
                translated_query=translated_query,
                final_query=final_query,
                entities=entity_data,
                hl_keywords=hl_keywords,
                ll_keywords=ll_keywords,
            )
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.warning("Query processing failed: %s", e)
            return ProcessedQuery(
                original_query=original_query, final_query=original_query, entities=[]
            )

    @staticmethod
    def _needs_query_entities(state: dict[str, Any]) -> bool:
        # Without a resolved strategy (direct step invocation) stay conservative
        # and extract, matching the pre-gating behavior.
        strategy = state.get("resolved_strategy")
        return (
            strategy is None
            or QueryInput.ENTITIES in get_strategy_spec(strategy).query_inputs
        )

    @staticmethod
    def _needs_dual_keywords(state: dict[str, Any]) -> bool:
        strategy = state.get("resolved_strategy")
        return (
            strategy is not None
            and QueryInput.DUAL_KEYWORDS in get_strategy_spec(strategy).query_inputs
        )

    # Phrases that signal the model returned commentary about the request rather
    # than a translation of it (seen when the query is already in the target
    # language). Matched case-insensitively against the start of the response.
    _TRANSLATION_REFUSAL_MARKERS: ClassVar[tuple[str, ...]] = (
        "i appreciate",
        "i notice",
        "i'm sorry",
        "i am sorry",
        "i cannot",
        "i can't",
        "as an ai",
        "it appears",
        "the text you",
        "there is no text",
        "no text was provided",
    )

    @classmethod
    def _looks_like_translation_refusal(cls, candidate: str, original: str) -> bool:
        """Heuristic: does the 'translation' look like LLM meta-output, not a
        translation? Only flags when the candidate differs from the original
        (an identical passthrough is fine) and opens with a known refusal/notice
        phrase, so genuine translations that merely contain such words mid-text
        are not caught."""
        text = candidate.strip().lower()
        if not text or text == original.strip().lower():
            return False
        return text.startswith(cls._TRANSLATION_REFUSAL_MARKERS)

    async def _extract_dual_keywords(
        self, query: str, target_language: Any
    ) -> tuple[list[str], list[str]]:
        """Extract LightRAG high/low-level keywords as two lists.

        Robust to the LLM wrapping the JSON in prose or code fences. Returns
        empty lists on failure when ``ignore_errors`` is set.
        """
        try:
            extractor = self._get_chain_for_prompt(
                KeywordsExtractionPrompt, StrOutputParser()
            )
            raw = await extractor.ainvoke(
                {"query": query, "target_language": target_language}
            )
            payload = self._parse_keyword_json(raw)
            hl = [str(k) for k in payload.get("high_level_keywords", []) if k]
            ll = [str(k) for k in payload.get("low_level_keywords", []) if k]
            return hl, ll
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.warning("Dual-keyword extraction failed: %s", e)
            return [], []

    @staticmethod
    def _parse_keyword_json(raw: str) -> dict[str, Any]:
        # Strict variant: unlike the shared degrade-to-{} parser, this RAISES on
        # malformed JSON so that with ignore_errors=False a broken keyword
        # extraction surfaces instead of silently yielding empty keyword lists.
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1] if "```" in text[3:] else text[3:]
            if text.lstrip().lower().startswith("json"):
                text = text.lstrip()[4:]
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}

    async def _load_memory_step(self, state: dict[str, Any]) -> dict[str, Any]:
        if not state.get("use_memory") or not (cid := state.get("conversation_id")):
            state["history"] = ""
            state["relevant_entities"] = []
            return state

        memory = await self.memory_manager.get_langchain_memory(cid)
        memory_variables = memory.load_memory_variables(state)
        state.update(memory_variables)
        return state

    async def _search_step(self, state: dict[str, Any]) -> SearchResult:
        # A single GraphRAGChain is reused for concurrent invocations (the
        # evaluation path runs queries through `abatch`). Keep the resolved
        # strategy in a LOCAL — storing it on `self` lets a concurrent
        # `_search_step` overwrite it between assignment and `await`, executing
        # one query against another query's strategy (silent cross-contamination).
        # The instance itself is cached per loop and holds no per-query state.
        strategy_instance = self._get_strategy_instance(state["resolved_strategy"])
        if state.get("filters"):
            self._validate_filter_keys(
                state["filters"], list(strategy_instance.retrievers.values())
            )
        processed: ProcessedQuery = state["processed_query"]
        # Ordered dedupe (not set()) so the joined entity query, and hence its
        # embedding, is reproducible across processes.
        entity_focus = list(
            dict.fromkeys(processed.entities + state.get("relevant_entities", []))
        )

        resolved_strategy: SearchStrategy = state["resolved_strategy"]
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
            # naive), so it reads the resolved mode from here.
            metadata={"search_strategy": resolved_strategy.value},
        )
        return await strategy_instance.asearch(search_query)

    @staticmethod
    def _validate_filter_keys(
        filters: dict[str, Any] | None, retrievers: list[BaseGraphRAGRetriever]
    ) -> None:
        """Reject caller filter keys that no retriever of the strategy declares.

        Each retriever applies a key only to the indexes / labels that declare
        it, so an undeclared key would be dropped everywhere and the query would
        silently run unfiltered. A retriever without a declared schema accepts
        every key.
        """
        if not filters:
            return
        schemas = [r.filter_fields() for r in retrievers]
        if any(schema is None for schema in schemas):
            return
        declared = union_filter_fields(s for s in schemas if s is not None)
        unknown = sorted(key for key in filters if not declared.declares(key))
        if unknown:
            raise InvalidFilterError(
                f"Unknown filter key(s): {', '.join(unknown)}. No index this "
                "search strategy reads has these fields, so the filter would "
                f"be ignored. Filterable keys: {', '.join(declared.describe())}."
            )

    def _get_strategy_instance(
        self, strategy_type: SearchStrategy
    ) -> BaseSearchStrategy:
        current_loop_id = self._sync_loop_caches()
        cache_key = (strategy_type, current_loop_id)
        if (cached := self._strategy_cache.get(cache_key)) is not None:
            return cached

        spec = get_strategy_spec(strategy_type)

        # Inject retrievers keyed by abstract role ("graph"/"document"), so the
        # strategy never names a concrete backend.
        retrievers = {
            role.value: self._get_retriever(role) for role in spec.required_roles
        }

        strategy = spec.strategy_class(
            config=self.config, retrievers=retrievers, providers=self.providers
        )
        self._strategy_cache[cache_key] = strategy
        return strategy

    def _build_graph_retriever(self) -> BaseGraphRAGRetriever:
        neptune_client = NeptuneClient(
            config=self.config, boto_session=self.boto_session
        )
        return NeptuneRetriever(
            config=self.config,
            neptune_client=neptune_client,
            boto_session=self.boto_session,
        )

    def _build_document_retriever(self) -> BaseGraphRAGRetriever:
        opensearch_client = OpenSearchClient(
            config=self.config, boto_session=self.boto_session
        )
        return OpenSearchRetriever(
            config=self.config,
            opensearch_client=opensearch_client,
            boto_session=self.boto_session,
            embedding_factory=self.providers.embedding_factory,
        )

    def _get_retriever(self, role: RetrieverRole) -> BaseGraphRAGRetriever:
        current_loop_id = self._sync_loop_caches()

        cache_key = (role, current_loop_id)
        if cache_key in self._retriever_cache:
            return self._retriever_cache[cache_key]

        # Role -> adapter builder. Swapping a backend means changing the builder
        # bound to a role here, not editing any strategy. Injected
        # retriever_builders take precedence over the AWS defaults (the backend
        # seam), so a custom store is wired without subclassing.
        builders: dict[RetrieverRole, Callable[[], BaseGraphRAGRetriever]] = {
            RetrieverRole.GRAPH: self._build_graph_retriever,
            RetrieverRole.DOCUMENT: self._build_document_retriever,
            **self._retriever_builders_override,
        }
        builder = builders.get(role)
        if builder is None:
            raise ValueError(f"No retriever bound to role: '{role}'")

        retriever = builder()
        self._retriever_cache[cache_key] = retriever
        return retriever

    def _sync_loop_caches(self) -> int | None:
        """Release loop-bound caches when the running event loop changed.

        Retrievers hold clients bound to the loop they were built on, so a
        query on another loop needs new ones. The evicted retrievers are closed
        (on their own loop when it is still running) instead of being left to
        leak their connection pools. Returns the current loop id (``None``
        outside a running loop).
        """
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            return None
        if current is not self._cached_loop:
            logger.debug(
                "Event loop changed (old=%s, new=%s), releasing cached retrievers "
                "and strategies",
                id(self._cached_loop) if self._cached_loop else None,
                id(current),
            )
            evicted, evicted_loop = self._take_cached_retrievers()
            self._release_retrievers(evicted, evicted_loop, wait=False)
            self._cached_loop = current
        return id(current)

    def _take_cached_retrievers(
        self,
    ) -> tuple[list[BaseGraphRAGRetriever], asyncio.AbstractEventLoop | None]:
        retrievers = list(self._retriever_cache.values())
        loop = getattr(self, "_cached_loop", None)
        self._retriever_cache.clear()
        # Strategies hold the retrievers, so they go with them.
        getattr(self, "_strategy_cache", {}).clear()
        self._cached_loop = None
        return retrievers, loop

    @staticmethod
    def _release_retrievers(
        retrievers: list[BaseGraphRAGRetriever],
        loop: asyncio.AbstractEventLoop | None,
        *,
        wait: bool,
    ) -> None:
        """Close retrievers from sync code (best-effort, never raises).

        Retrievers bound to a loop that is still running on another thread are
        closed there with their awaited ``aclose``; otherwise (no loop, or the
        loop is gone) their synchronous ``close`` releases what it can.
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
        for retriever in retrievers:
            close = getattr(retriever, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as e:  # noqa: BLE001 - teardown must never raise
                    logger.debug("Error closing retriever %r: %s", retriever, e)

    async def aclose(self) -> None:
        """Close every cached retriever's backing client (best-effort).

        Each retriever build opens a Neptune websocket + thread pool and/or an
        OpenSearch (a)sync HTTP pool that otherwise survive until GC. Call this
        when the chain is done (e.g. from the CLI ``finally``) so a process that
        finishes a query releases its sockets. Retrievers are closed on the
        loop they are bound to, then the chain's sync-API loop is stopped.
        Never raises.
        """
        retrievers, loop = self._take_cached_retrievers()
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
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
            self._release_retrievers(retrievers, None, wait=False)
        runner = getattr(self, "_loop_runner", None)
        if runner is not None:
            await asyncio.to_thread(runner.stop)

    def close(self) -> None:
        """Synchronous teardown of cached retrievers and the sync-API loop."""
        retrievers, loop = self._take_cached_retrievers()
        self._release_retrievers(retrievers, loop, wait=True)
        runner = getattr(self, "_loop_runner", None)
        if runner is not None:
            runner.stop()

    def _optimize_context(self, state: dict[str, Any]) -> OptimizedContext:
        query: ProcessedQuery = state["processed_query"]
        search_results: SearchResult = state["search_results"]
        return self.token_manager.optimize_context(
            retrieval_results=search_results.results,
            query=query.final_query,
            max_tokens=state.get("max_tokens"),
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
            logger.warning("Context optimization failed: %s", e, exc_info=True)
            return OptimizedContext(
                sections=[],
                total_tokens=0,
                sections_included=0,
                sections_excluded=len(state["search_results"].results),
                quality_score=0.0,
            )

    async def _context_building_step(self, state: dict[str, Any]) -> str:
        try:
            query: ProcessedQuery = state["processed_query"]
            optimized: OptimizedContext | None = state.get("optimized_context")
            if optimized is None:
                optimized = self._optimize_context(state)
            search_context = self.token_manager.build_context_string(optimized)
            history = state.get("history")

            # Decide emptiness on what retrieval actually produced, BEFORE the
            # history-aware rewrite: the builder LLM can turn an empty context
            # plus conversation history into a plausible narrative that would
            # slip past the answer step's empty-context guard.
            if not history or not optimized.sections:
                return search_context

            context_builder = self._get_chain_for_prompt(
                ContextBuildingPrompt, StrOutputParser()
            )
            result = await context_builder.ainvoke(
                {
                    "query": query.original_query,
                    "search_results": search_context,
                    "conversation_history": history,
                }
            )
            return str(result)
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.warning("Context building failed: %s", e, exc_info=True)
            return ""

    def _answer_generation_step(self, state: dict[str, Any]) -> Runnable:
        # Never ask the LLM to answer from an empty/placeholder context: retrieval
        # produced nothing, so generating anyway invites a confident hallucination
        # with no supporting sources. Short-circuit to an explicit "cannot answer".
        context = str(state.get("context") or "").strip()
        if not context or context == EMPTY_CONTEXT_PLACEHOLDER:
            return RunnableLambda(lambda _: NO_CONTEXT_ANSWER)

        enable_thinking = state.get("enable_thinking", False)
        return self._get_chain_for_prompt(
            AnswerGenerationPrompt,
            StrOutputParser(),
            enable_thinking=enable_thinking,
        )

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
        context = str(state.get("context") or "").strip()
        if optimized is None:
            sources = [
                GraphRAGChain._source_entry(r, r.content, truncated=False)
                for r in sr.results
                if not GraphRAGChain._is_synthesized(r.metadata)
            ]
        elif not context or context == EMPTY_CONTEXT_PLACEHOLDER:
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
        }
        if not context or context == EMPTY_CONTEXT_PLACEHOLDER:
            # The answer step returned the fixed NO_CONTEXT_ANSWER.
            metadata["abstained"] = True
        if optimized is not None:
            metadata["context_sections_included"] = optimized.sections_included
            metadata["context_sections_excluded"] = optimized.sections_excluded

        return RAGOutput(
            answer=state["answer"],
            sources=sources,
            search_results=sr,
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
        retriever-supplied metadata of the same name.
        """
        metadata = strip_embedding_fields(result.metadata or {})
        metadata.pop("truncated", None)
        raw_doc_ids = metadata.get("document_ids") or metadata.get("document_id")
        document_ids = (
            [str(d) for d in raw_doc_ids]
            if isinstance(raw_doc_ids, list | tuple | set)
            else [str(raw_doc_ids)] if raw_doc_ids else []
        )
        chunk_id = result.chunk_id
        if chunk_id is None and result.retriever_type == SectionType.TEXT.value:
            chunk_id = metadata.get("id") or result.source
        provenance = {
            "source_id": result.source,
            "document_ids": document_ids,
            "chunk_id": chunk_id,
            "section_type": result.retriever_type,
            "score": result.score,
        }
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
            truncated = bool(section.metadata.get("truncated"))
            index = section.result_index
            if index is not None and 0 <= index < len(results):
                result = results[index]
            else:
                result = RetrievalResult(
                    content=section.content,
                    score=section.priority,
                    source=section.source_id,
                    retriever_type=section.section_type.value,
                    metadata=section.metadata,
                )
            if GraphRAGChain._is_synthesized(result.metadata):
                continue
            sources.append(
                GraphRAGChain._source_entry(
                    result, section.content, truncated=truncated
                )
            )
        return sources

    @staticmethod
    def _format_search_output_step(state: dict[str, Any]) -> dict[str, Any]:
        sr: SearchResult = state["search_results"]
        sr.search_strategy = state["resolved_strategy"].value

        metadata = {
            "search_strategy": sr.search_strategy,
            "processing_time": time.time() - state["start_time"],
            "total_results": len(sr.results),
            **sr.metadata,
        }

        return {
            "search_results": sr.model_dump(),
            "processed_query": state["processed_query"].model_dump(),
            "metadata": metadata,
        }

    def invoke(
        self,
        input: RAGInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> RAGOutput | dict[str, Any]:
        # Runs on the chain's long-lived loop (not asyncio.run per call), so it
        # also works from a thread that already runs an event loop and reuses
        # the loop-bound retrievers across calls (batch() fans out to here).
        result: RAGOutput | dict[str, Any] = self._loop_runner.run(
            self.ainvoke(input, config, **kwargs)
        )
        return result

    async def ainvoke(
        self,
        input: RAGInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> RAGOutput | dict[str, Any]:
        rag_input, input_dict = self._prepare_invoke(input)
        tokens = bind_contextvars(
            query_id=uuid.uuid4().hex[:12], conversation_id=rag_input.conversation_id
        )

        try:
            output: RAGOutput | dict[str, Any] = await self.chain.ainvoke(
                input_dict, config
            )
            await self._save_memory(
                output, query_processing=rag_input.enable_query_processing
            )
            if isinstance(output, RAGOutput):
                return output
            if self.mode == ChainMode.SEARCH:
                return output
            return RAGOutput(**output)
        except Exception as e:
            if not self.ignore_errors:
                raise

            logger.error(
                "RAG chain execution failed for query '%s': %s", rag_input.query, e
            )
            processing_time = time.time() - input_dict["start_time"]

            search_result = SearchResult(
                query=SearchQuery(query=rag_input.query),
                results=[],
                total_results=0,
                search_strategy="error",
                processing_time=processing_time,
                metadata={"error": str(e)},
            )
            processed_query = ProcessedQuery(
                original_query=rag_input.query, final_query=rag_input.query
            )
            error_metadata = {"error": True, "processing_time": processing_time}

            if self.mode == ChainMode.SEARCH:
                return {
                    "search_results": search_result.model_dump(),
                    "processed_query": processed_query.model_dump(),
                    "metadata": error_metadata,
                }
            return RAGOutput(
                answer=DEFAULT_ERROR_MESSAGE,
                sources=[],
                search_results=search_result,
                conversation_id=rag_input.conversation_id,
                processed_query=processed_query,
                metadata=error_metadata,
            )
        finally:
            reset_contextvars(**tokens)

    @staticmethod
    def _prepare_invoke(
        inputs: RAGInput | dict[str, Any],
    ) -> tuple[RAGInput, dict[str, Any]]:
        rag_input = RAGInput(**inputs) if isinstance(inputs, dict) else inputs

        if rag_input.use_memory and not rag_input.conversation_id:
            rag_input.conversation_id = str(uuid.uuid4())

        input_dict = rag_input.model_dump()
        input_dict["start_time"] = time.time()
        return rag_input, input_dict

    async def _save_memory(
        self, output: RAGOutput | dict | None, *, query_processing: bool = False
    ) -> None:
        if not isinstance(output, RAGOutput) or not output.conversation_id:
            return

        try:
            query = (
                output.processed_query.original_query if output.processed_query else ""
            )
            entities = self._query_step_entities(output) if query_processing else None
            if entities is None:
                await self.memory_manager.add_message(
                    output.conversation_id, MessageRole.USER, query
                )
            else:
                # Reuse the query step's extraction (same prompt and model)
                # instead of a second entity-extraction LLM call.
                await self.memory_manager.add_message(
                    output.conversation_id, MessageRole.USER, query, entities=entities
                )
            await self.memory_manager.add_message(
                output.conversation_id, MessageRole.ASSISTANT, output.answer
            )
        except Exception as e:
            logger.error(
                "Failed to save conversation memory for conversation '%s': %s",
                output.conversation_id,
                e,
            )

    @staticmethod
    def _query_step_entities(output: RAGOutput) -> list[str] | None:
        """The entities the query step extracted, or None when it did not run.

        Extraction is gated on the resolved strategy declaring
        ``QueryInput.ENTITIES``; otherwise ``processed_query.entities`` is
        empty because nothing was extracted, not because there were none.
        """
        try:
            strategy = SearchStrategy(output.search_results.search_strategy)
            spec = get_strategy_spec(strategy)
        except (KeyError, ValueError):
            return None
        if QueryInput.ENTITIES not in spec.query_inputs:
            return None
        return list(output.processed_query.entities)

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

        Drives :meth:`astream` on the chain's long-lived event loop (the one
        ``invoke`` uses), pulling one chunk at a time. This works both from
        plain sync code and from a thread that already runs an event loop
        (where ``asyncio.run`` would raise), while still delivering chunks as
        they are produced rather than after the whole answer is generated.
        """
        if self.mode == ChainMode.SEARCH:
            logger.warning("Streaming is not supported in SEARCH mode.")
            return

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

        Retrieval, query processing and context building run once (the same
        steps ``ainvoke`` uses); only the answer-generation LLM is streamed.
        An empty context yields the standard no-data answer once without
        calling the LLM. On success the full answer is written to conversation
        memory exactly as ``ainvoke`` would.
        """
        if self.mode == ChainMode.SEARCH:
            logger.warning("Streaming is not supported in SEARCH mode.")
            return

        rag_input, input_dict = self._prepare_invoke(input)
        chunks: list[str] = []
        try:
            state: dict[str, Any] = await self._stream_context_chain.ainvoke(
                input_dict, config
            )
            answer_runnable = self._answer_generation_step(state)
            async for chunk in answer_runnable.astream(state, config):
                text = str(chunk)
                if not text:
                    continue
                chunks.append(text)
                yield text
        except Exception as e:
            if not self.ignore_errors:
                logger.error("RAG stream failed for query '%s': %s", rag_input.query, e)
                raise
            logger.error(
                "RAG stream failed for query '%s' after %s chunk(s): %s",
                rag_input.query,
                len(chunks),
                e,
            )
            # Mirror ainvoke's degraded answer. If part of the answer was
            # already delivered, append the notice so the truncation is visible.
            yield f"\n\n{DEFAULT_ERROR_MESSAGE}" if chunks else DEFAULT_ERROR_MESSAGE
            return

        state["answer"] = "".join(chunks)
        await self._save_memory(
            self._format_output_step(state),
            query_processing=rag_input.enable_query_processing,
        )


_T = TypeVar("_T")


# Upper bound on waiting for retrievers to close on another thread's loop.
_TEARDOWN_TIMEOUT_SECONDS = 30.0


async def _aclose_retrievers(retrievers: list[BaseGraphRAGRetriever]) -> None:
    for retriever in retrievers:
        aclose = getattr(retriever, "aclose", None)
        if aclose is None:
            continue
        try:
            await aclose()
        except Exception as e:  # noqa: BLE001 - teardown must never raise
            logger.debug("Error closing retriever %r: %s", retriever, e)


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

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                if self._io_workers is not None:
                    configure_event_loop(loop, self._io_workers)
                thread = threading.Thread(
                    target=loop.run_forever, name="graphrag-chain-loop", daemon=True
                )
                thread.start()
                self._loop, self._thread = loop, thread
            return self._loop

    def run(self, awaitable: Awaitable[_T]) -> _T:
        loop = self._ensure_loop()
        if threading.current_thread() is self._thread:
            raise RuntimeError(
                "A synchronous GraphRAGChain method was called from the chain's "
                "own event loop; await the async variant instead."
            )

        async def _await() -> _T:
            return await awaitable

        return asyncio.run_coroutine_threadsafe(_await(), loop).result()

    def stop(self) -> None:
        """Stop and close the loop (idempotent; restarted on next use)."""
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
        if loop is None or thread is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if thread is threading.current_thread():
            return  # stopping from inside the loop: it closes once unwound
        thread.join(timeout=_TEARDOWN_TIMEOUT_SECONDS)
        if not thread.is_alive():
            loop.close()


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
    try:
        while True:
            try:
                item = loop_runner.run(agen.__anext__())
            except StopAsyncIteration:
                return
            yield item
    finally:
        try:
            aclose = getattr(agen, "aclose", None)
            if aclose is not None:
                loop_runner.run(aclose())
        finally:
            if owned:
                loop_runner.stop()


async def create_rag_chain(
    config: Config,
    boto_session: boto3.Session | None = None,
    mode: ChainMode = ChainMode.RAG,
    **kwargs: Any,
) -> GraphRAGChain:
    return GraphRAGChain(config=config, boto_session=boto_session, mode=mode, **kwargs)
