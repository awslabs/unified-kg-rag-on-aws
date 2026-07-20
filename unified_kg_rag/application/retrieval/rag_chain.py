# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Iterator
from enum import Enum
from typing import Any, ClassVar

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

from unified_kg_rag.adapters.aws import (
    BedrockLanguageModelFactory,
    NeptuneClient,
    OpenSearchClient,
)
from unified_kg_rag.adapters.aws.chain_factory import setup_chain
from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
)
from unified_kg_rag.adapters.retrieval.memory_manager import get_memory_manager
from unified_kg_rag.adapters.retrieval.token_manager import (
    EMPTY_CONTEXT_PLACEHOLDER,
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
from unified_kg_rag.domain.models import (
    Config,
    LanguageModelId,
    MessageRole,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
)
from unified_kg_rag.domain.prompts import (
    AnswerGenerationPrompt,
    BasePrompt,
    ContextBuildingPrompt,
    EntityExtractionPrompt,
    KeywordsExtractionPrompt,
    StrategySelectionPrompt,
    TranslationPrompt,
)
from unified_kg_rag.domain.retrieval.strategy_registry import get_strategy_spec
from unified_kg_rag.ports.model_factory import LLMFactoryPort
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

DEFAULT_ERROR_MESSAGE: str = (
    "I apologize, but an error occurred while processing your request. Please try again in a moment."
)

# Strategies that use LightRAG dual-level keyword retrieval rather than the
# GraphRAG community-summary methodology.
LIGHTRAG_STRATEGIES: frozenset[SearchStrategy] = frozenset(
    {SearchStrategy.MIX, SearchStrategy.HYBRID, SearchStrategy.NAIVE}
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
        retriever_builders: (
            dict[RetrieverRole, Callable[[], BaseGraphRAGRetriever]] | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name
        )
        self.mode = mode
        self.ignore_errors = self.config.processing.ignore_errors
        self.memory_manager = get_memory_manager()
        self.token_manager = TokenManager(self.config, boto_session=self.boto_session)
        # Provider seam (hexagonal): inject a custom LLM factory (any
        # LLMFactoryPort — e.g. a local Ollama-backed one) instead of Bedrock.
        # Defaults to Bedrock so existing callers are unchanged.
        self.factory: LLMFactoryPort = model_factory or BedrockLanguageModelFactory(
            config=config,
            boto_session=boto_session,
            region_name=config.aws.bedrock.region_name,
        )
        # Backend seam: inject custom retriever builders keyed by abstract role
        # ("graph"/"document") to swap Neptune/OpenSearch for another store
        # without subclassing. Unspecified roles fall back to the AWS defaults.
        self._retriever_builders_override = retriever_builders or {}
        self._retriever_cache: dict[
            tuple[RetrieverRole, int | None], BaseGraphRAGRetriever
        ] = {}
        self._cached_loop_id: int | None = None
        self.chain = self._build_chain()

    def _build_chain(self) -> Runnable:
        # RunnableLambda accepts coroutine functions at runtime; the stubs only
        # type the sync-callable overload.
        base_chain: Runnable = (
            RunnableLambda(self._resolve_strategy)  # type: ignore[arg-type]
            | RunnablePassthrough.assign(
                processed_query=self._query_processing_branch()
            )
            | RunnableLambda(self._load_memory_step)  # type: ignore[arg-type]
            | RunnablePassthrough.assign(search_results=self._search_step)
        )

        rag_branch = (
            RunnablePassthrough.assign(context=self._context_building_step)
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
            selected_strategy_str = await router.ainvoke({"query": query})
            strategy = self._parse_routed_strategy(selected_strategy_str)
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.warning(
                "Strategy auto-selection failed, using LOCAL as fallback: %s", e
            )
            strategy = SearchStrategy.LOCAL

        state["resolved_strategy"] = strategy
        return state

    # AUTO routes within the GraphRAG strategies (the LightRAG mix/hybrid/naive
    # modes are a separate methodology the caller selects explicitly).
    _ROUTABLE_STRATEGIES = (
        SearchStrategy.SIMPLE,
        SearchStrategy.LOCAL,
        SearchStrategy.GLOBAL,
        SearchStrategy.DRIFT,
    )

    @classmethod
    def _parse_routed_strategy(cls, raw: str) -> SearchStrategy:
        """Map a router LLM response to a strategy, tolerating extra text.

        The router is asked for a bare word, but LLMs add punctuation/prose
        ("Local search.", "I'd use local"). Match a known strategy name as a
        substring instead of requiring an exact enum value (which raised
        ValueError and dropped to the expensive fallback). Defaults to LOCAL —
        a sensible general-purpose graph strategy — NOT DRIFT (the costliest).
        """
        text = (raw or "").strip().lower()
        # Exact match first, then substring (whole-word-ish) over routable names.
        for strat in cls._ROUTABLE_STRATEGIES:
            if text == strat.value:
                return strat
        for strat in cls._ROUTABLE_STRATEGIES:
            if strat.value in text:
                return strat
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
        model_id_map: dict[type[BasePrompt], LanguageModelId] = {
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

            entity_extractor = self._get_chain_for_prompt(
                EntityExtractionPrompt, CommaSeparatedListOutputParser()
            )

            tasks: dict[str, Any] = {
                "entities": entity_extractor.ainvoke(
                    {"query": original_query, "target_language": target_language}
                ),
            }
            if not skip_translation:
                translator = self._get_chain_for_prompt(
                    TranslationPrompt, StrOutputParser()
                )
                tasks["translation"] = translator.ainvoke(
                    {"query": original_query, "target_language": target_language}
                )

            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
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
            if self._is_lightrag_mode(inputs):
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
    def _is_lightrag_mode(state: dict[str, Any]) -> bool:
        strategy = state.get("resolved_strategy")
        return strategy in LIGHTRAG_STRATEGIES

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
        # evaluation path runs queries through `abatch`). Keep the per-query
        # strategy instance LOCAL — storing it on `self` lets a concurrent
        # `_search_step` overwrite it between assignment and `await`, executing
        # one query against another query's strategy (silent cross-contamination).
        strategy_instance = self._get_strategy_instance(state["resolved_strategy"])
        processed: ProcessedQuery = state["processed_query"]
        entity_focus = list(
            set(processed.entities + state.get("relevant_entities", []))
        )

        resolved_strategy: SearchStrategy = state["resolved_strategy"]
        metadata: dict[str, Any] = {}
        if resolved_strategy in LIGHTRAG_STRATEGIES:
            metadata["lightrag_mode"] = resolved_strategy.value

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
            metadata=metadata,
        )
        return await strategy_instance.asearch(search_query)

    def _get_strategy_instance(
        self, strategy_type: SearchStrategy
    ) -> BaseSearchStrategy:
        spec = get_strategy_spec(strategy_type)

        # Inject retrievers keyed by abstract role ("graph"/"document"), so the
        # strategy never names a concrete backend.
        retrievers = {
            role.value: self._get_retriever(role) for role in spec.required_roles
        }

        return spec.strategy_class(config=self.config, retrievers=retrievers)

    def _build_graph_retriever(self) -> BaseGraphRAGRetriever:
        neptune_client = NeptuneClient(
            config=self.config, boto_session=self.boto_session
        )
        return NeptuneRetriever(config=self.config, neptune_client=neptune_client)

    def _build_document_retriever(self) -> BaseGraphRAGRetriever:
        opensearch_client = OpenSearchClient(
            config=self.config, boto_session=self.boto_session
        )
        return OpenSearchRetriever(
            config=self.config, opensearch_client=opensearch_client
        )

    def _get_retriever(self, role: RetrieverRole) -> BaseGraphRAGRetriever:
        current_loop_id = self._get_current_loop_id()

        # Invalidate cache if event loop changed
        if current_loop_id is not None and self._cached_loop_id != current_loop_id:
            logger.debug(
                "Event loop changed (old=%s, new=%s), clearing retriever cache",
                self._cached_loop_id,
                current_loop_id,
            )
            self._retriever_cache.clear()
            self._cached_loop_id = current_loop_id

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

    def _get_current_loop_id(self) -> int | None:
        try:
            return id(asyncio.get_running_loop())
        except RuntimeError:
            return None

    async def aclose(self) -> None:
        """Close every cached retriever's backing client (best-effort).

        Each retriever build opens a Neptune websocket + thread pool and/or an
        OpenSearch (a)sync HTTP pool that otherwise survive until GC. Call this
        when the chain is done (e.g. from the CLI ``finally``) so a process that
        finishes a query releases its sockets. Never raises.
        """
        for retriever in self._retriever_cache.values():
            aclose = getattr(retriever, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception as e:  # noqa: BLE001 - teardown must never raise
                    logger.debug("Error closing retriever %r: %s", retriever, e)
        self._retriever_cache.clear()

    def close(self) -> None:
        """Synchronous teardown of cached retrievers' backing clients."""
        for retriever in self._retriever_cache.values():
            close = getattr(retriever, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as e:  # noqa: BLE001 - teardown must never raise
                    logger.debug("Error closing retriever %r: %s", retriever, e)
        self._retriever_cache.clear()

    def _context_building_step(self, state: dict[str, Any]) -> str:
        try:
            query: ProcessedQuery = state["processed_query"]
            search_results: SearchResult = state["search_results"]

            optimized = self.token_manager.optimize_context(
                retrieval_results=search_results.results,
                query=query.final_query,
                max_tokens=state.get("max_tokens"),
            )
            search_context = self.token_manager.build_context_string(optimized)
            history = state.get("history")

            if not history:
                return search_context

            context_builder = self._get_chain_for_prompt(
                ContextBuildingPrompt, StrOutputParser()
            )
            result = context_builder.invoke(
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
            return RunnableLambda(
                lambda _: "I could not find relevant information in the "
                "available data to answer this question."
            )

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
        sources = [
            r.model_dump(include={"source", "score", "metadata"}) for r in sr.results
        ]

        metadata = {
            "search_strategy": sr.search_strategy,
            "processing_time": time.time() - state["start_time"],
            "total_results": len(sr.results),
            **sr.metadata,
        }

        return RAGOutput(
            answer=state["answer"],
            sources=sources,
            search_results=sr,
            conversation_id=state.get("conversation_id"),
            processed_query=state["processed_query"],
            metadata=metadata,
        )

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
        return asyncio.run(self.ainvoke(input, config, **kwargs))

    async def ainvoke(
        self,
        input: RAGInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> RAGOutput | dict[str, Any]:
        rag_input, input_dict = self._prepare_invoke(input)

        try:
            output: RAGOutput | dict[str, Any] = await self.chain.ainvoke(
                input_dict, config
            )
            await self._save_memory(output)
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

    async def _save_memory(self, output: RAGOutput | dict | None) -> None:
        if not isinstance(output, RAGOutput) or not output.conversation_id:
            return

        try:
            query = (
                output.processed_query.original_query if output.processed_query else ""
            )
            await self.memory_manager.add_message(
                output.conversation_id, MessageRole.USER, query
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

    def stream(
        self,
        input: RAGInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[RAGOutput | dict[str, Any]]:
        if self.mode == ChainMode.SEARCH:
            logger.warning("Streaming is not supported in SEARCH mode.")
            return

        _, input_dict = self._prepare_invoke(input)
        answer_chain = self.chain | (lambda x: x["answer"])

        try:
            yield from answer_chain.stream(input_dict, config, **kwargs)
        except Exception as e:
            logger.error("RAG stream failed: %s", e)
            raise

    async def astream(
        self,
        input: RAGInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[RAGOutput, None]:
        if self.mode == ChainMode.SEARCH:
            logger.warning("Streaming is not supported in SEARCH mode.")
            return

        _, input_dict = self._prepare_invoke(input)
        answer_chain = self.chain | (lambda x: x["answer"])

        try:
            async for chunk in answer_chain.astream(input_dict, config, **kwargs):
                yield chunk
        except Exception as e:
            logger.error("Async RAG stream failed: %s", e)
            raise


async def create_rag_chain(
    config: Config,
    boto_session: boto3.Session | None = None,
    mode: ChainMode = ChainMode.RAG,
    **kwargs: Any,
) -> GraphRAGChain:
    return GraphRAGChain(config=config, boto_session=boto_session, mode=mode, **kwargs)
