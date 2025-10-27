import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, Iterator
from enum import Enum
from functools import lru_cache
from typing import Any

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

from aws_graphrag.aws import (
    BedrockLanguageModelFactory,
    NeptuneClient,
    OpenSearchClient,
)
from aws_graphrag.core import get_logger
from aws_graphrag.models import (
    Config,
    LanguageModelId,
    MessageRole,
    RetrieverType,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
)
from aws_graphrag.prompts import (
    AnswerGenerationPrompt,
    BasePrompt,
    ContextBuildingPrompt,
    EntityExtractionPrompt,
    StrategySelectionPrompt,
    TranslationPrompt,
)
from aws_graphrag.utils import setup_chain

from .base import BaseContextBuilder, BaseGraphRAGRetriever, BaseSearchStrategy
from .memory_manager import get_memory_manager
from .retrievers import NeptuneRetriever, OpenSearchRetriever
from .search_strategies import (
    DriftSearchStrategy,
    GlobalSearchStrategy,
    LocalSearchStrategy,
    SimpleSearchStrategy,
)
from .token_manager import TokenManager

logger = get_logger(__name__)

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
        description="The search strategy to use (auto, drift, global, local, simple)",
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
        self.token_manager = TokenManager(self.config)
        self.factory = BedrockLanguageModelFactory(
            config=config,
            boto_session=boto_session,
            region_name=config.aws.bedrock.region_name,
        )
        self.search_strategy_instance: BaseSearchStrategy | None = None
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
            strategy = SearchStrategy(selected_strategy_str.strip().lower())
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.warning(
                f"Strategy auto-selection failed, using DRIFT as fallback: {e}"
            )
            strategy = SearchStrategy.DRIFT

        state["resolved_strategy"] = strategy
        return state

    def _get_chain_for_prompt(
        self,
        prompt_class: type[BasePrompt],
        parser: BaseOutputParser,
        **kwargs: Any,
    ) -> Runnable:
        model_id_map: dict[type[BasePrompt], LanguageModelId] = {
            EntityExtractionPrompt: self.config.search.entity_extraction_model_id,
            TranslationPrompt: self.config.search.translation_model_id,
            StrategySelectionPrompt: self.config.search.strategy_selection_model_id,
            ContextBuildingPrompt: self.config.search.context_building_model_id,
            AnswerGenerationPrompt: self.config.search.answer_generation_model_id,
        }
        return setup_chain(
            factory=self.factory,
            model_id=model_id_map[prompt_class],
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
            target_language = (
                inputs.get("target_language")
                or self.config.processing.translation.target_language
            )

            translator = self._get_chain_for_prompt(
                TranslationPrompt, StrOutputParser()
            )
            entity_extractor = self._get_chain_for_prompt(
                EntityExtractionPrompt, CommaSeparatedListOutputParser()
            )

            tasks = {
                "translation": translator.ainvoke(
                    {"query": original_query, "target_language": target_language}
                ),
                "entities": entity_extractor.ainvoke(
                    {"query": original_query, "target_language": target_language}
                ),
            }

            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            results_map = dict(zip(tasks.keys(), results, strict=True))
            translated_query = results_map.get("translation")
            if not isinstance(translated_query, str):
                if isinstance(translated_query, Exception):
                    if self.ignore_errors:
                        logger.warning(f"Query translation failed: {translated_query}")
                    else:
                        raise translated_query
                translated_query = None

            entity_data = results_map.get("entities", [])
            if not isinstance(entity_data, list):
                if isinstance(entity_data, Exception):
                    if self.ignore_errors:
                        logger.warning(f"Entity extraction failed: {entity_data}")
                    else:
                        raise entity_data
                entity_data = []

            return ProcessedQuery(
                original_query=original_query,
                translated_query=translated_query,
                final_query=translated_query or original_query,
                entities=entity_data,
            )
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.warning(f"Query processing failed: {e}")
            return ProcessedQuery(
                original_query=original_query, final_query=original_query, entities=[]
            )

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
        self.search_strategy_instance = self._get_strategy_instance(
            state["resolved_strategy"]
        )
        processed: ProcessedQuery = state["processed_query"]
        entity_focus = list(
            set(processed.entities + state.get("relevant_entities", []))
        )

        search_query = SearchQuery(
            query=processed.final_query,
            search_type=state.get("search_type", SearchType.HYBRID),
            top_k=state.get("top_k", 10),
            retrieval_multiplier=state.get("retrieval_multiplier", 1),
            suffix=state.get("suffix"),
            max_tokens=state.get("max_tokens"),
            entity_focus=entity_focus or [],
            filters=state.get("filters"),
        )
        return await self.search_strategy_instance.asearch(search_query)

    def _get_strategy_instance(
        self, strategy_type: SearchStrategy
    ) -> BaseSearchStrategy:
        strategy_map: dict[SearchStrategy, type[BaseSearchStrategy]] = {
            SearchStrategy.SIMPLE: SimpleSearchStrategy,
            SearchStrategy.LOCAL: LocalSearchStrategy,
            SearchStrategy.GLOBAL: GlobalSearchStrategy,
            SearchStrategy.DRIFT: DriftSearchStrategy,
        }

        if strategy_type == SearchStrategy.SIMPLE:
            retrievers = {
                RetrieverType.OPENSEARCH.value: self._get_retriever(
                    RetrieverType.OPENSEARCH
                )
            }
        else:
            retrievers = {
                RetrieverType.OPENSEARCH.value: self._get_retriever(
                    RetrieverType.OPENSEARCH
                ),
                RetrieverType.NEPTUNE.value: self._get_retriever(RetrieverType.NEPTUNE),
            }

        strategy_class = strategy_map[strategy_type]
        context_builder: BaseContextBuilder | None = None
        return strategy_class(
            config=self.config, retrievers=retrievers, context_builder=context_builder
        )

    @lru_cache
    def _get_retriever(self, retriever_type: RetrieverType) -> BaseGraphRAGRetriever:
        if retriever_type == RetrieverType.OPENSEARCH:
            opensearch_client = OpenSearchClient(
                config=self.config, boto_session=self.boto_session
            )
            return OpenSearchRetriever(
                config=self.config, opensearch_client=opensearch_client
            )
        elif retriever_type == RetrieverType.NEPTUNE:
            neptune_client = NeptuneClient(
                config=self.config, boto_session=self.boto_session
            )
            return NeptuneRetriever(config=self.config, neptune_client=neptune_client)
        else:
            raise ValueError(f"Unknown retriever type: '{retriever_type}'")

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
            logger.warning(f"Context building failed: {e}", exc_info=True)
            return ""

    def _answer_generation_step(self, state: dict[str, Any]) -> Runnable:
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
            output = await self.chain.ainvoke(input_dict, config)
            await self._save_memory(output)
            if isinstance(output, RAGOutput):
                return output
            elif isinstance(output, dict):
                return RAGOutput(**output)
            else:
                return output
        except Exception as e:
            if not self.ignore_errors:
                raise

            logger.error(
                f"RAG chain execution failed for query '{rag_input.query}': {e}"
            )
            processing_time = time.time() - input_dict["start_time"]

            return RAGOutput(
                answer=DEFAULT_ERROR_MESSAGE,
                sources=[],
                search_results=SearchResult(
                    query=SearchQuery(query=rag_input.query),
                    results=[],
                    total_results=0,
                    search_strategy="error",
                    processing_time=processing_time,
                    metadata={"error": str(e)},
                ),
                conversation_id=rag_input.conversation_id,
                processed_query=ProcessedQuery(
                    original_query=rag_input.query, final_query=rag_input.query
                ),
                metadata={"error": True, "processing_time": processing_time},
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
                f"Failed to save conversation memory for conversation '{output.conversation_id}': {e}"
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
            logger.error(f"RAG stream failed: {e}")
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
            logger.error(f"Async RAG stream failed: {e}")
            raise


async def create_rag_chain(
    config: Config,
    boto_session: boto3.Session | None = None,
    mode: ChainMode = ChainMode.RAG,
    **kwargs: Any,
) -> GraphRAGChain:
    return GraphRAGChain(config=config, boto_session=boto_session, mode=mode, **kwargs)
