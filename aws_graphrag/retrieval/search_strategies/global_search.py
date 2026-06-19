# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import asyncio
import time
from typing import Any, ClassVar

import boto3
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from aws_graphrag.aws import BedrockLanguageModelFactory
from aws_graphrag.core import get_logger
from aws_graphrag.models import (
    Config,
    RetrievalResult,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from aws_graphrag.prompts import CommunityRelevancePrompt, MapReduceSummaryPrompt
from aws_graphrag.retrieval.base import (
    BaseContextBuilder,
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
)
from aws_graphrag.retrieval.strategy_registry import register_strategy
from aws_graphrag.retrieval.token_manager import SectionType
from aws_graphrag.utils import safe_float_parse, setup_chain

logger = get_logger(__name__)


@register_strategy(SearchStrategy.GLOBAL)
class GlobalSearchStrategy(BaseSearchStrategy):
    MAX_TEXT_UNITS: ClassVar[int] = 100

    def __init__(
        self,
        config: Config,
        retrievers: dict[str, BaseGraphRAGRetriever],
        context_builder: BaseContextBuilder | None = None,
        boto_session: boto3.Session | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, retrievers, context_builder, boto_session, **kwargs)
        self.global_search_config = config.search.global_search
        self.ignore_errors = config.processing.ignore_errors

        factory = BedrockLanguageModelFactory(
            config=config,
            boto_session=boto_session,
            region_name=config.aws.bedrock.region_name,
        )

        str_output_parser = StrOutputParser()
        self.community_relevance_scorer: Runnable = setup_chain(
            factory=factory,
            model_id=self.global_search_config.community_relevance_model_id,
            prompt_class=CommunityRelevancePrompt,
            parser=str_output_parser,
        )
        self.map_reducer: Runnable = setup_chain(
            factory=factory,
            model_id=self.global_search_config.map_reduce_model_id,
            prompt_class=MapReduceSummaryPrompt,
            parser=str_output_parser,
        )

    async def asearch(self, query: SearchQuery) -> SearchResult:
        start_time = time.time()
        logger.info(
            f"Global search started - query: '{query.query[:50]}...' ('{query.search_type.value}')"
        )

        retrieved_communities = await self._retrieve_and_fuse_communities(query)
        if not retrieved_communities:
            logger.warning(f"No results found for query: '{query.query[:50]}...'")
            return SearchResult(
                query=query,
                results=[],
                total_results=0,
                search_strategy="global_search",
                processing_time=time.time() - start_time,
                metadata={},
            )

        community_ids = self._get_ids(retrieved_communities, "community_id")
        logger.debug(
            f"Found {len(community_ids)} communities: '{', '.join(community_ids[:5])}{'...' if len(community_ids) > 5 else ''}'"
        )

        selected_communities = await self._select_relevant_communities(
            retrieved_communities, query
        )
        community_ids = self._get_ids(selected_communities, "id")
        logger.debug(
            f"Selected {len(community_ids)} communities: '{', '.join(community_ids[:5])}{'...' if len(community_ids) > 5 else ''}'"
        )

        final_results = await self._augment_and_rerank_communities(
            selected_communities, retrieved_communities, query
        )
        logger.debug(
            f"Augmented {len(final_results)} items: "
            f"'{', '.join(str(item.metadata.get('community_id') or item.metadata.get('id')) for item in final_results[:5] if item.metadata)}"
            f"{'...' if len(final_results) > 5 else ''}'"
        )

        if self.global_search_config.enable_map_reduce:
            final_results = await self._apply_map_reduce(final_results, query)

        final_results = final_results[: query.top_k]
        processing_time = time.time() - start_time

        self._record_search_metrics(
            processing_time,
            len(retrieved_communities),
            len(selected_communities),
            len(final_results),
        )

        logger.info(
            f"Search completed - retrieved: {len(final_results)} results ({processing_time:.3f}s)"
        )

        return SearchResult(
            query=query,
            results=final_results,
            total_results=len(final_results),
            search_strategy="global_search",
            processing_time=processing_time,
            metadata={
                "retrieved_community_count": len(retrieved_communities),
                "selected_community_count": len(selected_communities),
                "map_reduce_applied": self._was_map_reduce_applied(final_results),
            },
        )

    async def _retrieve_and_fuse_communities(
        self, query: SearchQuery
    ) -> list[RetrievalResult]:
        candidate_community_reports = await self._retrieve_community_reports(query)
        if not candidate_community_reports:
            return []

        candidate_community_ids = self._get_ids(
            candidate_community_reports, "community_id"
        )
        if not candidate_community_ids:
            return candidate_community_reports

        logger.debug(
            f"Found {len(candidate_community_ids)} candidate communities: "
            f"'{', '.join(str(cid) for cid in candidate_community_ids[:5])}"
            f"{'...' if len(candidate_community_ids) > 5 else ''}'"
        )

        expanded_community_nodes = await self._retrieve_community_nodes(
            query, candidate_community_ids
        )
        if not expanded_community_nodes:
            return candidate_community_reports

        expanded_community_ids = self._get_ids(expanded_community_nodes, "id")
        if not expanded_community_ids:
            return candidate_community_reports

        expanded_community_reports = await self._retrieve_reports_by_ids(
            expanded_community_ids, query
        )
        logger.debug(
            f"Expanded to {len(expanded_community_reports)} communities: "
            f"'{', '.join(str(cid) for cid in expanded_community_ids[:5])}"
            f"{'...' if len(expanded_community_ids) > 5 else ''}'"
        )

        return self.hybrid_scorer.fuse_and_rerank_results(
            {
                "opensearch_candidate_community_reports": candidate_community_reports,
                "opensearch_expanded_community_reports": expanded_community_reports,
            },
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
        )

    async def _retrieve_community_reports(
        self, query: SearchQuery
    ) -> list[RetrievalResult]:
        index_prefixes = [
            self.config.indexing.opensearch.community_reports_index_prefix
        ]
        return await self._retrieve_documents(query, index_prefixes)

    async def _retrieve_reports_by_ids(
        self, community_ids: list[str], query: SearchQuery
    ) -> list[RetrievalResult]:
        if not self.document_retriever or not community_ids:
            return []

        try:
            search_query = query.model_copy(deep=True)
            search_query.query = ""
            search_query.filters = (search_query.filters or {}).copy()
            search_query.filters["community_id"] = community_ids
            search_query.top_k = len(community_ids)

            index_prefixes = [
                self.config.indexing.opensearch.community_reports_index_prefix
            ]
            return await self._retrieve_documents(search_query, index_prefixes)
        except Exception as e:
            logger.error(f"OpenSearch retrieval failed: {e}")
            return []

    async def _retrieve_documents(
        self, query: SearchQuery, index_prefixes: list[str]
    ) -> list[RetrievalResult]:
        if not self.document_retriever:
            return []

        try:
            search_query = query.model_copy(deep=True)
            search_query.index_prefixes = index_prefixes
            return await self.document_retriever.aretrieve(search_query)
        except Exception as e:
            logger.error(f"OpenSearch retrieval failed: {e}")
            return []

    async def _retrieve_community_nodes(
        self, query: SearchQuery, community_ids: list[str]
    ) -> list[RetrievalResult]:
        if not self.graph_retriever:
            return []

        try:
            search_query = query.model_copy(deep=True)
            search_query.query = ""
            search_query.filters = (search_query.filters or {}).copy()
            search_query.filters["id"] = community_ids
            search_query.label_prefixes = [
                self.config.indexing.neptune.community_label_prefix
            ]

            return await asyncio.wait_for(
                self.graph_retriever.aretrieve(search_query), timeout=30.0
            )
        except Exception as e:
            logger.error(f"Neptune community retrieval failed: {e}")
            return []

    async def _select_relevant_communities(
        self, all_communities: list[RetrievalResult], query: SearchQuery
    ) -> list[RetrievalResult]:
        max_communities = (
            self.global_search_config.max_communities * query.retrieval_multiplier
        )
        if not self.global_search_config.use_dynamic_selection:
            return all_communities[:max_communities]

        async def score_item(item: RetrievalResult) -> tuple[RetrievalResult, float]:
            try:
                llm_output = await self.community_relevance_scorer.ainvoke(
                    {"community_summary": item.content, "query": query.query}
                )
                parsed_score = safe_float_parse(llm_output, default_value=5.0)
                relevance_score = parsed_score or 0.0

                if parsed_score is not None:
                    item.score = ((item.score or 0.5) * 0.4) + (
                        (relevance_score / 10.0) * 0.6
                    )

            except Exception as e:
                if not self.ignore_errors:
                    raise

                logger.warning(f"Community scoring failed: {e}")
                relevance_score = 0.0

            return item, relevance_score

        tasks = [score_item(item) for item in all_communities]
        evaluated_items = await asyncio.gather(*tasks)

        relevance_threshold = self.global_search_config.relevance_threshold
        relevant_items = [
            item for item, score in evaluated_items if score >= relevance_threshold
        ]
        relevant_items.sort(key=lambda x: x.score or 0.0, reverse=True)
        logger.debug(
            f"Filtered {len(evaluated_items) - len(relevant_items)} communities based on relevance threshold {relevance_threshold}"
        )

        return relevant_items[:max_communities]

    async def _augment_and_rerank_communities(
        self,
        selected: list[RetrievalResult],
        fallback: list[RetrievalResult],
        query: SearchQuery,
    ) -> list[RetrievalResult]:
        if not selected:
            return fallback

        context = await self._retrieve_community_context(selected, query)
        return self.hybrid_scorer.fuse_and_rerank_results(
            {
                "opensearch_community_reports": selected,
                "text_units": context,
            },
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
        )

    async def _retrieve_community_context(
        self, communities: list[RetrievalResult], query: SearchQuery
    ) -> list[RetrievalResult]:
        community_ids = self._get_ids(communities, "community_id")
        if not community_ids:
            return []

        search_query = query.model_copy(deep=True)
        search_query.filters = {
            **(query.filters or {}),
            "community_ids": list(community_ids),
        }
        search_query.top_k = min(query.top_k, self.MAX_TEXT_UNITS)
        index_prefixes = [self.config.indexing.opensearch.text_units_index_prefix]

        return await self._retrieve_documents(search_query, index_prefixes)

    async def _apply_map_reduce(
        self, results: list[RetrievalResult], query: SearchQuery
    ) -> list[RetrievalResult]:
        if len(results) < 3:
            return results

        try:
            context = "\n\n---\n\n".join([r.content for r in results])
            summary = await self.map_reducer.ainvoke(
                {"summaries": context, "query": query.query}
            )
            summary_result = RetrievalResult(
                content=summary,
                score=1.0,
                source="synthesized_summary",
                retriever_type=SectionType.GENERAL.value,
                metadata={"source_results_count": len(results)},
            )
            return [summary_result] + results
        except Exception as e:
            if not self.ignore_errors:
                raise

            logger.error(f"Map-reduce synthesis failed: {e}")
            return results

    def _record_search_metrics(
        self,
        processing_time: float,
        retrieved_count: int,
        selected_count: int,
        final_count: int,
    ) -> None:
        self._record_timing("processing_time", processing_time)
        self._record_metric("retrieved_count", retrieved_count)
        self._record_metric("selected_count", selected_count)
        self._record_metric("final_count", final_count)

    def _was_map_reduce_applied(self, final_results: list[RetrievalResult]) -> bool:
        if not self.global_search_config.enable_map_reduce:
            return False
        return any("synthesized" in (r.source or "") for r in final_results)
