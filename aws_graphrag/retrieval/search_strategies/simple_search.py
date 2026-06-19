# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import time
from typing import Any

import boto3

from aws_graphrag.core import get_logger
from aws_graphrag.models import (
    Config,
    RetrievalResult,
    RetrieverType,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from aws_graphrag.retrieval.base import (
    BaseContextBuilder,
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
)
from aws_graphrag.retrieval.strategy_registry import register_strategy

logger = get_logger(__name__)


@register_strategy(
    SearchStrategy.SIMPLE, required_retrievers=(RetrieverType.OPENSEARCH,)
)
class SimpleSearchStrategy(BaseSearchStrategy):
    def __init__(
        self,
        config: Config,
        retrievers: dict[str, BaseGraphRAGRetriever],
        context_builder: BaseContextBuilder | None = None,
        boto_session: boto3.Session | None = None,
        **kwargs: Any,
    ):
        super().__init__(config, retrievers, context_builder, boto_session, **kwargs)
        self.opensearch_retriever = retrievers.get(RetrieverType.OPENSEARCH.value)

    async def asearch(self, query: SearchQuery) -> SearchResult:
        start_time = time.time()
        logger.info(
            f"Simple search started - query: '{query.query[:50]}...' ('{query.search_type.value}')"
        )

        all_results = await self._execute_opensearch_retrieval(query)
        if not all_results:
            logger.warning(f"No results found for query: '{query.query[:50]}...'")
            return SearchResult(
                query=query,
                results=[],
                total_results=0,
                search_strategy="simple_search",
                processing_time=time.time() - start_time,
                metadata={},
            )

        final_results = self.hybrid_scorer.fuse_and_rerank_results(
            all_results,
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
        )

        processing_time = time.time() - start_time
        self._record_search_metrics(processing_time, len(final_results))

        logger.info(
            f"Search completed - retrieved: {len(final_results)} results in {processing_time:.3f}s"
        )

        return SearchResult(
            query=query,
            results=final_results,
            total_results=len(final_results),
            search_strategy="simple_search",
            processing_time=processing_time,
            metadata={},
        )

    async def _execute_opensearch_retrieval(
        self, query: SearchQuery
    ) -> dict[str, list[RetrievalResult]]:
        if not self.opensearch_retriever:
            return {}

        try:
            results = await self.opensearch_retriever.aretrieve(query)
            return {"opensearch_all": results} if results else {}
        except Exception as e:
            logger.error(f"OpenSearch retrieval failed: {e}")
            return {}

    def _record_search_metrics(
        self, processing_time: float, retrieved_count: int
    ) -> None:
        self._record_timing("processing_time", processing_time)
        self._record_metric("retrieved_count", retrieved_count)
