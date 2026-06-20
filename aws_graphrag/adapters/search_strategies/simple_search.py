# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import time

from aws_graphrag.core import get_logger
from aws_graphrag.domain.models import (
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from aws_graphrag.domain.retrieval.strategy_registry import register_strategy
from aws_graphrag.retrieval.base import (
    BaseSearchStrategy,
)

logger = get_logger(__name__)


@register_strategy(SearchStrategy.SIMPLE, required_roles=(RetrieverRole.DOCUMENT,))
class SimpleSearchStrategy(BaseSearchStrategy):
    async def asearch(self, query: SearchQuery) -> SearchResult:
        start_time = time.time()
        logger.info(
            f"Simple search started - query: '{query.query[:50]}...' ('{query.search_type.value}')"
        )

        all_results = await self._retrieve_documents(query)
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

    async def _retrieve_documents(
        self, query: SearchQuery
    ) -> dict[str, list[RetrievalResult]]:
        if not self.document_retriever:
            return {}

        try:
            results = await self.document_retriever.aretrieve(query)
            return {"opensearch_all": results} if results else {}
        except Exception as e:
            logger.error(f"OpenSearch retrieval failed: {e}")
            return {}

    def _record_search_metrics(
        self, processing_time: float, retrieved_count: int
    ) -> None:
        self._record_timing("processing_time", processing_time)
        self._record_metric("retrieved_count", retrieved_count)
