# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time

from aws_graphrag.adapters.retrieval.base import (
    BaseSearchStrategy,
)
from aws_graphrag.domain.models import (
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from aws_graphrag.domain.retrieval.strategy_registry import register_strategy
from aws_graphrag.shared import get_logger

logger = get_logger(__name__)


@register_strategy(SearchStrategy.SIMPLE, required_roles=(RetrieverRole.DOCUMENT,))
class SimpleSearchStrategy(BaseSearchStrategy):
    async def asearch(self, query: SearchQuery) -> SearchResult:
        start_time = time.time()
        logger.info(
            "Simple search started - query: '%s...' ('%s')",
            query.query[:50],
            query.search_type.value,
        )

        all_results = await self._retrieve_documents(query)
        if not all_results:
            logger.warning("No results found for query: '%s...'", query.query[:50])
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
            "Search completed - retrieved: %s results in %.3fs",
            len(final_results),
            processing_time,
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
            logger.error("OpenSearch retrieval failed: %s", e)
            return {}

    def _record_search_metrics(
        self, processing_time: float, retrieved_count: int
    ) -> None:
        self._record_timing("processing_time", processing_time)
        self._record_metric("retrieved_count", retrieved_count)
