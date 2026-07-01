# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time

from unified_kg_rag.adapters.retrieval.base import (
    BaseSearchStrategy,
    is_fatal_retrieval_error,
)
from unified_kg_rag.domain.models import (
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from unified_kg_rag.domain.retrieval.strategy_registry import register_strategy
from unified_kg_rag.shared import get_logger

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
            results = await self.document_retriever.aretrieve(
                self._apply_claim_gate(query)
            )
            return {"opensearch_all": results} if results else {}
        except Exception as e:
            if is_fatal_retrieval_error(e):
                raise
            logger.error("OpenSearch retrieval failed: %s", e)
            return {}

    def _apply_claim_gate(self, query: SearchQuery) -> SearchQuery:
        # Simple search sweeps every index by default (index_prefixes=None ->
        # all mappings, which includes the claims index). Keep claims inclusion
        # consistently gated on the enabled flag: when the caller hasn't pinned
        # index_prefixes and claim extraction is off, restrict the sweep to the
        # non-claims indexes so a claims-off run never queries that index.
        opensearch = self.config.indexing.opensearch
        if query.index_prefixes or self.config.processing.claim_extraction.enabled:
            return query

        non_claims_prefixes = [
            opensearch.text_units_index_prefix,
            opensearch.entities_index_prefix,
            opensearch.relationships_index_prefix,
            opensearch.community_reports_index_prefix,
        ]
        return query.model_copy(update={"index_prefixes": non_claims_prefixes})

    def _record_search_metrics(
        self, processing_time: float, retrieved_count: int
    ) -> None:
        self._record_timing("processing_time", processing_time)
        self._record_metric("retrieved_count", retrieved_count)
