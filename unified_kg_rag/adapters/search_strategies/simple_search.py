# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time

from langchain_core.runnables import RunnableConfig

from unified_kg_rag.adapters.retrieval.base import (
    BaseSearchStrategy,
)
from unified_kg_rag.adapters.retrieval.strategy_registry import register_strategy
from unified_kg_rag.domain.models import (
    IndexArtifact,
    Methodology,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from unified_kg_rag.domain.retrieval.index_prefixes import (
    all_index_prefixes,
    configured_index_prefixes,
    held_artifacts_of,
    held_index_prefixes,
)
from unified_kg_rag.domain.retrieval.source_fields import id_list
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import text_digest

logger = get_logger(__name__)


@register_strategy(
    SearchStrategy.SIMPLE,
    required_roles=(RetrieverRole.DOCUMENT,),
    # It sweeps the indexes the namespace holds: the text units at least.
    required_artifacts=frozenset({IndexArtifact.TEXT_UNITS}),
)
class SimpleSearchStrategy(BaseSearchStrategy):
    methodology = Methodology.GRAPHRAG

    def filtered_index_prefixes(self, strategy: SearchStrategy) -> list[str]:
        # Every index the pipeline builds (_apply_claim_gate).
        return configured_index_prefixes(self.config)

    async def asearch(
        self, query: SearchQuery, config: RunnableConfig | None = None
    ) -> SearchResult:
        start_time = time.time()
        rerank = self._new_rerank_trace()
        logger.info(
            "Simple search started - query: %s ('%s')",
            text_digest(query.query),
            query.search_type.value,
        )

        all_results = await self._retrieve_documents(query)
        if not all_results:
            logger.warning("No results found for query: %s", text_digest(query.query))
            return SearchResult(
                query=query,
                results=[],
                total_results=0,
                search_strategy="simple_search",
                processing_time=time.time() - start_time,
                metadata={"rerank": rerank.summary()},
            )

        final_results = await self._fuse_and_rerank(
            all_results,
            rerank_trace=rerank,
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
        )

        processing_time = time.time() - start_time

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
            metadata={"rerank": rerank.summary()},
        )

    async def _retrieve_documents(
        self, query: SearchQuery
    ) -> dict[str, list[RetrievalResult]]:
        if not self.document_retriever:
            return {}

        search_query = await self._apply_claim_gate(query)
        if self._source_kinds_skip_reports(query):
            without = self._without_reports(search_query)
            if without is None:
                return {}
            search_query = without
        results = await self._safe_aretrieve(
            self.document_retriever, search_query, "Document retrieval"
        )
        return {"vector_all": results} if results else {}

    def _without_reports(self, query: SearchQuery) -> SearchQuery | None:
        # A community report carries no source kind and summarizes seed-only
        # members too, so a source_kinds filter that keeps one kind cannot
        # be honoured on it (local and DRIFT read no report then either).
        # None: the caller pinned the reports index alone.
        reports = self.config.indexing.opensearch.community_reports_index_prefix
        listed = id_list(query.index_prefixes) or configured_index_prefixes(self.config)
        prefixes = [prefix for prefix in listed if prefix != reports]
        if len(prefixes) < len(listed):
            logger.info(
                "Community reports skipped: the source_kinds filter keeps one "
                "source kind and a report carries none"
            )
        if not prefixes:
            return None
        return query.model_copy(update={"index_prefixes": prefixes})

    async def _apply_claim_gate(self, query: SearchQuery) -> SearchQuery:
        # Simple search sweeps every index by default. When the caller hasn't
        # pinned index_prefixes and some optional index is not built by the
        # configured pipeline (claims off, no relationship vector index, no
        # community detection) or not held by the queried namespace (a
        # lightrag-only one has no reports, a raptor-only one no entities),
        # pin the sweep to the indices that do exist so any document backend
        # honours the same gates as OpenSearchRetriever. A claim extraction
        # that found no claim built no claims index: left out too, as local
        # leaves it (no "Index not found" WARNING on every query).
        if query.index_prefixes:
            return query
        held = held_artifacts_of(query.metadata)
        configured = (
            configured_index_prefixes(self.config)
            if held is None
            else held_index_prefixes(self.config, held)
        )
        claims = self.config.indexing.opensearch.claims_index_prefix
        if claims in configured and not await self._claims_index_exists(query.suffix):
            configured = [prefix for prefix in configured if prefix != claims]
        if configured == all_index_prefixes(self.config):
            return query
        return query.model_copy(update={"index_prefixes": configured})
