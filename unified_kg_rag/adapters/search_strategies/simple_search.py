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
    configured_index_prefixes,
    held_artifacts_of,
    held_index_prefixes,
    index_artifacts,
    readable_artifacts,
)
from unified_kg_rag.domain.retrieval.source_fields import id_list
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import text_digest
from unified_kg_rag.shared.utils.concurrency import gather_or_cancel

logger = get_logger(__name__)

# The fusion bucket of each index (search.fusion.fusion_weights keys).
_BUCKETS: dict[IndexArtifact | None, str] = {
    IndexArtifact.TEXT_UNITS: "simple_text_units",
    IndexArtifact.ENTITIES: "simple_entities",
    IndexArtifact.RELATIONSHIP_VECTORS: "simple_relationships",
    IndexArtifact.CLAIMS: "simple_claims",
    IndexArtifact.COMMUNITY_REPORTS: "simple_community_reports",
    IndexArtifact.SUMMARY_TREE: "simple_summary_nodes",
}


@register_strategy(
    SearchStrategy.SIMPLE,
    required_roles=(RetrieverRole.DOCUMENT,),
    # It sweeps the indexes the namespace holds: the text units at least.
    required_artifacts=frozenset({IndexArtifact.TEXT_UNITS}),
)
class SimpleSearchStrategy(BaseSearchStrategy):
    methodology = Methodology.GRAPHRAG

    def filtered_index_prefixes(self, strategy: SearchStrategy) -> list[str]:
        # Every index the pipeline builds (_swept_prefixes); the chain keeps
        # those the queried namespace holds.
        prefixes = configured_index_prefixes(self.config)
        if IndexArtifact.SUMMARY_TREE in readable_artifacts(self.config, None):
            prefixes.append(self.config.indexing.opensearch.summary_nodes_index_prefix)
        return prefixes

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
                metadata={"rerank": rerank.summary(), "sources": {}},
            )

        final_results = await self._fuse_and_rerank(
            all_results,
            rerank_trace=rerank,
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
            # One bucket per index, never sharing an item: by rank (RRF) they
            # took equal shares whatever their scores, which a reranker
            # corrects but a run without one kept.
            fusion="min_max",
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
            metadata={
                "rerank": rerank.summary(),
                "sources": {name: len(hits) for name, hits in all_results.items()},
            },
        )

    async def _retrieve_documents(
        self, query: SearchQuery
    ) -> dict[str, list[RetrievalResult]]:
        """One fusion bucket per index swept (``_BUCKETS``), each its own
        ``top_k * retrieval_multiplier`` best: fused by their normalized
        scores (``combine_min_max``), reranked as one
        pool and cut to ``top_k`` (``_fuse_and_rerank``). One search over
        every index kept only ``top_k * retrieval_multiplier`` hits across
        them all, about two or three of each index before the reranker
        saw any."""
        retriever = self.document_retriever
        if retriever is None:
            return {}
        prefixes = await self._swept_prefixes(query)
        artifacts = index_artifacts(self.config)
        buckets = [
            _BUCKETS.get(artifacts.get(prefix), f"simple_{prefix}")
            for prefix in prefixes
        ]
        found = await gather_or_cancel(
            [
                self._safe_aretrieve(
                    retriever,
                    query.model_copy(update={"index_prefixes": [prefix]}),
                    f"Simple retrieval ({bucket})",
                )
                for prefix, bucket in zip(prefixes, buckets, strict=True)
            ]
        )
        return {
            bucket: results
            for bucket, results in zip(buckets, found, strict=True)
            if results
        }

    async def _swept_prefixes(self, query: SearchQuery) -> list[str]:
        """The indexes ``query`` sweeps: those the caller pinned, else every
        one the configured pipeline builds and the queried namespace holds
        (``held_index_prefixes``; any document backend honours the same
        gates), the summary tree's among them. A claim extraction that
        found no claim built no claims index: left out, as local leaves it
        (no "Index not found" WARNING on every query). The reports are left
        out under a ``source_kinds`` filter that keeps one kind, the summary
        nodes under a filter key their index does not store
        (``_summary_unfiltered_keys``)."""
        o = self.config.indexing.opensearch
        prefixes = id_list(query.index_prefixes)
        if not prefixes:
            held = held_artifacts_of(query.metadata)
            prefixes = (
                configured_index_prefixes(self.config)
                if held is None
                else held_index_prefixes(self.config, held)
            )
            if (
                o.claims_index_prefix in prefixes
                and not await self._claims_index_exists(query.suffix)
            ):
                prefixes = [p for p in prefixes if p != o.claims_index_prefix]
            if self._reads_artifact(query, IndexArtifact.SUMMARY_TREE):
                prefixes = [*prefixes, o.summary_nodes_index_prefix]
        if self._source_kinds_skip_reports(query) and (
            o.community_reports_index_prefix in prefixes
        ):
            # A community report carries no source kind and summarizes
            # seed-only members too, so a source_kinds filter that keeps one
            # kind cannot be honoured on it (local and DRIFT read no report
            # then either).
            logger.info(
                "Community reports skipped: the source_kinds filter keeps one "
                "source kind and a report carries none"
            )
            prefixes = [p for p in prefixes if p != o.community_reports_index_prefix]
        if o.summary_nodes_index_prefix in prefixes and (
            unfiltered := self._summary_unfiltered_keys(query)
        ):
            logger.info(
                "Simple search reads no summary: the summary index does not "
                "store the filter keys %s",
                ", ".join(unfiltered),
            )
            prefixes = [p for p in prefixes if p != o.summary_nodes_index_prefix]
        return prefixes
