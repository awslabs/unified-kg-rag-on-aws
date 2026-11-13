# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HippoRAG 2 retrieval as a strategy: ``ppr`` (opt-in, docs/design.md §6.4).

The question's facts (relationships-index hits) and passages (text-unit
hits), both under the request's search (BM25 + kNN fused by default), seed
a Personalized PageRank over the graph neighbourhood of the facts' entities
and the passages those entities cite; the passages it ranks, the passage
hits and the facts are fused, reranked and answered as every strategy's
results are. No query-time LLM call besides the answer. Reimplemented from
HippoRAG 2 (Gutiérrez et al., ICML 2025; OSU-NLP-Group/HippoRAG, MIT).
"""

from __future__ import annotations

import time
from collections.abc import Mapping

from langchain_core.runnables import RunnableConfig

from unified_kg_rag.adapters.retrieval.ppr_ranking import (
    PPR_SEED_TEXT_UNITS,
    PPRRankingMixin,
    endpoint_ids,
    entity_id,
)
from unified_kg_rag.adapters.retrieval.strategy_registry import register_strategy
from unified_kg_rag.domain.models import (
    IndexArtifact,
    Methodology,
    RetrievalResult,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SectionType,
)
from unified_kg_rag.domain.retrieval.ppr import seed_weights
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import text_digest
from unified_kg_rag.shared.utils.concurrency import gather_or_cancel

logger = get_logger(__name__)

# The fusion buckets this strategy emits (search.fusion.fusion_weights).
PPR_PASSAGES = "ppr_passages"
PPR_FACTS = "ppr_facts"


def _relative(hits: list[RetrievalResult]) -> list[float]:
    """The hits' scores divided by the best (0 when none is positive)."""
    scores = [max(float(hit.score or 0.0), 0.0) for hit in hits]
    best = max(scores, default=0.0)
    return [score / best if best > 0 else 0.0 for score in scores]


@register_strategy(SearchStrategy.PPR)
class PPRSearchStrategy(PPRRankingMixin):
    """HippoRAG 2's fact- and passage-seeded PageRank over the graph."""

    methodology = Methodology.HIPPORAG

    def filtered_index_prefixes(self, strategy: SearchStrategy) -> list[str]:
        # Facts, passages, and the entities a filtered walk or a failed
        # expansion reads.
        o = self.config.indexing.opensearch
        return self._built_prefixes(
            o.entities_index_prefix,
            o.relationships_index_prefix,
            o.text_units_index_prefix,
        )

    async def asearch(
        self, query: SearchQuery, config: RunnableConfig | None = None
    ) -> SearchResult:
        start_time = time.time()
        rerank = self._new_rerank_trace()
        settings = self.config.search.ppr_search
        logger.info(
            "PPR search started - query: %s ('%s')",
            text_digest(query.query),
            query.search_type.value,
        )
        facts_built = self._reads_artifact(query, IndexArtifact.RELATIONSHIP_VECTORS)
        entity_k = settings.entity_seed_top_k or (
            0 if facts_built else settings.fact_top_k
        )
        facts, passage_hits, entity_hits = await gather_or_cancel(
            [
                self._fact_hits(query, settings.fact_top_k),
                self._passage_seed_hits(query),
                self._entity_seed_hits(query, entity_k),
            ]
        )
        # Each kind of hit scaled to its best (HippoRAG 2 min-max normalizes
        # the fact scores), so the passage weight means the same under any
        # search type.
        scored: list[tuple[float, list[str]]] = [
            (score, ends)
            for hit, score in zip(facts, _relative(facts), strict=True)
            if (ends := endpoint_ids(hit))
        ] + [
            (score, [str(hit.source)])
            for hit, score in zip(entity_hits, _relative(entity_hits), strict=True)
        ]
        budget: dict[str, float] = {}
        for score, ends in scored:
            for end in ends:
                budget[end] = budget.get(end, 0.0) + max(score, 0.0)
        seed_ids = list(budget)

        entities = await self._neighbourhood(query, seed_ids, budget)

        def reset(counts: Mapping[str, int]) -> dict[str, float]:
            # Only entities in the graph seed (and take a link_top_k slot), as
            # HippoRAG 2 seeds the phrases its graph holds; weighed over the
            # document units each cites (counted off the event loop).
            return seed_weights(
                [(score, [e for e in ends if e in counts]) for score, ends in scored],
                counts,
                settings.link_top_k,
            )

        ranked, stats = await self._ppr_rank(
            query,
            entities,
            reset,
            passage_hits,
            frequency_threshold=settings.entity_frequency_threshold,
        )
        if not ranked:
            # Nothing seeded a passage: the passage hits as found.
            ranked = [str(hit.source) for hit in passage_hits if hit.source]
        quota = self._per_type_quota(query.top_k)
        passages = await self._passages_in_order(
            query, ranked, quota[SectionType.TEXT.value], passage_hits
        )

        buckets: dict[str, list[RetrievalResult]] = {}
        for name, results in (
            (PPR_PASSAGES, passages),
            (PPR_SEED_TEXT_UNITS, passage_hits),
            (PPR_FACTS, facts),
        ):
            if results:
                buckets[name] = list(results)
        final_results = (
            await self._fuse_and_rerank(
                buckets,
                rerank_trace=rerank,
                top_k=query.top_k,
                retrieval_multiplier=query.retrieval_multiplier,
                query=query.query,
                per_type_quota={
                    SectionType.TEXT.value: quota[SectionType.TEXT.value],
                    SectionType.RELATIONSHIP.value: settings.fact_top_k,
                },
                rerank_only_types=(
                    {SectionType.TEXT.value} if settings.rerank else set()
                ),
            )
            if buckets
            else []
        )
        processing_time = time.time() - start_time
        logger.info(
            "PPR search completed - %s results in %.3fs",
            len(final_results),
            processing_time,
        )
        return SearchResult(
            query=query,
            results=final_results,
            total_results=len(final_results),
            search_strategy="ppr_search",
            processing_time=processing_time,
            metadata={
                "methodology": Methodology.HIPPORAG.value,
                "fact_count": len(facts),
                "seed_entity_count": len(seed_ids),
                "ppr": stats,
                "sources": {name: len(results) for name, results in buckets.items()},
                "rerank": rerank.summary(),
            },
        )

    async def _neighbourhood(
        self, query: SearchQuery, seed_ids: list[str], budget: dict[str, float]
    ) -> list[RetrievalResult]:
        """The seeds and the entities the expansion reaches from them
        (``search.ppr_search.max_hops`` deep, at most
        ``max_neighbourhood_entities`` besides the seeds, each seed's share
        in proportion to its fact score). Seeds the expansion did not return
        (a failed walk) are read from the entities index."""
        if not seed_ids:
            return []
        settings = self.config.search.ppr_search
        walk = query.model_copy(
            update={
                "top_k": settings.max_neighbourhood_entities,
                "retrieval_multiplier": 1,
                "max_hops": settings.max_hops,
            }
        )
        expanded = await self._expand_via_graph(
            walk,
            seed_ids,
            seed_weights=budget if all(v > 0 for v in budget.values()) else None,
        )
        reached = {entity_id(node) for node in expanded}
        if missing := [seed for seed in seed_ids if seed not in reached]:
            expanded = [*expanded, *await self._entities_by_id(query, missing)]
        return expanded
