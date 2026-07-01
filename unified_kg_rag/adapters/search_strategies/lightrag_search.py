# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LightRAG dual-level keyword search strategy.

Implements LightRAG's local/global/hybrid/mix/naive retrieval on top of
unified-kg-rag-on-aws's *shared* infrastructure rather than as a separate, reduced path:

- low-level keywords (``ll_keywords``) -> entities index (lexical + vector),
- high-level keywords (``hl_keywords``) -> relationships index (lexical + vector),
- the entity hits are expanded through Neptune graph traversal,
- ``mix`` additionally pulls the source chunks referenced by the matched
  entities/relationships (following ``text_unit_ids`` lineage, ranked by how
  many matched entities/relationships cite each chunk — LightRAG's
  ``_find_related_text_unit_from_entities``/``_from_relationships``) and blends
  a naive vector chunk retrieval,
- everything is fused and reranked via the shared :class:`HybridScorer`
  (BM25 lexical + vector semantic + graph + RRF + Bedrock rerank).

So a LightRAG-mode query enjoys the same hybrid scoring, multilingual handling,
and caching as the GraphRAG strategies — only the retrieval algorithm differs.
"""

from __future__ import annotations

import time
from typing import Any

import boto3

from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
    is_fatal_retrieval_error,
)
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
)
from unified_kg_rag.domain.retrieval.strategy_registry import register_strategy
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)


@register_strategy(SearchStrategy.MIX)
@register_strategy(SearchStrategy.HYBRID)
@register_strategy(SearchStrategy.NAIVE)
class LightRAGSearchStrategy(BaseSearchStrategy):
    """Dual-level keyword retrieval (LightRAG) over the shared hybrid stack.

    The same class serves three modes, distinguished by the resolved
    :class:`SearchStrategy` passed via ``query.metadata['lightrag_mode']``
    (default ``mix``):

    - ``naive``: vector chunk retrieval only (no graph).
    - ``hybrid``: ll->entities + hl->relationships + graph expansion.
    - ``mix``: hybrid graph retrieval blended with naive chunk retrieval.

    Backends are accessed only through the abstract GRAPH / DOCUMENT retriever
    roles (``self.graph_retriever`` / ``self.document_retriever``).
    """

    def __init__(
        self,
        config: Config,
        retrievers: dict[str, BaseGraphRAGRetriever],
        boto_session: boto3.Session | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, retrievers, boto_session, **kwargs)
        self._os_config = config.indexing.opensearch

    def _mode(self, query: SearchQuery) -> str:
        return str(query.metadata.get("lightrag_mode", SearchStrategy.MIX.value))

    def _apply_keyword_fallback(self, query: SearchQuery) -> SearchQuery:
        """Force the raw query as a low-level keyword when both lists are empty.

        Without this, a hybrid/mix query whose keyword extraction returned
        nothing would retrieve from no graph source at all. The length gate is
        config-driven (``search.lightrag_search.raw_query_fallback_max_len``).
        """
        if query.hl_keywords or query.ll_keywords:
            return query
        max_len = self.config.search.lightrag_search.raw_query_fallback_max_len
        if query.query and len(query.query) < max_len:
            logger.warning(
                "No keywords extracted; falling back to raw query as ll_keyword"
            )
            fallback = query.model_copy(deep=True)
            fallback.ll_keywords = [query.query]
            return fallback
        return query

    async def asearch(self, query: SearchQuery) -> SearchResult:
        start_time = time.time()
        mode = self._mode(query)
        logger.info(
            "LightRAG search started - mode: '%s', query: '%s...'",
            mode,
            query.query[:50],
        )

        results_by_source: dict[str, list[RetrievalResult]] = {}

        if mode == SearchStrategy.NAIVE.value:
            results_by_source.update(await self._retrieve_chunks(query))
        else:
            # hybrid / mix: dual-level keyword retrieval + graph expansion.
            query = self._apply_keyword_fallback(query)
            if query.ll_keywords:
                results_by_source.update(await self._retrieve_entities(query))
            if query.hl_keywords:
                results_by_source.update(await self._retrieve_relationships(query))

            # Seed graph expansion from BOTH the low-level entity hits and the
            # endpoints of the high-level relationship hits. Without the latter,
            # an hl-only (purely thematic/global) query — relationships but no
            # entities — gets no graph expansion and no entity grounding, which
            # diverges from LightRAG (its global mode reaches entities via the
            # matched relationships' endpoints).
            seed_entity_ids = list(
                dict.fromkeys(
                    self._get_ids(results_by_source.get("lightrag_entities", []), "id")
                    + self._relationship_endpoint_ids(
                        results_by_source.get("lightrag_relationships", [])
                    )
                )
            )
            if seed_entity_ids:
                expanded = await self._expand_via_graph(query, seed_entity_ids)
                if expanded:
                    results_by_source["graph_entities"] = expanded

            if mode == SearchStrategy.MIX.value:
                # KG-grounded chunks: follow text_unit_ids lineage from the
                # matched entities/relationships to their source chunks (ranked
                # by citation count), then blend a naive vector chunk query.
                linked = await self._retrieve_linked_chunks(
                    query,
                    results_by_source.get("lightrag_entities", []),
                    results_by_source.get("lightrag_relationships", []),
                )
                if linked:
                    results_by_source.update(linked)
                results_by_source.update(await self._retrieve_chunks(query))

        final_results = self.hybrid_scorer.fuse_and_rerank_results(
            results_by_source,
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
        )

        processing_time = time.time() - start_time
        self._record_timing("processing_time", processing_time)
        self._record_metric("retrieved_count", len(final_results))

        logger.info(
            "LightRAG search completed - %d results in %.3fs",
            len(final_results),
            processing_time,
        )

        return SearchResult(
            query=query,
            results=final_results,
            total_results=len(final_results),
            search_strategy=f"lightrag_{mode}",
            processing_time=processing_time,
            metadata={
                "mode": mode,
                "hl_keyword_count": len(query.hl_keywords),
                "ll_keyword_count": len(query.ll_keywords),
                "sources": {k: len(v) for k, v in results_by_source.items()},
            },
        )

    async def _retrieve_entities(
        self, query: SearchQuery
    ) -> dict[str, list[RetrievalResult]]:
        """Low-level keywords -> entities index (LightRAG local component)."""
        if not self.document_retriever:
            return {}
        search_query = SearchQuery(
            query=", ".join(query.ll_keywords),
            search_type=query.search_type,
            top_k=query.top_k,
            index_prefixes=[self._os_config.entities_index_prefix],
            suffix=query.suffix,
        )
        try:
            results = await self.document_retriever.aretrieve(search_query)
            return {"lightrag_entities": results}
        except Exception as e:
            if is_fatal_retrieval_error(e):
                raise
            logger.error("Entity retrieval (ll_keywords) failed: %s", e)
            return {}

    async def _retrieve_relationships(
        self, query: SearchQuery
    ) -> dict[str, list[RetrievalResult]]:
        """High-level keywords -> relationships index (LightRAG global component)."""
        if not self.document_retriever:
            return {}
        search_query = SearchQuery(
            query=", ".join(query.hl_keywords),
            search_type=query.search_type,
            top_k=query.top_k,
            index_prefixes=[self._os_config.relationships_index_prefix],
            suffix=query.suffix,
        )
        try:
            results = await self.document_retriever.aretrieve(search_query)
            return {"lightrag_relationships": results}
        except Exception as e:
            if is_fatal_retrieval_error(e):
                raise
            logger.error("Relationship retrieval (hl_keywords) failed: %s", e)
            return {}

    async def _retrieve_chunks(
        self, query: SearchQuery
    ) -> dict[str, list[RetrievalResult]]:
        """Naive vector chunk retrieval over the text-units index."""
        if not self.document_retriever:
            return {}
        search_query = SearchQuery(
            query=query.query,
            search_type=query.search_type,
            top_k=query.top_k,
            index_prefixes=[self._os_config.text_units_index_prefix],
            suffix=query.suffix,
        )
        try:
            results = await self.document_retriever.aretrieve(search_query)
            return {"lightrag_chunks": results}
        except Exception as e:
            if is_fatal_retrieval_error(e):
                raise
            logger.error("Chunk retrieval failed: %s", e)
            return {}

    @staticmethod
    def _collect_linked_chunk_ids(
        entity_results: list[RetrievalResult],
        relationship_results: list[RetrievalResult],
        limit: int,
    ) -> list[str]:
        """Rank source chunk ids by how many matched KG items cite them.

        Mirrors LightRAG's ``_find_related_text_unit_from_entities``/
        ``_from_relationships``: each matched entity/relationship contributes the
        chunks in its ``text_unit_ids`` lineage; chunks cited by more matched
        items are more central to the query and ranked first. Ties keep first-
        seen order so ranking is deterministic.
        """
        counts: dict[str, int] = {}
        order: dict[str, int] = {}
        seq = 0
        for result in [*entity_results, *relationship_results]:
            metadata = result.metadata or {}
            unit_ids = metadata.get("text_unit_ids")
            if not isinstance(unit_ids, list):
                continue
            for uid in unit_ids:
                if not isinstance(uid, str) or not uid:
                    continue
                if uid not in counts:
                    counts[uid] = 0
                    order[uid] = seq
                    seq += 1
                counts[uid] += 1
        ranked = sorted(counts, key=lambda uid: (-counts[uid], order[uid]))
        return ranked[:limit]

    async def _retrieve_linked_chunks(
        self,
        query: SearchQuery,
        entity_results: list[RetrievalResult],
        relationship_results: list[RetrievalResult],
    ) -> dict[str, list[RetrievalResult]]:
        """Fetch the source chunks cited by the matched entities/relationships.

        Collects ``text_unit_ids`` lineage from the entity/relationship hits,
        ranks chunk ids by citation count, and fetches the top chunks by id from
        the text-units index. Degrades to no section if lineage is absent (e.g.
        an index built before lineage was added) or retrieval fails.
        """
        if not self.document_retriever:
            return {}
        chunk_ids = self._collect_linked_chunk_ids(
            entity_results, relationship_results, limit=query.top_k
        )
        if not chunk_ids:
            return {}
        search_query = SearchQuery(
            query="",
            search_type=SearchType.LEXICAL,
            top_k=len(chunk_ids),
            index_prefixes=[self._os_config.text_units_index_prefix],
            suffix=query.suffix,
            filters={"id": chunk_ids},
        )
        try:
            results = await self.document_retriever.aretrieve(search_query)
            return {"lightrag_linked_chunks": results} if results else {}
        except Exception as e:
            logger.error("Linked chunk retrieval failed: %s", e)
            return {}

    @staticmethod
    def _relationship_endpoint_ids(
        relationship_results: list[RetrievalResult],
    ) -> list[str]:
        """Collect source/target entity ids from relationship hits.

        Relationship documents carry their endpoint entity ids in metadata
        (``source_id``/``target_id``); these ground a high-level (relationship)
        hit back to the graph so it can be expanded like an entity hit.
        """
        endpoint_ids: list[str] = []
        for result in relationship_results:
            metadata = result.metadata or {}
            for field in ("source_id", "target_id"):
                value = metadata.get(field)
                if isinstance(value, str) and value:
                    endpoint_ids.append(value)
        return endpoint_ids

    async def _expand_via_graph(
        self, query: SearchQuery, seed_entity_ids: list[str]
    ) -> list[RetrievalResult]:
        """Expand seed entities through the graph (shared with GraphRAG local)."""
        if not self.graph_retriever or not seed_entity_ids:
            return []
        search_query = query.model_copy(deep=True)
        search_query.search_type = SearchType.HYBRID
        search_query.label_prefixes = [self.config.indexing.neptune.entity_label_prefix]
        search_query.entity_focus = []
        search_query.filters = (search_query.filters or {}).copy()
        search_query.filters["id"] = seed_entity_ids
        try:
            return await self.graph_retriever.aretrieve(search_query)
        except Exception as e:
            logger.error("Neptune expansion failed: %s", e)
            return []
