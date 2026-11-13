# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HippoRAG 2's Personalized PageRank passage ranking over the stores.

The store-facing half of ``domain.retrieval.ppr``: the seed searches (facts,
passages, entities), the entity + passage graph of a neighbourhood the
expansion reached and its PageRank, shared by the ``ppr`` strategy and
``local``'s ``text_unit_ranking: ppr`` (docs/design.md §6.4). Every search
runs under the caller's suffix and filters with the request's
``search_type`` (BM25 + kNN fused by default).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from unified_kg_rag.adapters.retrieval.base import BaseSearchStrategy
from unified_kg_rag.domain.ingestion.seed_graph import is_seed_unit
from unified_kg_rag.domain.models import (
    GraphEdge,
    IndexArtifact,
    RetrievalResult,
    SearchQuery,
    SearchType,
)
from unified_kg_rag.domain.retrieval.lineage import merge as merge_lineages
from unified_kg_rag.domain.retrieval.ppr import (
    PassageGraph,
    PPRResult,
    passage_weights,
)
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

# The fusion bucket of the passage hits that seed the PageRank.
PPR_SEED_TEXT_UNITS = "ppr_seed_text_units"


def document_unit_ids(node: RetrievalResult) -> list[str]:
    """The document text units an entity (or relationship) cites, each once,
    in id order: a seed graph's units are no passages."""
    ids = (node.metadata or {}).get("text_unit_ids") or []
    if not isinstance(ids, list):
        return []
    return sorted({str(u) for u in ids if u and not is_seed_unit(str(u))})


def entity_id(node: RetrievalResult) -> str:
    return str((node.metadata or {}).get("id") or node.source or "")


def endpoint_ids(hit: RetrievalResult) -> list[str]:
    """A relationship hit's endpoint entity ids (source first)."""
    metadata = hit.metadata or {}
    return [
        str(value)
        for value in (metadata.get("source_id"), metadata.get("target_id"))
        if isinstance(value, str) and value
    ]


class PPRRankingMixin(BaseSearchStrategy):
    """The PageRank passage ranking a strategy mixes in (no ``asearch``)."""

    async def _index_hits(
        self,
        query: SearchQuery,
        prefix: str,
        top_k: int,
        label: str,
        search_type: SearchType | None = None,
    ) -> list[RetrievalResult]:
        """The ``top_k`` hits of the question on ``prefix``'s index, under
        ``search_type`` (the request's by default)."""
        if not self.document_retriever or top_k < 1 or not query.query:
            return []
        return await self._safe_aretrieve(
            self.document_retriever,
            SearchQuery(
                query=query.query,
                search_type=search_type or query.search_type,
                top_k=top_k,
                retrieval_multiplier=1,
                index_prefixes=[prefix],
                suffix=query.suffix,
                filters=self._scoped_filters(query),
            ),
            label,
        )

    async def _passage_seed_hits(self, query: SearchQuery) -> list[RetrievalResult]:
        """The text-unit hits of the question that seed passage nodes."""
        return await self._index_hits(
            query,
            self.config.indexing.opensearch.text_units_index_prefix,
            self.config.search.ppr_search.passage_seed_top_k,
            "PPR passage seeds",
        )

    async def _fact_hits(self, query: SearchQuery, top_k: int) -> list[RetrievalResult]:
        """The relationships-index hits of the question (the facts), under
        ``search.ppr_search.fact_search_type`` (kNN by default, HippoRAG 2's
        query-to-fact similarity); none without the relationship vector
        index."""
        if not self._reads_artifact(query, IndexArtifact.RELATIONSHIP_VECTORS):
            return []
        return await self._index_hits(
            query,
            self.config.indexing.opensearch.relationships_index_prefix,
            top_k,
            "PPR fact seeds",
            self.config.search.ppr_search.fact_search_type,
        )

    async def _entity_seed_hits(
        self, query: SearchQuery, top_k: int
    ) -> list[RetrievalResult]:
        return [
            hit
            for hit in await self._index_hits(
                query,
                self.config.indexing.opensearch.entities_index_prefix,
                top_k,
                "PPR entity seeds",
            )
            if hit.source
        ]

    async def _entities_by_id(
        self, query: SearchQuery, ids: list[str]
    ) -> list[RetrievalResult]:
        """The entities-index documents of ``ids`` under the caller's
        filters (an id that does not pass them is not returned)."""
        if not self.document_retriever or not ids:
            return []
        return await self._fetch_by_ids(
            self.document_retriever,
            SearchQuery(
                query="",
                search_type=SearchType.LEXICAL,
                top_k=len(ids),
                retrieval_multiplier=1,
                index_prefixes=[self.config.indexing.opensearch.entities_index_prefix],
                suffix=query.suffix,
                filters=self._scoped_filters(query, id=ids),
            ),
            "PPR entity read",
        )

    async def _ppr_rank(
        self,
        query: SearchQuery,
        entities: Sequence[RetrievalResult],
        entity_reset: Mapping[str, float],
        passage_hits: Sequence[RetrievalResult],
        *,
        frequency_threshold: int,
    ) -> tuple[list[str], dict[str, Any]]:
        """The passages by PageRank score and the run's counts.

        The graph: ``entities`` (each once), the relationships between them
        (``_neighbourhood_edges``), the document units each cites (none for
        an entity citing more than ``frequency_threshold``, a seed too) and
        the ``passage_hits``. The reset: ``entity_reset`` (entity ids; only
        those in the graph) and the passage hits at
        ``search.ppr_search.passage_node_weight`` times their min-max
        normalized score. ``([], stats)`` when nothing seeds a passage.
        """
        settings = self.config.search.ppr_search
        nodes: dict[str, RetrievalResult] = {}
        for node in entities:
            if (key := entity_id(node)) and key not in nodes:
                nodes[key] = node
        edges: list[GraphEdge] = await self._neighbourhood_edges(
            query, list(nodes), settings.max_edges_per_vertex
        )
        graph = PassageGraph()
        for edge in edges:
            graph.add_relationship(edge.source_id, edge.target_id)
        linked: list[tuple[list[str], float]] = []
        for key, node in nodes.items():
            graph.add_entity(key)
            units = document_unit_ids(node)
            if units and len(units) <= frequency_threshold:
                graph.link_passages(key, units)
                linked.append((units, float(node.score or 0.0)))
        hits = {
            str(hit.source): float(hit.score or 0.0)
            for hit in passage_hits
            if hit.source and not is_seed_unit(str(hit.source))
        }
        for unit in hits:
            graph.add_passage(unit)
        reset = {k: v for k, v in entity_reset.items() if k in nodes and v > 0}

        def rank() -> tuple[list[tuple[str, float]], PPRResult | None]:
            return graph.rank_passages(
                reset,
                passage_weights(hits, settings.passage_node_weight),
                damping=settings.damping,
                max_iterations=settings.max_iterations,
                tolerance=settings.tolerance,
                # Equal scores (the passages no seed reaches score 0) in
                # local's support order, not by id.
                tie_order=merge_lineages(
                    [units for units, _ in linked],
                    order="support",
                    scores=[score for _, score in linked],
                ),
            )

        # Milliseconds of numpy, and the lineage merge, off the event loop
        # (tens on thousands of passages).
        ranked, result = await asyncio.to_thread(rank)
        stats: dict[str, Any] = {
            "entities": len(nodes),
            "relationships": len(edges),
            "linked_entities": len(linked),
            "passages": len(graph.passages),
            "entity_seeds": len(reset),
            "passage_seeds": len(hits),
            "iterations": result.iterations if result else 0,
            "converged": result.converged if result else False,
        }
        if result is not None and not result.converged:
            logger.warning(
                "PPR did not converge in %s iterations (%s nodes): the passages "
                "are ranked by the last iteration's scores; raise "
                "search.ppr_search.max_iterations or tolerance",
                result.iterations,
                graph.node_count,
            )
        return [unit for unit, _ in ranked], stats

    async def _passages_in_order(
        self,
        query: SearchQuery,
        ranked: list[str],
        quota: int,
        known: Sequence[RetrievalResult] = (),
    ) -> list[RetrievalResult]:
        """The first ``quota`` of the ``ranked`` text units that pass the
        caller's filters, scored in that order (``_restore_rank``); the ones
        among ``known`` (search hits) are not fetched again."""
        chosen = await self._passing_ranked_units(query, ranked, quota)
        if not chosen:
            return []
        have = {str(hit.source): hit for hit in known if hit.source}
        missing = [unit for unit in chosen if unit not in have]
        fetched: list[RetrievalResult] = []
        if missing and self.document_retriever is not None:
            fetched = await self._fetch_by_ids(
                self.document_retriever,
                SearchQuery(
                    query="",
                    search_type=SearchType.LEXICAL,
                    top_k=len(missing),
                    retrieval_multiplier=1,
                    index_prefixes=[
                        self.config.indexing.opensearch.text_units_index_prefix
                    ],
                    suffix=query.suffix,
                    filters=self._scoped_filters(query, id=missing),
                ),
                "PPR passage fetch",
            )
        found = [have[unit] for unit in chosen if unit in have] + fetched
        return self._restore_rank(found, chosen)
