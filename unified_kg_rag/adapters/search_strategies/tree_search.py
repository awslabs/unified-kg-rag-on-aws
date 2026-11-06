# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``tree``: RAPTOR's summary tree searched hybrid (docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in).

RAPTOR's "collapsed tree" ranks every node of every level in one pool by
cosine similarity to the query. Here the pool is fused from up to four
buckets, each under the request's ``search_type`` (BM25 + kNN by default):

- ``tree_text_units``: the text units (``naive``'s search);
- ``tree_summaries``: every level of the summary tree;
- ``tree_graph_summaries`` / ``tree_graph_text_units`` (``graph_signal``,
  where the namespace builds the entities index): the summaries and the text
  units citing the text units of the entities a hybrid search finds with the
  raw query (no model call), scored by the seeds they cover.

The buckets are fused by RRF and the leaves and summaries reranked together
(the one comparator across the two), at most ``max_nodes_per_level``
summaries of a level are kept, and each is expanded to ``expand_children``
passages of its own lineage, so the answer cites original text. A summary
is index content a model wrote at ingestion, as a community report is: it
is reported as a source with its lineage, never as ``synthesized``.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from typing import Any

from langchain_core.runnables import RunnableConfig

from unified_kg_rag.adapters.retrieval.base import BaseSearchStrategy
from unified_kg_rag.adapters.retrieval.strategy_registry import register_strategy
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.domain.ingestion.methodologies import artifacts_to_build
from unified_kg_rag.domain.ingestion.seed_graph import is_seed_unit
from unified_kg_rag.domain.models import (
    IndexArtifact,
    Methodology,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
)
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import text_digest
from unified_kg_rag.shared.utils.concurrency import gather_bounded, gather_or_cancel

logger = get_logger(__name__)

TEXT_BUCKET = "tree_text_units"
SUMMARY_BUCKET = "tree_summaries"
GRAPH_SUMMARY_BUCKET = "tree_graph_summaries"
GRAPH_TEXT_BUCKET = "tree_graph_text_units"
# A summary's expanded passages sit just below it in the fused order.
_EXPANSION_STEP = 1e-6
# Levels of child lookups that resolve a cut lineage at most (max_levels).
_MAX_LINEAGE_DEPTH = 10


def _units(result: RetrievalResult) -> list[str]:
    """The document text units a result cites (seed units left out)."""
    return [
        str(unit)
        for unit in (result.metadata or {}).get("text_unit_ids") or []
        if unit and not is_seed_unit(str(unit))
    ]


@register_strategy(SearchStrategy.TREE, required_roles=(RetrieverRole.DOCUMENT,))
class TreeSearchStrategy(BaseSearchStrategy):
    """Collapsed-tree retrieval over the summary tree and the text units."""

    methodology = Methodology.RAPTOR
    # The text units ground the answer: they take what the summaries leave.
    spare_context_budget_first = (SectionType.TEXT,)

    def _graph_signal(self) -> bool:
        """Whether the graph buckets run: on (``auto``) where the namespace's
        methodologies build the entities index."""
        return (
            self.config.search.tree_search.graph_signal == "auto"
            and IndexArtifact.ENTITIES in artifacts_to_build(self.config)
        )

    def filtered_index_prefixes(self, strategy: SearchStrategy) -> list[str]:
        o = self.config.indexing.opensearch
        prefixes = [o.text_units_index_prefix, o.summary_nodes_index_prefix]
        if self._graph_signal():
            prefixes.append(o.entities_index_prefix)
        return prefixes

    async def asearch(
        self, query: SearchQuery, config: RunnableConfig | None = None
    ) -> SearchResult:
        start = time.time()
        rerank = self._new_rerank_trace()
        tree = self.config.search.tree_search
        logger.info("Tree search started - query: %s", text_digest(query.query))
        metadata: dict[str, Any] = {}
        buckets = await self._buckets(query, metadata)
        quota = {
            SectionType.TEXT.value: tree.text_top_k,
            SectionType.SUMMARY.value: tree.summary_top_k,
        }
        fused = (
            await self._fuse_and_rerank(
                buckets,
                rerank_trace=rerank,
                top_k=query.top_k,
                retrieval_multiplier=query.retrieval_multiplier,
                query=query.query,
                per_type_quota=quota,
            )
            if any(buckets.values())
            else []
        )
        kept = self._per_level(fused, tree.max_nodes_per_level)
        expanded = await self._expand(query, kept)
        results = sorted(
            [self._rendered(r) for r in kept] + expanded,
            key=lambda r: r.score,
            reverse=True,
        )
        levels = Counter(
            str((r.metadata or {}).get("level"))
            for r in results
            if r.retriever_type == SectionType.SUMMARY.value
        )
        metadata.update(
            {
                "rerank": rerank.summary(),
                "sources": {name: len(hits) for name, hits in buckets.items()},
                "summaries": sum(levels.values()),
                "summary_levels": dict(sorted(levels.items())),
                "expanded_passages": len(expanded),
            }
        )
        if not buckets.get(SUMMARY_BUCKET) and await self._tree_missing(query):
            metadata["summary_tree"] = "missing"
        logger.info(
            "Tree search completed - %d results (%d summaries) in %.3fs",
            len(results),
            metadata["summaries"],
            time.time() - start,
        )
        return SearchResult(
            query=query,
            results=results,
            total_results=len(results),
            search_strategy="tree_search",
            processing_time=time.time() - start,
            metadata=metadata,
        )

    # --- the buckets ------------------------------------------------------------

    async def _buckets(
        self, query: SearchQuery, metadata: dict[str, Any]
    ) -> dict[str, list[RetrievalResult]]:
        retriever = self.document_retriever
        if retriever is None:
            return {}
        o = self.config.indexing.opensearch
        tree = self.config.search.tree_search

        def search(prefix: str, top_k: int) -> SearchQuery:
            return SearchQuery(
                query=query.query,
                search_type=query.search_type,
                top_k=top_k,
                index_prefixes=[prefix],
                suffix=query.suffix,
                filters=self._scoped_filters(query),
            )

        searches = [
            self._safe_aretrieve(
                retriever,
                search(o.text_units_index_prefix, tree.text_top_k),
                "Tree text",
            ),
            self._safe_aretrieve(
                retriever,
                search(o.summary_nodes_index_prefix, tree.summary_top_k),
                "Tree summaries",
            ),
        ]
        graph = self._graph_signal()
        if graph:
            searches.append(
                self._safe_aretrieve(
                    retriever,
                    search(o.entities_index_prefix, tree.graph_seed_entities),
                    "Tree graph seeds",
                )
            )
        found = await gather_or_cancel(searches)
        buckets = {TEXT_BUCKET: found[0], SUMMARY_BUCKET: found[1]}
        metadata["graph_signal"] = "on" if graph else "unavailable"
        if tree.graph_signal == "off":
            metadata["graph_signal"] = "off"
        if graph:
            seeds = found[2][: tree.graph_seed_entities]
            metadata["graph_seeds"] = len(seeds)
            buckets.update(await self._graph_buckets(query, seeds))
        return buckets

    async def _graph_buckets(
        self, query: SearchQuery, seeds: list[RetrievalResult]
    ) -> dict[str, list[RetrievalResult]]:
        """The summaries and text units citing the seeds' text units, each
        scored by the seeds' weights (``1 / (1 + rank)``) it covers; a
        summary over ``sqrt(n_leaves)``, so a root does not win by size."""
        retriever = self.document_retriever
        weights: dict[str, float] = defaultdict(float)
        lineages: list[tuple[float, set[str]]] = []
        for rank, seed in enumerate(seeds):
            units = set(_units(seed))
            if not units:
                continue
            weight = 1.0 / (1 + rank)
            lineages.append((weight, units))
            for unit in units:
                weights[unit] += weight
        if retriever is None or not weights:
            return {}
        o = self.config.indexing.opensearch
        tree = self.config.search.tree_search
        limits = await retriever.aindex_limits(
            o.summary_nodes_index_prefix, query.suffix
        )
        cited = sorted(weights, key=lambda u: (-weights[u], u))[
            : limits.max_terms_count
        ]
        nodes = await self._safe_aretrieve(
            retriever,
            SearchQuery(
                query="",
                search_type=SearchType.LEXICAL,
                top_k=min(
                    limits.max_result_window,
                    max(
                        tree.summary_top_k,
                        len(cited) * self.config.indexing.summary_tree.max_levels,
                    ),
                ),
                retrieval_multiplier=1,
                index_prefixes=[o.summary_nodes_index_prefix],
                suffix=query.suffix,
                filters=self._scoped_filters(query, text_unit_ids=cited),
            ),
            "Tree graph summaries",
        )

        def node_score(node: RetrievalResult) -> float:
            held = set(_units(node))
            leaves = int((node.metadata or {}).get("n_leaves") or len(held) or 1)
            covered = sum(weight for weight, units in lineages if units & held)
            return covered / math.sqrt(leaves)

        ranked_nodes = sorted(nodes, key=lambda n: (-node_score(n), str(n.source)))[
            : tree.summary_top_k
        ]
        top_units = sorted(weights, key=lambda u: (-weights[u], u))[: tree.text_top_k]
        unit_hits = await self._fetch_by_ids(
            retriever,
            SearchQuery(
                query="",
                search_type=SearchType.LEXICAL,
                top_k=len(top_units),
                index_prefixes=[o.text_units_index_prefix],
                suffix=query.suffix,
                filters=self._scoped_filters(query, id=top_units),
            ),
            "Tree graph text",
        )
        order = {unit: index for index, unit in enumerate(top_units)}
        ranked_units = sorted(
            unit_hits, key=lambda r: order.get(str(r.source), len(order))
        )
        return {GRAPH_SUMMARY_BUCKET: ranked_nodes, GRAPH_TEXT_BUCKET: ranked_units}

    # --- selection, expansion, rendering ------------------------------------------

    @staticmethod
    def _per_level(
        results: list[RetrievalResult], per_level: int
    ) -> list[RetrievalResult]:
        """``results`` with at most ``per_level`` summaries of any level, in
        order."""
        seen: Counter[Any] = Counter()
        kept = []
        for result in results:
            if result.retriever_type == SectionType.SUMMARY.value:
                level = (result.metadata or {}).get("level")
                if seen[level] >= per_level:
                    continue
                seen[level] += 1
            kept.append(result)
        return kept

    async def _expand(
        self, query: SearchQuery, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Up to ``expand_children`` passages of each kept summary's own
        lineage, searched by the query under the caller's filters, scored
        just below the summary; a passage already in ``results`` (or another
        summary's) is not repeated."""
        count = self.config.search.tree_search.expand_children
        retriever = self.document_retriever
        summaries = [
            r for r in results if r.retriever_type == SectionType.SUMMARY.value
        ]
        if not count or retriever is None or not summaries:
            return []
        o = self.config.indexing.opensearch
        limits = await retriever.aindex_limits(o.text_units_index_prefix, query.suffix)

        async def passages(summary: RetrievalResult) -> list[RetrievalResult]:
            lineage = (await self._lineage(query, summary))[: limits.max_terms_count]
            if not lineage:
                return []
            return await self._safe_aretrieve(
                retriever,
                SearchQuery(
                    query=query.query,
                    search_type=query.search_type,
                    top_k=count,
                    index_prefixes=[o.text_units_index_prefix],
                    suffix=query.suffix,
                    filters=self._scoped_filters(query, id=lineage),
                ),
                "Tree expansion",
            )

        found = await gather_bounded(
            self.config.search.max_concurrency, (passages(s) for s in summaries)
        )
        taken = {
            str(r.source) for r in results if r.retriever_type == SectionType.TEXT.value
        }
        expanded: list[RetrievalResult] = []
        for summary, hits in zip(summaries, found, strict=True):
            for rank, hit in enumerate(hits[:count], start=1):
                if str(hit.source) in taken:
                    continue
                taken.add(str(hit.source))
                expanded.append(
                    hit.model_copy(
                        update={
                            "score": summary.score - rank * _EXPANSION_STEP,
                            "metadata": {
                                **(hit.metadata or {}),
                                "expanded_from": summary.source,
                            },
                        }
                    )
                )
        return expanded

    async def _lineage(self, query: SearchQuery, summary: RetrievalResult) -> list[str]:
        """A summary's text units: its stored lineage, or, when that was cut
        (``lineage_complete`` false), the union of its descendants' down to
        the nodes whose lineage is whole."""
        metadata = summary.metadata or {}
        if metadata.get("lineage_complete", True) is not False:
            return sorted(_units(summary))
        retriever = self.document_retriever
        assert retriever is not None
        prefix = self.config.indexing.opensearch.summary_nodes_index_prefix
        units: set[str] = set()
        frontier = [str(c) for c in metadata.get("child_ids") or []]
        for _ in range(_MAX_LINEAGE_DEPTH):
            if not frontier:
                break
            children = await self._fetch_by_ids(
                retriever,
                SearchQuery(
                    query="",
                    search_type=SearchType.LEXICAL,
                    top_k=len(frontier),
                    index_prefixes=[prefix],
                    suffix=query.suffix,
                    filters={"id": frontier},
                ),
                "Tree lineage",
            )
            nodes = {str(c.source) for c in children}
            # Children that are no node are text units.
            units.update(c for c in frontier if c not in nodes)
            frontier = []
            for child in children:
                if (child.metadata or {}).get("lineage_complete", True) is False:
                    frontier.extend(
                        str(c) for c in (child.metadata or {}).get("child_ids") or []
                    )
                else:
                    units.update(_units(child))
        return sorted(units)

    @staticmethod
    def _rendered(result: RetrievalResult) -> RetrievalResult:
        """A summary as the context shows it: its level and how many
        passages it covers before its text."""
        if result.retriever_type != SectionType.SUMMARY.value:
            return result
        metadata = result.metadata or {}
        text = metadata.get("text") or result.content
        level = metadata.get("level")
        leaves = metadata.get("n_leaves")
        return result.model_copy(
            update={"content": f"Summary (level {level}, {leaves} passages): {text}"}
        )

    async def _tree_missing(self, query: SearchQuery) -> bool:
        """Whether the namespace has no summary index (asked only when the
        summary search found nothing)."""
        retriever = self.document_retriever
        if retriever is None:
            return False
        exists = await retriever.ahas_index(
            self.config.indexing.opensearch.summary_nodes_index_prefix, query.suffix
        )
        if exists is False:
            logger.warning(
                "The index namespace has no summary tree: tree answers from the "
                "text units alone. Select the raptor methodology "
                "(indexing.methodologies) and run ingestion to build it."
            )
        return exists is False
