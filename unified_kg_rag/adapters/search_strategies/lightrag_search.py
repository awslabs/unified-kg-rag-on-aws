# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LightRAG dual-level keyword search strategy.

Implements LightRAG's ``mix``, ``hybrid`` and ``naive`` retrieval modes (the
ones registered; LightRAG's ``local`` and ``global`` are not: ``hybrid``
runs both levels) on top of unified-kg-rag-on-aws's *shared* infrastructure
rather than as a separate, reduced path:

- low-level keywords (``ll_keywords``) -> entities index (lexical + vector),
- high-level keywords (``hl_keywords``) -> relationships index (lexical + vector),
- the entity hits are expanded one hop to their incident relationships and
  the relationship hits to their endpoint entities (both from OpenSearch),
- ``hybrid`` and ``mix`` pull the source chunks referenced by the matched
  entities and, as a stream of its own, by the matched relationships
  (following ``text_unit_ids`` lineage: by weighted polling, every match
  keeping at least one chunk, or by similarity to the query —
  LightRAG's ``_find_related_text_unit_from_entities``/``_from_relations``,
  ``kg_chunk_pick_method``); ``mix`` additionally blends the naive chunk
  retrieval, deduplicating the cited chunks against it (``_merge_chunks``),
- everything is fused and reranked via the shared :class:`HybridScorer`
  (BM25 lexical + vector semantic + graph + RRF + Bedrock rerank).

So a LightRAG-mode query enjoys the same hybrid scoring, multilingual handling,
and caching as the GraphRAG strategies — only the retrieval algorithm differs.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableConfig

from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
)
from unified_kg_rag.adapters.retrieval.strategy_registry import (
    QueryInput,
    register_strategy,
)
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.domain.ingestion.seed_graph import is_seed_unit
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
)
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import text_digest
from unified_kg_rag.shared.utils.concurrency import gather_or_cancel

logger = get_logger(__name__)

# The two KG-linked chunk streams: the chunks the entity hits cite and those
# the relationship hits cite. Fused as separate streams, so RRF interleaves
# them (upstream round-robins its vector, entity and relation chunk lists).
_ENTITY_CHUNKS = "lightrag_linked_chunks"
_RELATION_CHUNKS = "lightrag_relation_chunks"

# Upstream LightRAG uses TWO separate retrieval widths, not one.
# `QueryParam.top_k` (DEFAULT_TOP_K = 40) is the width of the ENTITY and RELATIONSHIP
# vector queries; `QueryParam.chunk_top_k` (DEFAULT_CHUNK_TOP_K = 20) is the separate
# width of the chunk stream (lightrag/constants.py; operate.py line ~4281:
# `search_top_k = query_param.chunk_top_k or query_param.top_k`). Collapsing both onto
# the caller's single `top_k` starves the KG streams, because a `top_k` sized for the
# chunk stream is far narrower than the width the entity/relationship queries need.
#
# `top_k` stays the caller's knob for how wide the CHUNK stream is (the axis that maps
# onto the GraphRAG strategies' top_k). The KG streams get their own configured width
# (`search.lightrag_search.kg_stream_top_k`, defaulting to upstream's 40), scaled up
# from top_k when the caller asks for more than that.


def _pick_by_weighted_polling(
    lineages: list[list[str]],
    max_related_chunks: int,
    min_related_chunks: int = 1,
) -> list[str]:
    """Allocate chunk slots per KG hit on a linear-decreasing gradient.

    Port of upstream LightRAG's ``utils.pick_by_weighted_polling`` (the
    ``kg_chunk_pick_method = WEIGHT`` path of
    ``_find_related_text_unit_from_entities``/``_from_relations``; the
    ``VECTOR`` one is ``LightRAGSearchStrategy._vector_picked_chunks``).
    ``lineages`` is one chunk-id list per matched entity/relationship, already
    in retrieval (importance) order; each list keeps its order (chunk id,
    ``LightRAGSearchStrategy._lineages``), without upstream's sort of a hit's
    chunks by how many matched hits cite them.

    The point of the gradient is the ``min_related_chunks=1`` floor: the LAST
    matched KG item still gets one chunk. A global "rank all lineage chunks by
    citation count" cut — what this strategy did previously — instead
    lets chunks cited by many entities take every slot, and the one low-citation
    chunk that uniquely carries a multi-hop question's second hop is dropped.
    """
    if not lineages:
        return []
    n = len(lineages)
    if n == 1:
        return list(lineages[0][:max_related_chunks])

    expected_counts = [
        int(
            round(
                max_related_chunks
                - (i / (n - 1)) * (max_related_chunks - min_related_chunks)
            )
        )
        for i in range(n)
    ]

    selected: list[str] = []
    used = []
    total_remaining = 0
    for lineage, expected in zip(lineages, expected_counts, strict=True):
        actual = min(expected, len(lineage))
        selected.extend(lineage[:actual])
        used.append(actual)
        total_remaining += expected - actual

    # Re-distribute the quota that short lineages could not fill, one chunk at
    # a time, first-fit as upstream LightRAG does: each leftover slot goes to
    # the first item, in rank order, with a chunk left.
    for _ in range(total_remaining):
        allocated = False
        for i, lineage in enumerate(lineages):
            if used[i] < len(lineage):
                selected.append(lineage[used[i]])
                used[i] += 1
                allocated = True
                break
        if not allocated:
            break
    return selected


# NAIVE is chunk retrieval only (no graph), so it declares DOCUMENT-only roles;
# MIX/HYBRID keep the default (DOCUMENT, GRAPH) so the opt-in multi-hop graph
# expansion (`search.lightrag_search.enable_graph_expansion`) has a GRAPH
# retriever. The shared class can't carry one required_roles for all three, so
# NAIVE registers separately -- otherwise every naive query builds an unused
# Neptune retriever.
@register_strategy(
    SearchStrategy.MIX, query_inputs=frozenset({QueryInput.DUAL_KEYWORDS})
)
@register_strategy(
    SearchStrategy.HYBRID, query_inputs=frozenset({QueryInput.DUAL_KEYWORDS})
)
@register_strategy(SearchStrategy.NAIVE, required_roles=(RetrieverRole.DOCUMENT,))
class LightRAGSearchStrategy(BaseSearchStrategy):
    """Dual-level keyword retrieval (LightRAG) over the shared hybrid stack.

    The same class serves three modes, distinguished by the resolved
    :class:`SearchStrategy` passed via ``query.metadata['search_strategy']``
    (the chain sets it; default ``mix``):

    - ``naive``: chunk retrieval only (no graph), with the request's
      ``search_type`` (BM25 + vector by default).
    - ``hybrid``: ll->entities + hl->relationships + one-hop cross-type
      expansion + the chunks those items cite.
    - ``mix``: ``hybrid`` plus the ``naive`` chunk retrieval.

    Backends are accessed only through the abstract GRAPH / DOCUMENT retriever
    roles (``self.graph_retriever`` / ``self.document_retriever``).
    """

    # Upstream gives chunks what the KG sections leave unused
    # (operate.py: available_chunk_tokens = max_total_tokens - KG tokens).
    spare_context_budget_first = (SectionType.TEXT,)

    def __init__(
        self,
        config: Config,
        retrievers: dict[str, BaseGraphRAGRetriever],
        **kwargs: Any,
    ) -> None:
        super().__init__(config, retrievers, **kwargs)
        self._os_config = config.indexing.opensearch
        self._lightrag_config = config.search.lightrag_search

    def _kg_stream_top_k(self, top_k: int) -> int:
        """Width of the entity/relationship vector queries (`QueryParam.top_k`)."""
        return max(self._lightrag_config.kg_stream_top_k, top_k)

    def _chunk_stream_top_k(self, top_k: int) -> int:
        """Width of the chunk vector query (`QueryParam.chunk_top_k`)."""
        return max(self._lightrag_config.chunk_stream_top_k, top_k)

    def _mode(self, query: SearchQuery) -> str:
        return str(query.metadata.get("search_strategy") or SearchStrategy.MIX.value)

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
        # Upstream warns here too ("low_level_keywords and high_level_keywords
        # is empty"): hybrid then returns nothing and mix only chunks.
        logger.warning(
            "No keywords extracted and the query is too long (%s characters, "
            "limit %s) for the raw-query fallback; no entities or relationships "
            "are retrieved",
            len(query.query),
            max_len,
        )
        return query

    def filtered_index_prefixes(self, strategy: SearchStrategy) -> list[str]:
        # naive: the chunks; hybrid and mix: entities, relationships and the
        # chunks they cite.
        o = self._os_config
        if strategy is SearchStrategy.NAIVE:
            return [o.text_units_index_prefix]
        return self._built_prefixes(
            o.entities_index_prefix,
            o.relationships_index_prefix,
            o.text_units_index_prefix,
        )

    def filtered_retrievers(
        self, strategy: SearchStrategy
    ) -> list[BaseGraphRAGRetriever]:
        # The graph is read only by the opt-in graph expansion.
        if self._lightrag_config.enable_graph_expansion:
            return super().filtered_retrievers(strategy)
        return [
            r
            for role, r in self.retrievers.items()
            if role != RetrieverRole.GRAPH.value
        ]

    async def asearch(
        self, query: SearchQuery, config: RunnableConfig | None = None
    ) -> SearchResult:
        start_time = time.time()
        rerank = self._new_rerank_trace()
        mode = self._mode(query)
        logger.info(
            "LightRAG search started - mode: '%s', query: %s",
            mode,
            text_digest(query.query),
        )

        results_by_source: dict[str, list[RetrievalResult]] = {}

        if mode == SearchStrategy.NAIVE.value:
            results_by_source.update(await self._retrieve_chunks(query))
        else:
            query = self._apply_keyword_fallback(query)
            results_by_source.update(await self._retrieve_kg_streams(query, mode))

        # Keep entities/relationships/chunks as separate streams
        # with reserved slots (LightRAG uses ~40 entities / 20 chunks), instead of
        # collapsing all types into one flat top_k pool where chunks get starved.
        # Scaled off top_k: text (chunks) get the largest share since the multi-hop
        # answer lives in them; entities/relationships get the rest. Non-mix modes and
        # the no-quota path are unchanged.
        per_type_quota = None
        if mode in (SearchStrategy.MIX.value, SearchStrategy.HYBRID.value):
            k = query.top_k
            # The quota has to match upstream's TWO widths (see the module comment on
            # kg_stream_top_k / chunk_stream_top_k), otherwise widening the vector
            # queries changes nothing — the fusion quota, not the query width, is what
            # decides how much of each stream reaches the context.
            #
            # Upstream applies NO count cap to the KG lists —
            # `_find_most_related_edges_from_entities` /
            # `_find_most_related_entities_from_relationships` return every
            # cross-expanded item and `_apply_token_truncation` trims them against
            # per-type TOKEN limits (DEFAULT_MAX_ENTITY_TOKENS 6000 /
            # DEFAULT_MAX_RELATION_TOKENS 8000). So the KG quotas are the *floor*
            # `_kg_stream_top_k(k)` raised to however many candidates the expansion
            # actually produced; TokenManager's per-type budget is what binds, exactly
            # as upstream. (Chunks keep a hard count cap because upstream has one too:
            # `chunk_top_k` on the merged pool — see `_select_with_type_quota`.)
            #
            # Only the three types mix actually retrieves are listed. A type with no
            # quota entry is still eligible for the score-ordered back-fill, so this
            # reserves slots without silently excluding anything.
            candidates: dict[str, int] = {}
            for stream in results_by_source.values():
                for result in stream:
                    rtype = result.retriever_type
                    candidates[rtype] = candidates.get(rtype, 0) + 1
            kg_floor = self._kg_stream_top_k(k)
            per_type_quota = {
                SectionType.TEXT.value: self._chunk_stream_top_k(k),
                SectionType.ENTITY.value: max(
                    kg_floor, candidates.get(SectionType.ENTITY.value, 0)
                ),
                SectionType.RELATIONSHIP.value: max(
                    kg_floor, candidates.get(SectionType.RELATIONSHIP.value, 0)
                ),
            }
        # For mix, rerank ONLY text chunks; keep the KG entity/relationship streams on
        # their native (degree/cosine) order. Content-vs-query reranking buries
        # multi-hop bridge entities/relations whose description lacks the question's
        # surface terms (LightRAG reranks chunks only).
        rerank_only_types = (
            {SectionType.TEXT.value}
            if mode in (SearchStrategy.MIX.value, SearchStrategy.HYBRID.value)
            else None
        )
        # naive returns the chunk width, as upstream `naive_query` returns
        # `chunk_top_k` chunks rather than the KG `top_k`.
        fuse_top_k = (
            self._chunk_stream_top_k(query.top_k)
            if mode == SearchStrategy.NAIVE.value
            else query.top_k
        )
        final_results = await self._fuse_and_rerank(
            results_by_source,
            rerank_trace=rerank,
            top_k=fuse_top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
            per_type_quota=per_type_quota,
            rerank_only_types=rerank_only_types,
        )

        processing_time = time.time() - start_time

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
                "rerank": rerank.summary(),
            },
        )

    async def _retrieve_kg_streams(
        self, query: SearchQuery, mode: str
    ) -> dict[str, list[RetrievalResult]]:
        """hybrid / mix retrieval in three dependency-ordered rounds.

        1. The independent vector queries run concurrently: ll keywords ->
           entities, hl keywords -> relationships, and (mix only) the naive
           chunk query.
        2. The two cross-type expansions run concurrently, as both depend only
           on round 1: entity hits -> incident relationships (upstream
           ``_find_most_related_edges_from_entities``) and relationship hits ->
           endpoint entities (``_find_most_related_entities_from_relationships``).
        3. The source chunks cited by the matched entities and, as a stream
           of their own, by the matched relationships
           (``_find_related_text_unit_from_entities`` / ``_from_relations``),
           excluding chunks the mix vector query already returned, because
           upstream merges the chunk streams under one ``seen_chunk_ids`` set
           with the vector stream taking precedence (``_merge_chunks``).

        Upstream LightRAG has no multi-hop graph traversal: its graph context is
        the matched items plus their one-hop incident edges and endpoint nodes,
        which rounds 2 and 3 already cover from the OpenSearch indices. The
        Neptune neighbourhood expansion is therefore opt-in
        (``search.lightrag_search.enable_graph_expansion``); when enabled it runs
        in round 2, seeded from the entity hits and the relationship endpoints
        in rank order.
        """
        results: dict[str, list[RetrievalResult]] = {}

        first_round = []
        if query.ll_keywords:
            first_round.append(self._retrieve_entities(query))
        if query.hl_keywords:
            first_round.append(self._retrieve_relationships(query))
        if mode == SearchStrategy.MIX.value:
            first_round.append(self._retrieve_chunks(query))
        for streams in await gather_or_cancel(first_round):
            results.update(streams)

        entity_hits = results.get("lightrag_entities", [])
        relationship_hits = results.get("lightrag_relationships", [])
        second_round = []
        if entity_hits:
            second_round.append(
                self._retrieve_incident_relationships(
                    query, entity_hits, known_relationships=relationship_hits
                )
            )
        if relationship_hits:
            # Sourced from the hl VECTOR hits only, as upstream does, not from
            # the incident-edge expansion running alongside it.
            second_round.append(
                self._retrieve_endpoint_entities(
                    query, relationship_hits, known_entities=entity_hits
                )
            )
        if self._lightrag_config.enable_graph_expansion:
            second_round.append(
                self._graph_expansion(query, entity_hits, relationship_hits)
            )
        for streams in await gather_or_cancel(second_round):
            results.update(streams)

        vector_chunk_ids = {
            cid
            for cid in self._get_ids(results.get("lightrag_chunks", []), "id")
            if cid
        }
        results.update(
            await self._retrieve_linked_chunks(
                query, entity_hits, relationship_hits, exclude=vector_chunk_ids
            )
        )
        return results

    async def _graph_expansion(
        self,
        query: SearchQuery,
        entity_hits: list[RetrievalResult],
        relationship_hits: list[RetrievalResult],
    ) -> dict[str, list[RetrievalResult]]:
        """Opt-in Neptune neighbourhood of the entity hits + relationship endpoints.

        Every seed (entity hits, then endpoints in first-seen order, deduped)
        is expanded and kept. The expansion (the graph walk, or under a
        document filter the walk over the relationships index) splits its
        fetch budget (``top_k * retrieval_multiplier *
        traversal_fetch_multiplier``) across the seeds, so with many seeds
        each reaches as few as one neighbour, but together they still reach
        up to that budget for the ``top_k * retrieval_multiplier`` neighbour
        slots it keeps.
        """
        entity_ids = [
            str((hit.metadata or {}).get("id") or hit.source or "")
            for hit in entity_hits
        ]
        seed_ids = list(
            dict.fromkeys(
                eid
                for eid in entity_ids
                + self._relationship_endpoint_ids(relationship_hits)
                if eid
            )
        )
        if not seed_ids:
            return {}
        expanded = await self._expand_via_graph(query, seed_ids)
        return {"graph_entities": expanded} if expanded else {}

    async def _retrieve_entities(
        self, query: SearchQuery
    ) -> dict[str, list[RetrievalResult]]:
        """Low-level keywords -> entities index (LightRAG local component)."""
        if not self.document_retriever:
            return {}
        search_query = SearchQuery(
            query=", ".join(query.ll_keywords),
            search_type=query.search_type,
            top_k=self._kg_stream_top_k(query.top_k),
            index_prefixes=[self._os_config.entities_index_prefix],
            suffix=query.suffix,
            filters=self._scoped_filters(query),
        )
        results = await self._safe_aretrieve(
            self.document_retriever, search_query, "Entity retrieval (ll_keywords)"
        )
        return {"lightrag_entities": results}

    async def _retrieve_relationships(
        self, query: SearchQuery
    ) -> dict[str, list[RetrievalResult]]:
        """High-level keywords -> relationships index (LightRAG global component)."""
        if not self.document_retriever:
            return {}
        search_query = SearchQuery(
            query=", ".join(query.hl_keywords),
            search_type=query.search_type,
            top_k=self._kg_stream_top_k(query.top_k),
            index_prefixes=[self._os_config.relationships_index_prefix],
            suffix=query.suffix,
            filters=self._scoped_filters(query),
        )
        results = await self._safe_aretrieve(
            self.document_retriever,
            search_query,
            "Relationship retrieval (hl_keywords)",
        )
        return {"lightrag_relationships": results}

    async def _retrieve_chunks(
        self, query: SearchQuery
    ) -> dict[str, list[RetrievalResult]]:
        """Naive chunk retrieval over the text-units index.

        Runs the request's ``search_type``: BM25 + vector under the default
        ``hybrid``; ``vector`` matches upstream's pure-vector baseline.
        """
        if not self.document_retriever:
            return {}
        search_query = SearchQuery(
            query=query.query,
            search_type=query.search_type,
            top_k=self._chunk_stream_top_k(query.top_k),
            index_prefixes=[self._os_config.text_units_index_prefix],
            suffix=query.suffix,
            filters=self._scoped_filters(query),
        )
        results = await self._safe_aretrieve(
            self.document_retriever, search_query, "Chunk retrieval"
        )
        return {"lightrag_chunks": results}

    @staticmethod
    def _lineages(results: list[RetrievalResult]) -> list[list[str]]:
        """Per-hit ``text_unit_ids`` lineage, hits in retrieval (importance)
        order, each lineage deduplicated and sorted by chunk id, without the
        seed graphs' units: they are no chunks to follow (a hit only a seed
        graph states has none).

        Sorted here, not as stored: a full build stores a lineage sorted
        while an incremental merge appends the delta's units in arrival
        order, so the stored order made the weighted pick depend on how the
        graph was built."""
        lineages: list[list[str]] = []
        for result in results:
            metadata = result.metadata or {}
            unit_ids = metadata.get("text_unit_ids")
            if not isinstance(unit_ids, list):
                continue
            lineage = sorted(
                {
                    uid
                    for uid in unit_ids
                    if isinstance(uid, str) and uid and not is_seed_unit(uid)
                }
            )
            if lineage:
                lineages.append(lineage)
        return lineages

    def _collect_linked_chunk_ids(
        self,
        entity_results: list[RetrievalResult],
        relationship_results: list[RetrievalResult],
        limit: int,
        exclude: set[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """The source chunks cited by the matched entities, and those cited by
        the matched relationships: one list per side.

        Follows upstream LightRAG's two-stage selection
        (``_find_related_text_unit_from_entities`` then ``_from_relations``,
        its ``WEIGHT`` method), without its per-hit occurrence sort: a hit's
        chunks keep their id order (measured on dense graphs, the sort lowered
        hybrid recall):

        1. per-hit lineages are deduplicated keeping the EARLIEST (most
           important) citing hit of the side, so a chunk occupies one item's
           quota, not many;
        2. slots are allocated by :func:`_pick_by_weighted_polling`, which
           guarantees every matched hit at least one chunk, and each side is
           cut to ``limit`` (the chunk width);
        3. the relationship side drops the chunks the entity side SELECTED,
           and both drop ``exclude``: the chunks the naive vector stream
           already returned (upstream ``operate.py`` passes ``entity_chunks``
           into the relation pass for exactly this, then round-robin merges
           the three streams under one ``seen_chunk_ids``).

        The relationship side used to drop every chunk any entity hit cited,
        selected or not, and the two sides were cut to ``limit`` together, so
        the entity side's slots came first and a relationship's chunks almost
        never reached the context.
        """
        cap = self._lightrag_config.related_chunk_number
        base = set(exclude or ())

        def _select(results: list[RetrievalResult], excluded: set[str]) -> list[str]:
            deduped: list[list[str]] = []
            for lineage in self._lineages(results):
                keep = [uid for uid in lineage if uid not in excluded]
                # Keep the chunk on its earliest (most important) citing hit only.
                excluded.update(keep)
                if keep:
                    deduped.append(keep)
            return _pick_by_weighted_polling(deduped, cap)[:limit]

        entity_ids = _select(entity_results, set(base))
        relation_ids = _select(relationship_results, base | set(entity_ids))
        return entity_ids, relation_ids

    async def _retrieve_linked_chunks(
        self,
        query: SearchQuery,
        entity_results: list[RetrievalResult],
        relationship_results: list[RetrievalResult],
        exclude: set[str] | None = None,
    ) -> dict[str, list[RetrievalResult]]:
        """The source chunks cited by the matched entities
        (``lightrag_linked_chunks``) and by the matched relationships
        (``lightrag_relation_chunks``).

        ``search.lightrag_search.kg_chunk_pick_method`` picks them: ``weight``
        allocates chunk slots per hit by weighted polling and fetches both
        sides' chunks by id in one request; ``vector`` searches each side's
        cited chunks for the query (``_vector_picked_chunks``). ``exclude``
        carries the chunk ids the naive vector stream already returned.
        Each side is a stream of its own, so fusion interleaves them and the
        chunk quota (``chunk_stream_top_k``) cuts the merged chunks after the
        rerank, as upstream's round robin then ``chunk_top_k`` cut does.
        Degrades to no section if lineage is absent (e.g. an index built
        before lineage was added) or retrieval fails.
        """
        if not self.document_retriever:
            return {}
        # Each side is a CHUNK stream, so it is bounded by upstream's chunk
        # width, not by the caller's top_k. Upstream bounds it in two steps —
        # `related_chunk_number` (5) chunks per matched entity/relationship,
        # then a truncation of the merged pool to `chunk_top_k` — so with 40
        # KG hits its pool is far larger than a flat top_k=10 before the cut.
        limit = self._chunk_stream_top_k(query.top_k)
        exclude = set(exclude or ())
        # The caller's id filter: no slot for a chunk the fetch removes.
        linked = [
            unit
            for lineage in self._lineages([*entity_results, *relationship_results])
            for unit in lineage
        ]
        exclude.update(set(linked) - set(self._ids_allowed(query.filters, linked)))
        if self._lightrag_config.kg_chunk_pick_method == "vector":
            return await self._vector_picked_chunks(
                query, entity_results, relationship_results, limit, exclude
            )
        if filters := self._text_unit_filters(query.filters):
            entity_ids, relation_ids = await self._passing_linked_chunk_ids(
                query, entity_results, relationship_results, filters, limit, exclude
            )
        else:
            entity_ids, relation_ids = self._collect_linked_chunk_ids(
                entity_results, relationship_results, limit=limit, exclude=exclude
            )
        # Disjoint: the relationship side drops the entity side's chunks.
        chunk_ids = [*entity_ids, *relation_ids]
        if not chunk_ids:
            return {}
        search_query = SearchQuery(
            query="",
            search_type=SearchType.LEXICAL,
            top_k=len(chunk_ids),
            index_prefixes=[self._os_config.text_units_index_prefix],
            suffix=query.suffix,
            filters=self._scoped_filters(query, id=chunk_ids),
        )
        results = await self._fetch_by_ids(
            self.document_retriever, search_query, "Linked chunk retrieval"
        )
        relation_side = set(relation_ids)
        streams: dict[str, list[RetrievalResult]] = {}
        for name, ids, found in (
            (
                _ENTITY_CHUNKS,
                entity_ids,
                [r for r in results if str(r.source) not in relation_side],
            ),
            (
                _RELATION_CHUNKS,
                relation_ids,
                [r for r in results if str(r.source) in relation_side],
            ),
        ):
            if found:
                streams[name] = self._restore_rank(found, ids)
        return streams

    async def _vector_picked_chunks(
        self,
        query: SearchQuery,
        entity_results: list[RetrievalResult],
        relationship_results: list[RetrievalResult],
        limit: int,
        exclude: set[str],
    ) -> dict[str, list[RetrievalResult]]:
        """``kg_chunk_pick_method: vector``: per side, the cited chunks the
        query matches best.

        Upstream's ``VECTOR`` method (``pick_by_vector_similarity``) ranks a
        side's cited chunks by embedding similarity to the query and keeps
        ``related_chunk_number * hits / 2``. Here each side is one search of
        the text-units index restricted to its cited chunks (an ``id``
        filter, beside the caller's filters, which the search applies
        itself) with the request's ``search_type``, BM25 + vector by
        default, cut to the chunk width too. The relationship side drops the
        chunks the entity side picked, as upstream's relation pass does; it
        asks for that many more instead of waiting for them, so both run at
        once. A side citing more chunks than one request can name
        (``IndexLimits.id_batch_size``) searches the chunks of its most
        important hits only.
        """
        retriever = self.document_retriever
        assert retriever is not None  # the caller returns early without one
        prefix = self._os_config.text_units_index_prefix
        room = (await retriever.aindex_limits(prefix, query.suffix)).id_batch_size
        cap = self._lightrag_config.related_chunk_number

        def side(results: list[RetrievalResult]) -> tuple[list[str], int]:
            lineages = self._lineages(results)
            cited = list(dict.fromkeys(u for lineage in lineages for u in lineage))
            cited = [u for u in cited if u not in exclude][:room]
            return cited, min(limit, max(1, cap * len(lineages) // 2), len(cited))

        async def pick(
            cited: list[str], wanted: int, label: str
        ) -> list[RetrievalResult]:
            if not cited or wanted <= 0:
                return []
            return await self._safe_aretrieve(
                retriever,
                SearchQuery(
                    query=query.query,
                    search_type=query.search_type,
                    top_k=wanted,
                    retrieval_multiplier=1,
                    index_prefixes=[prefix],
                    suffix=query.suffix,
                    filters=self._scoped_filters(query, id=cited),
                ),
                label,
            )

        entity_cited, entity_wanted = side(entity_results)
        relation_cited, relation_wanted = side(relationship_results)
        entity_hits, relation_hits = await gather_or_cancel(
            [
                pick(entity_cited, entity_wanted, "Entity chunk pick (vector)"),
                pick(
                    relation_cited,
                    min(relation_wanted + entity_wanted, len(relation_cited)),
                    "Relationship chunk pick (vector)",
                ),
            ]
        )
        taken = {str(r.source) for r in entity_hits}
        relation_hits = [r for r in relation_hits if str(r.source) not in taken]
        streams: dict[str, list[RetrievalResult]] = {}
        if entity_hits:
            streams[_ENTITY_CHUNKS] = entity_hits
        if relation_hits := relation_hits[:relation_wanted]:
            streams[_RELATION_CHUNKS] = relation_hits
        return streams

    async def _passing_linked_chunk_ids(
        self,
        query: SearchQuery,
        entity_results: list[RetrievalResult],
        relationship_results: list[RetrievalResult],
        filters: dict[str, Any],
        limit: int,
        exclude: set[str],
    ) -> tuple[list[str], list[str]]:
        """``_collect_linked_chunk_ids`` under a filter on text units: the
        slots go to chunks that pass.

        Each hit gets at most ``related_chunk_number`` chunks. The first
        round checks (ids only) each hit's first ``related_chunk_number``
        chunks and the unfiltered selection, all in one request, and ends the
        checks when the selection is then all checked (a filter that passes
        it). Otherwise the hits still short of their slots have their whole
        remaining lineage checked, again in one request. So a hub hit's long
        lineage does not stop the check before the other hits' are asked, and
        a selective filter costs two requests, not one per round of a few
        ids. The selection then keeps a chunk on its first citing hit only
        and polling hands a short hit's slots to the others, so a slot can
        still land on a chunk neither round asked: each hit holding such a
        chunk has its lineage checked past the deepest one, by
        ``related_chunk_number`` + 1 chunks and twice as far each round (one
        request per round, not per chunk, and a run of failing chunks costs
        a few rounds; the whole rest of a hub's lineage cost a request per
        few ids), and the chunks selected again, until every selected chunk
        is known. A failed
        check counts what it asked as failing (fail-closed) and ends the
        checks; the slots then go to the next chunks, fetched under the
        filter (which the fetch applies too).
        """
        cap = self._lightrag_config.related_chunk_number
        lineages = [
            [unit for unit in lineage if unit not in exclude]
            for lineage in self._lineages([*entity_results, *relationship_results])
        ]
        prefix = self._os_config.text_units_index_prefix
        passed: set[str] = set()
        checked: set[str] = set()
        failing = set(exclude)

        def select() -> tuple[list[str], list[str]]:
            return self._collect_linked_chunk_ids(
                entity_results, relationship_results, limit=limit, exclude=failing
            )

        def flat(selection: tuple[list[str], list[str]]) -> list[str]:
            return [*selection[0], *selection[1]]

        async def check(ids: list[str]) -> bool:
            known = len(checked)
            passed.update(
                await self._ids_passing(query, prefix, ids, filters, checked=checked)
            )
            if len(checked) == known:  # the check failed: fail-closed
                failing.update(ids)
                return False
            failing.update(checked - passed)
            return True

        for second in (False, True):
            if second:  # the rest of each hit's lineage that is still short
                parts = [
                    lineage[cap:]
                    for lineage in lineages
                    if sum(1 for u in lineage[:cap] if u in passed) < cap
                ]
            else:
                # The unfiltered selection too: the slots land there unless
                # the filter fails some of it, which the order of a lineage
                # (sorted by id) decides otherwise.
                parts = [flat(select()), *(lineage[:cap] for lineage in lineages)]
            ask = [
                u
                for u in dict.fromkeys(u for part in parts for u in part)
                if u not in checked
            ]
            if ask and not await check(ask):
                return select()
            if not second and all(u in checked for u in flat(select())):
                return select()
        selected = select()
        # How far past the deepest unknown chunk a round asks; doubled each
        # round. Asking only the selected chunks cost a request per failing
        # chunk, a hub's whole rest a request per few ids (max_terms_count).
        reach = cap + 1

        def ahead(lineage: list[str], unknown: set[str]) -> list[str]:
            deepest = max(i for i, u in enumerate(lineage) if u in unknown)
            return lineage[: deepest + reach]

        while unknown := {u for u in flat(selected) if u not in checked}:
            ok = await check(
                list(
                    dict.fromkeys(
                        u
                        for lineage in lineages
                        if unknown.intersection(lineage)
                        for u in ahead(lineage, unknown)
                        if u not in checked
                    )
                )
            )
            selected = select()
            reach *= 2
            if not ok:
                break
        return selected

    @staticmethod
    def _relationship_endpoint_ids(
        relationship_results: list[RetrievalResult],
    ) -> list[str]:
        """Collect source/target entity ids from relationship hits.

        Relationship documents carry their endpoint entity ids in metadata
        (``source_id``/``target_id``); these ground a high-level (relationship)
        hit back to the graph so it can be expanded like an entity hit.

        Deduplicated in FIRST-SEEN order (src before tgt, edge by edge), which is
        the order upstream's ``_find_most_related_entities_from_relationships``
        builds ``entity_names`` in and then rebuilds ``node_datas`` to match. The
        order is load-bearing: the KG streams are not reranked (only ``text`` is),
        so stream position is what decides which endpoints survive fusion.
        """
        endpoint_ids: list[str] = []
        for result in relationship_results:
            metadata = result.metadata or {}
            for field in ("source_id", "target_id"):
                value = metadata.get(field)
                if isinstance(value, str) and value:
                    endpoint_ids.append(value)
        return list(dict.fromkeys(endpoint_ids))

    async def _retrieve_endpoint_entities(
        self,
        query: SearchQuery,
        relationship_results: list[RetrievalResult],
        known_entities: list[RetrievalResult] | None = None,
    ) -> dict[str, list[RetrievalResult]]:
        """The endpoint entities of the hl relationship hits.

        Upstream's global side does not run an entity vector query at all — its
        entity context comes entirely from the matched relationships:
        ``_get_edge_data`` (operate.py ~5419) hands its ``edge_datas`` to
        ``_find_most_related_entities_from_relationships`` (~5478), which collects
        every ``src_id``/``tgt_id`` in first-seen order and fetches those nodes via
        ``get_nodes_batch``. In hybrid/mix the result is round-robin merged with the
        local (ll vector) entity list under one ``seen_entities`` set
        (``_merge_context``). That is how upstream's entity section grows well past
        the width of the entity vector query itself.

        This emits the endpoints themselves as first-class entity candidates.

        Only the hl VECTOR hits are used as the source, not the
        incident-edge expansion: upstream's endpoint pass runs inside
        ``_get_edge_data`` on the vector results only, and feeding it the
        (much larger) incident-edge set would manufacture entities upstream never
        has.
        """
        retriever = self.document_retriever
        if not retriever:
            return {}
        already = {eid for eid in self._get_ids(known_entities or [], "id") if eid}
        endpoint_ids = [
            eid
            for eid in self._relationship_endpoint_ids(relationship_results)
            if eid not in already
        ]
        if not endpoint_ids:
            return {}
        search_query = SearchQuery(
            query="",
            search_type=SearchType.LEXICAL,
            top_k=len(endpoint_ids),
            index_prefixes=[self._os_config.entities_index_prefix],
            suffix=query.suffix,
            filters=self._scoped_filters(query, id=endpoint_ids),
        )
        results = await self._fetch_by_ids(
            retriever, search_query, "Endpoint entity retrieval"
        )
        if not results:
            return {}
        # Restore the endpoint order the fetch-by-id lost (OpenSearch returns a
        # terms filter's hits in score/index order, not in the order asked for);
        # upstream rebuilds `node_datas` in `entity_names` order for this reason.
        order = {eid: i for i, eid in enumerate(endpoint_ids)}

        def _position(result: RetrievalResult) -> int:
            metadata = result.metadata or {}
            eid = str(metadata.get("id") or result.source or "")
            return order.get(eid, len(order))

        results.sort(key=_position)
        logger.info(
            "Endpoint expansion: %s relationship hits -> %s endpoint entities",
            len(relationship_results),
            len(results),
        )
        return {"lightrag_endpoint_entities": results}

    async def _retrieve_incident_relationships(
        self,
        query: SearchQuery,
        entity_results: list[RetrievalResult],
        known_relationships: list[RetrievalResult] | None = None,
    ) -> dict[str, list[RetrievalResult]]:
        """Every relationship INCIDENT to a matched entity.

        Upstream LightRAG's `_find_most_related_edges_from_entities` (operate.py
        ~5204) takes the entity vector hits, pulls **all** edges incident to them
        via `get_nodes_edges_batch`, dedups them, and orders by
        `(rank, weight)` — with no top_k cut. That cross-type expansion is where the
        bulk of upstream's relationship context comes from; the relationship vector
        query alone supplies a small fraction of it. The relationship stream was never
        truncated on OUR side — the candidates simply did not exist.

        The relationships index stores its endpoints as `source_id`/`target_id`
        keywords, but `_build_filter_clauses` ANDs every filter, so
        `source_id OR target_id` needs two queries. `known_relationships` carries
        the hl vector query's hits, which are dropped from this stream by
        relationship id. Upstream keys on the endpoint pair (`_merge_context`'s
        `rel_key = tuple(sorted([src_id, tgt_id]))`) because its graph holds one
        edge per pair; ours keeps typed, directed edges between one pair.
        """
        entity_ids = [eid for eid in self._get_ids(entity_results, "id") if eid]
        deduped = await self._fetch_incident_relationships(
            query, entity_ids, known_relationships
        )
        if not deduped:
            return {}
        logger.info(
            "Incident expansion: %s entity hits -> %s incident relationships",
            len(entity_ids),
            len(deduped),
        )
        return {"lightrag_incident_relationships": deduped}
