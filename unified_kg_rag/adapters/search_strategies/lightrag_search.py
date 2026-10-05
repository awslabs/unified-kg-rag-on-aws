# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LightRAG dual-level keyword search strategy.

Implements LightRAG's local/global/hybrid/mix/naive retrieval on top of
unified-kg-rag-on-aws's *shared* infrastructure rather than as a separate, reduced path:

- low-level keywords (``ll_keywords``) -> entities index (lexical + vector),
- high-level keywords (``hl_keywords``) -> relationships index (lexical + vector),
- the entity hits are expanded one hop to their incident relationships and
  the relationship hits to their endpoint entities (both from OpenSearch),
- ``mix`` additionally pulls the source chunks referenced by the matched
  entities/relationships (following ``text_unit_ids`` lineage, allocating slots
  per hit by weighted polling so every match keeps at least one chunk, and
  deduplicating against the naive vector chunk stream — LightRAG's
  ``_find_related_text_unit_from_entities``/``_from_relations`` +
  ``pick_by_weighted_polling`` + ``_merge_chunks``) and blends a naive vector
  chunk retrieval,
- everything is fused and reranked via the shared :class:`HybridScorer`
  (BM25 lexical + vector semantic + graph + RRF + Bedrock rerank).

So a LightRAG-mode query enjoys the same hybrid scoring, multilingual handling,
and caching as the GraphRAG strategies — only the retrieval algorithm differs.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import boto3

from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
)
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
)
from unified_kg_rag.domain.retrieval.strategy_registry import (
    QueryInput,
    register_strategy,
)
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

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

    Port of upstream LightRAG's ``utils.pick_by_weighted_polling`` (the default
    ``kg_chunk_pick_method = WEIGHT`` path used by
    ``_find_related_text_unit_from_entities``/``_from_relations``). ``lineages``
    is one chunk-id list per matched entity/relationship, already in retrieval
    (importance) order.

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

    # Re-distribute the quota that short lineages could not fill, one chunk per
    # scan, so leftover slots still spread across items instead of piling onto
    # the single richest one.
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


# NAIVE is vector-chunk-only (no graph), so it declares DOCUMENT-only roles;
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
    (the chain sets it; ``query.metadata['lightrag_mode']`` is still read for
    direct callers; default ``mix``):

    - ``naive``: vector chunk retrieval only (no graph).
    - ``hybrid``: ll->entities + hl->relationships + one-hop cross-type
      expansion + the chunks those items cite.
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
        self._lightrag_config = config.search.lightrag_search

    def _kg_stream_top_k(self, top_k: int) -> int:
        """Width of the entity/relationship vector queries (`QueryParam.top_k`)."""
        return max(self._lightrag_config.kg_stream_top_k, top_k)

    def _chunk_stream_top_k(self, top_k: int) -> int:
        """Width of the chunk vector query (`QueryParam.chunk_top_k`)."""
        return max(self._lightrag_config.chunk_stream_top_k, top_k)

    def _incident_fetch_limit(self) -> int:
        """How many incident edges to request per endpoint side.

        Upstream's entity->incident-edge expansion has NO count limit — it fetches
        every edge touching the entity hits and lets the per-type TOKEN budget
        (DEFAULT_MAX_RELATION_TOKENS = 8000) do the trimming. An OpenSearch query
        cannot be unbounded, so ask for the largest page the retriever will
        actually grant (`opensearch.max_query_size`) rather than a larger constant
        the retriever would silently clamp. Applied per endpoint side, so up to
        ~2x that many edges reach the dedup, which keeps the token budget rather
        than a count the binding constraint, as upstream.
        """
        return self._os_config.max_query_size

    def _mode(self, query: SearchQuery) -> str:
        metadata = query.metadata
        mode = metadata.get("search_strategy") or metadata.get("lightrag_mode")
        return str(mode or SearchStrategy.MIX.value)

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
        final_results = self.hybrid_scorer.fuse_and_rerank_results(
            results_by_source,
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
            per_type_quota=per_type_quota,
            rerank_only_types=rerank_only_types,
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
        3. The source chunks cited by the matched entities/relationships
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
        for streams in await asyncio.gather(*first_round):
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
        for streams in await asyncio.gather(*second_round):
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

        Seeds keep rank order (entity hits first, then endpoints in first-seen
        order) because the graph retriever caps the seed count.
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
        """Naive vector chunk retrieval over the text-units index."""
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
        """Per-hit ``text_unit_ids`` lineage, in retrieval (importance) order."""
        lineages: list[list[str]] = []
        for result in results:
            metadata = result.metadata or {}
            unit_ids = metadata.get("text_unit_ids")
            # Neptune's `_clean_property_map` unwraps a
            # single-element value_map list into a bare scalar, so an entity or
            # relationship whose lineage is exactly ONE chunk arrives as a str.
            # The old `isinstance(..., list)` guard skipped those outright,
            # discarding precisely the most specific (single-document) bridge
            # items — the ones multi-hop questions hinge on.
            if isinstance(unit_ids, str):
                unit_ids = [unit_ids]
            if not isinstance(unit_ids, list):
                continue
            lineage = [uid for uid in unit_ids if isinstance(uid, str) and uid]
            if lineage:
                lineages.append(lineage)
        return lineages

    def _collect_linked_chunk_ids(
        self,
        entity_results: list[RetrievalResult],
        relationship_results: list[RetrievalResult],
        limit: int,
        exclude: set[str] | None = None,
    ) -> list[str]:
        """Select the source chunks cited by the matched entities/relationships.

        Mirrors upstream LightRAG's two-stage selection
        (``_find_related_text_unit_from_entities`` then ``_from_relations``):

        1. per-hit lineages are deduplicated keeping the EARLIEST (most
           important) citing hit, so a chunk occupies one item's quota, not many;
        2. slots are allocated by :func:`_pick_by_weighted_polling`, which
           guarantees every matched hit at least one chunk;
        3. relationship lineages additionally drop chunks already selected from
           the entity side, and — via ``exclude`` — chunks the naive vector
           stream already returned (upstream ``operate.py`` passes
           ``entity_chunks`` into the relation pass for exactly this, then
           round-robin merges the three streams under one ``seen_chunk_ids``).

        The previous implementation ranked the pooled lineage by global citation
        count and cut at ``limit``, with no dedup against the vector chunk stream, so
        a large share of the linked stream spent slots on chunks the vector stream had
        already retrieved.
        """
        excluded = set(exclude or ())

        def _select(results: list[RetrievalResult]) -> list[str]:
            deduped: list[list[str]] = []
            for lineage in self._lineages(results):
                keep = [uid for uid in lineage if uid not in excluded]
                # Keep the chunk on its earliest (most important) citing hit only.
                excluded.update(keep)
                if keep:
                    deduped.append(keep)
            return _pick_by_weighted_polling(
                deduped, self._lightrag_config.related_chunk_number
            )

        selected = _select(entity_results) + _select(relationship_results)
        return selected[:limit]

    async def _retrieve_linked_chunks(
        self,
        query: SearchQuery,
        entity_results: list[RetrievalResult],
        relationship_results: list[RetrievalResult],
        exclude: set[str] | None = None,
    ) -> dict[str, list[RetrievalResult]]:
        """Fetch the source chunks cited by the matched entities/relationships.

        Collects ``text_unit_ids`` lineage from the entity/relationship hits,
        allocates chunk slots per hit by weighted polling, and fetches the
        selected chunks by id from the text-units index. ``exclude`` carries the
        chunk ids the naive vector stream already returned.
        Degrades to no section if lineage is absent (e.g. an index built before
        lineage was added) or retrieval fails.
        """
        if not self.document_retriever:
            return {}
        # The lineage pool is a CHUNK stream, so it is bounded by
        # upstream's chunk width, not by the caller's top_k. Upstream bounds it in two
        # steps — `related_chunk_number` (5) chunks per matched entity/relationship, then
        # a truncation of the combined pool to `chunk_top_k` — so with 40 KG hits its
        # pool is far larger than a flat top_k=10 before the final cut.
        chunk_ids = self._collect_linked_chunk_ids(
            entity_results,
            relationship_results,
            limit=self._chunk_stream_top_k(query.top_k),
            exclude=exclude,
        )
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
        results = await self._safe_aretrieve(
            self.document_retriever, search_query, "Linked chunk retrieval"
        )
        return {"lightrag_linked_chunks": results} if results else {}

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
        results = await self._safe_aretrieve(
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

    @staticmethod
    def _endpoint_pair(result: RetrievalResult) -> tuple[str, str]:
        """The undirected endpoint key upstream dedups edges by (`tuple(sorted(e))`)."""
        metadata = result.metadata or {}
        return tuple(  # type: ignore[return-value]
            sorted((str(metadata.get("source_id")), str(metadata.get("target_id"))))
        )

    async def _retrieve_incident_relationships(
        self,
        query: SearchQuery,
        entity_results: list[RetrievalResult],
        known_relationships: list[RetrievalResult] | None = None,
    ) -> dict[str, list[RetrievalResult]]:
        """Every relationship INCIDENT to a matched entity.

        Upstream LightRAG's `_find_most_related_edges_from_entities` (operate.py
        ~5204) takes the entity vector hits, pulls **all** edges incident to them
        via `get_nodes_edges_batch`, dedups by sorted endpoint pair, and orders by
        `(rank, weight)` — with no top_k cut. That cross-type expansion is where the
        bulk of upstream's relationship context comes from; the relationship vector
        query alone supplies a small fraction of it. The relationship stream was never
        truncated on OUR side — the candidates simply did not exist.

        The relationships index stores its endpoints as `source_id`/`target_id`
        keywords, but `_build_filter_clauses` ANDs every filter, so
        `source_id OR target_id` needs two queries. `known_relationships` carries
        the hl vector query's hits, which are dropped from this stream by ENDPOINT
        PAIR (not doc id) — that is the key upstream's round-robin merge of the
        local and global relation lists dedups on (`_merge_context` builds
        `rel_key = tuple(sorted([src_id, tgt_id]))` over one `seen_relations` set).
        """
        retriever = self.document_retriever
        if not retriever:
            return {}
        entity_ids = [eid for eid in self._get_ids(entity_results, "id") if eid]
        if not entity_ids:
            return {}

        page = self._incident_fetch_limit()

        async def _by_endpoint(field: str) -> list[RetrievalResult]:
            search_query = SearchQuery(
                query="",
                search_type=SearchType.LEXICAL,
                top_k=page,
                index_prefixes=[self._os_config.relationships_index_prefix],
                suffix=query.suffix,
                filters=self._scoped_filters(query, **{field: entity_ids}),
            )
            side = await self._safe_aretrieve(
                retriever, search_query, f"Incident relationship retrieval ({field})"
            )
            # A full page means the count cap bound and edges were dropped —
            # upstream drops none. Say so rather than let a silent truncation read
            # as "all edges".
            if len(side) >= page:
                logger.warning(
                    "Incident-edge fetch on %s hit the %s-hit page cap; "
                    "the expansion is truncated (upstream truncates by tokens only)",
                    field,
                    page,
                )
            return side

        sides = await asyncio.gather(
            _by_endpoint("source_id"), _by_endpoint("target_id")
        )

        # Dedup by the undirected endpoint pair, as upstream does (`tuple(sorted(e))`),
        # so an edge reachable from both of its endpoints is carried once — and so an
        # edge the hl vector query already returned is not paid for twice.
        seen: set[tuple[str, str]] = {
            self._endpoint_pair(r) for r in (known_relationships or [])
        }
        deduped: list[RetrievalResult] = []
        for result in [r for side in sides for r in side]:
            pair = self._endpoint_pair(result)
            if pair in seen:
                continue
            seen.add(pair)
            deduped.append(result)

        if not deduped:
            return {}
        # Upstream orders these by (rank, weight) descending — degree first, so the
        # hub edges that carry a multi-hop chain outrank incidental leaf edges.
        deduped.sort(
            key=lambda r: (
                float((r.metadata or {}).get("rank") or 0.0),
                float((r.metadata or {}).get("weight") or 0.0),
            ),
            reverse=True,
        )
        logger.info(
            "Incident expansion: %s entity hits -> %s incident relationships",
            len(entity_ids),
            len(deduped),
        )
        return {"lightrag_incident_relationships": deduped}
