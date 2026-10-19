# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig

from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.retrieval.hybrid_scorer import HybridScorer, RerankTrace
from unified_kg_rag.adapters.retrieval.token_manager import (
    SectionType,
    TokenManager,
)
from unified_kg_rag.adapters.storage.filter_schema import (
    FilterFields,
    document_filters,
    id_batch_size,
    opensearch_filter_fields,
)
from unified_kg_rag.adapters.store_errors import is_fatal_retrieval_error
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchType,
)
from unified_kg_rag.shared import get_logger, observability
from unified_kg_rag.shared.utils.store_names import graph_label, store_name

logger = get_logger(__name__)

# Source ids recorded per retrieval span; the hit count is always complete.
_MAX_SPAN_SOURCE_IDS = 50

# What one query has learned about its stores (``query_scope``).
_QUERY_SCOPE: ContextVar[dict[str, Any] | None] = ContextVar(
    "kg_rag_query_scope", default=None
)


@contextmanager
def query_scope() -> Iterator[dict[str, Any]]:
    """A store-lookup memo for the sub-searches of one query.

    ``GraphRAGChain`` opens one per query; a retriever may keep what it
    learned there (``OpenSearchRetriever``'s attribute probes) and must not
    keep it past the query, so an ingestion or reset in the same process is
    seen by the next one. The memo is shared by every task the query spawns.
    """
    memo: dict[str, Any] = {}
    token = _QUERY_SCOPE.set(memo)
    try:
        yield memo
    finally:
        _QUERY_SCOPE.reset(token)


def current_query_scope() -> dict[str, Any] | None:
    """The memo of the query in progress (``query_scope``), if any."""
    return _QUERY_SCOPE.get()


def _retrieval_attributes(results: list[RetrievalResult]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    for result in results:
        key = str(result.retriever_type or "unknown")
        by_type[key] = by_type.get(key, 0) + 1
    attributes: dict[str, Any] = {
        "kg_rag.hits": len(results),
        "kg_rag.source_ids": [
            str(r.source) for r in results[:_MAX_SPAN_SOURCE_IDS] if r.source
        ],
    }
    for result_type, count in by_type.items():
        attributes[f"kg_rag.hits.{result_type}"] = count
    return attributes


def _text_unit_ids(result: RetrievalResult) -> list[str]:
    """The ``text_unit_ids`` a result cites (a graph store unwraps one to a str)."""
    ids = (result.metadata or {}).get("text_unit_ids") or []
    return [str(ids)] if isinstance(ids, str) else [str(i) for i in ids]


class BaseGraphRAGRetriever(BaseRetriever, ABC):
    def __init__(self, config: Config, **kwargs: Any) -> None:
        BaseRetriever.__init__(self, **kwargs)
        self._config = config

    # The LangChain retriever interface (``invoke``/``ainvoke``) over
    # :meth:`aretrieve`; the strategies call ``aretrieve`` directly.
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Sync ``invoke``: runs :meth:`aretrieve` on a new event loop, so it
        cannot be called from inside a running one (use ``ainvoke`` there)."""
        return self._to_documents(asyncio.run(self.aretrieve(SearchQuery(query=query))))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """``ainvoke``: awaits :meth:`aretrieve` on the caller's event loop."""
        return self._to_documents(await self.aretrieve(SearchQuery(query=query)))

    @staticmethod
    def _to_documents(results: list[RetrievalResult]) -> list[Document]:
        return [
            Document(
                page_content=result.content,
                metadata={
                    "score": result.score,
                    "source": result.source,
                    "retriever_type": result.retriever_type,
                    **(result.metadata or {}),
                },
            )
            for result in results
        ]

    @abstractmethod
    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        pass

    def filter_fields(self) -> FilterFields | None:
        """Filter keys some index or label this retriever searches can match.

        Used to reject caller filters no target store can apply. ``None`` (the
        default, for backends without a declared schema) accepts every key.
        """
        return None

    async def aread_build_record(self, suffix: str | None) -> dict[str, Any] | None:
        """The build record ingestion wrote on ``suffix``'s index
        (``VectorIndexer.record_build``), read by ``run-eval`` into its run
        manifest. ``None`` (the default, for a backend without index
        metadata) when there is none or it cannot be read."""
        return None

    def _get_name(self, base: str, suffix: str | None) -> str:
        return store_name(base, suffix, self._config.indexing.additional_suffix)

    def _get_label(self, prefix: str, suffix: str | None) -> str:
        """Graph vertex label for ``prefix`` under ``suffix``."""
        return graph_label(prefix, suffix, self._config.indexing.additional_suffix)


class BaseSearchStrategy(ABC):
    """A search algorithm over role-keyed retrievers.

    ``GraphRAGChain`` builds one instance per strategy and event loop and
    reuses it for every query, including concurrent ones. Subclasses must
    therefore keep per-query state local to ``asearch`` (and what it calls)
    and treat instance attributes as read-only after ``__init__``.
    """

    # Section types that take the context budget other types leave unused
    # before the rest share it (``TokenManager.optimize_context``).
    spare_context_budget_first: ClassVar[tuple[SectionType, ...]] = ()

    def __init__(
        self,
        config: Config,
        retrievers: dict[str, BaseGraphRAGRetriever],
        *,
        providers: Providers | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config
        # Retrievers are keyed by RetrieverRole value ("graph" / "document"),
        # not by concrete backend, so strategies stay backend-agnostic.
        self.retrievers = retrievers
        # Session and model providers come from the orchestrator's shared
        # bundle (GraphRAGChain passes its own), so an injected provider reaches
        # the strategy's LLM chains, scorer and token counter alike. Without
        # one, a default bundle is built from ``config``.
        self.providers = Providers.resolve(config, providers)
        self.hybrid_scorer = HybridScorer(self.config, providers=self.providers)
        self.token_manager = TokenManager(self.config, providers=self.providers)

    @property
    def graph_retriever(self) -> BaseGraphRAGRetriever | None:
        """The retriever bound to the GRAPH role (graph traversal/expansion)."""
        return self.retrievers.get(RetrieverRole.GRAPH.value)

    @property
    def document_retriever(self) -> BaseGraphRAGRetriever | None:
        """The retriever bound to the DOCUMENT role (vector/lexical lookup)."""
        return self.retrievers.get(RetrieverRole.DOCUMENT.value)

    async def _safe_aretrieve(
        self,
        retriever: BaseGraphRAGRetriever,
        query: SearchQuery,
        label: str,
    ) -> list[RetrievalResult]:
        """Run one sub-retrieval, re-raising fatal errors and degrading the rest.

        The single place a strategy decides whether a retrieval failure is
        surfaced or absorbed: fatal errors (an unreachable, hung or overloaded
        store, denied or invalid requests, a model call that cannot succeed:
        see `is_fatal_retrieval_error`) propagate so a broken deployment is not
        reported as "no results"; anything else (one request the store failed)
        is logged under ``label`` and degrades to an empty list so one failing
        section does not sink the whole search.
        """
        with observability.span(
            "kg_rag.retrieve",
            retriever=type(retriever).__name__,
            label=label,
            search_type=query.search_type,
            top_k=query.top_k,
        ) as span:
            try:
                results = await retriever.aretrieve(query)
            except Exception as e:
                if is_fatal_retrieval_error(e):
                    raise
                logger.error(
                    "%s: %s failed (degrading to empty results): %s",
                    type(self).__name__,
                    label,
                    e,
                )
                span.record_exception(e)
                span.set_attributes({"kg_rag.degraded": True, "kg_rag.hits": 0})
                return []
            # Hit counts and source ids only: never the retrieved text.
            if observability.is_enabled():
                span.set_attributes(_retrieval_attributes(results))
            return results

    async def _ids_passing(
        self,
        query: SearchQuery,
        prefix: str,
        ids: Iterable[str],
        filters: dict[str, Any],
        *,
        enough: int | None = None,
    ) -> set[str]:
        """The ``ids`` of ``prefix``'s index that pass ``filters``.

        Ids only come back, in batches one request can hold
        (``id_batch_size``), in the order given; with ``enough`` the batches
        stop once that many passed. A failed request ends the check with what
        passed so far (the caller treats the rest as failing: fail-closed)
        and marks the span degraded.
        """
        retriever = self.document_retriever
        ordered = list(dict.fromkeys(ids))
        passed: set[str] = set()
        if retriever is None or not ordered:
            return passed
        batch = id_batch_size(self.config.indexing.opensearch.index_settings)
        with observability.span(
            "kg_rag.retrieve",
            retriever=type(retriever).__name__,
            label="Attribute filter check",
            index=prefix,
        ) as span:
            for start in range(0, len(ordered), batch):
                chunk = ordered[start : start + batch]
                check = SearchQuery(
                    query="",
                    search_type=SearchType.LEXICAL,
                    top_k=len(chunk),
                    retrieval_multiplier=1,
                    index_prefixes=[prefix],
                    suffix=query.suffix,
                    filters={**filters, "id": chunk},
                    source_fields=[],
                )
                try:
                    passed |= {str(h.source) for h in await retriever.aretrieve(check)}
                except Exception as e:
                    if is_fatal_retrieval_error(e):
                        raise
                    logger.warning(
                        "%s: the attribute filter check on '%s' failed; what it "
                        "could not check counts as failing: %s",
                        type(self).__name__,
                        prefix,
                        e,
                    )
                    span.record_exception(e)
                    span.set_attributes({"kg_rag.degraded": True})
                    break
                if enough is not None and len(passed) >= enough:
                    break
        return passed

    def _text_unit_filters(self, filters: dict[str, Any] | None) -> dict[str, Any]:
        """The keys of ``filters`` that narrow the text units (every key the
        text-units index declares but ``id``): document filters, ``n_tokens``
        ranges and the like."""
        declared = opensearch_filter_fields(
            *self.config.processing.translation.translated_languages
        )["text_units"]
        return {
            k: v
            for k, v in (filters or {}).items()
            if k != "id" and declared.declares(k)
        }

    async def _passing_ranked_units(
        self, query: SearchQuery, ranked: list[str], quota: int
    ) -> list[str]:
        """The first ``quota`` of ``ranked`` text unit ids that pass the
        caller's filters, in rank order.

        Cutting the ranked ids to the quota before a filtered fetch left the
        slots of failing ids empty; fetching every ranked unit's content to
        cut afterwards cost a hub's whole lineage. Under a filter on text
        units, an ids-only check walks the ranked ids until ``quota`` pass,
        and only those are fetched with their content.
        """
        filters = self._text_unit_filters(query.filters)
        if not filters:
            return ranked[:quota]
        passed = await self._ids_passing(
            query,
            self.config.indexing.opensearch.text_units_index_prefix,
            ranked,
            filters,
            enough=quota,
        )
        return [unit for unit in ranked if unit in passed][:quota]

    async def _vouch_by_entities_index(
        self, query: SearchQuery, results: list[RetrievalResult], seeds: set[str]
    ) -> list[RetrievalResult]:
        """Graph-expansion ``results`` without the vertices that do not pass
        the document filters.

        Graph vertices hold no document attribute (``document_filters``), so
        a graph store cannot apply one. The entities index does: an entity
        document carries the filters of every document it comes from, so it
        passes when one of them does, and each vertex other than the
        ``seeds`` is checked there by id (``_ids_passing``). Those values are
        unioned per key, though, so under two or more keys an entity of two
        documents that each pass one key would pass while none passes both:
        its survivors are then also checked by their text units, each of
        which belongs to one document. Seeds, relationships and reports
        match each key independently (see the user guide).
        """
        filters = document_filters(query.filters)
        if not filters:
            return results
        o = self.config.indexing.opensearch
        others = [r for r in results if r.source not in seeds]
        passed = await self._ids_passing(
            query, o.entities_index_prefix, (str(r.source) for r in others), filters
        )
        if len(filters) > 1:
            units = {
                str(r.source): _text_unit_ids(r)
                for r in others
                if str(r.source) in passed
            }
            vouched = await self._ids_passing(
                query,
                o.text_units_index_prefix,
                (u for ids in units.values() for u in ids),
                filters,
            )
            passed = {v for v, ids in units.items() if vouched.intersection(ids)}
        kept = [r for r in results if r.source in seeds or str(r.source) in passed]
        if len(kept) < len(results):
            logger.debug(
                "Document filters dropped %s of %s expanded entities",
                len(results) - len(kept),
                len(results),
            )
        return kept

    def _new_rerank_trace(self) -> RerankTrace:
        """The per-query rerank record ``asearch`` passes to ``_fuse_and_rerank``
        and reports as ``SearchResult.metadata["rerank"]``."""
        return self.hybrid_scorer.new_trace()

    async def _fuse_and_rerank(
        self,
        results_dict: dict[str, list[RetrievalResult]],
        *,
        rerank_trace: RerankTrace,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """``HybridScorer.fuse_and_rerank_results`` off the event-loop thread.

        Reranking is a blocking model call (a Bedrock round trip by default),
        so running it inline would stall every other query on the loop. The
        rerank outcome is recorded on ``rerank_trace``, the calling query's own
        and required: a throwaway default would report the query as skipped
        while its rerank failed.
        """
        tracing = observability.is_enabled()
        with observability.span(
            "kg_rag.fusion",
            self._fusion_span_attributes(results_dict) if tracing else None,
        ) as span:
            # to_thread copies the context, so the rerank span nests here.
            results: list[RetrievalResult] = await asyncio.to_thread(
                self.hybrid_scorer.fuse_and_rerank_results,
                results_dict,
                rerank_trace=rerank_trace,
                **kwargs,
            )
            if tracing:
                span.set_attributes(
                    {
                        "kg_rag.results_out": len(results),
                        "kg_rag.rerank_status": rerank_trace.stages[-1].status.value,
                        "kg_rag.reranked": sum(
                            bool((r.metadata or {}).get("reranked")) for r in results
                        ),
                    }
                )
            return results

    def _fusion_span_attributes(
        self, results_dict: dict[str, list[RetrievalResult]]
    ) -> dict[str, Any]:
        return {
            "kg_rag.fusion.method": self.config.search.fusion.method,
            "kg_rag.streams": sorted(results_dict),
            "kg_rag.results_in": sum(len(v) for v in results_dict.values()),
            "kg_rag.rerank_enabled": self.hybrid_scorer.rerank_model is not None,
        }

    def _per_type_quota(self, top_k: int) -> dict[str, int]:
        """Reserved fusion slots per section type, from configured shares of top_k.

        Shared by local search and DRIFT (whose iterations are local searches,
        as in MS GraphRAG) so one flat top_k cut cannot let a single section
        type crowd out the others (``search.local_search.type_quota``).
        """
        quota_config = self.config.search.local_search.type_quota
        shares: list[tuple[SectionType, float, int]] = [
            (SectionType.TEXT, quota_config.text_multiplier, quota_config.text_floor),
            (
                SectionType.ENTITY,
                quota_config.entity_multiplier,
                quota_config.entity_floor,
            ),
            (
                SectionType.RELATIONSHIP,
                quota_config.relationship_multiplier,
                quota_config.relationship_floor,
            ),
            (
                SectionType.COMMUNITY,
                quota_config.community_multiplier,
                quota_config.community_floor,
            ),
            (
                SectionType.CLAIM,
                quota_config.claim_multiplier,
                quota_config.claim_floor,
            ),
        ]
        return {
            section_type.value: max(int(multiplier * top_k), floor)
            for section_type, multiplier, floor in shares
        }

    async def _expand_via_graph(
        self, query: SearchQuery, seed_entity_ids: list[str]
    ) -> list[RetrievalResult]:
        """Expand seed entities through the graph (GraphRAG local and LightRAG).

        Re-queries the GRAPH retriever pinned to the seed entity ids (an ``id``
        filter merged into the caller's filters).
        """
        if not self.graph_retriever or not seed_entity_ids:
            return []

        search_query = query.model_copy(deep=True)
        search_query.filters = (search_query.filters or {}).copy()
        search_query.filters["id"] = seed_entity_ids

        expanded = await self._safe_aretrieve(
            self.graph_retriever, search_query, "graph expansion"
        )
        return await self._vouch_by_entities_index(
            query, expanded, {str(s) for s in seed_entity_ids}
        )

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
        return self.config.indexing.opensearch.max_query_size

    @staticmethod
    def _relationship_identity(result: RetrievalResult) -> str:
        """The stored relationship's id (its content when it has none)."""
        metadata = result.metadata or {}
        return str(metadata.get("id") or result.source or result.content)

    async def _fetch_incident_relationships(
        self,
        query: SearchQuery,
        entity_ids: list[str],
        known_relationships: list[RetrievalResult] | None = None,
        bridge_first: bool = False,
    ) -> list[RetrievalResult]:
        """Every relationship incident to ``entity_ids``, deduped and ordered.

        The relationships index stores its endpoints as `source_id`/`target_id`
        keywords, but `_build_filter_clauses` ANDs every filter, so
        `source_id OR target_id` needs two queries. Edges are deduped by
        relationship id (also against ``known_relationships``) and
        ordered by `(rank, weight)` descending, as upstream LightRAG's
        `_find_most_related_edges_from_entities` does. With ``bridge_first``
        the edges whose BOTH endpoints are in ``entity_ids`` — the bridges
        that connect two retrieved entities, i.e. the hops of a multi-hop
        chain — come before edges that leave the set.
        """
        retriever = self.document_retriever
        if not retriever or not entity_ids:
            return []

        page = self._incident_fetch_limit()
        relationships_prefix = (
            self.config.indexing.opensearch.relationships_index_prefix
        )

        async def _by_endpoint(field: str) -> list[RetrievalResult]:
            search_query = SearchQuery(
                query="",
                search_type=SearchType.LEXICAL,
                top_k=page,
                index_prefixes=[relationships_prefix],
                suffix=query.suffix,
                filters=self._scoped_filters(query, **{field: entity_ids}),
            )
            side = await self._safe_aretrieve(
                retriever, search_query, f"Incident relationship retrieval ({field})"
            )
            # A text-less search is ordered by (rank, weight) server-side (the
            # retriever's importance sort), so a full page holds the best edges.
            # Still, the count cap bound and edges were dropped — upstream drops
            # none. Say so rather than let a silent truncation read as "all edges".
            if len(side) >= page:
                logger.warning(
                    "Incident-edge fetch on %s hit the %s-hit page cap; the "
                    "lowest-ranked edges are dropped (upstream truncates by "
                    "tokens only)",
                    field,
                    page,
                )
            return side

        sides = await asyncio.gather(
            _by_endpoint("source_id"), _by_endpoint("target_id")
        )

        # Dedup by relationship id, so an edge reachable from both of its
        # endpoints is carried once and an edge the caller already holds is not
        # paid for twice. Upstream dedups by the undirected endpoint pair
        # (`tuple(sorted(e))`) because its graph holds one edge per pair; ours
        # keeps typed, directed edges, and a pair key dropped every edge between
        # two entities but the first ("supplies" and "invoices" alike).
        seen = {self._relationship_identity(r) for r in (known_relationships or [])}
        deduped: list[RetrievalResult] = []
        for result in [r for side in sides for r in side]:
            identity = self._relationship_identity(result)
            if identity in seen:
                continue
            seen.add(identity)
            deduped.append(result)

        in_set = set(entity_ids)

        def _order(r: RetrievalResult) -> tuple[float, ...]:
            metadata = r.metadata or {}
            key = (
                float(metadata.get("rank") or 0.0),
                float(metadata.get("weight") or 0.0),
            )
            if not bridge_first:
                return key
            bridge = (
                str(metadata.get("source_id")) in in_set
                and str(metadata.get("target_id")) in in_set
            )
            return (float(bridge), *key)

        # Upstream orders these by (rank, weight) descending — degree first, so the
        # hub edges that carry a multi-hop chain outrank incidental leaf edges.
        deduped.sort(key=_order, reverse=True)
        return deduped

    @staticmethod
    def _restore_rank(
        results: list[RetrievalResult], ranked_ids: list[str]
    ) -> list[RetrievalResult]:
        """``results`` of a fetch by id, in ``ranked_ids`` order.

        A fetch by id (``query=""``, LEXICAL, pure id filter) comes back in
        index order with no meaningful relevance score (typically 0.0), which
        would throw away the order the ids were selected in (graph support in
        local search, weighted polling in LightRAG). Re-impose it, and project
        it into ``score`` as a normalized descending value so the shared fusion
        and per-type-quota path (which sorts by score) preserves it instead of
        tie-breaking arbitrarily.
        """
        if not results or not ranked_ids:
            return results
        rank_of = {unit_id: i for i, unit_id in enumerate(ranked_ids)}
        fallback = len(ranked_ids)
        ordered = sorted(results, key=lambda r: rank_of.get(str(r.source), fallback))
        total = len(ordered)
        rescored: list[RetrievalResult] = []
        for position, result in enumerate(ordered):
            scored = result.model_copy()
            scored.score = (total - position) / total
            rescored.append(scored)
        return rescored

    @abstractmethod
    async def asearch(
        self, query: SearchQuery, config: RunnableConfig | None = None
    ) -> SearchResult:
        """Run the strategy for ``query``.

        ``config`` is the caller's LangChain ``RunnableConfig`` (callbacks,
        tags, metadata). Pass it to every LLM call the strategy makes, so the
        caller's callbacks see those calls nested under its run, rather than
        relying on implicit contextvar propagation, which LangChain documents
        only for async code on Python 3.11+.
        """

    @staticmethod
    def _get_ids(results: list[RetrievalResult], key: str) -> list[str]:
        """The distinct ``key`` ids of ``results``, in rank order (first seen).

        Not a ``set``: its order of str ids changes with ``PYTHONHASHSEED``,
        so every caller would have to sort to stay reproducible.
        """
        ids: dict[str, None] = {}
        for result in results:
            # OpenSearch hits carry the canonical id in `source` and only echo
            # metadata[key] when the indexed _source happens to include that
            # field; fall back to `source` so graph expansion still gets seeds
            # (matching local/drift candidate-entity resolution).
            value = result.metadata.get(key)
            if not value and key == "id" and result.source:
                value = result.source
            if value:
                values = value if isinstance(value, list) else [value]
                ids.update(dict.fromkeys(str(v) for v in values))
        return list(ids)

    @staticmethod
    def _scoped_filters(query: SearchQuery, **scope: Any) -> dict[str, Any] | None:
        """Caller filters (``query.filters``) merged with sub-query scope filters.

        Strategies that build a fresh ``SearchQuery`` for a sub-retrieval must
        carry the caller's attribute filters through, or ``--filters`` is
        silently dropped on that path. ``scope`` holds the filters the strategy
        itself adds (e.g. ``id=[...]`` for a fetch-by-id); on a key collision
        the scope wins, since it pins the fetch to specific artifacts. Returns a
        new dict (never the caller's) or ``None`` when there is nothing to apply.
        Indexes have different fields, so the retrievers drop each caller key
        from the sub-queries whose index (or vertex label) lacks that field.
        """
        merged = {**(query.filters or {}), **scope}
        return merged or None
