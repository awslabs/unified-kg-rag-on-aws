# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig

from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.retrieval.hybrid_scorer import HybridScorer
from unified_kg_rag.adapters.retrieval.token_manager import (
    SectionType,
    TokenManager,
)
from unified_kg_rag.adapters.storage.filter_schema import FilterFields
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchType,
)
from unified_kg_rag.shared import (
    ConfigurationError,
    IndexNotFoundError,
    get_logger,
    observability,
)
from unified_kg_rag.shared.utils.store_names import graph_label, store_name

logger = get_logger(__name__)

# Substrings in an error message that indicate a non-transient (fatal)
# misconfiguration — auth, credentials, endpoint/config, or an outright
# connection failure. A retriever should surface these to the caller rather
# than masking them as "0 results", which is indistinguishable from a genuine
# empty match and silently hides broken auth/config from the user.
_FATAL_ERROR_MARKERS: tuple[str, ...] = (
    "credential",
    "not configured",
    "endpoint",
    "auth",
    "forbidden",
    "unauthorized",
    "access denied",
    "accessdenied",
    "security token",
    "expiredtoken",
    "expired token",
    "invalidclienttoken",
    "failed to connect",
    "failed to establish connection",
)


def is_fatal_retrieval_error(exc: BaseException) -> bool:
    """Classify a retrieval error as fatal (re-raise) vs transient (degrade).

    Fatal = a clearly non-recoverable misconfiguration (auth/credentials/
    endpoint/config) or a connection failure: returning ``[]`` for these turns a
    real failure into a misleading "no results". Everything else (timeouts,
    throttling, a single malformed query) is treated as transient and the caller
    may degrade to an empty result list. ``ConnectionError`` (the store
    clients raise ``StoreConnectionError`` for a store they cannot reach), a
    missing suffix (``IndexNotFoundError``) and a ``ConfigurationError`` (such
    as a query embedding model the index was not built with, or a request the
    store rejects as invalid) are always fatal. Other failures arrive as an
    ``AWSServiceError`` carrying the original message, which the markers are
    matched against.
    """
    if isinstance(exc, ConnectionError | IndexNotFoundError | ConfigurationError):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _FATAL_ERROR_MARKERS)


# Source ids recorded per retrieval span; the hit count is always complete.
_MAX_SPAN_SOURCE_IDS = 50


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


class BaseGraphRAGRetriever(BaseRetriever, ABC):
    def __init__(self, config: Config, **kwargs: Any) -> None:
        BaseRetriever.__init__(self, **kwargs)
        self._config = config

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        try:
            search_query = SearchQuery(query=query)
            results = asyncio.run(self.aretrieve(search_query))

            documents = []
            for result in results:
                doc = Document(
                    page_content=result.content,
                    metadata={
                        "score": result.score,
                        "source": result.source,
                        "retriever_type": result.retriever_type,
                        **(result.metadata or {}),
                    },
                )
                documents.append(doc)

            return documents
        except Exception as e:
            logger.error("Document retrieval failed: %s", str(e))
            raise

    @abstractmethod
    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        pass

    def filter_fields(self) -> FilterFields | None:
        """Filter keys some index or label this retriever searches can match.

        Used to reject caller filters no target store can apply. ``None`` (the
        default, for backends without a declared schema) accepts every key.
        """
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
        surfaced or absorbed: fatal errors (auth/credentials/endpoint/
        connection, see `is_fatal_retrieval_error`) propagate so a broken
        configuration is not reported as "no results"; anything else (timeouts,
        throttling, a malformed query) is logged under ``label`` and degrades to
        an empty list so one failing section does not sink the whole search.
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

    async def _fuse_and_rerank(
        self, results_dict: dict[str, list[RetrievalResult]], **kwargs: Any
    ) -> list[RetrievalResult]:
        """``HybridScorer.fuse_and_rerank_results`` off the event-loop thread.

        Reranking is a blocking model call (a Bedrock round trip by default),
        so running it inline would stall every other query on the loop.
        """
        tracing = observability.is_enabled()
        with observability.span(
            "kg_rag.fusion",
            self._fusion_span_attributes(results_dict) if tracing else None,
        ) as span:
            # to_thread copies the context, so the rerank span nests here.
            results: list[RetrievalResult] = await asyncio.to_thread(
                self.hybrid_scorer.fuse_and_rerank_results, results_dict, **kwargs
            )
            if tracing:
                span.set_attributes(
                    {
                        "kg_rag.results_out": len(results),
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
        filter merged into the caller's filters), with the entity focus cleared
        so the seeds, not name matching, drive the traversal.
        """
        if not self.graph_retriever or not seed_entity_ids:
            return []

        search_query = query.model_copy(deep=True)
        search_query.label_prefixes = [self.config.indexing.neptune.entity_label_prefix]
        search_query.entity_focus = []
        search_query.filters = (search_query.filters or {}).copy()
        search_query.filters["id"] = seed_entity_ids

        return await self._safe_aretrieve(
            self.graph_retriever, search_query, "Neptune graph expansion"
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
    def _endpoint_pair(result: RetrievalResult) -> tuple[str, str]:
        """The undirected endpoint key upstream dedups edges by (`tuple(sorted(e))`)."""
        metadata = result.metadata or {}
        return tuple(  # type: ignore[return-value]
            sorted((str(metadata.get("source_id")), str(metadata.get("target_id"))))
        )

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
        undirected endpoint pair (also against ``known_relationships``) and
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
        # edge the caller already holds is not paid for twice.
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
        ids_set: set[str] = set()
        for result in results:
            ids = result.metadata.get(key)
            # OpenSearch hits carry the canonical id in `source` and only echo
            # metadata[key] when the indexed _source happens to include that
            # field; fall back to `source` so graph expansion still gets seeds
            # (matching local/drift candidate-entity resolution).
            if not ids and key == "id" and result.source:
                ids = result.source
            if ids:
                if isinstance(ids, list):
                    ids_set.update(str(id_val) for id_val in ids)
                else:
                    ids_set.add(str(ids))
        return list(ids_set)

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
