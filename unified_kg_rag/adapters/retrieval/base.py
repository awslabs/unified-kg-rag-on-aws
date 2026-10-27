# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Collection, Iterable, Iterator
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
    IndexLimits,
    document_filters,
    opensearch_filter_fields,
)
from unified_kg_rag.adapters.store_errors import is_fatal_retrieval_error
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
)
from unified_kg_rag.domain.retrieval.index_prefixes import configured_index_prefixes
from unified_kg_rag.shared import get_logger, observability
from unified_kg_rag.shared.exceptions import InvalidFilterError
from unified_kg_rag.shared.utils.common import compute_hash
from unified_kg_rag.shared.utils.concurrency import gather_bounded, gather_or_cancel
from unified_kg_rag.shared.utils.store_names import graph_label, store_name

logger = get_logger(__name__)

# Source ids recorded per retrieval span; the hit count is always complete.
_MAX_SPAN_SOURCE_IDS = 50
# What the filtered walk reads of an edge: its endpoints and their names, the
# order (rank, weight) and, under two or more filter keys, its text units.
_WALK_EDGE_FIELDS = (
    "id",
    "source_id",
    "target_id",
    "source_name",
    "target_name",
    "rank",
    "weight",
    "text_unit_ids",
)

# The relationships index's endpoint fields: source OR target takes a request
# each (``_build_filter_clauses`` ANDs every filter).
_ENDPOINTS = ("source_id", "target_id")
_OTHER_END = {"source_id": "target_id", "target_id": "source_id"}

# The query memo's entries (``query_scope``): the incident-edge pages, the
# filter check's outcomes and whether the incident page cap was logged.
_MEMO_INCIDENT = "incident"
_MEMO_PASSING = "passing"
_MEMO_PAGE_CAP = "incident page cap"

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


def as_float(value: Any) -> float | None:
    """``value`` as a float; ``None`` when it is absent or not a number."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def rank_expanded(
    results: Iterable[RetrievalResult], seeds: set[str], limit: int | None = None
) -> list[RetrievalResult]:
    """Graph-expansion ``results`` best first: by score, then by id, so equal
    scores do not come out in the order a store happened to return them.
    With ``limit``, the seeds plus the ``limit`` best other entities: the
    expansion over-fetches (``traversal_fetch_multiplier``) so the cut
    follows relevance, not traversal order. Seeds are never cut and do not
    use up the ``limit``: they are what the query matched, and local search
    passes ``top_k`` of them without an entity focus. The one cut both the
    graph walk (``GraphExpansionRetriever``) and the filtered walk make."""
    ranked = sorted(results, key=lambda r: (-(r.score or 0.0), str(r.source)))
    if limit is None:
        return ranked
    kept: list[RetrievalResult] = []
    others = 0
    for result in ranked:
        if str(result.source) not in seeds:
            if others >= limit:
                continue
            others += 1
        kept.append(result)
    return kept


def seed_share(query: SearchQuery, fetch_multiplier: int, seeds: int) -> int:
    """How many entities each of ``seeds`` seeds reaches at most: the
    expansion's budget (``top_k * retrieval_multiplier *
    traversal_fetch_multiplier``) split evenly, rounded up, so every seed
    gets one and the seeds together still reach the budget for the slots
    ``rank_expanded`` keeps. The one split of the graph walks and the
    filtered walk."""
    return math.ceil(
        query.top_k * query.retrieval_multiplier * fetch_multiplier / seeds
    )


def graph_relevance(path_length: int, importance: float) -> float:
    """A graph-expansion entity's relevance in (0, 1]: a vertex ``n`` hops
    from the seeds (path length ``L = n + 1``) scores in the band
    ``(1 / (L + 1), 1 / L]``, its ``importance`` (0-1) placing it inside
    the band, so every nearer vertex outranks every farther one. The one
    owner of the formula (``GraphExpansionRetriever`` and the filtered
    expansion)."""
    length = path_length or 1
    importance = min(max(importance, 0.0), 1.0)
    near, far = 1.0 / length, 1.0 / (length + 1)
    # (1 + importance) / 2 keeps the band's lower edge open, so the most
    # important node one hop farther out never ties the least important one.
    return far + (near - far) * (1.0 + importance) / 2.0


def entity_content(node: dict[str, Any], path_names: list[str]) -> str:
    """A graph-expansion entity's content: its name, description and, when
    reached from a seed, the ``Path:`` of entity names (seed first). The one
    place the format is written (``GraphExpansionRetriever`` and the
    filtered expansion's redrawn paths)."""
    parts = [
        f"Entity: {node.get('name', 'Unknown')}",
        f"Description: {node.get('description', 'N/A')}",
    ]
    if names := [name for name in path_names if name]:
        parts.append(f"Path: {' -> '.join(names)}")
    return "\n".join(parts)


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
        """The hits of one search: at most ``top_k * retrieval_multiplier``
        and no more than the index's ``aindex_limits`` allow, leaving out
        what ``query.exclude`` matches. Past those limits a backend may cut
        or refuse a request: the strategies size theirs by
        :meth:`aindex_limits` (``BaseSearchStrategy._fetch_by_ids`` asks a
        fetch by id in batches), so a backend need not batch one itself."""

    def filter_fields(self) -> FilterFields | None:
        """Filter keys some index or label this retriever searches can match.

        Used to reject caller filters no target store can apply. ``None`` (the
        default, for backends without a declared schema) accepts every key.
        """
        return None

    def filter_fields_for(self, prefixes: Collection[str]) -> FilterFields | None:
        """:meth:`filter_fields` of the indexes ``prefixes`` names only (the
        ones a strategy applies the caller's filters to). By default every
        index's: a backend without per-index schemas, or one ``prefixes``
        do not name (a graph store)."""
        return self.filter_fields()

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        """The limits in force on ``prefix``'s index for ``suffix``: the hits
        one search returns and the values one ``terms`` clause takes. By
        default the configured ``index_settings``'."""
        return IndexLimits.of(self._config.indexing.opensearch.index_settings)

    async def ahas_index(self, prefix: str, suffix: str | None) -> bool | None:
        """Whether ``prefix``'s index exists for ``suffix``; ``None`` (the
        default) when the backend cannot tell."""
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


SCOPED_FILTER_KEYS = frozenset({"id", "source_id", "target_id"})
"""The keys strategies scope their sub-searches by (a fetch by id, a walk
step's incident edges): a caller value on one is intersected with the scope
(``scope_filters``), and a range there is refused (``GraphRAGChain``)."""


def scope_filters(
    filters: dict[str, Any] | None, **scope: Any
) -> dict[str, Any] | None:
    """``filters`` with the ``scope`` a strategy adds to a sub-search (e.g.
    ``id=[...]`` for a fetch by id).

    A key both set keeps the scope's values the caller's also allows, in the
    scope's order (an empty list matches nothing), so the scope narrows a
    caller filter and never replaces it. A range on either side of such a
    key cannot be intersected as a list and is refused
    (``InvalidFilterError``; the chain refuses a caller range on
    ``SCOPED_FILTER_KEYS`` before the search). Returns a new dict, or
    ``None`` when there is nothing to apply.
    """
    merged = {**(filters or {})}
    for key, value in scope.items():
        if key not in merged:
            merged[key] = value
            continue
        given = merged[key]
        if isinstance(given, dict) or isinstance(value, dict):
            raise InvalidFilterError(
                f"Filter '{key}' is a range, but the search scopes '{key}' by "
                "value; give a value or a list of values."
            )
        allowed = set(given) if isinstance(given, list) else {given}
        values = value if isinstance(value, list) else [value]
        merged[key] = [v for v in values if v in allowed]
    return merged or None


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

    def filtered_index_prefixes(self, strategy: SearchStrategy) -> list[str] | None:
        """The document indexes ``strategy``'s searches apply the caller's
        filters to, which a filter key must be declared on
        (``GraphRAGChain`` checks it): a key on an index the strategy never
        reads is dropped there, and the query ran unfiltered. ``None`` (the
        default): every index of the document retriever."""
        return None

    def reserved_filter_keys(self, strategy: SearchStrategy) -> frozenset[str]:
        """The filter keys ``strategy`` scopes its searches by with values a
        caller value cannot be intersected with (a range, a browse from the
        roots), which ``GraphRAGChain`` therefore refuses. None by default:
        the id keys a strategy scopes (``SCOPED_FILTER_KEYS``) are
        intersected with the caller's values instead (``scope_filters``)."""
        return frozenset()

    def filtered_retrievers(
        self, strategy: SearchStrategy
    ) -> list[BaseGraphRAGRetriever]:
        """The retrievers ``strategy``'s searches apply the caller's filters
        to (``GraphRAGChain`` checks the keys against them). By default every
        one the strategy holds."""
        return list(self.retrievers.values())

    def _built_prefixes(self, *prefixes: str) -> list[str]:
        """The ``prefixes`` the configured pipeline builds."""
        return [p for p in configured_index_prefixes(self.config) if p in prefixes]

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
        failures: list[Exception] | None = None,
    ) -> list[RetrievalResult]:
        """Run one sub-retrieval, re-raising fatal errors and degrading the rest.

        The single place a strategy decides whether a retrieval failure is
        surfaced or absorbed: fatal errors (an unreachable, hung or overloaded
        store, denied or invalid requests, a model call that cannot succeed:
        see `is_fatal_retrieval_error`) propagate so a broken deployment is not
        reported as "no results"; anything else (one request the store failed)
        is logged under ``label`` and degrades to an empty list so one failing
        section does not sink the whole search. ``failures`` collects what was
        absorbed, for a caller that must not keep a degraded result.
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
                if failures is not None:
                    failures.append(e)
                return []
            # Hit counts and source ids only: never the retrieved text.
            if observability.is_enabled():
                span.set_attributes(_retrieval_attributes(results))
            return results

    async def _fetch_by_ids(
        self,
        retriever: BaseGraphRAGRetriever,
        query: SearchQuery,
        label: str,
    ) -> list[RetrievalResult]:
        """``query``, a fetch by id (no text, ``filters["id"]`` listing the
        ids), through ``_safe_aretrieve`` in batches one search returns in
        full: at most ``IndexLimits.id_batch_size`` ids of every index it
        reads (``aindex_limits``), each batch with a ``top_k`` of its ids, at
        most ``search.max_concurrency`` at once. Asked at once, ids past the
        window came back as the window's worth in the store's order, not the
        best-ranked ones. The hits come back batch after batch, each id once
        (a repeated id would be asked, and returned, by two batches)."""
        ids = list(dict.fromkeys((query.filters or {}).get("id") or ()))
        prefixes = query.index_prefixes or []
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        limits = [await retriever.aindex_limits(p, query.suffix) for p in prefixes]
        step = max(1, min((lim.id_batch_size for lim in limits), default=len(ids)))
        parts = await gather_bounded(
            self.config.search.max_concurrency,
            (
                self._safe_aretrieve(
                    retriever,
                    query.model_copy(
                        update={
                            "filters": {**(query.filters or {}), "id": batch},
                            "top_k": len(batch),
                            "retrieval_multiplier": 1,
                        }
                    ),
                    label,
                )
                for batch in (ids[i : i + step] for i in range(0, len(ids), step))
            ),
        )
        return [hit for part in parts for hit in part]

    async def _ids_passing(
        self,
        query: SearchQuery,
        prefix: str,
        ids: Iterable[str],
        filters: dict[str, Any],
        *,
        enough: int | None = None,
        checked: set[str] | None = None,
    ) -> set[str]:
        """The ``ids`` of ``prefix``'s index that pass ``filters``.

        Ids only come back, in batches one request can hold
        (``IndexLimits.id_batch_size``) of the ids not known yet. With ``enough`` the ids
        are counted in the order given and the check stops once ``enough`` of
        them passed: every id ranked above the ``enough``-th passing one is
        known, so a caller cutting the ranked ids to the passing ones keeps
        the right ones (an outcome memoized lower down cannot stand in for
        an unchecked id above it). Within a query (``query_scope``) each id's
        outcome is asked once (DRIFT's follow-ups check the same units).
        ``checked`` collects the ids whose outcome is known. A failed request
        ends the check with what passed so far (the caller treats the rest as
        failing: fail-closed) and marks the span degraded.
        """
        retriever = self.document_retriever
        ordered = list(dict.fromkeys(ids))
        passed: set[str] = set()
        if retriever is None or not ordered:
            return passed
        memo = current_query_scope()
        key = f"{_MEMO_PASSING}\n" + json.dumps(
            [prefix, query.suffix, filters], sort_keys=True, default=str
        )
        known: dict[str, bool] = memo.setdefault(key, {}) if memo is not None else {}
        batch = (await retriever.aindex_limits(prefix, query.suffix)).id_batch_size

        def note(outcomes: dict[str, bool]) -> None:
            if checked is not None:
                checked.update(outcomes)

        async def check(chunk: list[str]) -> bool:
            request = SearchQuery(
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
                found = {str(h.source) for h in await retriever.aretrieve(request)}
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
                return False
            outcomes = {i: i in found for i in chunk}
            known.update(outcomes)
            note(outcomes)
            return True

        with observability.span(
            "kg_rag.retrieve",
            retriever=type(retriever).__name__,
            label="Attribute filter check",
            index=prefix,
        ) as span:
            # ``scan`` gathers the next unknown ids; ``position`` counts the
            # outcomes in rank order up to the last id asked (each id is
            # visited once by each: no rescan of the rest per batch).
            position = scan = 0
            while position < len(ordered):
                unknown: list[str] = []
                last = len(ordered) - 1
                while scan < len(ordered) and len(unknown) < batch:
                    if ordered[scan] not in known:
                        unknown.append(ordered[scan])
                        last = scan
                    scan += 1
                if unknown and not await check(unknown):
                    break
                if not unknown:
                    last = len(ordered) - 1
                # Count in rank order up to the last id known now.
                while position <= last:
                    vertex = ordered[position]
                    position += 1
                    note({vertex: known[vertex]})
                    if known[vertex]:
                        passed.add(vertex)
                        if enough is not None and len(passed) >= enough:
                            return passed
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

    def _graph_unapplied_filters(self, query: SearchQuery) -> dict[str, Any]:
        """The caller filters a graph walk cannot apply to the entities it
        reaches but the entities index can: the document filters (vertices
        hold none, ``document_filters``) and the entity keys the graph does
        not store (``name.keyword``, ``attributes.<path>``), which the graph
        drops (``_scope_filters_to_label``). Without schemas to compare (a
        custom backend's ``filter_fields`` is ``None``), the document
        filters only."""
        unapplied = document_filters(query.filters)
        graph, documents = self.graph_retriever, self.document_retriever
        if not query.filters or graph is None or documents is None:
            return unapplied
        if (graph_fields := graph.filter_fields()) is None:
            return unapplied
        entity_fields = documents.filter_fields_for(
            [self.config.indexing.opensearch.entities_index_prefix]
        )
        if entity_fields is None:
            return unapplied
        for key, value in query.filters.items():
            if entity_fields.declares(key) and not graph_fields.declares(key):
                unapplied[key] = value
        return unapplied

    async def _vouch_by_entities_index(
        self,
        query: SearchQuery,
        results: list[RetrievalResult],
        seeds: set[str],
        filters: dict[str, Any],
    ) -> list[RetrievalResult]:
        """Graph-expansion ``results`` without the vertices that do not pass
        ``filters`` (``_graph_unapplied_filters``) on the entities index.

        Graph vertices hold no document attribute (``document_filters``), so
        a graph store cannot apply one, nor an entity key it does not store.
        The entities index does: an entity document carries the filters of
        every document it comes from, so it passes when one of them does,
        and each vertex other than the ``seeds`` is checked there by id
        (``_ids_passing``). Those document values are unioned per key,
        though, so under two or more document keys an entity of two
        documents that each pass one key would pass while none passes both:
        its survivors are then also checked by their text units, each of
        which belongs to one document. Seeds, relationships and reports
        match each document key independently (see the user guide). The
        seeds are checked for the entity keys: a seed need not have been
        found under them (LightRAG seeds the walk with the endpoints of the
        relationships it found, which hold no entity key).
        """
        o = self.config.indexing.opensearch
        others = [r for r in results if r.source not in seeds]
        passed = await self._ids_passing(
            query, o.entities_index_prefix, (str(r.source) for r in others), filters
        )
        if len(documents := document_filters(filters)) > 1:
            passed = await self._one_text_unit_passes(
                query,
                {
                    str(r.source): _text_unit_ids(r)
                    for r in others
                    if str(r.source) in passed
                },
                documents,
            )
        if own := {k: v for k, v in filters.items() if k not in documents}:
            passed |= await self._ids_passing(
                query,
                o.entities_index_prefix,
                (str(r.source) for r in results if r.source in seeds),
                own,
            )
            kept = [r for r in results if str(r.source) in passed]
        else:
            kept = [r for r in results if r.source in seeds or str(r.source) in passed]
        if len(kept) < len(results):
            logger.debug(
                "Filters the graph cannot apply dropped %s of %s expanded entities",
                len(results) - len(kept),
                len(results),
            )
        return kept

    async def _one_text_unit_passes(
        self, query: SearchQuery, units: dict[str, list[str]], filters: dict[str, Any]
    ) -> set[str]:
        """The keys of ``units`` (an item's text unit ids) with a text unit
        that passes ``filters``.

        An entity or edge document unions its documents' values per key, so
        under two or more keys it can pass while no document passes them all;
        a text unit is one document's. All their units are checked at once.
        """
        if not units:
            return set()
        passed = await self._ids_passing(
            query,
            self.config.indexing.opensearch.text_units_index_prefix,
            (u for ids in units.values() for u in ids),
            filters,
        )
        return {key for key, ids in units.items() if passed.intersection(ids)}

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
        filter merged into the caller's filters). Under a document filter the
        graph cannot tell which edges a passing document states, so the walk
        runs over the relationships index instead
        (``_expand_over_passing_relationships``); without that index the
        graph walk stands, its entities checked against the entities index,
        as they are under an entity key the graph does not store
        (``name.keyword``, ``attributes.<path>``).
        """
        graph = self.graph_retriever
        if graph is None or not seed_entity_ids:
            return []
        seeds = {str(s) for s in seed_entity_ids}
        filtered = bool(document_filters(query.filters))
        if (
            filtered
            and self.document_retriever is not None
            and self.config.indexing.opensearch.build_relationship_vector_index
        ):
            walked = await self._expand_over_passing_relationships(
                query, list(dict.fromkeys(str(s) for s in seed_entity_ids))
            )
            if walked is not None:
                return walked
        search_query = query.model_copy(deep=True)
        search_query.filters = (search_query.filters or {}).copy()
        # The graph reads ``id`` as the seeds to walk from: a caller's ``id``
        # (whose entities the seeds already are) applies to what it reached.
        caller_ids = search_query.filters.pop("id", None)
        search_query.filters["id"] = seed_entity_ids
        expanded = await self._safe_aretrieve(graph, search_query, "graph expansion")
        if caller_ids is not None:
            passing = scope_filters({"id": caller_ids}, id=[r.source for r in expanded])
            kept = set((passing or {})["id"])
            expanded = [r for r in expanded if r.source in kept]
        if unapplied := self._graph_unapplied_filters(query):
            expanded = await self._vouch_by_entities_index(
                query, expanded, seeds, unapplied
            )
        return expanded

    async def _expand_over_passing_relationships(
        self, query: SearchQuery, seeds: list[str]
    ) -> list[RetrievalResult] | None:
        """Graph expansion under a document filter: a breadth-first walk from
        the seeds over the relationships the filter passes.

        The graph walk crossed every edge, and its per-vertex neighbour cap
        and cut ran before any check, so on a hub it spent its budget on
        what only excluded documents state (a 1,522-degree hub kept 0-4
        entities where an index of the passing documents alone keeps 10).
        Each hop instead asks the relationships index for the passing edges
        incident to the frontier (``_incident_relationships``: best first by
        rank, weight and id); under two or more keys an edge also needs one
        passing document, by its text units. The walk keeps the graph walk's
        bounds: ``max_hops``, ``max_results_per_hop`` new neighbours per
        vertex a hop and ``top_k * retrieval_multiplier *
        traversal_fetch_multiplier`` vertices split across the seeds: each
        seed reaches at most its share (rounded up, so the shares and no
        global cap bound the total), so a hub seed cannot use up the others'.
        A hop asks only the vertices whose seed is still short of its share.
        Edges back to a vertex already reached are left out by the store
        (``exclude`` on the other endpoint), so they cannot fill a page. A
        page that comes back full holds the busiest vertices' best edges
        only: the vertices still open are then asked together once more at
        the full end, and if that page is full too, each on its own, in
        waves of ``search.max_concurrency``, for what its wave may still
        take (two requests a hop; one more when a page is full; one per
        vertex still open when the next is full too). The reached
        entities are read from the entities index under the caller's
        filters (a vertex filter such as ``type`` keeps only its entities,
        the walk goes through the others), scored by distance and importance
        as the graph walk does (``degree`` reads as ``rank``: the index holds
        no degree), and cut to ``top_k * retrieval_multiplier``; ``Path:``
        follows the walk.

        ``None`` when the relationships index is missing at query time: the
        caller falls back to the graph walk.
        """
        retriever = self.document_retriever
        assert retriever is not None  # the caller walks the graph without one
        o = self.config.indexing.opensearch
        neptune = self.config.indexing.neptune
        filters = document_filters(query.filters)
        limit = query.top_k * query.retrieval_multiplier
        share = seed_share(query, neptune.traversal_fetch_multiplier, len(seeds))
        # Text units only decide an edge under two or more keys.
        fields = [
            f for f in _WALK_EDGE_FIELDS if len(filters) > 1 or f != "text_unit_ids"
        ]
        parent: dict[str, str | None] = dict.fromkeys(seeds)
        root = {seed: seed for seed in seeds}
        taken = dict.fromkeys(seeds, 0)
        names: dict[str, str] = {}
        frontier = list(seeds)
        per_vertex: dict[str, int] = {}  # new neighbours per vertex this hop

        def needed(vertex: str) -> int:
            """How many more neighbours ``vertex`` may take this hop."""
            return min(
                neptune.max_results_per_hop - per_vertex.get(vertex, 0),
                share - taken[root[vertex]],
            )

        def open_(vertex: str) -> bool:
            return needed(vertex) > 0

        def take(edges: list[RetrievalResult], asked: set[str]) -> None:
            """Walk ``edges`` from the ``asked`` vertices."""
            for edge in edges:
                metadata = edge.metadata or {}
                ends = (
                    (str(metadata.get("source_id")), metadata.get("source_name")),
                    (str(metadata.get("target_id")), metadata.get("target_name")),
                )
                for (vertex, _), (other, _) in (ends, ends[::-1]):
                    if vertex not in asked or other in parent or not open_(vertex):
                        continue
                    parent[other], root[other] = vertex, root[vertex]
                    taken[root[vertex]] += 1
                    per_vertex[vertex] = per_vertex.get(vertex, 0) + 1
                    reached.append(other)
                for vertex, name in ends:
                    if name:
                        names.setdefault(vertex, str(name))

        async def passing(
            vertices: list[str],
            size: int | None = None,
            ends: tuple[str, ...] = _ENDPOINTS,
        ) -> tuple[list[RetrievalResult], tuple[str, ...], Counter[tuple[str, str]]]:
            """The passing edges incident to ``vertices`` at the ``ends``
            named, the ends whose page came back full and how many edges of
            each (end, vertex) the check under two or more keys dropped
            (they come back on the vertex's next page)."""
            # An edge back to a vertex already reached adds nothing: left out
            # by the store, so such edges cannot fill a page.
            edges, full = await self._incident_relationships(
                query, vertices, fields, size, ends, away_from=list(parent)
            )
            dropped: Counter[tuple[str, str]] = Counter()
            if len(filters) > 1 and edges:
                kept = await self._edges_one_document_passes(query, edges, filters)
                ids = {self._relationship_identity(e) for e in kept}
                dropped.update(
                    (end, str((e.metadata or {}).get(end)))
                    for e in edges
                    if self._relationship_identity(e) not in ids
                    for end in ends
                )
                edges = kept
            return edges, full, dropped

        for hop in range(neptune.max_hops):
            per_vertex.clear()
            # Only the vertices whose seed is still short of its share.
            frontier = [vertex for vertex in frontier if open_(vertex)]
            if not frontier:
                break
            reached: list[str] = []
            edges, full, dropped = await passing(frontier)
            if hop == 0 and not edges:
                exists = await retriever.ahas_index(
                    o.relationships_index_prefix, query.suffix
                )
                if exists is False:
                    logger.warning(
                        "%s: no relationships index for suffix '%s'; graph "
                        "expansion under document filters walks the graph",
                        type(self).__name__,
                        query.suffix,
                    )
                    return None
            take(edges, set(frontier))
            still_open = [vertex for vertex in frontier if open_(vertex)]
            if full and still_open:
                # A full page held the busiest vertices' best edges only. The
                # vertices still open are asked again together at the full
                # ends: with the reached vertices left out, that page holds
                # new edges only.
                edges, full, more = await passing(still_open, None, full)
                dropped |= more  # the same edges come back: the larger count
                take(edges, set(still_open))
                still_open = [vertex for vertex in still_open if open_(vertex)]
            if full and still_open:
                # Full again: each vertex still open is asked on its own, in
                # waves of search.max_concurrency. Asked at once, the requests
                # share their neighbours and the first to take one keeps it,
                # so each asks for what its wave may still take, plus its
                # edges the check dropped (they come back). A wave leaves out
                # what the waves before it reached, so it asks for less than
                # all the open seeds' room (which a request on its own got).
                step = self.config.search.max_concurrency
                for start in range(0, len(still_open), step):
                    wave = [v for v in still_open[start : start + step] if open_(v)]
                    room = sum(needed(v) for v in wave)
                    pages = await gather_or_cancel(
                        passing(
                            [v],
                            # At most a page (_incident_relationships).
                            max(dropped[e, v] for e in full) + room,
                            full,
                        )
                        for v in wave
                    )
                    for vertex, (edges, _, _) in zip(wave, pages, strict=True):
                        take(edges, {vertex})
            frontier = reached
        # The seeds first, then by distance: the order the walk reached them.
        reached_ids = list(parent)
        entities = await self._fetch_by_ids(
            retriever,
            SearchQuery(
                query="",
                search_type=SearchType.LEXICAL,
                top_k=len(reached_ids),
                retrieval_multiplier=1,
                index_prefixes=[o.entities_index_prefix],
                suffix=query.suffix,
                filters=self._scoped_filters(query, id=reached_ids),
            ),
            "Filtered graph expansion entities",
        )
        seed_set = set(seeds)
        if len(filters) > 1:
            # The read applied each key on its own (an entity unions its
            # documents' values, as an edge does): one document must pass all.
            vouched = await self._one_text_unit_passes(
                query,
                {
                    str(e.source): _text_unit_ids(e)
                    for e in entities
                    if str(e.source) not in seed_set
                },
                filters,
            )
            entities = [e for e in entities if str(e.source) in seed_set | vouched]
        with_importance = neptune.entity_importance_source != "none"

        def importance(result: RetrievalResult) -> float | None:
            return (
                as_float((result.metadata or {}).get("rank"))
                if with_importance
                else None
            )

        top = max((importance(r) or 0.0 for r in entities), default=0.0)
        results: list[RetrievalResult] = []
        for entity in entities:
            vertex = str(entity.source)
            if vertex not in parent:
                continue
            chain: list[str] = []
            current: str | None = vertex
            while current is not None:
                chain.append(current)
                current = parent[current]
            metadata = entity.metadata or {}
            path = [
                (
                    str(metadata.get("name") or names.get(v, v))
                    if v == vertex
                    else names.get(v, v)
                )
                for v in reversed(chain)
            ]
            raw = importance(entity)
            results.append(
                entity.model_copy(
                    update={
                        "content": entity_content(metadata, path),
                        "score": graph_relevance(
                            len(chain),
                            raw / top if raw is not None and top > 0 else 0.5,
                        ),
                        "retriever_type": SectionType.ENTITY.value,
                    }
                )
            )
        return rank_expanded(results, seed_set, limit)

    async def _edges_one_document_passes(
        self, query: SearchQuery, edges: list[RetrievalResult], filters: dict[str, Any]
    ) -> list[RetrievalResult]:
        """Of the ``edges`` the relationships index passed, those one document
        passes under every key: the index unions each key's values over an
        edge's documents, so under two or more keys an edge of a
        ``{category: x, grp: a}`` and a ``{category: y, grp: b}`` document
        passes ``category:x grp:b``. Checked by the edges' text units (one
        document each), as the entity check is."""
        vouched = await self._one_text_unit_passes(
            query,
            {self._relationship_identity(e): _text_unit_ids(e) for e in edges},
            filters,
        )
        return [e for e in edges if self._relationship_identity(e) in vouched]

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
        """Every relationship incident to ``entity_ids``, deduped and ordered
        (``_incident_relationships``) without those the caller already holds
        (``known_relationships``, by relationship id), so an edge is not paid
        for twice. With ``bridge_first`` the edges whose BOTH endpoints are in
        ``entity_ids`` — the bridges that connect two retrieved entities,
        i.e. the hops of a multi-hop chain — come before edges that leave the
        set.
        """
        edges, _ = await self._incident_relationships(query, entity_ids)
        known = {self._relationship_identity(r) for r in known_relationships or []}
        edges = [e for e in edges if self._relationship_identity(e) not in known]
        if bridge_first:
            in_set = set(entity_ids)
            # Stable: each group keeps the (rank, weight, id) order.
            edges.sort(
                key=lambda r: not (
                    str((r.metadata or {}).get("source_id")) in in_set
                    and str((r.metadata or {}).get("target_id")) in in_set
                )
            )
        return edges

    async def _incident_relationships(
        self,
        query: SearchQuery,
        entity_ids: list[str],
        source_fields: list[str] | None = None,
        size: int | None = None,
        ends: tuple[str, ...] = _ENDPOINTS,
        away_from: list[str] | None = None,
    ) -> tuple[list[RetrievalResult], tuple[str, ...]]:
        """The relationships incident to ``entity_ids`` (only
        ``source_fields`` of each when given) at the ``ends`` named, at most
        ``size`` per end (``_incident_fetch_limit`` by default; never past
        the index's live ``max_result_window``), without those
        whose other endpoint is in ``away_from``, and the ends whose page came
        back full (the edges past it were dropped).

        The relationships index stores its endpoints as `source_id`/`target_id`
        keywords, but `_build_filter_clauses` ANDs every filter, so
        `source_id OR target_id` needs two requests. Edges are deduped by
        relationship id and ordered by `(rank, weight)` descending, as
        upstream LightRAG's `_find_most_related_edges_from_entities` does,
        then by id, so equal edges do not follow the order of the two
        requests. Within a query (``query_scope``) the same edges are asked
        once (DRIFT's follow-ups expand the same seeds); a concurrent caller
        waits for the request in flight. A failed request is not kept.
        """
        retriever = self.document_retriever
        if not retriever or not entity_ids:
            return [], ()
        prefix = self.config.indexing.opensearch.relationships_index_prefix
        step = (await retriever.aindex_limits(prefix, query.suffix)).max_terms_count
        # One terms clause per list: past max_terms_count the store refuses
        # the query, so the vertices are asked in batches, each with its own
        # page. The exclusion only saves page room: past max_terms_count it
        # keeps the vertices reached last (it is in reach order), so edges
        # back to the others can take a page.
        batches = [entity_ids[i : i + step] for i in range(0, len(entity_ids), step)]
        away = list(away_from or ())[-step:]
        memo = current_query_scope()
        # What the requests send: the batches' and the kept exclusion's
        # values (the order matters only past max_terms_count, by what it
        # puts in which batch or keeps). Hashed: the vertices left out alone
        # run to megabytes a walk.
        key = f"{_MEMO_INCIDENT}\n" + compute_hash(
            json.dumps(
                [
                    query.suffix,
                    query.filters,
                    sorted(sorted(set(batch)) for batch in batches),
                    source_fields,
                    size,
                    ends,
                    sorted(set(away)),
                ],
                sort_keys=True,
                default=str,
            ),
            length=64,
        )
        if memo is not None:
            if (pending := memo.get(key)) is not None:
                shared = await asyncio.shield(pending)
                if shared is not None:
                    return list(shared[0]), shared[1]
            else:
                outcome: asyncio.Future[Any] = (
                    asyncio.get_running_loop().create_future()
                )
                memo[key] = outcome
                try:
                    edges, full, failed = await self._ask_incident(
                        query, batches, source_fields, size, ends, away
                    )
                except BaseException:
                    del memo[key]
                    outcome.set_result(None)  # the waiters ask for themselves
                    raise
                if failed:
                    del memo[key]
                    outcome.set_result(None)
                else:
                    outcome.set_result((edges, full))
                return list(edges), full
        edges, full, _ = await self._ask_incident(
            query, batches, source_fields, size, ends, away
        )
        return edges, full

    async def _ask_incident(
        self,
        query: SearchQuery,
        batches: list[list[str]],
        source_fields: list[str] | None,
        size: int | None = None,
        ends: tuple[str, ...] = _ENDPOINTS,
        away_from: list[str] | None = None,
    ) -> tuple[list[RetrievalResult], tuple[str, ...], bool]:
        """``_incident_relationships``' requests (one per end and batch of
        vertices, at most ``search.max_concurrency`` at once): the edges,
        the ends whose page came back full and whether a request failed."""
        retriever = self.document_retriever
        assert retriever is not None
        prefix = self.config.indexing.opensearch.relationships_index_prefix
        limits = await retriever.aindex_limits(prefix, query.suffix)
        # A page is judged full against the hits a search returns: the live
        # window may be below max_query_size.
        cap = min(self._incident_fetch_limit(), limits.max_result_window)
        page = min(size or cap, cap)
        failures: list[Exception] = []

        async def by_batch(field: str, vertices: list[str]) -> list[RetrievalResult]:
            return await self._safe_aretrieve(
                retriever,
                SearchQuery(
                    query="",
                    search_type=SearchType.LEXICAL,
                    top_k=page,
                    index_prefixes=[prefix],
                    suffix=query.suffix,
                    filters=self._scoped_filters(query, **{field: vertices}),
                    exclude=({_OTHER_END[field]: away_from} if away_from else None),
                    source_fields=source_fields,
                ),
                f"Incident relationship retrieval ({field})",
                failures,
            )

        pages = await gather_bounded(
            self.config.search.max_concurrency,
            (by_batch(end, batch) for end in ends for batch in batches),
        )
        sides = [
            pages[i : i + len(batches)] for i in range(0, len(pages), len(batches))
        ]
        # A store that rejected the window was asked again with the live one
        # and returned that many hits: a page that size is full too.
        window = (await retriever.aindex_limits(prefix, query.suffix)).max_result_window
        cap, page = min(cap, window), min(page, window)
        # A text-less search is ordered by (rank, weight) server-side (the
        # retriever's importance sort), so a full page holds the best edges.
        # Still, the count cap bound and edges were dropped — upstream drops
        # none. Say so (once a query: DRIFT asks again and again) rather than
        # let a silent truncation read as "all edges".
        full = tuple(
            end
            for end, side in zip(ends, sides, strict=True)
            if any(len(hits) >= page for hits in side)
        )
        if full and page >= cap:  # not a sized page
            memo = current_query_scope()
            warned = memo is not None and _MEMO_PAGE_CAP in memo
            (logger.debug if warned else logger.warning)(
                "Incident-edge fetch hit the %s-hit page cap; the lowest-ranked "
                "edges are dropped (upstream truncates by tokens only)",
                page,
            )
            if memo is not None:
                memo[_MEMO_PAGE_CAP] = True

        # Dedup by relationship id, so an edge reachable from both of its
        # endpoints is carried once. Upstream dedups by the undirected endpoint
        # pair (`tuple(sorted(e))`) because its graph holds one edge per pair;
        # ours keeps typed, directed edges, and a pair key dropped every edge
        # between two entities but the first ("supplies" and "invoices" alike).
        deduped = list(
            {
                self._relationship_identity(r): r
                for side in sides
                for hits in side
                for r in hits
            }.values()
        )

        def order(r: RetrievalResult) -> tuple[float, float, str]:
            metadata = r.metadata or {}
            return (
                -(as_float(metadata.get("rank")) or 0.0),
                -(as_float(metadata.get("weight")) or 0.0),
                self._relationship_identity(r),
            )

        # Upstream orders these by (rank, weight) descending — degree first, so the
        # hub edges that carry a multi-hop chain outrank incidental leaf edges.
        deduped.sort(key=order)
        return deduped, full, bool(failures)

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
        only for async code.
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
        silently dropped on that path. Returns a new dict (never the
        caller's) or ``None`` when there is nothing to apply; see
        ``scope_filters``. Indexes have different fields, so the retrievers
        drop each caller key from the sub-queries whose index (or vertex
        label) lacks that field.
        """
        return scope_filters(query.filters, **scope)
