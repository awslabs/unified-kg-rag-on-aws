# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph expansion from seed entities, independent of the graph query language.

``GraphExpansionRetriever`` holds the ranking, de-duplication, cut and
content-building logic every graph backend shares. A backend subclass only
implements ``_traverse``, which walks the graph from the seeds and returns one
row per reached vertex in the projected shape ``NeptuneRetriever`` produces
with Gremlin::

    {
        "node": {"id": [...], "name": [...], "type": [...],
                 "description": [...],
                 "rank": [...], "text_unit_ids": [...],
                 "community_ids": [...], "iri": [...],
                 "source_kinds": [...]},  # value_map; the last three
                                          # if held
        "path": [{"name": [...]}, ...],  # seed first, the node last
        "node_type": "<vertex label>",
        "degree": <int>,  # only when entity_importance_source == "degree"
        "superseded": True,  # optional: an older copy of a vertex whose
                             # newer copy came back too
    }

The seeds themselves come back as rows with a path of length 1.
"""

import asyncio
import logging
import time
from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    entity_content,
    graph_relevance,
    rank_expanded,
)
from unified_kg_rag.adapters.storage.neptune_codec import decode_edge_list
from unified_kg_rag.adapters.store_errors import is_fatal_retrieval_error
from unified_kg_rag.domain.ingestion.seed_graph import SOURCE_KINDS_FIELD
from unified_kg_rag.domain.models import (
    Config,
    GraphEdge,
    RetrievalResult,
    SearchQuery,
    SectionType,
)
from unified_kg_rag.domain.models.config import GraphExpansionConfig
from unified_kg_rag.domain.retrieval import source_fields
from unified_kg_rag.domain.retrieval.filters import (
    ATTRIBUTE_KEY_PREFIX,
    GRAPH_ENTITY_FILTER_FIELDS,
    FieldKind,
    FilterFields,
    KindsRule,
    Range,
    Term,
    check_filter_values,
    kinds_rule,
    loose_values,
    own_attribute,
    parse_filters,
    typed_filter_values,
)
from unified_kg_rag.domain.retrieval.ppr import cap_edges_per_vertex
from unified_kg_rag.shared import InvalidFilterError, StoreConnectionError, get_logger
from unified_kg_rag.shared.logging import log_error
from unified_kg_rag.shared.utils import safe_float_parse, text_digest

logger = get_logger(__name__)

# Vertex list properties (source_fields.LIST_FIELDS) stay lists however many
# values they hold, where a value_map's one-element list is otherwise read as
# its value.
_LIST_PROPERTIES = source_fields.LIST_FIELDS


VertexClause = Term | Range | KindsRule


def vertex_clauses(filters: dict[str, Any] | None) -> tuple[VertexClause, ...]:
    """Filters scoped to entity vertices
    (``GraphExpansionRetriever._scope_filters_to_label``) as clauses on the
    vertex properties both graph stores hold (``neptune_codec``), which
    each backend compiles (Gremlin steps, the SQLite graph's predicate).

    ``id`` is the seed list, not a filter. A name is the ``name``
    property under either spelling; an entity's own attribute
    (``attributes.<key>``) its ``attr_<key>`` property, matched under either
    spelling of a number or boolean (``loose_values``) as the entities
    index, which maps it by its first value, converts a term.

    Raises:
        InvalidFilterError: A key no vertex holds (a document filter, a
            nested attribute path, analyzed text), which the scoping drops
            before, or a malformed value.
    """
    clauses: list[VertexClause] = []
    for clause in parse_filters(
        {k: v for k, v in (filters or {}).items() if k != "id"},
        allow_empty_lists=True,
    ):
        if not GRAPH_ENTITY_FILTER_FIELDS.declares(clause.field):
            raise InvalidFilterError(
                f"Filter '{clause.field}' names nothing a graph vertex holds."
            )
        if clause.kind is FieldKind.NAME:
            field = "name"
        elif clause.kind is FieldKind.ATTRIBUTE:
            field = f"{ATTRIBUTE_KEY_PREFIX}{own_attribute(clause.field)}"
        else:
            clauses.append(clause)
            continue
        if isinstance(clause, Term) and clause.kind is FieldKind.ATTRIBUTE:
            values = loose_values(clause.values)
            listed = clause.listed or len(values) > 1
            clauses.append(Term(field, values, clause.kind, listed))
        elif isinstance(clause, Term):
            clauses.append(Term(field, clause.values, clause.kind, clause.listed))
        else:
            assert isinstance(clause, Range)
            clauses.append(Range(field, clause.bounds, clause.kind))
    return tuple(clauses)


def edge_source_kinds(stored: Any) -> list[str] | None:
    """An edge's stored ``source_kinds`` (one JSON list string, the graph
    codec's encoding) as a list; ``None`` when absent or unreadable."""
    return decode_edge_list(stored)


def edge_predicate(
    filters: dict[str, Any] | None,
) -> Callable[[Mapping[str, Any]], bool] | None:
    """The caller's ``source_kinds`` rule (``kinds_rule``) as a predicate on
    one edge's stored properties (``source_kinds`` a JSON list string);
    ``None`` without one. The structured queries of the embedded graph and
    the reference executor's fake apply it to every edge of a walk."""
    rule = kinds_rule(filters)
    if rule is None:
        return None
    return lambda props: rule.passes(edge_source_kinds(props.get(SOURCE_KINDS_FIELD)))


class GraphExpansionRetriever(BaseGraphRAGRetriever):
    """Graph expansion from seed entities given as an ``id`` filter.

    Every search strategy finds its seed entities in the vector store and
    passes their ids (``filters["id"]``); a query without one returns no
    results. ``indexing.neptune`` holds the graph's labels and
    ``search.graph_expansion`` the walk's settings (hops, fetch budget,
    importance source) for every graph backend.
    """

    # Name used in log lines.
    _store_name: ClassVar[str] = "Graph"

    def __init__(self, config: Config, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._neptune_config = config.indexing.neptune
        self._max_hops = self._expansion.max_hops
        self._max_results_per_hop = self._expansion.max_results_per_hop

    @property
    def _expansion(self) -> GraphExpansionConfig:
        """The walk's settings (``search.graph_expansion``)."""
        return self._config.search.graph_expansion

    @abstractmethod
    async def _traverse(
        self, seed_ids: list[Any], query: SearchQuery
    ) -> list[dict[str, Any]]:
        """Rows for the seeds and the vertices reached from them (module doc)."""

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        start_time = time.time()
        logger.info(
            "%s retrieval started - query: %s ('%s')",
            self._store_name,
            text_digest(query.query),
            query.search_type.value,
        )
        # The filters a strategy sends are checked as the OpenSearch retriever
        # checks them: a range operator the walk has no step for (``in``,
        # ``eq``) was dropped, so the filter matched every vertex. A
        # strategy's own scope may allow nothing (an empty list).
        check_filter_values(query.filters, allow_empty_lists=True)
        self._refuse_any_of(query)

        try:
            seed_ids = self._seed_ids(query)
            if not seed_ids:
                logger.warning(
                    "%s retrieval needs seed entity ids (an 'id' filter); "
                    "none given for query: %s",
                    self._store_name,
                    text_digest(query.query),
                )
                return []
            traversal_results = await self._traverse(seed_ids, query)
            # Pure over the rows; a hub's thousands of lineage ids are
            # validated as its result is made, so off the event loop.
            results = await asyncio.to_thread(
                self._process_traversal_results, traversal_results, query
            )

            processing_time = time.time() - start_time
            logger.info(
                "%s retrieval completed - retrieved: %s results (%.2fs)",
                self._store_name,
                len(results),
                processing_time,
            )

            return results

        except Exception as e:
            error = self._request_error(e)
            if not is_fatal_retrieval_error(error):
                log_error(
                    logger,
                    logging.ERROR,
                    "%s retrieval failed (transient, degrading to empty "
                    "results): %s",
                    self._store_name,
                    error=e,
                )
                return []
            if isinstance(error, StoreConnectionError):
                # The traversal has dropped the connection already.
                log_error(
                    logger,
                    logging.ERROR,
                    "%s retrieval failed: %s",
                    self._store_name,
                    error=error,
                )
            else:
                # A broken configuration is not "no results".
                log_error(
                    logger,
                    logging.ERROR,
                    "%s retrieval failed (fatal): %s",
                    self._store_name,
                    error=e,
                )
            if error is e:
                raise
            raise error from e

    def _request_error(self, exc: Exception) -> Exception:
        """``exc`` typed by the backend's error policy (identity by default)."""
        return exc

    def _refuse_any_of(self, query: SearchQuery) -> None:
        """Refuse ``SearchQuery.any_of``: a walk filters its vertices by
        every key, and ignoring the groups would widen it."""
        if query.any_of:
            raise InvalidFilterError(
                f"{self._store_name} graph expansion applies no any_of filter "
                "groups."
            )

    async def aneighbourhood_edges(
        self, vertex_ids: list[str], query: SearchQuery, *, per_vertex: int
    ) -> list[GraphEdge] | None:
        """The relationships between two of ``vertex_ids`` (the base's
        contract), read by the backend (``_edges_within``) under
        ``query.suffix``'s entity label. Of the caller's filters only
        ``source_kinds`` applies to an edge (the vertices passed the rest
        already): an edge stated only by excluded kinds takes no slot. The
        error policy is :meth:`aretrieve`'s: a request the store failed
        degrades to no edges, a fatal one raises."""
        check_filter_values(query.filters, allow_empty_lists=True)
        self._refuse_any_of(query)
        ids = list(dict.fromkeys(str(v) for v in vertex_ids if v))
        if len(ids) < 2 or per_vertex < 1:
            return []
        label = self._get_label(self._neptune_config.entity_label_prefix, query.suffix)
        try:
            edges = await self._edges_within(
                ids, label, per_vertex, kinds_rule(query.filters)
            )
        except Exception as e:
            error = self._request_error(e)
            if not is_fatal_retrieval_error(error):
                log_error(
                    logger,
                    logging.ERROR,
                    "%s neighbourhood edges failed (transient, degrading to "
                    "none): %s",
                    self._store_name,
                    error=e,
                )
                return []
            if error is e:
                raise
            raise error from e
        if edges is None:
            return None
        return cap_edges_per_vertex(edges, ids, per_vertex)

    async def _edges_within(
        self,
        ids: list[str],
        label: str,
        per_vertex: int,
        kinds: KindsRule | None,
    ) -> list[GraphEdge] | None:
        """The edges between two of ``ids`` under ``label`` that pass the
        caller's ``source_kinds`` rule ``kinds``, each vertex's first
        ``per_vertex`` by relationship id. ``None`` (the default): the
        backend cannot tell."""
        return None

    @staticmethod
    def _seed_ids(query: SearchQuery) -> list[Any]:
        """The seed ids, each once (a repeated custom vertex id is a vertex
        looked up, and expanded, once per repeat)."""
        seed_ids = (query.filters or {}).get("id")
        return list(dict.fromkeys(seed_ids)) if isinstance(seed_ids, list) else []

    def filter_fields(self) -> FilterFields:
        # The traversal filters entity vertices only (_scope_filters_to_label).
        return GRAPH_ENTITY_FILTER_FIELDS

    @staticmethod
    def _scope_filters_to_label(
        label: str, filters: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """``filters`` without the keys entity vertices (``label``) do not
        declare, typed.

        Mirrors the OpenSearch per-index scoping: a filter on ``key`` drops
        every vertex lacking ``key``, so a key the label does not declare
        (``GRAPH_ENTITY_FILTER_FIELDS``) is dropped rather than emptying the
        traversal (the strategy checks the reached entities against the
        entities index for such a key,
        ``BaseSearchStrategy._graph_unapplied_filters``). A string value of
        a boolean or numeric field is converted to that type
        (``typed_filter_values``): a vertex property compares as stored, so
        ``"false"`` matched no ``in_schema``, where OpenSearch converts it.
        The backend compiles what is kept (``vertex_clauses``).
        """
        if not filters:
            return filters
        kept: dict[str, Any] = {}
        for key, value in (typed_filter_values(filters) or {}).items():
            if not GRAPH_ENTITY_FILTER_FIELDS.declares(key):
                logger.debug(
                    "Dropped filter key '%s': no %s vertex declares it", key, label
                )
                continue
            kept[key] = value
        return kept or None

    @staticmethod
    def _clean_property_map(prop_map: dict[str, Any]) -> dict[str, Any]:
        return {
            k: (
                v[0]
                if isinstance(v, list) and len(v) == 1 and k not in _LIST_PROPERTIES
                else v
            )
            for k, v in prop_map.items()
        }

    def _process_traversal_results(
        self, traversal_results: list[dict[str, Any]], query: SearchQuery
    ) -> list[RetrievalResult]:
        importance_key = self._entity_importance_key()
        max_importance = 0.0
        if importance_key is not None:
            max_importance = max(
                (
                    safe_float_parse(self._node_data(item).get(importance_key)) or 0.0
                    for item in traversal_results
                ),
                default=0.0,
            )

        # Entity expansion runs per seed, so a node two seeds reach arrives once
        # per seed; keep the copy with the shortest path, of equally short
        # paths the best-scored one, then the one from the better-ranked seed
        # (the caller's seed order), then the first by content, not the first
        # to arrive. A node stored twice (a Neptune label being switched)
        # arrives once per copy: the nearer, the newer of two as near (even
        # where the old copy's degree or rank scores it higher).
        seed_rank = self._seed_ranks(traversal_results, query)
        best: dict[Any, tuple[tuple[int, bool, float, int, str], RetrievalResult]] = {}
        seed_ids: set[Any] = set()
        for item in traversal_results:
            node_data = self._node_data(item)
            node_id = node_data.get("id")
            if not node_id:
                continue

            result = self._create_retrieval_result(
                item,
                node_data,
                importance_key=importance_key,
                max_importance=max_importance,
            )
            if len(item.get("path") or []) <= 1:
                # A path of one vertex: a seed, not something reached from one.
                seed_ids.add(node_id)
            key = (
                len(item.get("path") or []) or 1,
                bool(item.get("superseded")),
                -(result.score or 0.0),
                seed_rank(item),
                result.content,
            )
            kept = best.get(node_id)
            if kept is None or key < kept[0]:
                best[node_id] = (key, result)

        return rank_expanded(
            (result for _, result in best.values()),
            {str(s) for s in seed_ids},
            (
                query.top_k * query.retrieval_multiplier
                if self._expansion.traversal_fetch_multiplier > 1
                else None
            ),
        )

    def _seed_ranks(
        self, traversal_results: list[dict[str, Any]], query: SearchQuery
    ) -> Callable[[dict[str, Any]], int]:
        """The rank of the seed a row's path starts from: its position in the
        caller's seed ids, the paths naming it by name (of seeds sharing a
        name, the best rank); after every seed when not among the rows."""
        order = {str(s): i for i, s in enumerate(self._seed_ids(query))}
        by_name: dict[str, int] = {}
        for item in traversal_results:
            if len(item.get("path") or []) > 1:
                continue
            node_data = self._node_data(item)
            rank = order.get(str(node_data.get("id")), len(order))
            name = str(node_data.get("name", ""))
            by_name[name] = min(rank, by_name.get(name, rank))

        def seed_rank(item: dict[str, Any]) -> int:
            path = item.get("path") or [{}]
            first = path[0] if isinstance(path[0], dict) else {}
            names = first.get("name") or [""]
            return by_name.get(str(names[0]), len(order))

        return seed_rank

    def _entity_importance_key(self) -> str | None:
        """The node property graph expansion reads as entity importance."""
        source = self._expansion.entity_importance_source
        return None if source == "none" else source

    def _node_data(self, item: dict[str, Any]) -> dict[str, Any]:
        """The cleaned property map, with the projected degree folded in."""
        node_data = self._clean_property_map(item.get("node", {}))
        if "degree" in item:
            node_data["degree"] = item["degree"]
        return node_data

    def _create_retrieval_result(
        self,
        item: dict[str, Any],
        node_data: dict[str, Any],
        importance_key: str | None = None,
        max_importance: float = 0.0,
    ) -> RetrievalResult:
        path_data = item.get("path", [])
        return RetrievalResult(
            content=self._build_content(node_data, path_data),
            score=self._calculate_relevance(
                node_data,
                path_data,
                importance_key=importance_key,
                max_importance=max_importance,
            ),
            source=str(node_data.get("id")),
            retriever_type=SectionType.ENTITY,
            metadata={
                **node_data,
                "_node_type": item.get("node_type", "unknown"),
            },
        )

    @staticmethod
    def _build_content(
        node_data: dict[str, Any], path_data: list[dict[str, Any]]
    ) -> str:
        return entity_content(
            node_data,
            [p.get("name", [""])[0] for p in path_data if isinstance(p, dict)],
        )

    @staticmethod
    def _calculate_relevance(
        node: dict[str, Any],
        path: list[dict[str, Any]],
        importance_key: str | None = None,
        max_importance: float = 0.0,
    ) -> float:
        """Relevance in (0, 1]: distance to the seeds first, importance second.

        A node ``n`` hops from the seeds (path length ``L = n + 1``) scores in
        the band ``(1 / (L + 1), 1 / L]`` -- a seed in ``[0.75, 1]``, one hop
        out in ``[0.42, 0.5]`` -- and its importance places it inside the band.
        So every nearer node outranks every farther one and importance only
        orders nodes at the same distance: a one-hop hub can never displace a
        seed the query matched. (Averaging the two terms, as before, let a
        high-degree neighbour outrank a low-degree seed.) An entity's
        importance is its ``importance_key`` value (indexed ``rank`` or
        projected ``degree``) relative to the largest one in the same result,
        neutral 0.5 when absent or without an ``importance_key``
        (``entity_importance_source: none``). (A query-text term was dropped: graph
        expansion passes the whole question, which never occurs inside a node
        name.)
        """
        raw = (
            safe_float_parse(node.get(importance_key))
            if importance_key is not None
            else None
        )
        return graph_relevance(
            len(path) or 1,
            raw / max_importance if raw is not None and max_importance > 0 else 0.5,
        )
