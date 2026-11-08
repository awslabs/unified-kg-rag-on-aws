# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph expansion over the embedded SQLite graph store.

The walk and the filters reproduce ``NeptuneRetriever``'s Gremlin traversal
(see ``SQLiteGraphStore.traverse``); ranking, de-duplication and the cut come
from ``GraphExpansionRetriever``, so both backends rank the same graph alike.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, ClassVar

from unified_kg_rag.adapters.graph_query.sqlite import SQLiteQuery
from unified_kg_rag.adapters.local.sqlite_graph import SQLiteGraphStore, VertexFilter
from unified_kg_rag.adapters.retrieval.base import seed_shares
from unified_kg_rag.adapters.retrievers.graph_expansion import (
    GraphExpansionRetriever,
    VertexClause,
    edge_predicate,
    edge_source_kinds,
    vertex_clauses,
)
from unified_kg_rag.domain.ingestion.seed_graph import SOURCE_KINDS_FIELD
from unified_kg_rag.domain.models import Config, GraphEdge, SearchQuery
from unified_kg_rag.domain.models.graph_query import (
    GraphProfile,
    GraphQueryRequest,
    GraphQueryResult,
)
from unified_kg_rag.domain.retrieval.filters import KindsRule
from unified_kg_rag.domain.retrieval.graph_query import build_profile
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import safe_float_parse

logger = get_logger(__name__)


def _passes(clause: VertexClause, value_map: dict[str, Any]) -> bool:
    """Whether a vertex's stored valueMap passes one clause, as Gremlin's
    ``has`` does on a multi-valued property: a list is ``within`` (any
    stored value in it), a range holds for some stored value, a vertex
    without the property fails (but a ``source_kinds`` rule naming
    ``document``)."""
    stored = value_map.get(clause.field) or []
    if isinstance(clause, KindsRule):
        return clause.passes(stored)
    return clause.holds(stored)


def vertex_filter(filters: dict[str, Any] | None) -> VertexFilter | None:
    """The predicate ``NeptuneRetriever._apply_filters`` applies to a vertex:
    every clause of the scoped ``filters`` (``vertex_clauses``)."""
    clauses = vertex_clauses(filters)
    if not clauses:
        return None
    return lambda value_map: all(_passes(c, value_map) for c in clauses)


class SQLiteGraphRetriever(GraphExpansionRetriever):
    """Graph expansion from seed entities over ``SQLiteGraphStore``.

    The traversal runs in a worker thread (``asyncio.to_thread``); its errors
    are already typed by the store (a locked database is a
    ``StoreTimeoutError``, an unusable one a ``StoreConfigurationError``), so
    both fail the query.
    """

    _store_name: ClassVar[str] = "SQLite graph"

    def __init__(
        self,
        config: Config,
        store: SQLiteGraphStore | None = None,
        **kwargs: Any,
    ) -> None:
        """Args:
        config: Framework config; ``providers.graph.sqlite_path`` is the
            database unless ``store`` is given.
        store: The store to read (e.g. shared with an indexer).
        """
        super().__init__(config, **kwargs)
        self._store = store or SQLiteGraphStore.from_config(config)

    def close(self) -> None:
        self._store.close()

    async def aclose(self) -> None:
        await asyncio.to_thread(self._store.close)

    async def _traverse(
        self, seed_ids: list[Any], query: SearchQuery
    ) -> list[dict[str, Any]]:
        entity_label = self._get_label(
            self._neptune_config.entity_label_prefix, query.suffix
        )
        hops = query.max_hops or self._max_hops
        logger.info(
            "Traversing from entities with label: '%s', seed_count: %s, max_hops: %s",
            entity_label,
            len(seed_ids),
            hops,
        )
        filters = self._scope_filters_to_label(entity_label, query.filters)
        return await asyncio.to_thread(
            self._store.traverse,
            entity_label,
            [str(seed) for seed in seed_ids],
            max_hops=hops,
            max_results_per_hop=self._max_results_per_hop,
            per_seed_limit=seed_shares(
                query, self._neptune_config.traversal_fetch_multiplier, seed_ids
            ),
            keep=vertex_filter(filters),
            with_degree=self._neptune_config.entity_importance_source == "degree",
        )

    async def _edges_within(
        self,
        ids: list[str],
        label: str,
        per_vertex: int,
        kinds: KindsRule | None,
    ) -> list[GraphEdge]:
        rows = await asyncio.to_thread(self._store.edges_within, label, ids)
        edges = [
            GraphEdge(
                id=str(props.get("id") or edge_id),
                source_id=source,
                target_id=target,
                weight=safe_float_parse(props.get("weight")) or 1.0,
                source_kinds=edge_source_kinds(props.get(SOURCE_KINDS_FIELD)),
            )
            for edge_id, source, target, props in rows
        ]
        # aneighbourhood_edges keeps each vertex's first per_vertex.
        return [
            edge for edge in edges if kinds is None or kinds.passes(edge.source_kinds)
        ]

    # --- GraphQueryPort (the structured strategy, docs/design.md §6.6) -----

    async def adescribe(
        self,
        suffix: str | None,
        *,
        sample_vertices: int,
        edges_per_vertex: int,
        max_examples: int,
    ) -> GraphProfile:
        label = self._get_label(self._neptune_config.entity_label_prefix, suffix)
        vertices, edges = await asyncio.to_thread(
            self._store.sample_profile, label, sample_vertices, edges_per_vertex
        )
        return build_profile(
            vertices,
            edges,
            sampled=len(vertices) >= sample_vertices,
            max_examples=max_examples,
        )

    async def aexecute(self, request: GraphQueryRequest) -> GraphQueryResult:
        started = time.monotonic()
        label = self._get_label(
            self._neptune_config.entity_label_prefix, request.suffix
        )
        filters = self._scope_filters_to_label(label, request.filters)
        # Compiled and checked off the loop: a long walk's statement takes
        # tens of milliseconds to write and read back.
        query = await asyncio.to_thread(
            SQLiteQuery,
            request,
            label,
            vertex_ok=vertex_filter(filters),
            edge_ok=edge_predicate(filters),
        )
        logger.debug("Structured query SQL: %s", query.sql)
        rows = await asyncio.to_thread(
            self._store.query_read_only,
            query.sql,
            query.params,
            functions=query.functions(),
            timeout_seconds=min(
                request.timeout_seconds, self._config.search.store_timeout_seconds
            ),
        )
        fields = query.decode(rows)
        ids: dict[str, None] = {}
        for row in fields.get("rows", ()):
            ids.update(dict.fromkeys(row.values()))
        for path in fields.get("paths", ()):
            ids.update(dict.fromkeys(path.vertex_ids))
        if request.operation == "group" and request.group_key is None:
            ids.update(dict.fromkeys(str(g.key) for g in fields.get("groups", ())))
        vertices = (
            await asyncio.to_thread(
                self._store.read_vertex_maps, label, list(ids), request.hydrate
            )
            if ids and request.hydrate
            else {}
        )
        return GraphQueryResult(
            operation=request.operation,
            vertices=vertices,
            backend="sqlite",
            compiled_digest=query.digest,
            elapsed_ms=(time.monotonic() - started) * 1000,
            **fields,
        )
