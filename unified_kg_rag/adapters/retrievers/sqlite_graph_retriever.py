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
    edge_kinds_pass,
    edge_predicate,
    edge_source_kinds,
)
from unified_kg_rag.adapters.storage.filter_schema import (
    ATTRIBUTE_KEY_PREFIX,
    SOURCE_KINDS_FIELD,
    source_kinds_values,
)
from unified_kg_rag.domain.ingestion.seed_graph import DOCUMENT_SOURCE_KIND
from unified_kg_rag.domain.models import Config, GraphEdge, SearchQuery
from unified_kg_rag.domain.models.graph_query import (
    GraphProfile,
    GraphQueryRequest,
    GraphQueryResult,
)
from unified_kg_rag.domain.retrieval.graph_query import build_profile
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import safe_float_parse

logger = get_logger(__name__)

_COMPARISONS = {
    "gte": lambda stored, value: stored >= value,
    "lte": lambda stored, value: stored <= value,
    "gt": lambda stored, value: stored > value,
    "lt": lambda stored, value: stored < value,
}


def _matches(values: list[Any], value: Any) -> bool:
    """Whether a stored multi-valued property passes one filter entry.

    As Gremlin's ``has``: a list is ``within`` (any stored value in it), a
    dict holds range operators that must all hold for some stored value
    each, anything else is equality. A comparison across types fails.
    """
    if isinstance(value, list):
        return any(v in value for v in values)
    if isinstance(value, dict):
        for op, operand in value.items():
            compare = _COMPARISONS.get(op)
            if compare is None:
                continue
            try:
                if not any(compare(v, operand) for v in values):
                    return False
            except TypeError:
                return False
        return True
    return value in values


def vertex_filter(filters: dict[str, Any] | None) -> VertexFilter | None:
    """The predicate ``NeptuneRetriever._apply_filters`` applies to a vertex.

    ``id`` is the seed list, not a filter. An ``attr_*`` key applies only to
    vertices that have the property, and a ``source_kinds`` entry naming
    ``document`` passes vertices without it too (an entity no seed graph
    states); every other key also drops vertices without it. A dict without
    a range operator filters nothing; the retriever refuses one before
    (``check_filter_values``), as the OpenSearch retriever does.
    """
    entries = [
        (key, value)
        for key, value in (filters or {}).items()
        if key != "id"
        and not (
            isinstance(value, dict) and not any(op in _COMPARISONS for op in value)
        )
    ]
    if not entries:
        return None

    def keep(value_map: dict[str, Any]) -> bool:
        for key, value in entries:
            stored = value_map.get(key)
            if not stored:
                if key.startswith(ATTRIBUTE_KEY_PREFIX) or (
                    key == SOURCE_KINDS_FIELD
                    and isinstance(value, str | list)
                    and DOCUMENT_SOURCE_KIND in source_kinds_values(value)
                ):
                    continue
                return False
            if not _matches(stored, value):
                return False
        return True

    return keep


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
        kinds: tuple[list[str], bool] | None,
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
        return [edge for edge in edges if edge_kinds_pass(edge, kinds)]

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
        query = SQLiteQuery(
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
