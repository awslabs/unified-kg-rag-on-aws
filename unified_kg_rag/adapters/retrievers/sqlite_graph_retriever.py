# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph expansion over the embedded SQLite graph store.

The walk and the filters reproduce ``NeptuneRetriever``'s Gremlin traversal
(see ``SQLiteGraphStore.traverse``); ranking, de-duplication and the cut come
from ``GraphExpansionRetriever``, so both backends rank the same graph alike.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from unified_kg_rag.adapters.local.sqlite_graph import SQLiteGraphStore, VertexFilter
from unified_kg_rag.adapters.retrieval.base import seed_share
from unified_kg_rag.adapters.retrievers.graph_expansion import GraphExpansionRetriever
from unified_kg_rag.adapters.storage.filter_schema import ATTRIBUTE_KEY_PREFIX
from unified_kg_rag.domain.models import Config, SearchQuery
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

_COMPARISONS = {
    "gte": lambda stored, value: stored >= value,
    "lte": lambda stored, value: stored <= value,
    "gt": lambda stored, value: stored > value,
    "lt": lambda stored, value: stored < value,
    "eq": lambda stored, value: stored == value,
    "neq": lambda stored, value: stored != value,
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
    vertices that have the property; every other key also drops vertices
    without it. A dict without a range operator filters nothing.
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
                if key.startswith(ATTRIBUTE_KEY_PREFIX):
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
        logger.info(
            "Traversing from entities with label: '%s', seed_count: %s, max_hops: %s",
            entity_label,
            len(seed_ids),
            self._max_hops,
        )
        filters = self._scope_filters_to_label(entity_label, query.filters)
        return await asyncio.to_thread(
            self._store.traverse,
            entity_label,
            [str(seed) for seed in seed_ids],
            max_hops=self._max_hops,
            max_results_per_hop=self._max_results_per_hop,
            per_seed_limit=seed_share(
                query, self._neptune_config.traversal_fetch_multiplier, len(seed_ids)
            ),
            keep=vertex_filter(filters),
            with_degree=self._neptune_config.entity_importance_source == "degree",
        )
