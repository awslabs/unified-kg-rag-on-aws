# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph expansion from seed entities, independent of the graph query language.

``GraphExpansionRetriever`` holds the ranking, de-duplication, cut and
content-building logic every graph backend shares. A backend subclass only
implements ``_traverse``, which walks the graph from the seeds and returns one
row per reached vertex in the projected shape ``NeptuneRetriever`` produces
with Gremlin::

    {
        "node": {"id": [...], "name": [...], "description": [...],
                 "rank": [...], "text_unit_ids": [...]},  # value_map
        "path": [{"name": [...]}, ...],  # seed first, the node last
        "node_type": "<vertex label>",
        "degree": <int>,  # only when entity_importance_source == "degree"
    }

The seeds themselves come back as rows with a path of length 1.
"""

import time
from abc import abstractmethod
from typing import Any, ClassVar

from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    entity_content,
)
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.adapters.storage.filter_schema import (
    NEPTUNE_FILTER_FIELDS,
    FilterFields,
    union_filter_fields,
)
from unified_kg_rag.adapters.store_errors import is_fatal_retrieval_error
from unified_kg_rag.domain.models import Config, RetrievalResult, SearchQuery
from unified_kg_rag.shared import StoreConnectionError, get_logger
from unified_kg_rag.shared.utils import text_digest

logger = get_logger(__name__)


class GraphExpansionRetriever(BaseGraphRAGRetriever):
    """Graph expansion from seed entities given as an ``id`` filter.

    Every search strategy finds its seed entities in the vector store and
    passes their ids (``filters["id"]``); a query without one returns no
    results. ``indexing.neptune`` holds the logical graph settings (labels,
    hops, fetch budget, importance source) for every graph backend.
    """

    # Name used in log lines.
    _store_name: ClassVar[str] = "Graph"

    def __init__(self, config: Config, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._neptune_config = config.indexing.neptune
        self._max_hops = self._neptune_config.max_hops
        self._max_results_per_hop = self._neptune_config.max_results_per_hop

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
            results = self._process_traversal_results(traversal_results, query)

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
                logger.error(
                    "%s retrieval failed (transient, degrading to empty "
                    "results): %s",
                    self._store_name,
                    e,
                    exc_info=True,
                )
                return []
            if isinstance(error, StoreConnectionError):
                # The traversal has dropped the connection already.
                logger.error("%s retrieval failed: %s", self._store_name, error)
            else:
                # A broken configuration is not "no results".
                logger.error(
                    "%s retrieval failed (fatal): %s",
                    self._store_name,
                    e,
                    exc_info=True,
                )
            if error is e:
                raise
            raise error from e

    def _request_error(self, exc: Exception) -> Exception:
        """``exc`` typed by the backend's error policy (identity by default)."""
        return exc

    @staticmethod
    def _seed_ids(query: SearchQuery) -> list[Any]:
        seed_ids = (query.filters or {}).get("id")
        return list(seed_ids) if isinstance(seed_ids, list) else []

    def filter_fields(self) -> FilterFields:
        return union_filter_fields(NEPTUNE_FILTER_FIELDS.values())

    @staticmethod
    def _scope_filters_to_label(
        label: str, kind: str, filters: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """``filters`` without the keys ``label`` does not declare.

        ``kind`` is the label's ``NEPTUNE_FILTER_FIELDS`` kind. Mirrors the
        OpenSearch per-index scoping: a filter on ``key`` drops every vertex
        lacking ``key``, so a key the label does not declare is dropped
        rather than emptying the traversal.
        """
        if not filters:
            return filters
        fields = NEPTUNE_FILTER_FIELDS[kind]
        kept: dict[str, Any] = {}
        for key, value in filters.items():
            if not fields.declares(key):
                logger.debug(
                    "Dropped filter key '%s': no %s vertex declares it", key, label
                )
                continue
            kept[key] = value
        return kept or None

    @staticmethod
    def _clean_property_map(prop_map: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v[0] if isinstance(v, list) and len(v) == 1 else v
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
                    self._as_float(self._node_data(item).get(importance_key)) or 0.0
                    for item in traversal_results
                ),
                default=0.0,
            )

        # Entity expansion runs per seed, so a node two seeds reach arrives once
        # per seed; keep the copy with the best score (its shortest path, since
        # the score orders by distance first).
        best: dict[Any, RetrievalResult] = {}
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
            kept = best.get(node_id)
            if kept is None or (result.score or 0.0) > (kept.score or 0.0):
                best[node_id] = result

        results = sorted(best.values(), key=lambda x: x.score or 0.0, reverse=True)
        if self._neptune_config.traversal_fetch_multiplier > 1:
            results = self._cut_entities(
                results,
                query.top_k * query.retrieval_multiplier,
                seed_ids={str(s) for s in seed_ids},
            )
        return results

    def _entity_importance_key(self) -> str | None:
        """The node property graph expansion reads as entity importance."""
        source = self._neptune_config.entity_importance_source
        return None if source == "none" else source

    def _node_data(self, item: dict[str, Any]) -> dict[str, Any]:
        """The cleaned property map, with the projected degree folded in."""
        node_data = self._clean_property_map(item.get("node", {}))
        if "degree" in item:
            node_data["degree"] = item["degree"]
        return node_data

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _cut_entities(
        ranked: list[RetrievalResult],
        limit: int,
        seed_ids: set[str] | frozenset[str] = frozenset(),
    ) -> list[RetrievalResult]:
        """Keep the seeds plus the ``limit`` best-ranked other entities.

        The entity traversal over-fetches (``traversal_fetch_multiplier``) so the
        cut happens here, after relevance ranking, instead of in emit order.
        Seeds are never cut and do not use up the ``limit``: they are what the
        query matched, and counting them would leave the expansion no room
        whenever the caller passes ``limit`` or more seeds (local search passes
        ``top_k`` without an entity focus).
        """
        kept: list[RetrievalResult] = []
        expanded = 0
        for result in ranked:
            if result.source not in seed_ids:
                if expanded >= limit:
                    continue
                expanded += 1
            kept.append(result)
        return kept

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
            retriever_type=str(SectionType.ENTITY.value),
            metadata={**node_data, "_node_type": item.get("node_type", "unknown")},
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
        length = len(path) or 1
        raw = (
            GraphExpansionRetriever._as_float(node.get(importance_key))
            if importance_key is not None
            else None
        )
        importance = (
            raw / max_importance if raw is not None and max_importance > 0 else 0.5
        )
        importance = min(max(importance, 0.0), 1.0)
        near, far = 1.0 / length, 1.0 / (length + 1)
        # (1 + importance) / 2 keeps the band's lower edge open, so the most
        # important node one hop farther out never ties the least important one.
        return far + (near - far) * (1.0 + importance) / 2.0
