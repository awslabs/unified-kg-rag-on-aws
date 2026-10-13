# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import math
import time
import weakref
from typing import Any

from gremlin_python.process.graph_traversal import (
    GraphTraversal,
    GraphTraversalSource,
    __,
)
from gremlin_python.process.traversal import P, Traversal

from unified_kg_rag.adapters.aws.neptune import NeptuneClient, neptune_pool_size
from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    is_fatal_retrieval_error,
)
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.adapters.storage.filter_schema import (
    ATTRIBUTE_KEY_PREFIX,
    NEPTUNE_FILTER_FIELDS,
    FilterFields,
    union_filter_fields,
)
from unified_kg_rag.domain.models import Config, RetrievalResult, SearchQuery
from unified_kg_rag.shared import StoreConnectionError, get_logger
from unified_kg_rag.shared.utils import text_digest

logger = get_logger(__name__)


def _is_connect_failure(exc: BaseException) -> bool:
    """Whether ``exc`` is aiohttp's ``ClientConnectorError`` (or a subclass).

    gremlinpython's transport raises it when the websocket cannot be opened.
    Matched by class name, as aiohttp is gremlinpython's dependency, not ours.
    """
    return any(cls.__name__ == "ClientConnectorError" for cls in type(exc).__mro__)


class NeptuneRetriever(BaseGraphRAGRetriever):
    """Graph expansion from seed entities given as an ``id`` filter.

    Every search strategy finds its seed entities in OpenSearch and passes
    their ids (``filters["id"]``); a query without one returns no results.
    """

    def __init__(
        self,
        config: Config,
        neptune_client: NeptuneClient,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self._neptune_client = neptune_client
        self._neptune_config = config.indexing.neptune
        self._max_hops = self._neptune_config.max_hops
        self._max_results_per_hop = self._neptune_config.max_results_per_hop
        # At most one traversal per pooled connection runs in a worker thread;
        # the rest wait here, on the loop. Otherwise each waits inside
        # gremlinpython's blocking pool.get() while holding a default-executor
        # thread, and the Bedrock calls LangChain runs on that executor queue
        # behind them. One semaphore per loop (a semaphore is loop-bound).
        self._pool_size = neptune_pool_size(config)
        self._traversal_slots: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Semaphore
        ] = weakref.WeakKeyDictionary()

    def close(self) -> None:
        """Close the underlying Neptune websocket + thread pool (best-effort)."""
        self._neptune_client.close()

    async def aclose(self) -> None:
        """Async teardown: the blocking Neptune close runs off the loop thread."""
        await asyncio.to_thread(self._neptune_client.close)

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        start_time = time.time()
        logger.info(
            "Neptune retrieval started - query: %s ('%s')",
            text_digest(query.query),
            query.search_type.value,
        )

        try:
            seed_ids = self._seed_ids(query)
            if not seed_ids:
                logger.warning(
                    "Neptune retrieval needs seed entity ids (an 'id' filter); "
                    "none given for query: %s",
                    text_digest(query.query),
                )
                return []
            # Opening the connection is blocking: gremlinpython drives its
            # websocket handshake with run_until_complete on a private loop,
            # which fails on a thread that already runs one (the old code
            # only worked because the CLIs apply nest_asyncio).
            g = await asyncio.to_thread(lambda: self._neptune_client.g)

            traversal_results = await self._traverse_from_entities(g, seed_ids, query)
            results = self._process_traversal_results(traversal_results, query)

            processing_time = time.time() - start_time
            logger.info(
                "Neptune retrieval completed - retrieved: %s results (%.2fs)",
                len(results),
                processing_time,
            )

            return results

        except Exception as e:
            # Re-raise clearly-fatal errors (auth/credentials/endpoint/
            # connection) so a broken configuration is not silently reported as
            # "no results"; degrade to an empty list only on transient failures.
            if is_fatal_retrieval_error(e):
                logger.error("Neptune retrieval failed (fatal): %s", e, exc_info=True)
                raise
            logger.error(
                "Neptune retrieval failed (transient, degrading to empty "
                "results): %s",
                e,
                exc_info=True,
            )
            return []

    @staticmethod
    def _seed_ids(query: SearchQuery) -> list[Any]:
        seed_ids = (query.filters or {}).get("id")
        return list(seed_ids) if isinstance(seed_ids, list) else []

    def filter_fields(self) -> FilterFields:
        return union_filter_fields(NEPTUNE_FILTER_FIELDS.values())

    @staticmethod
    def _scope_filters_to_labels(
        label_kinds: dict[str, str], filters: dict[str, Any] | None
    ) -> tuple[dict[str, Any] | None, dict[str, list[str]]]:
        """Restrict filters to the vertex labels that declare each key.

        ``label_kinds`` maps each target label to its ``NEPTUNE_FILTER_FIELDS``
        kind. Mirrors the OpenSearch per-index scoping: ``has(key, ...)`` drops
        every vertex lacking ``key``, so keys no target label declares are
        dropped, and for a key only some labels declare the returned ``exempt``
        map lists the labels the filter must not touch.
        """
        if not filters:
            return filters, {}
        kept: dict[str, Any] = {}
        exempt: dict[str, list[str]] = {}
        for key, value in filters.items():
            lacking = [
                label
                for label, kind in label_kinds.items()
                if not NEPTUNE_FILTER_FIELDS[kind].declares(key)
            ]
            if len(lacking) == len(label_kinds):
                logger.debug(
                    "Dropped filter key '%s': no %s vertex declares it",
                    key,
                    "/".join(label_kinds),
                )
                continue
            kept[key] = value
            if lacking:
                exempt[key] = lacking
        return kept or None, exempt

    @staticmethod
    def _filter_steps(key: str, value: Any) -> list[tuple[str, Any]]:
        """``(key, predicate)`` pairs for one filter entry (``has`` arguments)."""
        if isinstance(value, list):
            return [(key, P.within(value))]
        if isinstance(value, dict):
            return [
                (key, getattr(P, op)(val))
                for op, val in value.items()
                if op in {"gte", "lte", "gt", "lt", "eq", "neq"}
            ]
        return [(key, value)]

    @staticmethod
    def _filter_predicate(key: str, steps: list[tuple[str, Any]]) -> GraphTraversal:
        """Anonymous traversal a vertex must match for one filter entry.

        ``attr_*`` keys apply where present: a vertex without the property
        passes. Entity vertices carry their own extracted attributes (e.g.
        ``attr_role``) but not document attribute filters (``attr_category``
        lives in OpenSearch), so a strict match would empty graph expansion for
        every document-attribute filter. Every other key is strict.
        """
        matched = __.has(*steps[0])
        for step in steps[1:]:
            matched = matched.has(*step)
        if key.startswith(ATTRIBUTE_KEY_PREFIX):
            return __.or_(matched, __.hasNot(key))
        return matched

    @classmethod
    def _apply_filters(
        cls,
        traversal: GraphTraversal,
        filters: dict[str, Any] | None,
        exempt_labels: dict[str, list[str]] | None = None,
    ) -> GraphTraversal:
        if not filters:
            return traversal

        for key, value in filters.items():
            if key == "id":
                continue
            steps = cls._filter_steps(key, value)
            if not steps:
                continue
            predicate = cls._filter_predicate(key, steps)
            exempt = (exempt_labels or {}).get(key)
            if exempt:
                # Only the labels that carry ``key`` are filtered; vertices of
                # the exempt labels pass through untouched.
                traversal = traversal.or_(__.hasLabel(*exempt), predicate)
            elif key.startswith(ATTRIBUTE_KEY_PREFIX):
                traversal = traversal.where(predicate)
            else:
                for step in steps:
                    traversal = traversal.has(*step)
        return traversal

    async def _traverse_from_entities(
        self, g: GraphTraversalSource, seed_ids: list[Any], query: SearchQuery
    ) -> list[dict[str, Any]]:
        # Every seed the caller chose; a fixed cap here dropped seeds silently.
        # max_hops is validated (>= 1) by NeptuneIndexingConfig.
        hops = self._max_hops
        entity_label = self._get_label(
            self._neptune_config.entity_label_prefix, query.suffix
        )
        logger.info(
            "Traversing from entities with label: '%s', seed_count: %s, max_hops: %s",
            entity_label,
            len(seed_ids),
            hops,
        )

        # Over-fetch, then let `_process_traversal_results` rank and cut the
        # non-seed entities to top_k * retrieval_multiplier. The fetch budget is split evenly across
        # the seeds and spent inside a per-seed local(), because a limit()
        # inside repeat() counts every traverser of the whole traversal: one
        # global cap let the first seeds' neighbourhoods use it up and left the
        # other seeds with none. Within a seed, local(both().limit(n)) caps how
        # many neighbours any one node contributes per hop (max_results_per_hop)
        # so a hub cannot fan out unbounded. The seeds come back through the
        # identity() branch (path length 1, the top relevance band, never cut);
        # repeat().emit() emits from the first hop on, never the seed itself,
        # and the expansion excludes the seeds so it spends its budget on new
        # entities. Both
        # shapes stay Neptune-friendly: repeat().times().emit() and
        # union(identity(), ...). An emit() before repeat() is not optimized by
        # Neptune and times out.
        fetch_limit = (
            query.top_k
            * query.retrieval_multiplier
            * self._neptune_config.traversal_fetch_multiplier
        )
        # No per-seed floor is needed: ceil() gives every seed at least one and
        # the seeds together still fetch >= fetch_limit candidates for the
        # top_k * retrieval_multiplier expansion slots the cut keeps.
        per_seed_limit = math.ceil(fetch_limit / len(seed_ids))
        traversal = (
            g.V()
            .hasLabel(entity_label)
            .has("id", P.within(seed_ids))
            .union(
                __.identity(),
                __.local(
                    __.repeat(
                        __.local(
                            __.both()
                            .hasLabel(entity_label)
                            .limit(self._max_results_per_hop)
                        )
                        .has("id", P.without(seed_ids))
                        .dedup()
                        .limit(per_seed_limit)
                    )
                    .times(hops)
                    .emit()
                ),
            )
        )
        filters, exempt = self._scope_filters_to_labels(
            {entity_label: "entity"}, query.filters
        )
        traversal = self._apply_filters(traversal, filters, exempt)
        traversal = self._with_projection(
            traversal,
            with_degree=self._neptune_config.entity_importance_source == "degree",
        )
        return await self._execute_traversal(traversal)

    @staticmethod
    def _with_projection(
        traversal: GraphTraversal, with_degree: bool = False
    ) -> GraphTraversal:
        # `rank` is what the indexer writes on entity vertices (see
        # NeptuneIndexer.index_entities).
        keys = ["node", "path", "node_type"] + (["degree"] if with_degree else [])
        projected = (
            traversal.project(*keys)
            .by(__.value_map("id", "name", "description", "rank", "text_unit_ids"))
            .by(__.path().by(__.value_map("name")))
            .by(__.label())
        )
        if with_degree:
            projected = projected.by(__.bothE().count())
        return projected

    async def _execute_traversal(self, traversal: Traversal) -> list[Any]:
        loop = asyncio.get_running_loop()
        slots = self._traversal_slots.get(loop)
        if slots is None:
            slots = self._traversal_slots.setdefault(
                loop, asyncio.Semaphore(self._pool_size)
            )
        try:
            # to_list() is a BLOCKING Gremlin round trip; running it directly in
            # this coroutine would block the event loop and serialize the seed
            # lookups that callers fan out via asyncio.gather. Offload it to a
            # worker thread so the gather actually overlaps.
            async with slots:
                result: list[Any] = await asyncio.to_thread(traversal.to_list)
            return result
        except Exception as e:
            if _is_connect_failure(e):
                # The websocket could not be (re)opened: the server is down or
                # unreachable (refused, DNS, TLS), as every later query will be.
                raise StoreConnectionError(
                    f"Neptune is not reachable (Gremlin connection failed): {e}"
                ) from e
            # Fatal errors (auth/credentials/endpoint/connection) must reach
            # `aretrieve`, which re-raises them; swallowing them here turned a
            # broken Neptune configuration into a silent "no seed nodes found".
            if is_fatal_retrieval_error(e):
                raise
            logger.error("Gremlin traversal execution failed: %s", e)
            return []

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
                # The identity() branch: a seed, not something reached from one.
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
        content_parts = [
            f"Entity: {node_data.get('name', 'Unknown')}",
            f"Description: {node_data.get('description', 'N/A')}",
        ]
        path_names = [p.get("name", [""])[0] for p in path_data if isinstance(p, dict)]
        if path_names:
            content_parts.append(f"Path: {' -> '.join(filter(None, path_names))}")

        return "\n".join(content_parts)

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
        neutral 0.5 when absent. Without an ``importance_key`` the legacy
        ``importance`` property is read (entity vertices never store it, so
        this is the neutral 0.5). (A query-text term was dropped: graph
        expansion passes the whole question, which never occurs inside a node
        name.)
        """
        length = len(path) or 1
        if importance_key is not None:
            raw = NeptuneRetriever._as_float(node.get(importance_key))
            importance = (
                raw / max_importance if raw is not None and max_importance > 0 else 0.5
            )
        else:
            importance = float(node.get("importance", 0.5))
        importance = min(max(importance, 0.0), 1.0)
        near, far = 1.0 / length, 1.0 / (length + 1)
        # (1 + importance) / 2 keeps the band's lower edge open, so the most
        # important node one hop farther out never ties the least important one.
        return far + (near - far) * (1.0 + importance) / 2.0
