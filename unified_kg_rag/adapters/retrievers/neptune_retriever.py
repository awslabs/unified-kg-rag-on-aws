# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

from gremlin_python.process.graph_traversal import (
    GraphTraversal,
    GraphTraversalSource,
    __,
)
from gremlin_python.process.traversal import P, Traversal

from unified_kg_rag.adapters.aws.neptune import (
    NeptuneClient,
    neptune_pool_size,
    neptune_request_error,
)
from unified_kg_rag.adapters.retrieval.base import seed_share
from unified_kg_rag.adapters.retrievers.graph_expansion import GraphExpansionRetriever
from unified_kg_rag.adapters.storage.filter_schema import ATTRIBUTE_KEY_PREFIX
from unified_kg_rag.adapters.store_errors import is_fatal_retrieval_error
from unified_kg_rag.domain.models import Config, SearchQuery
from unified_kg_rag.shared import StoreConnectionError, get_logger

logger = get_logger(__name__)


class NeptuneRetriever(GraphExpansionRetriever):
    """Graph expansion from seed entities over Neptune (Gremlin).

    The ranking and cut live in ``GraphExpansionRetriever``; this class runs
    the Gremlin traversal and owns the Neptune connection, its worker threads
    and its request-error policy.
    """

    _store_name: ClassVar[str] = "Neptune"

    def __init__(
        self,
        config: Config,
        neptune_client: NeptuneClient,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self._neptune_client = neptune_client
        # At most one traversal per pooled connection runs in a worker thread;
        # the rest wait here, on the loop, rather than inside gremlinpython's
        # blocking pool.get(). One semaphore per loop (a semaphore is
        # loop-bound).
        self._pool_size = neptune_pool_size(config)
        self._traversal_slots: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Semaphore
        ] = weakref.WeakKeyDictionary()
        # Traversals (and the blocking connection handshake) run on threads of
        # their own, never on the loop's default executor: a traversal that
        # does hang then holds one of these, not a thread the model calls,
        # reranking and every other strategy need. Created on first use.
        self._executor: ThreadPoolExecutor | None = None

    def _traversal_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._pool_size, thread_name_prefix="neptune-retriever"
            )
        return self._executor

    def _shutdown_executor(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def close(self) -> None:
        """Close the underlying Neptune websocket + thread pool (best-effort)."""
        self._neptune_client.close()
        self._shutdown_executor()

    async def aclose(self) -> None:
        """Async teardown: the blocking Neptune close runs off the loop thread."""
        await asyncio.to_thread(self._neptune_client.close)
        self._shutdown_executor()

    def _request_error(self, exc: Exception) -> Exception:
        """``exc`` typed by the Neptune client's policy (``neptune_request_error``).

        A lost connection fails every later request on it, and a request
        without an answer within ``search.store_timeout_seconds`` means the
        server is not answering: both become a ``StoreConnectionError``, which
        fails the query rather than reading as "no results".
        """
        return neptune_request_error(exc, self._config.search.store_timeout_seconds)

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
        """Anonymous traversal a vertex must match for one ``attr_*`` entry.

        ``attr_*`` keys apply where present: a vertex without the property
        passes. Entity vertices carry their own extracted attributes (e.g.
        ``attr_role``) but not document attribute filters (``attr_category``
        lives in OpenSearch), so a strict match would empty graph expansion for
        every document-attribute filter. Every other key is a strict ``has``
        (``_apply_filters``).
        """
        matched = __.has(*steps[0])
        for step in steps[1:]:
            matched = matched.has(*step)
        return __.or_(matched, __.hasNot(key))

    @classmethod
    def _apply_filters(
        cls,
        traversal: GraphTraversal,
        filters: dict[str, Any] | None,
    ) -> GraphTraversal:
        if not filters:
            return traversal

        for key, value in filters.items():
            if key == "id":
                continue
            steps = cls._filter_steps(key, value)
            if not steps:
                continue
            if key.startswith(ATTRIBUTE_KEY_PREFIX):
                traversal = traversal.where(cls._filter_predicate(key, steps))
            else:
                for step in steps:
                    traversal = traversal.has(*step)
        return traversal

    async def _traverse(
        self, seed_ids: list[Any], query: SearchQuery
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
        # non-seed entities to top_k * retrieval_multiplier. The fetch budget
        # is split evenly across the seeds and spent inside a per-seed
        # local(), so the first seeds' neighbourhoods cannot use it up. The
        # limit() after emit() caps each seed's total, in breadth-first emit
        # order (hop 1 first): Neptune scopes a dedup().limit() inside
        # repeat() to each traverser's step of each hop (a limit of 5 on a
        # tree of fan-out 8 fetched 5 + 25 entities), TinkerGraph to the
        # seed. That inner limit stays, as a bound on the work of each hop;
        # the outer dedup() keeps the total to distinct entities on both.
        # Within a seed, local(both().limit(n)) caps how many neighbours any
        # one node contributes per hop (max_results_per_hop) so a hub cannot
        # fan out unbounded. The seeds come back through the identity()
        # branch (path length 1, the top relevance band, never cut);
        # repeat().emit() emits from the first hop on, never the seed itself,
        # and the expansion excludes the seeds so it spends its budget on new
        # entities. Both shapes stay Neptune-friendly:
        # repeat().times().emit() and union(identity(), ...). An emit()
        # before repeat() is not optimized by Neptune and times out.
        per_seed_limit = seed_share(
            query, self._neptune_config.traversal_fetch_multiplier, len(seed_ids)
        )
        filters = self._scope_filters_to_label(entity_label, "entity", query.filters)
        with_degree = self._neptune_config.entity_importance_source == "degree"

        def build(g: GraphTraversalSource) -> Traversal:
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
                        .dedup()
                        .limit(per_seed_limit)
                    ),
                )
            )
            traversal = self._apply_filters(traversal, filters)
            return self._with_projection(
                traversal, degree_label=entity_label if with_degree else None
            )

        return await self._execute_traversal(build)

    @staticmethod
    def _with_projection(
        traversal: GraphTraversal, degree_label: str | None = None
    ) -> GraphTraversal:
        """Project each vertex; with ``degree_label``, also its degree: its
        edges to entities of that label (an older build's ``MemberOf`` edges
        to communities are not relationships)."""
        # `rank` is what the indexer writes on entity vertices (see
        # NeptuneIndexer.index_entities).
        with_degree = degree_label is not None
        keys = ["node", "path", "node_type"] + (["degree"] if with_degree else [])
        projected = (
            traversal.project(*keys)
            .by(__.value_map("id", "name", "description", "rank", "text_unit_ids"))
            .by(__.path().by(__.value_map("name")))
            .by(__.label())
        )
        if degree_label is not None:
            projected = projected.by(__.both().hasLabel(degree_label).count())
        return projected

    async def _execute_traversal(
        self, build: Callable[[GraphTraversalSource], Traversal]
    ) -> list[Any]:
        """Run the traversal ``build`` makes from the client's current source.

        The traversal is built in the worker thread, from the source of the
        moment, so a query never sends on a connection another query dropped
        meanwhile (``reset`` closes it and empties its pool). On an unusable
        connection the source the request used is dropped, so the next query
        reconnects.
        """
        loop = asyncio.get_running_loop()
        slots = self._traversal_slots.get(loop)
        if slots is None:
            slots = self._traversal_slots.setdefault(
                loop, asyncio.Semaphore(self._pool_size)
            )
        used: list[GraphTraversalSource] = []

        def run() -> list[Any]:
            # Opening the connection is blocking: gremlinpython drives its
            # websocket handshake with run_until_complete on a private loop,
            # which fails on a thread that already runs one.
            g = self._neptune_client.g
            used.append(g)
            return list(build(g).to_list())

        # Each request is bounded by store_timeout_seconds, but not the wait
        # for a pooled connection, which can block in gremlinpython's
        # pool.get(). This bound frees the slot, not the thread: the reset
        # below closes the connection, which wakes a thread blocked on its
        # pool and fails one whose write was in flight (_close_connection).
        timeout = 2 * self._config.search.store_timeout_seconds
        try:
            async with slots:
                result: list[Any] = await asyncio.wait_for(
                    loop.run_in_executor(self._traversal_executor(), run), timeout
                )
            return result
        except Exception as e:
            # Fatal errors (an unusable connection, a refused configuration)
            # must reach `aretrieve`, which re-raises them; swallowing them
            # here turned a broken Neptune into a silent "no seed nodes found".
            error = self._request_error(e)
            if is_fatal_retrieval_error(error):
                if isinstance(error, StoreConnectionError) and used:
                    await asyncio.to_thread(self._neptune_client.reset, used[0])
                if error is e:
                    raise
                raise error from e
            logger.error("Gremlin traversal execution failed: %s", e)
            return []
