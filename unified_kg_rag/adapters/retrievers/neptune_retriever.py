# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import functools
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
    CUSTOM_VERTEX_IDS,
    NeptuneClient,
    neptune_pool_size,
    neptune_request_error,
    sample_vertex_query,
    stored_vertex_id_mode,
    vertex_id,
    vertex_id_switch_query,
    vertex_ids,
)
from unified_kg_rag.adapters.retrieval.base import seed_share
from unified_kg_rag.adapters.retrievers.graph_expansion import GraphExpansionRetriever
from unified_kg_rag.adapters.storage.filter_schema import ATTRIBUTE_KEY_PREFIX
from unified_kg_rag.adapters.store_errors import is_fatal_retrieval_error
from unified_kg_rag.domain.models import Config, SearchQuery
from unified_kg_rag.shared import StoreConnectionError, get_logger
from unified_kg_rag.shared.logging import warn_once

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
        self._closed = False
        # indexing.neptune.vertex_ids: the labels found built in this mode
        # (and not being switched).
        self._checked_labels: set[str] = set()

    def _is_closed(self) -> bool:
        """``_closed`` read through a call, so a re-read after an earlier
        check is not narrowed away (close() may run on another thread)."""
        return self._closed

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
        self._closed = True
        self._neptune_client.close()
        self._shutdown_executor()

    async def aclose(self) -> None:
        """Async teardown: the blocking Neptune close runs off the loop thread."""
        self._closed = True
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
        # fan out unbounded: its first n by id, as the embedded store takes
        # them, not the first n the store happens to keep (two builds of one
        # graph, with their edges written in another order, expanded to
        # other entities); indexing.neptune.deterministic_order: false takes
        # the store's first n, without reading the rest of a hub. The seeds come back through the identity()
        # branch (path length 1, the top relevance band, never cut);
        # repeat().emit() emits from the first hop on, never the seed itself,
        # and the expansion excludes the seeds so it spends its budget on new
        # entities. Both shapes stay Neptune-friendly:
        # repeat().times().emit() and union(identity(), ...). An emit()
        # before repeat() is not optimized by Neptune and times out.
        per_seed_limit = seed_share(
            query, self._neptune_config.traversal_fetch_multiplier, len(seed_ids)
        )
        filters = self._scope_filters_to_label(entity_label, query.filters)
        with_degree = self._neptune_config.entity_importance_source == "degree"
        if not seed_ids:
            return []
        settled = self._settled(entity_label)
        custom_ids, switching = await self._vertex_lookup(entity_label)

        def seeds(g: GraphTraversalSource, custom_ids: bool) -> GraphTraversal:
            if custom_ids:
                return g.V(*vertex_ids(entity_label, [str(s) for s in seed_ids]))
            return g.V().hasLabel(entity_label).has("id", P.within(seed_ids))

        def neighbours() -> GraphTraversal:
            reached = __.both().hasLabel(entity_label)
            if self._neptune_config.deterministic_order:
                reached = reached.order().by("id")
            return reached.limit(self._max_results_per_hop)

        def build(
            g: GraphTraversalSource, *, custom_ids: bool, switching: str | None
        ) -> Traversal:
            traversal = seeds(g, custom_ids).union(
                __.identity(),
                __.local(
                    __.repeat(
                        __.local(neighbours())
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
            traversal = self._apply_filters(traversal, filters)
            return self._with_projection(
                traversal,
                degree_label=entity_label if with_degree else None,
                with_vertex_id=switching is not None,
            )

        rows = await self._execute_traversal(
            functools.partial(build, custom_ids=custom_ids, switching=switching)
        )
        if custom_ids and settled and not rows:
            # No seed under its custom vertex id: a full build may have
            # switched the label to property ids since it was found settled
            # (its sweep drops the switch vertex, and with it the old
            # copies). Checked again, and expanded again if so.
            self._checked_labels.discard(entity_label)
            lookup = await self._vertex_lookup(entity_label)
            if lookup != (custom_ids, switching):
                custom_ids, switching = lookup
                rows = await self._execute_traversal(
                    functools.partial(build, custom_ids=custom_ids, switching=switching)
                )
        if switching is not None:
            return self._rewritten_copies(rows, entity_label, switching)
        return rows

    def _settled(self, label: str) -> bool:
        """Whether ``label`` was found settled in the configured mode (and
        not being switched) on an earlier query."""
        return label in self._checked_labels

    async def _vertex_lookup(self, label: str) -> tuple[bool, str | None]:
        """How the expansion finds ``label``'s vertices: by custom vertex id
        or by the ``id`` property, and the mode a full build is switching the
        label to (``None``: no switch).

        Lookups go by custom vertex id only where every vertex has one: the
        configured mode is ``custom`` and the label is empty or was built so,
        and no build is switching it. Otherwise they go by the ``id``
        property, which vertices of both modes have, with a WARNING once:
        a label of the other mode (the query service switched before the
        full build's first write, or the build before the query service) is
        read as it is, and one being switched is read whole, the rewritten
        and the not yet rewritten vertices (see :meth:`_rewritten_copies`).
        Refusing the label failed every graph expansion until the build
        ended; the indexer still refuses writes of the other mode.

        A label found settled in the configured mode is only checked for a
        switch on each query (one indexed read), and sampled again once a
        custom-id lookup finds none of the seeds (``_traverse``); any other
        is sampled on each query, so lookups follow once a build has
        rewritten it.
        """
        mode = self._neptune_config.vertex_ids
        custom = mode == CUSTOM_VERTEX_IDS
        if self._settled(label):
            switching = await self._execute_traversal(vertex_id_switch_query(label))
            if not switching:
                return custom, None
            self._checked_labels.discard(label)
            return self._switch_lookup(label, str(switching[0]))
        stored = stored_vertex_id_mode(
            label, await self._execute_traversal(sample_vertex_query(label))
        )
        if stored is None:
            return custom, None
        switching = await self._execute_traversal(vertex_id_switch_query(label))
        if switching:
            return self._switch_lookup(label, str(switching[0]))
        if stored != mode:
            warn_once(
                logger,
                f"neptune-vertex-id-mode:{label}:{stored}",
                "Neptune label '%s' was built with vertex_ids '%s', but this "
                "query uses '%s': graph expansion looks its vertices up by "
                "their id property (slower) until a full build with "
                "vertex_ids: %s rewrites it, or set vertex_ids back to '%s'",
                label,
                stored,
                mode,
                mode,
                stored,
            )
            return False, None
        self._checked_labels.add(label)
        return custom, None

    @staticmethod
    def _switch_lookup(label: str, target: str) -> tuple[bool, str]:
        """The lookup of ``label`` while a full build switches it to
        ``target``: by the ``id`` property, with a WARNING once."""
        warn_once(
            logger,
            f"neptune-vertex-id-switch:{label}:{target}",
            "Neptune label '%s' is being rewritten with vertex_ids '%s': "
            "graph expansion looks its vertices up by their id property "
            "until the full build ends (rewritten vertices win over "
            "their old copies)",
            label,
            target,
        )
        return False, target

    @staticmethod
    def _rewritten_copies(
        rows: list[Any], label: str, target: str
    ) -> list[dict[str, Any]]:
        """``rows`` (projected with each vertex's ``vertex_id``) without the
        old copy of a vertex whose rewritten copy came back too.

        A label a full build is switching to ``target`` holds a vertex it
        has rewritten twice, under its old and its new vertex id, with its
        old and its new edges, until the sweep drops the old copy; the
        rewritten one is the one whose vertex id is of the ``target`` mode.
        A vertex not rewritten yet keeps its one (old) copy.
        """

        def item_id(row: dict[str, Any]) -> str | None:
            ids = (row.get("node") or {}).get("id") or []
            return str(ids[0]) if ids else None

        def rewritten(row: dict[str, Any]) -> bool:
            item = item_id(row)
            if item is None:
                return False
            custom = str(row.get("vertex_id")) == vertex_id(label, item)
            return custom == (target == CUSTOM_VERTEX_IDS)

        dict_rows = [row for row in rows if isinstance(row, dict)]
        new = {item_id(row) for row in dict_rows if rewritten(row)}
        kept: list[dict[str, Any]] = []
        for row in dict_rows:
            if rewritten(row) or item_id(row) not in new:
                kept.append({k: v for k, v in row.items() if k != "vertex_id"})
        return kept

    @staticmethod
    def _with_projection(
        traversal: GraphTraversal,
        degree_label: str | None = None,
        with_vertex_id: bool = False,
    ) -> GraphTraversal:
        """Project each vertex; with ``degree_label``, also its degree: its
        edges to entities of that label (an older build's ``MemberOf`` edges
        to communities are not relationships); with ``with_vertex_id``, its
        vertex id."""
        # `rank` is what the indexer writes on entity vertices (see
        # NeptuneIndexer.index_entities).
        with_degree = degree_label is not None
        keys = ["node", "path", "node_type"] + (["degree"] if with_degree else [])
        if with_vertex_id:
            keys.append("vertex_id")
        projected = (
            traversal.project(*keys)
            .by(__.value_map("id", "name", "description", "rank", "text_unit_ids"))
            .by(__.path().by(__.value_map("name")))
            .by(__.label())
        )
        if degree_label is not None:
            projected = projected.by(__.both().hasLabel(degree_label).count())
        if with_vertex_id:
            projected = projected.by(__.id_())
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
        closed = StoreConnectionError("The Neptune retriever is closed.")
        if self._closed:
            # No new worker pool or connection for a closed retriever.
            raise closed
        loop = asyncio.get_running_loop()
        slots = self._traversal_slots.get(loop)
        if slots is None:
            slots = self._traversal_slots.setdefault(
                loop, asyncio.Semaphore(self._pool_size)
            )
        used: list[GraphTraversalSource] = []

        def run() -> list[Any]:
            if self._closed:
                # Closed before this worker started: open nothing.
                raise closed
            # Opening the connection is blocking: gremlinpython drives its
            # websocket handshake with run_until_complete on a private loop,
            # which fails on a thread that already runs one.
            g = self._neptune_client.g
            # Read again: close() may run on another thread meanwhile.
            if self._is_closed():
                # Closed while this worker waited for the client (close()
                # holds its lock): ``g`` may be a connection opened after
                # the close, which nothing would close.
                self._neptune_client.close()
                raise closed
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
                if self._closed:
                    # Closed while this query waited for a slot: no new pool.
                    raise closed
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
