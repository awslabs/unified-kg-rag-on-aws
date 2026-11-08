# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import functools
import time
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
    is_evaluation_timeout,
    neptune_pool_size,
    neptune_request_error,
    sample_vertex_query,
    stored_vertex_id_mode,
    vertex_id,
    vertex_id_switch_query,
    vertex_ids,
)
from unified_kg_rag.adapters.graph_query.gremlin import (
    HYDRATE_BATCH,
    GremlinQuery,
    decode_profile,
    hydrate_traversal,
    named_vertex_ids,
    profile_traversal,
)
from unified_kg_rag.adapters.retrieval.base import seed_shares
from unified_kg_rag.adapters.retrievers.graph_expansion import (
    GraphExpansionRetriever,
    edge_kinds_filter,
    edge_source_kinds,
)
from unified_kg_rag.adapters.storage.filter_schema import (
    SOURCE_KINDS_FIELD,
    source_kinds_values,
)
from unified_kg_rag.adapters.store_errors import is_fatal_retrieval_error
from unified_kg_rag.domain.ingestion.seed_graph import DOCUMENT_SOURCE_KIND
from unified_kg_rag.domain.models import Config, GraphEdge, SearchQuery
from unified_kg_rag.domain.models.graph_query import (
    GraphProfile,
    GraphQueryRequest,
    GraphQueryResult,
)
from unified_kg_rag.domain.retrieval.graph_query import build_profile
from unified_kg_rag.shared import StoreConnectionError, get_logger
from unified_kg_rag.shared.exceptions import (
    GraphQueryError,
    GraphQueryExecutionError,
)
from unified_kg_rag.shared.logging import warn_once
from unified_kg_rag.shared.utils import safe_float_parse
from unified_kg_rag.shared.utils.concurrency import gather_or_cancel

logger = get_logger(__name__)

# The vertex properties a traversal row projects (the shape
# ``GraphExpansionRetriever`` reads).
_PROJECTED_PROPERTIES = (
    "id",
    "name",
    "type",
    "description",
    "rank",
    "text_unit_ids",
    "community_ids",
    "iri",
    SOURCE_KINDS_FIELD,
)


class NeptuneRetriever(GraphExpansionRetriever):
    """Graph expansion from seed entities over Neptune (Gremlin).

    The ranking and cut live in ``GraphExpansionRetriever``; this class runs
    the Gremlin traversal and owns the Neptune connection, its worker threads
    and its request-error policy.
    """

    _store_name: ClassVar[str] = "Neptune"
    # _edges_within walks bothE() of each vertex (base: scans_incident_edges).
    scans_incident_edges: ClassVar[bool] = True

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
                # Range bounds only: ``aretrieve`` refuses any other key
                # (``check_filter_values``), which this would drop.
                if op in {"gte", "lte", "gt", "lt"}
            ]
        return [(key, value)]

    @staticmethod
    def _source_kinds_predicate(value: Any) -> GraphTraversal:
        """Anonymous traversal a vertex must match for a ``source_kinds``
        entry (``filter_schema``): a vertex holds one of the kinds (a
        multi-valued property: any value), or, when they name ``document``,
        holds no ``source_kinds`` (an entity no seed graph states)."""
        kinds = source_kinds_values(value)
        held = __.has(
            SOURCE_KINDS_FIELD, kinds[0] if len(kinds) == 1 else P.within(kinds)
        )
        if DOCUMENT_SOURCE_KIND not in kinds:
            return held
        return __.or_(held, __.hasNot(SOURCE_KINDS_FIELD))

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
            if key == SOURCE_KINDS_FIELD and isinstance(value, str | list) and value:
                traversal = traversal.where(cls._source_kinds_predicate(value))
                continue
            for step in cls._filter_steps(key, value):
                traversal = traversal.has(*step)
        return traversal

    async def _traverse(
        self, seed_ids: list[Any], query: SearchQuery
    ) -> list[dict[str, Any]]:
        # Every seed the caller chose; a fixed cap here dropped seeds silently.
        # max_hops is validated (>= 1) by NeptuneIndexingConfig (and a
        # query's own by SearchQuery).
        hops = query.max_hops or self._max_hops
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
        # is split across the seeds (seed_shares) and spent inside a per-seed
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
        # the store's first n, without reading the rest of a hub. The seeds
        # come back through the identity() branch (path length 1, the top
        # relevance band, never cut); repeat().emit() emits from the first
        # hop on, never the seed itself, and the expansion excludes the
        # seeds so it spends its budget on new entities. Both shapes stay
        # Neptune-friendly: repeat().times().emit() and union(identity(),
        # ...). An emit() before repeat() is not optimized by Neptune and
        # times out. Seeds of unequal shares (``seed_shares``: local search
        # weights its seeds by score) are walked in one request per share,
        # concurrently, each the traversal above for its seeds.
        shares = seed_shares(
            query, self._neptune_config.traversal_fetch_multiplier, seed_ids
        )
        by_share: dict[int, list[Any]] = {}
        for seed in seed_ids:
            by_share.setdefault(shares[str(seed)], []).append(seed)
        filters = self._scope_filters_to_label(entity_label, query.filters)
        with_degree = self._neptune_config.entity_importance_source == "degree"
        if not seed_ids:
            return []
        settled = self._settled(entity_label)
        custom_ids, switching = await self._vertex_lookup(entity_label)

        def seeds(
            g: GraphTraversalSource, walked: list[Any], custom_ids: bool
        ) -> GraphTraversal:
            if custom_ids:
                return g.V(*vertex_ids(entity_label, [str(s) for s in walked]))
            return g.V().hasLabel(entity_label).has("id", P.within(walked))

        def neighbours() -> GraphTraversal:
            reached = __.both().hasLabel(entity_label)
            if self._neptune_config.deterministic_order:
                reached = reached.order().by("id")
            return reached.limit(self._max_results_per_hop)

        def build(
            g: GraphTraversalSource,
            *,
            walked: list[Any],
            per_seed_limit: int,
            custom_ids: bool,
            switching: str | None,
        ) -> Traversal:
            traversal = seeds(g, walked, custom_ids).union(
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

        async def walk(custom_ids: bool, switching: str | None) -> list[Any]:
            parts = await gather_or_cancel(
                self._execute_traversal(
                    functools.partial(
                        build,
                        walked=walked,
                        per_seed_limit=limit,
                        custom_ids=custom_ids,
                        switching=switching,
                    )
                )
                for limit, walked in by_share.items()
            )
            return [row for part in parts for row in part]

        rows = await walk(custom_ids, switching)
        if custom_ids and settled and not rows:
            # No seed under its custom vertex id: a full build may have
            # switched the label to property ids since it was found settled
            # (its sweep drops the switch vertex, and with it the old
            # copies). Checked again, and expanded again if so.
            self._checked_labels.discard(entity_label)
            lookup = await self._vertex_lookup(entity_label)
            if lookup != (custom_ids, switching):
                custom_ids, switching = lookup
                rows = await walk(custom_ids, switching)
        if switching is not None:
            return self._mark_old_copies(rows, entity_label, switching)
        return rows

    async def _edges_within(
        self,
        ids: list[str],
        label: str,
        per_vertex: int,
        kinds: tuple[list[str], bool] | None,
    ) -> list[GraphEdge]:
        """One request: from each vertex of ``ids``, its first ``per_vertex``
        edges (by relationship id under ``deterministic_order``, else the
        store's first) whose other end is in ``ids`` too and, under a
        ``source_kinds`` filter, that name one of its kinds. Custom vertex
        ids look the vertices up by vertex id; property ids (also while a
        full build switches the label: both copies are read, and their
        edges deduplicated by relationship id) by the ``id`` property."""
        custom_ids, _ = await self._vertex_lookup(label)
        deterministic = self._neptune_config.deterministic_order

        def build(g: GraphTraversalSource) -> Traversal:
            if custom_ids:
                members = vertex_ids(label, ids)
                start = g.V(*members)
                other = __.otherV().hasId(P.within(members))
            else:
                start = g.V().hasLabel(label).has("id", P.within(ids))
                other = __.otherV().hasLabel(label).has("id", P.within(ids))
            incident = __.bothE().where(other)
            if kinds is not None:
                encodings, without = kinds
                named = __.has(SOURCE_KINDS_FIELD, P.within(encodings))
                incident = incident.where(
                    __.or_(named, __.hasNot(SOURCE_KINDS_FIELD)) if without else named
                )
            if deterministic:
                incident = incident.order().by("id")
            return (
                start.local(incident.limit(per_vertex))
                .dedup()
                .project("id", "source_id", "target_id", "weight", SOURCE_KINDS_FIELD)
                .by(__.coalesce(__.values("id"), __.id_()))
                .by(__.outV().values("id"))
                .by(__.inV().values("id"))
                .by(__.coalesce(__.values("weight"), __.constant(1.0)))
                .by(__.coalesce(__.values(SOURCE_KINDS_FIELD), __.constant("")))
            )

        rows = await self._execute_traversal(build)
        return [
            GraphEdge(
                id=str(row["id"]),
                source_id=str(row["source_id"]),
                target_id=str(row["target_id"]),
                weight=safe_float_parse(row.get("weight")) or 1.0,
                source_kinds=edge_source_kinds(row.get(SOURCE_KINDS_FIELD)),
            )
            for row in rows
            if isinstance(row, dict)
        ]

    # --- GraphQueryPort (the structured strategy, docs/design.md §6.6) -----

    def _backend(self) -> str:
        return str(self._config.providers.graph.kind)

    def _query_timeout_ms(self, seconds: float) -> int:
        """A structured query's server-side limit: never past the client's
        own wait (search.store_timeout_seconds)."""
        return int(min(seconds, self._config.search.store_timeout_seconds) * 1000)

    async def adescribe(
        self,
        suffix: str | None,
        *,
        sample_vertices: int,
        edges_per_vertex: int,
        max_examples: int,
    ) -> GraphProfile:
        label = self._get_label(self._neptune_config.entity_label_prefix, suffix)
        timeout_ms = self._query_timeout_ms(self._config.search.store_timeout_seconds)
        rows = await self._execute_traversal(
            functools.partial(
                profile_traversal,
                label=label,
                sample=sample_vertices,
                edges_per_vertex=edges_per_vertex,
                timeout_ms=timeout_ms,
            ),
            degrade=False,
        )
        vertices, edges = decode_profile(rows)
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
        timeout_ms = self._query_timeout_ms(request.timeout_seconds)
        query = GremlinQuery(
            request,
            label,
            apply_filters=lambda traversal: self._apply_filters(traversal, filters),
            edge_rule=edge_kinds_filter(filters),
            timeout_ms=timeout_ms,
        )
        digest: list[str] = []

        def build(g: GraphTraversalSource) -> Traversal:
            traversal = query.build(g)
            digest.append(query.digest(g))
            logger.debug("Structured query bytecode: %s", traversal.bytecode)
            return traversal

        rows = await self._execute_traversal(build, degrade=False)
        fields = query.decode(rows)
        ids = named_vertex_ids(
            fields, request.operation == "group" and request.group_key is None
        )
        # One limit for the query and its hydration: the reads get what the
        # query left of it.
        left_ms = timeout_ms - int((time.monotonic() - started) * 1000)
        if ids and request.hydrate and left_ms <= 0:
            raise GraphQueryExecutionError(
                "The graph store query used up its time limit before its "
                "entities were read",
                timed_out=True,
            )
        vertices = await self._hydrate(label, ids, request.hydrate, max(left_ms, 1))
        return GraphQueryResult(
            operation=request.operation,
            vertices=vertices,
            backend=self._backend(),
            compiled_digest=digest[0] if digest else "",
            elapsed_ms=(time.monotonic() - started) * 1000,
            **fields,
        )

    async def _hydrate(
        self, label: str, ids: list[str], keys: tuple[str, ...], timeout_ms: int
    ) -> dict[str, dict[str, list[Any]]]:
        """The ``keys`` of ``label``'s vertices among ``ids``, by their id
        property (a vertex stored twice: one copy)."""
        found: dict[str, dict[str, list[Any]]] = {}
        if not ids or not keys:
            return found
        batches = [
            ids[i : i + HYDRATE_BATCH] for i in range(0, len(ids), HYDRATE_BATCH)
        ]
        parts = await gather_or_cancel(
            self._execute_traversal(
                functools.partial(
                    hydrate_traversal,
                    label=label,
                    ids=batch,
                    keys=keys,
                    timeout_ms=timeout_ms,
                ),
                degrade=False,
            )
            for batch in batches
        )
        for row in (row for part in parts for row in part):
            if not isinstance(row, dict):
                continue
            props = row.get("props") or {}
            found.setdefault(
                str(row.get("id")),
                {
                    str(k): list(v) if isinstance(v, list) else [v]
                    for k, v in props.items()
                },
            )
        return found

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
        and the not yet rewritten vertices (see :meth:`_mark_old_copies`).
        Refusing the label failed every graph expansion until the build
        ended; the indexer still refuses writes of the other mode.

        A label found settled in ``custom`` mode is only checked for a switch
        on each query (one indexed read), and sampled again once a custom-id
        lookup finds none of the seeds (``_traverse``). One settled in
        ``property`` mode is not read again: its lookups by the ``id``
        property find the vertices of both modes, and a switch can only be
        to custom ids, so each vertex id is read along (no request of its
        own) and a rewritten copy, the one with a custom vertex id, is told
        from its old one whether or not a build is switching the label. Any
        other label is sampled on each query, so lookups follow once a build
        has rewritten it.
        """
        mode = self._neptune_config.vertex_ids
        custom = mode == CUSTOM_VERTEX_IDS
        if self._settled(label):
            if not custom:
                return False, CUSTOM_VERTEX_IDS
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
            "until the full build ends (results may be partial until "
            "then: its edges are split between old and rewritten vertices)",
            label,
            target,
        )
        return False, target

    @staticmethod
    def _mark_old_copies(
        rows: list[Any], label: str, target: str
    ) -> list[dict[str, Any]]:
        """``rows`` (projected with each vertex's ``vertex_id``) without
        their ``vertex_id``, the old copy of a vertex whose rewritten copy
        came back too marked ``superseded``.

        A label a full build is switching to ``target`` holds a vertex it
        has rewritten twice, under its old and its new vertex id, with its
        old and its new edges, until the sweep drops the old copy; the
        rewritten one is the one whose vertex id is of the ``target`` mode.
        A vertex not rewritten yet keeps its one (old) copy. Both copies are
        kept: the ranking keeps the nearer one (the old copy may be nearer
        the seeds over edges not rewritten yet), the rewritten one of two as
        near.
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
        marked: list[dict[str, Any]] = []
        for row in dict_rows:
            kept = {k: v for k, v in row.items() if k != "vertex_id"}
            if not rewritten(row) and item_id(row) in new:
                kept["superseded"] = True
            marked.append(kept)
        return marked

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
            # A seed graph's iri and source_kinds, for the source's
            # provenance: absent from a vertex no seed graph states.
            .by(__.value_map(*_PROJECTED_PROPERTIES))
            .by(__.path().by(__.value_map("name")))
            .by(__.label())
        )
        if degree_label is not None:
            projected = projected.by(__.both().hasLabel(degree_label).count())
        if with_vertex_id:
            projected = projected.by(__.id_())
        return projected

    async def _execute_traversal(
        self,
        build: Callable[[GraphTraversalSource], Traversal],
        *,
        degrade: bool = True,
    ) -> list[Any]:
        """Run the traversal ``build`` makes from the client's current source.

        The traversal is built in the worker thread, from the source of the
        moment, so a query never sends on a connection another query dropped
        meanwhile (``reset`` closes it and empties its pool). On an unusable
        connection the source the request used is dropped, so the next query
        reconnects. A failure that leaves the store usable degrades to no
        rows, or, without ``degrade`` (a structured query, which falls back
        instead), raises ``GraphQueryExecutionError``; a ``GraphQueryError``
        raised building the traversal (a checker refusal) passes as is.
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
            if isinstance(e, GraphQueryError):
                raise
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
            if not degrade:
                timed_out = is_evaluation_timeout(e)
                raise GraphQueryExecutionError(
                    "The graph store "
                    + (
                        "stopped the query at its time limit"
                        if timed_out
                        else "failed the query"
                    )
                    + f": {(str(e).splitlines() or [type(e).__name__])[0][:200]}",
                    timed_out=timed_out,
                ) from e
            logger.error("Gremlin traversal execution failed: %s", e)
            return []
