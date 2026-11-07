# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An in-memory ``GraphQueryPort`` over the reference executor.

``FakeGraphQueryRetriever`` is a graph retriever (``FakeRetriever``'s
shapes) that answers structured queries over one ``PropertyGraph`` per
suffix with ``domain.retrieval.graph_query.reference_operation``, the
caller's filters applied as the stores apply them (the graph expansion's
own predicates). :func:`graph_of` reads a ``FakeGraphStore``'s suffix as
such a graph, so a test can index through the fake and query it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from tests.fixtures.fakes.retrievers import FakeRetriever
from tests.fixtures.fakes.stores import FakeGraphStore
from unified_kg_rag.adapters.retrievers.graph_expansion import (
    GraphExpansionRetriever,
    edge_kinds_filter,
    edge_source_kinds,
)
from unified_kg_rag.adapters.retrievers.sqlite_graph_retriever import vertex_filter
from unified_kg_rag.domain.ingestion.seed_graph import encoded_source_kinds
from unified_kg_rag.domain.models import Constants
from unified_kg_rag.domain.models.graph_query import (
    GraphEdgeRecord,
    GraphProfile,
    GraphQueryRequest,
    GraphQueryResult,
    PropertyGraph,
)
from unified_kg_rag.domain.retrieval.graph_query import (
    build_profile,
    reference_operation,
)


def graph_of(store: FakeGraphStore, suffix: str | None = None) -> PropertyGraph:
    """The entity vertices and relationship edges ``store`` holds for
    ``suffix`` (the default suffix when ``None``)."""
    scope = suffix or Constants.DEFAULT_SUFFIX.value
    vertices = {
        item_id: {k: list(v) for k, v in value_map.items()}
        for (s, item_id), value_map in store.data.get("entities", {}).items()
        if s == scope
    }
    edges = [
        GraphEdgeRecord(
            id=item_id,
            type=row["label"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            props=dict(row["props"]),
        )
        for (s, item_id), row in store.data.get("relationships", {}).items()
        if s == scope
    ]
    return PropertyGraph(vertices=vertices, edges=edges)


def edge_predicate(
    filters: Mapping[str, Any] | None,
) -> Callable[[Mapping[str, Any]], bool] | None:
    """The caller's ``source_kinds`` rule on one edge's stored properties,
    as the graph retrievers apply it."""
    rule = edge_kinds_filter(dict(filters or {}))
    if rule is None:
        return None
    encodings, without = rule

    def keep(props: Mapping[str, Any]) -> bool:
        kinds = edge_source_kinds(props.get("source_kinds"))
        if not kinds:
            return without
        return encoded_source_kinds(kinds) in encodings

    return keep


class FakeGraphQueryRetriever(FakeRetriever):
    """A recording graph retriever with a ``GraphQueryPort`` over in-memory
    graphs (``graphs``: suffix -> graph; ``None`` keys the default)."""

    def __init__(
        self,
        graphs: Mapping[str | None, PropertyGraph] | PropertyGraph,
        tag: str = "entity",
        *,
        fail: Exception | None = None,
    ) -> None:
        super().__init__(tag)
        self.graphs = (
            {None: graphs} if isinstance(graphs, PropertyGraph) else dict(graphs)
        )
        self.fail = fail
        self.requests: list[GraphQueryRequest] = []
        self.describes: list[str | None] = []

    def _graph(self, suffix: str | None) -> PropertyGraph:
        """The graph of ``suffix``: another suffix's is never read."""
        if suffix in self.graphs:
            return self.graphs[suffix]
        default = Constants.DEFAULT_SUFFIX.value
        if suffix in (None, default):
            for key in (None, default):
                if key in self.graphs:
                    return self.graphs[key]
        return PropertyGraph()

    async def adescribe(
        self,
        suffix: str | None,
        *,
        sample_vertices: int,
        edges_per_vertex: int,
        max_examples: int,
    ) -> GraphProfile:
        self.describes.append(suffix)
        graph = self._graph(suffix)
        sampled = sorted(graph.vertices)[:sample_vertices]
        out: dict[str, list[GraphEdgeRecord]] = {}
        for edge in sorted(graph.edges, key=lambda e: e.id):
            out.setdefault(edge.source_id, []).append(edge)

        def kind(vertex_id: str) -> str:
            return str((graph.vertices.get(vertex_id, {}).get("type") or [""])[0])

        return build_profile(
            (graph.vertices[v] for v in sampled),
            [
                (kind(v), edge.type, kind(edge.target_id))
                for v in sampled
                for edge in out.get(v, [])[:edges_per_vertex]
            ],
            sampled=len(sampled) < len(graph.vertices),
            max_examples=max_examples,
        )

    async def aexecute(self, request: GraphQueryRequest) -> GraphQueryResult:
        self.requests.append(request)
        if self.fail is not None:
            raise self.fail
        filters = GraphExpansionRetriever._scope_filters_to_label(
            "fake", request.filters
        )
        return reference_operation(
            request,
            self._graph(request.suffix),
            vertex_ok=vertex_filter(filters),
            edge_ok=edge_predicate(filters),
        )
