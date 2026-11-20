# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The visualization-data JSON that ``run-visualization`` renders from.

Ingestion writes it at the end of a full build
(``graph.visualization.export_data``), and ``run-visualization`` after an
incremental run, from what the index namespace's stores hold
(:func:`stored_visualization_data`): the whole graph and the community
hierarchy, also beside another corpus's documents, built here in one pass over the nodes and edges (no
embedding or model call, no centrality, no layout). ``run-visualization`` computes the layout and centrality when it
renders; a file that carries them (written by an earlier release) is drawn
with its own.

The layout helpers (:func:`displayed_graph`, :func:`spring_layout`) are
renderer-neutral: the one layout and node cap of a drawn graph, which
``run-visualization`` and the console's Graph page share.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import networkx as nx

from unified_kg_rag.domain.ingestion.graph_builder import GraphBuilder
from unified_kg_rag.domain.models import Claim, Entity, Relationship
from unified_kg_rag.shared.utils import strip_embedding_fields

__all__ = [
    "DEFAULT_OUTPUTS_DIRECTORY",
    "VISUALIZATION_DATA_FILENAME",
    "displayed_graph",
    "graph_data",
    "spring_layout",
    "stored_visualization_data",
    "strip_heavy_attributes",
]

# File written into the visualization outputs directory; it is the input
# ``run-visualization --data-path`` re-renders from without ingestion.
VISUALIZATION_DATA_FILENAME = "visualization_data.json"
DEFAULT_OUTPUTS_DIRECTORY = "outputs/visualization"
# The packed layout (spring_layout): a component of n nodes takes a square
# of side sqrt(n); a row is _ROW_WIDTH times the largest's side wide, and
# _COMPONENT_GAP apart from the next. Presentation: the console's
# layout before it moved here.
_ROW_WIDTH = 2.2
_COMPONENT_GAP = 0.4


def strip_heavy_attributes(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the attributes no renderer reads from exported nodes and edges.

    Vectors are large and dominate file size. ``relationships`` is the copy of
    every relationship an edge merges (see ``GraphBuilder``), which only
    community detection on the live graph reads; exported, it repeats each
    edge's own attributes.
    """
    return [
        {
            **item,
            "attributes": {
                key: value
                for key, value in strip_embedding_fields(
                    item.get("attributes") or {}
                ).items()
                if key != "relationships"
            },
        }
        for item in items
    ]


def graph_data(
    graph: nx.Graph,
    communities: dict[str, Any],
    node_communities: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Nodes, edges and ``communities`` (a ``hierarchy`` list of
    ``community_id``, ``level``, ``nodes``, ``size``, ``parent`` and
    ``children``), without a layout or centrality.

    ``node_communities`` (``{node id: level-0 community id}``) is written into
    each node's ``community_id`` attribute, which the renderers color by; the
    graph itself is not modified.
    """
    node_communities = node_communities or {}

    def _node_attributes(node_id: Any, attrs: dict[str, Any]) -> dict[str, Any]:
        community = node_communities.get(str(node_id))
        if community is None:
            return attrs
        return {**attrs, "community_id": community}

    return {
        "nodes": strip_heavy_attributes(
            [
                {"id": str(node_id), "attributes": _node_attributes(node_id, attrs)}
                for node_id, attrs in graph.nodes(data=True)
            ]
        ),
        "edges": strip_heavy_attributes(
            [
                {"source": str(u), "target": str(v), "attributes": attrs}
                for u, v, attrs in graph.edges(data=True)
            ]
        ),
        "layout": {},
        "communities": communities,
    }


def _level(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def stored_visualization_data(
    entities: Iterable[Entity],
    relationships: Iterable[Relationship],
    claims: Iterable[Claim],
    reports: Iterable[dict[str, Any]],
    *,
    namespace: str,
    built_at: str | None = None,
) -> dict[str, Any]:
    """The visualization data of an index namespace from what its stores
    hold: the graph ``GraphBuilder`` builds of its entities, relationships
    and claims, and the community hierarchy of its stored community reports
    (``StoreInspection.scan_items`` records), each community's members the
    entities that list it in ``community_ids``. Deterministic: items in id
    order, communities by level and id. ``namespace`` names it, so
    ``run-visualization --namespace`` finds the newest export of one, and
    ``built_at`` (its build record's, when it has one) tells which build
    the stores held."""
    built = GraphBuilder(
        sorted(entities, key=lambda e: e.id),
        sorted(relationships, key=lambda r: r.id),
        sorted(claims, key=lambda c: c.id),
    ).build()
    stored = sorted(
        (r for r in reports if r.get("community_id")),
        key=lambda r: (_level(r.get("level")), str(r["community_id"])),
    )
    levels = {str(r["community_id"]): _level(r.get("level")) for r in stored}
    members: dict[str, list[str]] = {community: [] for community in levels}
    node_communities: dict[str, str] = {}
    for node_id, attrs in built.nodes(data=True):
        for community in attrs.get("community_ids") or []:
            if (community := str(community)) not in members:
                continue
            members[community].append(str(node_id))
            if levels[community] == 0:
                node_communities.setdefault(str(node_id), community)
    hierarchy = [
        {
            "community_id": str(r["community_id"]),
            "level": levels[str(r["community_id"])],
            "nodes": sorted(members[str(r["community_id"])]),
            "size": len(members[str(r["community_id"])]),
            "parent": r.get("parent_community_id") or None,
            "children": [str(c) for c in r.get("child_community_ids") or []],
        }
        for r in stored
    ]
    # The namespace and the build first: a reader finds them in the file's
    # first bytes.
    return {
        "namespace": namespace,
        "built_at": built_at,
        **graph_data(built, {"hierarchy": hierarchy}, node_communities),
    }


def displayed_graph(
    graph: nx.Graph, max_nodes: int | None, max_edges: int | None = None
) -> nx.Graph:
    """The part of ``graph`` the interactive page draws: its top ``max_nodes``
    by degree (``graph.visualization.interactive.max_nodes``; None or 0 draws
    every node), and of the links between them at most ``max_edges``
    (``interactive.max_edges``; None or 0 keeps them all), as
    :func:`kept_edges` picks them."""
    if max_nodes and graph.number_of_nodes() > max_nodes:
        ranked = sorted(graph.degree(), key=lambda nd: nd[1], reverse=True)
        graph = graph.subgraph(node for node, _ in ranked[:max_nodes])
    if not max_edges or graph.number_of_edges() <= max_edges:
        return graph
    # Every drawn node stays, one left with no link too.
    pruned = nx.Graph()
    pruned.add_nodes_from(graph.nodes(data=True))
    pruned.add_edges_from(
        (u, v, graph.adj[u][v]) for u, v in kept_edges(graph, max_edges)
    )
    return pruned


def kept_edges(graph: nx.Graph, max_edges: int) -> list[tuple[Any, Any]]:
    """``max_edges`` of an undirected ``graph``'s links, chosen so every node
    keeps its own strongest: a node ranks its links by ``weight`` (heaviest
    first), then by the other end's degree, and a link counts at the better
    of its two ends' ranks; the lowest counts are kept, each node's first
    link before any node's second (local-degree sparsification, Lindner et
    al. 2015, "Structure-preserving sparsification methods for social
    networks", with the weight first). In the graph's own order on ties, so
    the same graph keeps the same links."""
    nodes = list(graph)
    order = {node: i for i, node in enumerate(nodes)}
    adjacency = graph.adj
    degree = {node: len(adjacency[node]) for node in nodes}

    def weight(attributes: dict[str, Any]) -> float:
        value = attributes.get("weight")
        return float(value) if isinstance(value, int | float) else 0.0

    # Each link, by its ends' positions (lower first): its best rank and
    # its weight.
    best: dict[tuple[int, int], tuple[int, float]] = {}
    for node, neighbours in adjacency.items():
        here = order[node]
        ranked = sorted(
            neighbours,
            key=lambda other: (
                -weight(neighbours[other]),
                -degree[other],
                order[other],
            ),
        )
        for rank, other in enumerate(ranked):
            there = order[other]
            key = (here, there) if here < there else (there, here)
            seen = best.get(key)
            if seen is None or rank < seen[0]:
                best[key] = (rank, weight(neighbours[other]))
    chosen = sorted(best, key=lambda key: (best[key][0], -best[key][1], key))
    return [(nodes[a], nodes[b]) for a, b in chosen[:max_edges]]


def spring_layout(
    graph: nx.Graph, max_nodes: int | None, max_edges: int | None = None
) -> dict[str, list[float]]:
    """The one layout of a drawn graph (``run-visualization``'s pages and
    the console's Graph page): a seeded spring layout of each connected
    component of the drawn nodes (:func:`displayed_graph`), the components
    packed in rows largest first, so small islands do not push the main
    component into a corner (one spring layout of a disconnected graph
    flings its islands far apart).

    A spring layout is quadratic in the node count, so laying out a large
    graph and then drawing its top ``max_nodes`` spent minutes on nodes
    the page never shows. A graph within the cap is laid out whole. The
    links laid out are those drawn (``max_edges``).
    """
    # In the graph's own node order, never a set's: a set of strings
    # iterates in an order of the process's hash seed, so the seeded layout
    # moved from one run to the next.
    drawn = displayed_graph(graph, max_nodes, max_edges)
    order = {node: i for i, node in enumerate(graph)}
    components = sorted(
        (
            sorted(component, key=order.__getitem__)
            for component in nx.connected_components(drawn)
        ),
        key=lambda nodes: (-len(nodes), order[nodes[0]]),
    )
    positions: dict[str, list[float]] = {}
    x = y = row_height = 0.0
    width = max(1.0, len(components[0]) ** 0.5 * _ROW_WIDTH) if components else 1.0
    for component in components:
        size = len(component) ** 0.5
        if x > 0 and x + size > width:
            x, y = 0.0, y + row_height + _COMPONENT_GAP
            row_height = 0.0
        if len(component) == 1:
            (node,) = component
            local = {node: (0.0, 0.0)}
        else:
            ordered = nx.Graph()
            ordered.add_nodes_from(component)
            ordered.add_edges_from(drawn.edges(component))
            local = nx.spring_layout(ordered, seed=42)
        for node, (px, py) in local.items():
            positions[str(node)] = [
                x + size / 2 + float(px) * size / 2,
                y + size / 2 + float(py) * size / 2,
            ]
        x += size + _COMPONENT_GAP
        row_height = max(row_height, size)
    return positions
