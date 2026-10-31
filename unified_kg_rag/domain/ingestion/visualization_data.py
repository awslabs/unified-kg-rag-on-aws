# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The visualization-data JSON that ``run-visualization`` renders from.

Ingestion writes it after community detection (``graph.visualization.
export_data``): the graph and the community hierarchy, built here in one pass
over the nodes and edges (no embedding or model call, no centrality, no
layout). ``run-visualization`` computes the layout and centrality when it
renders; a file that carries them (written by an earlier release) is drawn
with its own.

The layout helpers (:func:`displayed_graph`, :func:`spring_layout`) are
renderer-neutral: ``run-visualization`` lays out the nodes the interactive
page will draw without depending on it.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from unified_kg_rag.shared.utils import strip_embedding_fields

__all__ = [
    "DEFAULT_OUTPUTS_DIRECTORY",
    "VISUALIZATION_DATA_FILENAME",
    "displayed_graph",
    "graph_data",
    "spring_layout",
    "strip_heavy_attributes",
]

# File written into the visualization outputs directory; it is the input
# ``run-visualization --data-path`` re-renders from without ingestion.
VISUALIZATION_DATA_FILENAME = "visualization_data.json"
DEFAULT_OUTPUTS_DIRECTORY = "outputs/visualization"


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
    """Nodes, edges and ``communities`` (``export_community_data`` output),
    without a layout or centrality.

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


def displayed_graph(graph: nx.Graph, max_nodes: int | None) -> nx.Graph:
    """The part of ``graph`` the interactive page draws: its top ``max_nodes``
    by degree (``graph.visualization.interactive.max_nodes``; None or 0 draws
    every node)."""
    if not max_nodes or graph.number_of_nodes() <= max_nodes:
        return graph
    ranked = sorted(graph.degree(), key=lambda nd: nd[1], reverse=True)
    return graph.subgraph(node for node, _ in ranked[:max_nodes])


def spring_layout(graph: nx.Graph, max_nodes: int | None) -> dict[str, list[float]]:
    """Seeded spring layout of the drawn nodes only (:func:`displayed_graph`).

    A spring layout is quadratic in the node count, so laying out a large
    graph and then drawing its top ``max_nodes`` spent minutes on nodes
    the page never shows. A graph within the cap is laid out whole.
    """
    positions = nx.spring_layout(displayed_graph(graph, max_nodes), seed=42)
    return {str(node): pos.tolist() for node, pos in positions.items()}
