# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Graph page's server side: the ``visualization_data.json`` files the
ingestion exported, read into what the page draws (nodes, edges,
communities, a seeded layout) and the analysis it lists (degrees,
communities, schema conformance, seed versus document evidence)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx

from unified_kg_rag.application.cli.console.paths import Roots, walk_files
from unified_kg_rag.domain.ingestion.visualization_data import (
    VISUALIZATION_DATA_FILENAME,
    displayed_graph,
)
from unified_kg_rag.domain.retrieval import source_fields
from unified_kg_rag.shared.exceptions import DataProcessingError

# The node attributes the page reads; the rest stay on the server.
MAX_DESCRIPTION = 1200
# Nodes the page lays out and draws at once (the rest are listed and found
# by search); the layout is quadratic in the drawn nodes.
MAX_DRAWN_NODES = 600
MAX_FILE_BYTES = 200 * 1024 * 1024


class GraphDataError(DataProcessingError):
    """The file is not exported visualization data."""


def find_graphs(roots: Roots, names: tuple[str, ...]) -> list[dict[str, Any]]:
    """Every ``visualization_data.json`` under the roots named ``names``,
    newest first."""
    found: dict[Path, dict[str, Any]] = {}
    for name in names:
        root = roots.roots.get(name)
        if root is None or not root.path.is_dir():
            continue
        for path in walk_files(root.path, limit=20000):
            if path.name != VISUALIZATION_DATA_FILENAME or path in found:
                continue
            ref = roots.ref(path)
            if ref is None:
                continue
            stat = path.stat()
            found[path] = {
                "ref": ref,
                "label": _label(path),
                "modified": stat.st_mtime,
                "bytes": stat.st_size,
            }
    return sorted(found.values(), key=lambda g: g["modified"], reverse=True)


def _label(path: Path) -> str:
    """``<pipeline id>`` for ``<cache>/<pipeline id>/visualization/...``."""
    parent = path.parent
    if parent.name == "visualization" and parent.parent.name:
        return parent.parent.name
    return parent.name or path.name


def _text(value: Any, limit: int = MAX_DESCRIPTION) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def load_graph(path: Path) -> dict[str, Any]:
    """The graph of an exported file, as the page draws it.

    Raises:
        GraphDataError: The file is too large, not JSON, or not shaped like
            an export.
    """
    if path.stat().st_size > MAX_FILE_BYTES:
        raise GraphDataError(f"'{path.name}' is over {MAX_FILE_BYTES // 2**20} MiB")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_nodes = data["nodes"]
        raw_edges = data["edges"]
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise TypeError("nodes and edges must be lists")
    except (ValueError, KeyError, TypeError) as e:
        raise GraphDataError(
            f"'{path.name}' is not exported visualization data: {e}"
        ) from e

    graph = nx.Graph()
    nodes: list[dict[str, Any]] = []
    for raw in raw_nodes:
        attrs = raw.get("attributes") or {}
        node_id = str(raw.get("id"))
        graph.add_node(node_id)
        nodes.append(
            {
                "id": node_id,
                "name": str(attrs.get("name") or attrs.get("title") or node_id),
                "type": attrs.get("type"),
                "description": _text(attrs.get("description")),
                "community_id": attrs.get("community_id"),
                "text_unit_ids": [str(t) for t in attrs.get("text_unit_ids") or []],
                "source_kinds": source_fields.id_list(attrs.get("source_kinds")),
                "iri": attrs.get("iri"),
                "type_iri": attrs.get("type_iri"),
                "in_schema": attrs.get("in_schema"),
                "node_type": attrs.get("node_type"),
            }
        )
    edges: list[dict[str, Any]] = []
    for raw in raw_edges:
        attrs = raw.get("attributes") or {}
        source, target = str(raw.get("source")), str(raw.get("target"))
        graph.add_edge(source, target)
        edges.append(
            {
                "id": str(attrs.get("id") or f"{source}->{target}"),
                "source": source,
                "target": target,
                "type": attrs.get("type"),
                "description": _text(attrs.get("description"), 600),
                "weight": attrs.get("weight"),
                "text_unit_ids": [str(t) for t in attrs.get("text_unit_ids") or []],
                "source_kinds": source_fields.id_list(attrs.get("source_kinds")),
                "in_schema": attrs.get("in_schema"),
                "predicate_iri": attrs.get("predicate_iri"),
            }
        )
    degrees = dict(graph.degree())
    for node in nodes:
        node["degree"] = degrees.get(node["id"], 0)
    # The page draws the best-connected MAX_DRAWN_NODES, at an exported
    # layout's positions when it has them (an export may lay out more).
    drawn = displayed_graph(graph, MAX_DRAWN_NODES)
    exported = data.get("layout") or {}
    layout = {k: v for k, v in exported.items() if k in drawn}
    if not layout and drawn.number_of_nodes():
        layout = packed_layout(drawn)
    communities = _communities(data.get("communities"), {n["id"]: n for n in nodes})
    return {
        "nodes": nodes,
        "edges": edges,
        "layout": layout,
        "communities": communities,
        "stats": _stats(graph, nodes, edges, communities),
    }


def packed_layout(graph: nx.Graph) -> dict[str, list[float]]:
    """A seeded spring layout of each connected component, the components
    packed in rows largest first, so small islands do not push the main
    component into a corner (one spring layout of a disconnected graph
    flings its islands far apart)."""
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    positions: dict[str, list[float]] = {}
    x = y = row_height = 0.0
    width = max(1.0, len(components[0]) ** 0.5 * 2.2) if components else 1.0
    for component in components:
        size = len(component) ** 0.5
        if x > 0 and x + size > width:
            x, y = 0.0, y + row_height + 0.4
            row_height = 0.0
        if len(component) == 1:
            (node,) = component
            local = {node: (0.0, 0.0)}
        else:
            local = nx.spring_layout(graph.subgraph(component), seed=42)
        for node, (px, py) in local.items():
            positions[str(node)] = [
                x + size / 2 + px * size / 2,
                y + size / 2 + py * size / 2,
            ]
        x += size + 0.4
        row_height = max(row_height, size)
    return positions


def _communities(raw: Any, nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    hierarchy = raw.get("hierarchy") if isinstance(raw, dict) else raw
    found = []
    for entry in hierarchy or []:
        if not isinstance(entry, dict):
            continue
        members = [str(n) for n in entry.get("nodes") or []]
        top = sorted(
            (nodes[m] for m in members if m in nodes), key=lambda n: -n["degree"]
        )[:5]
        found.append(
            {
                "id": str(entry.get("community_id")),
                "level": int(entry.get("level") or 0),
                "parent": entry.get("parent"),
                "children": list(entry.get("children") or []),
                "size": int(entry.get("size") or len(members)),
                "nodes": members,
                "top_entities": [n["name"] for n in top],
            }
        )
    return sorted(found, key=lambda c: (c["level"], -c["size"]))


def _stats(
    graph: nx.Graph,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    communities: list[dict[str, Any]],
) -> dict[str, Any]:
    def evidence(item: dict[str, Any]) -> str:
        kinds = set(item["source_kinds"])
        if "seed_graph" in kinds and "document" in kinds:
            return "both"
        if "seed_graph" in kinds:
            return "seed_graph"
        return "document"

    def conformance(items: list[dict[str, Any]]) -> dict[str, int]:
        counts = Counter(
            (
                "in_schema"
                if i["in_schema"] is True
                else "off_schema" if i["in_schema"] is False else "unannotated"
            )
            for i in items
        )
        return {k: counts.get(k, 0) for k in ("in_schema", "off_schema", "unannotated")}

    degrees = sorted((n["degree"] for n in nodes), reverse=True)
    histogram = Counter(degrees)
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "components": (
            nx.number_connected_components(graph) if graph.number_of_nodes() else 0
        ),
        "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0.0,
        "mean_degree": (sum(degrees) / len(degrees)) if degrees else 0.0,
        "max_degree": degrees[0] if degrees else 0,
        "degree_histogram": sorted(histogram.items()),
        "top_entities": [
            {"id": n["id"], "name": n["name"], "degree": n["degree"], "type": n["type"]}
            for n in sorted(nodes, key=lambda n: -n["degree"])[:10]
        ],
        "types": Counter(str(n["type"] or "unknown") for n in nodes).most_common(),
        "relation_types": Counter(
            str(e["type"] or "unknown") for e in edges
        ).most_common(),
        "communities_by_level": sorted(
            Counter(c["level"] for c in communities).items()
        ),
        "schema": {"entities": conformance(nodes), "relationships": conformance(edges)},
        "evidence": {
            "entities": dict(Counter(evidence(n) for n in nodes)),
            "relationships": dict(Counter(evidence(e) for e in edges)),
        },
    }
