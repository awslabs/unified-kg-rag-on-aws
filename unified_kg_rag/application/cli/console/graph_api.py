# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Graph page's server side: the ``visualization_data.json`` files the
ingestion exported, read into what the page draws (nodes, edges,
communities, a seeded layout) and the analysis it lists (degrees,
communities, schema conformance, seed versus document evidence).

A file is read once per version (path, modification time, size): the page
gets the drawn entities and the edges between them in full and every other
entity as ``{id, name, degree}`` (search and the table), and asks for one
of those by id (:func:`entity`).

An ingestion job exports nothing of its own (an incremental run writes no
graph): the page, once opened, gets the namespace's graph as
``run-visualization`` does (:func:`current_graph`), rebuilt from the stores
only when no export shows the build they hold (:class:`GraphBuilder`)."""

from __future__ import annotations

import json
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

from unified_kg_rag.application.cli.console.paths import Roots, walk_files
from unified_kg_rag.application.storage.visualization_export import (
    export_build_time,
    export_namespace,
)
from unified_kg_rag.domain.ingestion.visualization_data import (
    VISUALIZATION_DATA_FILENAME,
    displayed_graph,
)
from unified_kg_rag.domain.retrieval import source_fields
from unified_kg_rag.domain.retrieval.store_names import resolve_namespace
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import DataProcessingError

if TYPE_CHECKING:
    from unified_kg_rag.application.bootstrap import Providers
    from unified_kg_rag.domain.models import Config

logger = get_logger(__name__)

# The node attributes the page reads; the rest stay on the server.
MAX_DESCRIPTION = 1200
# Nodes the page lays out and draws at once (the rest are listed and found
# by search); the layout is quadratic in the drawn nodes.
MAX_DRAWN_NODES = 600
MAX_FILE_BYTES = 200 * 1024 * 1024
# Read graphs kept (the newest few files the page switches between).
CACHED_GRAPHS = 4


# Where the page's on-demand graph of a namespace goes, under the cache
# directory (one file per namespace, labelled with it).
ON_DEMAND_DIRECTORY = "console-graphs"


class GraphDataError(DataProcessingError):
    """The file is not exported visualization data."""


@dataclass(frozen=True)
class CurrentGraph:
    """The Graph page's graph of the config's index namespace
    (:func:`current_graph`). ``state``: ``current`` (``path`` shows what the
    stores hold, from the build that finished at ``built_at``),
    ``export_off`` (``graph.visualization.export_data`` off: nothing is
    built) or ``unrecorded`` (no build record: nothing ingested, or indexed
    by a release before build records, whose graph cannot be told current);
    ``failed`` when building it raised (:class:`GraphBuilder`), with the
    build's ``built_at`` once the build record was read, so the exports
    older than it are still marked outdated."""

    state: str
    namespace: str | None
    built_at: str | None = None
    path: Path | None = None

    def view(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "namespace": self.namespace,
            "built_at": self.built_at,
        }


def current_graph(
    config: Config,
    cache_directory: Path,
    providers: Callable[[], Providers] | None = None,
) -> CurrentGraph:
    """The visualization data of the config's index namespace that shows
    what its stores hold (``current_visualization_data``, the builder
    ``run-visualization`` uses): the newest export under the cache (and
    ``graph.visualization.outputs_directory``) that names the build its
    build record names, else one built from the stores into
    ``graph.visualization.outputs_directory``, else
    ``<cache>/console-graphs/<namespace>/``. Nothing is built (nor a
    provider bundle) with ``graph.visualization.export_data`` off or no
    build recorded (an empty namespace). ``providers`` gives the bundle to
    read the stores with (default: one built for ``config``). A build that
    raises is ``failed`` (logged), with the recorded ``built_at``.

    Raises:
        ConfigurationError: The vector store cannot page its items.
        StoreConnectionError: A store cannot be reached.
    """
    from unified_kg_rag.application.bootstrap import build_providers
    from unified_kg_rag.application.storage.visualization_export import (
        current_visualization_data,
        stored_export,
    )

    suffix = resolve_namespace(config, None)
    if not config.graph.visualization.export_data:
        return CurrentGraph("export_off", suffix)
    bundle = providers() if providers is not None else build_providers(config)
    built_at, _ = stored_export(bundle, suffix)
    if built_at is None:
        return CurrentGraph("unrecorded", suffix)
    configured = config.graph.visualization.outputs_directory
    directory = (
        Path(configured)
        if configured
        else cache_directory / ON_DEMAND_DIRECTORY / suffix
    )
    try:
        path, _ = current_visualization_data(
            bundle,
            suffix,
            [cache_directory, *([Path(configured)] if configured else [])],
            directory / VISUALIZATION_DATA_FILENAME,
        )
    except Exception as e:  # noqa: BLE001 - the page lists the exports there are
        logger.warning("Building the graph of '%s' failed: %s", suffix, e)
        return CurrentGraph("failed", suffix, built_at)
    return CurrentGraph("current", suffix, built_at, path)


class GraphBuilder:
    """Brings the Graph page's graph up to date when the page opens
    (:func:`current_graph`), one build at a time, reading the stores with
    one provider bundle per config: the console keeps its config object
    until Settings saves another, so opening the page again builds none."""

    def __init__(self, cache_directory: Path) -> None:
        self.cache_directory = cache_directory
        self._lock = threading.Lock()
        self._providers: tuple[Config, Providers] | None = None

    def current(self, load_config: Callable[[], Config]) -> CurrentGraph:
        """:func:`current_graph` of the config ``load_config`` reads;
        ``failed`` (logged) when either raises, the page then listing the
        exports there are."""
        with self._lock:
            namespace = None
            try:
                config = load_config()
                namespace = resolve_namespace(config, None)
                return current_graph(
                    config, self.cache_directory, lambda: self._bundle(config)
                )
            except Exception as e:  # noqa: BLE001 - the page still lists the exports
                logger.warning("Building the graph for the Graph page failed: %s", e)
                return CurrentGraph("failed", namespace)

    def _bundle(self, config: Config) -> Providers:
        from unified_kg_rag.application.bootstrap import build_providers

        if self._providers is None or self._providers[0] is not config:
            self._providers = (config, build_providers(config))
        return self._providers[1]


def mark_outdated(graphs: list[dict[str, Any]], current: CurrentGraph) -> None:
    """Mark each of ``graphs`` (:func:`find_graphs`) ``outdated`` when a
    later build of its namespace exists: the build record's, for the
    config's namespace once it is known, else the latest an export of the
    namespace names. An export that names no namespace is never marked."""
    latest: dict[str, str] = {}
    for g in graphs:
        if g["namespace"] and g["built_at"]:
            latest[g["namespace"]] = max(latest.get(g["namespace"], ""), g["built_at"])
    if current.namespace is not None and current.built_at is not None:
        latest[current.namespace] = current.built_at
    for g in graphs:
        build = latest.get(g["namespace"]) if g["namespace"] else None
        g["outdated"] = build is not None and g["built_at"] != build


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
            # An export holds its namespace's whole graph: named by it, the
            # run that wrote it beside (an export of an earlier release
            # names no namespace).
            namespace = export_namespace(path)
            run = _label(path)
            found[path] = {
                "ref": ref,
                "label": namespace or run,
                "namespace": namespace,
                "run": run,
                "built_at": export_build_time(path),
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


@dataclass(frozen=True)
class LoadedGraph:
    """One version of an exported file: what the page draws (``page``,
    encoded once) and every entity and edge, for :func:`entity`."""

    page: dict[str, Any]
    encoded: bytes
    nodes: dict[str, dict[str, Any]]
    edges: list[dict[str, Any]]


_cache: OrderedDict[tuple[str, int, int], LoadedGraph] = OrderedDict()
_cache_lock = threading.Lock()


def cached_graph(path: Path) -> LoadedGraph:
    """The graph of an exported file, read again only when the file changed
    (its modification time or size). Its ``page`` is what the page draws:
    the drawn entities (``nodes``, at most :data:`MAX_DRAWN_NODES`, with
    their ``layout``) and the edges between them in full, every other
    entity as ``{id, name, degree}`` (``others``, most connected first), and
    the communities and analysis of the whole graph.

    Raises:
        GraphDataError: The file is too large, not JSON, or not shaped like
            an export.
    """
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    loaded = _read_graph(path)
    with _cache_lock:
        _cache[key] = loaded
        while len(_cache) > CACHED_GRAPHS:
            _cache.popitem(last=False)
    return loaded


def entity(path: Path, entity_id: str) -> dict[str, Any] | None:
    """An entity of the graph of ``path`` (drawn or not), every edge that
    touches it and the names of the entities at their other ends; None for
    an id the graph lacks."""
    loaded = cached_graph(path)
    node = loaded.nodes.get(entity_id)
    if node is None:
        return None
    edges = [e for e in loaded.edges if entity_id in (e["source"], e["target"])]
    others = {e["target"] if e["source"] == entity_id else e["source"] for e in edges}
    return {
        "node": node,
        "edges": edges,
        "names": {
            other: loaded.nodes[other]["name"]
            for other in others
            if other in loaded.nodes
        },
    }


def _read_graph(path: Path) -> LoadedGraph:
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
    by_id = {n["id"]: n for n in nodes}
    communities = _communities(data.get("communities"), by_id)
    page = {
        "nodes": [n for n in nodes if n["id"] in layout],
        "others": [
            {"id": n["id"], "name": n["name"], "degree": n["degree"]}
            for n in sorted(nodes, key=lambda n: -n["degree"])
            if n["id"] not in layout
        ],
        "edges": [e for e in edges if e["source"] in layout and e["target"] in layout],
        "layout": layout,
        "communities": communities,
        "stats": _stats(graph, nodes, edges, communities),
    }
    encoded = json.dumps(page, ensure_ascii=False, separators=(",", ":")).encode()
    return LoadedGraph(page=page, encoded=encoded, nodes=by_id, edges=edges)


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
