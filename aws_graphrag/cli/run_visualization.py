# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Standalone graph visualization CLI.

Renders visualizations from a previously exported visualization-data JSON
(produced by ``GraphVisualizationManager.export_visualization_data``) WITHOUT
re-running ingestion. Drives the renderer registry, so any registered renderer
(``--renderers interactive static``) can be produced independently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx

from aws_graphrag.core import get_config, get_logger
from aws_graphrag.ingestion.community_detector import HierarchicalCommunity
from aws_graphrag.models import Community
from aws_graphrag.visualization.renderers import (
    RenderContext,
    get_renderer_class,
    registered_renderers,
)

logger = get_logger(__name__)


def _hierarchy_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the community hierarchy list from exported data.

    ``export_visualization_data`` nests community data under ``communities`` as
    a dict with a ``hierarchy`` list (see CommunityDetector.export_community_data).
    """
    communities = data.get("communities", [])
    if isinstance(communities, dict):
        hierarchy = communities.get("hierarchy", [])
        return list(hierarchy) if isinstance(hierarchy, list) else []
    return list(communities) if isinstance(communities, list) else []


def _to_hierarchical_communities(
    entries: list[dict[str, Any]],
) -> list[HierarchicalCommunity]:
    return [
        HierarchicalCommunity.model_validate(
            {
                "community_id": str(entry.get("community_id", "")),
                "level": int(entry.get("level", 0)),
                "nodes": set(entry.get("nodes", []) or []),
                "parent_id": entry.get("parent"),
                "children_ids": list(entry.get("children", []) or []),
            }
        )
        for entry in entries
    ]


def _to_communities(entries: list[dict[str, Any]]) -> list[Community]:
    communities: list[Community] = []
    for entry in entries:
        nodes = entry.get("nodes", []) or []
        size = entry.get("size")
        communities.append(
            Community.model_validate(
                {
                    "id": str(entry.get("community_id", "")),
                    "name": str(entry.get("community_id", "")),
                    "level": str(entry.get("level", 0)),
                    "parent": entry.get("parent") or "",
                    "children": list(entry.get("children", []) or []),
                    "size": size if size is not None else len(nodes),
                }
            )
        )
    return communities


def load_render_context(data_path: Path) -> RenderContext:
    """Reconstruct a :class:`RenderContext` from exported visualization JSON."""
    data = json.loads(data_path.read_text(encoding="utf-8"))

    graph = nx.Graph()
    for node in data.get("nodes", []):
        graph.add_node(str(node["id"]), **(node.get("attributes") or {}))
    for edge in data.get("edges", []):
        graph.add_edge(
            str(edge["source"]), str(edge["target"]), **(edge.get("attributes") or {})
        )

    entries = _hierarchy_entries(data)
    return RenderContext(
        graph=graph,
        layout=data.get("layout", {}),
        communities=_to_communities(entries),
        community_hierarchy=_to_hierarchical_communities(entries),
        centrality=data.get("centrality", {}),
    )


def run_visualization(
    data_path: Path, output_dir: Path, renderer_names: list[str], config: Any
) -> list[Path]:
    """Render the requested renderers from exported data; return written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    context = load_render_context(data_path)
    viz_config = config.graph.visualization

    written: list[Path] = []
    for name in renderer_names:
        renderer_cls = get_renderer_class(name)
        renderer_config = (
            viz_config.interactive if name == "interactive" else viz_config.static
        )
        try:
            paths = renderer_cls(renderer_config).render(context, output_dir)
            written.extend(paths)
            logger.info("Renderer '%s' wrote %d file(s)", name, len(paths))
        except Exception as e:
            logger.error("Renderer '%s' failed: %s", name, e)

    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render graph visualizations from exported data (no ingestion)."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        type=Path,
        help="Path to exported visualization-data JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualization_outputs"),
        help="Directory to write visualization files",
    )
    parser.add_argument(
        "--renderers",
        nargs="+",
        default=list(registered_renderers()),
        help=f"Renderers to run. Available: {', '.join(registered_renderers())}",
    )
    parser.add_argument(
        "--config-path", type=str, default=None, help="Config YAML path"
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = get_config(args.config_path)
    written = run_visualization(args.data_path, args.output_dir, args.renderers, config)
    logger.info(
        "Visualization complete: %d files in '%s'", len(written), args.output_dir
    )
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
