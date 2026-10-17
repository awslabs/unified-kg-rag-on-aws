# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone graph visualization CLI.

Renders visualizations from a previously exported visualization-data JSON
(``visualization_data.json``, written into ``graph.visualization.outputs_directory``
during ingestion: by ``GraphVisualizationManager.run`` with a layout and
centrality when ``graph.visualization.enabled``, else as graph and communities
only, ``export_data``) WITHOUT re-running ingestion. A data-only file gets a
spring layout and the configured centrality metrics here. Drives the renderer registry, so any registered renderer
(``--renderers interactive static``) can be produced independently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx

from unified_kg_rag.adapters.renderers import (
    RenderContext,
    registered_renderers,
    render_all,
)
from unified_kg_rag.adapters.renderers.interactive import InteractiveRenderer
from unified_kg_rag.application.cli.config_args import (
    config_path_arg,
    exit_on_config_error,
)
from unified_kg_rag.application.cli.conventions import (
    EXIT_FAILURE,
    EXIT_INTERRUPTED,
    add_output_format_argument,
    exits_on_sigterm,
    json_output,
    print_json,
    usage_error,
)
from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.cli.provider_args import (
    add_provider_arguments,
    apply_provider_arguments,
)
from unified_kg_rag.domain.ingestion.graph_analyzer import (
    CentralityMetrics,
    GraphAnalyzer,
)
from unified_kg_rag.domain.models import Community, HierarchicalCommunity
from unified_kg_rag.shared import get_config, get_logger, setup_logging
from unified_kg_rag.shared.config import load_cli_dotenv

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
    centrality = {
        node_id: CentralityMetrics(**metrics)
        for node_id, metrics in (data.get("centrality") or {}).items()
        if isinstance(metrics, dict)
    }
    return RenderContext(
        graph=graph,
        layout=data.get("layout", {}),
        communities=_to_communities(entries),
        community_hierarchy=_to_hierarchical_communities(entries),
        centrality=centrality,
    )


def _fill_missing_analysis(context: RenderContext, config: Any) -> None:
    """Compute what a data-only export (``graph.visualization.export_data``)
    leaves out: a seeded spring layout from the graph topology and the
    configured centrality metrics."""
    if context.graph.number_of_nodes() == 0:
        return
    if not context.layout:
        logger.info(
            "No layout in the data (exported without rendering); using a spring "
            "layout from the graph topology."
        )
        context.layout = InteractiveRenderer(
            config.graph.visualization.interactive or {}
        ).spring_layout(context.graph)
    if not context.centrality:
        context.centrality = GraphAnalyzer(config, context.graph).calculate_centrality()


def run_visualization(
    data_path: Path, output_dir: Path, renderer_names: list[str], config: Any
) -> list[Path]:
    """Render the requested renderers from exported data; return written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    context = load_render_context(data_path)
    _fill_missing_analysis(context, config)
    written = render_all(
        context, output_dir, renderer_names, config.graph.visualization
    )
    if not written:
        logger.error(
            "No visualization files were rendered from '%s' (empty graph or "
            "every renderer failed).",
            data_path,
        )
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Render graph visualizations from exported data (no ingestion).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=doc_epilog(
            "7-visualization-run-visualization",
            "run-visualization --data-path visualization_data.json "
            "--output-directory ./viz",
        ),
    )
    parser.add_argument(
        "--data-path",
        required=True,
        type=Path,
        help="Path to exported visualization-data JSON",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("visualization_outputs"),
        help="Directory to write visualization files",
    )
    parser.add_argument(
        "--renderers",
        nargs="+",
        default=list(registered_renderers()),
        # Rejected here, before the data is loaded and centrality computed.
        choices=registered_renderers(),
        metavar="RENDERER",
        help=f"Renderers to run. Available: {', '.join(registered_renderers())}",
    )
    parser.add_argument(
        "--config-path", type=config_path_arg, default=None, help="Config YAML path"
    )
    add_output_format_argument(parser)
    # Accepted like on the other CLIs, so one command line (and a config of
    # either backend) serves them all; rendering makes no model call.
    add_provider_arguments(parser)
    return parser


@exits_on_sigterm
def main() -> int:
    load_cli_dotenv()
    args = _build_parser().parse_args()
    as_json = json_output(args)
    try:
        with exit_on_config_error():
            config = apply_provider_arguments(get_config(args.config_path), args)
        if not args.data_path.is_file():
            usage_error(f"visualization data not found: '{args.data_path}'")
        setup_logging(config, stream=sys.stderr if as_json else None)
        written = run_visualization(
            args.data_path, args.output_directory, args.renderers, config
        )
        logger.info(
            "Visualization complete: %d files in '%s'",
            len(written),
            args.output_directory,
        )
        if as_json:
            print_json(
                {
                    "output_directory": str(args.output_directory),
                    "files": [str(path) for path in written],
                }
            )
        return 0 if written else EXIT_FAILURE
    except KeyboardInterrupt:
        logger.warning("Visualization interrupted by user.")
        return EXIT_INTERRUPTED
    except Exception as e:
        # Clean error line rather than a raw traceback (parity with the other
        # CLIs); full detail goes to the log at debug level.
        logger.error("Visualization failed: %s", e)
        logger.debug("Traceback:", exc_info=True)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
