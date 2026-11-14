# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone graph visualization CLI.

Renders visualizations from a visualization-data JSON
(``visualization_data.json``, the index namespace's graph and its
communities, which ingestion writes from the stores at the end of a full
build, ``graph.visualization.export_data``) WITHOUT re-running ingestion:
``--data-path``, else the namespace's (``--suffix``, else the config's)
newest export where ingestion writes them when it shows the build the
stores hold, else one built from the stores into ``--output-directory``
(after an incremental run, which exports nothing). The file
gets a spring layout here unless it carries its own, and the configured
centrality metrics when a requested renderer reads them (``static``). Drives the renderer registry, so any registered renderer
(``--renderers interactive static``) can be produced independently. The
``interactive`` renderer needs the ``viz`` extra; without it a run that
requests it stops before rendering (exit 2) naming the extra. ``static``
needs no extra.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import networkx as nx

from unified_kg_rag.adapters.renderers import (
    RenderContext,
    get_renderer_class,
    registered_renderers,
    render_resolved,
    resolve_renderers,
)
from unified_kg_rag.application.cli.config_args import (
    add_config_arguments,
    exit_on_config_error,
    load_cli_config,
)
from unified_kg_rag.application.cli.conventions import (
    EXIT_FAILURE,
    EXIT_INTERRUPTED,
    EXIT_USAGE,
    ArgumentParser,
    add_output_format_argument,
    exits_on_sigterm,
    failed,
    failure_fields,
    json_output,
    print_json,
    setup_cli_logging,
    usage_error,
)
from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.cli.preflight import missing_extras_error
from unified_kg_rag.application.cli.provider_args import add_provider_arguments
from unified_kg_rag.application.config import load_cli_dotenv
from unified_kg_rag.application.storage.visualization_export import (
    current_visualization_data,
    newest_export,
)
from unified_kg_rag.domain.ingestion.graph_analyzer import (
    CentralityMetrics,
    GraphAnalyzer,
)
from unified_kg_rag.domain.ingestion.visualization_data import (
    DEFAULT_OUTPUTS_DIRECTORY,
    VISUALIZATION_DATA_FILENAME,
    spring_layout,
)
from unified_kg_rag.domain.models import (
    Community,
    HierarchicalCommunity,
    validate_safe_name,
)
from unified_kg_rag.domain.models.config import InteractiveGraphSettings
from unified_kg_rag.domain.retrieval.store_names import resolve_namespace
from unified_kg_rag.shared import (
    ConfigurationError,
    DataProcessingError,
    StoreConnectionError,
    get_logger,
)
from unified_kg_rag.shared.logging import log_error
from unified_kg_rag.shared.registry import Registry

logger = get_logger(__name__)


def _hierarchy_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the community hierarchy list from exported data.

    The export nests community data under ``communities`` as a dict with a
    ``hierarchy`` list (``stored_visualization_data``).
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


class InvalidVisualizationDataError(DataProcessingError):
    """The visualization data file is not exported visualization JSON (exit 2,
    as an invalid ``run-eval`` dataset)."""


def load_render_context(data_path: Path) -> RenderContext:
    """Reconstruct a :class:`RenderContext` from exported visualization JSON.

    Raises:
        InvalidVisualizationDataError: The file is not valid JSON or not
            shaped like an export.
    """
    try:
        return _render_context(json.loads(data_path.read_text(encoding="utf-8")))
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        # ValueError covers invalid JSON and UTF-8 and a failed validation.
        raise InvalidVisualizationDataError(
            f"'{data_path}' is not exported visualization data: {e!r}"
        ) from e


def _render_context(data: dict[str, Any]) -> RenderContext:

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


def _fill_missing_analysis(
    context: RenderContext, config: Any, *, centrality: bool = True
) -> None:
    """Compute what the export (``graph.visualization.export_data``) leaves
    out: a seeded spring layout from the graph topology and, with
    ``centrality``, the configured centrality metrics."""
    if context.graph.number_of_nodes() == 0:
        return
    if not context.layout:
        logger.info(
            "No layout in the data; using a spring layout from the graph topology."
        )
        context.layout = spring_layout(
            context.graph,
            InteractiveGraphSettings.model_validate(
                config.graph.visualization.interactive
            ).max_nodes,
        )
    if centrality and not context.centrality:
        context.centrality = GraphAnalyzer(config, context.graph).calculate_centrality()


def run_visualization(
    data_path: Path,
    output_dir: Path,
    renderer_names: list[str],
    config: Any,
    *,
    failures: dict[str, str] | None = None,
    skipped: dict[str, str] | None = None,
) -> list[Path]:
    """Render the requested renderers from exported data; return written paths.

    Centrality is computed only when a requested renderer reads it
    (``BaseRenderer.needs_centrality``). ``failures``, when given, gets each
    renderer that failed and its error. With ``skipped`` (the defaults, not
    a requested list), a renderer whose dependency is not installed is left
    out with a WARNING and recorded there, unless every one of them is.

    Raises:
        ConfigurationError: A renderer's dependency is not installed (the
            ``viz`` extra), checked before the data is loaded.
    """
    if skipped is None:
        renderers = resolve_renderers(renderer_names)
    else:
        renderers = []
        missing: list[ConfigurationError] = []
        for name in renderer_names:
            try:
                renderers += resolve_renderers([name])
            except ConfigurationError as e:
                missing.append(e)
                skipped[name] = str(e)
        if not renderers and missing:
            raise missing[0]
        for name, reason in skipped.items():
            logger.warning("Skipping the '%s' renderer: %s", name, reason)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = load_render_context(data_path)
    _fill_missing_analysis(
        context,
        config,
        centrality=any(renderer.needs_centrality for _, renderer in renderers),
    )
    written = render_resolved(
        context, output_dir, renderers, config.graph.visualization, failures=failures
    )
    if not written:
        logger.error(
            "No visualization files were rendered from '%s' (empty graph or "
            "every renderer failed).",
            data_path,
        )
    return written


def _suffix(value: str) -> str:
    try:
        return validate_safe_name(value, "--suffix")
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def _namespace_data(config: Any, args: argparse.Namespace) -> Path:
    """The visualization data of ``--suffix``'s namespace (else the
    config's): its newest export under the directories ingestion writes to
    when it shows the build the stores hold, else one built from the stores
    into ``--output-directory`` (``current_visualization_data``). With the
    stores unreachable or not configured, the newest export, with a
    WARNING; exit 2 when the namespace holds no graph."""
    from unified_kg_rag.application.bootstrap import build_providers

    if error := missing_extras_error(config, seeds=False):
        usage_error(error)
    suffix = resolve_namespace(config, args.suffix)
    configured = config.graph.visualization.outputs_directory
    directories = [
        *([Path(configured)] if configured else []),
        Path(DEFAULT_OUTPUTS_DIRECTORY),
        args.cache_directory,
    ]
    target = args.output_directory / VISUALIZATION_DATA_FILENAME
    try:
        path, counts = current_visualization_data(
            build_providers(config), suffix, directories, target
        )
    except (StoreConnectionError, ConfigurationError) as e:
        found = newest_export([*directories, target.parent], suffix)
        if found is None:
            raise
        log_error(
            logger,
            logging.WARNING,
            "The stores of namespace '%s' could not be read: rendering its "
            "newest export, '%s', which may be older than they are (pass "
            "--config-path to read them): %s",
            suffix,
            found,
            error=e,
        )
        return found
    if counts is None:
        logger.info("The export '%s' shows the stores of '%s'", path, suffix)
    elif not counts["nodes"]:
        usage_error(
            f"the index namespace '{suffix}' holds no graph: run run-ingestion "
            "first, check --suffix, or pass --data-path"
        )
    else:
        logger.info(
            "Built the visualization data of namespace '%s' from its stores "
            "(%d nodes, %d edges) into '%s'",
            suffix,
            counts["nodes"],
            counts["edges"],
            path,
        )
    return path


def _requested_renderers(names: list[str]) -> list[str]:
    """``--renderers`` as registered names, each once, checked against the
    registry once the plugins are loaded (a plugin's renderer is one, and a
    name in any case, the registry rule); exit 2 naming an unknown one."""
    requested: list[str] = []
    for name in names:
        try:
            get_renderer_class(name)
        except ValueError as e:
            usage_error(f"--renderers: {e}")
        key = Registry.normalize(name)
        if key not in requested:
            requested.append(key)
    return requested


def _build_parser() -> argparse.ArgumentParser:
    parser = ArgumentParser(
        allow_abbrev=False,
        description="Render graph visualizations from exported data (no ingestion).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=doc_epilog(
            "guides/visualize.md#examples",
            "run-visualization --data-path visualization_data.json "
            "--output-directory ./viz",
        ),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to exported visualization-data JSON (default: --suffix's "
        "namespace's newest visualization_data.json under "
        "graph.visualization.outputs_directory, outputs/visualization and "
        "--cache-directory when it shows the build the stores hold, else "
        "one built from the stores into --output-directory)",
    )
    parser.add_argument(
        "--suffix",
        type=_suffix,
        default=None,
        help="Index namespace to render without --data-path (the name "
        "run-ingestion --suffix wrote it under; default: the config's "
        "index_value, else 'default')",
    )
    parser.add_argument(
        "--cache-directory",
        type=Path,
        default=Path("cache"),
        help="The run-ingestion cache directory searched for an export "
        "without --data-path (default: cache)",
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
        default=None,
        # Checked once the plugins are loaded (_requested_renderers), before
        # the data is loaded and centrality computed.
        metavar="RENDERER",
        help=f"Renderers to run, any case: {', '.join(registered_renderers())} "
        "or one a plugin registers (default: all of them, leaving out with a "
        "warning one whose extra is not installed; a renderer named here "
        "needs its extra)",
    )
    add_config_arguments(parser)
    add_output_format_argument(parser)
    add_provider_arguments(parser)
    return parser


@exits_on_sigterm
def main() -> int:
    load_cli_dotenv()
    args = _build_parser().parse_args()
    as_json = json_output(args)
    try:
        with exit_on_config_error():
            config = load_cli_config(args)
        setup_cli_logging(config, args)
        if args.renderers is not None:
            args.renderers = _requested_renderers(args.renderers)
        if args.data_path is None:
            args.data_path = _namespace_data(config, args)
        if not args.data_path.is_file():
            usage_error(f"visualization data not found: '{args.data_path}'")
        logger.info("Rendering '%s'", args.data_path)
        failures: dict[str, str] = {}
        # Only the default list leaves out a renderer it cannot run.
        skipped: dict[str, str] | None = {} if args.renderers is None else None
        written = run_visualization(
            args.data_path,
            args.output_directory,
            args.renderers or list(registered_renderers()),
            config,
            failures=failures,
            skipped=skipped,
        )
        logger.info(
            "Visualization complete: %d files in '%s'",
            len(written),
            args.output_directory,
        )
        error = None
        if not written:
            error = f"No visualization files were rendered from '{args.data_path}'"
        elif failures:
            # The others rendered, but a requested output is missing.
            error = f"Renderer(s) failed: {', '.join(sorted(failures))}"
        if as_json:
            print_json(
                {
                    # The data rendered (an export, or built from the stores).
                    "data_path": str(args.data_path),
                    "output_directory": str(args.output_directory),
                    "files": [str(path) for path in written],
                    "failed_renderers": failures,
                    "skipped_renderers": skipped or {},
                    **(failure_fields(EXIT_FAILURE, error) if error else {}),
                }
            )
        if error:
            return failed(EXIT_FAILURE, error)
        return 0
    except KeyboardInterrupt:
        logger.warning("Visualization interrupted by user.")
        # Raised, not returned: exits_on_sigterm prints the JSON status.
        raise SystemExit(EXIT_INTERRUPTED) from None
    except (ConfigurationError, InvalidVisualizationDataError) as e:
        log_error(logger, logging.ERROR, "Visualization input error: %s", error=e)
        return failed(EXIT_USAGE, e)
    except Exception as e:  # noqa: BLE001 - logged, exit code
        # A clean error line rather than a raw traceback, by the one rule of
        # every surface (log_error): a store that could not be reached with
        # its text, an unexpected failure's type and where it happened (text
        # and traceback at DEBUG).
        log_error(logger, logging.ERROR, "Visualization failed: %s", error=e)
        return failed(EXIT_FAILURE, e)


if __name__ == "__main__":
    sys.exit(main())
