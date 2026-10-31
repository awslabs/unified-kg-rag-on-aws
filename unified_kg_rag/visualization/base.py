# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from typing import Any

from unified_kg_rag.adapters.ingestion.community_detector import CommunityDetector
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.renderers import (
    RenderContext,
    registered_renderers,
    render_all,
)
from unified_kg_rag.domain.ingestion.graph_analyzer import GraphAnalyzer
from unified_kg_rag.domain.ingestion.visualization_data import (
    DEFAULT_OUTPUTS_DIRECTORY,
    VISUALIZATION_DATA_FILENAME,
    displayed_graph,
    spring_layout,
    strip_heavy_attributes,
)
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import InteractiveGraphSettings
from unified_kg_rag.shared import get_logger

from .embeddings.dimensionality import DimensionalityReducer
from .embeddings.node_embedder import BedrockNodeEmbedder
from .exporters.html_exporter import HTMLExporter

logger = get_logger(__name__)

# Re-exported: callers import them from here as well.
__all__ = [
    "DEFAULT_OUTPUTS_DIRECTORY",
    "VISUALIZATION_DATA_FILENAME",
    "GraphVisualizationManager",
]


class GraphVisualizationManager:
    def __init__(
        self,
        config: Config,
        graph_analyzer: GraphAnalyzer,
        community_detector: CommunityDetector,
        outputs_dir: Path | None = None,
        embedder: BedrockNodeEmbedder | None = None,
        *,
        providers: Providers | None = None,
    ) -> None:
        self.config = config
        self.viz_config = self.config.graph.visualization
        self.analyzer = graph_analyzer
        self.community_detector = community_detector
        self.outputs_dir = outputs_dir or Path(
            self.viz_config.outputs_directory or DEFAULT_OUTPUTS_DIRECTORY
        )
        self.providers = Providers.resolve(config, providers)

        # Built lazily: with embedding_method == "none" no Bedrock embedding
        # client is ever constructed.
        self._embedder = embedder

        self.reducer = DimensionalityReducer(self.viz_config.layout)
        self.html_exporter = HTMLExporter()
        # Set by _generate_layout when the embedding-based layout could not be
        # produced and a topology-only spring layout was used instead.
        self.layout_degraded = False

    @property
    def embedder(self) -> BedrockNodeEmbedder:
        if self._embedder is None:
            self._embedder = BedrockNodeEmbedder(
                self.config, embedding_factory=self.providers.embedding_factory
            )
        return self._embedder

    def run(self) -> None:
        # ``graph.visualization.enabled`` gates the ingestion call site; a
        # caller that builds the manager directly asked for the output.
        if not self.analyzer.graph:
            logger.warning("No graph is available for visualization. Skipping.")
            return

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Starting comprehensive visualization report creation in '%s'.",
            self.outputs_dir,
        )

        self.analyzer.set_community_data(
            self.community_detector.node_to_community_l0,
            self.community_detector.get_community_metrics(),
        )

        layout = self._generate_layout()

        written = self._generate_visualizations(self.outputs_dir, layout)
        self._export_summary_report(self.outputs_dir, written)
        self.export_visualization_data(
            self.outputs_dir / VISUALIZATION_DATA_FILENAME, layout=layout
        )

        if self.layout_degraded:
            logger.warning(
                "Visualization report created with a DEGRADED layout (node "
                "embeddings or their reduction unavailable; spring layout used "
                "instead)."
            )
        else:
            logger.info("Comprehensive visualization report created successfully.")

    def _generate_layout(self) -> dict[str, Any]:
        self.layout_degraded = False
        if not self.analyzer.graph or self.viz_config.embedding_method == "none":
            logger.info(
                "Skipping embedding generation. Using spring layout as fallback."
            )
            if self.analyzer.graph:
                return self._spring_layout()
            return {}

        # Only the interactive page uses positions, and it draws its top
        # max_nodes; embed just those (one embedding call per node).
        embeddings = self.embedder.generate_embeddings(
            displayed_graph(self.analyzer.graph, self._max_drawn_nodes())
        )
        # Failed generation yields no embeddings (never random substitutes).
        if not embeddings.embeddings:
            return self._degraded_layout("node embedding generation failed")
        try:
            return self.reducer.reduce_dimensions(
                embeddings, method=self.viz_config.layout_method
            )
        except Exception as e:
            return self._degraded_layout(f"dimensionality reduction failed: {e}")

    def _degraded_layout(self, reason: str) -> dict[str, Any]:
        """Seeded spring layout from graph topology, flagged as degraded."""
        self.layout_degraded = True
        logger.error(
            "%s; the layout is DEGRADED (spring layout from graph topology, not "
            "semantic embeddings).",
            reason,
        )
        return self._spring_layout()

    def _max_drawn_nodes(self) -> int | None:
        """The nodes the interactive page draws (its ``max_nodes``)."""
        return InteractiveGraphSettings.model_validate(
            self.viz_config.interactive
        ).max_nodes

    def _spring_layout(self) -> dict[str, Any]:
        # Like the embeddings, the spring layout covers only the drawn nodes.
        return spring_layout(self.analyzer.graph, self._max_drawn_nodes())

    def _generate_visualizations(
        self, outputs_dir: Path, layout: dict[str, Any]
    ) -> list[Path]:
        """Run every registered renderer; return the files they wrote."""
        if not self.analyzer.graph:
            return []

        # Drive the registered renderers through the shared registry so the
        # manager and the standalone CLI use one rendering path.
        context = RenderContext(
            graph=self.analyzer.graph,
            layout=layout,
            communities=self.community_detector.generate_community_objects(),
            community_hierarchy=list(self.community_detector.all_communities.values()),
            centrality=self.analyzer.calculate_centrality(),
        )
        written = render_all(
            context, outputs_dir, registered_renderers(), self.viz_config
        )
        if not written:
            logger.warning("No visualization files were rendered.")
        return written

    def _export_summary_report(self, outputs_dir: Path, written: list[Path]) -> None:
        report_data = {
            "graph_stats": self.analyzer.get_graph_statistics(),
            "centrality_data": self.analyzer.calculate_centrality(),
            "community_metrics": self.community_detector.get_community_metrics(),
            "visualizations": written,
        }
        self.html_exporter.create_report(outputs_dir, report_data)

    def export_visualization_data(
        self, output_path: str | Path, layout: dict[str, Any] | None = None
    ) -> Path | None:
        """Write the JSON ``run-visualization --data-path`` renders from.

        Pass the already-computed ``layout`` to avoid a second (Bedrock-backed)
        embedding pass. Returns the written path, or ``None`` if nothing was
        written.
        """
        if not self.analyzer.graph:
            logger.warning("No graph available for data export.")
            return None

        output_path = Path(output_path)
        logger.info("Exporting visualization data to '%s'...", output_path)
        data = self.analyzer.export_graph_data()
        data["nodes"] = strip_heavy_attributes(data.get("nodes", []))
        data["edges"] = strip_heavy_attributes(data.get("edges", []))
        data["layout"] = layout if layout is not None else self._generate_layout()
        data["layout_degraded"] = self.layout_degraded
        data["communities"] = self.community_detector.export_community_data()
        # Serialize centrality so the standalone CLI can render the centrality
        # comparison plot without re-running analysis.
        data["centrality"] = {
            node_id: metrics.model_dump(exclude_none=True)
            for node_id, metrics in self.analyzer.calculate_centrality().items()
        }

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except Exception as e:
            logger.exception("Failed to export data to JSON: %s", e)
            return None
        logger.info("Successfully exported visualization data to '%s'", output_path)
        return output_path
