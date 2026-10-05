# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from typing import Any

import boto3
import networkx as nx

from unified_kg_rag.adapters.ingestion.community_detector import CommunityDetector
from unified_kg_rag.adapters.renderers import (
    RenderContext,
    get_renderer_class,
    registered_renderers,
)
from unified_kg_rag.domain.ingestion.graph_analyzer import GraphAnalyzer
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import get_logger

from .embeddings.dimensionality import DimensionalityReducer
from .embeddings.node2vec import BedrockNodeEmbedder
from .exporters.html_exporter import HTMLExporter

logger = get_logger(__name__)

# File written into the visualization outputs directory on every run; it is the
# input ``run-visualization --data-path`` re-renders from without ingestion.
VISUALIZATION_DATA_FILENAME = "visualization_data.json"


def _is_heavy_attribute(key: str) -> bool:
    """Vector attributes (``embedding``/``*_embedding``) are excluded from the
    export: they are large, unused by any renderer, and dominate file size."""
    return key == "embedding" or key.endswith("_embedding")


def _strip_heavy_attributes(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for item in items:
        attrs = item.get("attributes") or {}
        stripped.append(
            {
                **item,
                "attributes": {
                    k: v for k, v in attrs.items() if not _is_heavy_attribute(k)
                },
            }
        )
    return stripped


class GraphVisualizationManager:
    def __init__(
        self,
        config: Config,
        graph_analyzer: GraphAnalyzer,
        community_detector: CommunityDetector,
        outputs_dir: Path | None = None,
        boto_session: boto3.Session | None = None,
        embedder: BedrockNodeEmbedder | None = None,
    ) -> None:
        self.config = config
        self.viz_config = self.config.graph.visualization
        self.analyzer = graph_analyzer
        self.community_detector = community_detector
        self.outputs_dir = outputs_dir or Path(self.viz_config.outputs_directory)
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name
        )

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
            self._embedder = BedrockNodeEmbedder(self.config, self.boto_session)
        return self._embedder

    def run(self) -> None:
        if not self.viz_config.enabled:
            logger.info("Visualization pipeline is disabled in the configuration.")
            return

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

        self._generate_visualizations(self.outputs_dir, layout)
        self._export_summary_report(self.outputs_dir)
        self.export_visualization_data(
            self.outputs_dir / VISUALIZATION_DATA_FILENAME, layout=layout
        )

        if self.layout_degraded:
            logger.warning(
                "Visualization report created with a DEGRADED layout (node "
                "embeddings unavailable; spring layout used instead)."
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
                spring_layout = nx.spring_layout(self.analyzer.graph, seed=42)
                return {str(k): v.tolist() for k, v in spring_layout.items()}
            return {}

        embeddings = self.embedder.generate_embeddings(self.analyzer.graph)
        # Failed generation yields no embeddings (never random substitutes).
        if not embeddings.embeddings:
            self.layout_degraded = True
            logger.error(
                "Node embedding generation failed; the layout is DEGRADED "
                "(spring layout from graph topology, not semantic embeddings)."
            )
            spring_layout = nx.spring_layout(self.analyzer.graph, seed=42)
            return {str(k): v.tolist() for k, v in spring_layout.items()}

        return self.reducer.reduce_dimensions(
            embeddings, method=self.viz_config.layout_method
        )

    def _generate_visualizations(
        self, outputs_dir: Path, layout: dict[str, Any]
    ) -> None:
        if not self.analyzer.graph:
            return

        # Drive the registered renderers through the shared registry so the
        # manager and the standalone CLI use one rendering path.
        context = RenderContext(
            graph=self.analyzer.graph,
            layout=layout,
            communities=self.community_detector.generate_community_objects(),
            community_hierarchy=list(self.community_detector.all_communities.values()),
            centrality=self.analyzer.calculate_centrality(),
        )
        written: list[Path] = []
        for name in registered_renderers():
            try:
                renderer_cls = get_renderer_class(name)
                # Resolve each renderer's config block generically (viz_config
                # attribute named after the renderer); a renderer without a
                # dedicated config block gets an empty dict so renderers can call
                # ``.get(...)`` uniformly. No hardcoded renderer list.
                renderer_config = getattr(self.viz_config, name, None) or {}
                written += renderer_cls(renderer_config).render(context, outputs_dir)
            except Exception as e:
                logger.warning("Renderer '%s' failed: %s", name, e)
        if not written:
            logger.warning("No visualization files were rendered.")

    def _export_summary_report(self, outputs_dir: Path) -> None:
        report_data = {
            "graph_stats": self.analyzer.get_graph_statistics(),
            "centrality_data": self.analyzer.calculate_centrality(),
            "community_metrics": self.community_detector.get_community_metrics(),
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
        data["nodes"] = _strip_heavy_attributes(data.get("nodes", []))
        data["edges"] = _strip_heavy_attributes(data.get("edges", []))
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
