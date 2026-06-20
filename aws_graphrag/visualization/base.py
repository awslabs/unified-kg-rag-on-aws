# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import json
from pathlib import Path
from typing import Any

import boto3
import networkx as nx

from aws_graphrag.adapters.renderers import (
    RenderContext,
    get_renderer_class,
    registered_renderers,
)
from aws_graphrag.domain.models import Config
from aws_graphrag.ingestion import CommunityDetector, GraphAnalyzer
from aws_graphrag.shared import get_logger

from .embeddings.dimensionality import DimensionalityReducer
from .embeddings.node2vec import BedrockNodeEmbedder
from .exporters.html_exporter import HTMLExporter

logger = get_logger(__name__)


class GraphVisualizationManager:
    def __init__(
        self,
        config: Config,
        graph_analyzer: GraphAnalyzer,
        community_detector: CommunityDetector,
        outputs_dir: Path | None = None,
        boto_session: boto3.Session | None = None,
    ) -> None:
        self.config = config
        self.viz_config = self.config.graph.visualization
        self.analyzer = graph_analyzer
        self.community_detector = community_detector
        self.outputs_dir = outputs_dir or Path(self.viz_config.outputs_directory)
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name
        )

        self.embedder = BedrockNodeEmbedder(self.config, self.boto_session)

        self.reducer = DimensionalityReducer(self.viz_config.layout)
        self.html_exporter = HTMLExporter()

    def run(self) -> None:
        if not self.viz_config.enabled:
            logger.info("Visualization pipeline is disabled in the configuration.")
            return

        if not self.analyzer.graph:
            logger.warning("No graph is available for visualization. Skipping.")
            return

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Starting comprehensive visualization report creation in '{self.outputs_dir}'."
        )

        self.analyzer.set_community_data(
            self.community_detector.node_to_community_l0,
            self.community_detector.get_community_metrics(),
        )

        layout = self._generate_layout()

        self._generate_visualizations(self.outputs_dir, layout)
        self._export_summary_report(self.outputs_dir)

        logger.info("Comprehensive visualization report created successfully.")

    def _generate_layout(self) -> dict[str, Any]:
        if not self.analyzer.graph or self.viz_config.embedding_method == "none":
            logger.info(
                "Skipping embedding generation. Using spring layout as fallback."
            )
            if self.analyzer.graph:
                spring_layout = nx.spring_layout(self.analyzer.graph, seed=42)
                return {str(k): v.tolist() for k, v in spring_layout.items()}
            return {}

        embeddings = self.embedder.generate_embeddings(self.analyzer.graph)
        if not embeddings.embeddings:
            logger.warning(
                "Embedding generation failed. Falling back to spring layout."
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
        for name in registered_renderers():
            try:
                renderer_cls = get_renderer_class(name)
                # Resolve each renderer's config block generically (viz_config
                # attribute named after the renderer); a renderer without a
                # dedicated config block gets None. No hardcoded renderer list.
                renderer_config = getattr(self.viz_config, name, None)
                renderer_cls(renderer_config).render(context, outputs_dir)
            except Exception as e:
                logger.warning("Renderer '%s' failed: %s", name, e)

    def _export_summary_report(self, outputs_dir: Path) -> None:
        report_data = {
            "graph_stats": self.analyzer.get_graph_statistics(),
            "centrality_data": self.analyzer.calculate_centrality(),
            "community_metrics": self.community_detector.get_community_metrics(),
        }
        self.html_exporter.create_report(outputs_dir, report_data)

    def export_visualization_data(self, output_path: str) -> None:
        if not self.analyzer.graph:
            logger.warning("No graph available for data export.")
            return

        logger.info(f"Exporting visualization data to '{output_path}'...")
        data = self.analyzer.export_graph_data()
        data["layout"] = self._generate_layout()
        data["communities"] = self.community_detector.export_community_data()
        # Serialize centrality so the standalone CLI can render the centrality
        # comparison plot without re-running analysis.
        data["centrality"] = {
            node_id: metrics.model_dump(exclude_none=True)
            for node_id, metrics in self.analyzer.calculate_centrality().items()
        }

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Successfully exported visualization data to '{output_path}'")
        except Exception as e:
            logger.error(f"Failed to export data to JSON: {e}", exc_info=True)
