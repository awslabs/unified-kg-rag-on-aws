import json
from pathlib import Path
from typing import Any

import boto3
import networkx as nx

from aws_graphrag.core import get_logger
from aws_graphrag.ingestion import CommunityDetector, GraphAnalyzer
from aws_graphrag.models import Config

from .embeddings.dimensionality import DimensionalityReducer
from .embeddings.node2vec import BedrockNodeEmbedder
from .exporters.html_exporter import HTMLExporter
from .renderers.interactive import InteractiveRenderer
from .renderers.static import StaticRenderer

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
        self.interactive_renderer = InteractiveRenderer(self.viz_config.interactive)
        self.static_renderer = StaticRenderer(self.viz_config.static)
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

    def _generate_visualizations(self, outputs_dir: Path, layout: dict[str, Any]):
        if not self.analyzer.graph:
            return

        self.interactive_renderer.create_network_visualization(
            self.analyzer.graph, layout, str(outputs_dir / "interactive_graph.html")
        )
        self.interactive_renderer.create_community_hierarchy(
            list(self.community_detector.all_communities.values()),
            str(outputs_dir / "community_hierarchy.html"),
        )

        self.static_renderer.plot_degree_distribution(
            self.analyzer.graph, str(outputs_dir / "degree_distribution.html")
        )
        centrality_data = self.analyzer.calculate_centrality()
        if centrality_data:
            self.static_renderer.plot_centrality_comparison(
                centrality_data, str(outputs_dir / "centrality_comparison.html")
            )

        communities = self.community_detector.generate_community_objects()
        if communities:
            self.static_renderer.plot_community_size_distribution(
                communities, str(outputs_dir / "community_size_distribution.html")
            )

    def _export_summary_report(self, outputs_dir: Path):
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

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Successfully exported visualization data to '{output_path}'")
        except Exception as e:
            logger.error(f"Failed to export data to JSON: {e}", exc_info=True)
