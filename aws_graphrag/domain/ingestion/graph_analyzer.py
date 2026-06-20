# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
from collections.abc import Callable
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

from aws_graphrag.domain.models import CommunityMetrics, Config
from aws_graphrag.shared import GraphError, get_logger

logger = get_logger(__name__)


class CentralityMetrics(BaseModel):
    node_id: str = Field(description="Unique identifier for the graph node")
    node_name: str | None = Field(
        default=None,
        description="Human-readable name or label for the graph node",
    )
    degree: float | None = Field(
        default=None,
        description="Degree centrality: measures the fraction of nodes the node is connected to",
    )
    betweenness: float | None = Field(
        default=None,
        description="Betweenness centrality: measures the extent to which a node lies on paths between other nodes",
    )
    pagerank: float | None = Field(
        default=None,
        description="PageRank centrality: measures the importance of a node based on the structure of incoming links",
    )
    closeness: float | None = Field(
        default=None,
        description="Closeness centrality: measures how close a node is to all other nodes in the graph",
    )
    eigenvector: float | None = Field(
        default=None,
        description="Eigenvector centrality: measures the influence of a node based on the centrality of its neighbors",
    )


class GraphStatistics(BaseModel):
    num_nodes: int = Field(description="Total number of nodes in the graph")
    num_edges: int = Field(description="Total number of edges in the graph")
    density: float | None = Field(
        default=None,
        description="Graph density: ratio of actual edges to possible edges (0-1)",
    )
    average_clustering: float | None = Field(
        default=None,
        description="Average clustering coefficient: measures the degree to which nodes cluster together",
    )
    diameter: int | None = Field(
        default=None,
        description="Graph diameter: longest shortest path between any two nodes",
    )
    num_connected_components: int | None = Field(
        default=None, description="Number of connected components in the graph"
    )
    largest_component_size: int | None = Field(
        default=None, description="Number of nodes in the largest connected component"
    )
    community_metrics: CommunityMetrics | None = Field(
        default=None, description="Metrics related to community structure"
    )


class GraphAnalyzer:
    def __init__(self, config: Config, graph: nx.Graph | None = None) -> None:
        self.config = config
        self.analysis_config = config.graph.analysis
        self._graph = graph
        self._centrality_cache: dict[str, dict[str, float]] = {}
        self._statistics_cache: GraphStatistics | None = None

    def set_community_data(
        self,
        node_to_community_map: dict[str, str],
        community_metrics: CommunityMetrics | None,
    ) -> None:
        if not self._graph:
            return
        nx.set_node_attributes(self._graph, node_to_community_map, "community_id")
        if self._statistics_cache:
            self._statistics_cache.community_metrics = community_metrics
        logger.info("Community data integrated into the graph and statistics.")

    def calculate_centrality(self) -> dict[str, CentralityMetrics]:
        if not self._graph or self._graph.number_of_nodes() == 0:
            logger.warning("Cannot calculate centrality for an empty graph")
            return {}

        logger.info(
            "Starting centrality calculation for graph with %s nodes",
            self._graph.number_of_nodes(),
        )

        centrality_data = {}
        for node_id in self._graph.nodes():
            attrs = self._graph.nodes[node_id]
            if attrs.get("node_type") == "claim":
                node_name = f"{attrs.get('source_name', '')} -> {attrs.get('type', str(node_id))} -> {attrs.get('target_name', '')}"
            else:
                node_name = attrs.get("name", str(node_id))

            centrality_data[node_id] = CentralityMetrics(
                node_id=str(node_id),
                node_name=node_name,
            )

        centrality_methods = {
            "degree": (
                self.analysis_config.centrality.calculate_degree,
                nx.degree_centrality,
                {},
            ),
            "betweenness": (
                self.analysis_config.centrality.calculate_betweenness,
                nx.betweenness_centrality,
                {"k": self.analysis_config.centrality.betweenness_k},
            ),
            "pagerank": (
                self.analysis_config.centrality.calculate_pagerank,
                nx.pagerank,
                {
                    "alpha": self.analysis_config.centrality.pagerank_alpha,
                    "max_iter": self.analysis_config.centrality.pagerank_max_iter,
                },
            ),
            "closeness": (
                self.analysis_config.centrality.calculate_closeness,
                nx.closeness_centrality,
                {},
            ),
            "eigenvector": (
                self.analysis_config.centrality.calculate_eigenvector,
                nx.eigenvector_centrality,
                {
                    "max_iter": self.analysis_config.centrality.eigenvector_max_iter,
                    "tol": self.analysis_config.centrality.eigenvector_tol,
                },
            ),
        }

        enabled_methods = [
            name for name, (enabled, _, _) in centrality_methods.items() if enabled
        ]
        logger.info("Enabled centrality methods: %s", enabled_methods)

        for name, (enabled, func, kwargs) in centrality_methods.items():
            if enabled:
                self._compute_and_cache_centrality(name, func, kwargs, centrality_data)

        logger.info(
            "Centrality calculation completed for %s nodes", len(centrality_data)
        )
        return centrality_data

    def _compute_and_cache_centrality(
        self,
        metric_name: str,
        computation_func: Callable,
        kwargs: Any,
        centrality_data: dict[str, CentralityMetrics],
    ) -> None:
        try:
            logger.info("Calculating %s centrality...", metric_name)
            result = computation_func(self._graph, **kwargs)
            self._update_centrality_results(centrality_data, metric_name, result)
            self._centrality_cache[metric_name] = result
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError) as e:
            logger.warning("Failed to calculate %s centrality: %s", metric_name, e)
        except Exception as e:
            logger.error(
                "An unexpected error occurred during %s calculation: %s",
                metric_name,
                e,
                exc_info=True,
            )

    @staticmethod
    def _update_centrality_results(
        data: dict[str, CentralityMetrics], name: str, result: dict[str, float]
    ) -> None:
        for node_id, value in result.items():
            if node_id in data:
                setattr(data[node_id], name, value)

    def clear_cache(self) -> None:
        self._centrality_cache.clear()
        self._statistics_cache = None
        logger.info("Graph analyzer cache cleared.")

    def export_graph_data(self) -> dict[str, Any]:
        if not self._graph:
            logger.warning("Cannot export data from an empty graph")
            return {"nodes": [], "edges": [], "statistics": {}}

        logger.info(
            "Exporting graph data with %s nodes and %s edges",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

        nodes_data = [
            {
                "id": str(node_id),
                "attributes": self._graph.nodes[node_id],
                **self._get_node_centrality(node_id),
            }
            for node_id in self._graph.nodes()
        ]
        edges_data = [
            {"source": str(u), "target": str(v), "attributes": attrs}
            for u, v, attrs in self._graph.edges(data=True)
        ]
        stats = self.get_graph_statistics()

        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "statistics": stats.model_dump(exclude_none=True) if stats else {},
        }

    def _get_node_centrality(self, node_id: str) -> dict[str, float | None]:
        centralities = {}
        for name, values in self._centrality_cache.items():
            centralities[f"{name}_centrality"] = values.get(node_id)
        return centralities

    def get_graph_statistics(self) -> GraphStatistics:
        if self._statistics_cache is not None:
            return self._statistics_cache

        if not self._graph:
            logger.warning("Cannot calculate statistics for an empty graph")
            return GraphStatistics(num_nodes=0, num_edges=0)

        logger.info(
            "Calculating graph statistics for graph with %s nodes",
            self._graph.number_of_nodes(),
        )

        stats = GraphStatistics(
            num_nodes=self._graph.number_of_nodes(),
            num_edges=self._graph.number_of_edges(),
        )

        if stats.num_nodes > 0:
            self._calculate_optional_statistics(stats)

        self._statistics_cache = stats
        logger.info(
            "Graph statistics calculation completed: %s",
            stats.model_dump(exclude_none=True),
        )
        return stats

    def _calculate_optional_statistics(self, stats: GraphStatistics) -> None:
        if self.analysis_config.statistics.calculate_density:
            self._compute_metric(
                "density",
                nx.density,
                lambda r: setattr(stats, "density", r),
            )

        if self.analysis_config.statistics.calculate_clustering:
            self._compute_metric(
                "average_clustering",
                nx.average_clustering,
                lambda r: setattr(stats, "average_clustering", r),
            )

        if self.analysis_config.statistics.calculate_components:
            self._compute_metric(
                "connected_components",
                lambda g: (
                    list(nx.connected_components(g)) if not nx.is_directed(g) else []
                ),
                lambda result: self._update_component_stats(result, stats),
            )
            if (
                stats.num_connected_components == 1
                and self.analysis_config.statistics.calculate_diameter
            ):
                self._compute_metric(
                    "diameter",
                    nx.diameter,
                    lambda r: setattr(stats, "diameter", r),
                )

    def _compute_metric(
        self,
        metric_name: str,
        computation_func: Callable[[nx.Graph], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        if not self._graph:
            return
        try:
            result = computation_func(self._graph)
            on_success(result)
        except Exception as e:
            logger.warning("Failed to calculate %s: %s", metric_name, e)

    @staticmethod
    def _update_component_stats(result: list[set], stats: GraphStatistics) -> None:
        stats.num_connected_components = len(result)
        if result:
            stats.largest_component_size = len(max(result, key=len))

    def get_top_nodes_by_centrality(
        self, centrality_type: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        if centrality_type not in self._centrality_cache:
            available = list(self._centrality_cache.keys())
            msg = (
                f"Centrality type '{centrality_type}' not found. Available: {available}"
            )
            logger.error(msg)
            raise GraphError(msg)

        centrality_values = self._centrality_cache[centrality_type]
        top_nodes = sorted(
            centrality_values.items(), key=lambda item: item[1], reverse=True
        )[:top_k]

        logger.info(
            "Retrieved top %s nodes by %s centrality", len(top_nodes), centrality_type
        )
        return [(str(node), val) for node, val in top_nodes]

    @property
    def centrality_cache(self) -> dict[str, dict[str, float]]:
        return self._centrality_cache

    @property
    def graph(self) -> nx.Graph | None:
        return self._graph

    @graph.setter
    def graph(self, graph: nx.Graph | None) -> None:
        self._graph = graph
        self.clear_cache()
