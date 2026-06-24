# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

import boto3
import networkx as nx
from graspologic.partition import leiden
from pydantic import BaseModel, Field

from aws_graphrag.adapters.aws import BedrockLanguageModelFactory
from aws_graphrag.adapters.aws.chain_factory import (
    create_robust_xml_output_parser,
    setup_chain,
)
from aws_graphrag.domain.ingestion.base_processor import BaseProcessor
from aws_graphrag.domain.models import (
    Community,
    CommunityMetrics,
    CommunityReport,
    Config,
)
from aws_graphrag.domain.prompts import CommunityReportPrompt
from aws_graphrag.shared import GraphError, get_logger
from aws_graphrag.shared.utils import (
    BatchProcessor,
    generate_stable_id,
)

logger = get_logger(__name__)


class HierarchicalCommunity(BaseModel):
    community_id: str = Field(
        description="Unique string identifier for this community within the hierarchical structure"
    )
    level: int = Field(
        description="Depth level in the community hierarchy where 0 represents the base level with finest granularity"
    )
    nodes: set[str] = Field(
        description="Collection of graph node identifiers that are members of this community"
    )
    parent_id: str | None = Field(
        None,
        description="Identifier of the parent community at the next higher hierarchical level, if any",
    )
    children_ids: list[str] = Field(
        default_factory=list,
        description="Ordered list of child community identifiers at the next lower hierarchical level",
    )

    def __hash__(self) -> int:
        return hash(self.community_id)


class CommunityDetector(BaseProcessor):
    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        show_progress: bool = True,
    ) -> None:
        super().__init__(config)
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name
        )
        self.community_detection_config = self.config.graph.community_detection
        self.ignore_errors = self.config.processing.ignore_errors

        self.graph: nx.Graph = nx.Graph()
        self.all_communities: dict[str, HierarchicalCommunity] = {}
        self.node_to_community_l0: dict[str, str] = {}
        self.base_modularity: float = 0.0
        self.show_progress = show_progress

        self.factory = BedrockLanguageModelFactory(
            config=self.config,
            boto_session=self.boto_session,
            region_name=self.config.aws.bedrock.region_name,
        )
        self.batch_processor = BatchProcessor()

        if self.community_detection_config.report_generation.enabled:
            parser = create_robust_xml_output_parser(
                factory=self.factory,
                enable_output_fixing=self.config.fixing.enabled,
                output_fixing_model_id=self.config.fixing.fixing_model_id,
            )
            self.report_generator = setup_chain(
                factory=self.factory,
                model_id=self.community_detection_config.report_generation.report_generation_model_id,
                prompt_class=CommunityReportPrompt,
                parser=parser,
            )

    def __call__(self, graph: nx.Graph) -> "CommunityDetector":
        self.graph = graph
        logger.info(
            "Starting community detection on graph with %s nodes and %s edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        self.detect_communities()
        return self

    def detect_communities(self) -> None:
        if not self.graph or self.graph.number_of_nodes() == 0:
            logger.warning("Community detection skipped for empty or invalid graph.")
            return

        self.analyze_hierarchy()
        logger.info(
            "Detected %s communities across all levels.", len(self.all_communities)
        )

    def analyze_hierarchy(self) -> None:
        try:
            base_partition = self._get_base_partition()
        except ValueError as e:
            logger.error("Failed to detect base communities: %s", e)
            return

        if not base_partition:
            logger.error(
                "Failed to detect base communities - aborting hierarchy analysis"
            )
            return

        self.base_modularity = nx.community.modularity(self.graph, base_partition)

        logger.info(
            "Base partition found with %s communities and modularity %.4f",
            len(base_partition),
            self.base_modularity,
        )

        partitions = [base_partition]
        max_levels = self.community_detection_config.max_levels

        current_partition = base_partition

        for level in range(1, max_levels):
            logger.info("Creating level %s communities...", level)

            try:
                cluster_graph = self._create_cluster_graph(current_partition)

                if cluster_graph.number_of_nodes() < 2:
                    logger.info(
                        "Cluster graph has only %s nodes, stopping hierarchy at level %s",
                        cluster_graph.number_of_nodes(),
                        level - 1,
                    )
                    break

                logger.info(
                    "Created cluster graph with %s nodes and %s edges",
                    cluster_graph.number_of_nodes(),
                    cluster_graph.number_of_edges(),
                )

                next_partition = self._get_base_partition(cluster_graph)

                if (
                    next_partition
                    and 1 < len(next_partition) < cluster_graph.number_of_nodes()
                ):
                    logger.info(
                        "Selected partition with %s communities for level %s",
                        len(next_partition),
                        level,
                    )
                else:
                    logger.info(
                        "No meaningful partition found at level %s, stopping hierarchy construction",
                        level,
                    )
                    break

                partitions.append(next_partition)
                current_partition = next_partition

            except Exception as e:
                logger.error(
                    "Failed to create cluster graph for level %s: %s", level, e
                )
                break

        logger.info("Hierarchy construction completed with %s levels", len(partitions))
        self._process_hierarchical_partitions(partitions)

    def _get_base_partition(
        self, graph: nx.Graph | None = None
    ) -> list[set[str]] | None:
        target_graph = graph or self.graph

        if not target_graph or target_graph.number_of_edges() == 0:
            logger.warning(
                "Graph has no edges. Treating each node as a separate community."
            )
            return [{node} for node in target_graph.nodes()]

        try:
            config = self.community_detection_config

            if config.auto_resolution:
                resolution = self._find_optimal_resolution(target_graph)
            else:
                resolution = config.resolution

            all_nodes = set(target_graph.nodes())

            partition_dict = leiden(
                target_graph,
                resolution=resolution,
                random_seed=config.random_state,
                trials=config.trials,
                extra_forced_iterations=config.extra_forced_iterations,
            )

            communities = self._partition_dict_to_communities(partition_dict)
            partitioned_nodes = set(partition_dict.keys())
            missing_nodes = all_nodes - partitioned_nodes
            next_label = max(communities.keys()) + 1 if communities else 0
            for node in missing_nodes:
                communities[next_label] = {node}
                next_label += 1

            if config.min_community_size > 1:
                communities = self._merge_small_communities(
                    dict(communities), target_graph, config.min_community_size
                )

            return list(communities.values())

        except Exception as e:
            logger.error("Error during Leiden partitioning: %s", e)
            return None

    @staticmethod
    def _partition_dict_to_communities(
        partition_dict: dict[str, int],
    ) -> dict[int, set[str]]:
        communities: dict[int, set[str]] = defaultdict(set)
        for node, label in partition_dict.items():
            communities[label].add(node)
        return dict(communities)

    def _find_optimal_resolution(self, graph: nx.Graph) -> float:
        best_resolution = self.community_detection_config.resolution
        best_modularity = -1.0

        resolution_candidates = (
            self.community_detection_config.auto_resolution_candidates
        )

        for resolution in resolution_candidates:
            try:
                partition_dict = leiden(
                    graph,
                    resolution=resolution,
                    random_seed=self.community_detection_config.random_state,
                    trials=1,
                )

                communities = self._partition_dict_to_communities(partition_dict)
                partition = list(communities.values())

                if len(partition) < 2:
                    continue

                modularity = nx.community.modularity(graph, partition)

                if modularity > best_modularity:
                    best_modularity = modularity
                    best_resolution = resolution

            except Exception as e:
                logger.debug("Resolution %s failed: %s", resolution, e)
                continue

        logger.info(
            "Auto-selected resolution %s with modularity %.4f",
            best_resolution,
            best_modularity,
        )
        return best_resolution

    def _merge_small_communities(
        self,
        communities: dict[int, set[str]],
        graph: nx.Graph,
        min_size: int,
    ) -> dict[int, set[str]]:
        node_to_comm: dict[str, int] = {}
        for comm_id, nodes in communities.items():
            for node in nodes:
                node_to_comm[node] = comm_id

        # Iterate to convergence: a small community can merge into another small
        # one (or one already processed), so a single pass can leave communities
        # still below min_size. Repeat until no merge happens (or no small
        # community has an eligible neighbor — isolated ones can't be merged).
        while True:
            small_comms = [
                cid for cid, nodes in communities.items() if len(nodes) < min_size
            ]
            merged_any = False

            for small_comm_id in small_comms:
                if small_comm_id not in communities:
                    continue

                nodes = communities[small_comm_id]

                neighbor_counts: dict[int, int] = defaultdict(int)
                for node in nodes:
                    for neighbor in graph.neighbors(node):
                        if neighbor in node_to_comm:
                            neighbor_comm = node_to_comm[neighbor]
                            if (
                                neighbor_comm != small_comm_id
                                and neighbor_comm in communities
                            ):
                                neighbor_counts[neighbor_comm] += 1

                if neighbor_counts:
                    target_comm = max(neighbor_counts, key=lambda x: neighbor_counts[x])
                    communities[target_comm].update(nodes)
                    for node in nodes:
                        node_to_comm[node] = target_comm
                    del communities[small_comm_id]
                    merged_any = True

            if not merged_any:
                break

        return communities

    def _create_cluster_graph(self, partition: list[set[str]]) -> nx.Graph:
        num_communities = len(partition)
        cluster_graph = nx.Graph()

        for i in range(num_communities):
            cluster_graph.add_node(i, name=f"cluster_{i}")

        node_to_community = {}
        for comm_idx, nodes in enumerate(partition):
            for node in nodes:
                node_to_community[node] = comm_idx

        edge_weights: dict[tuple[int, int], float] = defaultdict(float)

        for source, target, data in self.graph.edges(data=True):
            source_community = node_to_community.get(source)
            target_community = node_to_community.get(target)

            if source_community is None or target_community is None:
                continue
            if source_community == target_community:
                continue

            edge_weight = data.get("weight", 1.0)
            comm_pair = (
                min(source_community, target_community),
                max(source_community, target_community),
            )
            edge_weights[comm_pair] += edge_weight

        for (comm1, comm2), weight in edge_weights.items():
            if weight > 0:
                cluster_graph.add_edge(comm1, comm2, weight=weight)

        return cluster_graph

    def _process_hierarchical_partitions(
        self, partitions: list[list[set[str]]]
    ) -> None:
        self.all_communities.clear()
        self.node_to_community_l0.clear()

        logger.info("Processing %s levels of partitions", len(partitions))

        l0_partition = partitions[0]
        # Keys are partition member identities, which Leiden may emit as ints
        # or node-label strings depending on the input graph.
        level_0_communities: dict[Any, str] = {}

        for i, nodes in enumerate(l0_partition):
            comm_id = f"L0_C{i}"

            for node_id in nodes:
                self.node_to_community_l0[node_id] = comm_id

            community = HierarchicalCommunity(
                community_id=comm_id, level=0, nodes=nodes, parent_id=None
            )
            self.all_communities[comm_id] = community
            level_0_communities[i] = comm_id

        logger.info("Created %s level 0 communities", len(level_0_communities))

        previous_level_map = level_0_communities

        for level in range(1, len(partitions)):
            partition = partitions[level]
            current_level_map: dict[Any, str] = {}

            for i, cluster_nodes in enumerate(partition):
                parent_comm_id = f"L{level}_C{i}"
                parent_nodes: set[str] = set()
                child_comm_ids: list[str] = []

                for cluster_idx in cluster_nodes:
                    if cluster_idx in previous_level_map:
                        child_comm_id = previous_level_map[cluster_idx]
                        child_comm = self.all_communities.get(child_comm_id)

                        if child_comm:
                            child_comm.parent_id = parent_comm_id
                            parent_nodes.update(child_comm.nodes)
                            child_comm_ids.append(child_comm_id)

                if parent_nodes:
                    parent_community = HierarchicalCommunity(
                        community_id=parent_comm_id,
                        level=level,
                        nodes=parent_nodes,
                        parent_id=None,
                        children_ids=child_comm_ids,
                    )
                    self.all_communities[parent_comm_id] = parent_community
                    current_level_map[i] = parent_comm_id

            logger.info(
                "Created %s level %s communities", len(current_level_map), level
            )
            previous_level_map = current_level_map

            if not current_level_map:
                logger.warning(
                    "No communities created at level %s, stopping hierarchy", level
                )
                break

        logger.info(
            "Final hierarchy: %s total communities across %s levels",
            len(self.all_communities),
            len(partitions),
        )

    def export_community_data(self) -> dict[str, Any]:
        if not self.all_communities:
            raise GraphError("Communities must be detected before exporting.")

        metrics = self.get_community_metrics()
        hierarchy_output = [
            {
                "community_id": comm.community_id,
                "level": comm.level,
                "nodes": list(comm.nodes),
                "size": len(comm.nodes),
                "parent": comm.parent_id,
                "children": comm.children_ids,
            }
            for comm in self.all_communities.values()
        ]

        return {
            "resolution": self.community_detection_config.resolution,
            "metrics_level_0": metrics.model_dump() if metrics else {},
            "hierarchy": hierarchy_output,
        }

    def generate_community_objects(self) -> list[Community]:
        if not self.graph or self.graph.number_of_nodes() == 0:
            return []

        if not self.all_communities:
            raise GraphError("Communities must be detected before generating objects.")

        attributes = self._extract_attributes_from_graph()
        communities = [
            self._create_community_object(hier_comm, attributes)
            for hier_comm in self.all_communities.values()
        ]

        logger.info("Generated %s community objects", len(communities))
        return communities

    def get_community_metrics(self) -> CommunityMetrics | None:
        if not self.all_communities or not self.graph:
            return None

        level_0_communities = [
            comm.nodes for comm in self.all_communities.values() if comm.level == 0
        ]
        if not level_0_communities:
            return None

        community_sizes = [len(nodes) for nodes in level_0_communities]
        if not community_sizes:
            return CommunityMetrics(
                modularity=0.0,
                num_communities=0,
                average_community_size=0.0,
                largest_community_size=0,
                smallest_community_size=0,
                community_size_distribution={},
            )

        size_distribution: dict[int, int] = defaultdict(int)
        for size in community_sizes:
            size_distribution[size] += 1

        return CommunityMetrics(
            modularity=self.base_modularity,
            num_communities=len(level_0_communities),
            average_community_size=sum(community_sizes) / len(community_sizes),
            largest_community_size=max(community_sizes),
            smallest_community_size=min(community_sizes),
            community_size_distribution=dict(size_distribution),
        )

    def _extract_attributes_from_graph(self) -> dict[str, Any]:
        if hasattr(self.graph, "graph") and self.graph.graph:
            attributes = self.graph.graph.get("attributes", {})
            if isinstance(attributes, dict):
                return attributes
            return {}
        for _, node_data in self.graph.nodes(data=True):
            node_attributes = node_data.get("attributes")
            if isinstance(node_attributes, dict):
                return node_attributes
        return {}

    def _create_community_object(
        self, hier_comm: HierarchicalCommunity, attributes: dict[str, Any]
    ) -> Community:
        subgraph = self.graph.subgraph(hier_comm.nodes)
        max_entities = (
            self.community_detection_config.report_generation.max_entities_per_report
        )
        entity_names = [
            self.graph.nodes[nid].get("name", nid)
            for nid in list(hier_comm.nodes)[:max_entities]
        ]

        community_attributes = {
            **attributes,
            "size": len(hier_comm.nodes),
            "density": nx.density(subgraph),
            "entity_names": entity_names,
            "resolution": self.community_detection_config.resolution,
            "num_relationships": subgraph.number_of_edges(),
        }

        return Community(
            id=hier_comm.community_id,
            short_id=hier_comm.community_id,
            name=f"Level {hier_comm.level} Community {hier_comm.community_id.split('_C')[1]}",
            name_embedding=None,
            level=str(hier_comm.level),
            parent=hier_comm.parent_id or "",
            children=hier_comm.children_ids,
            entity_ids=list(hier_comm.nodes),
            relationship_ids=[
                rel_id
                for source, target in subgraph.edges()
                if (rel_id := self.graph.edges[source, target].get("id")) is not None
            ],
            covariate_ids={},
            text_unit_ids=[],
            size=len(hier_comm.nodes),
            period=None,
            attributes=community_attributes,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    def generate_reports(self, communities: list[Community]) -> list[CommunityReport]:
        report_config = self.community_detection_config.report_generation
        if not report_config.enabled or not hasattr(self, "report_generator"):
            logger.info("Community report generation is disabled")
            return []

        if not communities:
            logger.warning("No communities provided for report generation")
            return []

        try:
            logger.info("Generating reports for %s communities.", len(communities))
            start_time = time.time()
            graph_attributes = self._extract_attributes_from_graph()
            report_inputs = [self._prepare_report_input(c) for c in communities]

            report_results = self.batch_processor.execute_with_fallback(
                items_to_process=report_inputs,
                prepare_inputs_func=self._create_report_chain_inputs,
                batch_func=self.report_generator.batch,
                sequential_func=self.report_generator.invoke,
                task_name="Community report generation",
                run_config=self.config.processing.model_dump(),
                show_progress=self.show_progress,
            )

            reports = []
            failed = 0
            for community, result in zip(communities, report_results, strict=True):
                if result:
                    report = self._create_community_report(
                        community, result, graph_attributes
                    )
                    reports.append(report)
                else:
                    # Empty LLM result: count and log rather than silently
                    # producing fewer reports while the run still 'succeeds'.
                    failed += 1
                    logger.warning(
                        "No report generated for community '%s' (empty LLM result)",
                        community.id,
                    )

            if failed:
                logger.warning(
                    "%s of %s community reports had no result", failed, len(communities)
                )
            logger.info(
                "Generated %s reports in %.2fs.",
                len(reports),
                time.time() - start_time,
            )

        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.error("Error during community report generation: %s", e)
            return []

        return reports

    def _prepare_report_input(self, community: Community) -> dict[str, Any]:
        report_config = self.community_detection_config.report_generation
        entity_ids = (community.entity_ids or [])[
            : report_config.max_entities_per_report
        ]

        entities = [
            {
                "id": eid,
                "name": self.graph.nodes[eid].get("name", eid),
                "type": self.graph.nodes[eid].get("type", "unknown"),
                "description": self.graph.nodes[eid].get("description", ""),
            }
            for eid in entity_ids
            if eid in self.graph
        ]

        relationships = [
            {
                # Use entity NAMES, not raw node ids (hashes): the report LLM
                # otherwise sees relationships between unintelligible ids,
                # degrading report quality. Fall back to the id only if a node
                # has no name.
                "source": self.graph.nodes[u].get("name", u),
                "target": self.graph.nodes[v].get("name", v),
                "type": data.get("type", "related"),
                "description": data.get("description", ""),
            }
            for u, v, data in self.graph.subgraph(entity_ids).edges(data=True)
        ]

        return {
            "community_id": community.id,
            "entities": entities,
            "relationships": relationships,
            **report_config.model_dump(
                include={"content_length", "include_statistics", "include_key_entities"}
            ),
        }

    @staticmethod
    def _create_report_chain_inputs(
        report_inputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "community_id": r["community_id"],
                "entities": "\n".join(
                    f"- {e['name']} ({e['type']}): {e['description']}"
                    for e in r["entities"]
                ),
                "relationships": "\n".join(
                    f"- {rel['source']} -> {rel['target']} ({rel['type']}): {rel['description']}"
                    for rel in r["relationships"]
                ),
                "content_length": r["content_length"],
                "include_statistics": str(r["include_statistics"]),
                "include_key_entities": str(r["include_key_entities"]),
            }
            for r in report_inputs
        ]

    def _create_community_report(
        self,
        community: Community,
        result: dict[str, Any],
        graph_attributes: dict[str, Any],
    ) -> CommunityReport:
        report_config = self.community_detection_config.report_generation

        name = self._extract_text_from_result(result.get("community_name", ""))
        summary = self._extract_text_from_result(
            result.get(
                "summary", f"No summary generated for community {community.name}."
            )
        )
        full_content = self._extract_text_from_result(
            result.get(
                "full_content", f"No report generated for community {community.name}."
            )
        )

        report_id = generate_stable_id(f"report:{community.id}")
        current_time = datetime.now()

        attributes = {
            "generation_model": report_config.report_generation_model_id,
            "entity_count": len(community.entity_ids or []),
            "relationship_count": len(community.relationship_ids or []),
            "content_length": report_config.content_length,
            "generation_timestamp": current_time.isoformat(),
            **graph_attributes,
        }

        report_name = (
            f"Report for {community.name}: {name}"
            if name
            else f"Report for {community.name}"
        )

        return CommunityReport(
            id=report_id,
            short_id=report_id[:8],
            name=report_name,
            name_embedding=None,
            community_id=community.id,
            summary=summary,
            summary_embedding=None,
            full_content=full_content,
            full_content_embedding=None,
            rank=1,
            size=community.size,
            period=community.period,
            attributes=attributes,
            created_at=current_time,
            updated_at=current_time,
        )

    @staticmethod
    def _extract_text_from_result(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if "#text" in value:
                return str(value["#text"])
            return str(value)
        return str(value)
