# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

import boto3
import networkx as nx
from graspologic.partition import leiden
from pydantic import BaseModel, Field

from unified_kg_rag.adapters.aws import BedrockLanguageModelFactory
from unified_kg_rag.adapters.aws.chain_factory import (
    create_robust_xml_output_parser,
    setup_chain,
)
from unified_kg_rag.adapters.aws.token_counter import estimate_token_count
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.models import (
    Community,
    CommunityFinding,
    CommunityMetrics,
    CommunityReport,
    Config,
)
from unified_kg_rag.domain.prompts import CommunityReportPrompt
from unified_kg_rag.shared import GraphError, get_logger
from unified_kg_rag.shared.utils import (
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
        # The graph to coarsen at the NEXT level. Level 1 coarsens the original
        # entity graph; level >= 2 must coarsen the PREVIOUS level's cluster
        # graph (whose nodes are previous-level community indices), because the
        # partition members at level >= 2 are cluster indices, not entity ids.
        current_graph = self.graph

        for level in range(1, max_levels):
            logger.info("Creating level %s communities...", level)

            try:
                cluster_graph = self._create_cluster_graph(
                    current_partition, current_graph
                )

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
                # Coarsen the cluster graph we just built next, so deeper levels
                # aggregate over the previous level rather than the entity graph.
                current_graph = cluster_graph

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

    def _collect_community_text_unit_ids(self, node_ids: Any) -> list[str]:
        """Union the source text-unit ids of a community's member entities.

        Graph nodes carry the full entity dump (graph_builder adds each entity
        with **entity.model_dump()), so each node's ``text_unit_ids`` is the list
        of text units that mentioned it. Deduplicated + sorted for a stable,
        reproducible community record.
        """
        collected: set[str] = set()
        for nid in node_ids:
            attrs = self.graph.nodes[nid] if nid in self.graph else {}
            tu = attrs.get("text_unit_ids") or []
            if isinstance(tu, (list, tuple, set)):
                collected.update(str(t) for t in tu if t)
        return sorted(collected)

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

        # Size short-circuit: the sweep runs a full Leiden partition + modularity
        # computation for EACH candidate (~10x the base cost) at every hierarchy
        # level. On a large graph that dominates the analysis phase, so above the
        # configured node cap we skip the sweep and use the fixed resolution.
        max_nodes = self.community_detection_config.auto_resolution_max_nodes
        if graph.number_of_nodes() > max_nodes:
            logger.info(
                "Graph has %s nodes (> %s); skipping the auto-resolution sweep "
                "and using the fixed resolution %s.",
                graph.number_of_nodes(),
                max_nodes,
                best_resolution,
            )
            return best_resolution

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

    def _create_cluster_graph(
        self, partition: list[set[str]], source_graph: nx.Graph | None = None
    ) -> nx.Graph:
        # ``source_graph`` is the graph whose nodes the partition members refer
        # to: the original entity graph at level 1, or the previous level's
        # cluster graph at level >= 2. Defaults to the entity graph for the
        # first coarsening / backward compatibility.
        graph = source_graph if source_graph is not None else self.graph
        num_communities = len(partition)
        cluster_graph = nx.Graph()

        for i in range(num_communities):
            cluster_graph.add_node(i, name=f"cluster_{i}")

        node_to_community = {}
        for comm_idx, nodes in enumerate(partition):
            for node in nodes:
                node_to_community[node] = comm_idx

        edge_weights: dict[tuple[int, int], float] = defaultdict(float)

        for source, target, data in graph.edges(data=True):
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
            # Union the member entities' source text units. Nodes are added with
            # the full entity dump (graph_builder), so text_unit_ids is present.
            # Without this the community's text_unit_ids stayed empty, so
            # _enrich_text_units never tagged text units with community_ids and
            # global search's community text-unit fusion bucket matched nothing.
            text_unit_ids=self._collect_community_text_unit_ids(hier_comm.nodes),
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

            if (
                self.community_detection_config.report_generation.enable_sub_community_rollup
            ):
                reports, failed = self._generate_reports_with_rollup(
                    communities, graph_attributes
                )
            else:
                reports, failed = self._generate_reports_flat(
                    communities, graph_attributes
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

    def _run_report_batch(
        self,
        report_inputs: list[dict[str, Any]],
        communities: list[Community],
        graph_attributes: dict[str, Any],
    ) -> tuple[list[CommunityReport], int]:
        """Run one batch of prepared report inputs through the LLM chain."""
        report_results = self.batch_processor.execute_with_fallback(
            items_to_process=report_inputs,
            prepare_inputs_func=self._create_report_chain_inputs,
            batch_func=self.report_generator.batch,
            sequential_func=self.report_generator.invoke,
            task_name="Community report generation",
            run_config=self.config.processing.model_dump(),
            show_progress=self.show_progress,
        )

        reports: list[CommunityReport] = []
        failed = 0
        for community, result in zip(communities, report_results, strict=True):
            if result:
                reports.append(
                    self._create_community_report(community, result, graph_attributes)
                )
            else:
                # Empty LLM result: count and log rather than silently producing
                # fewer reports while the run still 'succeeds'.
                failed += 1
                logger.warning(
                    "No report generated for community '%s' (empty LLM result)",
                    community.id,
                )
        return reports, failed

    def _generate_reports_flat(
        self, communities: list[Community], graph_attributes: dict[str, Any]
    ) -> tuple[list[CommunityReport], int]:
        """Original behaviour: every community summarized independently."""
        report_inputs = [self._prepare_report_input(c) for c in communities]
        return self._run_report_batch(report_inputs, communities, graph_attributes)

    def _generate_reports_with_rollup(
        self, communities: list[Community], graph_attributes: dict[str, Any]
    ) -> tuple[list[CommunityReport], int]:
        """Bottom-up per-level generation with sub-community roll-up.

        Communities are grouped by level and processed finest-first (level 0 =
        leaf). Each level's reports accumulate into ``reports_by_id``, so when a
        coarser parent's raw context overflows the token budget, the already-
        generated child reports are available to substitute (MS GraphRAG's
        local-vs-sub-community trade-off). Batching/fallback is preserved within
        each level.
        """

        def _level(c: Community) -> int:
            try:
                return int(c.level)
            except (TypeError, ValueError):
                return 0

        reports_by_id: dict[str, CommunityReport] = {}
        all_reports: list[CommunityReport] = []
        total_failed = 0

        for level in sorted({_level(c) for c in communities}):
            level_communities = [c for c in communities if _level(c) == level]
            report_inputs = [
                self._prepare_report_input(c, reports_by_id) for c in level_communities
            ]
            level_reports, failed = self._run_report_batch(
                report_inputs, level_communities, graph_attributes
            )
            total_failed += failed
            for report in level_reports:
                reports_by_id[report.community_id] = report
            all_reports.extend(level_reports)

        return all_reports, total_failed

    def _compute_report_rank(self, community: Community) -> int:
        """Deterministic community-importance rank (no LLM call).

        MS GraphRAG uses a report ``rank`` to reflect community importance (it
        prefilters/sorts global-search candidates by it). We derive it from
        graph structure: the sum of in-graph degrees of the community's member
        entities — a connectivity signal that is stable across runs and needs no
        model call. Higher = more central/important. Falls back to the member
        count when the graph is unavailable.
        """
        entity_ids = community.entity_ids or []
        if not self.graph or self.graph.number_of_nodes() == 0:
            return len(entity_ids)
        return sum(self.graph.degree(eid) for eid in entity_ids if eid in self.graph)

    def _select_report_entities(self, community: Community) -> list[str]:
        """Pick the community's entity ids most worth putting in its report.

        MS GraphRAG builds report context most-connected-first so a large
        community's report leads with its central, defining entities rather
        than whatever happened to appear first in list order. We mirror that:
        rank the community's in-graph entities by graph degree (descending),
        then cap at ``max_entities_per_report``. The token-budget pack in
        ``_prepare_report_input`` may keep fewer; degree-sort guarantees that
        whatever is dropped is the least-connected (least-important) entity.
        Ties break on a stable id sort so selection is deterministic.
        """
        max_entities = (
            self.community_detection_config.report_generation.max_entities_per_report
        )
        present = [eid for eid in (community.entity_ids or []) if eid in self.graph]
        present.sort(key=lambda eid: (-self.graph.degree(eid), str(eid)))
        return present[:max_entities]

    def _prepare_report_input(
        self,
        community: Community,
        reports_by_id: dict[str, CommunityReport] | None = None,
    ) -> dict[str, Any]:
        report_config = self.community_detection_config.report_generation
        token_budget = report_config.max_report_context_tokens

        # Degree-sorted candidates (most-connected first), capped at the entity
        # upper bound. We then pack into the token budget so a huge community
        # neither overflows the prompt nor gets arbitrarily truncated by list
        # order — truncation drops the least-connected entities first.
        candidate_ids = self._select_report_entities(community)

        entities: list[dict[str, Any]] = []
        selected_ids: list[str] = []
        used_tokens = 0
        truncated = False
        for eid in candidate_ids:
            node = self.graph.nodes[eid]
            entity = {
                "id": eid,
                "name": node.get("name", eid),
                "type": node.get("type", "unknown"),
                "description": node.get("description", ""),
            }
            cost = estimate_token_count(self._format_entity_line(entity))
            # Always admit the first (highest-degree) entity even if it alone
            # exceeds the budget, so a report is never left with no context.
            if entities and used_tokens + cost > token_budget:
                truncated = True
                break
            entities.append(entity)
            selected_ids.append(eid)
            used_tokens += cost

        # Relationships only among the selected entities, ordered by combined
        # endpoint degree (most-connected pair first, edge weight as tiebreak)
        # so the most salient connections survive the same token budget.
        selected_set = set(selected_ids)
        edges = sorted(
            self.graph.subgraph(selected_ids).edges(data=True),
            key=lambda e: (
                -(self.graph.degree(e[0]) + self.graph.degree(e[1])),
                -float(e[2].get("weight", 1.0)),
            ),
        )
        relationships: list[dict[str, Any]] = []
        for u, v, data in edges:
            if u not in selected_set or v not in selected_set:
                continue
            rel = {
                # Use entity NAMES, not raw node ids (hashes): the report LLM
                # otherwise sees relationships between unintelligible ids,
                # degrading report quality. Fall back to the id only if a node
                # has no name.
                "source": self.graph.nodes[u].get("name", u),
                "target": self.graph.nodes[v].get("name", v),
                "type": data.get("type", "related"),
                "description": data.get("description", ""),
            }
            cost = estimate_token_count(self._format_relationship_line(rel))
            if used_tokens + cost > token_budget:
                truncated = True
                break
            relationships.append(rel)
            used_tokens += cost

        # Sub-community roll-up: when this (parent) community's raw context was
        # truncated and its child sub-communities already have reports, fold
        # their report summaries in (MS GraphRAG parity) so the synthesis of the
        # parts that did not fit is not lost. Bounded by its own token budget so
        # the prompt stays roughly within 2x the raw budget in the overflow case.
        sub_reports = ""
        if truncated and reports_by_id:
            sub_reports = self._build_sub_community_context(
                community, reports_by_id, token_budget
            )

        return {
            "community_id": community.id,
            "entities": entities,
            "relationships": relationships,
            "sub_community_reports": sub_reports,
            **report_config.model_dump(
                include={"content_length", "include_statistics", "include_key_entities"}
            ),
        }

    def _build_sub_community_context(
        self,
        community: Community,
        reports_by_id: dict[str, CommunityReport],
        token_budget: int,
    ) -> str:
        """Pack child sub-community report summaries into a token budget.

        Children are taken biggest-first (by report size, then id for stable
        ordering) so the largest sub-communities — whose raw detail is most
        likely to have overflowed the parent — are summarized first. Each entry
        is the child's executive summary prefixed with its importance rating.
        """
        child_reports = [
            reports_by_id[cid]
            for cid in (community.children or [])
            if cid in reports_by_id
        ]
        if not child_reports:
            return ""

        child_reports.sort(key=lambda r: (-(r.size or 0), r.id))

        lines: list[str] = []
        used = 0
        for report in child_reports:
            summary = (report.summary or report.full_content or "").strip()
            if not summary:
                continue
            header = report.name or f"Sub-community {report.community_id}"
            entry = f"- {header} (rating {report.rating:.1f}/10): {summary}"
            cost = estimate_token_count(entry)
            if lines and used + cost > token_budget:
                break
            lines.append(entry)
            used += cost
        return "\n".join(lines)

    @staticmethod
    def _format_entity_line(entity: dict[str, Any]) -> str:
        return f"- {entity['name']} ({entity['type']}): {entity['description']}"

    @staticmethod
    def _format_relationship_line(rel: dict[str, Any]) -> str:
        return (
            f"- {rel['source']} -> {rel['target']} "
            f"({rel['type']}): {rel['description']}"
        )

    @classmethod
    def _create_report_chain_inputs(
        cls,
        report_inputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "community_id": r["community_id"],
                "entities": "\n".join(
                    cls._format_entity_line(e) for e in r["entities"]
                ),
                "relationships": "\n".join(
                    cls._format_relationship_line(rel) for rel in r["relationships"]
                ),
                "sub_community_reports": r.get("sub_community_reports", ""),
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
        findings = self._extract_findings(result)
        rating = self._extract_rating(result)
        rating_explanation = self._extract_text_from_result(
            result.get("rating_explanation", "")
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

        report = CommunityReport(
            id=report_id,
            short_id=report_id[:8],
            name=report_name,
            name_embedding=None,
            community_id=community.id,
            summary=summary,
            summary_embedding=None,
            full_content="",
            full_content_embedding=None,
            findings=findings,
            rating=rating,
            rating_explanation=rating_explanation,
            rank=self._compute_report_rank(community),
            size=community.size,
            period=community.period,
            attributes=attributes,
            created_at=current_time,
            updated_at=current_time,
        )
        # Derive the free-text body from the structured fields so embeddings,
        # global-search map-reduce, and display all stay consistent with the
        # findings/rating without a second LLM call.
        rendered = report.render_full_content()
        report.full_content = (
            rendered or f"No report generated for community {community.name}."
        )
        return report

    @staticmethod
    def _extract_text_from_result(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if "#text" in value:
                return str(value["#text"])
            return str(value)
        return str(value)

    @classmethod
    def _extract_rating(cls, result: dict[str, Any]) -> float:
        """Parse the 0-10 importance rating, tolerating noisy LLM output."""
        raw = cls._extract_text_from_result(result.get("rating", "")).strip()
        if not raw:
            return 0.0
        match = re.search(r"-?\d+(?:\.\d+)?", raw)
        if not match:
            return 0.0
        try:
            return max(0.0, min(10.0, float(match.group())))
        except ValueError:
            return 0.0

    @classmethod
    def _extract_findings(cls, result: dict[str, Any]) -> list[CommunityFinding]:
        """Parse the <findings>/<finding> block into structured findings.

        The XML parser collapses a single repeated child tag to a scalar and
        multiple to a list, so normalize both ``{"finding": {...}}`` and
        ``{"finding": [{...}, ...]}`` (and the rare bare-list) shapes.
        """
        findings_block = result.get("findings")
        raw_findings: list[Any]
        if isinstance(findings_block, dict):
            inner = findings_block.get("finding", findings_block)
            raw_findings = inner if isinstance(inner, list) else [inner]
        elif isinstance(findings_block, list):
            raw_findings = findings_block
        else:
            return []

        findings: list[CommunityFinding] = []
        for item in raw_findings:
            if isinstance(item, dict):
                summary = cls._extract_text_from_result(item.get("summary", "")).strip()
                explanation = cls._extract_text_from_result(
                    item.get("explanation", "")
                ).strip()
            else:
                summary = cls._extract_text_from_result(item).strip()
                explanation = ""
            if summary or explanation:
                findings.append(
                    CommunityFinding(summary=summary, explanation=explanation)
                )
        return findings
