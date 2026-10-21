# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from typing import Any

import networkx as nx

from unified_kg_rag.adapters.llm.chain_factory import (
    create_robust_xml_output_parser,
    setup_chain,
)
from unified_kg_rag.adapters.llm.retry import transient_error_predicate
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.ingestion.base_processor import (
    edge_weight,
    merge_attributes,
)
from unified_kg_rag.domain.ingestion.graph_builder import edge_relationships
from unified_kg_rag.domain.models import (
    Community,
    CommunityFinding,
    CommunityMetrics,
    CommunityReport,
    Config,
    Constants,
    HierarchicalCommunity,
    ModelPurpose,
    ModelRole,
    TextUnit,
)
from unified_kg_rag.domain.prompts import CommunityReportPrompt
from unified_kg_rag.shared import GraphError, get_logger
from unified_kg_rag.shared.utils import generate_stable_id
from unified_kg_rag.shared.utils.langchain import BATCH_ITEM_FAILED, BatchProcessor

logger = get_logger(__name__)


def community_content_id(level: int, nodes: Any) -> str:
    """Content-derived community id: the level plus the sorted member entity ids.

    Positional ids (``L0_C0`` ...) named a *different* community on every run, so
    an incremental run that clusters only the delta subgraph upserted its
    ``L0_C0`` over the full corpus's ``L0_C0`` (and its report, whose id hashes
    the community id). A content id only matches when the membership matches,
    so delta communities are appended next to the existing ones instead of
    replacing them, and an unchanged community keeps its id across rebuilds.
    """
    members = "\x1f".join(sorted(str(node) for node in nodes))
    return generate_stable_id(f"community:{level}:{members}")


def _canonical_graph(graph: nx.Graph) -> nx.Graph:
    """Copy ``graph`` with nodes and edges inserted in sorted order.

    Leiden's result depends on node/edge insertion order even with a fixed seed,
    and the graph builder inserts in extraction order, which varies with LLM
    response order. Sorting first makes the same graph partition the same way.
    """
    canonical = nx.Graph()
    canonical.graph.update(graph.graph)
    canonical.add_nodes_from(sorted(graph.nodes(data=True), key=lambda n: str(n[0])))
    canonical.add_edges_from(
        sorted(
            graph.edges(data=True),
            key=lambda e: tuple(sorted((str(e[0]), str(e[1])))),
        )
    )
    return canonical


def leiden(
    graph: nx.Graph,
    *,
    resolution: float = 1.0,
    random_seed: int | None = None,
    trials: int = 1,
    extra_forced_iterations: int = 0,
) -> dict[Any, int]:
    """Leiden partition of ``graph``: node -> community label.

    Same algorithm, arguments and result as ``graspologic.partition.leiden``
    on an undirected networkx graph (modularity, randomness 0.001, edge
    ``weight`` defaulting to 1.0); isolated nodes are not in the result. It
    calls ``graspologic_native`` directly because importing any graspologic
    module runs ``graspologic/__init__``, which takes ~10 s and imports boto3
    through gensim and smart_open.

    Raises:
        ValueError: An argument is out of range, ``graph`` is directed or a
            multigraph, or two nodes have the same ``str``.
        GraphError: ``graspologic_native`` panicked.
    """
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Leiden supports only undirected non-multi graphs")
    if resolution <= 0:
        raise ValueError("resolution must be a positive float")
    if random_seed is not None and random_seed <= 0:
        raise ValueError("random_seed must be a positive integer")
    if trials < 1:
        raise ValueError("trials must be a positive integer")
    if extra_forced_iterations < 0:
        raise ValueError("extra_forced_iterations must be a non-negative integer")

    import graspologic_native

    # The native module takes string node ids; map them back afterwards.
    originals: dict[str, Any] = {}

    def node_key(node: Any) -> str:
        key = str(node)
        if originals.setdefault(key, node) != node:
            raise ValueError(f"Two distinct nodes share the string id {key!r}")
        return key

    edges = [
        (node_key(source), node_key(target), float(weight))
        for source, target, weight in graph.edges(data="weight", default=1.0)
    ]
    try:
        _quality, partition = graspologic_native.leiden(
            edges=edges,
            starting_communities=None,
            resolution=resolution,
            randomness=0.001,
            iterations=extra_forced_iterations + 1,
            use_modularity=True,
            seed=random_seed,
            trials=trials,
        )
    except BaseException as e:
        # A Rust panic surfaces as pyo3's PanicException, a BaseException
        # (not importable: pyo3 creates its module at runtime), so the
        # stage's error handling would not see it and every resume would die
        # the same way. KeyboardInterrupt and SystemExit pass through.
        if type(e).__module__ != "pyo3_runtime":
            raise
        raise GraphError(f"graspologic_native panicked: {e}") from e
    return {originals[key]: label for key, label in partition.items()}


def _community_sort_key(nodes: set[Any]) -> str:
    """Order communities by their smallest member's string id (members are
    disjoint, so the key is unique within a partition)."""
    return min(str(node) for node in nodes)


def _modularity(graph: nx.Graph, partition: list[set[str]]) -> float:
    """Modularity of ``partition``; ``0.0`` for a graph without edges.

    NetworkX divides by the total edge weight, so an edgeless graph (entities
    but no relationships, such as an incremental delta of a one-entity
    document) would raise ``ZeroDivisionError``.
    """
    if graph.number_of_edges() == 0 or graph.size(weight="weight") == 0:
        return 0.0
    return float(nx.community.modularity(graph, partition))


class CommunityDetector:
    def __init__(
        self,
        config: Config,
        show_progress: bool = True,
        *,
        providers: Providers | None = None,
    ) -> None:
        self.config = config
        self.providers = Providers.resolve(config, providers)
        self.community_detection_config = self.config.graph.community_detection
        self.ignore_errors = self.config.processing.ignore_errors

        self.graph: nx.Graph = nx.Graph()
        self.all_communities: dict[str, HierarchicalCommunity] = {}
        self.node_to_community_l0: dict[str, str] = {}
        # Entities of the components too small to form a community.
        self.unclustered: set[Any] = set()
        self.base_modularity: float = 0.0
        self.show_progress = show_progress

        self.factory = self.providers.llm_factory
        self.batch_processor = BatchProcessor.from_processing(
            self.config.processing,
            is_transient_error=transient_error_predicate(self.factory),
        )

        if self.community_detection_config.report_generation.enabled:
            parser = create_robust_xml_output_parser(
                model_purpose=ModelPurpose.INGESTION,
                factory=self.factory,
                enable_output_fixing=self.config.fixing.enabled,
                output_fixing_model_id=self.config.model_id(ModelRole.FIXING),
                min_output_tokens=CommunityReportPrompt.output_floor(self.config),
                output_tags=[
                    "community_name",
                    "summary",
                    "rating",
                    "rating_explanation",
                    "findings",
                ],
            )
            self.report_generator = setup_chain(
                model_purpose=ModelPurpose.INGESTION,
                factory=self.factory,
                model_id=self.config.model_id(ModelRole.COMMUNITY_REPORT),
                prompt_class=CommunityReportPrompt,
                min_output_tokens=CommunityReportPrompt.output_floor(self.config),
                parser=parser,
                custom_prompts=self.config.custom_prompts,
            )

    def __call__(
        self,
        graph: nx.Graph,
        whole_graph: Callable[[list[set[Any]]], set[Any]] | None = None,
    ) -> "CommunityDetector":
        """Detect the communities of ``graph``.

        ``whole_graph`` (an incremental run) returns the entities of the
        given small components that reach ``min_community_size`` entities in
        the whole graph: they are clustered (see ``_small_component_nodes``).
        """
        self.graph = self._with_usable_weights(self._entity_subgraph(graph))
        graph = self.graph
        self.unclustered = self._small_component_nodes(graph, whole_graph)
        logger.info(
            "Starting community detection on graph with %s nodes and %s edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        self.detect_communities()
        return self

    @staticmethod
    def _entity_subgraph(graph: nx.Graph) -> nx.Graph:
        """Return the entity-only graph that communities are detected on.

        The knowledge graph also carries claim nodes (``node_type="claim"``,
        see ``GraphBuilder``). Communities group entities, as in GraphRAG:
        claim nodes would otherwise join entity communities, and every claim
        whose subject/object did not resolve became an isolated singleton
        community costing its own report LLM call. Nodes without a
        ``node_type`` are treated as entities.
        """
        non_entity_nodes = [
            node
            for node, data in graph.nodes(data=True)
            if data.get("node_type", "entity") != "entity"
        ]
        if not non_entity_nodes:
            return graph
        logger.info(
            "Excluding %s non-entity (claim) nodes from community detection",
            len(non_entity_nodes),
        )
        entity_graph = graph.copy()
        entity_graph.remove_nodes_from(non_entity_nodes)
        return entity_graph

    def _small_component_nodes(
        self,
        graph: nx.Graph,
        whole_graph: Callable[[list[set[Any]]], set[Any]] | None = None,
    ) -> set[Any]:
        """The entities of ``graph``'s connected components of fewer than
        ``min_community_size`` entities, which get no community.

        An isolated entity, or a pair only linked to each other, cannot merge
        into a neighbouring community, so each became a root community of
        its own: a report LLM call that restates one entity, and a report in
        every cut of the hierarchy, which pushed global search's cut past its
        cap on medium corpora (151 of 153 roots at 10k entities). MS GraphRAG
        clusters only the largest connected component (``use_lcc``); this
        leaves out only the components too small to form a community. Leiden
        still partitions the whole graph, so the other communities are the
        ones it finds with them (``_process_hierarchical_partitions`` only
        skips these); the left-out entities stay reachable through local
        search.

        An incremental run's graph holds only its delta, so a new entity
        whose one relationship is to a stored entity is a small component
        here but may not be in the whole graph: a component ``whole_graph``
        finds large enough there is clustered, as a full build would.
        """
        min_size = self.community_detection_config.min_community_size
        components = [
            component
            for component in nx.connected_components(graph)
            if len(component) < min_size
        ]
        if components and whole_graph is not None:
            large = whole_graph(components)
            components = [c for c in components if large.isdisjoint(c)]
        small = {node for component in components for node in component}
        if small:
            logger.info(
                "Leaving %s entities in connected components of fewer than %s "
                "entities (graph.community_detection.min_community_size) out "
                "of the communities",
                len(small),
                min_size,
            )
        return small

    @staticmethod
    def _with_usable_weights(graph: nx.Graph) -> nx.Graph:
        """Return ``graph`` with every edge weight usable (see ``edge_weight``).

        New extractions are already clean (``edge_weight`` in the parser), but
        a graph read back from a store or cache written before that may still
        carry a NaN, infinite, negative, denormal or huge weight, and Leiden
        panics or fails on one every time the stage is resumed. Such weights
        are replaced on a copy, so the caller's graph is left as it is.
        """
        bad = {
            (source, target): edge_weight(weight)
            for source, target, weight in graph.edges(data="weight", default=1.0)
            if not (isinstance(weight, int | float) and edge_weight(weight) == weight)
        }
        if not bad:
            return graph
        logger.warning(
            "Replacing %s non-finite, negative or out-of-range edge weights "
            "before community detection",
            len(bad),
        )
        graph = graph.copy()
        for (source, target), weight in bad.items():
            graph.edges[source, target]["weight"] = weight
        return graph

    def detect_communities(self) -> None:
        if not self.graph or self.graph.number_of_nodes() == 0:
            logger.warning("Community detection skipped for empty or invalid graph.")
            return

        self.analyze_hierarchy()
        logger.info(
            "Detected %s communities across all levels.", len(self.all_communities)
        )

    def analyze_hierarchy(self) -> None:
        base_partition = self._get_base_partition()

        self.base_modularity = _modularity(self.graph, base_partition)

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

            cluster_graph = self._create_cluster_graph(current_partition, current_graph)

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
            if cluster_graph.size(weight="weight") == 0:
                # No edge joins two communities: the normal top of the
                # hierarchy, not the edgeless input _get_base_partition warns of.
                logger.info(
                    "No edges between the communities of level %s, stopping "
                    "hierarchy construction",
                    level - 1,
                )
                break

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

        logger.info("Hierarchy construction completed with %s levels", len(partitions))
        self._process_hierarchical_partitions(partitions)

    def _get_base_partition(self, graph: nx.Graph | None = None) -> list[set[Any]]:
        """Partition ``graph`` (the entity graph by default) into communities.

        The communities come back in canonical order (see
        ``_partition_dict_to_communities``), because the next level's cluster
        graph numbers its nodes by position in this list.

        Raises:
            GraphError: Leiden failed. Silently falling back would truncate the
                hierarchy or leave no communities to report on.
        """
        target_graph = graph if graph is not None else self.graph

        # Leiden needs some edge weight: on a graph whose every edge weighs 0
        # (strength-0 relationships, clamped by ``edge_weight``; an incremental
        # delta clusters only the new document's subgraph) graspologic_native
        # panics with a ``BaseException`` that ends the run and every resume.
        # Such a graph has no structure to partition, like an edgeless one.
        if target_graph.size(weight="weight") == 0:
            logger.warning(
                "Graph has no weighted edges. Treating each node as a separate "
                "community."
            )
            return [{node} for node in sorted(target_graph.nodes(), key=str)]

        config = self.community_detection_config

        if config.auto_resolution:
            resolution = self._find_optimal_resolution(target_graph)
        else:
            resolution = config.resolution

        communities = self._leiden_communities(target_graph, resolution)

        # Every level is bounded by max_cluster_size: in entities at level 0,
        # in child communities above (a node of a cluster graph is a whole
        # community), so a parent's roll-up packs a bounded number of child
        # report summaries. Only entity communities are held to
        # min_community_size: merging cluster-graph nodes would fold a level
        # into the one below it.
        communities = self._split_large_communities(
            communities, target_graph, resolution, config.max_cluster_size
        )
        if graph is None:
            if config.min_community_size > 1:
                communities = self._merge_small_communities(
                    communities,
                    target_graph,
                    config.min_community_size,
                    config.max_cluster_size,
                )

        return sorted(communities.values(), key=_community_sort_key)

    def _leiden_communities(
        self, graph: nx.Graph, resolution: float
    ) -> dict[int, set[Any]]:
        """Leiden partition of ``graph`` as canonically numbered communities.

        Raises:
            GraphError: Leiden failed.
        """
        config = self.community_detection_config
        try:
            partition_dict = leiden(
                _canonical_graph(graph),
                resolution=resolution,
                random_seed=config.random_state,
                trials=config.trials,
                extra_forced_iterations=config.extra_forced_iterations,
            )
        except Exception as e:
            raise GraphError(
                f"Leiden partitioning failed on a graph with "
                f"{graph.number_of_nodes()} nodes and "
                f"{graph.number_of_edges()} edges: {e}"
            ) from e
        return self._partition_dict_to_communities(partition_dict, graph)

    def _split_large_communities(
        self,
        communities: dict[int, set[Any]],
        graph: nx.Graph,
        resolution: float,
        max_size: int,
    ) -> dict[int, set[Any]]:
        """Re-partition every community above ``max_size`` members, recursively.

        Modularity has a resolution limit: on a large graph, one Leiden pass
        merges well-separated groups into communities that grow with the
        corpus. A level-0 report then sees only the ``max_entities_per_report``
        best-connected members of such a leaf (MS GraphRAG bounds its leaves
        with ``max_cluster_size`` the same way), and a parent's roll-up only
        the child summaries that fit its token budget. Leiden run on the
        community's own subgraph judges it against that subgraph alone, so it
        splits. A community Leiden leaves whole (one piece) stays as it is.
        """
        done: list[set[Any]] = []
        pending = list(communities.values())
        split = 0
        while pending:
            nodes = pending.pop()
            if len(nodes) <= max_size:
                done.append(nodes)
                continue
            subgraph = graph.subgraph(nodes)
            pieces = (
                list(self._leiden_communities(subgraph, resolution).values())
                if subgraph.size(weight="weight") > 0
                else []
            )
            if len(pieces) < 2:
                logger.info(
                    "Community of %s members has no finer partition; keeping "
                    "it above max_cluster_size %s",
                    len(nodes),
                    max_size,
                )
                done.append(nodes)
                continue
            split += 1
            pending.extend(pieces)
        if split:
            logger.info(
                "Split %s communities above max_cluster_size %s",
                split,
                max_size,
            )
        return dict(enumerate(sorted(done, key=_community_sort_key)))

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
        partition_dict: dict[Any, int],
        graph: nx.Graph | None = None,
    ) -> dict[int, set[Any]]:
        """Group a node -> label map into communities, numbered canonically.

        ``graspologic_native`` returns a Rust ``HashMap``, whose iteration
        order (and so the labels' first-seen order) changes from process to
        process. The communities are renumbered by their smallest member, so
        the small-community merge and the next level's cluster-graph node
        numbers do not depend on that order.
        """
        groups: dict[int, set[Any]] = defaultdict(set)
        for node, label in partition_dict.items():
            groups[label].add(node)
        ordered = sorted(groups.values(), key=_community_sort_key)
        if graph is not None:
            # Leiden leaves isolated nodes out of its result; each becomes its
            # own community so the partition covers the graph (modularity
            # rejects anything less).
            missing = [node for node in graph.nodes() if node not in partition_dict]
            ordered.extend({node} for node in sorted(missing, key=str))
        return dict(enumerate(ordered))

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
        canonical = _canonical_graph(graph)

        for resolution in resolution_candidates:
            try:
                partition_dict = leiden(
                    canonical,
                    resolution=resolution,
                    random_seed=self.community_detection_config.random_state,
                    trials=1,
                )

                communities = self._partition_dict_to_communities(partition_dict, graph)
                partition = list(communities.values())

                if len(partition) < 2:
                    continue

                modularity = _modularity(graph, partition)

                if modularity > best_modularity:
                    best_modularity = modularity
                    best_resolution = resolution

            except Exception as e:
                logger.warning("Resolution %s failed: %s", resolution, e)
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
        max_size: int | None = None,
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

                # Total edge weight to each neighboring community: the merge
                # goes where the community is most strongly tied, not where it
                # merely has the most edges. Summed with ``fsum``: ``nodes`` is
                # a set iterated in hash-seed order, and a plain float sum in
                # that order can split an exact tie by one ulp, overriding the
                # lowest-label rule below differently from run to run.
                neighbor_weights: dict[int, list[float]] = defaultdict(list)
                for node in nodes:
                    for neighbor, data in graph[node].items():
                        if neighbor in node_to_comm:
                            neighbor_comm = node_to_comm[neighbor]
                            if (
                                neighbor_comm != small_comm_id
                                and neighbor_comm in communities
                                # A merge must not push a community back
                                # above max_cluster_size.
                                and (
                                    max_size is None
                                    or len(communities[neighbor_comm]) + len(nodes)
                                    <= max_size
                                )
                            ):
                                neighbor_weights[neighbor_comm].append(
                                    data.get("weight", 1.0)
                                )

                if neighbor_weights:
                    # Ties go to the lowest (canonical) label, not to whichever
                    # neighbor a set iteration happened to reach first.
                    totals = {c: math.fsum(w) for c, w in neighbor_weights.items()}
                    target_comm = max(totals, key=lambda x: (totals[x], -x))
                    communities[target_comm].update(nodes)
                    for node in nodes:
                        node_to_comm[node] = target_comm
                    del communities[small_comm_id]
                    merged_any = True

            if not merged_any:
                break

        return communities

    def _create_cluster_graph(
        self, partition: list[set[str]], graph: nx.Graph
    ) -> nx.Graph:
        # ``graph`` is the graph whose nodes the partition members refer to:
        # the original entity graph at level 1, or the previous level's
        # cluster graph at level >= 2.
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
            if nodes <= self.unclustered:
                # A too-small component (or part of one): no community. It
                # has no edge to another community, so no coarser community
                # contains it either.
                continue
            comm_id = community_content_id(0, nodes)

            for node_id in nodes:
                self.node_to_community_l0[node_id] = comm_id

            community = HierarchicalCommunity(
                community_id=comm_id,
                level=0,
                nodes=nodes,
                parent_id=None,
                label=f"L0_C{i}",
            )
            self.all_communities[comm_id] = community
            level_0_communities[i] = comm_id

        logger.info("Created %s level 0 communities", len(level_0_communities))

        previous_level_map = level_0_communities

        for level in range(1, len(partitions)):
            partition = partitions[level]
            current_level_map: dict[Any, str] = {}

            for i, cluster_nodes in enumerate(partition):
                parent_nodes: set[str] = set()
                children: list[HierarchicalCommunity] = []

                for cluster_idx in sorted(cluster_nodes, key=str):
                    if cluster_idx in previous_level_map:
                        child_comm = self.all_communities.get(
                            previous_level_map[cluster_idx]
                        )
                        if child_comm:
                            parent_nodes.update(child_comm.nodes)
                            children.append(child_comm)

                if len(children) == 1:
                    # A cluster Leiden left on its own (often an isolated one)
                    # would be a parent identical to its only child, embedded,
                    # indexed and retrieved once per level. The child stands
                    # for it at this level instead.
                    current_level_map[i] = children[0].community_id
                elif parent_nodes:
                    # The id derives from the members, so it is known only once
                    # every child has been folded in.
                    parent_comm_id = community_content_id(level, parent_nodes)
                    for child_comm in children:
                        child_comm.parent_id = parent_comm_id
                    parent_community = HierarchicalCommunity(
                        community_id=parent_comm_id,
                        level=level,
                        nodes=parent_nodes,
                        parent_id=None,
                        children_ids=[c.community_id for c in children],
                        label=f"L{level}_C{i}",
                    )
                    self.all_communities[parent_comm_id] = parent_community
                    current_level_map[i] = parent_comm_id

            logger.info(
                "Created %s level %s communities",
                sum(
                    self.all_communities[cid].level == level
                    for cid in current_level_map.values()
                ),
                level,
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
                # ``nodes`` is a set: its order of str ids follows the hash seed.
                "nodes": sorted(comm.nodes, key=str),
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
        # No entity, or none in a component large enough for a community.
        if len(self.unclustered) == self.graph.number_of_nodes():
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
        """The index suffix the communities and reports are written under.

        Only that placement key: the rest of a node's attributes (its
        document ``filters``, its own extracted attributes) describe that one
        entity. Copying the first node's map gave every report that entity's
        document filters, so an attribute filter kept or dropped reports by an
        arbitrary entity's documents; a report's filters come from its own
        text units (``_attach_report_lineage``).
        """
        key = Constants.INDEX.value
        attributes: Any = None
        if hasattr(self.graph, "graph") and self.graph.graph:
            attributes = self.graph.graph.get("attributes")
        else:
            attributes = next(
                (
                    data.get("attributes")
                    for _, data in self.graph.nodes(data=True)
                    if isinstance(data.get("attributes"), dict)
                ),
                None,
            )
        if isinstance(attributes, dict) and key in attributes:
            return {key: attributes[key]}
        return {}

    def _create_community_object(
        self, hier_comm: HierarchicalCommunity, attributes: dict[str, Any]
    ) -> Community:
        subgraph = self.graph.subgraph(hier_comm.nodes)
        max_entities = (
            self.community_detection_config.report_generation.max_entities_per_report
        )
        member_ids = sorted(hier_comm.nodes, key=str)
        entity_names = [
            self.graph.nodes[nid].get("name", nid) for nid in member_ids[:max_entities]
        ]

        # An edge carries every relationship between its two entities.
        relationship_ids = sorted(
            rel["id"]
            for _, _, data in subgraph.edges(data=True)
            for rel in edge_relationships(data)
            if rel.get("id") is not None
        )
        community_attributes = {
            **attributes,
            "size": len(hier_comm.nodes),
            "density": nx.density(subgraph),
            "entity_names": entity_names,
            "resolution": self.community_detection_config.resolution,
            "num_relationships": len(relationship_ids),
        }

        label = hier_comm.label or hier_comm.community_id
        return Community(
            id=hier_comm.community_id,
            short_id=label,
            name=f"Level {hier_comm.level} Community {label.rsplit('_C', 1)[-1]}",
            name_embedding=None,
            level=str(hier_comm.level),
            parent=hier_comm.parent_id or "",
            children=hier_comm.children_ids,
            entity_ids=member_ids,
            relationship_ids=relationship_ids,
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

    def generate_reports(
        self,
        communities: list[Community],
        text_units: list[TextUnit] | None = None,
    ) -> tuple[list[CommunityReport], list[str]]:
        """Generate one report per community.

        Args:
            communities: Communities to summarize.
            text_units: The corpus's text units, used to resolve each report's
                candidate source documents (community membership provenance,
                see ``_attach_report_lineage``). Omit to leave ``document_ids``
                empty; the report's ``text_unit_ids`` are recorded either way.

        Returns:
            One report per community that produced a non-empty LLM result, and
            the ids of the communities that got none (every community when
            generation raised under ``ignore_errors``), so the caller can retry
            their documents.
        """
        report_config = self.community_detection_config.report_generation
        if not report_config.enabled or not hasattr(self, "report_generator"):
            logger.info("Community report generation is disabled")
            return [], []

        if not communities:
            logger.warning("No communities provided for report generation")
            return [], []

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

            self._attach_report_lineage(reports, communities, text_units)

            if failed:
                logger.warning(
                    "%s of %s community reports had no result",
                    len(failed),
                    len(communities),
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
            return [], [community.id for community in communities]

        return reports, failed

    @staticmethod
    def _attach_report_lineage(
        reports: list[CommunityReport],
        communities: list[Community],
        text_units: list[TextUnit] | None,
    ) -> None:
        """Record each report's candidate source text units, documents and
        document ``filters``.

        A report is the only artifact in the index with no pointer back to its
        sources, so a retrieved report cannot be filtered by origin. Detection
        already unions its member entities' ``text_unit_ids`` onto the Community,
        so the report inherits them; the source documents come from those units.

        This is community MEMBERSHIP provenance, not report-input provenance.
        ``_prepare_report_input`` caps the prompt at ``max_entities_per_report``
        and a token budget, and the lineage deliberately ignores that cap: a
        community with ``e1`` from ``d1`` and ``e2`` from ``d2`` records
        ``document_ids=["d1", "d2"]`` even when only ``e1`` reached the prompt.
        The fields therefore may name material omitted from the report and do
        not establish sentence-level citation support; they exist so a consumer
        can filter reports by source or re-summarize them per reader.

        The report's ``filters`` attribute is the union of its text units'
        document filters, so an attribute filter matches a report when one of
        its documents matches, as it does an entity. Without text units, or
        when none has filters, the report has none and an attribute filter
        excludes it (fail-closed).

        Args:
            reports: Reports to annotate, mutated in place.
            communities: The communities the reports were generated from.
            text_units: Corpus text units; omit to leave ``document_ids`` empty.
        """
        unit_to_documents = {
            unit.id: list(unit.document_ids or []) for unit in (text_units or [])
        }
        filters_key = Constants.FILTERS.value
        unit_filters = {
            unit.id: filters
            for unit in (text_units or [])
            if isinstance(filters := (unit.attributes or {}).get(filters_key), dict)
            and filters
        }
        community_units = {c.id: list(c.text_unit_ids or []) for c in communities}

        for report in reports:
            unit_ids = community_units.get(report.community_id, [])
            report.text_unit_ids = unit_ids
            document_ids: set[str] = set()
            for unit_id in unit_ids:
                document_ids.update(unit_to_documents.get(unit_id, ()))
            # Sorted for a stable, reproducible record.
            report.document_ids = sorted(document_ids)
            attributes = {
                k: v for k, v in (report.attributes or {}).items() if k != filters_key
            }
            maps = [
                {filters_key: unit_filters[u]} for u in unit_ids if u in unit_filters
            ]
            filters = (merge_attributes(maps) or {}).get(filters_key)
            if filters:
                attributes[filters_key] = filters
            report.attributes = attributes

    def _run_report_batch(
        self,
        report_inputs: list[dict[str, Any]],
        communities: list[Community],
        graph_attributes: dict[str, Any],
    ) -> tuple[list[CommunityReport], list[str]]:
        """Run one batch of report inputs; return the reports and failed ids."""
        report_results = self.batch_processor.execute_with_fallback(
            items_to_process=report_inputs,
            prepare_inputs_func=self._create_report_chain_inputs,
            sequential_func=self.report_generator.invoke,
            task_name="Community report generation",
            show_progress=self.show_progress,
        )

        reports: list[CommunityReport] = []
        failed: list[str] = []
        for community, result in zip(communities, report_results, strict=True):
            if result and result is not BATCH_ITEM_FAILED:
                reports.append(
                    self._create_community_report(community, result, graph_attributes)
                )
            else:
                # Empty LLM result: report it so the caller retries the
                # community's documents instead of silently storing fewer reports.
                failed.append(community.id)
                logger.warning(
                    "No report generated for community '%s' (empty LLM result)",
                    community.id,
                )
        return reports, failed

    def _generate_reports_flat(
        self, communities: list[Community], graph_attributes: dict[str, Any]
    ) -> tuple[list[CommunityReport], list[str]]:
        """Original behaviour: every community summarized independently."""
        report_inputs = [self._prepare_report_input(c) for c in communities]
        return self._run_report_batch(report_inputs, communities, graph_attributes)

    def _generate_reports_with_rollup(
        self, communities: list[Community], graph_attributes: dict[str, Any]
    ) -> tuple[list[CommunityReport], list[str]]:
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
        total_failed: list[str] = []

        for level in sorted({_level(c) for c in communities}):
            level_communities = [c for c in communities if _level(c) == level]
            report_inputs = [
                self._prepare_report_input(c, reports_by_id) for c in level_communities
            ]
            level_reports, failed = self._run_report_batch(
                report_inputs, level_communities, graph_attributes
            )
            total_failed.extend(failed)
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
        # Members over the entity cap are left out as well: like a token
        # overflow, that rolls the sub-community reports in.
        truncated = len(candidate_ids) < sum(
            1 for eid in community.entity_ids or [] if eid in self.graph
        )
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
        # An edge carries every relationship between its two entities; each
        # is its own line, in the relationship's own direction.
        edges = sorted(
            (
                (data.get("source_id", u), data.get("target_id", v), data)
                for u, v, edge in self.graph.subgraph(selected_ids).edges(data=True)
                for data in edge_relationships(edge)
            ),
            key=lambda e: (
                -(self.graph.degree(e[0]) + self.graph.degree(e[1])),
                -float(e[2].get("weight") or 1.0),
                str(e[2].get("id", "")),
            ),
        )
        relationships: list[dict[str, Any]] = []
        for u, v, data in edges:
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
        if truncated and not sub_reports:
            logger.info(
                "Report context of community '%s' keeps %s of its %s entities",
                community.id,
                len(entities),
                len(community.entity_ids or []),
            )

        return {
            "community_id": community.id,
            "entities": entities,
            "relationships": relationships,
            "sub_community_reports": sub_reports,
            "content_length": report_config.content_length,
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
            "generation_model": self.config.model_id(ModelRole.COMMUNITY_REPORT),
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
            full_content="",
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
            if not value:  # an empty element (<summary></summary>)
                return ""
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
