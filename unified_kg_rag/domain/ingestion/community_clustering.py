# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Community clustering: the hierarchy of Leiden partitions of the entity
graph and its content-named communities (pure: no model call, no store).

Report generation, which asks a model about these communities, is
``adapters.ingestion.community_detector``'s.
"""

import math
from collections import defaultdict
from collections.abc import Callable, Collection
from typing import Any

import networkx as nx

from unified_kg_rag.domain.ingestion.base_processor import edge_weight
from unified_kg_rag.domain.models import HierarchicalCommunity
from unified_kg_rag.domain.models.config import CommunityDetectionConfig
from unified_kg_rag.shared import GraphError, get_logger
from unified_kg_rag.shared.utils import generate_stable_id

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


def community_label(level: int, community_id: str) -> str:
    """Display label of a community: its level and the head of its content id.

    A positional label (``L0_C3``, the community's rank in the sorted
    partition) renamed every unchanged community behind an added one, and the
    name reaches the report's record key and every parent's roll-up text, so
    a reset re-asked the report model for communities that had not changed.
    """
    return f"L{level}_C{community_id[:8]}"


def canonical_graph(graph: nx.Graph) -> nx.Graph:
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

    With its defaults, the same algorithm, arguments and result as
    ``graspologic.partition.leiden`` on an undirected networkx graph
    (modularity, randomness 0.001, edge ``weight`` defaulting to 1.0, one
    trial, no extra iteration); isolated nodes are not in the result. The
    stage passes ``CommunityDetectionConfig.trials`` (3) and
    ``extra_forced_iterations`` (2), where MS GraphRAG runs one trial of one
    iteration: a deliberate deviation (docs/design.md §4.3). It
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


def community_sort_key(nodes: set[Any]) -> str:
    """Order communities by their smallest member's string id (members are
    disjoint, so the key is unique within a partition)."""
    return min(str(node) for node in nodes)


def graph_modularity(graph: nx.Graph, partition: list[set[str]]) -> float:
    """Modularity of ``partition``; ``0.0`` for a graph without edges.

    NetworkX divides by the total edge weight, so an edgeless graph (entities
    but no relationships, such as an incremental delta of a one-entity
    document) would raise ``ZeroDivisionError``.
    """
    if graph.number_of_edges() == 0 or graph.size(weight="weight") == 0:
        return 0.0
    return float(nx.community.modularity(graph, partition))


def entity_subgraph(graph: nx.Graph) -> nx.Graph:
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


def with_usable_weights(graph: nx.Graph) -> nx.Graph:
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


def small_component_nodes(
    graph: nx.Graph,
    min_size: int,
    whole_graph: Callable[[list[set[Any]]], set[Any]] | None = None,
) -> set[Any]:
    """The entities of ``graph``'s connected components of fewer than
    ``min_size`` (``min_community_size``) entities, which get no community.

    An isolated entity, or a pair only linked to each other, cannot merge
    into a neighbouring community, so each became a root community of
    its own: a report LLM call that restates one entity, and a report in
    every cut of the hierarchy, which pushed global search's cut past its
    cap on medium corpora (151 of 153 roots at 10k entities). MS GraphRAG
    clusters only the largest connected component (``use_lcc``); this
    leaves out only the components too small to form a community. Leiden
    still partitions the whole graph, so the other communities are the
    ones it finds with them (``hierarchical_communities`` only
    skips these); the left-out entities stay reachable through local
    search.

    An incremental run's graph holds only its delta, so a new entity
    whose one relationship is to a stored entity is a small component
    here but may not be in the whole graph: a component ``whole_graph``
    finds large enough there is clustered, as a full build would.
    """
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


class ClusterNode:
    """A community of the hierarchy under construction: its ``members`` (the
    nodes of the graph partitioned: entities, or the communities of the level
    below) and the finer communities it was split into (``children``, none
    for a leaf)."""

    __slots__ = ("members", "children", "level")

    def __init__(
        self,
        members: set[Any],
        children: list["ClusterNode"] | None = None,
        level: int = 0,
    ) -> None:
        self.members = members
        self.children: list[ClusterNode] = children or []
        self.level = level


def _postorder(roots: list[ClusterNode]) -> list[ClusterNode]:
    """Every node under ``roots``, each after its children (iteratively: a
    chain of splits can be deeper than the recursion limit)."""
    order: list[ClusterNode] = []
    stack = list(roots)
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(node.children)
    order.reverse()
    return order


def _leaves(roots: list[ClusterNode]) -> list[ClusterNode]:
    return [node for node in _postorder(roots) if not node.children]


def base_partition(
    graph: nx.Graph, config: CommunityDetectionConfig, *, entity_level: bool = True
) -> list[set[Any]]:
    """The finest communities of ``graph`` (the leaves of
    :func:`partition_tree`), in canonical order."""
    return sorted(
        (
            leaf.members
            for leaf in _leaves(
                partition_tree(graph, config, entity_level=entity_level)
            )
        ),
        key=community_sort_key,
    )


def partition_tree(
    graph: nx.Graph, config: CommunityDetectionConfig, *, entity_level: bool = True
) -> list[ClusterNode]:
    """Partition ``graph`` into communities: the entity graph
    (``entity_level``), or a cluster graph of the level below.

    The roots are one Leiden partition of ``graph``; a root over
    ``max_cluster_size`` members is split (:func:`split_large_communities`)
    and stays the parent of its pieces, as graspologic's
    ``hierarchical_leiden`` (MS GraphRAG's clustering) keeps it.

    Raises:
        GraphError: Leiden failed. Silently falling back would truncate the
            hierarchy or leave no communities to report on.
    """
    # Leiden needs some edge weight: on a graph whose every edge weighs 0
    # (strength-0 relationships, clamped by ``edge_weight``; an incremental
    # delta clusters only the new document's subgraph) graspologic_native
    # panics with a ``BaseException`` that ends the run and every resume.
    # Such a graph has no structure to partition, like an edgeless one.
    if graph.size(weight="weight") == 0:
        logger.warning(
            "Graph has no weighted edges. Treating each node as a separate "
            "community."
        )
        return [ClusterNode({node}) for node in sorted(graph.nodes(), key=str)]

    if config.auto_resolution:
        resolution = find_optimal_resolution(graph, config)
    else:
        resolution = config.resolution

    communities = leiden_communities(graph, resolution, config)

    # Every leaf is bounded by max_cluster_size: in entities at level 0, in
    # child communities above (a node of a cluster graph is a whole
    # community), so a leaf report and a parent's roll-up pack a bounded
    # number of entities or child report summaries. Only entity communities
    # are held to min_community_size: merging cluster-graph nodes would fold
    # a level into the one below it.
    roots = split_large_communities(
        communities, graph, resolution, config.max_cluster_size, config
    )
    if entity_level and config.min_community_size > 1:
        roots = _merge_small_leaves(roots, graph, config)
    return sorted(roots, key=lambda root: community_sort_key(root.members))


def _merge_small_leaves(
    roots: list[ClusterNode], graph: nx.Graph, config: CommunityDetectionConfig
) -> list[ClusterNode]:
    """Merge the leaves under ``min_community_size`` into their strongest
    neighbour (``merge_small_communities``) and re-derive the parents: each
    is the union of its remaining children, a parent left with one child is
    that child, and one left with none is gone."""
    leaves = sorted(_leaves(roots), key=lambda leaf: community_sort_key(leaf.members))
    merged = merge_small_communities(
        {i: set(leaf.members) for i, leaf in enumerate(leaves)},
        graph,
        config.min_community_size,
        config.max_cluster_size,
    )
    kept: dict[int, ClusterNode | None] = {}
    for i, leaf in enumerate(leaves):
        members = merged.get(i)
        if members is not None:
            leaf.members = members
        kept[id(leaf)] = leaf if members is not None else None
    for node in _postorder(roots):
        if not node.children:
            continue
        children = [
            child for child in (kept[id(c)] for c in node.children) if child is not None
        ]
        if len(children) < 2:
            kept[id(node)] = children[0] if children else None
            continue
        node.children = children
        node.members = set().union(*(child.members for child in children))
        kept[id(node)] = node
    return [root for root in (kept[id(r)] for r in roots) if root is not None]


def leiden_communities(
    graph: nx.Graph, resolution: float, config: CommunityDetectionConfig
) -> dict[int, set[Any]]:
    """Leiden partition of ``graph`` as canonically numbered communities.

    Raises:
        GraphError: Leiden failed.
    """
    try:
        partition_dict = leiden(
            canonical_graph(graph),
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
    return partition_dict_to_communities(partition_dict, graph)


def split_large_communities(
    communities: dict[int, set[Any]],
    graph: nx.Graph,
    resolution: float,
    max_size: int,
    config: CommunityDetectionConfig,
) -> list[ClusterNode]:
    """Re-partition every community above ``max_size`` members, recursively:
    the trees of ``communities`` (one root each), whose leaves fit
    ``max_size`` or have no finer partition.

    Modularity has a resolution limit: on a large graph, one Leiden pass
    merges well-separated groups into communities that grow with the
    corpus. A level-0 report then sees only the ``max_entities_per_report``
    best-connected members of such a leaf (MS GraphRAG bounds its leaves
    with ``max_cluster_size`` the same way), and a parent's roll-up only
    the child summaries that fit its token budget. Leiden run on the
    community's own subgraph judges it against that subgraph alone, so it
    splits. A community Leiden leaves whole (one piece) stays a leaf.

    The split community stays as the parent of its pieces, as in
    graspologic's ``hierarchical_leiden``: the pass over the whole graph is
    the best partition at its scale, and regrouping the pieces afresh (one
    Leiden pass over their cluster graph) recovered it worse on planted and
    LFR benchmark graphs (``test_community_hierarchy_quality``).
    """
    roots = [ClusterNode(set(nodes)) for nodes in communities.values()]
    pending = list(roots)
    split = 0
    while pending:
        node = pending.pop()
        if len(node.members) <= max_size:
            continue
        subgraph = graph.subgraph(node.members)
        pieces = (
            list(leiden_communities(subgraph, resolution, config).values())
            if subgraph.size(weight="weight") > 0
            else []
        )
        if len(pieces) < 2:
            logger.info(
                "Community of %s members has no finer partition; keeping "
                "it above max_cluster_size %s",
                len(node.members),
                max_size,
            )
            continue
        split += 1
        node.children = [ClusterNode(piece) for piece in pieces]
        pending.extend(node.children)
    if split:
        logger.info(
            "Split %s communities above max_cluster_size %s",
            split,
            max_size,
        )
    return roots


def partition_dict_to_communities(
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
    ordered = sorted(groups.values(), key=community_sort_key)
    if graph is not None:
        # Leiden leaves isolated nodes out of its result; each becomes its
        # own community so the partition covers the graph (modularity
        # rejects anything less).
        missing = [node for node in graph.nodes() if node not in partition_dict]
        ordered.extend({node} for node in sorted(missing, key=str))
    return dict(enumerate(ordered))


def find_optimal_resolution(graph: nx.Graph, config: CommunityDetectionConfig) -> float:
    best_resolution = config.resolution
    best_modularity = -1.0

    # Size short-circuit: the sweep runs a full Leiden partition + modularity
    # computation for EACH candidate (~10x the base cost) at every hierarchy
    # level. On a large graph that dominates the analysis phase, so above the
    # configured node cap we skip the sweep and use the fixed resolution.
    max_nodes = config.auto_resolution_max_nodes
    if graph.number_of_nodes() > max_nodes:
        logger.info(
            "Graph has %s nodes (> %s); skipping the auto-resolution sweep "
            "and using the fixed resolution %s.",
            graph.number_of_nodes(),
            max_nodes,
            best_resolution,
        )
        return best_resolution

    resolution_candidates = config.auto_resolution_candidates
    canonical = canonical_graph(graph)

    for resolution in resolution_candidates:
        try:
            partition_dict = leiden(
                canonical,
                resolution=resolution,
                random_seed=config.random_state,
                trials=1,
            )

            communities = partition_dict_to_communities(partition_dict, graph)
            partition = list(communities.values())

            if len(partition) < 2:
                continue

            modularity = graph_modularity(graph, partition)

            if modularity > best_modularity:
                best_modularity = modularity
                best_resolution = resolution

        except Exception as e:  # noqa: BLE001 - a failed resolution is skipped
            logger.warning("Resolution %s failed: %s", resolution, e)
            continue

    logger.info(
        "Auto-selected resolution %s with modularity %.4f",
        best_resolution,
        best_modularity,
    )
    return best_resolution


def merge_small_communities(
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


def build_cluster_graph(partition: list[set[str]], graph: nx.Graph) -> nx.Graph:
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


def _set_heights(roots: list[ClusterNode], base: int = 0) -> None:
    """Set each node's ``level`` to ``base`` plus its height (0 for a leaf)."""
    for node in _postorder(roots):
        node.level = base + (
            1 + max(child.level - base for child in node.children)
            if node.children
            else 0
        )


def _truncate(roots: list[ClusterNode], max_levels: int) -> list[ClusterNode]:
    """``roots`` without the nodes at level ``max_levels`` or above (their
    children become roots)."""
    kept: list[ClusterNode] = []
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node.level >= max_levels:
            stack.extend(node.children)
        else:
            kept.append(node)
    return sorted(kept, key=lambda root: community_sort_key(root.members))


def _tree_partitions(roots: list[ClusterNode]) -> list[list[set[Any]]]:
    """The hierarchy under ``roots`` as :func:`build_hierarchy` returns it.

    Level ``L`` holds every node at level ``L`` or below whose parent is
    above ``L`` (or which is a root): a node at level ``L`` groups its
    children, a lower one stands alone (``hierarchical_communities`` lets
    it stand for itself, so it is no new community).
    """
    nodes = _postorder(roots)
    parent_level: dict[int, int] = {}
    for node in nodes:
        for child in node.children:
            parent_level[id(child)] = node.level

    def frontier(level: int) -> list[ClusterNode]:
        return sorted(
            (
                node
                for node in nodes
                if node.level <= level and parent_level.get(id(node), level + 1) > level
            ),
            key=lambda node: community_sort_key(node.members),
        )

    below = frontier(0)
    partitions: list[list[set[Any]]] = [[set(node.members) for node in below]]
    for level in range(1, max((root.level for root in roots), default=0) + 1):
        index = {id(node): i for i, node in enumerate(below)}
        current = frontier(level)
        partitions.append(
            [
                (
                    {index[id(child)] for child in node.children}
                    if node.level == level
                    else {index[id(node)]}
                )
                for node in current
            ]
        )
        below = current
    return partitions


def _coarsen(
    roots: list[ClusterNode], graph: nx.Graph, config: CommunityDetectionConfig
) -> list[ClusterNode] | None:
    """Group ``roots`` (entity communities) into coarser communities, all
    above the highest root: a partition tree of their cluster graph, or
    ``None`` when it has none worth a level."""
    cluster_graph = build_cluster_graph([root.members for root in roots], graph)
    if cluster_graph.number_of_nodes() < 2:
        logger.info(
            "Cluster graph has only %s nodes, stopping the hierarchy",
            cluster_graph.number_of_nodes(),
        )
        return None
    logger.info(
        "Created cluster graph with %s nodes and %s edges",
        cluster_graph.number_of_nodes(),
        cluster_graph.number_of_edges(),
    )
    if cluster_graph.size(weight="weight") == 0:
        # No edge joins two communities: the normal top of the hierarchy,
        # not the edgeless input partition_tree warns of.
        logger.info("No edges between the top communities, stopping the hierarchy")
        return None

    groups = partition_tree(cluster_graph, config, entity_level=False)
    # Judged over the communities an edge joins to another: an isolated one
    # (an entity without relationships, a component of its own) is never
    # merged, so it must not make a single root over the others read as a
    # partition (and a level of its own, shifting every depth global search
    # cuts at). Such a root is no level, but the pieces it was split into
    # may be.
    linked = {node for node in cluster_graph if cluster_graph.degree(node) > 0}
    while True:
        holding = [group for group in groups if group.members & linked]
        if len(holding) != 1 or not holding[0].children:
            break
        groups = [g for g in groups if g is not holding[0]] + holding[0].children
    if len(holding) < 2 or all(len(group.members) == 1 for group in holding):
        logger.info("No meaningful coarser partition, stopping the hierarchy")
        return None

    # Every new community sits above every current root, so a level is one
    # scale: the roots of shallow and deep split trees are side by side
    # below the first coarser level.
    _set_heights(groups, base=max(root.level for root in roots) + 1)
    lifted: dict[int, ClusterNode] = {}
    for node in _postorder(groups):
        # A cluster-graph community over root indices -> one over entities.
        if not node.children and len(node.members) == 1:
            lifted[id(node)] = roots[next(iter(node.members))]
            continue
        children = (
            [lifted[id(child)] for child in node.children]
            if node.children
            else [roots[i] for i in sorted(node.members)]
        )
        lifted[id(node)] = ClusterNode(
            set().union(*(child.members for child in children)), children, node.level
        )
    return sorted(
        (lifted[id(group)] for group in groups),
        key=lambda root: community_sort_key(root.members),
    )


def build_hierarchy(
    graph: nx.Graph, config: CommunityDetectionConfig
) -> tuple[list[list[set[Any]]], float]:
    """The partitions of the entity ``graph`` at each level of the
    community hierarchy, the finest first (a level partitions the
    communities of the one below), and the modularity of the finest.

    The lowest levels are the partition tree of the entity graph (a split
    community over its pieces, :func:`partition_tree`), at its leaves'
    heights; above the highest of its roots, each round partitions the
    graph of the current top communities the same way, up to
    ``max_levels`` levels.
    """
    max_levels = config.max_levels
    roots = partition_tree(graph, config)
    _set_heights(roots)
    roots = _truncate(roots, max_levels)
    leaves = [leaf.members for leaf in _leaves(roots)]
    base_modularity = graph_modularity(graph, leaves)
    logger.info(
        "Base partition found with %s communities and modularity %.4f",
        len(leaves),
        base_modularity,
    )

    while roots and max(root.level for root in roots) + 1 < max_levels:
        coarser = _coarsen(roots, graph, config)
        if coarser is None:
            break
        roots = _truncate(coarser, max_levels)

    partitions = _tree_partitions(roots)
    logger.info("Hierarchy construction completed with %s levels", len(partitions))
    return partitions, base_modularity


def hierarchical_communities(
    partitions: list[list[set[str]]], unclustered: Collection[Any] = ()
) -> tuple[dict[str, HierarchicalCommunity], dict[str, str]]:
    """The communities of each level of ``partitions`` (see
    :func:`build_hierarchy`) by id, each named after its content
    (:func:`community_content_id`, :func:`community_label`), and the
    level-0 community of each entity; the ``unclustered`` entities get
    none."""
    all_communities: dict[str, HierarchicalCommunity] = {}
    node_to_community_l0: dict[str, str] = {}
    unclustered = set(unclustered)

    logger.info("Processing %s levels of partitions", len(partitions))

    l0_partition = partitions[0]
    # Keys are partition member identities, which Leiden may emit as ints
    # or node-label strings depending on the input graph.
    level_0_communities: dict[Any, str] = {}

    for i, nodes in enumerate(l0_partition):
        if nodes <= unclustered:
            # A too-small component (or part of one): no community. It
            # has no edge to another community, so no coarser community
            # contains it either.
            continue
        comm_id = community_content_id(0, nodes)

        for node_id in nodes:
            node_to_community_l0[node_id] = comm_id

        community = HierarchicalCommunity(
            community_id=comm_id,
            level=0,
            nodes=nodes,
            parent_id=None,
            label=community_label(0, comm_id),
        )
        all_communities[comm_id] = community
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
                    child_comm = all_communities.get(previous_level_map[cluster_idx])
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
                    label=community_label(level, parent_comm_id),
                )
                all_communities[parent_comm_id] = parent_community
                current_level_map[i] = parent_comm_id

        logger.info(
            "Created %s level %s communities",
            sum(
                all_communities[cid].level == level
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
        len(all_communities),
        len(partitions),
    )
    return all_communities, node_to_community_l0
