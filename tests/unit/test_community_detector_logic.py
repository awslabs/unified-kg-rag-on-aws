# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for CommunityDetector's pure graph logic.

These exercise the Leiden-backed community detection, the small-community
merge-to-convergence loop, hierarchy construction (cluster graph + level
building), modularity/resolution selection, and the report-input shaping that
turns node ids into entity *names* — without touching Bedrock. Report
generation is disabled so ``__init__`` never builds an LLM chain; the Leiden
call (graspologic) is real CPU on tiny hand-built graphs.
"""

from __future__ import annotations

import networkx as nx
import pytest

from aws_graphrag.adapters.ingestion.community_detector import (
    CommunityDetector,
    HierarchicalCommunity,
)
from aws_graphrag.domain.models import Community, Config
from aws_graphrag.shared import GraphError

pytestmark = pytest.mark.unit


def _detector(
    *,
    report: bool = False,
    auto_resolution: bool = True,
    min_community_size: int = 3,
    max_levels: int = 5,
) -> CommunityDetector:
    """Construct a detector AWS-free (report generation off => no LLM chain)."""
    config = Config()
    cd_config = config.graph.community_detection
    cd_config.report_generation.enabled = report
    cd_config.auto_resolution = auto_resolution
    cd_config.min_community_size = min_community_size
    cd_config.max_levels = max_levels
    return CommunityDetector(config, show_progress=False)


def _two_triangle_graph() -> nx.Graph:
    """Two triangles joined by a single bridge edge -> two natural communities."""
    g = nx.Graph()
    for i in range(6):
        g.add_node(f"n{i}", name=f"N{i}")
    # triangle A
    g.add_edge("n0", "n1")
    g.add_edge("n1", "n2")
    g.add_edge("n0", "n2")
    # triangle B
    g.add_edge("n3", "n4")
    g.add_edge("n4", "n5")
    g.add_edge("n3", "n5")
    # bridge
    g.add_edge("n2", "n3")
    return g


class TestConstruction:
    def test_constructs_aws_free_without_report_chain(self) -> None:
        cd = _detector(report=False)
        assert not hasattr(cd, "report_generator")
        assert cd.all_communities == {}
        assert cd.base_modularity == 0.0


class TestPartitionDictToCommunities:
    def test_groups_nodes_by_label(self) -> None:
        partition = {"a": 0, "b": 0, "c": 1, "d": 1, "e": 1}
        out = CommunityDetector._partition_dict_to_communities(partition)
        assert out == {0: {"a", "b"}, 1: {"c", "d", "e"}}

    def test_empty_dict_yields_empty(self) -> None:
        assert CommunityDetector._partition_dict_to_communities({}) == {}


class TestMergeSmallCommunities:
    def test_merges_small_into_largest_neighbor(self) -> None:
        cd = _detector()
        # A 5-clique (big), and a single straggler node attached to it.
        g = nx.Graph()
        for n in ("a", "b", "c", "d", "e"):
            g.add_node(n)
        for u in ("a", "b", "c", "d", "e"):
            for v in ("a", "b", "c", "d", "e"):
                if u < v:
                    g.add_edge(u, v)
        g.add_node("z")
        g.add_edge("z", "a")
        communities = {0: {"a", "b", "c", "d", "e"}, 1: {"z"}}
        merged = cd._merge_small_communities(communities, g, min_size=2)
        # 'z' (size 1 < 2) must be absorbed into the clique community.
        assert len(merged) == 1
        assert merged[0] == {"a", "b", "c", "d", "e", "z"}

    def test_iterates_to_convergence_chained_small_communities(self) -> None:
        # Path a-b-c-d, each its own community. With min_size=2 a single pass
        # can leave a community still too small; the loop repeats until stable.
        cd = _detector()
        g = nx.path_graph(["a", "b", "c", "d"])
        communities = {0: {"a"}, 1: {"b"}, 2: {"c"}, 3: {"d"}}
        merged = cd._merge_small_communities(communities, g, min_size=2)
        # Every surviving community has >= 2 nodes, and all 4 nodes preserved.
        all_nodes = set().union(*merged.values())
        assert all_nodes == {"a", "b", "c", "d"}
        assert all(len(nodes) >= 2 for nodes in merged.values())

    def test_isolated_small_community_cannot_merge(self) -> None:
        # An isolated node has no eligible neighbor -> stays as its own
        # community (loop terminates without merging it).
        cd = _detector()
        g = nx.Graph()
        g.add_edge("a", "b")
        g.add_node("iso")
        communities = {0: {"a", "b"}, 1: {"iso"}}
        merged = cd._merge_small_communities(communities, g, min_size=2)
        assert {"iso"} in merged.values()

    def test_no_merge_when_all_above_min_size(self) -> None:
        cd = _detector()
        g = nx.Graph()
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        g.add_edge("b", "c")
        communities = {0: {"a", "b"}, 1: {"c", "d"}}
        merged = cd._merge_small_communities(dict(communities), g, min_size=2)
        assert merged == communities


class TestBasePartition:
    def test_no_edges_each_node_is_its_own_community(self) -> None:
        cd = _detector()
        g = nx.Graph()
        g.add_nodes_from(["a", "b", "c"])
        partition = cd._get_base_partition(g)
        assert partition is not None
        assert sorted(partition, key=lambda s: sorted(s)) == [{"a"}, {"b"}, {"c"}]

    def test_finds_two_communities_in_two_triangles(self) -> None:
        cd = _detector(auto_resolution=True, min_community_size=1)
        g = _two_triangle_graph()
        partition = cd._get_base_partition(g)
        assert partition is not None
        # The bridged double-triangle should split into (at least) two clusters.
        assert len(partition) >= 2
        # Every original node appears exactly once across the partition.
        assert set().union(*partition) == set(g.nodes())
        assert sum(len(s) for s in partition) == g.number_of_nodes()

    def test_isolated_node_added_back_as_singleton(self) -> None:
        # Leiden only partitions nodes touched by edges; isolated nodes must be
        # re-added as their own community so no node is silently dropped.
        cd = _detector(min_community_size=1)
        g = _two_triangle_graph()
        g.add_node("lonely", name="Lonely")
        partition = cd._get_base_partition(g)
        assert partition is not None
        assert "lonely" in set().union(*partition)


class TestOptimalResolution:
    def test_selects_resolution_from_candidates(self) -> None:
        cd = _detector(auto_resolution=True)
        cd.community_detection_config.auto_resolution_candidates = [0.1, 1.0, 3.0]
        g = _two_triangle_graph()
        chosen = cd._find_optimal_resolution(g)
        assert chosen in {0.1, 1.0, 3.0}

    def test_falls_back_to_configured_resolution_when_no_split(self) -> None:
        # A single clique cannot be split into >=2 communities at any
        # resolution candidate; selection keeps the configured default.
        cd = _detector(auto_resolution=True)
        cd.community_detection_config.resolution = 1.0
        cd.community_detection_config.auto_resolution_candidates = [0.1, 0.2]
        g = nx.complete_graph(["a", "b", "c", "d"])
        chosen = cd._find_optimal_resolution(g)
        assert chosen == 1.0


class TestClusterGraph:
    def test_aggregates_inter_community_edges_with_weights(self) -> None:
        cd = _detector()
        # Two communities: {a,b} and {c,d}; two cross edges with weights.
        g = nx.Graph()
        g.add_edge("a", "b", weight=5.0)  # intra-community, ignored
        g.add_edge("c", "d", weight=5.0)  # intra-community, ignored
        g.add_edge("a", "c", weight=1.5)
        g.add_edge("b", "d", weight=2.5)
        cd.graph = g
        partition = [{"a", "b"}, {"c", "d"}]
        cluster = cd._create_cluster_graph(partition)
        assert cluster.number_of_nodes() == 2
        assert cluster.number_of_edges() == 1
        # The single cluster edge weight is the SUM of inter-community edges.
        assert cluster[0][1]["weight"] == pytest.approx(4.0)

    def test_no_inter_community_edges_yields_no_edges(self) -> None:
        cd = _detector()
        g = nx.Graph()
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        cd.graph = g
        cluster = cd._create_cluster_graph([{"a", "b"}, {"c", "d"}])
        assert cluster.number_of_nodes() == 2
        assert cluster.number_of_edges() == 0


class TestHierarchyProcessing:
    def test_single_level_creates_l0_communities(self) -> None:
        cd = _detector()
        cd.graph = _two_triangle_graph()
        partitions = [[{"n0", "n1", "n2"}, {"n3", "n4", "n5"}]]
        cd._process_hierarchical_partitions(partitions)
        assert set(cd.all_communities) == {"L0_C0", "L0_C1"}
        assert all(c.level == 0 for c in cd.all_communities.values())
        # node_to_community_l0 maps every node to its L0 community.
        assert cd.node_to_community_l0["n0"] == "L0_C0"
        assert cd.node_to_community_l0["n3"] == "L0_C1"

    def test_two_levels_builds_parent_child_links(self) -> None:
        cd = _detector()
        cd.graph = _two_triangle_graph()
        # L0 has two communities (indices 0, 1); L1 groups both under one parent.
        partitions = [
            [{"n0", "n1", "n2"}, {"n3", "n4", "n5"}],
            [{0, 1}],
        ]
        cd._process_hierarchical_partitions(partitions)
        assert "L1_C0" in cd.all_communities
        parent = cd.all_communities["L1_C0"]
        assert parent.level == 1
        assert set(parent.children_ids) == {"L0_C0", "L0_C1"}
        # Parent's node set is the union of children.
        assert parent.nodes == {"n0", "n1", "n2", "n3", "n4", "n5"}
        # Children point back at the parent.
        assert cd.all_communities["L0_C0"].parent_id == "L1_C0"
        assert cd.all_communities["L0_C1"].parent_id == "L1_C0"


class TestEndToEndDetection:
    def test_detect_communities_on_two_triangles(self) -> None:
        cd = _detector(auto_resolution=True, min_community_size=1)
        g = _two_triangle_graph()
        result = cd(g)
        assert result is cd
        assert len(cd.all_communities) >= 2
        # base_modularity is set from the level-0 partition.
        assert cd.base_modularity != 0.0

    def test_empty_graph_is_skipped(self) -> None:
        cd = _detector()
        cd(nx.Graph())
        assert cd.all_communities == {}


class TestMetricsAndExport:
    def test_get_community_metrics_summarizes_l0(self) -> None:
        cd = _detector()
        cd.graph = _two_triangle_graph()
        cd.base_modularity = 0.42
        cd.all_communities = {
            "L0_C0": HierarchicalCommunity(
                community_id="L0_C0", level=0, nodes={"n0", "n1", "n2"}
            ),
            "L0_C1": HierarchicalCommunity(
                community_id="L0_C1", level=0, nodes={"n3", "n4"}
            ),
        }
        metrics = cd.get_community_metrics()
        assert metrics is not None
        assert metrics.num_communities == 2
        assert metrics.modularity == pytest.approx(0.42)
        assert metrics.largest_community_size == 3
        assert metrics.smallest_community_size == 2
        assert metrics.average_community_size == pytest.approx(2.5)
        assert metrics.community_size_distribution == {3: 1, 2: 1}

    def test_get_community_metrics_none_without_communities(self) -> None:
        cd = _detector()
        cd.graph = _two_triangle_graph()
        assert cd.get_community_metrics() is None

    def test_export_requires_detected_communities(self) -> None:
        cd = _detector()
        cd.graph = _two_triangle_graph()
        with pytest.raises(GraphError):
            cd.export_community_data()

    def test_export_community_data_shape(self) -> None:
        cd = _detector()
        cd.graph = _two_triangle_graph()
        cd.all_communities = {
            "L0_C0": HierarchicalCommunity(
                community_id="L0_C0", level=0, nodes={"n0", "n1", "n2"}
            )
        }
        exported = cd.export_community_data()
        assert "hierarchy" in exported
        assert "metrics_level_0" in exported
        entry = exported["hierarchy"][0]
        assert entry["community_id"] == "L0_C0"
        assert entry["size"] == 3
        assert entry["level"] == 0


class TestGenerateCommunityObjects:
    def test_empty_graph_returns_empty(self) -> None:
        cd = _detector()
        cd.graph = nx.Graph()
        assert cd.generate_community_objects() == []

    def test_requires_detected_communities(self) -> None:
        cd = _detector()
        cd.graph = _two_triangle_graph()
        with pytest.raises(GraphError):
            cd.generate_community_objects()

    def test_builds_community_objects_with_attributes(self) -> None:
        cd = _detector()
        g = _two_triangle_graph()
        # Tag edges with ids so relationship_ids is populated.
        for i, (u, v) in enumerate(g.edges()):
            g.edges[u, v]["id"] = f"r{i}"
        cd.graph = g
        cd.all_communities = {
            "L0_C0": HierarchicalCommunity(
                community_id="L0_C0", level=0, nodes={"n0", "n1", "n2"}
            )
        }
        objs = cd.generate_community_objects()
        assert len(objs) == 1
        obj = objs[0]
        assert isinstance(obj, Community)
        assert obj.id == "L0_C0"
        assert obj.size == 3
        assert set(obj.entity_ids) == {"n0", "n1", "n2"}
        # density and entity_names are computed onto attributes.
        assert "density" in obj.attributes
        assert "entity_names" in obj.attributes
        # The three intra-triangle edges all carry ids.
        assert len(obj.relationship_ids) == 3


def _community(entity_ids: list[str]) -> Community:
    return Community(
        id="L0_C0",
        name="Level 0 Community 0",
        level="0",
        parent="",
        children=[],
        entity_ids=entity_ids,
        text_unit_ids=[],
        size=len(entity_ids),
    )


class TestReportInputShaping:
    def test_prepare_report_input_uses_entity_names_for_relationships(self) -> None:
        # Relationships in the report input must reference entity NAMES, not raw
        # node ids, so the LLM sees intelligible text.
        cd = _detector()
        g = nx.Graph()
        g.add_node("e1", name="Alice", type="PERSON", description="A person")
        g.add_node("e2", name="Acme", type="ORG", description="A company")
        g.add_edge("e1", "e2", type="works_at", description="Alice works at Acme")
        cd.graph = g
        report_input = cd._prepare_report_input(_community(["e1", "e2"]))

        assert report_input["community_id"] == "L0_C0"
        # entities carry id + name + type + description
        names = {e["name"] for e in report_input["entities"]}
        assert names == {"Alice", "Acme"}
        # relationship endpoints are NAMES, never the node ids.
        rel = report_input["relationships"][0]
        assert {rel["source"], rel["target"]} == {"Alice", "Acme"}
        assert "e1" not in (rel["source"], rel["target"])
        assert rel["type"] == "works_at"

    def test_prepare_report_input_falls_back_to_id_without_name(self) -> None:
        cd = _detector()
        g = nx.Graph()
        g.add_node("e1")  # no name attribute
        g.add_node("e2", name="Acme")
        g.add_edge("e1", "e2", type="related")
        cd.graph = g
        report_input = cd._prepare_report_input(_community(["e1", "e2"]))
        rel = report_input["relationships"][0]
        # node 'e1' has no name -> id is used as the fallback.
        assert {rel["source"], rel["target"]} == {"e1", "Acme"}

    def test_prepare_report_input_skips_unknown_entities(self) -> None:
        cd = _detector()
        g = nx.Graph()
        g.add_node("e1", name="Alice")
        cd.graph = g
        report_input = cd._prepare_report_input(_community(["e1", "missing"]))
        # 'missing' is not in the graph -> excluded from entities.
        assert [e["id"] for e in report_input["entities"]] == ["e1"]

    def test_create_report_chain_inputs_serializes_to_strings(self) -> None:
        report_inputs = [
            {
                "community_id": "L0_C0",
                "entities": [
                    {"name": "Alice", "type": "PERSON", "description": "A person"}
                ],
                "relationships": [
                    {
                        "source": "Alice",
                        "target": "Acme",
                        "type": "works_at",
                        "description": "rel",
                    }
                ],
                "content_length": 500,
                "include_statistics": True,
                "include_key_entities": False,
            }
        ]
        out = CommunityDetector._create_report_chain_inputs(report_inputs)
        assert len(out) == 1
        chain_in = out[0]
        assert chain_in["community_id"] == "L0_C0"
        assert "Alice (PERSON): A person" in chain_in["entities"]
        assert "Alice -> Acme (works_at): rel" in chain_in["relationships"]
        # bool flags are stringified.
        assert chain_in["include_statistics"] == "True"
        assert chain_in["include_key_entities"] == "False"


class TestReportRank:
    """_compute_report_rank: graph-importance rank (sum of member degrees),
    deterministic and LLM-free. Replaces the old hardcoded rank=1."""

    def test_rank_is_sum_of_member_degrees(self) -> None:
        cd = _detector()
        g = nx.Graph()
        # triangle: each of a,b,c has degree 2 -> sum = 6 for the 3-member community
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("a", "c")
        cd.graph = g
        assert cd._compute_report_rank(_community(["a", "b", "c"])) == 6

    def test_more_connected_community_outranks_sparser_one(self) -> None:
        cd = _detector()
        cd.graph = _star_graph(4)  # hub degree 4, each leaf degree 1
        hub_community = _community(["hub", "leaf0", "leaf1"])  # 4 + 1 + 1 = 6
        leaf_community = _community(["leaf2", "leaf3"])  # 1 + 1 = 2
        assert cd._compute_report_rank(hub_community) > cd._compute_report_rank(
            leaf_community
        )

    def test_entities_absent_from_graph_are_ignored(self) -> None:
        cd = _detector()
        g = nx.Graph()
        g.add_edge("a", "b")  # both degree 1
        cd.graph = g
        # 'ghost' is not in the graph -> contributes 0.
        assert cd._compute_report_rank(_community(["a", "b", "ghost"])) == 2

    def test_empty_graph_falls_back_to_member_count(self) -> None:
        cd = _detector()  # default graph is empty
        assert cd._compute_report_rank(_community(["a", "b", "c"])) == 3


def _star_graph(n_leaves: int) -> nx.Graph:
    """A hub 'hub' connected to n leaves -> hub has the highest degree."""
    g = nx.Graph()
    g.add_node("hub", name="Hub", type="ORG", description="central node")
    for i in range(n_leaves):
        lid = f"leaf{i}"
        g.add_node(lid, name=f"Leaf{i}", type="PERSON", description="x")
        g.add_edge("hub", lid, type="rel", description="connects")
    return g


class TestReportContextFidelity:
    """Degree-sorted selection + token budgeting for report-input assembly."""

    def test_degree_sort_puts_high_degree_entities_first(self) -> None:
        cd = _detector()
        # hub (degree 3) plus three leaves (degree 1 each). Feed entity_ids in
        # an order that does NOT lead with the hub, to prove it is reordered.
        cd.graph = _star_graph(3)
        community = _community(["leaf2", "leaf0", "hub", "leaf1"])
        selected = cd._select_report_entities(community)
        # The hub (highest degree) must come first regardless of input order.
        assert selected[0] == "hub"

    def test_degree_sort_drops_low_degree_under_small_cap(self) -> None:
        cd = _detector()
        cd.community_detection_config.report_generation.max_entities_per_report = 2
        cd.graph = _star_graph(4)
        community = _community(["leaf0", "leaf1", "hub", "leaf2", "leaf3"])
        report_input = cd._prepare_report_input(community)
        ids = [e["id"] for e in report_input["entities"]]
        # Cap of 2 keeps the hub (highest degree) + exactly one leaf; the rest
        # (all degree 1) are dropped.
        assert len(ids) == 2
        assert "hub" in ids

    def test_token_budget_caps_context(self) -> None:
        cd = _detector()
        # Generous entity cap, but a tiny token budget should still truncate.
        cd.community_detection_config.report_generation.max_entities_per_report = 100
        cd.community_detection_config.report_generation.max_report_context_tokens = 4
        cd.graph = _star_graph(10)
        community = _community(["hub", *[f"leaf{i}" for i in range(10)]])
        report_input = cd._prepare_report_input(community)
        # The budget is far below what 11 entities cost, so far fewer survive.
        assert len(report_input["entities"]) < 11
        # The single highest-degree entity is always admitted (never empty).
        assert len(report_input["entities"]) >= 1
        assert report_input["entities"][0]["id"] == "hub"

    def test_relationships_among_selected_entities_included(self) -> None:
        cd = _detector()
        cd.graph = _star_graph(3)
        community = _community(["hub", "leaf0", "leaf1", "leaf2"])
        report_input = cd._prepare_report_input(community)
        # Every relationship endpoint must be one of the selected entities.
        selected_names = {e["name"] for e in report_input["entities"]}
        for rel in report_input["relationships"]:
            assert rel["source"] in selected_names
            assert rel["target"] in selected_names
        # The hub-leaf edges are present (3 of them).
        assert len(report_input["relationships"]) == 3

    def test_relationships_dropped_when_endpoint_truncated(self) -> None:
        # With only the hub selected (cap 1), no relationship can survive since
        # both endpoints must be in the selected set.
        cd = _detector()
        cd.community_detection_config.report_generation.max_entities_per_report = 1
        cd.graph = _star_graph(3)
        community = _community(["hub", "leaf0", "leaf1", "leaf2"])
        report_input = cd._prepare_report_input(community)
        assert [e["id"] for e in report_input["entities"]] == ["hub"]
        assert report_input["relationships"] == []

    def test_config_default_present(self) -> None:
        cfg = Config()
        report_cfg = cfg.graph.community_detection.report_generation
        assert report_cfg.max_report_context_tokens == 4000


class TestExtractTextFromResult:
    def test_passthrough_string(self) -> None:
        assert CommunityDetector._extract_text_from_result("hello") == "hello"

    def test_dict_with_hashtext_key(self) -> None:
        assert CommunityDetector._extract_text_from_result({"#text": "body"}) == "body"

    def test_dict_without_hashtext_stringifies(self) -> None:
        out = CommunityDetector._extract_text_from_result({"a": 1})
        assert "a" in out

    def test_non_string_non_dict_stringifies(self) -> None:
        assert CommunityDetector._extract_text_from_result(42) == "42"


class TestGenerateReportsGuards:
    def test_returns_empty_when_report_generation_disabled(self) -> None:
        cd = _detector(report=False)
        # No report_generator attribute -> short-circuits to [].
        assert cd.generate_reports([_community(["e1"])]) == []

    def test_returns_empty_when_no_communities(self) -> None:
        cd = _detector(report=False)
        assert cd.generate_reports([]) == []


class _StubBatchProcessor:
    """Stands in for BatchProcessor: returns a caller-supplied result list
    without touching Bedrock."""

    def __init__(self, results: list[dict | None]) -> None:
        self._results = results

    def execute_with_fallback(self, **_: object) -> list[dict | None]:
        return self._results


class TestGenerateReportsCounting:
    def _ready_detector(self, results: list[dict | None]) -> CommunityDetector:
        # Build a detector whose report path is fully stubbed (no LLM/Bedrock).
        cd = _detector(report=False)
        cd.community_detection_config.report_generation.enabled = True
        g = nx.Graph()
        g.add_node("e1", name="Alice")
        cd.graph = g

        # report_generator only needs .batch / .invoke attributes for the
        # batch processor signature; the stubbed processor ignores them.
        class _StubChain:
            batch = staticmethod(lambda *a, **k: [])
            invoke = staticmethod(lambda *a, **k: {})

        cd.report_generator = _StubChain()
        cd.batch_processor = _StubBatchProcessor(results)
        return cd

    def test_creates_reports_for_nonempty_results(self) -> None:
        cd = self._ready_detector(
            [
                {
                    "community_name": "Cluster A",
                    "summary": "A summary",
                    "full_content": "Full body",
                }
            ]
        )
        reports = cd.generate_reports([_community(["e1"])])
        assert len(reports) == 1
        assert reports[0].community_id == "L0_C0"
        assert reports[0].summary == "A summary"
        assert reports[0].full_content == "Full body"

    def test_empty_result_is_counted_not_silently_dropped(self) -> None:
        # One real result, one empty (None): only one report, no crash.
        cd = self._ready_detector(
            [{"community_name": "A", "summary": "s", "full_content": "c"}, None]
        )
        reports = cd.generate_reports([_community(["e1"]), _community(["e1"])])
        assert len(reports) == 1
