# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for NeptuneRetriever Gremlin/parse logic.

Covers the pure pieces of the graph-expansion retriever: label-prefix
normalization, filter application onto a (recording) traversal, the
projection shape, property-map cleaning, traversal-result parsing into
RetrievalResult, relevance scoring, and content building. The retriever is
built via ``__new__`` so its AWS-client ``__init__`` never runs; a recording
fake stands in for the Gremlin traversal where shape matters, and NeptuneClient
is never constructed.
"""

from __future__ import annotations

import pytest
from gremlin_python.process.translator import Translator
from gremlin_python.structure.graph import Graph

from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from unified_kg_rag.domain.models import Config, SearchQuery

pytestmark = pytest.mark.unit


class RecordingTraversal:
    """Chainable fake recording (step_name, args) for each step received."""

    def __init__(self, calls: list[tuple]) -> None:
        self._calls = calls

    def __getattr__(self, name: str):
        def _step(*args, **kwargs):
            self._calls.append((name, args))
            return self

        return _step


@pytest.fixture
def retriever(config: Config) -> NeptuneRetriever:
    inst = NeptuneRetriever.__new__(NeptuneRetriever)
    object.__setattr__(inst, "_config", config)
    object.__setattr__(inst, "_neptune_config", config.indexing.neptune)
    object.__setattr__(inst, "_max_hops", config.indexing.neptune.max_hops)
    object.__setattr__(
        inst, "_max_results_per_hop", config.indexing.neptune.max_results_per_hop
    )
    return inst


# --------------------------------------------------------------------------- #
# _normalize_label_prefixes
# --------------------------------------------------------------------------- #


def test_normalize_label_prefixes_string(retriever) -> None:
    assert retriever._normalize_label_prefixes("entity") == ["entity"]


def test_normalize_label_prefixes_none_returns_entity_and_community(
    retriever, config
) -> None:
    out = retriever._normalize_label_prefixes(None)
    assert out == [
        config.indexing.neptune.entity_label_prefix,
        config.indexing.neptune.community_label_prefix,
    ]


def test_normalize_label_prefixes_list_passthrough(retriever) -> None:
    assert retriever._normalize_label_prefixes(["entity"]) == ["entity"]


# --------------------------------------------------------------------------- #
# _apply_filters
# --------------------------------------------------------------------------- #


def test_apply_filters_none_is_noop(retriever) -> None:
    calls: list[tuple] = []
    t = RecordingTraversal(calls)
    out = retriever._apply_filters(t, None)
    assert out is t
    assert calls == []


def test_apply_filters_skips_id_key(retriever) -> None:
    calls: list[tuple] = []
    retriever._apply_filters(RecordingTraversal(calls), {"id": ["a", "b"]})
    # The reserved "id" key is handled as seeds elsewhere, not as a has() filter.
    assert calls == []


def test_apply_filters_list_uses_within(retriever) -> None:
    calls: list[tuple] = []
    retriever._apply_filters(RecordingTraversal(calls), {"type": ["PERSON", "ORG"]})
    has_calls = [c for c in calls if c[0] == "has"]
    assert len(has_calls) == 1
    key, predicate = has_calls[0][1]
    assert key == "type"
    # P.within(...) predicate object carries the list as its value.
    assert list(predicate.value) == ["PERSON", "ORG"]


def test_apply_filters_scalar_uses_equality(retriever) -> None:
    calls: list[tuple] = []
    retriever._apply_filters(RecordingTraversal(calls), {"type": "PERSON"})
    has_calls = [c for c in calls if c[0] == "has"]
    assert has_calls[0][1] == ("type", "PERSON")


def test_apply_filters_dict_range_operators(retriever) -> None:
    calls: list[tuple] = []
    retriever._apply_filters(RecordingTraversal(calls), {"rank": {"gte": 5}})
    has_calls = [c for c in calls if c[0] == "has"]
    assert len(has_calls) == 1
    key, predicate = has_calls[0][1]
    assert key == "rank"
    assert predicate.value == 5


def test_apply_filters_dict_ignores_unknown_operator(retriever) -> None:
    calls: list[tuple] = []
    retriever._apply_filters(RecordingTraversal(calls), {"rank": {"bogus": 5}})
    assert [c for c in calls if c[0] == "has"] == []


# --------------------------------------------------------------------------- #
# _with_projection
# --------------------------------------------------------------------------- #


def test_with_projection_shape(retriever) -> None:
    calls: list[tuple] = []
    retriever._with_projection(RecordingTraversal(calls))
    project_calls = [c for c in calls if c[0] == "project"]
    assert project_calls
    assert project_calls[0][1] == ("node", "path", "node_type")
    # Three .by() modulators (node valueMap, path, label).
    assert sum(1 for c in calls if c[0] == "by") == 3


# --------------------------------------------------------------------------- #
# _clean_property_map
# --------------------------------------------------------------------------- #


def test_clean_property_map_unwraps_singletons(retriever) -> None:
    cleaned = retriever._clean_property_map(
        {"id": ["e1"], "name": ["Alice"], "tags": ["a", "b"], "scalar": 7}
    )
    assert cleaned["id"] == "e1"
    assert cleaned["name"] == "Alice"
    # Multi-element lists are left intact.
    assert cleaned["tags"] == ["a", "b"]
    assert cleaned["scalar"] == 7


# --------------------------------------------------------------------------- #
# _process_traversal_results
# --------------------------------------------------------------------------- #


def test_process_traversal_results_dedups_by_id(retriever) -> None:
    query = SearchQuery(query="alice")
    items = [
        {
            "node": {"id": ["e1"], "name": ["Alice"], "importance": [0.9]},
            "path": [],
            "node_type": "Entity-default",
        },
        {  # duplicate id -> dropped
            "node": {"id": ["e1"], "name": ["Alice again"]},
            "path": [],
            "node_type": "Entity-default",
        },
        {
            "node": {"id": ["e2"], "name": ["Bob"], "importance": [0.5]},
            "path": [],
            "node_type": "Entity-default",
        },
    ]
    results = retriever._process_traversal_results(items, query)
    assert [r.source for r in results] == ["e1", "e2"]
    # Sorted by score descending: e1 (importance 0.9) before e2 (0.5).
    assert results[0].score >= results[1].score


def test_process_traversal_results_skips_missing_id(retriever) -> None:
    query = SearchQuery(query="x")
    items = [{"node": {"name": ["NoId"]}, "path": [], "node_type": "Entity-default"}]
    assert retriever._process_traversal_results(items, query) == []


# --------------------------------------------------------------------------- #
# _create_retrieval_result (entity vs community typing)
# --------------------------------------------------------------------------- #


def test_create_retrieval_result_entity(retriever) -> None:
    query = SearchQuery(query="alice")
    node = {"id": "e1", "name": "Alice", "description": "researcher"}
    item = {"node_type": "Entity-default", "path": []}
    result = retriever._create_retrieval_result(item, node, query)
    assert result.source == "e1"
    assert result.retriever_type == SectionType.ENTITY.value
    assert result.metadata["_node_type"] == "Entity-default"
    assert "Entity: Alice" in result.content


def test_create_retrieval_result_community(retriever) -> None:
    query = SearchQuery(query="cluster")
    node = {"id": "c1", "name": "Cluster", "size": 10}
    item = {"node_type": "Community-default", "path": []}
    result = retriever._create_retrieval_result(item, node, query)
    assert result.retriever_type == SectionType.COMMUNITY.value
    assert "Community: Cluster" in result.content


# --------------------------------------------------------------------------- #
# _build_content
# --------------------------------------------------------------------------- #


def test_build_content_entity_with_path(retriever) -> None:
    content = retriever._build_content(
        {"name": "Alice", "description": "researcher"},
        [{"name": ["Alice"]}, {"name": ["Acme"]}],
        is_community=False,
    )
    assert "Entity: Alice" in content
    assert "Description: researcher" in content
    assert "Path: Alice -> Acme" in content


def test_build_content_community(retriever) -> None:
    content = retriever._build_content(
        {"name": "Cluster", "size": 12}, [], is_community=True
    )
    assert "Community: Cluster" in content
    assert "Size: 12" in content


# --------------------------------------------------------------------------- #
# _calculate_relevance
# --------------------------------------------------------------------------- #


def test_calculate_relevance_is_mean_of_importance_and_proximity(
    retriever,
) -> None:
    seed = retriever._calculate_relevance(
        {"importance": 1.0}, [{"name": ["a"]}], is_community=False
    )
    one_hop = retriever._calculate_relevance(
        {"importance": 1.0}, [{"name": ["a"]}, {"name": ["b"]}], is_community=False
    )
    assert seed == pytest.approx(1.0)
    assert one_hop == pytest.approx(0.75)


def test_calculate_relevance_nearer_node_outranks_more_important_far_one(
    retriever,
) -> None:
    near = retriever._calculate_relevance({"importance": 0.2}, [{}], False)
    far = retriever._calculate_relevance({"importance": 0.6}, [{}, {}], False)
    assert near > far


def test_calculate_relevance_community_size_relative_to_largest(retriever) -> None:
    largest = retriever._calculate_relevance(
        {"size": 50}, [{}], is_community=True, max_community_size=50
    )
    half = retriever._calculate_relevance(
        {"size": 25}, [{}], is_community=True, max_community_size=50
    )
    assert largest == pytest.approx(1.0)
    assert half == pytest.approx(0.75)
    assert 0.0 <= retriever._calculate_relevance({"size": 5}, [], True) <= 1.0


# --------------------------------------------------------------------------- #
# _get_seed_nodes (id-filter short-circuit path; pure, no traversal)
# --------------------------------------------------------------------------- #


async def test_get_seed_nodes_id_filter_entity(retriever, config, mocker) -> None:
    g = mocker.MagicMock()
    # Community-search detection compares the requested label_prefixes against
    # the config's (capitalized) prefixes, so pass the actual config value.
    entity_prefix = config.indexing.neptune.entity_label_prefix
    query = SearchQuery(
        query="x", label_prefixes=[entity_prefix], filters={"id": ["e1"]}
    )
    entities, communities = await retriever._get_seed_nodes(g, query)
    assert entities == [{"id": "e1"}]
    assert communities == []


async def test_get_seed_nodes_id_filter_community(retriever, config, mocker) -> None:
    g = mocker.MagicMock()
    community_prefix = config.indexing.neptune.community_label_prefix
    query = SearchQuery(
        query="x", label_prefixes=[community_prefix], filters={"id": ["c1", "c2"]}
    )
    entities, communities = await retriever._get_seed_nodes(g, query)
    assert entities == []
    assert communities == [{"id": "c1"}, {"id": "c2"}]


async def test_get_seed_nodes_id_filter_defaults_to_entities(retriever, mocker) -> None:
    # With no community/entity prefix match (and an id filter), the path returns
    # the seeds as ENTITY seeds (the implementation's default fall-through).
    g = mocker.MagicMock()
    query = SearchQuery(query="x", label_prefixes=["entity"], filters={"id": ["e1"]})
    entities, communities = await retriever._get_seed_nodes(g, query)
    assert entities == [{"id": "e1"}]
    assert communities == []


async def test_get_seed_nodes_no_label_prefixes_returns_empty(
    retriever, mocker
) -> None:
    g = mocker.MagicMock()
    query = SearchQuery(query="x", label_prefixes=["nonsense"])
    entities, communities = await retriever._get_seed_nodes(g, query)
    assert entities == []
    assert communities == []


@pytest.mark.parametrize("configured_hops", [1, 2, 5])
async def test_entity_traversal_honors_configured_max_hops(
    retriever, configured_hops
) -> None:
    # Regression: hops was max(self._max_hops, DEFAULT_MAX_HOPS=3), so a
    # configured max_hops < 3 was silently raised to 3 (and the config was
    # ignored). The configured value must be used directly.
    object.__setattr__(retriever, "_max_hops", configured_hops)
    text = await _entity_traversal_text(retriever, ["e1"], SearchQuery(query="x"))
    assert f".times({configured_hops}).emit()" in text


# --------------------------------------------------------------------------- #
# _find_seeds_by_type (entity_focus vs text-search fallback)
# --------------------------------------------------------------------------- #


async def test_find_seeds_by_type_uses_entity_focus_when_present(
    retriever, mocker
) -> None:
    # With entity_focus, seeds are found by exact name membership (P.within),
    # not the text-search fallback.
    calls: list[tuple] = []
    g = mocker.MagicMock()
    g.V.return_value = RecordingTraversal(calls)
    captured: dict = {}

    async def _fake_execute(traversal):
        captured["executed"] = True
        return []

    object.__setattr__(retriever, "_execute_traversal", _fake_execute)
    query = SearchQuery(query="ignored text", entity_focus=["Alice", "Bob"])

    out = await retriever._find_seeds_by_type(g, query, is_community=False)

    assert out == []
    step_names = [c[0] for c in calls]
    # A `has("name", P.within(...))` step is emitted for the focus names.
    has_name = [c for c in calls if c[0] == "has" and c[1] and c[1][0] == "name"]
    assert has_name, f"expected has('name', ...) in {step_names}"
    assert captured.get("executed") is True


async def test_find_seeds_by_type_text_search_fallback_without_focus(
    retriever, mocker
) -> None:
    # No entity_focus -> falls back to a text-search filter over query terms via
    # a .where(...) step (not a direct name-equality lookup).
    calls: list[tuple] = []
    g = mocker.MagicMock()
    g.V.return_value = RecordingTraversal(calls)

    async def _fake_execute(traversal):
        return []

    object.__setattr__(retriever, "_execute_traversal", _fake_execute)
    query = SearchQuery(query="acme supply agreement")

    await retriever._find_seeds_by_type(g, query, is_community=False)

    step_names = [c[0] for c in calls]
    assert "where" in step_names, f"expected a text-search where() in {step_names}"


async def test_find_seeds_by_type_empty_query_terms_returns_empty(
    retriever, mocker
) -> None:
    # No focus and a query with no word characters -> no seeds (and no traversal
    # execution), rather than an unbounded label scan.
    g = mocker.MagicMock()
    g.V.return_value = RecordingTraversal([])
    executed = {"called": False}

    async def _fake_execute(traversal):
        executed["called"] = True
        return []

    object.__setattr__(retriever, "_execute_traversal", _fake_execute)
    query = SearchQuery(query="!!! ??? ...")

    out = await retriever._find_seeds_by_type(g, query, is_community=False)

    assert out == []
    assert executed["called"] is False


# --------------------------------------------------------------------------- #
# Entity importance source + over-fetch-then-rank
# --------------------------------------------------------------------------- #


def _entity_item(node_id: str, hops: int, **props) -> dict:
    node = {"id": [node_id], "name": [node_id.upper()]}
    node.update({k: [v] for k, v in props.items() if k != "degree"})
    item = {
        "node": node,
        "path": [{"name": [f"p{i}"]} for i in range(hops + 1)],
        "node_type": "Entity-default",
    }
    if "degree" in props:
        item["degree"] = props["degree"]
    return item


def test_projection_reads_the_rank_the_indexer_writes(retriever, mocker) -> None:
    calls: list[tuple] = []
    anonymous: list[tuple] = []
    mocker.patch(
        "unified_kg_rag.adapters.retrievers.neptune_retriever.__",
        RecordingTraversal(anonymous),
    )
    retriever._with_projection(RecordingTraversal(calls))
    value_map_args = [c[1] for c in anonymous if c[0] == "value_map"]
    assert "rank" in value_map_args[0]
    assert "importance" not in value_map_args[0]


def test_projection_with_degree_counts_edges(retriever) -> None:
    calls: list[tuple] = []
    retriever._with_projection(RecordingTraversal(calls), with_degree=True)
    assert [c for c in calls if c[0] == "project"][0][1] == (
        "node",
        "path",
        "node_type",
        "degree",
    )
    assert sum(1 for c in calls if c[0] == "by") == 4


def test_rank_is_normalized_within_the_result(retriever) -> None:
    items = [_entity_item("low", 1, rank=1), _entity_item("high", 1, rank=4)]
    results = retriever._process_traversal_results(items, SearchQuery(query="x"))
    by_id = {r.source: r.score for r in results}
    assert [r.source for r in results] == ["high", "low"]
    assert by_id["high"] == pytest.approx((1.0 + 0.5) / 2)
    assert by_id["low"] == pytest.approx((0.25 + 0.5) / 2)


def test_degree_source_uses_the_projected_edge_count(retriever, config) -> None:
    config.indexing.neptune.entity_importance_source = "degree"
    items = [_entity_item("leaf", 1, degree=2), _entity_item("hub", 1, degree=10)]
    results = retriever._process_traversal_results(items, SearchQuery(query="x"))
    assert [r.source for r in results] == ["hub", "leaf"]
    assert results[0].metadata["degree"] == 10


def test_none_source_keeps_the_neutral_importance(retriever, config) -> None:
    config.indexing.neptune.entity_importance_source = "none"
    items = [_entity_item("a", 1, rank=1), _entity_item("b", 1, rank=9)]
    results = retriever._process_traversal_results(items, SearchQuery(query="x"))
    assert [r.score for r in results] == pytest.approx([0.5, 0.5])


def test_overfetched_entities_are_cut_after_ranking(retriever, config) -> None:
    config.indexing.neptune.traversal_fetch_multiplier = 3
    query = SearchQuery(query="x", top_k=2)
    # Emit order puts the far nodes first; ranking must keep the near ones.
    items = [_entity_item(f"far{i}", 3, rank=1) for i in range(4)] + [
        _entity_item("near0", 1, rank=1),
        _entity_item("near1", 1, rank=1),
    ]
    results = retriever._process_traversal_results(items, query)
    assert [r.source for r in results] == ["near0", "near1"]


def test_fetch_multiplier_one_keeps_every_traversed_entity(retriever, config) -> None:
    config.indexing.neptune.traversal_fetch_multiplier = 1
    query = SearchQuery(query="x", top_k=2)
    items = [_entity_item(f"e{i}", 1, rank=1) for i in range(3)]
    assert len(retriever._process_traversal_results(items, query)) == 3


async def _entity_traversal_text(retriever, seed_ids: list[str], query) -> str:
    """The Gremlin text `_traverse_from_entities` sends, projection left out."""
    captured: list = []

    async def _capture(traversal):
        captured.append(traversal)
        return []

    object.__setattr__(retriever, "_with_projection", lambda t, **_: t)
    object.__setattr__(retriever, "_execute_traversal", _capture)
    await retriever._traverse_from_entities(
        Graph().traversal(), [{"id": s} for s in seed_ids], query
    )
    return Translator("g").translate(captured[0].bytecode)


async def test_entity_traversal_returns_the_seeds_and_expands_each_one(
    retriever, config
) -> None:
    # A limit() inside repeat() counts every traverser of the traversal, so a
    # single cap let the first seeds use it up; emit() after repeat() never
    # emits the seed itself. The seeds come back via identity() and the
    # expansion runs, with its own budget, per seed inside local().
    config.indexing.neptune.traversal_fetch_multiplier = 3
    object.__setattr__(retriever, "_max_hops", 2)
    object.__setattr__(retriever, "_max_results_per_hop", 7)
    query = SearchQuery(query="x", top_k=10, retrieval_multiplier=1)
    text = await _entity_traversal_text(retriever, ["e1", "e2", "e3", "e4"], query)
    # Fetch width 10 * 1 * 3 = 30 split over 4 seeds -> 8 new entities per seed.
    assert text == (
        "g.V().hasLabel('Entity-default')"
        ".has('id',within(['e1','e2','e3','e4']))"
        ".union(__.identity(),__.local(__.repeat(__.local(__.both()"
        ".hasLabel('Entity-default').limit(7))"
        ".has('id',without(['e1','e2','e3','e4'])).dedup().limit(8))"
        ".times(2).emit()))"
    )


@pytest.mark.parametrize(("multiplier", "per_seed"), [(1, 4), (3, 12)])
async def test_entity_traversal_budget_scales_with_fetch_multiplier(
    retriever, config, multiplier, per_seed
) -> None:
    config.indexing.neptune.traversal_fetch_multiplier = multiplier
    query = SearchQuery(query="x", top_k=4, retrieval_multiplier=2)
    text = await _entity_traversal_text(retriever, ["e1", "e2"], query)
    assert f".dedup().limit({per_seed}))" in text


def test_a_node_reached_from_two_seeds_keeps_its_shortest_path(retriever) -> None:
    items = [
        _entity_item("seed", 0, rank=1),
        _entity_item("shared", 2, rank=1),
        _entity_item("shared", 1, rank=1),
    ]
    results = retriever._process_traversal_results(items, SearchQuery(query="x"))
    by_id = {r.source: r.score for r in results}
    assert by_id == {"seed": pytest.approx(1.0), "shared": pytest.approx(0.75)}


# --------------------------------------------------------------------------- #
# _find_seeds_by_type
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("is_community", "prop"), [(False, "rank"), (True, "size")])
async def test_text_seeds_rank_by_a_stored_property(
    retriever, is_community: bool, prop: str
) -> None:
    # Seeds are ordered by a property the indexer writes ("rank" on entities,
    # "size" on communities); no threshold on an unwritten property drops them.
    calls: list[tuple] = []

    class _G:
        def V(self):
            return RecordingTraversal(calls)

    async def _execute(traversal):
        return [{"id": ["v1"], "name": ["vendor"]}]

    object.__setattr__(retriever, "_execute_traversal", _execute)
    query = SearchQuery(query="vendor terms", entity_focus=["vendor"])
    seeds = await retriever._find_seeds_by_type(_G(), query, is_community=is_community)
    assert seeds == [{"id": "v1", "name": "vendor"}]
    by_steps = [args for name, args in calls if name == "by"]
    assert by_steps and by_steps[0][0] == prop
    assert not any(name == "has" and args[0] == "importance" for name, args in calls)
