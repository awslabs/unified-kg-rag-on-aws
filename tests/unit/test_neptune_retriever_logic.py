# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for NeptuneRetriever Gremlin/parse logic.

Covers the pure pieces of the graph-expansion retriever: seed ids, filter
application onto a (recording) traversal, the
projection shape, property-map cleaning, traversal-result parsing into
RetrievalResult, relevance scoring, and content building. The retriever is
built via ``__new__`` so its AWS-client ``__init__`` never runs; a recording
fake stands in for the Gremlin traversal where shape matters, and NeptuneClient
is never constructed.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

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
# _create_retrieval_result
# --------------------------------------------------------------------------- #


def test_create_retrieval_result_entity(retriever) -> None:
    node = {"id": "e1", "name": "Alice", "description": "researcher"}
    item = {"node_type": "Entity-default", "path": []}
    result = retriever._create_retrieval_result(item, node)
    assert result.source == "e1"
    assert result.retriever_type == SectionType.ENTITY.value
    assert result.metadata["_node_type"] == "Entity-default"
    assert "Entity: Alice" in result.content


# --------------------------------------------------------------------------- #
# _build_content
# --------------------------------------------------------------------------- #


def test_build_content_entity_with_path(retriever) -> None:
    content = retriever._build_content(
        {"name": "Alice", "description": "researcher"},
        [{"name": ["Alice"]}, {"name": ["Acme"]}],
    )
    assert "Entity: Alice" in content
    assert "Description: researcher" in content
    assert "Path: Alice -> Acme" in content


# --------------------------------------------------------------------------- #
# _calculate_relevance
# --------------------------------------------------------------------------- #


def _relevance(retriever, rank: float, path: list) -> float:
    return retriever._calculate_relevance(
        {"rank": rank}, path, importance_key="rank", max_importance=1.0
    )


def test_calculate_relevance_bands_by_distance(retriever) -> None:
    # Path length L scores in (1 / (L + 1), 1 / L], importance places it inside
    # (the upper half of the band, so the open lower edge never ties).
    seed = _relevance(retriever, 1.0, [{"name": ["a"]}])
    weak_seed = _relevance(retriever, 0.0, [{"name": ["a"]}])
    one_hop = _relevance(retriever, 1.0, [{"name": ["a"]}, {"name": ["b"]}])
    assert seed == pytest.approx(1.0)
    assert weak_seed == pytest.approx(0.75)
    assert one_hop == pytest.approx(0.5)


def test_calculate_relevance_nearer_node_outranks_more_important_far_one(
    retriever,
) -> None:
    near = _relevance(retriever, 0.0, [{}])
    far = _relevance(retriever, 1.0, [{}, {}])
    assert near > far


def test_calculate_relevance_importance_orders_within_a_distance(retriever) -> None:
    low = _relevance(retriever, 0.2, [{}, {}])
    high = _relevance(retriever, 0.8, [{}, {}])
    assert high > low


def test_calculate_relevance_without_importance_key_is_neutral(retriever) -> None:
    """``entity_importance_source: none`` scores every node at mid-band; no
    other vertex property stands in for importance."""
    assert retriever._calculate_relevance({"importance": 1.0}, [{}]) == pytest.approx(
        0.875
    )


# --------------------------------------------------------------------------- #
# Seeds: the caller's id filter
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"id": ["e1", "e2"]}, ["e1", "e2"]),
        ({"id": "e1"}, []),
        ({"type": "ORG"}, []),
        (None, []),
    ],
)
def test_seed_ids_come_from_the_id_filter(retriever, filters, expected) -> None:
    query = SearchQuery(query="x", filters=filters)
    assert retriever._seed_ids(query) == expected


async def test_query_without_seed_ids_returns_nothing_without_connecting(
    retriever,
) -> None:
    class _Unconnectable:
        @property
        def g(self):
            raise AssertionError("connected without seeds")

    object.__setattr__(retriever, "_neptune_client", _Unconnectable())
    query = SearchQuery(query="Vendor", entity_focus=["Vendor"])
    assert await retriever.aretrieve(query) == []


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
    # One hop out: the (1/3, 1/2] band, importance rank / max rank inside it.
    assert by_id["high"] == pytest.approx(1 / 3 + (1 / 6) * (1.0 + 1.0) / 2)
    assert by_id["low"] == pytest.approx(1 / 3 + (1 / 6) * (1.0 + 0.25) / 2)


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
    neutral = 1 / 3 + (1 / 6) * (1.0 + 0.5) / 2
    assert [r.score for r in results] == pytest.approx([neutral, neutral])


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


@pytest.mark.parametrize("source", ["rank", "degree", "none"])
def test_cut_keeps_every_seed_over_higher_ranked_hubs(
    retriever, config, source
) -> None:
    # 10 low-degree seeds and 30 one-hop hubs: averaging importance and
    # proximity let the hubs push 8 of the 10 seeds out of the cut.
    config.indexing.neptune.entity_importance_source = source
    config.indexing.neptune.traversal_fetch_multiplier = 3
    seeds = [
        _entity_item(f"seed{i}", 0, rank=1 + i % 3, degree=1 + i % 3) for i in range(10)
    ]
    hubs = [
        _entity_item(f"hub{i}", 1, rank=4 + i % 7, degree=4 + i % 7) for i in range(30)
    ]
    results = retriever._process_traversal_results(
        hubs + seeds, SearchQuery(query="x", top_k=10)
    )
    sources = [r.source for r in results]
    assert set(sources[:10]) == {f"seed{i}" for i in range(10)}
    # The expansion still gets its own top_k slots, best hubs first.
    assert len(sources) == 20
    assert all(s.startswith("hub") for s in sources[10:])
    if source != "none":
        kept = {int(s[3:]) % 7 for s in sources[10:]}
        assert min(kept) >= 4  # degrees 8-10 outrank the weaker hubs


def test_fetch_multiplier_one_keeps_every_traversed_entity(retriever, config) -> None:
    config.indexing.neptune.traversal_fetch_multiplier = 1
    query = SearchQuery(query="x", top_k=2)
    items = [_entity_item(f"e{i}", 1, rank=1) for i in range(3)]
    assert len(retriever._process_traversal_results(items, query)) == 3


async def _entity_traversal_text(retriever, seed_ids: list[str], query) -> str:
    """The Gremlin text `_traverse` sends, projection left out."""
    captured: list = []

    async def _capture(build):
        captured.append(build(Graph().traversal()))
        return []

    object.__setattr__(retriever, "_with_projection", lambda t, **_: t)
    object.__setattr__(retriever, "_execute_traversal", _capture)
    await retriever._traverse(seed_ids, query)
    return Translator("g").translate(captured[0].bytecode)


async def test_entity_traversal_returns_the_seeds_and_expands_each_one(
    retriever, config
) -> None:
    # A limit() inside repeat() counts every traverser of the traversal, so a
    # single cap let the first seeds use it up; emit() after repeat() never
    # emits the seed itself. The seeds come back via identity() and the
    # expansion runs, with its own budget, per seed inside local(). Neptune
    # applies the limit inside repeat() per traverser and hop, so the seed's
    # total is capped after emit().
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
        ".times(2).emit().dedup().limit(8)))"
    )


async def test_entity_traversal_keeps_every_seed(retriever, config) -> None:
    # A fixed cap of 10 silently dropped the seeds a caller passed beyond it
    # (local search and DRIFT pass up to top_k candidate ids).
    seeds = [f"e{i}" for i in range(15)]
    text = await _entity_traversal_text(retriever, seeds, SearchQuery(query="x"))
    assert "'e14'" in text


@pytest.mark.parametrize(("multiplier", "per_seed"), [(1, 4), (3, 12)])
async def test_entity_traversal_budget_scales_with_fetch_multiplier(
    retriever, config, multiplier, per_seed
) -> None:
    config.indexing.neptune.traversal_fetch_multiplier = multiplier
    query = SearchQuery(query="x", top_k=4, retrieval_multiplier=2)
    text = await _entity_traversal_text(retriever, ["e1", "e2"], query)
    assert f".emit().dedup().limit({per_seed})))" in text


def test_a_node_reached_from_two_seeds_keeps_its_shortest_path(retriever) -> None:
    items = [
        _entity_item("seed", 0, rank=1),
        _entity_item("shared", 2, rank=1),
        _entity_item("shared", 1, rank=1),
    ]
    results = retriever._process_traversal_results(items, SearchQuery(query="x"))
    by_id = {r.source: r.score for r in results}
    # One hop (the shorter path) tops the (1/3, 1/2] band; two hops would cap
    # at 1/3.
    assert by_id == {"seed": pytest.approx(1.0), "shared": pytest.approx(0.5)}


# --------------------------------------------------------------------------- #
# _execute_traversal: concurrency bound by the connection pool
# --------------------------------------------------------------------------- #


class _SlowTraversal:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def to_list(self) -> list:
        cls = type(self)
        with cls.lock:
            cls.active += 1
            cls.max_active = max(cls.max_active, cls.active)
        time.sleep(0.02)
        with cls.lock:
            cls.active -= 1
        return [1]


async def test_traversals_beyond_the_pool_wait_on_the_loop(config: Config) -> None:
    # gremlinpython blocks a worker thread in pool.get() for every request
    # beyond pool_size; those threads come from the loop's default executor,
    # which LangChain also runs Bedrock calls on. Excess traversals must wait
    # on the loop instead of holding an executor thread.
    config.aws.neptune.pool_size = 3
    config.indexing.neptune.index_concurrency = 1
    retriever = NeptuneRetriever(config=config, neptune_client=MagicMock())
    results = await asyncio.gather(
        *(
            retriever._execute_traversal(lambda g: _SlowTraversal())  # type: ignore[arg-type,return-value]
            for _ in range(12)
        )
    )
    assert results == [[1]] * 12
    assert _SlowTraversal.max_active == 3
