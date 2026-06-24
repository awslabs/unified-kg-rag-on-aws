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

from aws_graphrag.adapters.retrieval.token_manager import SectionType
from aws_graphrag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from aws_graphrag.domain.models import Config, SearchQuery

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
    object.__setattr__(
        inst, "_min_entity_importance", config.indexing.neptune.min_entity_importance
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


def test_calculate_relevance_clamped_to_one(retriever) -> None:
    query = SearchQuery(query="alice")
    score = retriever._calculate_relevance(
        {"importance": 1.0, "name": "alice", "description": "alice"},
        [],
        query,
        is_community=False,
    )
    assert 0.0 <= score <= 1.0


def test_calculate_relevance_text_match_boosts_entity(retriever) -> None:
    query = SearchQuery(query="alice")
    matched = retriever._calculate_relevance(
        {"importance": 0.5, "name": "alice", "description": ""},
        [],
        query,
        is_community=False,
    )
    unmatched = retriever._calculate_relevance(
        {"importance": 0.5, "name": "bob", "description": ""},
        [],
        query,
        is_community=False,
    )
    assert matched > unmatched


def test_calculate_relevance_community_uses_size(retriever) -> None:
    query = SearchQuery(query="x")
    score = retriever._calculate_relevance(
        {"size": 50, "name": "Cluster"}, [], query, is_community=True
    )
    assert 0.0 <= score <= 1.0


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
    retriever, mocker, configured_hops
) -> None:
    # Regression: hops was max(self._max_hops, DEFAULT_MAX_HOPS=3), so a
    # configured max_hops < 3 was silently raised to 3 (and the config was
    # ignored). The configured value must be used directly.
    object.__setattr__(retriever, "_max_hops", configured_hops)

    captured: dict[str, int] = {}

    class FluentTraversal:
        """Records every step; .times() captures its arg."""

        def times(self, n, *a, **k):
            captured["times"] = n
            return self

        def __getattr__(self, name):
            return lambda *a, **k: self

    g = mocker.MagicMock()
    g.V.return_value = FluentTraversal()

    # Patch via object.__setattr__ (not mocker.patch.object): the retriever is a
    # __new__-constructed pydantic model whose attribute deletion at teardown
    # raises, so set bound replacements directly.
    async def _fake_execute(_traversal):
        return []

    object.__setattr__(retriever, "_apply_filters", lambda t, f: t)
    object.__setattr__(retriever, "_with_projection", lambda t: t)
    object.__setattr__(retriever, "_execute_traversal", _fake_execute)

    await retriever._traverse_from_entities(g, [{"id": "e1"}], SearchQuery(query="x"))

    assert captured.get("times") == configured_hops
