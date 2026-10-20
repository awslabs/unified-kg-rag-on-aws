# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph expansion under a document filter walks the relationships it passes.

The graph walk crossed every edge, and its per-vertex cap and cut ran before
any check, so entities only excluded documents connect took the slots (a
1,522-degree hub kept 0-4 entities where an index of the passing documents
alone keeps 10). Under a document filter the walk runs over the
relationships index instead, one hop at a time. Also: the filter check's
per-query memo and rank order.
"""

from __future__ import annotations

from typing import Any

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401
from unified_kg_rag.adapters.retrieval.base import query_scope
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.domain.retrieval.strategy_registry import get_strategy_spec

pytestmark = pytest.mark.unit

# S - A (excluded), S - B, A - D, B - D, A - C. Passing: S-B, B-D, A-C.
_EDGES = [
    ("r-sa", "S", "A", 5.0),
    ("r-sb", "S", "B", 4.0),
    ("r-ad", "A", "D", 3.0),
    ("r-bd", "B", "D", 2.0),
    ("r-ac", "A", "C", 1.0),
]
_PASSING = {"r-sb", "r-bd", "r-ac"}
_PEOPLE = {"S", "B", "D"}


class _Graph:
    """The graph walk: every vertex, unchecked (what the fallback returns)."""

    def __init__(self) -> None:
        self.walks = 0

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.walks += 1
        return [_entity(v) for v in ("S", "A", "B", "D", "C")]


def _entity(vid: str, rank: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        content=f"Entity: {vid}",
        score=1.0,
        source=vid,
        retriever_type="entity",
        metadata={"id": vid, "name": vid, "rank": rank, "text_unit_ids": [f"u-{vid}"]},
    )


class _Index:
    """The relationships, entities and text-units indexes."""

    def __init__(
        self,
        config: Config,
        edges: list[tuple[str, str, str, float]] = _EDGES,
        passing: set[str] = _PASSING,
        has_relationships: bool = True,
        units_passing: set[str] | None = None,
    ) -> None:
        self.o = config.indexing.opensearch
        self.edges, self.passing = edges, passing
        self.has_relationships = has_relationships
        self.units_passing = units_passing
        self.calls: list[tuple[str, dict]] = []
        self.edge_fields: list[list[str] | None] = []

    async def ahas_index(self, prefix: str, suffix: str | None) -> bool:
        return prefix != self.o.relationships_index_prefix or self.has_relationships

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        prefix = (query.index_prefixes or [""])[0]
        filters = dict(query.filters or {})
        self.calls.append((prefix, filters))
        if prefix == self.o.relationships_index_prefix:
            self.edge_fields.append(query.source_fields)
            if not self.has_relationships:
                return []
            side = "source_id" if "source_id" in filters else "target_id"
            ends = set(filters[side])
            out = []
            for eid, a, b, rank in self.edges:
                if eid in self.passing and (a if side == "source_id" else b) in ends:
                    out.append(
                        RetrievalResult(
                            content=eid,
                            score=0.0,
                            source=eid,
                            retriever_type="relationship",
                            metadata={
                                "id": eid,
                                "source_id": a,
                                "target_id": b,
                                "source_name": a,
                                "target_name": b,
                                "rank": rank,
                                "text_unit_ids": [f"u-{eid}"],
                            },
                        )
                    )
            return out
        if prefix == self.o.entities_index_prefix:
            ids = filters.get("id") or []
            return [
                _entity(v)
                for v in ids
                if filters.get("type") != "person" or v in _PEOPLE
            ]
        if prefix == self.o.text_units_index_prefix:
            ids = filters.get("id") or []
            keep = self.units_passing
            return [
                RetrievalResult(content="", score=0.0, source=i, retriever_type="text")
                for i in ids
                if keep is None or i in keep
            ]
        return []


def _strategy(config: Config, graph: Any, document: Any) -> Any:
    return get_strategy_spec(SearchStrategy.LOCAL).strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: document,
            RetrieverRole.GRAPH.value: graph,
        },
    )


async def _expand(config: Config, index: Any, filters: dict, graph: Any = None) -> Any:
    strategy = _strategy(config, graph or _Graph(), index)
    return await strategy._expand_via_graph(
        SearchQuery(query="q", filters=filters), ["S"]
    )


def _paths(kept: list[RetrievalResult]) -> dict[str, str]:
    return {
        str(r.source): r.content.split("Path: ")[1] if "Path: " in r.content else ""
        for r in kept
    }


async def test_only_entities_a_walk_over_passing_edges_reaches_are_kept(
    config: Config,
) -> None:
    graph, index = _Graph(), _Index(config)
    kept = await _expand(config, index, {"attr_grp": "a"}, graph)
    # A only over an excluded edge, C only through A: never reached. D is
    # reached through B; the path names no excluded vertex.
    assert _paths(kept) == {"S": "S", "B": "S -> B", "D": "S -> B -> D"}
    assert graph.walks == 0
    assert all(r.content.count("Path: ") == 1 for r in kept)
    # An edge's description and the like are not read: the walk needs its
    # endpoints, names, order and text units.
    assert index.edge_fields and all(
        f and "description" not in f and "text_unit_ids" in f for f in index.edge_fields
    )
    # Nearer first: the score bands follow the walk's distance.
    by_id = {r.source: r for r in kept}
    assert by_id["S"].score > by_id["B"].score > by_id["D"].score


async def test_an_edge_stored_towards_the_seed_is_walked(config: Config) -> None:
    edges = [("r-sb", "B", "S", 4.0), ("r-bd", "D", "B", 2.0)]
    kept = await _expand(config, _Index(config, edges=edges), {"attr_grp": "a"})
    assert _paths(kept) == {"S": "S", "B": "S -> B", "D": "S -> B -> D"}


async def test_hops_and_per_vertex_neighbours_bound_the_walk(config: Config) -> None:
    config.indexing.neptune.max_hops = 1
    kept = await _expand(config, _Index(config), {"attr_grp": "a"})
    assert sorted(_paths(kept)) == ["B", "S"]

    config.indexing.neptune.max_hops = 3
    config.indexing.neptune.max_results_per_hop = 1
    edges = [("r1", "S", "X", 9.0), ("r2", "S", "Y", 1.0)]
    kept = await _expand(
        config, _Index(config, edges=edges, passing={"r1", "r2"}), {"attr_grp": "a"}
    )
    # One new neighbour of S per hop: the best-ranked edge's.
    assert sorted(_paths(kept)) == ["S", "X"]


async def test_a_vertex_filter_keeps_its_entities_the_walk_goes_through_others(
    config: Config,
) -> None:
    # {grp: a, type: person}: A is no person and excluded anyway; with B an
    # organization the walk still reaches D through it.
    global _PEOPLE
    people, _PEOPLE = _PEOPLE, {"S", "D"}
    try:
        kept = await _expand(
            config, _Index(config), {"attr_grp": "a", "type": "person"}
        )
    finally:
        _PEOPLE = people
    assert _paths(kept) == {"S": "S", "D": "S -> B -> D"}


async def test_two_keys_need_one_document_passing_an_edge(config: Config) -> None:
    # The relationships index unions an edge's documents' values per key:
    # under two keys an edge needs a text unit (one document) passing both.
    index = _Index(config, units_passing={"u-r-sb", "u-S", "u-B", "u-D"})
    kept = await _expand(config, index, {"attr_grp": "a", "attr_category": "x"})
    assert sorted(_paths(kept)) == ["B", "S"]


async def test_without_a_document_filter_the_graph_is_walked(config: Config) -> None:
    graph, index = _Graph(), _Index(config)
    kept = await _expand(config, index, {"type": "org"}, graph)
    assert len(kept) == 5 and graph.walks == 1
    assert index.calls == []


@pytest.mark.parametrize("why", ["flag-off", "index-missing"])
async def test_without_a_relationships_index_the_graph_walk_is_checked(
    config: Config, why: str
) -> None:
    if why == "flag-off":
        config.indexing.opensearch.build_relationship_vector_index = False
        index = _Index(config)
    else:
        index = _Index(config, has_relationships=False)
    graph = _Graph()
    kept = await _expand(config, index, {"attr_grp": "a"}, graph)
    assert graph.walks == 1
    assert len(kept) == 5  # the entities index passes every walked entity


async def test_a_failed_relationships_read_keeps_the_seeds_only(
    config: Config,
) -> None:
    class _Failing(_Index):
        async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
            if query.index_prefixes == [self.o.relationships_index_prefix]:
                raise RuntimeError("store hiccup")
            return await super().aretrieve(query)

    kept = await _expand(config, _Failing(config), {"attr_grp": "a"})
    assert [r.source for r in kept] == ["S"]


async def test_the_walk_is_cut_to_the_expansion_limit(config: Config) -> None:
    edges = [(f"r{i}", "S", f"v{i:02d}", float(30 - i)) for i in range(30)]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    strategy = _strategy(config, _Graph(), index)
    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", top_k=4, filters={"attr_grp": "a"}), ["S"]
    )
    assert len(kept) == 5  # the seed and top_k others


# --------------------------------------------------------------------------- #
# The filter check
# --------------------------------------------------------------------------- #


class _Document:
    """Every id passes but those of relationships not in ``_PASSING``."""

    def __init__(self, config: Config) -> None:
        self.o = config.indexing.opensearch
        self.calls: list[tuple[str, list[str]]] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        ids = list((query.filters or {}).get("id") or [])
        prefix = (query.index_prefixes or [""])[0]
        self.calls.append((prefix, ids))
        if prefix == self.o.relationships_index_prefix:
            ids = [i for i in ids if i in _PASSING]
        return [
            RetrievalResult(content="", score=0.0, source=i, retriever_type="entity")
            for i in ids
        ]


async def test_a_query_asks_each_ids_outcome_once(config: Config) -> None:
    # DRIFT's follow-ups check the same chunks and edges again.
    document = _Document(config)
    strategy = _strategy(config, _Graph(), document)
    query = SearchQuery(query="q", filters={"attr_grp": "a"})
    prefix = config.indexing.opensearch.relationships_index_prefix
    with query_scope():
        first = await strategy._ids_passing(
            query, prefix, ["r-sa", "r-sb"], {"attr_grp": "a"}
        )
        second = await strategy._ids_passing(
            query, prefix, ["r-sb", "r-sa", "r-bd"], {"attr_grp": "a"}
        )
    assert first == {"r-sb"} and second == {"r-sb", "r-bd"}
    assert document.calls == [(prefix, ["r-sa", "r-sb"]), (prefix, ["r-bd"])]


async def test_the_check_stops_once_enough_passed(config: Config) -> None:
    config.indexing.opensearch.index_settings = {"index.max_result_window": 2}
    document = _Document(config)
    strategy = _strategy(config, _Graph(), document)
    prefix = config.indexing.opensearch.entities_index_prefix
    passed = await strategy._ids_passing(
        SearchQuery(query="q"),
        prefix,
        [f"e{i}" for i in range(7)],
        {"attr_grp": "a"},
        enough=3,
    )
    # Two batches of two: four ids, three passed in rank order, no third batch.
    assert passed == {"e0", "e1", "e2"}
    assert [ids for _, ids in document.calls] == [["e0", "e1"], ["e2", "e3"]]


async def test_a_memoized_outcome_does_not_displace_a_higher_ranked_id(
    config: Config,
) -> None:
    document = _Document(config)
    strategy = _strategy(config, _Graph(), document)
    prefix = config.indexing.opensearch.entities_index_prefix
    query = SearchQuery(query="q")
    with query_scope():
        await strategy._ids_passing(query, prefix, ["u9", "u8"], {"attr_grp": "a"})
        passed = await strategy._ids_passing(
            query,
            prefix,
            ["u0", "u1", "u2", "u8", "u9"],
            {"attr_grp": "a"},
            enough=2,
        )
    assert passed == {"u0", "u1"}
    assert [ids for _, ids in document.calls] == [["u9", "u8"], ["u0", "u1", "u2"]]


async def test_the_check_walks_a_long_list_once(config: Config) -> None:
    # The remaining ids were rescanned per batch: quadratic in the list.
    lookups = 0

    class _Id(str):
        __slots__ = ()

        def __hash__(self) -> int:
            nonlocal lookups
            lookups += 1
            return str.__hash__(self)

    config.indexing.opensearch.index_settings = {"index.max_result_window": 10}
    document = _Document(config)
    strategy = _strategy(config, _Graph(), document)
    ids = [_Id(f"e{i}") for i in range(2_000)]
    passed = await strategy._ids_passing(
        SearchQuery(query="q"),
        config.indexing.opensearch.entities_index_prefix,
        ids,
        {"attr_grp": "a"},
    )
    assert len(passed) == 2_000
    assert lookups < 10 * len(ids)  # the rescan: about 100 per id
