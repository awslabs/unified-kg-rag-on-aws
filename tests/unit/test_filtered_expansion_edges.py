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

import asyncio
import logging
from typing import Any

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401
from unified_kg_rag.adapters.retrieval.base import query_scope
from unified_kg_rag.adapters.storage.filter_schema import id_batch_size
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

    async def aid_batch_size(self, prefix: str, suffix: str | None) -> int:
        return id_batch_size(self.o.index_settings)

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
            # Best first, as the retriever's importance sort: a page holds
            # the top_k best edges only.
            for eid, a, b, rank in sorted(self.edges, key=lambda e: (-e[3], e[0])):
                if len(out) >= query.top_k:
                    break
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
    # endpoints, names and order (its text units only under two keys).
    assert index.edge_fields and all(
        f and "description" not in f and "text_unit_ids" not in f
        for f in index.edge_fields
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
    assert all(f and "text_unit_ids" in f for f in index.edge_fields)


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


class _Ranked(_Index):
    """Entities with their own rank and indexed name."""

    def __init__(self, config: Config, ranks: dict[str, float], **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.ranks = ranks
        self.names: dict[str, str] = {}

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        results = await super().aretrieve(query)
        if (query.index_prefixes or [""])[0] != self.o.entities_index_prefix:
            return results
        out = []
        for r in results:
            r = _entity(str(r.source), self.ranks.get(str(r.source), 0.0))
            if name := self.names.get(str(r.source)):
                r.metadata["name"] = name
            out.append(r)
        return out


async def test_the_score_band_follows_the_walks_distance_and_rank(
    config: Config,
) -> None:
    # A vertex n hops out scores in (1/(n+2), 1/(n+1)], its rank over the
    # highest rank read placing it inside the band, as the graph walk
    # scores (graph_relevance): B one hop out at half the top rank, D two
    # hops out at a quarter.
    index = _Ranked(config, {"S": 4.0, "B": 2.0, "D": 1.0})
    by_id = {r.source: r for r in await _expand(config, index, {"attr_grp": "a"})}
    assert by_id["B"].score == pytest.approx(1 / 3 + (1 / 2 - 1 / 3) * 0.75)
    assert by_id["D"].score == pytest.approx(1 / 4 + (1 / 3 - 1 / 4) * 0.625)

    # No importance source: the middle of each band.
    config.indexing.neptune.entity_importance_source = "none"
    by_id = {r.source: r for r in await _expand(config, index, {"attr_grp": "a"})}
    assert by_id["D"].score == pytest.approx(1 / 4 + (1 / 3 - 1 / 4) * 0.75)


async def test_a_paths_last_name_is_the_entitys_own(config: Config) -> None:
    # The entities index names the reached entity; an edge's endpoint name
    # (the relationships index's copy) names only the vertices before it.
    index = _Ranked(config, {})
    index.names = {"D": "Depot Inc"}
    kept = await _expand(config, index, {"attr_grp": "a"})
    assert _paths(kept)["D"] == "S -> B -> Depot Inc"


async def test_the_walk_stops_at_its_vertex_budget(config: Config) -> None:
    # top_k * retrieval_multiplier * traversal_fetch_multiplier vertices in
    # all: no further vertex is taken, and no further hop is asked for.
    config.indexing.neptune.traversal_fetch_multiplier = 1
    config.indexing.neptune.max_results_per_hop = 10
    config.indexing.neptune.max_hops = 3
    edges = [(f"r{i}", "S", f"v{i}", float(10 - i)) for i in range(5)]
    edges += [("r-far", "v0", "far", 1.0)]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    strategy = _strategy(config, _Graph(), index)
    await strategy._expand_via_graph(
        SearchQuery(query="q", top_k=2, filters={"attr_grp": "a"}), ["S"]
    )
    o = config.indexing.opensearch
    (entity_ids,) = [
        f["id"] for prefix, f in index.calls if prefix == o.entities_index_prefix
    ]
    assert sorted(entity_ids) == ["S", "v0", "v1"]
    hops = [f for prefix, f in index.calls if prefix == o.relationships_index_prefix]
    assert len(hops) == 2  # one hop: its two requests (source and target side)


async def test_the_vertex_budget_counts_the_traversal_fetch_multiplier(
    config: Config,
) -> None:
    # top_k * retrieval_multiplier * traversal_fetch_multiplier: with 2 the
    # walk reaches twice the expansion limit (the cut comes after it).
    config.indexing.neptune.traversal_fetch_multiplier = 2
    config.indexing.neptune.max_results_per_hop = 10
    edges = [(f"r{i}", "S", f"v{i}", float(10 - i)) for i in range(6)]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    strategy = _strategy(config, _Graph(), index)
    await strategy._expand_via_graph(
        SearchQuery(query="q", top_k=2, filters={"attr_grp": "a"}), ["S"]
    )
    o = config.indexing.opensearch
    (entity_ids,) = [
        f["id"] for prefix, f in index.calls if prefix == o.entities_index_prefix
    ]
    assert sorted(entity_ids) == ["S", "v0", "v1", "v2", "v3"]


class _Named(_Index):
    """Relationships whose endpoints carry names that are not their ids."""

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        results = await super().aretrieve(query)
        if (query.index_prefixes or [""])[0] == self.o.relationships_index_prefix:
            for r in results:
                for end in ("source", "target"):
                    r.metadata[f"{end}_name"] = f"{r.metadata[f'{end}_id']} Corp"
        return results


async def test_a_path_names_the_vertices_before_the_entity_by_their_edges(
    config: Config,
) -> None:
    # The vertices a path goes through are named by the relationships the
    # walk crossed (the entities index is read for the reached entity only).
    kept = await _expand(config, _Named(config), {"attr_grp": "a"})
    assert _paths(kept)["D"] == "S Corp -> B Corp -> D"


async def test_an_unknown_relationships_index_keeps_the_walk(config: Config) -> None:
    # A backend that cannot tell whether the index exists (ahas_index None)
    # and returns no passing edge: the walk keeps the seeds rather than
    # falling back to the unchecked graph walk; only a missing index does.
    class _Unknown(_Index):
        async def ahas_index(self, prefix: str, suffix: str | None) -> bool | None:
            return None

    graph = _Graph()
    kept = await _expand(
        config, _Unknown(config, passing=set()), {"attr_grp": "a"}, graph
    )
    assert graph.walks == 0
    assert [r.source for r in kept] == ["S"]


async def test_two_keys_need_one_document_passing_a_reached_entity(
    config: Config,
) -> None:
    # Both edges have a text unit passing both keys, D none: the entities
    # index unions D's documents' values per key, so D is checked by its
    # text units too.
    index = _Index(config, units_passing={"u-r-sb", "u-r-bd", "u-S", "u-B"})
    kept = await _expand(config, index, {"attr_grp": "a", "attr_category": "x"})
    assert sorted(_paths(kept)) == ["B", "S"]


def _relationship_calls(config: Config, index: _Index) -> list[dict]:
    prefix = config.indexing.opensearch.relationships_index_prefix
    return [f for p, f in index.calls if p == prefix]


@pytest.mark.parametrize("bound", ["share", "per-hop"])
async def test_a_hub_seed_does_not_crowd_out_another_seeds_neighbours(
    config: Config, bound: str
) -> None:
    # Regression: one budget for all seeds and a page of the frontier's best
    # edges: the hub's 150 high-ranked edges filled the 100-hit page and
    # the budget, and S2's only neighbour was never reached. Each seed now
    # reaches at most its share of the budget (the graph walk splits it per
    # seed), and a full page is asked again for the vertices still open:
    # the hub closes on its share, or on max_results_per_hop.
    edges = [(f"h{i:03d}", "HUB", f"N{i:03d}", 100.0) for i in range(150)]
    edges.append(("s2", "S2", "X", 1.0))
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    strategy = _strategy(config, _Graph(), index)
    top_k = 10 if bound == "share" else 100  # share 15 or 150 > 50 per hop
    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", top_k=top_k, filters={"attr_grp": "a"}), ["HUB", "S2"]
    )
    o = config.indexing.opensearch
    (reached,) = [f["id"] for p, f in index.calls if p == o.entities_index_prefix]
    assert "X" in reached
    assert len([v for v in reached if v.startswith("N")]) == (
        15 if bound == "share" else 50
    )
    if bound == "per-hop":  # no cut: every reached entity is kept
        assert _paths(kept)["X"] == "S2 -> X"


async def test_the_budget_is_split_across_the_seeds(config: Config) -> None:
    # A's edges rank above B's: one budget went to A's neighbours alone.
    config.indexing.neptune.traversal_fetch_multiplier = 1
    edges = [(f"a{i}", "A", f"va{i}", 9.0 - i) for i in range(3)]
    edges += [(f"b{i}", "B", f"vb{i}", 3.0 - i) for i in range(3)]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    strategy = _strategy(config, _Graph(), index)
    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", top_k=2, filters={"attr_grp": "a"}), ["A", "B"]
    )
    assert sorted(_paths(kept)) == ["A", "B", "va0", "vb0"]


async def test_every_seed_reaches_its_share(config: Config) -> None:
    # Regression: the shares round up but one global budget capped the
    # total, so with 21 seeds (share 2 of 30) the last 6 by edge rank
    # reached nothing. The shares bound the total, as on the graph walk.
    edges = [
        (f"e{j:02d}-{i}", f"S{j:02d}", f"V{j:02d}-{i}", 100.0 - j)
        for j in range(21)
        for i in range(5)
    ]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    seeds = [f"S{j:02d}" for j in range(21)]
    await _strategy(config, _Graph(), index)._expand_via_graph(
        SearchQuery(query="q", top_k=10, filters={"attr_grp": "a"}), seeds
    )
    o = config.indexing.opensearch
    (reached,) = [f["id"] for p, f in index.calls if p == o.entities_index_prefix]
    for seed in seeds:
        assert len([v for v in reached if v.startswith(f"V{seed[1:]}-")]) == 2


async def test_a_full_page_asks_each_open_vertex_once_more(config: Config) -> None:
    # Regression: a full page was asked again for the still-open vertices
    # together, round after round, each round closing about one hub (20
    # hubs: 30 sequential requests, and L's neighbour never reached). Each
    # open vertex is now asked once more on its own, all at once, at the
    # full end only: one more round.
    edges = [
        (f"m{j:02d}-{i:03d}", f"M{j:02d}", f"V{j:02d}-{i:03d}", 100.0)
        for j in range(20)
        for i in range(120)
    ]
    edges.append(("l0", "L", "LX", 1.0))
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    seeds = [f"M{j:02d}" for j in range(20)] + ["L"]
    config.indexing.neptune.max_hops = 1
    kept = await _strategy(config, _Graph(), index)._expand_via_graph(
        SearchQuery(query="q", top_k=10, filters={"attr_grp": "a"}), seeds
    )
    o = config.indexing.opensearch
    (reached,) = [f["id"] for p, f in index.calls if p == o.entities_index_prefix]
    assert "LX" in reached and len(reached) == 21 + 2 * 20 + 1
    asks = _relationship_calls(config, index)
    # One request per end, then one per vertex still open at the full end
    # (the source side): the first page closed one hub of the 20.
    assert len(asks) == 2 + 20
    assert all(len(f["source_id"]) == 1 for f in asks[2:])
    assert len(kept) == 21 + 10  # the seeds and the cut


async def test_a_closed_seeds_vertices_are_not_asked(config: Config) -> None:
    # Before a hop the frontier keeps only the vertices whose seed is short
    # of its share: a1 belongs to A, which took its one (B found nothing,
    # so the budget of 2 was not spent and a1 was asked at hop 2).
    config.indexing.neptune.traversal_fetch_multiplier = 1
    edges = [("r1", "A", "a1", 2.0), ("r2", "a1", "a2", 1.0)]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    await _strategy(config, _Graph(), index)._expand_via_graph(
        SearchQuery(query="q", top_k=2, filters={"attr_grp": "a"}), ["A", "B"]
    )
    calls = _relationship_calls(config, index)
    assert len(calls) == 2  # hop 1 only
    assert all(
        sorted(f.get("source_id", f.get("target_id"))) == ["A", "B"] for f in calls
    )


async def test_two_keys_do_not_let_a_hubs_failing_edges_starve_a_seed(
    config: Config,
) -> None:
    # Regression: the hub's 150 edges pass the relationships index (each key
    # by another document) but no document passes both; they filled the
    # page, closed nothing, and the "same page again" guard stopped before
    # S2 was asked: X was never reached.
    hub = [(f"h{i:03d}", "H", f"N{i:03d}", 100.0) for i in range(150)]
    edges = [*hub, ("s2", "S2", "X", 1.0)]
    index = _Index(
        config,
        edges=edges,
        passing={e[0] for e in edges},
        units_passing={"u-s2", "u-X", "u-S2", "u-H"},
    )
    kept = await _strategy(config, _Graph(), index)._expand_via_graph(
        SearchQuery(query="q", filters={"attr_grp": "a", "attr_category": "x"}),
        ["H", "S2"],
    )
    assert _paths(kept)["X"] == "S2 -> X"


async def test_a_query_asks_for_a_frontiers_edges_once(config: Config) -> None:
    # DRIFT's follow-ups expand the same seeds again, concurrently too: the
    # same frontier's edges are fetched once a query, the request in flight
    # shared.
    index = _Index(config)
    strategy = _strategy(config, _Graph(), index)
    query = SearchQuery(query="q", filters={"attr_grp": "a"})
    with query_scope():
        first, second = await asyncio.gather(
            strategy._expand_via_graph(query, ["S"]),
            strategy._expand_via_graph(query, ["S"]),
        )
        third = await strategy._expand_via_graph(query, ["S"])
    assert _paths(first) == _paths(second) == _paths(third)
    one = _Index(config)
    await _strategy(config, _Graph(), one)._expand_via_graph(query, ["S"])
    assert len(_relationship_calls(config, index)) == len(
        _relationship_calls(config, one)
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"filters": {"attr_grp": "a"}}, {"filters": {"attr_grp": "b"}}),
        ({"source_fields": ["id"]}, {"source_fields": ["id", "rank"]}),
    ],
    ids=["filters", "source_fields"],
)
async def test_incident_edges_asked_otherwise_are_asked_again(
    config: Config, first: dict, second: dict
) -> None:
    # The per-query cache keys on the filters and the fields read too: an
    # edge page read under another filter, or without a field, answered for
    # the other.
    index = _Index(config)
    strategy = _strategy(config, _Graph(), index)

    async def ask(how: dict) -> None:
        query = SearchQuery(query="q", filters=how.get("filters", {"attr_grp": "a"}))
        await strategy._incident_relationships(
            query, ["S"], source_fields=how.get("source_fields")
        )

    with query_scope():
        await ask(first)
        await ask(first)
        asked = len(_relationship_calls(config, index))
        await ask(second)
    assert len(_relationship_calls(config, index)) == 2 * asked


async def test_a_waiter_asks_again_after_the_request_in_flight_failed(
    config: Config,
) -> None:
    # The owner's request failed: the concurrent caller waiting for it asks
    # for itself instead of taking "no edges".
    release = asyncio.Event()

    class _FailsFirst(_Index):
        failed = False

        async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
            if query.index_prefixes == [self.o.relationships_index_prefix]:
                await release.wait()
                if not self.failed:
                    self.failed = True
                    raise RuntimeError("store hiccup")
            return await super().aretrieve(query)

    strategy = _strategy(config, _Graph(), _FailsFirst(config))
    query = SearchQuery(query="q", filters={"attr_grp": "a"})
    with query_scope():
        owner = asyncio.ensure_future(strategy._incident_relationships(query, ["S"]))
        await asyncio.sleep(0)
        waiter = asyncio.ensure_future(strategy._incident_relationships(query, ["S"]))
        await asyncio.sleep(0)
        release.set()
        await owner
        edges, _ = await waiter
    assert sorted(str(e.source) for e in edges) == ["r-sb"]


async def test_a_failed_edge_fetch_is_asked_again(config: Config) -> None:
    class _FailsOnce(_Index):
        failed = False

        async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
            if query.index_prefixes == [self.o.relationships_index_prefix] and not (
                self.failed
            ):
                self.failed = True
                raise RuntimeError("store hiccup")
            return await super().aretrieve(query)

    strategy = _strategy(config, _Graph(), _FailsOnce(config))
    query = SearchQuery(query="q", filters={"attr_grp": "a"})
    with query_scope():
        degraded = await strategy._expand_via_graph(query, ["S"])
        again = await strategy._expand_via_graph(query, ["S"])
    assert sorted(_paths(degraded)) == ["S"]
    assert sorted(_paths(again)) == ["B", "D", "S"]


async def test_two_keys_read_the_reached_entities_once(config: Config) -> None:
    # The entities were read under the caller's filters: checking their ids
    # against the entities index again asked the same thing twice.
    index = _Index(config, units_passing={"u-r-sb", "u-r-bd", "u-B", "u-D"})
    await _expand(config, index, {"attr_grp": "a", "attr_category": "x"})
    o = config.indexing.opensearch
    assert len([p for p, _ in index.calls if p == o.entities_index_prefix]) == 1


async def test_equal_edges_and_entities_are_ordered_by_id(config: Config) -> None:
    # Ties followed the store's order (the order of the two endpoint
    # requests, the index order of the entities read): a subset index of
    # the passing documents and the filtered index kept different ones.
    config.indexing.neptune.max_results_per_hop = 1
    edges = [("r-b", "S", "B", 1.0), ("r-a", "A", "S", 1.0)]
    kept = await _expand(
        config, _Index(config, edges=edges, passing={"r-a", "r-b"}), {"attr_grp": "a"}
    )
    assert sorted(_paths(kept)) == ["A", "S"]  # r-a before r-b

    class _Reversed(_Index):
        async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
            return list(reversed(await super().aretrieve(query)))

    config.indexing.neptune.max_results_per_hop = 50
    edges = [(f"r{i}", "S", f"v{i}", 1.0) for i in range(9)]
    index = _Reversed(config, edges=edges, passing={e[0] for e in edges})
    kept = await _strategy(config, _Graph(), index)._expand_via_graph(
        SearchQuery(query="q", top_k=3, filters={"attr_grp": "a"}), ["S"]
    )
    assert sorted(_paths(kept)) == ["S", "v0", "v1", "v2"]


async def test_a_full_page_is_logged_once_a_query(
    config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    # Three hubs, one walk each: every walk asks a full page of its own (one
    # hub walked three times reads its edges once, so it could not tell).
    hubs = ("HUB1", "HUB2", "HUB3")
    edges = [
        (f"{hub}-{i:03d}", hub, f"{hub}-N{i:03d}", 1.0)
        for hub in hubs
        for i in range(120)
    ]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    strategy = _strategy(config, _Graph(), index)
    with caplog.at_level(logging.WARNING), query_scope():
        for hub in hubs:
            await strategy._expand_via_graph(
                SearchQuery(query="q", top_k=5, filters={"attr_grp": "a"}), [hub]
            )
    assert caplog.text.count("page cap") == 1
    caplog.clear()
    with caplog.at_level(logging.WARNING), query_scope():
        await strategy._expand_via_graph(
            SearchQuery(query="q", top_k=5, filters={"attr_grp": "a"}), ["HUB1"]
        )
    assert caplog.text.count("page cap") == 1  # once a query, not once ever


# --------------------------------------------------------------------------- #
# The filter check
# --------------------------------------------------------------------------- #


class _Document:
    """Every id passes but those of relationships not in ``_PASSING``."""

    def __init__(self, config: Config) -> None:
        self.o = config.indexing.opensearch
        self.calls: list[tuple[str, list[str]]] = []

    async def aid_batch_size(self, prefix: str, suffix: str | None) -> int:
        return id_batch_size(self.o.index_settings)

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
