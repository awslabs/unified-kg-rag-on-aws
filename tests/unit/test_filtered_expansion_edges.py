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
from unified_kg_rag.adapters.retrieval.base import query_scope, seed_share
from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
    SearchType,
)
from unified_kg_rag.domain.retrieval.strategy_registry import get_strategy_spec
from unified_kg_rag.shared.exceptions import InvalidFilterError, StoreTimeoutError

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

    def filter_fields(self) -> None:
        return None  # no declared schema: the document filters are checked

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
        self.excludes: list[tuple[str, dict | None]] = []  # (asked end, exclude)
        self.honours_exclude = True
        self.edge_fields: list[list[str] | None] = []

    async def ahas_index(self, prefix: str, suffix: str | None) -> bool:
        return prefix != self.o.relationships_index_prefix or self.has_relationships

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits.of(self.o.index_settings)

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        prefix = (query.index_prefixes or [""])[0]
        limits = await self.aindex_limits(prefix, query.suffix)
        # A store that batches nothing: a terms clause (or exclusion) over
        # max_terms_count is refused and a search returns at most
        # max_result_window hits. The strategies size every list themselves.
        filters = dict(query.filters or {})
        self.calls.append((prefix, filters))
        for values in [*filters.values(), *(query.exclude or {}).values()]:
            if isinstance(values, list) and len(values) > limits.max_terms_count:
                raise InvalidFilterError(f"{len(values)} values")
        query = query.model_copy(
            update={"top_k": min(query.top_k, limits.max_result_window)}
        )
        if prefix == self.o.relationships_index_prefix:
            self.edge_fields.append(query.source_fields)
            if not self.has_relationships:
                return []
            side = "source_id" if "source_id" in filters else "target_id"
            ends = set(filters[side])
            self.excludes.append((side, query.exclude))
            # The store leaves the excluded edges out before the page is cut.
            excluded = {
                k: set(v)
                for k, v in (query.exclude or {}).items()
                if self.honours_exclude
            }
            out = []
            # Best first, as the retriever's importance sort: a page holds
            # the top_k best edges only.
            for eid, a, b, rank in sorted(self.edges, key=lambda e: (-e[3], e[0])):
                if len(out) >= query.top_k:
                    break
                endpoints = {"source_id": a, "target_id": b}
                if any(endpoints[k] in v for k, v in excluded.items()):
                    continue
                if eid in self.passing and endpoints[side] in ends:
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


async def test_a_full_page_asks_the_open_vertices_together_then_each(
    config: Config,
) -> None:
    # Regression: a full page was asked again for the still-open vertices
    # together, round after round, each round closing about one hub (20
    # hubs: 30 sequential requests, and L's neighbour never reached). The
    # open vertices are now asked together once (with the reached vertices
    # left out, a page of new edges), then, when that page is full too,
    # each on its own, all at once, at the full end only.
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
    # One request per end; the open vertices together at the full end (the
    # source side; the first page closed one hub of the 20); then one per
    # vertex still open (the second page closed one more).
    assert len(asks) == 2 + 1 + 19
    assert len(asks[2]["source_id"]) == 20
    assert all(len(f["source_id"]) == 1 for f in asks[3:])
    assert len(kept) == 21 + 10  # the seeds and the cut


async def test_the_requests_on_their_own_run_max_concurrency_at_a_time(
    config: Config,
) -> None:
    # Regression: every open vertex was asked at once (DRIFT under a broad
    # filter: 420 requests in flight).
    edges = [
        (f"m{j:02d}-{i:03d}", f"M{j:02d}", f"V{j:02d}-{i:03d}", 100.0)
        for j in range(20)
        for i in range(120)
    ]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    in_flight, peak = 0, 0
    ask = index.aretrieve

    async def counted(query: SearchQuery) -> list[RetrievalResult]:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.001)
            return await ask(query)
        finally:
            in_flight -= 1

    index.aretrieve = counted  # type: ignore[method-assign]
    config.search.max_concurrency = 3
    config.indexing.neptune.max_hops = 1
    await _strategy(config, _Graph(), index)._expand_via_graph(
        SearchQuery(query="q", top_k=10, filters={"attr_grp": "a"}),
        [f"M{j:02d}" for j in range(20)],
    )
    assert len(_relationship_calls(config, index)) == 2 + 1 + 18
    assert peak == 3


async def test_a_fatal_error_on_its_own_cancels_the_rest_of_its_wave(
    config: Config,
) -> None:
    # The requests of a wave run together: one's fatal error (a denied
    # store) left the others going to the store after the query had failed.
    from unified_kg_rag.shared.exceptions import StoreAccessDeniedError

    edges = [
        (f"m{j:02d}-{i:03d}", f"M{j:02d}", f"V{j:02d}-{i:03d}", 100.0)
        for j in range(20)
        for i in range(120)
    ]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    o = config.indexing.opensearch
    on_its_own: list[str] = []
    cancelled: list[str] = []
    ask = index.aretrieve

    async def failing(query: SearchQuery) -> list[RetrievalResult]:
        ends = (query.filters or {}).get("source_id") or []
        if query.index_prefixes != [o.relationships_index_prefix] or len(ends) != 1:
            return await ask(query)
        on_its_own.append(ends[0])
        if len(on_its_own) == 1:
            await asyncio.sleep(0.01)
            raise StoreAccessDeniedError("denied")
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(ends[0])
            raise
        return []

    index.aretrieve = failing  # type: ignore[method-assign]
    config.search.max_concurrency = 3
    config.indexing.neptune.max_hops = 1
    with pytest.raises(StoreAccessDeniedError):
        await asyncio.wait_for(
            _strategy(config, _Graph(), index)._expand_via_graph(
                SearchQuery(query="q", top_k=10, filters={"attr_grp": "a"}),
                [f"M{j:02d}" for j in range(20)],
            ),
            2,
        )
    # The first wave only; the two beside the failed request cancelled.
    assert len(on_its_own) == 3
    assert sorted(cancelled) == sorted(on_its_own[1:])


async def test_a_live_window_below_the_page_is_judged_full(
    config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    # Regression: a page asked for max_query_size edges and the store
    # returned the live max_result_window's worth, so it never read as full:
    # the walk stopped after 2 of S's 6 neighbours, with no warning.
    caplog.set_level("WARNING")
    edges = [(f"s{i}", "S", f"N{i}", 10.0 - i) for i in range(6)]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    o = config.indexing.opensearch

    async def limits(prefix: str, suffix: str | None) -> IndexLimits:
        small = prefix == o.relationships_index_prefix
        return IndexLimits(max_result_window=2) if small else IndexLimits()

    index.aindex_limits = limits  # type: ignore[method-assign]
    o.max_query_size = 3
    config.indexing.neptune.max_hops = 1
    await _strategy(config, _Graph(), index)._expand_via_graph(
        SearchQuery(query="q", top_k=10, filters={"attr_grp": "a"}), ["S"]
    )
    (reached,) = [f["id"] for p, f in index.calls if p == o.entities_index_prefix]
    assert set(reached) == {"S"} | {f"N{i}" for i in range(6)}
    assert "hit the 2-hit page cap" in caplog.text


async def test_a_window_shrunk_under_the_walk_is_judged_full(config: Config) -> None:
    # Regression: the page was sized by the limits read before the request;
    # a store that rejected that window (an index rebuilt smaller) was asked
    # again with the live one and returned 2 edges, which the stale page of
    # 3 did not read as full: the walk stopped at 2 of S's 6 neighbours.
    edges = [(f"s{i}", "S", f"N{i}", 10.0 - i) for i in range(6)]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    o = config.indexing.opensearch
    reads = 0

    async def limits(prefix: str, suffix: str | None) -> IndexLimits:
        nonlocal reads
        if prefix != o.relationships_index_prefix:
            return IndexLimits()
        reads += 1
        return IndexLimits(max_result_window=10 if reads == 1 else 2)

    index.aindex_limits = limits  # type: ignore[method-assign]
    o.max_query_size = 3
    config.indexing.neptune.max_hops = 1
    await _strategy(config, _Graph(), index)._expand_via_graph(
        SearchQuery(query="q", top_k=10, filters={"attr_grp": "a"}), ["S"]
    )
    (reached,) = [f["id"] for p, f in index.calls if p == o.entities_index_prefix]
    assert set(reached) == {"S"} | {f"N{i}" for i in range(6)}


async def test_internal_lists_over_max_terms_count_do_not_fail_the_query(
    config: Config,
) -> None:
    # Regression: the walk's exclusion (every reached vertex) and its read of
    # the reached entities named more values than max_terms_count, and the
    # query failed with InvalidFilterError; then the read kept the vertices
    # reached first only (S and A of 5). The read now goes in batches, and
    # the exclusion (it only saves page room) keeps the vertices reached
    # last, the frontier's neighbourhood.
    chain = ["S", "A", "B", "C", "D"]
    pairs = zip(chain, chain[1:], strict=False)
    edges = [(f"e{i}", a, b, 1.0) for i, (a, b) in enumerate(pairs)]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    config.indexing.opensearch.index_settings = {"max_terms_count": 2}
    config.indexing.neptune.max_hops = 4
    kept = await _strategy(config, _Graph(), index)._expand_via_graph(
        SearchQuery(query="q", top_k=10, filters={"attr_grp": "a"}), ["S"]
    )
    o = config.indexing.opensearch
    reads = [f["id"] for p, f in index.calls if p == o.entities_index_prefix]
    assert [v for read in reads for v in read] == chain
    assert {r.source for r in kept} == set(chain)
    # The last hop (from C, both ends) leaves out B and C, not S and A.
    assert [list((x or {}).values()) for _, x in index.excludes[-2:]] == [
        [["B", "C"]],
        [["B", "C"]],
    ]


async def test_a_fanned_out_frontier_is_asked_in_batches_of_max_terms_count(
    config: Config,
) -> None:
    # Regression: the second hop's frontier (S's 3 neighbours) was cut to
    # max_terms_count (2): the third neighbour's own neighbour was never
    # reached. The store refuses a list over the limit, so it must batch.
    edges = [(f"s{i}", "S", f"A{i}", 2.0) for i in range(3)]
    edges += [(f"a{i}", f"A{i}", f"B{i}", 1.0) for i in range(3)]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    config.indexing.opensearch.index_settings = {"max_terms_count": 2}
    config.indexing.neptune.max_hops = 2
    kept = await _strategy(config, _Graph(), index)._expand_via_graph(
        SearchQuery(query="q", top_k=10, filters={"attr_grp": "a"}), ["S"]
    )
    assert {r.source for r in kept} == {"S"} | {
        f"{x}{i}" for x in "AB" for i in range(3)
    }
    hop2 = [f["source_id"] for f in _relationship_calls(config, index)[2:4]]
    assert sorted(map(len, hop2)) == [1, 2]


async def test_a_frontier_over_max_terms_count_is_asked_in_batches(
    config: Config,
) -> None:
    # Regression: a frontier of more vertices than max_terms_count was cut
    # to that many: the seeds past it were never asked (2 of 5 neighbours).
    seeds = [f"S{i}" for i in range(5)]
    edges = [(f"e{i}", f"S{i}", f"N{i}", 1.0) for i in range(5)]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    config.indexing.opensearch.index_settings = {"max_terms_count": 2}
    config.indexing.neptune.max_hops = 1
    kept = await _strategy(config, _Graph(), index)._expand_via_graph(
        SearchQuery(query="q", top_k=10, filters={"attr_grp": "a"}), seeds
    )
    assert {r.source for r in kept} == set(seeds) | {f"N{i}" for i in range(5)}
    asked = [f.get("source_id") for f in _relationship_calls(config, index)]
    assert sorted(len(a) for a in asked if a) == [1, 2, 2]


async def test_the_requests_on_their_own_are_sized_by_their_wave(
    config: Config,
) -> None:
    # Regression: every vertex asked on its own asked for what all the open
    # seeds might still take (100 seeds: 8,300 edges read for 200 reached).
    # They go in waves of max_concurrency; each asks for what its wave may
    # take, with what the waves before it reached left out.
    seeds = [f"M{j}" for j in range(6)]
    edges = [
        (f"m{j}-{i:03d}", f"M{j}", f"V{j}-{i:03d}", 100.0)
        for j in range(6)
        for i in range(120)
    ]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    asked: list[tuple[list[str], int, list[str]]] = []
    ask = index.aretrieve

    async def recorded(query: SearchQuery) -> list[RetrievalResult]:
        ends = (query.filters or {}).get("source_id")
        if ends is not None and len(ends) == 1:
            exclude = (query.exclude or {}).get("target_id") or []
            asked.append((ends, query.top_k, exclude))
        return await ask(query)

    index.aretrieve = recorded  # type: ignore[method-assign]
    config.search.max_concurrency = 2
    config.indexing.neptune.max_hops = 1
    query = SearchQuery(query="q", top_k=10, filters={"attr_grp": "a"})
    await _strategy(config, _Graph(), index)._expand_via_graph(query, seeds)

    share = seed_share(query, config.indexing.neptune.traversal_fetch_multiplier, 6)
    # The first page closed M0, the joint one M1; M2-M5 go two at a time.
    assert [ends for ends, _, _ in asked] == [["M2"], ["M3"], ["M4"], ["M5"]]
    assert {size for _, size, _ in asked} == {2 * share}
    first_wave = {f"V{j}-{i:03d}" for j in (2, 3) for i in range(share)}
    assert first_wave <= set(asked[2][2]) and first_wave <= set(asked[3][2])


async def test_the_requests_on_their_own_never_ask_past_a_page(
    config: Config,
) -> None:
    # A wave's room (what its vertices may still take, here more than a
    # page) is capped at the page the store returns: a request on its own
    # asks for at most a page.
    seeds = [f"M{j}" for j in range(6)]
    edges = [
        (f"m{j}-{i:03d}", f"M{j}", f"V{j}-{i:03d}", 100.0)
        for j in range(6)
        for i in range(120)
    ]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    sizes: list[int] = []
    ask = index.aretrieve

    async def recorded(query: SearchQuery) -> list[RetrievalResult]:
        ends = (query.filters or {}).get("source_id")
        if ends is not None and len(ends) == 1:
            sizes.append(query.top_k)
        return await ask(query)

    index.aretrieve = recorded  # type: ignore[method-assign]
    config.indexing.opensearch.max_query_size = 20
    config.search.max_concurrency = 6
    config.indexing.neptune.max_hops = 1
    await _strategy(config, _Graph(), index)._expand_via_graph(
        SearchQuery(query="q", top_k=30, filters={"attr_grp": "a"}), seeds
    )
    assert sizes and set(sizes) == {20}


async def test_seeds_sharing_their_neighbours_each_reach_their_share(
    config: Config,
) -> None:
    # Regression: the concurrent requests per open vertex left out the same
    # reached vertices and each asked for what its vertex needed: they all
    # got the same best shared neighbours, the first seed took them and the
    # others nothing (20 seeds around a shared core: 4 entities reached
    # instead of 40). Each now asks for what all the open seeds may take.
    seeds = [f"S{i:02d}" for i in range(20)]
    edges = [
        (f"{s}-c{j:03d}", s, f"C{j:03d}", 100.0) for s in seeds for j in range(150)
    ]
    edges += [(f"{s}-p{j}", s, f"{s}P{j}", 1.0) for s in seeds for j in range(5)]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    config.indexing.neptune.max_hops = 1
    await _strategy(config, _Graph(), index)._expand_via_graph(
        SearchQuery(query="q", top_k=10, filters={"attr_grp": "a"}), seeds
    )
    o = config.indexing.opensearch
    (reached,) = [f["id"] for p, f in index.calls if p == o.entities_index_prefix]
    assert len(reached) == 20 + 20 * 2  # share: ceil(10 * 3 / 20)


@pytest.mark.parametrize("respected", [True, False])
async def test_edges_back_to_reached_vertices_do_not_fill_a_page(
    config: Config, respected: bool
) -> None:
    # Regression: B's 150 best edges lead back to A (a seed: reached) and
    # filled every page of B's, so B's one new neighbour X was never
    # reached. The store leaves edges to reached vertices out (exclude on
    # the other end); a store that ignored it would starve B again.
    edges = [(f"ba{i:03d}", "B", "A", 100.0) for i in range(150)]
    edges += [("bx", "B", "X", 1.0), ("ay", "A", "Y", 1.0)]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    index.honours_exclude = respected
    config.indexing.neptune.max_hops = 1
    await _strategy(config, _Graph(), index)._expand_via_graph(
        SearchQuery(query="q", top_k=10, filters={"attr_grp": "a"}), ["A", "B"]
    )
    o = config.indexing.opensearch
    (reached,) = [f["id"] for p, f in index.calls if p == o.entities_index_prefix]
    assert ("X" in reached) is respected
    # Left out at the other end of the asked one: an edge from B to A is B's.
    other = {"source_id": "target_id", "target_id": "source_id"}
    assert index.excludes
    for side, exclude in index.excludes:
        assert exclude is not None and set(exclude) == {other[side]}
        assert {"A", "B"} <= set(exclude[other[side]])


async def test_two_keys_ask_again_past_the_edges_the_check_dropped(
    config: Config,
) -> None:
    # B's best edges pass the relationships index but no document passes
    # both keys; they come back on each of B's pages (10 on the first, all
    # 15 on the joint one). B's request on its own asks for them on top of
    # what the open seeds may take, so B's one passing edge (ranked last)
    # is on its page.
    config.indexing.opensearch.max_query_size = 20
    edges = [(f"a{i}", "A", f"VA{i}", 200.0) for i in range(10)]
    edges += [(f"bf{i:02d}", "B", f"F{i:02d}", 100.0) for i in range(15)]
    edges += [(f"hf{i:02d}", "H", f"G{i:02d}", 100.0) for i in range(50)]
    edges.append(("bx", "B", "X", 1.0))
    units = {f"u-{v}" for v in ("bx", "X", "A", "B", "H")}
    units |= {f"u-a{i}" for i in range(10)} | {f"u-VA{i}" for i in range(10)}
    index = _Index(
        config, edges=edges, passing={e[0] for e in edges}, units_passing=units
    )
    config.indexing.neptune.max_hops = 1
    await _strategy(config, _Graph(), index)._expand_via_graph(
        SearchQuery(
            query="q", top_k=2, filters={"attr_grp": "a", "attr_category": "x"}
        ),
        ["A", "H", "B"],
    )
    o = config.indexing.opensearch
    (reached,) = [f["id"] for p, f in index.calls if p == o.entities_index_prefix]
    assert "X" in reached
    asked = [
        f["source_id"] for f in _relationship_calls(config, index) if "source_id" in f
    ]
    # A took its share on the first page; H and B together, then each.
    assert asked[1] == ["H", "B"] and sorted(asked[2:]) == [["B"], ["H"]]


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
        ({"size": 1}, {"size": 3}),
        ({"away_from": ["X"]}, {"away_from": ["A"]}),
    ],
    ids=["filters", "source_fields", "size", "away_from"],
)
async def test_incident_edges_asked_otherwise_are_asked_again(
    config: Config, first: dict, second: dict
) -> None:
    # The per-query cache keys on the filters and the fields read too: an
    # edge page read under another filter, or without a field, answered for
    # the other. So does a page of another size or leaving out other
    # vertices (the walk's pages per vertex differ in both).
    index = _Index(config)
    strategy = _strategy(config, _Graph(), index)

    async def ask(how: dict) -> None:
        query = SearchQuery(query="q", filters=how.get("filters", {"attr_grp": "a"}))
        await strategy._incident_relationships(
            query,
            ["S"],
            source_fields=how.get("source_fields"),
            size=how.get("size"),
            away_from=how.get("away_from"),
        )

    with query_scope():
        await ask(first)
        await ask(first)
        asked = len(_relationship_calls(config, index))
        await ask(second)
    assert len(_relationship_calls(config, index)) == 2 * asked


async def test_the_incident_edge_memo_keys_on_what_the_requests_send(
    config: Config,
) -> None:
    # Regression: the memo keyed on the vertex sets, but past
    # max_terms_count the exclusion keeps the vertices listed last: the same
    # sets in another order leave out other vertices, and the second call
    # got the first's edges.
    edges = [("r-sx", "S", "X", 3.0), ("r-sy", "S", "Y", 2.0), ("r-sz", "S", "Z", 1.0)]
    config.indexing.opensearch.index_settings = {"max_terms_count": 1}
    query = SearchQuery(query="q", filters={"attr_grp": "a"})

    async def ask(away: list[str]) -> list[str]:
        index = _Index(config, edges=edges, passing={e[0] for e in edges})
        strategy = _strategy(config, _Graph(), index)
        found, _ = await strategy._incident_relationships(
            query, ["S"], ends=("source_id",), away_from=away
        )
        return sorted(str(e.source) for e in found)

    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    strategy = _strategy(config, _Graph(), index)
    with query_scope():
        await strategy._incident_relationships(
            query, ["S"], ends=("source_id",), away_from=["X", "Y"]
        )
        found, _ = await strategy._incident_relationships(
            query, ["S"], ends=("source_id",), away_from=["Y", "X"]
        )
    assert sorted(str(e.source) for e in found) == await ask(["Y", "X"])
    assert await ask(["Y", "X"]) == ["r-sy", "r-sz"]


async def test_one_full_batch_makes_its_end_full_and_warns(
    config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    # Past max_terms_count the vertices go in batches: an end whose page
    # came back full in any batch dropped edges, though the others did not.
    config.indexing.opensearch.index_settings = {"max_terms_count": 1}
    config.indexing.opensearch.max_query_size = 2
    edges = [("r-a1", "A", "X1", 3.0), ("r-a2", "A", "X2", 2.0)]
    edges += [("r-a3", "A", "X3", 1.0), ("r-b1", "B", "Y1", 1.0)]
    index = _Index(config, edges=edges, passing={e[0] for e in edges})
    with caplog.at_level(logging.WARNING):
        found, full = await _strategy(config, _Graph(), index)._incident_relationships(
            SearchQuery(query="q", filters={"attr_grp": "a"}),
            ["A", "B"],
            ends=("source_id",),
        )
    assert sorted(str(e.source) for e in found) == ["r-a1", "r-a2", "r-b1"]
    assert full == ("source_id",)
    assert "page cap" in caplog.text


async def test_the_incident_edge_batches_respect_the_search_concurrency(
    config: Config,
) -> None:
    # Regression: every batch of both ends went to the store at once.
    config.indexing.opensearch.index_settings = {"max_terms_count": 1}
    config.search.max_concurrency = 3
    vertices = [f"V{i}" for i in range(10)]
    edges = [(f"e{i}", v, "S", 1.0) for i, v in enumerate(vertices)]
    in_flight = peak = 0

    class _Counting(_Index):
        async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.005)
            in_flight -= 1
            return await super().aretrieve(query)

    index = _Counting(config, edges=edges, passing={e[0] for e in edges})
    found, _ = await _strategy(config, _Graph(), index)._incident_relationships(
        SearchQuery(query="q", filters={"attr_grp": "a"}), vertices
    )
    assert len(found) == 10
    assert peak == 3


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

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits.of(self.o.index_settings)

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


class _Window:
    """A custom retriever following the documented contract over an index
    whose live window is ``window``: it batches nothing and answers a search
    with at most a window of hits, in its own (reverse) order."""

    def __init__(self, window: int) -> None:
        self.window = window
        self.asked: list[list[str]] = []

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits(max_result_window=self.window)

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        ids = list((query.filters or {}).get("id") or [])
        self.asked.append(ids)
        return [_entity(i) for i in sorted(ids, reverse=True)][: self.window]


async def test_a_fetch_by_id_is_batched_by_the_smallest_step_of_its_indexes(
    config: Config,
) -> None:
    # Over two indexes the batch fits the one whose window is smaller;
    # the step is the window (2) here, not max_terms_count (3).
    class _TwoIndexes(_Window):
        async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
            window = 2 if prefix == "small" else 5
            return IndexLimits(max_result_window=window, max_terms_count=3)

    store = _TwoIndexes(window=2)
    strategy = _strategy(config, _Graph(), store)
    ids = [f"e{i}" for i in range(5)]
    found = await strategy._fetch_by_ids(
        store,
        SearchQuery(
            query="",
            search_type=SearchType.LEXICAL,
            top_k=len(ids),
            index_prefixes=["large", "small"],
            filters={"id": ids},
        ),
        "fetch",
    )
    assert store.asked == [["e0", "e1"], ["e2", "e3"], ["e4"]]
    assert sorted(str(r.source) for r in found) == ids


async def test_a_fetch_by_id_reads_the_limits_of_a_single_named_index(
    config: Config,
) -> None:
    # index_prefixes may name one index as a string: its limits, not those
    # of a prefix per character.
    class _ByName(_Window):
        async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
            return IndexLimits(max_result_window=2 if prefix == "small" else 5)

    store = _ByName(window=2)
    ids = [f"e{i}" for i in range(5)]
    await _strategy(config, _Graph(), store)._fetch_by_ids(
        store,
        SearchQuery(
            query="",
            search_type=SearchType.LEXICAL,
            top_k=len(ids),
            index_prefixes="small",
            filters={"id": ids},
        ),
        "fetch",
    )
    assert store.asked == [["e0", "e1"], ["e2", "e3"], ["e4"]]


async def test_a_fetch_by_id_asks_a_repeated_id_once(config: Config) -> None:
    # Regression: a repeated id was asked by two batches, and its hit came
    # back twice.
    store = _Window(window=2)
    found = await _strategy(config, _Graph(), store)._fetch_by_ids(
        store,
        SearchQuery(
            query="",
            search_type=SearchType.LEXICAL,
            top_k=4,
            index_prefixes=["entities"],
            filters={"id": ["e1", "e2", "e1", "e3"]},
        ),
        "fetch",
    )
    assert store.asked == [["e1", "e2"], ["e3"]]
    assert sorted(str(r.source) for r in found) == ["e1", "e2", "e3"]


class _Slow(_Window):
    """A ``_Window`` whose searches take a moment, counted in flight; the
    batch naming ``fail_on`` fails the query with ``error``."""

    def __init__(self, window: int, fail_on: str = "", error: Exception | None = None):
        super().__init__(window)
        self.fail_on, self.error = fail_on, error
        self.in_flight = self.peak = self.started = 0

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.started += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.005)
            if self.error is not None and self.fail_on in query.filters["id"]:
                raise self.error
            return await super().aretrieve(query)
        finally:
            self.in_flight -= 1


async def test_a_fetch_by_id_keeps_to_the_search_concurrency(config: Config) -> None:
    config.search.max_concurrency = 3
    store = _Slow(window=1)
    ids = [f"e{i}" for i in range(20)]
    found = await _strategy(config, _Graph(), store)._fetch_by_ids(
        store,
        SearchQuery(
            query="",
            search_type=SearchType.LEXICAL,
            top_k=len(ids),
            index_prefixes=["entities"],
            filters={"id": ids},
        ),
        "fetch",
    )
    assert len(found) == 20
    assert store.peak == 3


async def test_a_fatal_batch_stops_the_fetch_by_id_from_sending_the_rest(
    config: Config,
) -> None:
    # Regression: a fatal error (a timeout, denied access) of one batch
    # failed the query, and the queued batches still went to the store.
    config.search.max_concurrency = 2
    store = _Slow(window=1, fail_on="e0", error=StoreTimeoutError("timed out"))
    ids = [f"e{i}" for i in range(20)]
    with pytest.raises(StoreTimeoutError):
        await _strategy(config, _Graph(), store)._fetch_by_ids(
            store,
            SearchQuery(
                query="",
                search_type=SearchType.LEXICAL,
                top_k=len(ids),
                index_prefixes=["entities"],
                filters={"id": ids},
            ),
            "fetch",
        )
    sent = store.started
    await asyncio.sleep(0.05)
    assert store.started == sent <= 4
    assert store.in_flight == 0


def _relationship(source: str, target: str) -> RetrievalResult:
    return RetrievalResult(
        content=f"{source}-{target}",
        score=1.0,
        source=f"{source}-{target}",
        retriever_type="relationship",
        metadata={"source_id": source, "target_id": target},
    )


@pytest.mark.parametrize(
    "fetch", ["local text units", "lightrag endpoints", "lightrag chunks"]
)
async def test_a_fetch_by_id_names_every_ranked_id(config: Config, fetch: str) -> None:
    # Regression: a fetch by id was cut to the ids one search returns (the
    # live window), so a window of 100 kept 100 of 500 ranked text units;
    # then only OpenSearchRetriever batched, so a custom retriever that
    # answers a window kept the window's worth in its own order, not the
    # best-ranked ids. The strategy asks in batches of the window and keeps
    # the rank order.
    store = _Window(window=2)
    ranked = ["e1", "e2", "e3", "e4"]
    if fetch == "local text units":
        strategy = _strategy(config, _Graph(), store)
        found = await strategy._retrieve_documents(ranked, None)
        got = [r.source for r in found["text_units"]]
    elif fetch == "lightrag endpoints":
        strategy = get_strategy_spec(SearchStrategy.MIX).strategy_class(
            config=config, retrievers={RetrieverRole.DOCUMENT.value: store}
        )
        found = await strategy._retrieve_endpoint_entities(
            SearchQuery(query="q"),
            [_relationship("e1", "e2"), _relationship("e3", "e4")],
        )
        got = [r.source for r in found["lightrag_endpoint_entities"]]
    else:
        strategy = get_strategy_spec(SearchStrategy.MIX).strategy_class(
            config=config, retrievers={RetrieverRole.DOCUMENT.value: store}
        )
        cited = _entity("X").model_copy(
            update={"metadata": {"id": "X", "text_unit_ids": ranked}}
        )
        found = await strategy._retrieve_linked_chunks(
            SearchQuery(query="q"), [cited], []
        )
        got = [r.source for r in found["lightrag_linked_chunks"]]
    assert store.asked == [["e1", "e2"], ["e3", "e4"]]
    assert got == ranked


@pytest.mark.parametrize(
    ("scoped", "expected"),
    [
        # Regression: the edges were asked under the caller's id (edge ids
        # that are entity ids: none), so the walk reached no neighbour.
        ({"id": ["S", "D"]}, {"S": "S", "D": "S -> B -> D"}),
        # ... and under its source_id, intersected with the frontier.
        ({"source_id": ["X"]}, {"S": "S", "B": "S -> B", "D": "S -> B -> D"}),
        # ... and under its target_id.
        ({"target_id": ["X"]}, {"S": "S", "B": "S -> B", "D": "S -> B -> D"}),
    ],
)
async def test_the_walk_crosses_edges_without_the_callers_id_keys(
    config: Config, scoped: dict[str, list[str]], expected: dict[str, str]
) -> None:
    # The graph walk applies a caller id to what it reached only; so does
    # the walk over the relationships index, its entity read keeps it.
    index = _Index(config)
    kept = await _expand(config, index, {"attr_grp": "a", **scoped})
    assert _paths(kept) == expected
    edge_calls = [
        f for prefix, f in index.calls if prefix == index.o.relationships_index_prefix
    ]
    assert edge_calls and all("id" not in f for f in edge_calls)
    # The caller's document filter stays on every edge request (the fake
    # store answers the passing edges whatever the filters say).
    assert all(f.get("attr_grp") == "a" for f in edge_calls)
    assert all(
        "X" not in f.get(end, []) and f.get(end) != []
        for f in edge_calls
        for end in ("source_id", "target_id")
    )
    reads = [f for prefix, f in index.calls if prefix == index.o.entities_index_prefix]
    assert reads and all(f.get(k) is not None for f in reads for k in scoped)


async def test_the_walk_stops_once_every_caller_id_is_reached(config: Config) -> None:
    # The entity read keeps only the caller's ids and a vertex's path is
    # fixed once reached: with S and B reached at the first hop, the next
    # hop (from B to D) cannot change the result, so it is not asked.
    index = _Index(config)
    kept = await _expand(config, index, {"attr_grp": "a", "id": ["S", "B"]})
    assert _paths(kept) == {"S": "S", "B": "S -> B"}
    edge_calls = [
        f for prefix, f in index.calls if prefix == index.o.relationships_index_prefix
    ]
    assert len(edge_calls) == 2  # one hop: a request per endpoint
    # An id not reached yet keeps the walk going (D is two hops away).
    index = _Index(config)
    kept = await _expand(config, index, {"attr_grp": "a", "id": ["S", "D"]})
    assert _paths(kept) == {"S": "S", "D": "S -> B -> D"}
    assert sum(p == index.o.relationships_index_prefix for p, _ in index.calls) > 1
