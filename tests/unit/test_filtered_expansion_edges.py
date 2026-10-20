# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph expansion under a document filter follows passing edges only.

The walk crosses every edge, so an edge only excluded documents state
brought in entities (and their chunks) that an index of the passing
documents never reaches. Under a document filter an expanded entity is kept
only when a walk over edges the relationships index passes reaches it from
a seed. Also: the per-query memo of filter checks and probes, and the
filter values the retriever refuses.
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


def _vertex(vid: str, *walk: str) -> RetrievalResult:
    path = " -> ".join([*walk, vid]) if walk else vid
    return RetrievalResult(
        content=f"Entity: {vid}\nDescription: d\nPath: {path}",
        score=1.0,
        source=vid,
        retriever_type="entity",
        metadata={"id": vid, "name": vid, "text_unit_ids": [f"u-{vid}"]},
    )


# The walk: S - A (excluded edge), S - B, A - D (the walk reached D through
# A), B - D, A - C. Only S-B, B-D and A-C pass.
_EDGES = [
    ("r-sa", "S", "A"),
    ("r-sb", "S", "B"),
    ("r-ad", "A", "D"),
    ("r-bd", "B", "D"),
    ("r-ac", "A", "C"),
]
_PASSING = {"r-sb", "r-bd", "r-ac"}


class _Graph:
    def __init__(self, edges: list | None = _EDGES) -> None:
        self.edges = edges
        self.asked: list[list[str]] = []

    async def aedges_among(self, ids: list[str], suffix: str | None) -> Any:
        self.asked.append(ids)
        return self.edges

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        return [
            _vertex("S"),
            _vertex("A", "S"),
            _vertex("B", "S"),
            _vertex("D", "S", "A"),
            _vertex("C", "S", "A"),
        ]


class _Document:
    """Every entity passes the entities-index check; relationships pass when
    in ``_PASSING``; ``calls`` counts the checks by index."""

    def __init__(self, config: Config) -> None:
        self.o = config.indexing.opensearch
        self.calls: list[tuple[str, list[str]]] = []

    async def ahas_index(self, prefix: str, suffix: str | None) -> bool:
        return True

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


def _strategy(config: Config, graph: Any, document: Any) -> Any:
    return get_strategy_spec(SearchStrategy.LOCAL).strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: document,
            RetrieverRole.GRAPH.value: graph,
        },
    )


async def test_only_entities_a_passing_walk_reaches_are_kept(config: Config) -> None:
    graph, document = _Graph(), _Document(config)
    strategy = _strategy(config, graph, document)

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_grp": "a"}), ["S"]
    )

    # A only over an excluded edge, C only through A: dropped. D was walked
    # through A but a passing walk reaches it through B: kept, path redrawn.
    by_id = {r.source: r for r in kept}
    assert list(by_id) == ["S", "B", "D"]
    assert by_id["D"].content.endswith("Path: S -> B -> D")
    assert by_id["B"].content.endswith("Path: S -> B")
    rel_checks = [
        ids
        for prefix, ids in document.calls
        if prefix == config.indexing.opensearch.relationships_index_prefix
    ]
    assert rel_checks == [["r-sa", "r-sb", "r-ad", "r-bd", "r-ac"]]


async def test_hops_bound_the_passing_walk(config: Config) -> None:
    config.indexing.neptune.max_hops = 1
    strategy = _strategy(config, _Graph(), _Document(config))
    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_grp": "a"}), ["S"]
    )
    assert [r.source for r in kept] == ["S", "B"]


async def test_without_a_document_filter_nothing_is_asked(config: Config) -> None:
    graph, document = _Graph(), _Document(config)
    strategy = _strategy(config, graph, document)
    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"type": "org"}), ["S"]
    )
    assert len(kept) == 5
    assert graph.asked == [] and document.calls == []


async def test_a_graph_that_cannot_tell_keeps_the_entity_check(config: Config) -> None:
    strategy = _strategy(config, _Graph(edges=None), _Document(config))
    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_grp": "a"}), ["S"]
    )
    assert len(kept) == 5


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


async def test_without_a_relationship_index_the_entity_check_decides(
    config: Config,
) -> None:
    # Regression: with build_relationship_vector_index off there is no index
    # to check the walked edges in; none passed and only the seeds were kept.
    config.indexing.opensearch.build_relationship_vector_index = False
    graph, document = _Graph(), _Document(config)
    strategy = _strategy(config, graph, document)
    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_grp": "a"}), ["S"]
    )
    assert len(kept) == 5
    assert graph.asked == []
    assert all(
        prefix != config.indexing.opensearch.relationships_index_prefix
        for prefix, _ in document.calls
    )


class _MissingRelationshipIndex(_Document):
    """The relationships index is gone at query time: it reads as no hits."""

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        prefix = (query.index_prefixes or [""])[0]
        if prefix == self.o.relationships_index_prefix:
            self.calls.append((prefix, list((query.filters or {}).get("id") or [])))
            return []
        return await super().aretrieve(query)

    async def ahas_index(self, prefix: str, suffix: str | None) -> bool:
        return prefix != self.o.relationships_index_prefix


async def test_a_relationship_index_missing_at_query_time_is_noticed(
    config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    strategy = _strategy(config, _Graph(), _MissingRelationshipIndex(config))
    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_grp": "a"}), ["S"]
    )
    assert len(kept) == 5
    assert "no relationships index" in caplog.text


async def test_an_index_that_exists_but_passes_no_edge_still_drops(
    config: Config,
) -> None:
    class _NothingPasses(_MissingRelationshipIndex):
        async def ahas_index(self, prefix: str, suffix: str | None) -> bool:
            return True

    strategy = _strategy(config, _Graph(), _NothingPasses(config))
    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_grp": "a"}), ["S"]
    )
    assert [r.source for r in kept] == ["S"]


async def test_a_memoized_outcome_does_not_displace_a_higher_ranked_id(
    config: Config,
) -> None:
    # Regression: under ``enough`` the ids a query already knew passing were
    # counted wherever they ranked, so with u9 and u8 known the check
    # returned at once and u0, u1 were never asked: two lower-ranked chunks
    # took their slots.
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


async def test_an_intermediate_vertex_a_vertex_filter_left_out_still_connects(
    config: Config,
) -> None:
    # Regression: under {grp: a, type: person} the walk S -> X -> D returned
    # S and D only (X is an organization): the edges among the results had
    # no S-D edge, so D was dropped (4 entities against 6 in the subset).
    class _Typed(_Graph):
        async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
            seed, far = _vertex("S"), _vertex("D", "S", "X")
            far.metadata["_path"] = [["S", "S"], ["X", "Xavier"], ["D", "D"]]
            return [seed, far]

    edges = [("r-sx", "S", "X"), ("r-xd", "X", "D")]
    graph = _Typed(edges=edges)
    document = _Document(config)
    strategy = _strategy(config, graph, document)
    _PASSING.update({"r-sx", "r-xd"})
    try:
        kept = await strategy._expand_via_graph(
            SearchQuery(query="q", filters={"attr_grp": "a", "type": "person"}), ["S"]
        )
    finally:
        _PASSING.difference_update({"r-sx", "r-xd"})
    assert [r.source for r in kept] == ["S", "D"]
    assert kept[1].content.endswith("Path: S -> Xavier -> D")
    assert sorted(graph.asked[0]) == ["D", "S", "X"]


async def test_filtered_expansion_walks_wider_and_cuts_after_the_checks(
    config: Config,
) -> None:
    # Regression: the walk's per-seed budget and its cut ran before the
    # checks, so excluded entities used up the slots (2-4 kept against 12).
    asked: list[int] = []

    class _Wide(_Graph):
        async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
            asked.append(query.retrieval_multiplier)
            out = [_vertex("S")]
            for i in range(30):
                v = _vertex(f"v{i}", "S")
                v.score = 0.5 - i / 1000
                out.append(v)
            return out

    edges = [(f"r{i}", "S", f"v{i}") for i in range(30)]
    _PASSING.update(f"r{i}" for i in range(1, 30, 2))
    try:
        strategy = _strategy(config, _Wide(edges=edges), _Document(config))
        kept = await strategy._expand_via_graph(
            SearchQuery(query="q", top_k=4, filters={"attr_grp": "a"}), ["S"]
        )
    finally:
        _PASSING.difference_update(f"r{i}" for i in range(1, 30, 2))
    assert asked == [4]  # the walk ran four times wider
    # The seed plus the four best that passed, not the four best walked.
    assert [r.source for r in kept] == ["S", "v1", "v3", "v5", "v7"]


async def test_two_keys_need_one_document_passing_an_edge(config: Config) -> None:
    # The relationships index unions an edge's documents' values per key:
    # under two keys an edge passes there that no one document passes.
    o = config.indexing.opensearch

    class _Joint(_Document):
        async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
            prefix = (query.index_prefixes or [""])[0]
            ids = list((query.filters or {}).get("id") or [])
            if prefix == o.relationships_index_prefix and query.source_fields:
                self.calls.append((prefix, ids))
                return [
                    RetrievalResult(
                        content="",
                        score=0.0,
                        source=i,
                        retriever_type="relationship",
                        metadata={"text_unit_ids": [f"u-{i}"]},
                    )
                    for i in ids
                ]
            if prefix == o.text_units_index_prefix:
                self.calls.append((prefix, ids))
                return [
                    RetrievalResult(
                        content="", score=0.0, source=i, retriever_type="text"
                    )
                    for i in ids
                    # Every entity's own unit; of the edges' units, S-B's only.
                    if i == "u-r-sb" or not i.startswith("u-r-")
                ]
            return await super().aretrieve(query)

    strategy = _strategy(config, _Graph(), _Joint(config))
    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_grp": "a", "attr_category": "x"}), ["S"]
    )
    # r-bd and r-ac pass the index but no one document: only S-B is walked.
    assert [r.source for r in kept] == ["S", "B"]


async def test_a_redrawn_path_moves_the_score_to_its_distance_band(
    config: Config,
) -> None:
    class _Scored(_Graph):
        async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
            out = await super().aretrieve(query)
            for r in out:
                # D was walked S -> A -> D: a two-hop band (1/4, 1/3].
                hops = len(r.content.rsplit("Path: ", 1)[1].split(" -> "))
                r.metadata["_path"] = [[v, v] for v in range(hops)]
                r.score = 1 / (hops + 1) + 0.75 * (1 / hops - 1 / (hops + 1))
            return out

    strategy = _strategy(config, _Scored(), _Document(config))
    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_grp": "a"}), ["S"]
    )
    d = next(r for r in kept if r.source == "D")
    # Redrawn S -> B -> D: still two hops, same band and place in it.
    assert d.score == pytest.approx(1 / 4 + 0.75 * (1 / 3 - 1 / 4))


async def test_a_failed_edge_read_keeps_the_seeds_only(config: Config) -> None:
    class _Failing(_Graph):
        async def aedges_among(self, ids: list[str], suffix: str | None) -> Any:
            raise RuntimeError("graph read failed")

    strategy = _strategy(config, _Failing(), _Document(config))
    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_grp": "a"}), ["S"]
    )
    assert [r.source for r in kept] == ["S"]


async def test_an_edge_stored_towards_the_seed_is_walked(config: Config) -> None:
    # Edge direction is the extractor's: S-B stored as B -> S still connects.
    edges = [
        ("r-sa", "S", "A"),
        ("r-sb", "B", "S"),
        ("r-ad", "A", "D"),
        ("r-bd", "D", "B"),
        ("r-ac", "A", "C"),
    ]
    strategy = _strategy(config, _Graph(edges), _Document(config))
    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_grp": "a"}), ["S"]
    )
    assert [r.source for r in kept] == ["S", "B", "D"]


async def test_the_redrawn_path_does_not_name_the_excluded_walk(
    config: Config,
) -> None:
    strategy = _strategy(config, _Graph(), _Document(config))
    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_grp": "a"}), ["S"]
    )
    d = {r.source: r for r in kept}["D"]
    assert d.content.count("Path: ") == 1
    assert "A" not in d.content.split("Path: ")[1]


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
