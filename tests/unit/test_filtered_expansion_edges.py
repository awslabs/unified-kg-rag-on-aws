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
