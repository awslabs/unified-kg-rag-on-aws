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
    # Two batches of two: four ids, then three passed, so no third batch.
    assert passed == {"e0", "e1", "e2", "e3"}
    assert [ids for _, ids in document.calls] == [["e0", "e1"], ["e2", "e3"]]
