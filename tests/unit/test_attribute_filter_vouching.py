# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Document filters on graph expansion, whose store has no document attribute.

The relationship and claim indexes carry their documents' filters, so the
store applies them. Graph vertices are kept under an ``attr_<key>`` /
``attributes.filters.<key>`` filter only when their entities-index document,
which carries the filters of every document the entity comes from, passes it
(the expansion seeds aside).
"""

from __future__ import annotations

from typing import Any

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.domain.retrieval.strategy_registry import get_strategy_spec

pytestmark = pytest.mark.unit


def _hit(source: str, kind: str = "entity") -> RetrievalResult:
    return RetrievalResult(
        content=source,
        score=1.0,
        source=source,
        retriever_type=kind,
        metadata={"id": source},
    )


class _Store:
    """The entities of ``passing`` pass the filter (in ``entities_suffix``);
    every other search returns ``hits``."""

    def __init__(
        self,
        entities_prefix: str,
        hits: list[RetrievalResult],
        passing: frozenset[str] = frozenset({"near", "one"}),
        fail: bool = False,
        entities_suffix: str | None = None,
    ) -> None:
        self.entities_prefix = entities_prefix
        self.hits = hits
        self.passing = passing
        self.fail = fail
        self.entities_suffix = entities_suffix
        self.queries: list[SearchQuery] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.queries.append(query)
        if query.index_prefixes == [self.entities_prefix] and query.query == "":
            if self.fail:
                raise RuntimeError("store hiccup")
            if query.suffix != self.entities_suffix:
                return []
            ids = (query.filters or {}).get("id") or []
            return [_hit(i) for i in ids if i in self.passing]
        return [h.model_copy() for h in self.hits]


def _strategy(config: Config, document: _Store, graph: _Store) -> Any:
    return get_strategy_spec(SearchStrategy.LOCAL).strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: document,
            RetrieverRole.GRAPH.value: graph,
        },
    )


def _graph(prefix: str) -> _Store:
    return _Store(prefix, [_hit(i) for i in ("seed", "near", "one", "far")])


@pytest.mark.parametrize("key", ["attr_category", "attributes.filters.category"])
async def test_graph_expansion_keeps_seeds_and_vouched_vertices(
    config: Config, key: str
) -> None:
    prefix = config.indexing.opensearch.entities_index_prefix
    document = _Store(prefix, [])
    strategy = _strategy(config, document, _graph(prefix))

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={key: "x", "type": "t"}), ["seed"]
    )

    assert [r.source for r in kept] == ["seed", "near", "one"]
    (check,) = document.queries
    assert check.index_prefixes == [prefix]
    # One request for the non-seed vertices, document filter keys only, and
    # ids only back.
    assert check.filters == {key: "x", "id": ["near", "one", "far"]}
    assert check.source_fields == []


async def test_the_check_searches_the_query_suffix(config: Config) -> None:
    # A query on another tenant/version suffix must check that suffix's
    # entities, not the default one's (which would drop every vertex).
    prefix = config.indexing.opensearch.entities_index_prefix
    document = _Store(prefix, [], entities_suffix="tenant-b")
    strategy = _strategy(config, document, _graph(prefix))

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", suffix="tenant-b", filters={"attr_category": "x"}),
        ["seed"],
    )

    assert [r.source for r in kept] == ["seed", "near", "one"]
    assert [q.suffix for q in document.queries] == ["tenant-b"]


async def test_an_entity_attribute_filter_is_not_vouched(config: Config) -> None:
    # An entity's own attribute (attributes.role) is not a document filter.
    prefix = config.indexing.opensearch.entities_index_prefix
    document = _Store(prefix, [])
    strategy = _strategy(config, document, _graph(prefix))

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attributes.role": "buyer"}), ["seed"]
    )

    assert len(kept) == 4
    assert document.queries == []


async def test_the_check_batches_under_one_requests_limits(config: Config) -> None:
    # One fetch must name no more ids than a terms clause and a result
    # window hold, or the store rejects it and the query fails.
    config.indexing.opensearch.index_settings = {"index.max_result_window": 2}
    prefix = config.indexing.opensearch.entities_index_prefix
    document = _Store(prefix, [], passing=frozenset({"v3"}))
    graph = _Store(prefix, [_hit(f"v{i}") for i in range(5)])
    strategy = _strategy(config, document, graph)

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_category": "x"}), ["v0"]
    )

    assert [r.source for r in kept] == ["v0", "v3"]
    assert [q.filters["id"] for q in document.queries] == [["v1", "v2"], ["v3", "v4"]]


async def test_a_failed_check_drops_what_it_could_not_vouch_for(
    config: Config,
) -> None:
    prefix = config.indexing.opensearch.entities_index_prefix
    strategy = _strategy(config, _Store(prefix, [], fail=True), _graph(prefix))

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_category": "x"}), ["seed"]
    )

    assert [r.source for r in kept] == ["seed"]


async def test_document_store_results_are_filtered_by_the_store(
    config: Config,
) -> None:
    # Relationship and claim indexes apply document filters server-side:
    # nothing post-filters them (an extra fetch per sub-search).
    prefix = config.indexing.opensearch.entities_index_prefix
    store = _Store(prefix, [_hit("r-no", "relationship")])
    strategy = _strategy(config, store, _Store(prefix, []))

    kept = await strategy._safe_aretrieve(
        store, SearchQuery(query="q", filters={"attr_category": "x"}), "test"
    )

    assert [r.source for r in kept] == ["r-no"]
    assert len(store.queries) == 1
