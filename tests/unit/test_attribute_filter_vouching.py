# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Document filters on graph expansion, whose store has no document attribute.

The relationship and claim indexes carry their documents' filters, so the
store applies them. Graph vertices are kept under an ``attr_<key>`` /
``attributes.filters.<key>`` filter only when one of their text units passes
it (the expansion seeds aside).
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


def _hit(source: str, kind: str, units: Any) -> RetrievalResult:
    return RetrievalResult(
        content=source,
        score=1.0,
        source=source,
        retriever_type=kind,
        metadata={"id": source, "text_unit_ids": units},
    )


class _Store:
    """Text units ``u-ok`` pass the filter; every other search returns ``hits``."""

    def __init__(
        self,
        units_prefix: str,
        hits: list[RetrievalResult],
        fail: bool = False,
        units_suffix: str | None = None,
    ) -> None:
        self.units_prefix = units_prefix
        self.hits = hits
        self.fail = fail
        self.units_suffix = units_suffix  # the suffix whose units are stored
        self.queries: list[SearchQuery] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.queries.append(query)
        if query.index_prefixes == [self.units_prefix]:
            if self.fail:
                raise RuntimeError("store hiccup")
            if query.suffix != self.units_suffix:
                return []
            ids = (query.filters or {}).get("id") or []
            return [_hit(u, "text", []) for u in ids if u == "u-ok"]
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
    return _Store(
        prefix,
        [
            _hit("seed", "entity", ["u-no"]),
            _hit("near", "entity", ["u-no", "u-ok"]),
            _hit("one", "entity", "u-ok"),  # a graph store unwraps one id
            _hit("far", "entity", ["u-no"]),
            _hit("bare", "entity", []),
        ],
    )


@pytest.mark.parametrize("key", ["attr_category", "attributes.filters.category"])
async def test_graph_expansion_keeps_seeds_and_vouched_vertices(
    config: Config, key: str
) -> None:
    prefix = config.indexing.opensearch.text_units_index_prefix
    document = _Store(prefix, [])
    strategy = _strategy(config, document, _graph(prefix))

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={key: "x", "type": "t"}), ["seed"]
    )

    assert [r.source for r in kept] == ["seed", "near", "one"]
    (check,) = document.queries
    assert check.index_prefixes == [prefix]
    # Only the document filter keys plus the cited units, and ids only back.
    assert check.filters == {key: "x", "id": ["u-no", "u-ok"]}
    assert check.source_fields == []


async def test_the_check_searches_the_query_suffix(config: Config) -> None:
    # A query on another tenant/version suffix must check that suffix's text
    # units, not the default one's (which would drop every expanded vertex).
    prefix = config.indexing.opensearch.text_units_index_prefix
    document = _Store(prefix, [], units_suffix="tenant-b")
    strategy = _strategy(config, document, _graph(prefix))

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", suffix="tenant-b", filters={"attr_category": "x"}),
        ["seed"],
    )

    assert [r.source for r in kept] == ["seed", "near", "one"]
    assert [q.suffix for q in document.queries] == ["tenant-b"]


async def test_an_entity_attribute_filter_is_not_vouched(config: Config) -> None:
    # Regression: an entity's own attribute (attributes.role) was checked
    # against the text units, which never carry it, cutting the expansion
    # to the seeds.
    prefix = config.indexing.opensearch.text_units_index_prefix
    document = _Store(prefix, [])
    strategy = _strategy(config, document, _graph(prefix))

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attributes.role": "buyer"}), ["seed"]
    )

    assert len(kept) == 5
    assert document.queries == []


async def test_the_check_batches_and_stops_once_every_vertex_is_vouched(
    config: Config,
) -> None:
    # Regression: one fetch named every cited unit (a hub cites tens of
    # thousands): over max_terms_count the store rejected it and the query
    # failed, and whole units came back for an id check.
    config.indexing.opensearch.index_settings = {"index.max_result_window": 2}
    prefix = config.indexing.opensearch.text_units_index_prefix
    document = _Store(prefix, [])
    graph = _Store(
        prefix,
        [
            _hit("a", "entity", ["u-ok", "u-1", "u-2", "u-3"]),
            _hit("b", "entity", ["u-4", "u-5", "u-6"]),
        ],
    )
    strategy = _strategy(config, document, graph)

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_category": "x"}), ["seed"]
    )

    assert [r.source for r in kept] == ["a"]
    asked = [q.filters["id"] for q in document.queries]
    assert all(len(ids) <= 2 for ids in asked)
    # Each unit is asked once, and none of "a" once "a" is vouched for.
    flat = [u for ids in asked for u in ids]
    assert len(flat) == len(set(flat))
    assert {"u-4", "u-5", "u-6"} <= set(flat)
    assert not {"u-1", "u-2", "u-3", "u-ok"} & set(flat[flat.index("u-ok") + 1 :])


async def test_a_failed_check_drops_what_it_could_not_vouch_for(
    config: Config,
) -> None:
    prefix = config.indexing.opensearch.text_units_index_prefix
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
    prefix = config.indexing.opensearch.text_units_index_prefix
    store = _Store(prefix, [_hit("r-no", "relationship", ["u-no"])])
    strategy = _strategy(config, store, _Store(prefix, []))

    kept = await strategy._safe_aretrieve(
        store, SearchQuery(query="q", filters={"attr_category": "x"}), "test"
    )

    assert [r.source for r in kept] == ["r-no"]
    assert len(store.queries) == 1
