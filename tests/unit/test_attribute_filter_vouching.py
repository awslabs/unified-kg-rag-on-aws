# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attribute filters on results whose store has no document attribute.

Relationships, claims and graph vertices are kept under an ``attr_<key>`` /
``attributes.<path>`` filter only when one of their text units passes it.
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

    def __init__(self, units_prefix: str, hits: list[RetrievalResult]) -> None:
        self.units_prefix = units_prefix
        self.hits = hits
        self.queries: list[SearchQuery] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.queries.append(query)
        if query.index_prefixes == [self.units_prefix]:
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


@pytest.mark.parametrize("key", ["attr_category", "attributes.filters.category"])
async def test_relationships_and_claims_need_a_passing_text_unit(
    config: Config, key: str
) -> None:
    prefix = config.indexing.opensearch.text_units_index_prefix
    store = _Store(
        prefix,
        [
            _hit("r-ok", "relationship", ["u-no", "u-ok"]),
            _hit("r-no", "relationship", ["u-no"]),
            _hit("c-ok", "claim", "u-ok"),  # a graph store unwraps one id
            _hit("c-none", "claim", []),
            _hit("e-1", "entity", ["u-no"]),  # filtered by its own index
        ],
    )
    strategy = _strategy(config, store, _Store(prefix, []))

    kept = await strategy._safe_aretrieve(
        store, SearchQuery(query="q", filters={key: "x", "type": "t"}), "test"
    )

    assert [r.source for r in kept] == ["r-ok", "c-ok", "e-1"]
    check = store.queries[-1]
    assert check.index_prefixes == [prefix]
    # Only the attribute keys, plus the cited units.
    assert check.filters == {key: "x", "id": ["u-no", "u-ok"]}


async def test_graph_expansion_keeps_seeds_and_vouched_vertices(
    config: Config,
) -> None:
    prefix = config.indexing.opensearch.text_units_index_prefix
    graph = _Store(
        prefix,
        [
            _hit("seed", "entity", ["u-no"]),
            _hit("near", "entity", ["u-ok"]),
            _hit("far", "entity", ["u-no"]),
        ],
    )
    strategy = _strategy(config, _Store(prefix, []), graph)

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_category": "x"}), ["seed"]
    )

    assert [r.source for r in kept] == ["seed", "near"]


async def test_no_attribute_filter_checks_nothing(config: Config) -> None:
    prefix = config.indexing.opensearch.text_units_index_prefix
    store = _Store(prefix, [_hit("r-no", "relationship", ["u-no"])])
    strategy = _strategy(config, store, _Store(prefix, []))

    kept = await strategy._safe_aretrieve(
        store, SearchQuery(query="q", filters={"type": "t"}), "test"
    )

    assert [r.source for r in kept] == ["r-no"]
    assert len(store.queries) == 1
