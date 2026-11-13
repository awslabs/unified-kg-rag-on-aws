# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one text-unit fetch by id (``application.retrieval.text_units``) the
chain's citations, the MCP server and the evaluator's resolver share."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import pytest

from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.application.retrieval.text_units import fetch_text_units
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    SearchQuery,
    SectionType,
)
from unified_kg_rag.domain.retrieval.filters import FilterFields

pytestmark = pytest.mark.unit


class _Store:
    """Text units by id, each with a region; filters on ``attr_region``."""

    def __init__(self, regions: dict[str, str], batch: int = 2) -> None:
        self.regions = regions
        self.batch = batch
        self.queries: list[SearchQuery] = []

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits(max_result_window=self.batch)

    def filter_fields_for(self, prefixes: Collection[str]) -> FilterFields | None:
        return None

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.queries.append(query)
        filters: dict[str, Any] = query.filters or {}
        region = filters.get("attr_region")
        return [
            RetrievalResult(
                content="",
                score=0.0,
                source=i,
                retriever_type=SectionType.TEXT,
                metadata={"id": i, "attributes": {"file_name": f"{i}.txt"}},
            )
            for i in filters.get("id", [])
            if i in self.regions and region in (None, self.regions[i])
        ][: query.top_k]


async def test_batches_dedupes_and_keeps_the_order_asked(config: Config) -> None:
    store = _Store({"a": "n", "b": "n", "c": "s"})
    fetched = await fetch_text_units(store, config, "v1", ["c", "a", "c", "b", "x"])
    assert list(fetched.units) == ["c", "a", "b"]
    assert [q.filters for q in store.queries] == [
        {"id": ["c", "a"]},
        {"id": ["b", "x"]},
    ]
    assert all(q.suffix == "v1" and q.top_k <= 2 for q in store.queries)
    assert fetched.settled == {"a", "b", "c", "x"}


async def test_the_callers_filters_apply(config: Config) -> None:
    store = _Store({"a": "north", "b": "south"})
    fetched = await fetch_text_units(
        store, config, None, ["a", "b"], filters={"attr_region": "north"}
    )
    assert list(fetched.units) == ["a"]


async def test_a_batch_cut_by_a_reindex_is_not_settled(config: Config) -> None:
    store = _Store({"a": "n", "b": "n"})
    fetch = store.aretrieve

    async def shrinking(query: SearchQuery) -> list[RetrievalResult]:
        store.batch = 1
        return (await fetch(query))[:1]

    store.aretrieve = shrinking  # type: ignore[method-assign]
    fetched = await fetch_text_units(store, config, None, ["a", "b"])
    assert list(fetched.units) == ["a"] and fetched.settled == {"a"}


async def test_a_batch_without_hits_is_not_settled(config: Config) -> None:
    store = _Store({})
    fetched = await fetch_text_units(store, config, None, ["a"])
    assert fetched.units == {} and fetched.settled == set()
    assert await fetch_text_units(store, config, None, []) == type(fetched)()


async def test_the_strategies_and_the_unit_fetch_batch_alike(config: Config) -> None:
    # One batched read by id (architecture rv25): a strategy's fetch and the
    # text-unit fetch both go through fetch_by_ids, with the same batches.
    from unified_kg_rag.adapters.retrieval.base import fetch_by_ids
    from unified_kg_rag.adapters.retrieval.strategy_registry import get_strategy_spec
    from unified_kg_rag.domain.models import RetrieverRole, SearchStrategy

    ids = ["c", "a", "c", "b", "x"]
    regions = {"a": "n", "b": "n", "c": "s"}
    units = _Store(regions)
    await fetch_text_units(units, config, None, ids)
    store = _Store(regions)
    strategy: Any = get_strategy_spec(SearchStrategy.NAIVE).strategy_class(
        config=config, retrievers={RetrieverRole.DOCUMENT.value: store}
    )
    prefix = config.indexing.opensearch.text_units_index_prefix
    hits = await strategy._fetch_by_ids(
        store,
        SearchQuery(query="", index_prefixes=[prefix], filters={"id": ids}),
        "fetch",
    )
    assert [h.source for h in hits] == ["c", "a", "b"]
    assert [q.filters for q in store.queries] == [q.filters for q in units.queries]
    # The query's own values of the key narrow the batches.
    store.queries.clear()
    parts = await fetch_by_ids(
        store,
        SearchQuery(query="", index_prefixes=[prefix], filters={"id": ["a", "b"]}),
        ids,
        max_concurrency=2,
    )
    assert [batch for batch, _ in parts] == [["c", "a"], ["b", "x"]]
    assert [q.filters for q in store.queries] == [{"id": ["a"]}, {"id": ["b"]}]
