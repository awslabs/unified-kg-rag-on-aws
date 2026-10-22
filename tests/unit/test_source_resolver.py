# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for text-unit -> file attribution (AWS-free)."""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.domain.models import Config, RetrievalResult, SearchQuery
from unified_kg_rag.evaluation.source_resolver import (
    TextUnitFileResolver,
    file_name_of,
)

pytestmark = pytest.mark.unit


class _Store:
    """Fake document retriever: answers id-batch fetches from a dict."""

    def __init__(self, files: dict[str, str], batch: int = 2) -> None:
        self.files = files
        self.batch = batch
        self.queries: list[SearchQuery] = []

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits(max_result_window=self.batch)

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.queries.append(query)
        ids = (query.filters or {}).get("id", [])
        return [
            RetrievalResult(
                content="",
                score=0.0,
                source=i,
                retriever_type="text",
                metadata={"id": i, "attributes": {"file_name": self.files[i]}},
            )
            for i in ids
            if i in self.files
        ]


async def test_resolves_in_batches_and_caches(config: Config) -> None:
    store = _Store({"t1": "vendor.pdf", "t2": "buyer.txt"})
    resolver = TextUnitFileResolver(config, lambda: store)

    assert await resolver.aresolve(["t1", "t3", "t2"], "v2") == {
        "t1": "vendor.pdf",
        "t2": "buyer.txt",
    }
    assert [q.filters for q in store.queries] == [{"id": ["t1", "t3"]}, {"id": ["t2"]}]
    query = store.queries[0]
    assert query.suffix == "v2"
    assert query.index_prefixes == [config.indexing.opensearch.text_units_index_prefix]

    # Known and unknown ids (of a batch that answered) are cached per
    # suffix: no second lookup.
    assert await resolver.aresolve(["t1", "t3"], "v2") == {"t1": "vendor.pdf"}
    assert len(store.queries) == 2
    # A different suffix is a different index.
    await resolver.aresolve(["t1"], None)
    assert len(store.queries) == 3


async def test_batches_follow_the_live_limits(config: Config) -> None:
    # Batches were a fixed 500 ids: under a live max_result_window of 200 a
    # search returned 200 of them, and the other 300 were cached as unknown.
    store = _Store({f"t{i}": f"f{i % 7}.txt" for i in range(600)}, batch=200)
    resolver = TextUnitFileResolver(config, lambda: store)

    assert len(await resolver.aresolve(list(store.files), None)) == 600
    assert [len(q.filters["id"]) for q in store.queries] == [200, 200, 200]

    # Limits lowered by a reindex between the batching and the fetch: the
    # misses of the cut batch are asked again next time, not cached.
    fetch = store.aretrieve

    async def shrinking(query: SearchQuery) -> list[RetrievalResult]:
        store.batch = 1
        return (await fetch(query))[:1]

    store.aretrieve = shrinking  # type: ignore[method-assign]
    other = TextUnitFileResolver(config, lambda: store)
    assert await other.aresolve(["t1", "t2"], None) == {"t1": "f1.txt"}
    store.aretrieve = fetch  # type: ignore[method-assign]
    assert await other.aresolve(["t1", "t2"], None) == {
        "t1": "f1.txt",
        "t2": "f2.txt",
    }


async def test_a_batch_without_any_hit_is_not_cached(config: Config) -> None:
    # Regression: the retriever degrades a failed request to no hits, and
    # every id of the batch was cached as unknown for the evaluator's life.
    store = _Store({"t1": "vendor.pdf"})
    fetch = store.aretrieve
    calls = 0

    async def failing_once(query: SearchQuery) -> list[RetrievalResult]:
        nonlocal calls
        calls += 1
        return [] if calls == 1 else await fetch(query)

    store.aretrieve = failing_once  # type: ignore[method-assign]
    resolver = TextUnitFileResolver(config, lambda: store)
    assert await resolver.aresolve(["t1"], None) == {}
    assert await resolver.aresolve(["t1"], None) == {"t1": "vendor.pdf"}


def test_file_name_of() -> None:
    assert file_name_of({"attributes": {"file_path": "/in/a/Terms.pdf"}}) == "Terms.pdf"
    assert file_name_of({"file_name": " x.md "}) == "x.md"
    assert file_name_of({"file_name": ""}) is None
