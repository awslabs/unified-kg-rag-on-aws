# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for text-unit -> file attribution (AWS-free)."""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.models import Config, RetrievalResult, SearchQuery
from unified_kg_rag.evaluation.source_resolver import (
    TextUnitFileResolver,
    file_name_of,
)

pytestmark = pytest.mark.unit


class _Store:
    """Fake document retriever: answers id-batch fetches from a dict."""

    def __init__(self, files: dict[str, str]) -> None:
        self.files = files
        self.queries: list[SearchQuery] = []

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
    resolver.BATCH_SIZE = 2

    assert await resolver.aresolve(["t1", "t2", "t3"], "v2") == {
        "t1": "vendor.pdf",
        "t2": "buyer.txt",
    }
    assert [q.filters for q in store.queries] == [{"id": ["t1", "t2"]}, {"id": ["t3"]}]
    query = store.queries[0]
    assert query.suffix == "v2"
    assert query.index_prefixes == [config.indexing.opensearch.text_units_index_prefix]

    # Known and unknown ids are cached per suffix: no second lookup.
    assert await resolver.aresolve(["t1", "t3"], "v2") == {"t1": "vendor.pdf"}
    assert len(store.queries) == 2
    # A different suffix is a different index.
    await resolver.aresolve(["t1"], None)
    assert len(store.queries) == 3


def test_file_name_of() -> None:
    assert file_name_of({"attributes": {"file_path": "/in/a/Terms.pdf"}}) == "Terms.pdf"
    assert file_name_of({"file_name": " x.md "}) == "x.md"
    assert file_name_of({"file_name": ""}) is None
