# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A recording retriever for the search-strategy tests (``RetrieverPort``).

Each call returns one hit named after the queried index prefix (or the
retriever's tag), so a test can tell which index a strategy searched.
"""

from __future__ import annotations

from unified_kg_rag.domain.models import RetrievalResult, SearchQuery


class FakeRetriever:
    """Records the index_prefixes/query of each aretrieve call."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls: list[SearchQuery] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        prefix = query.index_prefixes[0] if query.index_prefixes else self.tag
        return [
            RetrievalResult(
                content=f"{prefix} result",
                score=1.0,
                source=f"{prefix}-1",
                retriever_type=self.tag,
                metadata={"id": f"{prefix}-id", "text_unit_ids": []},
            )
        ]
