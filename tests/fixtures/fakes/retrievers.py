# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A recording retriever for the search-strategy tests.

It stands in for a ``BaseGraphRAGRetriever`` (duck-typed: the base builds
infrastructure in its ``__init__``), with the base's own methods and call
shapes (``tests/unit/test_fake_port_signatures.py`` pins them). Each call
returns one hit named after the queried index prefix (or the retriever's
tag), so a test can tell which index a strategy searched.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from unified_kg_rag.adapters.storage.filter_schema import (
    IndexLimits,
)
from unified_kg_rag.domain.models import (
    Config,
    GraphEdge,
    RetrievalResult,
    SearchQuery,
    SectionType,
)
from unified_kg_rag.domain.retrieval.filters import (
    ITEM_FILTER_FIELDS,
    FilterFields,
    union_filter_fields,
)


class FakeRetriever:
    """Records the index_prefixes/query of each aretrieve call."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls: list[SearchQuery] = []

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    def filter_fields(self) -> None:
        """No declared schema: every filter key is accepted (the
        ``BaseGraphRAGRetriever`` default)."""
        return None

    def filter_fields_for(self, prefixes: object) -> None:
        return None

    async def ahas_index(self, prefix: str, suffix: str | None) -> bool | None:
        """Cannot tell (the ``BaseGraphRAGRetriever`` default)."""
        return None

    async def arequire_namespace(self, suffix: str | None) -> None:
        """Cannot tell (the ``BaseGraphRAGRetriever`` default)."""
        return None

    async def aread_build_record(self, suffix: str | None) -> dict[str, Any] | None:
        """No index metadata (the ``BaseGraphRAGRetriever`` default)."""
        return None

    async def aneighbourhood_edges(
        self, vertex_ids: list[str], query: SearchQuery, *, per_vertex: int
    ) -> list[GraphEdge] | None:
        """Cannot tell (the ``BaseGraphRAGRetriever`` default)."""
        return None

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


def hit(content: str, source: str = "doc-1", score: float = 0.9) -> RetrievalResult:
    """One document hit, as a document retriever returns it."""
    return RetrievalResult(
        content=content,
        score=score,
        source=source,
        retriever_type=SectionType.TEXT,
        metadata={"id": source, "text_unit_ids": []},
    )


class StaticRetriever:
    """Answers every query with copies of ``results`` (or raises
    ``RuntimeError("backend unavailable")`` with ``fail``); records the
    queries in ``calls``. For the chain tests that need a retriever, not a
    store."""

    def __init__(self, results: list[RetrievalResult], *, fail: bool = False) -> None:
        self.results = results
        self.fail = fail
        self.calls: list[SearchQuery] = []

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        if self.fail:
            raise RuntimeError("backend unavailable")
        return [r.model_copy(deep=True) for r in self.results]


def opensearch_fields_for(config: Config, prefixes: Collection[str]) -> FilterFields:
    """The filter keys the OpenSearch indices of ``prefixes`` declare: what
    ``OpenSearchRetriever.filter_fields_for`` answers, for a fake standing in
    for it."""
    o = config.indexing.opensearch
    by_kind = ITEM_FILTER_FIELDS
    by_prefix = {
        o.text_units_index_prefix: by_kind["text_units"],
        o.entities_index_prefix: by_kind["entities"],
        o.relationships_index_prefix: by_kind["relationships"],
        o.claims_index_prefix: by_kind["claims"],
        o.community_reports_index_prefix: by_kind["community_reports"],
    }
    return union_filter_fields(
        fields for prefix, fields in by_prefix.items() if prefix in prefixes
    )


_ENDS = ("source_id", "target_id")


def incident_ends(query: SearchQuery) -> dict[str, set[str]]:
    """The endpoint fields an incident-edge search names and their ids: the
    ``any_of`` groups' (an edge passes at either end) or else the filters'
    (one end per request), for a fake relationships index."""
    groups = query.any_of or [query.filters or {}]
    asked: dict[str, set[str]] = {}
    for group in groups:
        for end in _ENDS:
            if end in group:
                values = group[end]
                listed = values if isinstance(values, list) else [values]
                asked.setdefault(end, set()).update(str(v) for v in listed)
    return asked
