# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entity-linked chunks under a filter on text units fill their quota.

Local and LightRAG mix rank the chunks their entities cite and fetch them by
id under the caller's filters. Cutting the ranked ids to the quota first
spent the quota on ids the filter then dropped (#370 then fetched every
ranked chunk's content to cut afterwards); an ids-only check now finds the
first chunks that pass, and only those are fetched with their content.
"""

from __future__ import annotations

from typing import Any

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.domain.retrieval.strategy_registry import get_strategy_spec

pytestmark = pytest.mark.unit

# Ranked by how many entities cite them: u0 first. The filter passes odd ids.
_UNITS = [f"u{i}" for i in range(12)]


def _entity(eid: str, units: list[str]) -> RetrievalResult:
    return RetrievalResult(
        content=eid,
        score=1.0,
        source=eid,
        retriever_type="entity",
        metadata={"id": eid, "text_unit_ids": units},
    )


class _Store:
    def __init__(self, units_prefix: str, entities: list[RetrievalResult]) -> None:
        self.units_prefix = units_prefix
        self.entities = entities
        self.fetches: list[list[str]] = []  # content fetches
        self.checks: list[list[str]] = []  # ids-only checks

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        filters = query.filters or {}
        if query.index_prefixes == [self.units_prefix] and "id" in filters:
            ids = list(filters["id"])
            (self.checks if query.source_fields == [] else self.fetches).append(ids)
            passing = [
                i
                for i in ids
                if ("attr_grp" not in filters and "n_tokens" not in filters)
                or int(i[1:]) % 2 == 1
            ]
            return [
                RetrievalResult(content=i, score=0.0, source=i, retriever_type="text")
                for i in reversed(passing)  # index order, not rank order
            ]
        if query.index_prefixes and "entities" in str(query.index_prefixes):
            return [e.model_copy() for e in self.entities]
        return []


def _entities() -> list[RetrievalResult]:
    # u_i is cited by 12 - i entities, so the rank is u0, u1, ..., u11.
    return [_entity(f"e{k}", _UNITS[: 12 - k]) for k in range(12)]


_FILTERS = {
    "none": None,
    "document": {"attr_grp": "a"},
    "text-unit": {"n_tokens": {"gte": 1}},
}


@pytest.mark.parametrize("kind", list(_FILTERS))
async def test_local_fills_the_text_quota_with_chunks_that_pass(
    config: Config, kind: str
) -> None:
    filtered = _FILTERS[kind] is not None
    config.search.local_search.type_quota.text_multiplier = 0.0
    config.search.local_search.type_quota.text_floor = 4
    config.search.local_search.entity_frequency_threshold = 100
    units_prefix = config.indexing.opensearch.text_units_index_prefix
    store = _Store(units_prefix, _entities())
    strategy = get_strategy_spec(SearchStrategy.LOCAL).strategy_class(
        config=config, retrievers={RetrieverRole.DOCUMENT.value: store}
    )
    sections, stats = await strategy._retrieve_entity_chain(
        SearchQuery(query="q", entity_focus=["e0"], filters=_FILTERS[kind])
    )

    chunks = [r.source for r in sections["text_units"]]
    assert chunks == (["u1", "u3", "u5", "u7"] if filtered else _UNITS[:4])
    assert stats["text_unit_count"] == 4
    # Content is fetched for the quota only; a filter adds one ids-only check.
    assert store.fetches == [chunks if filtered else _UNITS[:4]]
    assert store.checks == ([_UNITS] if filtered else [])
    assert strategy._per_type_quota(10)[SectionType.TEXT.value] == 4


@pytest.mark.parametrize("kind", list(_FILTERS))
async def test_mix_fills_the_linked_chunk_stream_with_chunks_that_pass(
    config: Config, kind: str
) -> None:
    filtered = _FILTERS[kind] is not None
    config.search.lightrag_search.chunk_stream_top_k = 3
    # A per-hit cap smaller than the lineage: the slots must go to chunks that
    # pass, not to the first ones cited.
    config.search.lightrag_search.related_chunk_number = 3
    units_prefix = config.indexing.opensearch.text_units_index_prefix
    store = _Store(units_prefix, [])
    strategy: Any = get_strategy_spec(SearchStrategy.MIX).strategy_class(
        config=config, retrievers={RetrieverRole.DOCUMENT.value: store}
    )
    linked = await strategy._retrieve_linked_chunks(
        SearchQuery(query="q", top_k=3, filters=_FILTERS[kind]),
        [_entity("e0", _UNITS)],
        [],
    )

    chunks = [r.source for r in linked["lightrag_linked_chunks"]]
    assert chunks == (["u1", "u3", "u5"] if filtered else _UNITS[:3])
    assert store.fetches == [chunks]
    # The hit's lineage is checked until its three slots pass: the rest of a
    # long lineage is never asked.
    assert store.checks == (
        [["u0", "u1", "u2"], ["u3", "u4"], ["u5"]] if filtered else []
    )


async def test_mix_checks_every_hits_lineage_not_only_the_first(
    config: Config,
) -> None:
    # Regression: the lineage check stopped once the stream's worth of chunks
    # passed, all of them the first (hub) hit's, and excluded every other
    # hit's chunks unchecked: 5 chunks over 11 hits instead of 20 over 38.
    config.search.lightrag_search.chunk_stream_top_k = 4
    config.search.lightrag_search.related_chunk_number = 2
    units_prefix = config.indexing.opensearch.text_units_index_prefix
    store = _Store(units_prefix, [])
    strategy: Any = get_strategy_spec(SearchStrategy.MIX).strategy_class(
        config=config, retrievers={RetrieverRole.DOCUMENT.value: store}
    )
    hub = _entity("hub", [f"u{i}" for i in range(1, 40, 2)])  # 20 passing
    others = [_entity("x", ["u41", "u43"]), _entity("y", ["u45", "u47"])]

    linked = await strategy._retrieve_linked_chunks(
        SearchQuery(query="q", top_k=4, filters={"attr_grp": "a"}),
        [hub, *others],
        [],
    )

    chunks = [r.source for r in linked["lightrag_linked_chunks"]]
    assert {"u41", "u45"} & set(chunks), chunks
    assert len(chunks) == 4
