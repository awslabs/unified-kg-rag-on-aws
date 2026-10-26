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
from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
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

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

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
    # The hit's first three chunks, then (short of three passing) the rest of
    # its lineage: two requests, not one per round of a few ids.
    assert store.checks == ([_UNITS[:3], _UNITS[3:]] if filtered else [])


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


async def test_a_selective_filter_on_a_hub_lineage_costs_two_checks(
    config: Config,
) -> None:
    # Regression: each round asked a few more ids per short hit, so a filter
    # one hub chunk in 500 passes made a request per round: mix went from 1
    # request to 539 on 6,000 documents.
    config.search.lightrag_search.chunk_stream_top_k = 4
    config.search.lightrag_search.related_chunk_number = 5
    units_prefix = config.indexing.opensearch.text_units_index_prefix
    store = _Store(units_prefix, [])
    strategy: Any = get_strategy_spec(SearchStrategy.MIX).strategy_class(
        config=config, retrievers={RetrieverRole.DOCUMENT.value: store}
    )
    # Only u499 is odd-numbered and so passes; the rest are even.
    hub = _entity("hub", [f"u{i}" for i in range(0, 498, 2)] + ["u499"])
    linked = await strategy._retrieve_linked_chunks(
        SearchQuery(query="q", top_k=4, filters={"attr_grp": "a"}),
        [hub, _entity("x", ["u2000", "u2001"])],
        [],
    )
    assert {r.source for r in linked["lightrag_linked_chunks"]} == {"u499", "u2001"}
    assert len(store.checks) == 2


async def test_a_memoized_lineage_keeps_its_passing_chunks(config: Config) -> None:
    # Within a query (DRIFT's follow-ups) the second lineage check finds
    # every unit's outcome memoized and asks nothing: the units counted from
    # the memo must still count as checked, or the hit keeps none of them.
    from unified_kg_rag.adapters.retrieval.base import query_scope

    config.search.lightrag_search.chunk_stream_top_k = 3
    config.search.lightrag_search.related_chunk_number = 3
    store = _Store(config.indexing.opensearch.text_units_index_prefix, [])
    strategy: Any = get_strategy_spec(SearchStrategy.MIX).strategy_class(
        config=config, retrievers={RetrieverRole.DOCUMENT.value: store}
    )
    query = SearchQuery(query="q", top_k=3, filters={"attr_grp": "a"})
    with query_scope():
        first = await strategy._retrieve_linked_chunks(
            query, [_entity("e0", _UNITS)], []
        )
        second = await strategy._retrieve_linked_chunks(
            query, [_entity("e0", _UNITS)], []
        )
    expected = ["u1", "u3", "u5"]
    assert [r.source for r in first["lightrag_linked_chunks"]] == expected
    assert [r.source for r in second.get("lightrag_linked_chunks", [])] == expected


@pytest.mark.parametrize("failing_round", [1, 2])
async def test_a_failed_lineage_check_excludes_what_it_asked(
    config: Config, failing_round: int
) -> None:
    # Fail-closed: the chunks a failed check asked about count as failing
    # and take no slot. The slots go to the next chunks of the lineage,
    # fetched under the filter (which the fetch applies too).
    config.search.lightrag_search.chunk_stream_top_k = 3
    config.search.lightrag_search.related_chunk_number = 3

    class _Failing(_Store):
        async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
            if query.source_fields == [] and len(self.checks) + 1 == failing_round:
                self.checks.append(list((query.filters or {})["id"]))
                raise RuntimeError("store hiccup")
            return await super().aretrieve(query)

    store = _Failing(config.indexing.opensearch.text_units_index_prefix, [])
    strategy: Any = get_strategy_spec(SearchStrategy.MIX).strategy_class(
        config=config, retrievers={RetrieverRole.DOCUMENT.value: store}
    )
    query = SearchQuery(query="q", top_k=3, filters={"attr_grp": "a"})
    sections = await strategy._retrieve_linked_chunks(
        query, [_entity("e0", _UNITS)], []
    )
    # Round 1 asks u0-u2 (u1 passes, the hit is short); round 2 the rest.
    assert store.checks == [_UNITS[:3], _UNITS[3:]][:failing_round]
    if failing_round == 1:
        assert store.fetches == [["u3", "u4", "u5"]]
    else:
        assert store.fetches == [["u1"]]
        assert [r.source for r in sections["lightrag_linked_chunks"]] == ["u1"]


@pytest.mark.parametrize("case", ["dedup", "redistribute"])
async def test_mix_fills_the_slots_a_subset_index_fills(case: str) -> None:
    # Regression: the check asked each hit's first slots, but the selection
    # keeps a chunk on its first citing hit only and polling hands a short
    # hit's slots to the others, so slots landed on unchecked chunks the
    # filtered fetch then dropped. Selected and checked again until every
    # selected chunk is known, the filter fills what an index of the passing
    # (odd) chunks alone fills.
    odd = [f"u{i}" for i in (1, 3, 5, 7, 9)]
    even = [f"u{i}" for i in (0, 2, 4, 6, 8)]
    late = [f"u{i}" for i in (11, 13, 15)]
    if case == "dedup":  # e1's first five chunks are e0's
        hits = [_entity("e0", odd), _entity("e1", odd + even + late)]
    else:  # e1 and e2 cite one chunk each: their slots go to e0
        hits = [_entity("e0", odd + even + late), _entity("e1", ["u17"])]
        hits.append(_entity("e2", ["u19"]))

    async def linked(hits: list[RetrievalResult], filters: Any) -> list[str]:
        config = Config()
        config.search.lightrag_search.chunk_stream_top_k = 20
        config.search.lightrag_search.related_chunk_number = 5
        store = _Store(config.indexing.opensearch.text_units_index_prefix, [])
        strategy: Any = get_strategy_spec(SearchStrategy.MIX).strategy_class(
            config=config, retrievers={RetrieverRole.DOCUMENT.value: store}
        )
        out = await strategy._retrieve_linked_chunks(
            SearchQuery(query="q", top_k=10, filters=filters), hits, []
        )
        return sorted(r.source for r in out.get("lightrag_linked_chunks", []))

    subset = [
        _entity(
            str(h.source),
            [u for u in h.metadata["text_unit_ids"] if int(u[1:]) % 2 == 1],
        )
        for h in hits
    ]
    assert await linked(hits, {"attr_grp": "a"}) == await linked(subset, None)


async def test_slots_on_unchecked_chunks_cost_a_request_a_round(config: Config) -> None:
    # Regression: a slot moved past the checked chunks (e1's first five are
    # e0's, so its slots go to the rest) was checked chunk by chunk: 400
    # failing chunks cost about 80 rounds. Each round asks twice as far
    # past the deepest unchecked chunk as the one before.
    config.search.lightrag_search.chunk_stream_top_k = 20
    config.search.lightrag_search.related_chunk_number = 5
    store = _Store(config.indexing.opensearch.text_units_index_prefix, [])
    strategy: Any = get_strategy_spec(SearchStrategy.MIX).strategy_class(
        config=config, retrievers={RetrieverRole.DOCUMENT.value: store}
    )
    shared = [f"u{i}" for i in (1, 3, 5, 7, 9)]
    failing = [f"u{i}" for i in range(10, 810, 2)]  # 400 even ids
    hits = [_entity("e0", shared), _entity("e1", [*shared, *failing, "u811"])]
    out = await strategy._retrieve_linked_chunks(
        SearchQuery(query="q", top_k=10, filters={"attr_grp": "a"}), hits, []
    )
    assert sorted(r.source for r in out["lightrag_linked_chunks"]) == sorted(
        [*shared, "u811"]
    )
    assert len(store.checks) <= 10
    assert sum(len(ids) for ids in store.checks) <= 2 * (len(shared) + 401)


async def test_a_slot_on_an_unchecked_chunk_checks_a_hub_lineage_only_near_it(
    config: Config,
) -> None:
    # A slot that moved onto one unchecked chunk had the whole unchecked rest
    # of the hit's lineage checked: a hub's 1000 more chunks for one slot.
    config.search.lightrag_search.chunk_stream_top_k = 20
    config.search.lightrag_search.related_chunk_number = 5
    store = _Store(config.indexing.opensearch.text_units_index_prefix, [])
    strategy: Any = get_strategy_spec(SearchStrategy.MIX).strategy_class(
        config=config, retrievers={RetrieverRole.DOCUMENT.value: store}
    )
    shared = [f"u{i}" for i in (1, 3, 5, 7, 9)]
    hub = [*shared, "u10", *(f"u{i}" for i in range(11, 2011, 2))]
    hits = [_entity("e0", shared), _entity("e1", hub)]
    out = await strategy._retrieve_linked_chunks(
        SearchQuery(query="q", top_k=10, filters={"attr_grp": "a"}), hits, []
    )
    assert sorted(r.source for r in out["lightrag_linked_chunks"]) == sorted(
        [*shared, "u11"]  # e1's slot: u10 fails
    )
    assert sum(len(ids) for ids in store.checks) < 30, store.checks


async def test_a_filter_passing_the_unfiltered_selection_costs_one_check(
    config: Config,
) -> None:
    # The first round asks the unfiltered selection with each hit's first
    # chunks: when the filter passes it, no second round asks the rest of a
    # lineage whose first chunks another hit holds.
    config.search.lightrag_search.chunk_stream_top_k = 20
    config.search.lightrag_search.related_chunk_number = 5
    store = _Store(config.indexing.opensearch.text_units_index_prefix, [])
    strategy: Any = get_strategy_spec(SearchStrategy.MIX).strategy_class(
        config=config, retrievers={RetrieverRole.DOCUMENT.value: store}
    )
    shared = [f"u{i}" for i in (1, 3, 5, 7, 9)]
    hits = [
        _entity("e0", shared),
        _entity("e1", [*shared, *(f"u{i}" for i in range(11, 211, 2))]),
    ]
    await strategy._retrieve_linked_chunks(
        SearchQuery(query="q", top_k=10, filters={"attr_grp": "a"}), hits, []
    )
    assert len(store.checks) == 1, store.checks
