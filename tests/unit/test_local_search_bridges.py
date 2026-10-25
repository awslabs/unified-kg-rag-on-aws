# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local search's bridge-relationship section (in-network edges first).

The relationship vector query only finds edges whose description resembles the
question; the hop edges of a multi-hop chain rarely do. Local search therefore
also fetches the edges incident to its graph-expanded entities, putting the
edges whose both endpoints were retrieved first. Fake retrievers stand in for
OpenSearch / Neptune and fusion is stubbed to capture its input buckets.
"""

from __future__ import annotations

import asyncio

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401
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


def _edge(
    rid: str, src: str, tgt: str, rank: float, rel_type: str = "related"
) -> RetrievalResult:
    return RetrievalResult(
        content=f"{src} -{rel_type}-> {tgt}",
        score=0.0,
        source=rid,
        retriever_type="relationship",
        metadata={
            "id": rid,
            "source_id": src,
            "target_id": tgt,
            "rank": rank,
            "type": rel_type,
        },
    )


# Entities "vendor" and "buyer" are retrieved; "carrier" and "bank" are not.
_EDGES = {
    "source_id": [
        _edge("r-out", "vendor", "carrier", 9.0),
        _edge("r-bridge", "vendor", "buyer", 1.0),
    ],
    "target_id": [
        _edge("r-bridge", "vendor", "buyer", 1.0),  # same edge, other endpoint
        # Another typed edge between the same pair: a distinct relationship.
        _edge("r-bridge-typed", "buyer", "vendor", 2.0, "invoices"),
        _edge("r-in", "bank", "buyer", 5.0),
    ],
}


class FakeDocumentRetriever:
    def __init__(self, relationships_prefix: str) -> None:
        self.relationships_prefix = relationships_prefix
        self.calls: list[SearchQuery] = []

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        filters = query.filters or {}
        if query.index_prefixes == [self.relationships_prefix]:
            for field in ("source_id", "target_id"):
                if field in filters:
                    return [r.model_copy() for r in _EDGES[field]]
        prefix = query.index_prefixes[0] if query.index_prefixes else "doc"
        return [
            RetrievalResult(
                content=f"{prefix} hit",
                score=1.0,
                source="vendor",
                retriever_type="entity",
                metadata={"id": "vendor", "text_unit_ids": []},
            )
        ]


class FakeGraphRetriever:
    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                content=f"Entity: {eid}",
                score=1.0,
                source=eid,
                retriever_type="entity",
                metadata={"id": eid, "text_unit_ids": []},
            )
            for eid in ("vendor", "buyer")
        ]


def _make_strategy(config: Config):
    spec = get_strategy_spec(SearchStrategy.LOCAL)
    doc = FakeDocumentRetriever(config.indexing.opensearch.relationships_index_prefix)
    strategy = spec.strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: doc,
            RetrieverRole.GRAPH.value: FakeGraphRetriever(),
        },
    )
    captured: dict[str, list[RetrievalResult]] = {}

    def _fuse(results_dict, top_k, retrieval_multiplier=1, query=None, **_kw):
        captured.update(results_dict)
        return [r for results in results_dict.values() for r in results]

    strategy.hybrid_scorer.fuse_and_rerank_results = _fuse  # type: ignore[method-assign]
    return strategy, doc, captured


async def test_bridge_edges_come_first_and_edges_dedupe_by_id(
    config: Config,
) -> None:
    strategy, _, _ = _make_strategy(config)

    edges = await strategy._fetch_incident_relationships(
        SearchQuery(query="q"), ["vendor", "buyer"], bridge_first=True
    )

    # Bridges first despite their low rank; then out-of-network by rank. The
    # bridge reached from both endpoints is carried once, and the other typed
    # edge between the same pair is kept (an endpoint-pair key dropped it).
    assert [e.source for e in edges] == [
        "r-bridge-typed",
        "r-bridge",
        "r-out",
        "r-in",
    ]


async def test_an_edge_the_caller_holds_is_not_fetched_again(
    config: Config,
) -> None:
    strategy, _, _ = _make_strategy(config)

    edges = await strategy._fetch_incident_relationships(
        SearchQuery(query="q"),
        ["vendor", "buyer"],
        known_relationships=[_edge("r-bridge", "vendor", "buyer", 1.0)],
    )

    assert [e.source for e in edges] == ["r-out", "r-in", "r-bridge-typed"]


async def test_without_bridge_first_the_upstream_rank_order_holds(
    config: Config,
) -> None:
    strategy, _, _ = _make_strategy(config)

    edges = await strategy._fetch_incident_relationships(
        SearchQuery(query="q"), ["vendor", "buyer"]
    )

    assert [e.source for e in edges] == ["r-out", "r-in", "r-bridge-typed", "r-bridge"]


async def test_local_search_adds_bridge_section_for_expanded_entities(
    config: Config,
) -> None:
    config.indexing.opensearch.build_relationship_vector_index = True
    strategy, doc, captured = _make_strategy(config)

    await strategy.asearch(SearchQuery(query="q", entity_focus=["Vendor"]))

    assert [r.source for r in captured["bridge_relationships"]] == [
        "r-bridge-typed",
        "r-bridge",
        "r-out",
        "r-in",
    ]
    bridge_calls = [
        q
        for q in doc.calls
        if q.filters and ({"source_id", "target_id"} & set(q.filters))
    ]
    assert len(bridge_calls) == 2
    for call in bridge_calls:
        (ids,) = [v for k, v in call.filters.items() if k in ("source_id", "target_id")]
        assert set(ids) == {"vendor", "buyer"}


async def test_a_failed_bridge_lookup_cancels_the_text_unit_lookup(
    config: Config,
) -> None:
    # The two lookups after the expansion run together: a fatal error of one
    # (a denied store) left the other's store requests running.
    from unified_kg_rag.shared.exceptions import StoreAccessDeniedError

    strategy, _, _ = _make_strategy(config)
    cancelled: list[str] = []

    async def denied(*_args: object) -> list[RetrievalResult]:
        await asyncio.sleep(0.01)
        raise StoreAccessDeniedError("denied")

    async def slow(*_args: object) -> list[str]:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append("text units")
            raise
        return []

    strategy._retrieve_bridge_relationships = denied  # type: ignore[method-assign]
    strategy._passing_ranked_units = slow  # type: ignore[method-assign]
    with pytest.raises(StoreAccessDeniedError):
        await asyncio.wait_for(
            strategy._retrieve_entity_chain(
                SearchQuery(query="q", entity_focus=["Vendor"])
            ),
            2,
        )
    assert cancelled == ["text units"]


@pytest.mark.parametrize(
    ("flag", "vector_index"),
    [(False, True), (True, False)],
)
async def test_bridge_section_off_by_flag_or_missing_index(
    config: Config, flag: bool, vector_index: bool
) -> None:
    config.search.local_search.include_bridge_relationships = flag
    config.indexing.opensearch.build_relationship_vector_index = vector_index
    strategy, doc, captured = _make_strategy(config)

    await strategy.asearch(SearchQuery(query="q", entity_focus=["Vendor"]))

    assert "bridge_relationships" not in captured
    assert not [
        q
        for q in doc.calls
        if q.filters and ({"source_id", "target_id"} & set(q.filters))
    ]


async def test_bridge_stream_is_capped_at_the_relationship_quota(
    config: Config,
) -> None:
    config.indexing.opensearch.build_relationship_vector_index = True
    config.search.local_search.type_quota.relationship_multiplier = 0.1
    config.search.local_search.type_quota.relationship_floor = 2
    strategy, _, captured = _make_strategy(config)

    await strategy.asearch(SearchQuery(query="q", entity_focus=["Vendor"], top_k=5))

    # In-network edges first, up to the quota.
    assert [r.source for r in captured["bridge_relationships"]] == [
        "r-bridge-typed",
        "r-bridge",
    ]
