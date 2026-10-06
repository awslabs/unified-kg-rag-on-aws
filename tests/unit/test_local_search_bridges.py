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


def _edge(rid: str, src: str, tgt: str, rank: float) -> RetrievalResult:
    return RetrievalResult(
        content=f"{src} -> {tgt}",
        score=0.0,
        source=rid,
        retriever_type="relationship",
        metadata={"id": rid, "source_id": src, "target_id": tgt, "rank": rank},
    )


# Entities "vendor" and "buyer" are retrieved; "carrier" and "bank" are not.
_EDGES = {
    "source_id": [
        _edge("r-out", "vendor", "carrier", 9.0),
        _edge("r-bridge", "vendor", "buyer", 1.0),
    ],
    "target_id": [
        _edge("r-bridge-dup", "buyer", "vendor", 1.0),  # same pair, other side
        _edge("r-in", "bank", "buyer", 5.0),
    ],
}


class FakeDocumentRetriever:
    def __init__(self, relationships_prefix: str) -> None:
        self.relationships_prefix = relationships_prefix
        self.calls: list[SearchQuery] = []

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


async def test_bridge_edges_come_first_and_pairs_dedupe(config: Config) -> None:
    strategy, _, _ = _make_strategy(config)

    edges = await strategy._fetch_incident_relationships(
        SearchQuery(query="q"), ["vendor", "buyer"], bridge_first=True
    )

    # Bridge first despite its low rank; then out-of-network by rank. The
    # reversed duplicate of the bridge pair is carried once.
    assert [e.source for e in edges] == ["r-bridge", "r-out", "r-in"]


async def test_without_bridge_first_the_upstream_rank_order_holds(
    config: Config,
) -> None:
    strategy, _, _ = _make_strategy(config)

    edges = await strategy._fetch_incident_relationships(
        SearchQuery(query="q"), ["vendor", "buyer"]
    )

    assert [e.source for e in edges] == ["r-out", "r-in", "r-bridge"]


async def test_local_search_adds_bridge_section_for_expanded_entities(
    config: Config,
) -> None:
    config.indexing.opensearch.build_relationship_vector_index = True
    strategy, doc, captured = _make_strategy(config)

    await strategy.asearch(SearchQuery(query="q", entity_focus=["Vendor"]))

    assert [r.source for r in captured["bridge_relationships"]] == [
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

    # In-network edge first, then the best out-of-network one.
    assert [r.source for r in captured["bridge_relationships"]] == ["r-bridge", "r-out"]
