# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``iterative``'s follow-up round on every backend with retrievers.

The OpenSearch retriever over the embedded SQLite engine, the OpenSearch
container and Gremlin Server (``LOCAL_STORES=1``) and the AWS stores
(``-m aws``); see ``tests/fixtures/store_backends.py``. A three-hop chain is
written straight into the stores: a one-hop walk from the first company
reaches the carrier but not the depot, so round 0 lacks the passage that
places the depot, and the round seeded with the depot finds it, under a
document filter that keeps a decoy passage about the same depot out of
every round. The controller is a scripted model; there is no reranker.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.store_backends import (
    RETRIEVERS,
    StoreHarness,
    backends,
    open_harness,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.search_strategies.iterative_search import (
    IterativeSearchStrategy,
)
from unified_kg_rag.domain.models import (
    Entity,
    Relationship,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    TextUnit,
)

pytestmark = pytest.mark.integration

QUESTION = "Where is the storage site of the firm Vendor hires?"
FOLLOW_UP = "Where is Depot located?"
GOLD = "Depot is located in Harbor."
DECOY = "Depot is located in Marsh."


@pytest.fixture(params=backends(RETRIEVERS))
def harness(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StoreHarness]:
    yield from open_harness(request.param, tmp_path)


def _unit(unit_id: str, text: str, category: str) -> TextUnit:
    return TextUnit(
        id=unit_id, text=text, attributes={"filters": {"category": category}}
    )


def _entity(name: str, units: list[str], categories: list[str]) -> Entity:
    return Entity(
        id=f"e-{name.lower()}",
        name=name,
        type="ORG",
        description=f"{name} in the supply chain",
        text_unit_ids=units,
        rank=1,
        attributes={"filters": {"category": categories}},
    )


def _edge(source: str, target: str, unit: str, category: str) -> Relationship:
    return Relationship(
        id=f"r-{source.lower()}-{target.lower()}",
        source_id=f"e-{source.lower()}",
        target_id=f"e-{target.lower()}",
        source_name=source,
        target_name=target,
        description=f"{source} works with {target}",
        text_unit_ids=[unit],
        attributes={"filters": {"category": category}},
    )


def _ingest(h: StoreHarness) -> None:
    h.initialize()
    units = [
        _unit("t-orders", "Vendor hires Carrier for freight.", "ops"),
        _unit("t-freight", "Carrier rents Depot for storage.", "ops"),
        _unit("t-harbor", GOLD, "ops"),
        _unit("t-decoy", DECOY, "archive"),
    ]
    entities = [
        _entity("Vendor", ["t-orders"], ["ops"]),
        _entity("Carrier", ["t-orders", "t-freight"], ["ops"]),
        _entity("Depot", ["t-freight", "t-harbor", "t-decoy"], ["ops", "archive"]),
        _entity("Harbor", ["t-harbor"], ["ops"]),
        _entity("Marsh", ["t-decoy"], ["archive"]),
    ]
    edges = [
        _edge("Vendor", "Carrier", "t-orders", "ops"),
        _edge("Carrier", "Depot", "t-freight", "ops"),
        _edge("Depot", "Harbor", "t-harbor", "ops"),
        _edge("Depot", "Marsh", "t-decoy", "archive"),
    ]
    vectors = h.vector_indexer
    for stats in (
        vectors.index_text_units(units),
        vectors.index_entities(entities),
        vectors.index_relationships(edges),
        h.graph_indexer.index_entities(entities),
        h.graph_indexer.index_relationships(edges),
    ):
        assert stats.failed_items == 0, stats.errors


def _controller(system: str, human: str) -> str:
    assert "You control the retrieval" in system, system[:60]
    found = GOLD in human
    return json.dumps(
        {
            "sufficient": found,
            "known_facts": [],
            "bridge_entities": [] if found else ["Depot"],
            "next_queries": [] if found else [FOLLOW_UP],
        }
    )


class _Tokens:
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def truncate_to_token_limit(self, text: str, max_tokens: int) -> tuple[str, int]:
        words = text.split()[:max_tokens]
        return " ".join(words), len(words)


def _search(h: StoreHarness, filters: dict[str, Any] | None) -> SearchResult:
    config = h.config.model_copy(deep=True)
    config.search.iterative_search.base = "local"
    config.providers.rerank.kind = "none"
    providers = Providers(
        config,
        llm_factory=ScriptedLLMFactory(_controller),
        embedding_factory=h.embedding_factory,
        token_counter_factory=lambda *_, **__: _Tokens(),
    )

    async def run() -> SearchResult:
        assert h.graph_retriever is not None and h.vector_retriever is not None
        retrievers = {
            RetrieverRole.GRAPH.value: h.graph_retriever(),
            RetrieverRole.DOCUMENT.value: h.vector_retriever(),
        }
        try:
            strategy = IterativeSearchStrategy(config, retrievers, providers=providers)
            return await strategy.asearch(
                SearchQuery(
                    query=QUESTION,
                    entity_focus=["Vendor"],
                    filters=filters,
                    metadata={"query_language": "English"},
                )
            )
        finally:
            for retriever in retrievers.values():
                await retriever.aclose()

    return asyncio.run(run())


def _texts(result: SearchResult) -> list[str]:
    return [r.content for r in result.results if r.retriever_type == "text"]


@pytest.mark.parametrize(
    "filters", [None, {"attr_category": "ops"}], ids=["unfiltered", "filtered"]
)
def test_the_seeded_round_finds_the_last_hop_on_every_backend(
    harness: StoreHarness, filters: dict[str, Any] | None
) -> None:
    # One hop from the first company reaches the carrier, not the depot.
    h = harness.derive(lambda c: setattr(c.indexing.neptune, "max_hops", 1))
    _ingest(h)

    result = _search(h, filters)

    meta = result.metadata["iterative"]
    assert (meta["rounds"], meta["stop_reason"]) == (1, "sufficient")
    # Round 0 (the first verdict) lacked the passage; round 1 found it.
    assert [v["sufficient"] for v in meta["verdicts"]] == [False, True]
    texts = _texts(result)
    assert GOLD in texts
    (gold,) = [r for r in result.results if r.content == GOLD]
    assert gold.source == "t-harbor"
    if filters:
        assert not any(DECOY in r.content for r in result.results)
