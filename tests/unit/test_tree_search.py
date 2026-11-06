# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free tests of the ``tree`` strategy's graph buckets on a fake
document retriever (the embedded end-to-end runs are in
``tests/integration/test_raptor_e2e.py``)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.search_strategies.tree_search import (
    GRAPH_TEXT_BUCKET,
    TreeSearchStrategy,
)
from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.domain.models import Config, RetrievalResult, SearchQuery

pytestmark = pytest.mark.unit

PASSING = {"unit-3", "unit-4"}


class _Units:
    """Text units by id; under a caller filter only ``PASSING`` match."""

    def __init__(self) -> None:
        self.calls: list[SearchQuery] = []

    def filter_fields_for(self, prefixes: object) -> None:
        return None

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        filters = query.filters or {}
        ids = filters.get("id") or []
        return [
            RetrievalResult(
                content=f"Passage {unit}.",
                score=0.0,
                source=unit,
                retriever_type="text",
                metadata={"id": unit},
            )
            for unit in ids
            if "attr_category" not in filters or unit in PASSING
        ]


def _strategy(units: _Units, config: Config) -> TreeSearchStrategy:
    config.providers.rerank.kind = "none"
    providers = Providers(
        config,
        boto_session=MagicMock(),
        llm_factory=ScriptedLLMFactory(lambda _s, _h: "unused"),
        embedding_factory=MagicMock(),
    )
    return TreeSearchStrategy(
        config, {"document": units}, providers=providers  # type: ignore[dict-item]
    )


async def test_graph_text_units_are_the_best_that_pass_the_filters() -> None:
    config = Config()
    config.search.tree_search.text_top_k = 2
    units = _Units()
    strategy = _strategy(units, config)
    seed = RetrievalResult(
        content="Depot",
        score=1.0,
        source="entity-depot",
        retriever_type="entity",
        metadata={"text_unit_ids": ["unit-1", "unit-2", "unit-3", "unit-4"]},
    )
    query = SearchQuery(query="Where is Depot?", filters={"attr_category": "ops"})

    buckets: dict[str, Any] = await strategy._graph_buckets(
        query, [seed], summaries=False
    )

    # unit-1 and unit-2 rank first but fail the filter: the two slots go to
    # the next ones that pass, not to nothing.
    assert [r.source for r in buckets[GRAPH_TEXT_BUCKET]] == ["unit-3", "unit-4"]
