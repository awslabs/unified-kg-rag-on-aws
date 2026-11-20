# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a query's trace and metrics say of its evidence and spend: bounded
scores and ids on the retrieval and fusion spans, the cited ids, query id
and USD on the query span, ``metadata["cost_usd"]``, the token and cost
query metrics, and the package version on the trace resource (G-3..G-5)."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from tests.unit.test_observability import (
    _SIMPLE_QUERY,
    Tracing,
    _attrs,
    _by_name,
    _FakeModelFactory,
    _one,
    _RecordingSink,
    _simple_chain,
)
from tests.unit.test_query_audit import _MODEL, _counted
from unified_kg_rag.application.observability import configure_observability
from unified_kg_rag.domain.models.config import ModelPrice, ObservabilityConfig
from unified_kg_rag.domain.pricing import snapshot_cost_usd
from unified_kg_rag.shared import metrics, observability

pytestmark = pytest.mark.unit

_PRICE = ModelPrice(input_per_million_tokens=1.0, output_per_million_tokens=2.0)


@pytest.fixture
def tracing() -> Iterator[Tracing]:
    """Tracing into an in-memory exporter (test_observability's)."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    def enable() -> InMemorySpanExporter:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        configure_observability(
            ObservabilityConfig(enabled=True), tracer_provider=provider
        )
        return exporter

    yield enable
    observability.uninstall()


class _CitingFactory(_FakeModelFactory):
    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        return GenericFakeChatModel(
            messages=itertools.repeat(AIMessage(content="Vendor ships 120 units [1]."))
        )


async def test_the_spans_carry_scores_cited_ids_the_query_id_and_cost(
    tracing: Tracing,
) -> None:
    exporter = tracing()
    chain = _simple_chain(_counted(_CitingFactory()))
    chain.config.models.prices = {_MODEL: _PRICE}
    try:
        output = await chain.ainvoke(_SIMPLE_QUERY)
    finally:
        await chain.aclose()
    query = _attrs(_one(exporter, "kg_rag.query"))
    # G-4: the query span names the id its log lines and result carry.
    assert query["kg_rag.query_id"] == output.metadata["query_id"]
    # G-5: and what the query cost.
    assert output.metadata["cost_usd"] > 0
    assert output.metadata["unpriced_models"] == []
    assert query["kg_rag.cost_usd"] == output.metadata["cost_usd"]
    # G-3: the ids of the sections the answer cites.
    cited = [c.source_id for c in output.citations if c.cited]
    assert cited and list(query["kg_rag.cited_source_ids"]) == cited
    retrieve, *_ = _by_name(exporter)["kg_rag.retrieve"]
    assert list(_attrs(retrieve)["kg_rag.source_ids"]) == ["doc-0", "doc-1"]
    assert list(_attrs(retrieve)["kg_rag.scores"]) == [0.9, 0.9]
    fusion = _attrs(_one(exporter, "kg_rag.fusion"))
    fused = output.search_results.results
    assert list(fusion["kg_rag.source_ids"]) == [r.source for r in fused]
    assert list(fusion["kg_rag.scores"]) == [r.score for r in fused]


async def test_an_unpriced_query_has_no_cost_and_names_the_model(
    tracing: Tracing,
) -> None:
    exporter = tracing()
    chain = _simple_chain(_counted(_FakeModelFactory()))
    sink = _RecordingSink()
    chain.providers.metrics_sink = sink
    try:
        output = await chain.ainvoke(_SIMPLE_QUERY)
    finally:
        await chain.aclose()
    assert output.metadata["cost_usd"] is None
    assert output.metadata["unpriced_models"] == [_MODEL]
    assert "kg_rag.cost_usd" not in _attrs(_one(exporter, "kg_rag.query"))
    ((_, values, _),) = sink.emitted
    # Its tokens are counted; no cost series without a price.
    assert values[metrics.QUERY_INPUT_TOKENS_METRIC] > 0
    assert values[metrics.QUERY_OUTPUT_TOKENS_METRIC] > 0
    assert metrics.QUERY_COST_METRIC not in values


async def test_a_priced_query_publishes_its_tokens_and_cost() -> None:
    chain = _simple_chain(_counted(_FakeModelFactory()))
    chain.config.models.prices = {_MODEL: _PRICE}
    sink = _RecordingSink()
    chain.providers.metrics_sink = sink
    try:
        output = await chain.ainvoke(_SIMPLE_QUERY)
    finally:
        await chain.aclose()
    ((_, values, _),) = sink.emitted
    usage = output.metadata["usage"]
    assert values[metrics.QUERY_INPUT_TOKENS_METRIC] == usage["input_tokens"]
    assert values[metrics.QUERY_OUTPUT_TOKENS_METRIC] == usage["output_tokens"]
    assert values[metrics.QUERY_COST_METRIC] == pytest.approx(
        output.metadata["cost_usd"]
    )
    assert values[metrics.QUERY_COST_METRIC] == pytest.approx(
        (usage["input_tokens"] * 1.0 + usage["output_tokens"] * 2.0) / 1e6
    )


def test_a_snapshot_without_usage_has_no_cost() -> None:
    assert snapshot_cost_usd(None, {}) == (None, [])
    assert snapshot_cost_usd({"llm_calls": "many"}, {}) == (None, [])
    # No model called: nothing to price.
    assert snapshot_cost_usd({"llm_calls": 0}, {}) == (0.0, [])
