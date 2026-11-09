# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a query answered without: one ledger of its degraded steps.

A store read a strategy absorbed returned ``[]`` silently (simple and naive
reported a clean empty search), and the chain's own ``ignore_errors``
fallbacks (auto's router, the follow-up rewrite, the context budgeting) set
nothing. Every failed step is now one entry of the query's
``DegradationLedger``: ``metadata["degraded"]``, with ``degraded_error`` its
first error type. A findings-lineage read the answer does not cite with is
recorded apart (``citations_unresolved``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401 - registers them
from unified_kg_rag.adapters.retrieval.base import (
    current_ledger,
    degradation_scope,
    query_scope,
    record_degradation,
)
from unified_kg_rag.adapters.retrieval.strategy_registry import get_strategy_spec
from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    ProcessedQuery,
)
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from unified_kg_rag.domain.retrieval.degradation import (
    DegradationLedger,
    DegradationReport,
    DegradationStep,
)
from unified_kg_rag.shared import AWSServiceError

pytestmark = pytest.mark.unit


def test_a_ledger_passes_its_entries_to_the_one_it_opened_in() -> None:
    query = DegradationLedger()
    search = DegradationLedger(parent=query)
    query.record(DegradationStep.TRANSLATION, "searched untranslated", "TimeoutError")
    search.record(DegradationStep.RETRIEVAL, "Entity search", RuntimeError("x"))
    search.record(DegradationStep.ITERATIVE_CONTROLLER, "controller_unparsed")
    assert [e.step for e in search.entries] == ["retrieval", "iterative_controller"]
    assert [e.step for e in query.entries] == [
        "translation",
        "retrieval",
        "iterative_controller",
    ]
    assert search.metadata()["degraded_error"] == "RuntimeError"
    assert query.metadata()["degraded_error"] == "TimeoutError"


def test_the_first_error_type_skips_an_entry_without_one() -> None:
    ledger = DegradationLedger()
    ledger.record(DegradationStep.ITERATIVE_CONTROLLER, "controller_unparsed")
    assert ledger.metadata() == {
        "degraded": [
            {
                "step": "iterative_controller",
                "reason": "controller_unparsed",
                "error_type": None,
            }
        ]
    }
    ledger.record(DegradationStep.GLOBAL_MAP, "1/2 reports unrated", "Throttling")
    assert ledger.metadata()["degraded_error"] == "Throttling"


def test_an_unresolved_lineage_read_degrades_no_answer() -> None:
    ledger = DegradationLedger()
    ledger.record_unresolved(
        DegradationStep.FINDINGS_LINEAGE, "Report findings records", "RuntimeError"
    )
    # Regression (rv8): reported as True only, so no surface could say
    # which read failed; its entries are reported as the degraded ones are.
    assert ledger.metadata() == {
        "citations_unresolved": [
            {
                "step": "findings_lineage",
                "reason": "Report findings records",
                "error_type": "RuntimeError",
            }
        ]
    }
    assert not ledger.entries
    report = DegradationReport.of(ledger.metadata())
    assert report.citations_unresolved == ledger.unresolved
    assert report.degraded == [] and report.degraded_error is None


def test_a_report_reads_back_what_the_ledger_reports() -> None:
    ledger = DegradationLedger()
    ledger.record(DegradationStep.GLOBAL_MAP, "1/2 reports unrated", "Throttling")
    report = DegradationReport.of(ledger.metadata())
    assert report.degraded == ledger.entries
    assert report.degraded_error == "Throttling"
    # A third-party strategy's own list, not entries: left out, not a crash.
    assert DegradationReport.of({"degraded": ["odd", {"step": 1}]}).degraded == []
    assert DegradationReport.of(None) == DegradationReport()


def test_recording_outside_a_scope_is_nothing() -> None:
    assert current_ledger() is None
    record_degradation(DegradationStep.RETRIEVAL, "x", "RuntimeError")
    with query_scope() as memo:
        outer = current_ledger()
        with query_scope() as inner_memo:
            # A scope within the query keeps its memo, and passes entries on.
            assert inner_memo is memo
            record_degradation(DegradationStep.RETRIEVAL, "y", "RuntimeError")
    assert outer is not None and [e.reason for e in outer.entries] == ["y"]


class _Failing:
    """A document retriever whose every read fails (not fatally)."""

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        raise AWSServiceError("Read timed out")


@pytest.mark.parametrize("name", [SearchStrategy.SIMPLE, SearchStrategy.NAIVE])
async def test_an_absorbed_store_read_degrades_the_search(
    name: SearchStrategy,
) -> None:
    # Regression (rv7 unify #2): simple and naive reported a clean empty
    # search, and the chain answered "no information found".
    spec = get_strategy_spec(name)
    strategy = spec.strategy_class(
        config=Config(), retrievers={RetrieverRole.DOCUMENT.value: _Failing()}
    )
    strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda d, *a, **k: [r for v in d.values() for r in v]
    )
    with degradation_scope() as ledger:
        result = await strategy.asearch(
            SearchQuery(query="q", metadata={"search_strategy": name.value})
        )
    assert result.results == []
    assert result.metadata["degraded_error"] == "AWSServiceError"
    assert {e["step"] for e in result.metadata["degraded"]} == {"retrieval"}
    # The query's ledger holds what the strategy's did.
    assert [e.model_dump() for e in ledger.entries] == result.metadata["degraded"]


def _ignoring_chain() -> GraphRAGChain:
    chain = GraphRAGChain.__new__(GraphRAGChain)
    config = Config()
    config.search.auto_routable_strategies = [SearchStrategy.GLOBAL, SearchStrategy.MIX]
    chain.config = config
    chain.ignore_errors = True

    def _failing(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("model down")

    chain._get_chain_for_prompt = _failing  # type: ignore[method-assign]
    return chain


async def test_the_router_fallback_degrades_the_answer() -> None:
    chain = _ignoring_chain()
    with query_scope():
        await chain._resolve_strategy(
            {"search_strategy": SearchStrategy.AUTO, "query": "q"}
        )
        ledger = current_ledger()
        assert ledger is not None
        assert [(e.step, e.error_type) for e in ledger.entries] == [
            ("routing", "RuntimeError")
        ]


async def test_the_follow_up_rewrite_fallback_degrades_the_answer() -> None:
    chain = _ignoring_chain()
    with query_scope():
        assert (
            await chain._rewrite_query_step({"query": "and its price?", "history": "h"})
            is None
        )
        ledger = current_ledger()
        assert ledger is not None
        assert [e.step for e in ledger.entries] == ["follow_up_rewrite"]


def test_a_failed_context_budgeting_degrades_the_answer() -> None:
    chain = _ignoring_chain()

    def _broken(state: dict[str, Any]) -> Any:
        raise ValueError("count failed")

    chain._optimize_context = _broken  # type: ignore[method-assign]
    sr = SearchResult(
        query=SearchQuery(query="q"),
        results=[],
        total_results=0,
        search_strategy="local",
        processing_time=0.0,
    )
    with query_scope():
        chain._context_optimization_step({"search_results": sr})
        state = {
            "search_results": sr,
            "resolved_strategy": SearchStrategy.LOCAL,
            "start_time": 0.0,
            "answer": "a",
            "processed_query": ProcessedQuery(original_query="q", final_query="q"),
        }
        out = GraphRAGChain._format_output_step(state)
    assert out.metadata["degraded"] == [
        {
            "step": "context_optimization",
            "reason": "answered without context",
            "error_type": "ValueError",
        }
    ]
    assert out.metadata["degraded_error"] == "ValueError"


def test_the_console_names_every_step() -> None:
    # Regression (rv9 query L12): the console showed the raw step ids. The
    # English catalog names each framework step (the vitest catalog test
    # holds every locale to the English keys).
    catalog = json.loads(
        (Path(__file__).parents[2] / "console/web/src/i18n/locales/en.json").read_text(
            encoding="utf-8"
        )
    )
    steps = {
        value
        for name, value in vars(DegradationStep).items()
        if not name.startswith("_") and isinstance(value, str)
    }
    assert steps and {f"ask.step.{step}" for step in steps} <= set(catalog)
