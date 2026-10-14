# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-query rerank status and ``search.reranking.required`` (#83, AWS-free).

Regression: a rerank model that could not be built, a rerank call that
failed and one that returned nothing usable all fell back to the fused order
with a log line only, so a deployment could run on fusion alone without
anyone noticing (the per-result ``reranked`` flag cannot tell disabled from
failed). Every query now reports ``metadata["rerank"]`` (applied, disabled,
unavailable, skipped or failed, per stage plus a worst-case summary), and
``required: true`` fails the query instead of degrading it.

Rerank models come from fake factories through the ``rerank_factory`` seam;
nothing mocks boto3.
"""

from __future__ import annotations

import asyncio
import io
import itertools
import logging
import time
from typing import Any

import pytest
from botocore.exceptions import ClientError
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from rich.console import Console

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from tests.fixtures.fakes.chat_models import fake_model_info
from unified_kg_rag.adapters.local.rerank import NoRerankFactory
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.retrieval.hybrid_scorer import HybridScorer, RerankTrace
from unified_kg_rag.adapters.search_strategies import SimpleSearchStrategy
from unified_kg_rag.application.cli import run_rag_chain
from unified_kg_rag.application.cli.run_rag_chain import RAGChainRunner
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    ProcessedQuery,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationQuery,
    EvaluatorType,
    RerankStage,
    RerankStatus,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from unified_kg_rag.evaluation import EvaluationManager
from unified_kg_rag.shared import RerankModelError

pytestmark = pytest.mark.unit


class _FakeReranker:
    """Reverses the documents; raises or returns nothing on request.

    The behaviour is picked per query text, so concurrent queries on one
    shared model can have different outcomes.
    """

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.top_n: int | None = None

    def compress_documents(self, documents: Any, query: str) -> list[Document]:
        mode = "boom" if "boom" in query else "empty" if "empty" in query else self.mode
        if "slow" in query:
            time.sleep(0.1)
        if mode == "boom":
            raise RuntimeError("rerank backend down")
        if mode == "empty":
            return []
        return [
            Document(
                page_content=d.page_content,
                metadata={**d.metadata, "relevance_score": 1.0 - i / 10},
            )
            for i, d in enumerate(reversed(list(documents)))
        ]


class _FakeRerankFactory:
    kind = "fake"

    def __init__(self, model: Any = None, *, raises: Exception | None = None) -> None:
        self.model = model
        self.raises = raises

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        if self.raises is not None:
            raise self.raises
        return self.model

    def get_model_info(self, model_id: Any) -> None:
        return None


def _config(*, enabled: bool = True, required: bool = False) -> Config:
    config = Config()
    config.search.reranking.enabled = enabled
    config.search.reranking.required = required
    config.search.fusion.diversity_lambda = 1.0
    return config


def _scorer(factory: Any = None, **config_kwargs: Any) -> HybridScorer:
    return HybridScorer(_config(**config_kwargs), rerank_factory=factory)


def _results(*types: str) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            content=f"passage {i}",
            score=1.0 - i / 10,
            source=f"doc-{i}",
            retriever_type=rtype,
        )
        for i, rtype in enumerate(types)
    ]


def _fuse(
    scorer: HybridScorer,
    results: list[RetrievalResult],
    query: str | None = "q",
    **kwargs: Any,
) -> tuple[list[RetrievalResult], dict[str, Any]]:
    trace = scorer.new_trace()
    out = scorer.fuse_and_rerank_results(
        {"a": results}, top_k=10, query=query, rerank_trace=trace, **kwargs
    )
    return out, trace.summary()


# --- one fusion call -------------------------------------------------------------


def test_applied_reorders_and_reports_applied() -> None:
    out, rerank = _fuse(
        _scorer(_FakeRerankFactory(_FakeReranker())), _results("text", "text")
    )

    assert [r.source for r in out] == ["doc-1", "doc-0"]
    assert all(r.metadata["reranked"] for r in out)  # the per-result flag stays
    assert rerank == {
        "status": "applied",
        "reason": None,
        "stages": [{"status": "applied", "scope": "all", "reason": None}],
    }


def test_disabled_in_config() -> None:
    factory = _FakeRerankFactory(_FakeReranker())
    _, rerank = _fuse(_scorer(factory, enabled=False), _results("text"))
    assert rerank["status"] == "disabled"
    assert rerank["reason"] == "search.reranking.enabled is false"


def test_provider_none_is_disabled_even_when_required() -> None:
    config = _config(required=True)
    scorer = HybridScorer(config, rerank_factory=NoRerankFactory(config))
    _, rerank = _fuse(scorer, _results("text"))
    assert rerank["status"] == "disabled"
    assert rerank["reason"] == "providers.rerank.kind is none"


def test_init_failure_is_unavailable_and_logged_at_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    factory = _FakeRerankFactory(raises=ValueError("model not in this region"))
    with caplog.at_level(logging.ERROR):
        scorer = _scorer(factory)
    assert any(
        r.levelno == logging.ERROR
        and "Reranking initialization failed" in r.getMessage()
        for r in caplog.records
    )

    out, rerank = _fuse(scorer, _results("text", "text"))
    assert [r.source for r in out] == ["doc-0", "doc-1"]  # fused order kept
    assert rerank["status"] == "unavailable"
    assert rerank["reason"] == "rerank model initialization failed: ValueError"


def test_a_provider_that_builds_no_model_is_unavailable() -> None:
    _, rerank = _fuse(_scorer(_FakeRerankFactory(None)), _results("text"))
    assert rerank["status"] == "unavailable"


def test_runtime_failure_keeps_the_fused_order_and_reports_failed() -> None:
    scorer = _scorer(_FakeRerankFactory(_FakeReranker("boom")))
    out, rerank = _fuse(scorer, _results("text", "text"))
    assert [r.source for r in out] == ["doc-0", "doc-1"]
    assert not any((r.metadata or {}).get("reranked") for r in out)
    assert rerank["status"] == "failed"
    assert rerank["reason"] == "rerank call failed: RuntimeError"


def test_empty_rerank_output_is_failed() -> None:
    scorer = _scorer(_FakeRerankFactory(_FakeReranker("empty")))
    out, rerank = _fuse(scorer, _results("text", "text"))
    assert [r.source for r in out] == ["doc-0", "doc-1"]
    assert rerank["status"] == "failed"
    assert rerank["reason"] == "rerank returned no usable results"


@pytest.mark.parametrize(
    ("results", "query", "kwargs", "reason", "scope"),
    [
        ([], "q", {}, "no candidates", "all"),
        (_results("text"), None, {}, "no query text", "all"),
        (_results("text"), "", {}, "no query text", "all"),
        (
            _results("entity", "relationship"),
            "q",
            {"rerank_only_types": {"text"}},
            "no rerank-eligible results",
            "text",
        ),
    ],
)
def test_skipped_paths(
    results: list[RetrievalResult],
    query: str | None,
    kwargs: dict[str, Any],
    reason: str,
    scope: str,
) -> None:
    model = _FakeReranker("boom")  # would fail if it were called
    _, rerank = _fuse(_scorer(_FakeRerankFactory(model)), results, query, **kwargs)
    assert rerank["status"] == "skipped"
    assert rerank["reason"] == reason
    assert rerank["stages"][0]["scope"] == scope


def test_a_query_that_never_reached_fusion_is_skipped() -> None:
    # An early return (no candidates retrieved) records no stage.
    rerank = _scorer(_FakeRerankFactory(_FakeReranker())).new_trace().summary()
    assert rerank["status"] == "skipped"
    assert rerank["reason"] == "no candidates"
    # ...unless reranking is off or unavailable, which says more.
    assert _scorer(enabled=False).new_trace().summary()["status"] == "disabled"
    assert _scorer(_FakeRerankFactory(None)).new_trace().summary()["status"] == (
        "unavailable"
    )


_ARN = "arn:aws:sts::123456789012:assumed-role/Example/session"


def _client_error(code: str) -> Exception:
    return ClientError(
        {"Error": {"Code": code, "Message": f"denied for {_ARN}"}}, "Rerank"
    )


@pytest.mark.parametrize(
    "code", ["AccessDeniedException", "ValidationException", "ResourceNotFound"]
)
def test_a_client_error_reports_its_error_code_not_its_message(code: str) -> None:
    class _ClientErrorReranker(_FakeReranker):
        def compress_documents(self, documents: Any, query: str) -> list[Document]:
            # The Bedrock wrapper re-raises every error as RerankModelError.
            try:
                raise _client_error(code)
            except ClientError as e:
                raise RerankModelError(f"Reranking failed: {e}") from e

    scorer = _scorer(_FakeRerankFactory(_ClientErrorReranker()))
    _, rerank = _fuse(scorer, _results("text", "text"))
    assert rerank["reason"] == f"rerank call failed: {code}"
    assert _ARN not in str(rerank)

    init = _scorer(_FakeRerankFactory(raises=_client_error(code)))
    reason = init.new_trace().summary()["reason"]
    assert reason == f"rerank model initialization failed: {code}"


# --- several stages (global search reranked twice) -------------------------------


def test_two_stages_are_recorded_in_order_with_the_worst_as_summary() -> None:
    scorer = _scorer(_FakeRerankFactory(_FakeReranker()))
    trace = scorer.new_trace()
    scorer.fuse_and_rerank_results(
        {"reports": _results("community")}, top_k=5, query="q", rerank_trace=trace
    )
    scorer.fuse_and_rerank_results(
        {"chunks": _results("text", "text")},
        top_k=5,
        query="boom",
        rerank_only_types={"text"},
        rerank_trace=trace,
    )
    assert trace.summary() == {
        "status": "failed",
        "reason": "rerank call failed: RuntimeError",
        "stages": [
            {"status": "applied", "scope": "all", "reason": None},
            {
                "status": "failed",
                "scope": "text",
                "reason": "rerank call failed: RuntimeError",
            },
        ],
    }


@pytest.mark.parametrize(
    ("first", "second"),
    list(itertools.permutations(list(RerankStatus), 2)),
)
def test_summary_order_is_failed_unavailable_applied_skipped_disabled(
    first: RerankStatus, second: RerankStatus
) -> None:
    order = ["failed", "unavailable", "applied", "skipped", "disabled"]
    trace = RerankTrace(RerankStage(status=RerankStatus.SKIPPED, scope="all"))
    trace.record(RerankStage(status=first, scope="all", reason=first.value))
    trace.record(RerankStage(status=second, scope="text", reason=second.value))
    expected = min((first.value, second.value), key=order.index)
    assert trace.summary()["status"] == expected
    assert trace.summary()["reason"] == expected


# --- required: true --------------------------------------------------------------


def test_required_raises_on_init_failure() -> None:
    factory = _FakeRerankFactory(raises=ValueError("no such model"))
    with pytest.raises(RerankModelError, match="could not be built"):
        _scorer(factory, required=True)


def test_required_raises_when_the_provider_builds_no_model() -> None:
    with pytest.raises(RerankModelError, match="no rerank model"):
        _scorer(_FakeRerankFactory(None), required=True)


def test_required_raises_on_runtime_failure() -> None:
    scorer = _scorer(_FakeRerankFactory(_FakeReranker("boom")), required=True)
    with pytest.raises(RerankModelError, match="rerank backend down"):
        _fuse(scorer, _results("text"))


def test_required_raises_on_empty_output() -> None:
    scorer = _scorer(_FakeRerankFactory(_FakeReranker("empty")), required=True)
    with pytest.raises(RerankModelError, match="no usable results"):
        _fuse(scorer, _results("text"))


def test_required_does_not_fail_the_legitimate_no_op_paths() -> None:
    scorer = _scorer(_FakeRerankFactory(_FakeReranker("boom")), required=True)
    assert _fuse(scorer, [])[1]["status"] == "skipped"
    assert _fuse(scorer, _results("text"), query=None)[1]["status"] == "skipped"
    assert _scorer(enabled=False, required=True).rerank_model is None


# --- strategies ------------------------------------------------------------------


class _FakeRetriever:
    def __init__(self, results: list[RetrievalResult] | None = None) -> None:
        self.results = _results("text", "text") if results is None else results

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        return [r.model_copy() for r in self.results]


def _providers(config: Config, factory: Any, **kwargs: Any) -> Providers:
    from unittest.mock import MagicMock

    return Providers(
        config,
        boto_session=MagicMock(),
        rerank_factory=factory,
        token_counter_factory=lambda *_, **__: MagicMock(),
        **kwargs,
    )


def _simple(
    factory: Any, retriever: _FakeRetriever | None = None, **config_kwargs: Any
) -> SimpleSearchStrategy:
    config = _config(**config_kwargs)
    return SimpleSearchStrategy(
        config=config,
        retrievers={RetrieverRole.DOCUMENT.value: retriever or _FakeRetriever()},  # type: ignore[dict-item]
        providers=_providers(config, factory),
    )


async def test_strategy_reports_status_in_search_result_metadata() -> None:
    result = await _simple(_FakeRerankFactory(_FakeReranker())).asearch(
        SearchQuery(query="q")
    )
    assert result.metadata["rerank"]["status"] == "applied"


async def test_strategy_early_return_reports_skipped() -> None:
    strategy = _simple(_FakeRerankFactory(_FakeReranker()), _FakeRetriever([]))
    result = await strategy.asearch(SearchQuery(query="q"))
    assert result.results == []
    assert result.metadata["rerank"]["status"] == "skipped"


async def test_concurrent_queries_on_one_strategy_keep_their_own_status() -> None:
    # One shared strategy, scorer and model (as GraphRAGChain.abatch runs
    # them): the slow query is still reranking while the failing one ends.
    strategy = _simple(_FakeRerankFactory(_FakeReranker()))
    slow, failing = await asyncio.gather(
        strategy.asearch(SearchQuery(query="slow")),
        strategy.asearch(SearchQuery(query="boom")),
    )
    assert slow.metadata["rerank"]["status"] == "applied"
    assert failing.metadata["rerank"]["status"] == "failed"
    assert len(slow.metadata["rerank"]["stages"]) == 1
    assert len(failing.metadata["rerank"]["stages"]) == 1


async def test_required_failure_propagates_out_of_the_strategy() -> None:
    strategy = _simple(_FakeRerankFactory(_FakeReranker("boom")), required=True)
    with pytest.raises(RerankModelError):
        await strategy.asearch(SearchQuery(query="q"))


# --- the chain and processing.ignore_errors ---------------------------------------


class _FakeModelFactory:
    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        return GenericFakeChatModel(
            messages=itertools.repeat(AIMessage(content="Vendor ships 120 units."))
        )

    def get_model_info(self, model_id: Any) -> Any:
        return fake_model_info(model_id)


class _FakeMemoryManager:
    async def load_history(self, conv_id: str) -> str:
        return ""

    async def add_turn(self, *args: Any, **kwargs: Any) -> None:
        return None


async def _ask(*, required: bool, ignore_errors: bool, mode: str = "boom") -> RAGOutput:
    config = _config(required=required)
    config.processing.ignore_errors = ignore_errors
    chain = GraphRAGChain(
        config=config,
        providers=_providers(
            config,
            _FakeRerankFactory(_FakeReranker(mode)),
            llm_factory=_FakeModelFactory(),
            retriever_builders={
                RetrieverRole.DOCUMENT: _FakeRetriever,
                RetrieverRole.GRAPH: _FakeRetriever,
            },
        ),
        memory_manager=_FakeMemoryManager(),  # type: ignore[arg-type]
    )
    chain.token_manager.count_tokens = lambda text: len((text or "").split())  # type: ignore[method-assign]
    try:
        output = await chain.ainvoke(
            RAGInput(
                query="Who ships?",
                search_strategy=SearchStrategy.SIMPLE,
                enable_query_processing=False,
            )
        )
    finally:
        await chain.aclose()
    assert isinstance(output, RAGOutput)
    return output


@pytest.mark.parametrize("ignore_errors", [False, True])
async def test_a_failed_rerank_still_answers_and_reaches_rag_output(
    ignore_errors: bool,
) -> None:
    output = await _ask(required=False, ignore_errors=ignore_errors)
    assert output.answer == "Vendor ships 120 units."
    assert not output.metadata.get("error")
    assert output.metadata["rerank"]["status"] == "failed"
    assert output.search_results.metadata["rerank"]["status"] == "failed"


async def test_required_failure_raises_without_ignore_errors() -> None:
    with pytest.raises(RerankModelError):
        await _ask(required=True, ignore_errors=False)


async def test_required_failure_is_the_error_fallback_under_ignore_errors() -> None:
    output = await _ask(required=True, ignore_errors=True)
    assert output.metadata["error"] is True
    assert "Reranking failed" in output.search_results.metadata["error"]


# --- CLI and evaluation ----------------------------------------------------------


def _print(metadata: dict[str, Any]) -> str:
    out = io.StringIO()
    console = Console(file=out, width=300, color_system=None)
    original = run_rag_chain.console
    run_rag_chain.console = console
    try:
        RAGChainRunner._print_result(
            {"success": True, "answer": "ok", "metadata": metadata}, verbose=True
        )
    finally:
        run_rag_chain.console = original
    return out.getvalue()


def test_verbose_cli_prints_one_rerank_line() -> None:
    rendered = _print(
        {
            "rerank": {
                "status": "failed",
                "reason": "rerank call failed: RuntimeError",
                "stages": [
                    {"status": "failed", "scope": "all", "reason": "x"},
                ],
            }
        }
    )
    assert "Rerank: failed (rerank call failed: RuntimeError)" in rendered


def test_verbose_cli_lists_the_stages_of_a_multi_stage_query() -> None:
    rendered = _print(
        {
            "rerank": {
                "status": "failed",
                "reason": "rerank call failed: RuntimeError",
                "stages": [
                    {"status": "applied", "scope": "all", "reason": None},
                    {"status": "failed", "scope": "text", "reason": "x"},
                ],
            }
        }
    )
    assert "[all: applied; text: failed]" in rendered


def _rag_output(metadata: dict[str, Any]) -> RAGOutput:
    return RAGOutput(
        answer="Vendor",
        sources=[],
        search_results=SearchResult(
            query=SearchQuery(query="?"),
            results=[],
            total_results=0,
            search_strategy="simple",
            processing_time=0.1,
        ),
        conversation_id=None,
        processed_query=ProcessedQuery(original_query="?", final_query="?"),
        metadata=metadata,
    )


class _FakeChain:
    def __init__(self, outputs: dict[str, RAGOutput]) -> None:
        self.outputs = outputs

    async def ainvoke(self, inputs: Any, config: Any = None) -> RAGOutput:
        return self.outputs[inputs["query"]]

    async def abatch(
        self, inputs: Any, config: Any = None, return_exceptions: bool = False
    ) -> list[RAGOutput]:
        return [self.outputs[i["query"]] for i in inputs]


async def test_evaluation_results_and_summary_carry_the_rerank_status() -> None:
    config = Config()
    config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
    chain = _FakeChain(
        {
            "q1": _rag_output({"rerank": {"status": "applied"}}),
            "q2": _rag_output({"rerank": {"status": "failed"}}),
            "q3": _rag_output({"rerank": {"status": "failed"}}),
            "q4": _rag_output({}),
        }
    )
    manager = EvaluationManager(config, rag_chain=chain)
    queries = [EvaluationQuery(query_id=q, question=q) for q in chain.outputs]
    results, _, summary = await manager.evaluate_dataset(
        queries,
        [
            EvaluationGroundTruth(query_id=q.query_id, ground_truth="Vendor")
            for q in queries
        ],
        show_progress=False,
    )
    assert [r.rerank_status for r in results] == ["applied", "failed", "failed", None]
    assert summary.rerank_status_counts == {"applied": 1, "failed": 2}
    assert results[1].model_dump(mode="json")["rerank_status"] == "failed"
