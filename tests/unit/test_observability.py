# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry tracing through the ``shared.observability`` port.

Spans are collected with the SDK's in-memory exporter on a test-owned tracer
provider (``configure_observability(..., tracer_provider=...)``), so nothing
is exported and no global provider is set. Covers the no-op default, the
fail-fast missing-extra error, ingestion and query span trees, the GenAI
callback handler (token usage, content capture and redaction), trace ids in
log records and context propagation across the executor.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind, StatusCode

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from tests.fixtures.fakes.chat_models import fake_model_info
from unified_kg_rag.adapters.observability.langchain_callback import (
    GenAITracingCallbackHandler,
)
from unified_kg_rag.application.observability import configure_observability
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, RAGInput
from unified_kg_rag.domain.models import (
    Config,
    ModelPurpose,
    ObservabilityBackend,
    ObservabilityConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.shared import ConfigurationError, observability
from unified_kg_rag.shared.logging import _SHARED_PROCESSORS, add_trace_ids
from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor
from unified_kg_rag.shared.utils.langchain import BatchProcessor

pytestmark = pytest.mark.unit

_PASSAGE = "Vendor and Buyer signed the supply agreement for 120 units."


Tracing = Callable[..., InMemorySpanExporter]


@pytest.fixture
def tracing() -> Iterator[Tracing]:
    """Enable tracing into an in-memory exporter; reset to no-op afterwards."""

    def _enable(**settings: Any) -> InMemorySpanExporter:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        configure_observability(
            ObservabilityConfig(enabled=True, **settings), tracer_provider=provider
        )
        return exporter

    yield _enable
    observability.uninstall()


def _by_name(exporter: InMemorySpanExporter) -> dict[str, list[ReadableSpan]]:
    spans: dict[str, list[ReadableSpan]] = {}
    for span in exporter.get_finished_spans():
        spans.setdefault(span.name, []).append(span)
    return spans


def _one(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    spans = _by_name(exporter).get(name, [])
    assert len(spans) == 1, f"expected one '{name}' span, got {len(spans)}"
    return spans[0]


def _is_child(child: ReadableSpan, parent: ReadableSpan) -> bool:
    return (
        child.parent is not None
        and parent.context is not None
        and child.parent.span_id == parent.context.span_id
        and child.context is not None
        and child.context.trace_id == parent.context.trace_id
    )


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


# --- disabled (default) ----------------------------------------------------


def test_disabled_by_default_and_every_call_is_a_shared_noop() -> None:
    assert Config().observability.enabled is False
    assert not observability.is_enabled()
    first, second = observability.span("a", x=1), observability.span("b")
    assert first is second  # one shared null context, nothing allocated
    with first as span:
        assert span is observability.NOOP_SPAN
        span.set_attributes({"k": "v"})
    assert observability.start_span("c") is observability.NOOP_SPAN
    assert observability.current_trace_ids() is None
    assert observability.content_policy().capture_content is False
    assert configure_observability(ObservabilityConfig()) is False


def test_disabled_path_never_imports_the_otel_adapter() -> None:
    # A fresh interpreter in which OpenTelemetry cannot be imported at all: the
    # composition roots, the port and the batch processor still work, and the
    # OTel adapter is never loaded.
    script = textwrap.dedent("""
        import sys
        sys.modules["opentelemetry"] = None
        from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
        from unified_kg_rag.application.observability import configure_observability
        from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
        from unified_kg_rag.domain.models import Config
        from unified_kg_rag.shared import observability
        from unified_kg_rag.shared.utils.langchain import BatchProcessor

        assert configure_observability(Config().observability) is False
        with observability.span("kg_rag.x", a=1) as span:
            span.set_attribute("b", 2)
        out = BatchProcessor(max_concurrency=2).execute_with_fallback(
            [1, 2], lambda items: [{"x": i} for i in items], lambda d: d["x"], "t",
            show_progress=False,
        )
        assert out == [1, 2], out
        loaded = sorted(m for m in sys.modules if m.startswith(
            "unified_kg_rag.adapters.observability"))
        print(loaded)
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"


def test_enabled_without_the_otel_extra_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    monkeypatch.delitem(
        sys.modules, "unified_kg_rag.adapters.observability.otel", raising=False
    )
    with pytest.raises(ConfigurationError, match="'otel' extra"):
        configure_observability(ObservabilityConfig(enabled=True))
    assert not observability.is_enabled()


def test_otel_sdk_disabled_env_wins(
    monkeypatch: pytest.MonkeyPatch, tracing: Tracing
) -> None:
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert configure_observability(ObservabilityConfig(enabled=True)) is False


# --- config ---------------------------------------------------------------


def test_config_defaults_and_validation() -> None:
    cfg = ObservabilityConfig()
    assert cfg.backend == ObservabilityBackend.NONE
    assert (cfg.capture_content, cfg.sample_rate, cfg.redact_patterns) == (
        False,
        1.0,
        [],
    )
    assert ObservabilityConfig(backend="otlp").backend == ObservabilityBackend.OTLP
    with pytest.raises(ValueError):
        ObservabilityConfig(backend="cloudwatch")  # use otlp + an ADOT collector
    with pytest.raises(ValueError):
        ObservabilityConfig(sample_rate=1.5)
    with pytest.raises(ValueError, match="redact_patterns"):
        ObservabilityConfig(redact_patterns=["(unclosed"])


def test_otlp_backend_without_exporter_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unified_kg_rag.adapters.observability import otel

    for module in (
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
    ):
        monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ConfigurationError, match="OTLP exporter"):
        otel.build_backend(ObservabilityConfig(enabled=True, backend="otlp"))


def test_otlp_backend_rejects_an_unknown_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unified_kg_rag.adapters.observability import otel

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json")
    with pytest.raises(ConfigurationError, match="http/json"):
        otel.build_backend(ObservabilityConfig(enabled=True, backend="otlp"))


def test_otlp_backend_builds_a_sampled_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    from unified_kg_rag.adapters.observability import otel

    installed: list[Any] = []
    monkeypatch.setattr(otel.trace, "set_tracer_provider", installed.append)
    monkeypatch.setattr(
        otel.trace, "get_tracer_provider", lambda: otel.trace.ProxyTracerProvider()
    )
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    backend = otel.build_backend(
        ObservabilityConfig(
            enabled=True, backend="otlp", sample_rate=0.25, service_name="svc"
        )
    )
    try:
        (provider,) = installed
        assert provider.resource.attributes["service.name"] == "svc"
        assert "0.25" in provider.sampler.get_description()
    finally:
        backend.shutdown()


def test_attribute_values_are_coerced() -> None:
    from unified_kg_rag.adapters.observability.otel import to_attribute_value

    assert to_attribute_value(None) is None
    assert to_attribute_value(SearchStrategy.LOCAL) == "local"
    assert to_attribute_value(["a", SearchStrategy.MIX]) == ["a", "mix"]
    assert to_attribute_value([1, 2]) == [1, 2]
    assert to_attribute_value([1, 2.5]) == [1.0, 2.5]
    assert to_attribute_value([1, "a"]) == ["1", "a"]
    assert to_attribute_value({"k": 1}) == "{'k': 1}"


# --- port behaviour -----------------------------------------------------------


def test_span_records_namespaced_attributes_and_escaping_errors(
    tracing: Tracing,
) -> None:
    exporter = tracing()
    with pytest.raises(RuntimeError):
        with observability.span(
            "outer", {"gen_ai.request.model": "m"}, hits=3
        ) as handle:
            handle.set_attributes({"kg_rag.extra": True})
            raise RuntimeError("boom")
    span = _one(exporter, "outer")
    assert _attrs(span) == {
        "gen_ai.request.model": "m",
        "kg_rag.hits": 3,
        "kg_rag.extra": True,
        "error.type": "RuntimeError",
    }
    assert span.kind == SpanKind.INTERNAL
    assert span.status.status_code == StatusCode.ERROR
    assert span.events[0].name == "exception"


def test_trace_ids_reach_log_records(tracing: Tracing) -> None:
    assert add_trace_ids in _SHARED_PROCESSORS
    assert add_trace_ids(None, "info", {}) == {}
    tracing()
    assert add_trace_ids(None, "info", {}) == {}  # enabled, no active span
    with observability.span("logged"):
        trace_id, span_id = observability.current_trace_ids() or ("", "")
        event = add_trace_ids(None, "info", {"event": "x"})
    assert event == {"event": "x", "trace_id": trace_id, "span_id": span_id}
    assert len(trace_id) == 32 and len(span_id) == 16


def test_context_propagates_across_the_executor(tracing: Tracing) -> None:
    exporter = tracing()

    def _work(n: int) -> int:
        with observability.span("worker", n=n):
            return n

    with observability.span("parent"):
        with ContextThreadPoolExecutor(max_workers=2) as executor:
            assert sorted(executor.map(_work, [1, 2])) == [1, 2]
        with ThreadPoolExecutor(max_workers=1) as plain:
            plain.submit(_work, 3).result()

    parent = _one(exporter, "parent")
    workers = {_attrs(s)["kg_rag.n"]: s for s in _by_name(exporter)["worker"]}
    assert _is_child(workers[1], parent) and _is_child(workers[2], parent)
    # The premise: a plain pool thread starts a new trace.
    assert workers[3].parent is None


# --- ingestion ------------------------------------------------------------------


def _stage(
    stage_type: PipelineStageType,
    outcome: Callable[[], tuple[int, int]],
    metrics: dict[str, Any] | None = None,
):
    from unified_kg_rag.application.ingestion.pipeline_stages import PipelineStage

    class _Stage(PipelineStage):
        def __init__(self) -> None:  # no boto session / providers
            self.config = Config()
            self.stage_type = stage_type

        def _execute_core(
            self, context: PipelineContext
        ) -> tuple[int, int, dict[str, Any] | None]:
            inputs, outputs = outcome()
            return inputs, outputs, metrics

    return _Stage()


def test_ingestion_run_and_stage_spans(tracing: Tracing) -> None:
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline

    exporter = tracing()
    context = PipelineContext(
        pipeline_id="p-1",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=Path("."),
    )
    ok = _stage(
        PipelineStageType.DOCUMENT_PARSING,
        lambda: (3, 2),
        {"failed_files": 1, "duration": 0.5, "partial": True, "source": "s3"},
    )

    def _fail() -> tuple[int, int]:
        raise RuntimeError("synthetic stage failure")

    failing = _stage(PipelineStageType.TEXT_CHUNKING, _fail)

    def _run(*_: Any) -> PipelineContext:
        context.stage_results = [ok.execute(context), failing.execute(context)]
        context.status = PipelineStageStatus.FAILED
        return context

    pipeline = object.__new__(DataIngestionPipeline)
    pipeline._run = _run  # type: ignore[method-assign]
    pipeline.run(".")

    run = _one(exporter, "kg_rag.ingestion.run")
    stages = {
        _attrs(s)["kg_rag.stage"]: s
        for s in _by_name(exporter)["kg_rag.ingestion.stage"]
    }
    assert _attrs(run)["kg_rag.pipeline_id"] == "p-1"
    assert _attrs(run)["kg_rag.stages_completed"] == 1
    parsing = stages[PipelineStageType.DOCUMENT_PARSING.value]
    assert _is_child(parsing, run)
    assert _attrs(parsing)["kg_rag.input_count"] == 3
    assert _attrs(parsing)["kg_rag.output_count"] == 2
    assert _attrs(parsing)["kg_rag.stage.status"] == PipelineStageStatus.COMPLETED.value
    # Numeric metrics only: a completed stage can still report dropped work.
    assert _attrs(parsing)["kg_rag.stage.failed_files"] == 1
    assert _attrs(parsing)["kg_rag.stage.duration"] == 0.5
    assert "kg_rag.stage.partial" not in _attrs(parsing)
    assert "kg_rag.stage.source" not in _attrs(parsing)
    chunking = stages[PipelineStageType.TEXT_CHUNKING.value]
    assert chunking.status.status_code == StatusCode.ERROR
    assert _attrs(chunking)["error.type"] == "RuntimeError"
    assert [e.name for e in chunking.events] == ["exception"]
    # The pipeline reports the failure on the context, not by raising.
    assert run.status.status_code == StatusCode.ERROR


def test_batch_span_parents_the_calls_on_worker_threads(tracing: Tracing) -> None:
    exporter = tracing()
    processor = BatchProcessor(max_concurrency=2, max_attempts=1)

    def call(item: dict[str, int]) -> int:
        with observability.span("kg_rag.test.item"):
            return item["x"] * 10

    results = processor.execute_with_fallback(
        [1, 2, 3, 4, 5],
        lambda items: [{"x": i} for i in items],
        call,
        "synthetic task",
        show_progress=False,
    )
    assert results == [10, 20, 30, 40, 50]
    batch = _one(exporter, "kg_rag.batch")
    calls = _by_name(exporter)["kg_rag.test.item"]
    assert _attrs(batch)["kg_rag.task"] == "synthetic task"
    assert (_attrs(batch)["kg_rag.items"], _attrs(batch)["kg_rag.failed"]) == (5, 0)
    assert len(calls) == 5 and all(_is_child(c, batch) for c in calls)


# --- query ------------------------------------------------------------------------


class _FakeRetriever:
    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                content=_PASSAGE,
                score=0.9,
                source=f"doc-{i}",
                retriever_type="text",
                metadata={"id": f"doc-{i}", "text_unit_ids": []},
            )
            for i in range(2)
        ]


class _FakeModelFactory:
    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        return GenericFakeChatModel(
            messages=itertools.repeat(AIMessage(content="Vendor ships 120 units."))
        )

    def get_model_info(self, model_id: Any) -> Any:
        return fake_model_info(model_id)


class _FakeMemoryManager:
    async def load_history(self, conv_id: str) -> str:
        raise AssertionError("memory is off in these queries")

    async def add_turn(self, *args: Any, **kwargs: Any) -> None:
        return None


async def test_query_span_tree_carries_counts_and_ids_not_content(
    tracing: Tracing,
) -> None:
    exporter = tracing()
    config = Config()
    config.search.reranking.enabled = False
    chain = GraphRAGChain(
        config=config,
        model_factory=_FakeModelFactory(),
        memory_manager=_FakeMemoryManager(),  # type: ignore[arg-type]
        retriever_builders={
            RetrieverRole.DOCUMENT: _FakeRetriever,  # type: ignore[dict-item]
            RetrieverRole.GRAPH: _FakeRetriever,  # type: ignore[dict-item]
        },
    )
    chain.token_manager.count_tokens = lambda text: len((text or "").split())
    try:
        output = await chain.ainvoke(
            RAGInput(
                query="What does Vendor supply?",
                search_strategy=SearchStrategy.SIMPLE,
                enable_query_processing=False,
            )
        )
    finally:
        await chain.aclose()
    assert output.answer  # type: ignore[union-attr]

    query = _one(exporter, "kg_rag.query")
    strategy = _one(exporter, "kg_rag.strategy")
    retrieve = _by_name(exporter)["kg_rag.retrieve"]
    fusion = _one(exporter, "kg_rag.fusion")
    context = _one(exporter, "kg_rag.context_build")

    assert query.parent is None
    assert _attrs(query)["kg_rag.requested_strategy"] == "simple"
    assert _attrs(query)["kg_rag.sources"] >= 1
    assert _attrs(query)["kg_rag.error"] is False
    assert _is_child(strategy, query) and _is_child(context, query)
    assert _attrs(strategy)["kg_rag.strategy"] == "simple"
    assert all(_is_child(r, strategy) for r in retrieve)
    assert _is_child(fusion, strategy)
    assert _attrs(retrieve[0])["kg_rag.hits"] == 2
    assert list(_attrs(retrieve[0])["kg_rag.source_ids"]) == ["doc-0", "doc-1"]
    assert _attrs(fusion)["kg_rag.rerank_enabled"] is False
    assert _attrs(context)["kg_rag.context.tokens_used"] > 0
    assert (
        _attrs(context)["kg_rag.context.tokens_used"]
        <= _attrs(context)["kg_rag.context.token_budget"]
    )
    # Spans carry no retrieved text and no query text.
    for span in exporter.get_finished_spans():
        for value in _attrs(span).values():
            assert "supply agreement" not in str(value)
            assert "What does Vendor supply" not in str(value)


def _simple_chain() -> GraphRAGChain:
    config = Config()
    config.search.reranking.enabled = False
    chain = GraphRAGChain(
        config=config,
        model_factory=_FakeModelFactory(),
        memory_manager=_FakeMemoryManager(),  # type: ignore[arg-type]
        retriever_builders={
            RetrieverRole.DOCUMENT: _FakeRetriever,  # type: ignore[dict-item]
            RetrieverRole.GRAPH: _FakeRetriever,  # type: ignore[dict-item]
        },
    )
    chain.token_manager.count_tokens = lambda text: len((text or "").split())
    return chain


_SIMPLE_QUERY = RAGInput(
    query="What does Vendor supply?",
    search_strategy=SearchStrategy.SIMPLE,
    enable_query_processing=False,
)


async def test_captured_query_span_carries_redacted_query_and_answer(
    tracing: Tracing,
) -> None:
    # Regression: the root span had no input/output, so the Langfuse and
    # Phoenix trace lists showed every query trace with empty input/output.
    exporter = tracing(capture_content=True, redact_patterns=[r"\d+ units"])
    chain = _simple_chain()
    try:
        output = await chain.ainvoke(_SIMPLE_QUERY)
    finally:
        await chain.aclose()
    assert output.answer == "Vendor ships 120 units."  # type: ignore[union-attr]
    query = _attrs(_one(exporter, "kg_rag.query"))
    assert query["input.value"] == "What does Vendor supply?"
    assert query["output.value"] == "Vendor ships [REDACTED]."


async def test_streamed_query_span_carries_only_the_redacted_query(
    tracing: Tracing,
) -> None:
    # The streamed answer is produced after the query span has ended.
    exporter = tracing(capture_content=True, redact_patterns=["Vendor"])
    chain = _simple_chain()
    try:
        chunks = [chunk async for chunk in chain.astream(_SIMPLE_QUERY)]
    finally:
        await chain.aclose()
    assert "".join(chunks) == "Vendor ships 120 units."
    query = _attrs(_one(exporter, "kg_rag.query"))
    assert query["input.value"] == "What does [REDACTED] supply?"
    assert "output.value" not in query


async def test_query_span_has_no_input_or_output_without_capture(
    tracing: Tracing,
) -> None:
    exporter = tracing()
    chain = _simple_chain()
    try:
        await chain.ainvoke(_SIMPLE_QUERY)
        _ = [chunk async for chunk in chain.astream(_SIMPLE_QUERY)]
    finally:
        await chain.aclose()
    for span in _by_name(exporter)["kg_rag.query"]:
        assert not {"input.value", "output.value"} & set(_attrs(span))


async def test_fusion_and_strategy_spans_carry_the_rerank_status(
    tracing: Tracing,
) -> None:
    from unittest.mock import MagicMock

    from unified_kg_rag.adapters.providers import Providers

    class _FailingReranker:
        top_n = None

        def compress_documents(self, documents: Any, query: str) -> Any:
            raise RuntimeError("rerank backend down")

    class _RerankFactory:
        kind = "fake"

        def get_model(self, model_id: Any, **kwargs: Any) -> Any:
            return _FailingReranker()

        def get_model_info(self, model_id: Any) -> None:
            return None

    exporter = tracing()
    config = Config()
    chain = GraphRAGChain(
        config=config,
        providers=Providers(
            config,
            boto_session=MagicMock(),
            llm_factory=_FakeModelFactory(),
            rerank_factory=_RerankFactory(),
            retriever_builders={
                RetrieverRole.DOCUMENT: _FakeRetriever,  # type: ignore[dict-item]
                RetrieverRole.GRAPH: _FakeRetriever,  # type: ignore[dict-item]
            },
        ),
        memory_manager=_FakeMemoryManager(),  # type: ignore[arg-type]
    )
    chain.token_manager.count_tokens = lambda text: len((text or "").split())
    try:
        await chain.ainvoke(
            RAGInput(
                query="What does Vendor supply?",
                search_strategy=SearchStrategy.SIMPLE,
                enable_query_processing=False,
            )
        )
    finally:
        await chain.aclose()

    assert _attrs(_one(exporter, "kg_rag.fusion"))["kg_rag.rerank_status"] == "failed"
    assert _attrs(_one(exporter, "kg_rag.strategy"))["kg_rag.rerank_status"] == (
        "failed"
    )


async def test_a_degraded_strategy_marks_its_span_with_the_error_type(
    tracing: Tracing,
) -> None:
    # Regression: a map/reduce failure ignore_errors absorbed left no
    # error.type on the kg_rag.strategy span.
    exporter = tracing()
    config = Config()
    config.search.reranking.enabled = False
    chain = GraphRAGChain(
        config=config,
        model_factory=_FakeModelFactory(),
        memory_manager=_FakeMemoryManager(),  # type: ignore[arg-type]
        retriever_builders={
            RetrieverRole.DOCUMENT: _FakeRetriever,  # type: ignore[dict-item]
            RetrieverRole.GRAPH: _FakeRetriever,  # type: ignore[dict-item]
        },
    )
    chain.token_manager.count_tokens = lambda text: len((text or "").split())
    strategy = chain._get_strategy_instance(SearchStrategy.SIMPLE)
    real_asearch = strategy.asearch

    async def degraded_asearch(query: Any, config: Any = None) -> Any:
        result = await real_asearch(query, config=config)
        result.metadata["degraded_error"] = "TimeoutError"
        return result

    strategy.asearch = degraded_asearch  # type: ignore[method-assign]
    try:
        output = await chain.ainvoke(
            RAGInput(
                query="What does Vendor supply?",
                search_strategy=SearchStrategy.SIMPLE,
                enable_query_processing=False,
            )
        )
    finally:
        await chain.aclose()

    assert output.metadata["degraded_error"] == "TimeoutError"  # type: ignore[union-attr]
    assert _attrs(_one(exporter, "kg_rag.strategy"))["error.type"] == "TimeoutError"


async def test_query_spans_carry_the_conversation_as_session_and_the_user(
    tracing: Tracing,
) -> None:
    exporter = tracing()
    config = Config()
    config.search.reranking.enabled = False
    chain = GraphRAGChain(
        config=config,
        model_factory=_FakeModelFactory(),
        memory_manager=_FakeMemoryManager(),  # type: ignore[arg-type]
        retriever_builders={
            RetrieverRole.DOCUMENT: _FakeRetriever,  # type: ignore[dict-item]
            RetrieverRole.GRAPH: _FakeRetriever,  # type: ignore[dict-item]
        },
    )
    chain.token_manager.count_tokens = lambda text: len((text or "").split())
    try:
        await chain.ainvoke(
            RAGInput(
                query="What does Vendor supply?",
                search_strategy=SearchStrategy.SIMPLE,
                enable_query_processing=False,
                conversation_id="conv-42",
                user_id="user-7",
            )
        )
    finally:
        await chain.aclose()
    spans = exporter.get_finished_spans()
    assert {s.name for s in spans} >= {"kg_rag.query", "kg_rag.retrieve"}
    for span in spans:
        assert _attrs(span)["session.id"] == "conv-42", span.name
        assert _attrs(span)["user.id"] == "user-7", span.name
    assert _attrs(_one(exporter, "kg_rag.query"))["gen_ai.conversation.id"] == (
        "conv-42"
    )


async def test_a_query_degraded_under_ignore_errors_is_an_error_span(
    tracing: Tracing,
) -> None:
    # ignore_errors turns the failure into a fallback answer; the query span
    # must still say the query failed instead of staying UNSET.
    exporter = tracing()
    config = Config()
    config.processing.ignore_errors = True

    def _broken() -> Any:
        raise RuntimeError("synthetic store failure")

    chain = GraphRAGChain(
        config=config,
        model_factory=_FakeModelFactory(),
        memory_manager=_FakeMemoryManager(),  # type: ignore[arg-type]
        retriever_builders={
            RetrieverRole.DOCUMENT: _broken,
            RetrieverRole.GRAPH: _broken,
        },
    )
    try:
        output = await chain.ainvoke(
            RAGInput(
                query="What does Vendor supply?",
                search_strategy=SearchStrategy.SIMPLE,
                enable_query_processing=False,
            )
        )
    finally:
        await chain.aclose()

    assert output.metadata["error"] is True  # type: ignore[union-attr]
    query = _one(exporter, "kg_rag.query")
    assert query.status.status_code == StatusCode.ERROR
    assert query.status.description == "RuntimeError"
    # error.type too, as on a query that raised (OTel error-recording rule).
    assert _attrs(query)["error.type"] == "RuntimeError"
    assert [e.name for e in query.events] == ["exception"]
    assert _attrs(query)["kg_rag.error"] is True


# --- GenAI callback handler ---------------------------------------------------------


def _usage_message(text: str) -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "input_token_details": {"cache_read": 100, "cache_creation": 8},
        },
        response_metadata={"stopReason": "end_turn", "model_id": "vendor.model-v1"},
    )


def _traced_model(text: str) -> GenericFakeChatModel:
    handler = GenAITracingCallbackHandler(
        "vendor.model-v1",
        purpose=ModelPurpose.QUERY.value,
        max_tokens=512,
        temperature=0.1,
    )
    return GenericFakeChatModel(
        messages=itertools.repeat(_usage_message(text)), callbacks=[handler]
    )


_PROMPT = [
    SystemMessage(content="Answer from context."),
    HumanMessage(content="Account 123-456 asks about the Vendor contract."),
]


async def test_model_span_records_genai_usage_and_nests(tracing: Tracing) -> None:
    exporter = tracing()
    with observability.span("caller"):
        _traced_model("Vendor ships parts.").invoke(_PROMPT)
        await _traced_model("Vendor ships parts.").ainvoke(_PROMPT)

    caller = _one(exporter, "caller")
    calls = _by_name(exporter)["chat vendor.model-v1"]
    assert len(calls) == 2
    for call in calls:
        attrs = _attrs(call)
        assert _is_child(call, caller)
        assert call.kind == SpanKind.CLIENT
        assert attrs["gen_ai.operation.name"] == "chat"
        assert attrs["gen_ai.provider.name"] == "aws.bedrock"
        assert attrs["gen_ai.request.model"] == "vendor.model-v1"
        assert attrs["gen_ai.request.max_tokens"] == 512
        assert attrs["gen_ai.request.temperature"] == 0.1
        assert attrs["gen_ai.response.model"] == "vendor.model-v1"
        assert attrs["gen_ai.usage.input_tokens"] == 120
        assert attrs["gen_ai.usage.output_tokens"] == 30
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 100
        assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 8
        assert list(attrs["gen_ai.response.finish_reasons"]) == ["end_turn"]
        assert attrs["kg_rag.model.purpose"] == "query"
        # Content capture is off by default.
        assert "gen_ai.input.messages" not in attrs
        assert "gen_ai.output.messages" not in attrs


def test_model_span_captures_redacted_content_when_enabled(tracing: Tracing) -> None:
    exporter = tracing(capture_content=True, redact_patterns=[r"\d{3}-\d{3}"])
    _traced_model("Account 987-654 is in scope.").invoke(_PROMPT)

    attrs = _attrs(_one(exporter, "chat vendor.model-v1"))
    inputs = json.loads(attrs["gen_ai.input.messages"])
    outputs = json.loads(attrs["gen_ai.output.messages"])
    assert [m["role"] for m in inputs] == ["system", "user"]
    assert inputs[1]["parts"][0]["content"] == (
        "Account [REDACTED] asks about the Vendor contract."
    )
    assert outputs[0]["parts"][0]["content"] == "Account [REDACTED] is in scope."
    assert outputs[0]["finish_reason"] == "end_turn"


def test_model_error_ends_the_span_with_error_status(tracing: Tracing) -> None:
    exporter = tracing()

    class _Failing(GenericFakeChatModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("synthetic model failure")

    model = _Failing(
        messages=iter([]),
        callbacks=[GenAITracingCallbackHandler("vendor.model-v1")],
    )
    with pytest.raises(RuntimeError):
        model.invoke("hi")
    span = _one(exporter, "chat vendor.model-v1")
    assert span.status.status_code == StatusCode.ERROR
    assert _attrs(span)["error.type"] == "RuntimeError"


@pytest.mark.parametrize("asynchronous", [False, True], ids=["stream", "astream"])
async def test_a_stream_stopped_early_ends_its_span_without_error(
    tracing: Tracing, asynchronous: bool
) -> None:
    exporter = tracing()
    handler = GenAITracingCallbackHandler("vendor.model-v1")
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="Vendor ships parts in two lots.")]),
        callbacks=[handler],
    )
    if asynchronous:
        chunks = model.astream(_PROMPT)
        async for _ in chunks:
            break
        await chunks.aclose()
    else:
        stream = model.stream(_PROMPT)
        next(stream)
        stream.close()
    span = _one(exporter, "chat vendor.model-v1")
    assert span.status.status_code != StatusCode.ERROR
    assert "error.type" not in _attrs(span)
    assert not span.events
    assert handler._spans == {}


async def test_model_span_takes_session_and_user_from_run_metadata(
    tracing: Tracing,
) -> None:
    exporter = tracing()
    # No bind_identity block: a streamed answer's model call can outlive it.
    await _traced_model("Vendor ships parts.").ainvoke(
        _PROMPT, config={"metadata": {"session_id": "conv-42", "user_id": "user-7"}}
    )
    _traced_model("Vendor ships parts.").invoke(_PROMPT)
    with_ids, without_ids = _by_name(exporter)["chat vendor.model-v1"]
    assert (_attrs(with_ids)["session.id"], _attrs(with_ids)["user.id"]) == (
        "conv-42",
        "user-7",
    )
    assert "session.id" not in _attrs(without_ids)


@pytest.mark.parametrize(
    "metadata",
    [
        {"stopReason": "end_turn"},
        {"stop_reason": "end_turn"},
        {"finish_reason": "end_turn"},
        {"done_reason": "end_turn"},
    ],
    ids=["converse", "invoke-model", "openai-compatible", "ollama"],
)
def test_finish_reason_is_read_from_every_provider_shape(
    tracing: Tracing, metadata: dict[str, str]
) -> None:
    exporter = tracing()
    handler = GenAITracingCallbackHandler("vendor.model-v1")
    run_id = uuid4()
    handler.on_chat_model_start({}, [_PROMPT], run_id=run_id)
    message = AIMessage(content="Done.", response_metadata=metadata)
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id
    )
    attrs = _attrs(_one(exporter, "chat vendor.model-v1"))
    assert list(attrs["gen_ai.response.finish_reasons"]) == ["end_turn"]


def test_bedrock_factory_attaches_the_handler_only_when_enabled(
    tracing: Tracing,
) -> None:
    from unified_kg_rag.adapters.aws.bedrock import BedrockLanguageModelFactory

    disabled: dict[str, Any] = {"callbacks": None}
    BedrockLanguageModelFactory._apply_tracing(disabled, "m", ModelPurpose.QUERY)
    assert disabled == {"callbacks": None}

    tracing()
    existing = object()
    # InvokeModel carries the request parameters in model_kwargs.
    enabled: dict[str, Any] = {
        "callbacks": [existing],
        "model_kwargs": {"max_tokens": 256, "temperature": 0.2},
    }
    BedrockLanguageModelFactory._apply_tracing(enabled, "m", ModelPurpose.INGESTION)
    first, handler = enabled["callbacks"]
    assert first is existing
    assert isinstance(handler, GenAITracingCallbackHandler)
    assert (handler.model_id, handler.purpose) == ("m", "ingestion")
    assert (handler.max_tokens, handler.temperature) == (256, 0.2)
    # Converse carries them at the top level.
    converse: dict[str, Any] = {"max_tokens": 128}
    BedrockLanguageModelFactory._apply_tracing(converse, "m", ModelPurpose.QUERY)
    (handler,) = converse["callbacks"]
    assert (handler.max_tokens, handler.temperature) == (128, None)


# --- Bedrock embedding and rerank wrappers ------------------------------------------


def test_embedding_span_counts_texts_once_per_call(
    tracing: Tracing, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_aws import BedrockEmbeddings

    from unified_kg_rag.adapters.aws.bedrock import BedrockEmbeddingsWrapper

    exporter = tracing()
    monkeypatch.setattr(
        BedrockEmbeddings,
        "embed_documents",
        lambda self, texts: [[0.0] for _ in texts],
    )
    model = BedrockEmbeddingsWrapper(
        client=object(), model_id="amazon.titan-embed-text-v2:0"
    )
    model.embed_documents(["alpha", "beta"])

    span = _one(exporter, "embeddings amazon.titan-embed-text-v2:0")
    assert _attrs(span)["gen_ai.operation.name"] == "embeddings"
    assert (_attrs(span)["kg_rag.texts"], _attrs(span)["kg_rag.characters"]) == (2, 9)


async def test_async_embedding_fan_out_adds_no_nested_spans(
    tracing: Tracing, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_aws import BedrockEmbeddings

    from unified_kg_rag.adapters.aws.bedrock import BedrockEmbeddingsWrapper

    exporter = tracing()
    monkeypatch.setattr(BedrockEmbeddings, "embed_query", lambda self, text: [0.0])
    model = BedrockEmbeddingsWrapper(
        client=object(), model_id="amazon.titan-embed-text-v2:0"
    )
    assert await model.aembed_documents(["alpha", "beta", "gamma"]) == [[0.0]] * 3
    span = _one(exporter, "embeddings amazon.titan-embed-text-v2:0")
    assert _attrs(span)["kg_rag.texts"] == 3


@pytest.mark.parametrize(
    ("returned", "status"),
    [([Document(page_content="x", metadata={"key": "a"})], "ok"), ([], "empty")],
)
def test_rerank_span_reports_status_and_counts(
    tracing: Tracing,
    monkeypatch: pytest.MonkeyPatch,
    returned: list[Document],
    status: str,
) -> None:
    from langchain_aws.document_compressors.rerank import BedrockRerank

    from unified_kg_rag.adapters.aws.bedrock import BedrockRerankWrapper

    exporter = tracing()
    monkeypatch.setattr(
        BedrockRerank, "compress_documents", lambda self, docs, query, **_: returned
    )
    model = BedrockRerankWrapper(
        model_arn="arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0",
        client=object(),
        region_name="us-east-1",
        top_n=2,
    )
    docs = [Document(page_content=t) for t in ("a", "b", "c")]
    assert model.compress_documents(docs, "q") == returned

    attrs = _attrs(_one(exporter, "kg_rag.rerank"))
    assert attrs["kg_rag.rerank.status"] == status
    assert (attrs["kg_rag.documents_in"], attrs["kg_rag.documents_out"]) == (
        3,
        len(returned),
    )
    assert attrs["gen_ai.request.model"] == "cohere.rerank-v3-5:0"


def test_rerank_failure_is_an_error_span(
    tracing: Tracing, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_aws.document_compressors.rerank import BedrockRerank

    from unified_kg_rag.adapters.aws.bedrock import BedrockRerankWrapper
    from unified_kg_rag.shared import RerankModelError

    exporter = tracing()

    def _raise(*_: Any, **__: Any) -> Any:
        raise RuntimeError("endpoint unreachable")

    monkeypatch.setattr(BedrockRerank, "compress_documents", _raise)
    model = BedrockRerankWrapper(
        model_arn="arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0",
        client=object(),
        region_name="us-east-1",
    )
    with pytest.raises(RerankModelError):
        model.compress_documents([Document(page_content="a")], "q")
    span = _one(exporter, "kg_rag.rerank")
    assert _attrs(span)["kg_rag.rerank.status"] == "error"
    assert span.status.status_code == StatusCode.ERROR


def test_guardrail_intervention_and_invoke_model_usage_fallback(
    tracing: Tracing,
) -> None:
    exporter = tracing()
    handler = GenAITracingCallbackHandler("vendor.model-v1")
    run_id = uuid4()
    handler.on_chat_model_start({}, [_PROMPT], run_id=run_id)
    # InvokeModel with guardrail trace: on_llm_error first, then on_llm_end.
    handler.on_llm_error(
        RuntimeError("blocked"), run_id=run_id, reason="GUARDRAIL_INTERVENED"
    )
    handler.on_llm_end(
        LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="Blocked."))]],
            llm_output={"usage": {"prompt_tokens": 11, "completion_tokens": 2}},
        ),
        run_id=run_id,
    )
    attrs = _attrs(_one(exporter, "chat vendor.model-v1"))
    assert attrs["aws.bedrock.guardrail.intervened"] is True
    assert (
        attrs["gen_ai.usage.input_tokens"],
        attrs["gen_ai.usage.output_tokens"],
    ) == (11, 2)
