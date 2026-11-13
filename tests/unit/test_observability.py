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

import asyncio
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
from unified_kg_rag.adapters.llm.langchain import BatchProcessor
from unified_kg_rag.adapters.observability.langchain_callback import (
    GenAITracingCallbackHandler,
)
from unified_kg_rag.application.observability import configure_observability
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, RAGInput
from unified_kg_rag.domain.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationQuery,
    EvaluationResult,
    EvaluatorType,
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
from unified_kg_rag.evaluation import EvaluationManager
from unified_kg_rag.evaluation.evaluation_manager import GraphAwareEvaluator
from unified_kg_rag.shared import ConfigurationError, PipelineInterrupted, observability
from unified_kg_rag.shared.logging import _SHARED_PROCESSORS, add_trace_ids
from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor

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


@pytest.mark.slow
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
        from unified_kg_rag.adapters.llm.langchain import BatchProcessor

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


@pytest.mark.parametrize("stop", [KeyboardInterrupt, SystemExit, PipelineInterrupted])
def test_an_interrupted_span_is_an_error(tracing: Tracing, stop) -> None:
    # Before, an interrupt left the span UNSET, so a stopped query exported
    # as a success.
    exporter = tracing()
    with pytest.raises(stop):
        with observability.span("kg_rag.query"):
            raise stop(15) if stop is not KeyboardInterrupt else stop()
    span = _one(exporter, "kg_rag.query")
    assert span.status.status_code == StatusCode.ERROR
    assert _attrs(span)["error.type"] == stop.__name__


def test_a_cancelled_span_is_an_error(tracing: Tracing) -> None:
    # Before, a query coroutine cancelled on Ctrl-C/SIGTERM (or by a host
    # cancelling ainvoke) exported its span as UNSET.
    exporter = tracing()

    async def query() -> None:
        with observability.span("kg_rag.query"):
            await asyncio.sleep(10)

    async def main() -> None:
        task = asyncio.create_task(query())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    span = _one(exporter, "kg_rag.query")
    assert span.status.status_code == StatusCode.ERROR
    assert _attrs(span)["error.type"] == "CancelledError"


@pytest.mark.parametrize("stop", [GeneratorExit, SystemExit])
def test_a_closed_generator_or_clean_exit_is_not_an_error(
    tracing: Tracing, stop
) -> None:
    exporter = tracing()
    with pytest.raises(stop):
        with observability.span("kg_rag.query"):
            raise stop() if stop is GeneratorExit else stop(0)
    assert _one(exporter, "kg_rag.query").status.status_code == StatusCode.UNSET


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


_MARKER = "corpus text: Vendor's secret price 4242"


def _exported_text(span: ReadableSpan) -> str:
    """Everything ``span`` exports: attributes, status, events."""
    return json.dumps(
        {
            "attributes": dict(span.attributes or {}),
            "status": span.status.description,
            "events": [
                {"name": e.name, "attributes": dict(e.attributes or {})}
                for e in span.events
            ],
        },
        default=str,
    )


def test_an_exception_reaches_a_span_by_its_type_not_its_text(
    tracing: Tracing,
) -> None:
    # Regression (rv24): the SDK recorded an escaping error's message, stack
    # and status description: a judge's OutputParserException quoting
    # corpus text reached the trace with capture_content off.
    exporter = tracing()
    with pytest.raises(ValueError):
        with observability.span("escaping"):
            raise ValueError(_MARKER)
    with observability.span("recorded") as handle:
        handle.record_exception(ValueError(_MARKER))
    for name in ("escaping", "recorded"):
        span = _one(exporter, name)
        assert _MARKER not in _exported_text(span), name
        (event,) = span.events
        assert event.attributes is not None
        assert event.attributes["exception.type"] == "ValueError"
        assert str(event.attributes["kg_rag.error_digest"]).startswith("ValueError (")
    assert _one(exporter, "escaping").status.description == "ValueError"


def test_capture_content_records_the_exception_text_redacted(
    tracing: Tracing,
) -> None:
    exporter = tracing(capture_content=True, redact_patterns=[r"price \d+"])
    with pytest.raises(ValueError):
        with observability.span("escaping"):
            raise ValueError(_MARKER)
    (event,) = _one(exporter, "escaping").events
    assert event.attributes is not None
    assert event.attributes["exception.message"] == (
        "corpus text: Vendor's secret [REDACTED]"
    )
    assert "4242" not in str(event.attributes["exception.stacktrace"])


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
    from unified_kg_rag.adapters.providers import Providers
    from unified_kg_rag.application.ingestion.pipeline_stages import PipelineStage

    class _Stage(PipelineStage):
        def __init__(self) -> None:  # no boto session
            self.config = Config()
            self.stage_type = stage_type
            # A factory without is_item_error: the generic classifier.
            self.providers = Providers(self.config, llm_factory=object())  # type: ignore[arg-type]

        def _execute_core(
            self, context: PipelineContext
        ) -> tuple[int, int, dict[str, Any] | None]:
            inputs, outputs = outcome()
            return inputs, outputs, metrics

    return _Stage()


def _failing_items_stage(error: Exception, *, known_good_failed: bool = False):
    """A graph extraction stage whose every model call fails with ``error``
    (through the batch processor, as the extractors call)."""
    from unified_kg_rag.adapters.llm import retry

    def _outcome() -> tuple[int, int]:
        def call(_: dict[str, int]) -> Any:
            raise error

        BatchProcessor(max_concurrency=2, max_attempts=1).execute_with_fallback(
            [1, 2, 3],
            lambda items: [{"x": i} for i in items],
            call,
            "t",
            show_progress=False,
        )
        if known_good_failed and (ledger := retry._LEDGER.get()) is not None:
            ledger.model_failed = True
        return 3, 0

    return _stage(PipelineStageType.GRAPH_EXTRACTION, _outcome)


def _empty_context() -> PipelineContext:
    return PipelineContext(
        pipeline_id="p-4",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=Path("."),
    )


def test_a_stage_whose_model_server_is_down_fails_naming_it(tracing: Tracing) -> None:
    # Regression (rv24): "Critical stage 'graph_extraction' processed 1
    # inputs but produced 0 outputs. Check the stage configuration", class
    # failure, although every call was refused by a server that was down.
    import httpx

    tracing()
    result = _failing_items_stage(
        httpx.ConnectError("All connection attempts failed")
    ).execute(_empty_context())
    assert result.status == PipelineStageStatus.FAILED
    assert (result.error_class, result.error_type) == (
        "model_unreachable",
        "ModelUnreachableError",
    )
    assert "A model server could not be reached" in (result.error_message or "")
    assert "Check the stage configuration" not in (result.error_message or "")


class _StatusError(Exception):
    """A provider error with an HTTP status (as openai's and httpx's)."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_a_stage_the_model_refuses_alike_says_so(tracing: Tracing) -> None:
    tracing()
    refused = _failing_items_stage(
        _StatusError("rejected: Vendor Corp secret text", 400),
        known_good_failed=True,
    ).execute(_empty_context())
    assert "a known-good request failed alike (_StatusError, HTTP 400)" in (
        refused.error_message or ""
    )
    # Named by its type and status, never its text (it may quote the input).
    assert "secret text" not in (refused.error_message or "")
    assert (refused.error_class, refused.error_type) == ("failure", "_StatusError")
    # An item's own error without a known-good request that failed: the
    # documents may be at fault, so the stage keeps its own error.
    own = _failing_items_stage(_StatusError("bad item", 400)).execute(_empty_context())
    assert "produced 0 outputs" in (own.error_message or "")
    assert own.error_type == "PipelineStageError"


@pytest.mark.parametrize(
    ("error", "named"),
    [
        (_StatusError("server error quoting the item", 500), "_StatusError, HTTP 500"),
        (_StatusError("slow down", 429), "_StatusError, HTTP 429"),
        (TimeoutError("read timed out"), "TimeoutError"),
    ],
)
def test_a_stage_whose_calls_failed_for_no_item_of_theirs_blames_the_model(
    tracing: Tracing, error: Exception, named: str
) -> None:
    # Regression (rv25): every call answered HTTP 500 (or was throttled past
    # its retries) and the stage said "produced 0 outputs. Check the stage
    # configuration and input data quality", sending the operator to the
    # documents: the known-good request is sent only for items' own errors.
    tracing()
    result = _failing_items_stage(error).execute(_empty_context())
    message = result.error_message or ""
    assert f"3 of its 3 failed model calls ended in {named}" in message
    assert "not of the documents" in message
    assert "produced 0 outputs" not in message
    assert "quoting the item" not in message
    assert result.error_type == type(error).__name__


def test_the_failure_ledger_counts_per_cause_and_keeps_the_first_of_each() -> None:
    # Old behaviour: every failed item's exception was kept, tracebacks
    # included, for the whole stage.
    from unified_kg_rag.adapters.llm.retry import FailureLedger

    ledger = FailureLedger()
    assert ledger.dominant() is None and ledger.failures == 0
    first = _StatusError("a", 500)
    ledger.record(first)
    ledger.record(ValueError("b"))
    for _ in range(3):
        ledger.record(_StatusError("c", 500))
    assert ledger.failures == 5
    assert ledger.dominant_count() == (first, 4)
    assert not hasattr(ledger, "errors")


def test_a_failed_batch_item_logs_its_error_digest_not_its_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Regression (rv25): a provider error quoting the item's text reached the
    # ERROR log of every item that failed.
    def call(_: dict[str, int]) -> Any:
        raise RuntimeError("cannot process: Vendor Corp confidential terms")

    with caplog.at_level("ERROR"):
        BatchProcessor(max_concurrency=1, max_attempts=1).execute_with_fallback(
            [1], lambda items: [{"x": i} for i in items], call, "t", show_progress=False
        )
    assert "Item failed" in caplog.text
    assert "confidential terms" not in caplog.text


@pytest.mark.parametrize("stop", [PipelineInterrupted(15), RuntimeError("boom")])
def test_a_stopped_or_crashed_ingestion_run_span_names_its_pipeline(
    tracing: Tracing, stop: BaseException
) -> None:
    # Before, kg_rag.pipeline_id was set only from a returned context, so a
    # stopped or crashed run's trace did not say which run to resume.
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline

    exporter = tracing()
    pipeline = object.__new__(DataIngestionPipeline)
    pipeline.config = Config()

    def _run(*_: Any) -> PipelineContext:
        pipeline.pipeline_id = "p-2"
        raise stop

    pipeline._run = _run  # type: ignore[method-assign]
    with pytest.raises(type(stop)):
        pipeline.run(".")
    run = _one(exporter, "kg_rag.ingestion.run")
    assert _attrs(run)["kg_rag.pipeline_id"] == "p-2"
    assert run.status.status_code == StatusCode.ERROR


def test_an_incremental_run_span_counts_its_delta(tracing: Tracing) -> None:
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
    from unified_kg_rag.domain.models import DocumentDelta

    exporter = tracing()
    context = PipelineContext(
        pipeline_id="p-3",
        config={},
        status=PipelineStageStatus.COMPLETED,
        start_time=datetime(2026, 1, 1),
        source_directory=Path("."),
    )
    context.incremental_delta = DocumentDelta(
        new=["a", "b"], changed=["c"], deleted=["d", "e", "f"]
    )
    pipeline = object.__new__(DataIngestionPipeline)
    pipeline.config = Config()
    pipeline.config.processing.document_parsing.index_value = "tenant-a"
    pipeline._run = lambda *_: context  # type: ignore[method-assign]
    pipeline.run(".")
    run = _attrs(_one(exporter, "kg_rag.ingestion.run"))
    assert (run["kg_rag.mode"], run["kg_rag.namespace"]) == ("incremental", "tenant-a")
    assert (
        run["kg_rag.delta.new"],
        run["kg_rag.delta.changed"],
        run["kg_rag.delta.deleted"],
    ) == (2, 1, 3)


def test_a_reused_pipeline_does_not_name_its_last_run(tracing: Tracing) -> None:
    # A run that fails before it resolves its id must not report the id of
    # the pipeline's previous run (the CLI resumes a stopped run by it).
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline

    exporter = tracing()
    pipeline = object.__new__(DataIngestionPipeline)
    pipeline.config = Config()
    ids = iter(["p-1", None])

    def _run(*_: Any) -> PipelineContext:
        if (resolved := next(ids)) is not None:
            pipeline.pipeline_id = resolved
        raise RuntimeError("boom")

    pipeline._run = _run  # type: ignore[method-assign]
    for _ in range(2):
        with pytest.raises(RuntimeError):
            pipeline.run(".")
    assert pipeline.pipeline_id is None
    first, second = sorted(
        (s for s in exporter.get_finished_spans() if s.name == "kg_rag.ingestion.run"),
        key=lambda s: s.start_time or 0,
    )
    assert _attrs(first)["kg_rag.pipeline_id"] == "p-1"
    assert "kg_rag.pipeline_id" not in _attrs(second)


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
    pipeline.config = Config()
    pipeline._run = _run  # type: ignore[method-assign]
    pipeline.run(".")

    run = _one(exporter, "kg_rag.ingestion.run")
    stages = {
        _attrs(s)["kg_rag.stage"]: s
        for s in _by_name(exporter)["kg_rag.ingestion.stage"]
    }
    assert _attrs(run)["kg_rag.pipeline_id"] == "p-1"
    assert _attrs(run)["kg_rag.stages_completed"] == 1
    # The namespace it writes, on every span of the run, and a full build.
    assert all(
        _attrs(span)["kg_rag.namespace"] == "default"
        for span in exporter.get_finished_spans()
    )
    assert _attrs(run)["kg_rag.mode"] == "full"
    assert "kg_rag.delta.new" not in _attrs(run)
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
    config.providers.rerank.kind = "none"
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


def _simple_chain(model_factory: Any = None) -> GraphRAGChain:
    config = Config()
    config.providers.rerank.kind = "none"
    chain = GraphRAGChain(
        config=config,
        model_factory=model_factory or _FakeModelFactory(),
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


class _RecordingSink:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict[str, float], dict[str, str]]] = []

    def emit(
        self,
        namespace: str,
        metrics: dict[str, float],
        dimensions: dict[str, str] | None = None,
    ) -> None:
        self.emitted.append((namespace, metrics, dict(dimensions or {})))


class _ContextRetriever(_FakeRetriever):
    """Records the log context its retrieval runs in."""

    seen: list[dict[str, Any]] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        from structlog.contextvars import get_contextvars

        _ContextRetriever.seen.append(get_contextvars())
        return await super().aretrieve(query)


async def test_a_query_is_tied_to_its_namespace_logs_trace_and_metrics(
    tracing: Tracing,
) -> None:
    # Regression (rv24): nothing named the namespace (the tenant boundary)
    # on the query trace, its logs or its metrics, the query id was in the
    # logs only, and only run-rag emitted query metrics (latency only).
    from unified_kg_rag.shared import metrics

    exporter = tracing()
    chain = _simple_chain()
    sink = _RecordingSink()
    chain.providers.metrics_sink = sink
    chain.providers._retriever_builders[RetrieverRole.DOCUMENT] = _ContextRetriever
    _ContextRetriever.seen.clear()
    try:
        output = await chain.ainvoke(_SIMPLE_QUERY)
    finally:
        await chain.aclose()
    query = _one(exporter, "kg_rag.query")
    assert all(
        _attrs(span)["kg_rag.namespace"] == "default"
        for span in exporter.get_finished_spans()
    )
    query_id = output.metadata["query_id"]
    assert len(query_id) == 12
    assert output.metadata["trace_id"] == format(query.context.trace_id, "032x")
    assert _ContextRetriever.seen
    assert all(
        (seen["query_id"], seen["namespace"]) == (query_id, "default")
        for seen in _ContextRetriever.seen
    )
    ((namespace, values, dimensions),) = sink.emitted
    assert namespace == metrics.QUERY_METRICS_NAMESPACE
    assert values[metrics.QUERY_LATENCY_METRIC] > 0
    assert [
        values[name]
        for name in (
            metrics.QUERY_FAILED_METRIC,
            metrics.QUERY_DEGRADED_METRIC,
            metrics.QUERY_ABSTAINED_METRIC,
        )
    ] == [0.0, 0.0, 0.0]
    assert values[metrics.QUERY_THROTTLED_METRIC] == 0.0
    # The namespace by the key the ingestion records use (one spelling).
    assert dimensions == {"Strategy": "simple", "namespace": "default"}


async def test_a_failed_query_is_counted_and_a_search_is_not(
    tracing: Tracing,
) -> None:
    from unified_kg_rag.shared import metrics

    class _Down(GenericFakeChatModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("model down")

        async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("model down")

    class _Failing(_FakeModelFactory):
        def get_model(self, model_id: Any, **kwargs: Any) -> Any:
            return _Down(messages=iter(()))

    tracing()
    chain = _simple_chain(model_factory=_Failing())
    sink = _RecordingSink()
    chain.providers.metrics_sink = sink
    try:
        with pytest.raises(RuntimeError, match="model down"):
            await chain.ainvoke(_SIMPLE_QUERY)
        await chain.asearch(_SIMPLE_QUERY)
    finally:
        await chain.aclose()
    ((_, values, dimensions),) = sink.emitted
    assert values[metrics.QUERY_FAILED_METRIC] == 1.0
    assert dimensions["Strategy"] == "simple"
    assert dimensions["error_class"] == "failure"


class _DownChat(GenericFakeChatModel):
    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("model down")

    async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("model down")

    async def _astream(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("model down")
        yield  # pragma: no cover - an async generator


class _DownFactory(_FakeModelFactory):
    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        return _DownChat(messages=iter(()))


async def test_a_raised_query_failure_carries_its_query_and_trace_ids(
    tracing: Tracing,
) -> None:
    # Old behaviour: only an answer named its query_id and trace_id, so the
    # failure that most needed its log lines and trace named neither, on
    # any surface (run-rag's failure document, MCP tool errors, the console).
    from unified_kg_rag.application.cli.conventions import failure_fields
    from unified_kg_rag.application.cli.mcp_tools import describe_error

    exporter = tracing()
    chain = _simple_chain(model_factory=_DownFactory())
    try:
        with pytest.raises(RuntimeError, match="model down") as raised:
            await chain.ainvoke(_SIMPLE_QUERY)
    finally:
        await chain.aclose()
    query = _one(exporter, "kg_rag.query")
    error = raised.value
    trace_id = format(query.context.trace_id, "032x")
    assert (error.query_id, error.trace_id) == (  # type: ignore[attr-defined]
        error.query_id,  # type: ignore[attr-defined]
        trace_id,
    )
    assert len(error.query_id) == 12  # type: ignore[attr-defined]
    fields = failure_fields(1, error)
    assert (fields["query_id"], fields["trace_id"]) == (error.query_id, trace_id)  # type: ignore[attr-defined]
    message, expected = describe_error(error)
    assert not expected and f"trace_id: {trace_id}" in message
    assert "model down" not in message


async def test_a_streamed_query_names_its_namespace_and_publishes_its_metrics(
    tracing: Tracing,
) -> None:
    # Regression (rv25): stream/astream bound no namespace on their spans
    # and published no query metrics, so a host serving streamed answers
    # got neither tenant filtering nor alarms.
    from unified_kg_rag.shared import metrics

    exporter = tracing()
    # Its answer's model span too, which ends after the query span.
    chain = _simple_chain(_TracedModelFactory())
    sink = _RecordingSink()
    chain.providers.metrics_sink = sink
    try:
        chunks = [chunk async for chunk in chain.astream(_SIMPLE_QUERY)]
    finally:
        await chain.aclose()
    assert "".join(chunks)
    _one(exporter, "chat vendor.model-v1")
    spans = exporter.get_finished_spans()
    assert spans and all(
        _attrs(span).get("kg_rag.namespace") == "default" for span in spans
    ), [(s.name, _attrs(s).get("kg_rag.namespace")) for s in spans]
    ((namespace, values, properties),) = sink.emitted
    assert namespace == metrics.QUERY_METRICS_NAMESPACE
    assert values[metrics.QUERY_LATENCY_METRIC] > 0
    assert values[metrics.QUERY_FAILED_METRIC] == 0.0
    assert properties == {"Strategy": "simple", "namespace": "default"}

    # A failed stream counts, and its error carries the ids.
    failing = _simple_chain(model_factory=_DownFactory())
    sink = _RecordingSink()
    failing.providers.metrics_sink = sink
    try:
        with pytest.raises(Exception, match="model down") as raised:
            async for _ in failing.astream(_SIMPLE_QUERY):
                pass
    finally:
        await failing.aclose()
    ((_, values, properties),) = sink.emitted
    assert values[metrics.QUERY_FAILED_METRIC] == 1.0
    assert properties["error_class"] == "failure"
    assert len(raised.value.query_id) == 12  # type: ignore[attr-defined]

    # A stream the caller stops reading early is no failure.
    chain = _simple_chain()
    sink = _RecordingSink()
    chain.providers.metrics_sink = sink
    try:
        stream = chain.astream(_SIMPLE_QUERY)
        await anext(stream)
        await stream.aclose()
    finally:
        await chain.aclose()
    ((_, values, _),) = sink.emitted
    assert values[metrics.QUERY_FAILED_METRIC] == 0.0


class _DegradingRetriever(_FakeRetriever):
    """Answers, recording a step it answered without (as a failed rerank)."""

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        from unified_kg_rag.adapters.retrieval.base import record_degradation
        from unified_kg_rag.domain.retrieval.degradation import DegradationStep

        record_degradation(DegradationStep.RETRIEVAL, "reranker down", "Down")
        return await super().aretrieve(query)


def _throttled_request() -> None:
    from unified_kg_rag.shared.usage import record_request

    record_request("query", "vendor.model-v1", throttled=True)


class _ThrottledChat(GenericFakeChatModel):
    """Answers, streamed or not, after one throttled request (a 429)."""

    async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
        _throttled_request()
        return await super()._agenerate(*args, **kwargs)

    async def _astream(self, *args: Any, **kwargs: Any) -> Any:
        _throttled_request()
        async for chunk in super()._astream(*args, **kwargs):
            yield chunk


class _AbstainingFactory(_FakeModelFactory):
    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        return _ThrottledChat(
            messages=itertools.repeat(AIMessage(content="[NO_ANSWER]"))
        )


@pytest.mark.parametrize("streamed", [False, True], ids=["invoke", "stream"])
async def test_a_streamed_query_counts_degraded_abstained_and_throttled_as_invoke(
    streamed: bool,
) -> None:
    # Regression (rv26): astream published query_degraded=0 and
    # query_abstained=0 whatever its answer was, and never
    # query_throttled_requests, so streamed answers were missing from the
    # "Query outcomes" widget's degraded, abstained and throttled series.
    from unified_kg_rag.shared import metrics

    chain = _simple_chain(_AbstainingFactory())
    chain.config.search.abstention.mode = "mark"
    chain.providers._retriever_builders[RetrieverRole.DOCUMENT] = _DegradingRetriever
    sink = _RecordingSink()
    chain.providers.metrics_sink = sink
    try:
        if streamed:
            async for _ in chain.astream(_SIMPLE_QUERY):
                pass
        else:
            await chain.ainvoke(_SIMPLE_QUERY)
    finally:
        await chain.aclose()
    ((_, values, properties),) = sink.emitted
    assert [
        values.get(name)
        for name in (
            metrics.QUERY_FAILED_METRIC,
            metrics.QUERY_DEGRADED_METRIC,
            metrics.QUERY_ABSTAINED_METRIC,
            metrics.QUERY_THROTTLED_METRIC,
        )
    ] == [0.0, 1.0, 1.0, 1.0]
    assert properties == {"Strategy": "simple", "namespace": "default"}


@pytest.mark.parametrize("streamed", [False, True])
async def test_a_step_outside_the_strategy_degrades_a_streamed_query_too(
    streamed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pins rv27 quality o2: a degradation only the query's ledger holds (a
    # chain step, not the strategy's result metadata) reached a streamed
    # answer's metrics only through the ledger's metadata.
    from unified_kg_rag.shared import metrics

    chain = _simple_chain()
    chain.ignore_errors = True

    def unbudgeted(state: dict[str, Any]) -> Any:
        raise RuntimeError("context optimizer down")

    monkeypatch.setattr(chain, "_optimize_context", unbudgeted)
    sink = _RecordingSink()
    chain.providers.metrics_sink = sink
    try:
        if streamed:
            async for _ in chain.astream(_SIMPLE_QUERY):
                pass
        else:
            await chain.ainvoke(_SIMPLE_QUERY)
    finally:
        await chain.aclose()
    ((_, values, _),) = sink.emitted
    assert values[metrics.QUERY_FAILED_METRIC] == 0.0
    assert values[metrics.QUERY_DEGRADED_METRIC] == 1.0


async def test_admission_failures_count_but_a_callers_error_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Old behaviour: a failure in the query's admission (a build-record read,
    # the deadline) published no metric, while a caller's mistake (a typo'd
    # namespace, an empty question) counted in query_failed and paged.
    from unified_kg_rag.application.retrieval import methodology_check
    from unified_kg_rag.shared import StoreConnectionError, metrics
    from unified_kg_rag.shared.exceptions import InvalidArgumentError

    chain = _simple_chain()
    sink = _RecordingSink()
    chain.providers.metrics_sink = sink

    async def unreadable(*_: Any, **__: Any) -> Any:
        raise StoreConnectionError("build record unreadable")

    try:
        with pytest.raises(InvalidArgumentError):
            await chain.ainvoke({"query": "  "})
        with monkeypatch.context() as patch:
            patch.setattr(methodology_check, "check_strategy_holdings", unreadable)
            with pytest.raises(StoreConnectionError) as raised:
                await chain.ainvoke(_SIMPLE_QUERY)
    finally:
        await chain.aclose()
    (_, refused, refused_properties), (_, failed, failed_properties) = sink.emitted
    assert refused[metrics.QUERY_FAILED_METRIC] == 0.0
    assert refused_properties["error_class"] == "invalid_argument"
    assert failed[metrics.QUERY_FAILED_METRIC] == 1.0
    assert failed_properties == {
        "Strategy": "simple",
        "namespace": "default",
        "error_class": "store_unreachable",
    }
    assert len(raised.value.query_id) == 12  # type: ignore[attr-defined]


async def test_a_refused_inputs_metrics_keep_its_error_and_one_strategy_series() -> (
    None
):
    # Old behaviour: a refused dict input put its raw search_strategy string
    # into the Strategy dimension (a new billed series per string), and an
    # input that was neither a dict nor a RAGInput made the metrics' own
    # AttributeError replace the caller's error.
    from unified_kg_rag.shared import metrics
    from unified_kg_rag.shared.exceptions import InvalidArgumentError

    chain = _simple_chain()
    sink = _RecordingSink()
    chain.providers.metrics_sink = sink
    try:
        with pytest.raises(InvalidArgumentError):
            await chain.ainvoke({"query": "Vendor?", "search_strategy": "made-up"})
        with pytest.raises(AttributeError) as odd:
            await chain.ainvoke("Vendor?")  # type: ignore[arg-type]
    finally:
        await chain.aclose()
    # The admission's own error, not the metrics' reading of the input.
    assert "search_strategy" not in str(odd.value)
    (_, first, first_properties), (_, second, second_properties) = sink.emitted
    assert first_properties["Strategy"] == "invalid"
    assert second_properties["Strategy"] == "auto"
    assert first[metrics.QUERY_FAILED_METRIC] == 0.0


async def test_a_model_clients_own_error_reaches_the_caller_as_a_model_error() -> None:
    from unified_kg_rag.shared.exceptions import LanguageModelError

    class ResponseError(Exception):  # Ollama's: an unknown model
        status_code = 404

    class _Refused(GenericFakeChatModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            raise ResponseError("model 'x' not found")

        async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
            raise ResponseError("model 'x' not found")

    class _Factory(_FakeModelFactory):
        def get_model(self, model_id: Any, **kwargs: Any) -> Any:
            return _Refused(messages=iter(()))

    chain = _simple_chain(model_factory=_Factory())
    try:
        with pytest.raises(LanguageModelError, match="ResponseError") as raised:
            await chain.ainvoke(_SIMPLE_QUERY)
    finally:
        await chain.aclose()
    assert isinstance(raised.value.__cause__, ResponseError)


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


class _TracedModelFactory(_FakeModelFactory):
    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        return GenericFakeChatModel(
            messages=itertools.repeat(AIMessage(content="Vendor ships 120 units.")),
            callbacks=[GenAITracingCallbackHandler("vendor.model-v1")],
        )


def _traced_chain() -> GraphRAGChain:
    return _simple_chain(_TracedModelFactory())


def _assert_streamed_model_span_ended_ok(exporter: InMemorySpanExporter) -> None:
    span = _one(exporter, "chat vendor.model-v1")
    attrs = _attrs(span)
    assert span.status.status_code != StatusCode.ERROR
    assert "error.type" not in attrs
    assert json.loads(attrs["gen_ai.output.messages"])[0]["parts"][0]["content"] == (
        "Vendor ships 120 units."
    )


@pytest.mark.parametrize("per_task", [False, True], ids=["astream", "anext_per_task"])
async def test_a_streamed_answer_ends_its_model_span_ok(
    tracing: Tracing, per_task: bool
) -> None:
    # Before (#386), the task that started the model call ended before the
    # stream did, and its done-callback exported the span as ERROR
    # NoEndCallback, without the answer.
    exporter = tracing(capture_content=True)
    chain = _traced_chain()
    try:
        stream = chain.astream(_SIMPLE_QUERY).__aiter__()
        chunks = []
        while True:
            try:
                step = stream.__anext__()
                chunks.append(
                    await (asyncio.create_task(step) if per_task else step)  # type: ignore[arg-type]
                )
            except StopAsyncIteration:
                break
    finally:
        await chain.aclose()
    assert "".join(chunks) == "Vendor ships 120 units."
    _assert_streamed_model_span_ended_ok(exporter)


def test_a_sync_streamed_answer_ends_its_model_span_ok(tracing: Tracing) -> None:
    exporter = tracing(capture_content=True)
    chain = _traced_chain()
    try:
        assert "".join(chain.stream(_SIMPLE_QUERY)) == "Vendor ships 120 units."
    finally:
        chain.close()
    _assert_streamed_model_span_ended_ok(exporter)


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
    config.providers.rerank.kind = "none"
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
    config.providers.rerank.kind = "none"
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


def test_a_cancelled_model_call_ends_its_span(tracing: Tracing) -> None:
    # LangChain calls no end callback for a cancelled ainvoke: before, the
    # span stayed open in the handler (never exported, _spans kept growing).
    exporter = tracing()
    handler = GenAITracingCallbackHandler("vendor.model-v1")
    finished = uuid4()

    async def call(run_id) -> None:
        handler.on_chat_model_start({}, [_PROMPT], run_id=run_id)
        await asyncio.sleep(10)

    async def main() -> None:
        # A call that ended normally is not touched when its task ends.
        handler.on_chat_model_start({}, [_PROMPT], run_id=finished)
        handler.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=AIMessage(content="a"))]]),
            run_id=finished,
        )
        task = asyncio.create_task(call(uuid4()))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    statuses = sorted(
        (span.status.status_code.name, _attrs(span).get("error.type"))
        for span in exporter.get_finished_spans()
    )
    assert statuses == [("ERROR", "CancelledError"), ("UNSET", None)]
    assert handler._spans == {} and handler._watches == {}


class _SlowChatModel(GenericFakeChatModel):
    async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(10)
        return await super()._agenerate(*args, **kwargs)


@pytest.mark.parametrize(
    "deadline",
    ["wait_for", "timeout"],
)
async def test_a_timed_out_model_call_ends_its_span(
    tracing: Tracing, deadline: str
) -> None:
    # A timeout cancels the call but not its task, which finishes normally
    # (how RAGAS applies ragas_timeout). Before, only a cancelled task ended
    # the span: a timed-out call's span was never exported and stayed in
    # _spans.
    exporter = tracing()
    handler = GenAITracingCallbackHandler("vendor.model-v1")
    model = _SlowChatModel(
        messages=itertools.repeat(AIMessage(content="a")), callbacks=[handler]
    )

    async def judge() -> str:
        try:
            if deadline == "wait_for":
                await asyncio.wait_for(model.ainvoke(_PROMPT), 0.05)
            else:
                async with asyncio.timeout(0.05):
                    await model.ainvoke(_PROMPT)
        except TimeoutError:
            return "timed out"
        return "answered"

    assert await asyncio.create_task(judge()) == "timed out"
    await asyncio.sleep(0)  # let the task's done-callbacks run
    span = _one(exporter, "chat vendor.model-v1")
    assert span.status.status_code == StatusCode.ERROR
    assert _attrs(span)["error.type"] == "CancelledError"
    assert handler._spans == {} and handler._watches == {} and not handler._streamed


class _StreamingInsideChatModel(GenericFakeChatModel):
    """A model whose ``_agenerate`` streams (as ChatOllama's does): an
    ``ainvoke`` reports tokens, then stalls."""

    async def _agenerate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Any:
        assert run_manager is not None
        await run_manager.on_llm_new_token("a")
        await asyncio.sleep(10)
        return await super()._agenerate(messages, stop, run_manager, **kwargs)


async def test_a_timed_out_call_that_streams_internally_ends_its_span(
    tracing: Tracing,
) -> None:
    # ChatOllama streams inside ainvoke: before, its first token marked the
    # call as streamed, so a judge call timed out after it kept its span open.
    exporter = tracing()
    handler = GenAITracingCallbackHandler("vendor.model-v1")
    model = _StreamingInsideChatModel(
        messages=itertools.repeat(AIMessage(content="a")), callbacks=[handler]
    )

    async def judge() -> str:
        try:
            await asyncio.wait_for(model.ainvoke(_PROMPT), 0.05)
        except TimeoutError:
            return "timed out"
        return "answered"

    assert await asyncio.create_task(judge()) == "timed out"
    await asyncio.sleep(0)  # let the task's done-callbacks run
    span = _one(exporter, "chat vendor.model-v1")
    assert _attrs(span)["error.type"] == "CancelledError"
    assert handler._spans == {} and handler._watches == {} and not handler._streamed


async def test_a_stream_started_in_one_task_ends_ok_in_another(
    tracing: Tracing,
) -> None:
    # The task that pulled the first token ends normally before the stream
    # does: the span must stay open for the consumer, not end as an error.
    exporter = tracing(capture_content=True)
    handler = GenAITracingCallbackHandler("vendor.model-v1")
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="Vendor ships parts in two lots.")]),
        callbacks=[handler],
    )
    stream = model.astream(_PROMPT).__aiter__()
    first = await asyncio.create_task(stream.__anext__())  # type: ignore[arg-type]
    await asyncio.sleep(0)
    assert exporter.get_finished_spans() == () and len(handler._spans) == 1

    async def consume() -> str:
        return "".join([str(chunk.content) async for chunk in stream])

    assert str(first.content) + await asyncio.create_task(consume()) == (
        "Vendor ships parts in two lots."
    )
    span = _one(exporter, "chat vendor.model-v1")
    assert span.status.status_code != StatusCode.ERROR
    assert "error.type" not in _attrs(span)
    assert json.loads(_attrs(span)["gen_ai.output.messages"])[0]["parts"][0][
        "content"
    ] == ("Vendor ships parts in two lots.")
    assert handler._spans == {} and handler._watches == {} and not handler._streamed


def test_invoke_model_usage_fallback(tracing: Tracing) -> None:
    exporter = tracing()
    handler = GenAITracingCallbackHandler("vendor.model-v1")
    run_id = uuid4()
    handler.on_chat_model_start({}, [_PROMPT], run_id=run_id)
    handler.on_llm_end(
        LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="Blocked."))]],
            llm_output={"usage": {"prompt_tokens": 11, "completion_tokens": 2}},
        ),
        run_id=run_id,
    )
    attrs = _attrs(_one(exporter, "chat vendor.model-v1"))
    # No model call carries a guardrail (it screens the query and answer).
    assert "aws.bedrock.guardrail.intervened" not in attrs
    assert (
        attrs["gen_ai.usage.input_tokens"],
        attrs["gen_ai.usage.output_tokens"],
    ) == (11, 2)


# --- evaluation -----------------------------------------------------------------


async def test_judge_calls_nest_under_an_evaluate_span(
    tracing: Tracing, mocker
) -> None:
    # run-eval's judge model calls each started a root trace of their own.
    exporter = tracing()
    config = Config()
    config.evaluation.enabled_evaluators = [EvaluatorType.GRAPH_AWARE]
    manager = EvaluationManager(config, rag_chain=object())

    async def judge(*_args: Any) -> list:
        async def call() -> None:
            with observability.span("chat judge"):
                await asyncio.sleep(0)

        await asyncio.gather(call(), call())
        return []

    mocker.patch.object(GraphAwareEvaluator, "aevaluate_batch", side_effect=judge)
    await manager._evaluate_results(
        [EvaluationQuery(query_id="q1", question="?")],
        [
            EvaluationResult(
                query_id="q1", question="?", generated_answer="a", ground_truth=""
            )
        ],
        [EvaluationGroundTruth(query_id="q1", ground_truth="a")],
    )
    evaluate = _one(exporter, "kg_rag.evaluate")
    assert _attrs(evaluate)["kg_rag.evaluator"] == "graph_aware"
    assert _attrs(evaluate)["kg_rag.items"] == 1
    calls = _by_name(exporter)["chat judge"]
    assert len(calls) == 2 and all(_is_child(c, evaluate) for c in calls)


def test_bound_attributes_tag_every_span_and_nest(tracing) -> None:
    exporter = tracing()
    with observability.bind_attributes(**{"eval.strategy": "mix"}):
        with observability.bind_identity(session_id="s1"):
            with observability.span("inner", strategy="local"):
                pass
        with observability.span("outer"):
            pass
    with observability.span("after"):
        pass
    inner, outer, after = (
        _attrs(_one(exporter, n)) for n in ("inner", "outer", "after")
    )
    assert inner["kg_rag.eval.strategy"] == "mix" and inner["session.id"] == "s1"
    assert inner["kg_rag.strategy"] == "local"
    assert outer["kg_rag.eval.strategy"] == "mix" and "session.id" not in outer
    assert "kg_rag.eval.strategy" not in after


@pytest.mark.parametrize("order", ["sequential", "interleaved"])
async def test_run_eval_root_spans_name_the_run_strategy_and_repeat(
    tracing, mocker, tmp_path, order: str
) -> None:
    # The kg_rag.batch and kg_rag.evaluate roots of a comparison's runs had
    # no strategy or repeat, so the runs' traces could not be told apart.
    import json

    from unified_kg_rag.application.cli import run_evaluation
    from unified_kg_rag.domain.models import EvaluatorType

    exporter = tracing()
    config = Config()
    config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
    mocker.patch.object(run_evaluation, "display_ascii_art")
    data = tmp_path / "eval.json"
    data.write_text(json.dumps([{"question": "Who ships?", "answer": "Vendor"}]))

    class _Chain:
        async def ainvoke(self, inputs, config=None):
            with observability.span("probe"):
                meta = {"search_strategy": inputs["search_strategy"]}
                return {"answer": "Vendor", "sources": [], "metadata": meta}

    args = run_evaluation.CommandLineInterface._setup_arguments().parse_args(
        [
            "--data-path",
            str(data),
            "--output-directory",
            str(tmp_path / "out"),
            "--search-strategy",
            "local,mix",
            "--repeats",
            "2",
            "--run-order",
            order,
        ]
    )
    assert await run_evaluation.EvaluationRunner(args, _Chain(), config).run() == 0
    runs = [("local", 1), ("local", 2), ("mix", 1), ("mix", 2)]
    spans = _by_name(exporter)

    def tagged(name: str) -> list[tuple[str, int]]:
        return sorted(
            (_attrs(s)["kg_rag.eval.strategy"], _attrs(s)["kg_rag.eval.repeat"])
            for s in spans[name]
            if "kg_rag.eval.strategy" in _attrs(s)
        )

    # Interleaved, one batch answers every run: each answer's spans carry
    # its run (the warm-up queries, one per strategy, carry none).
    roots = ["kg_rag.evaluate"]
    if order == "sequential":
        roots.append("kg_rag.batch")
    else:
        (batch,) = spans["kg_rag.batch"]
        assert batch.parent is None and _attrs(batch)["kg_rag.items"] == 4
    for name in roots:
        assert all(s.parent is None for s in spans[name]), name
        assert tagged(name) == runs, name
    assert tagged("probe") == runs


def test_an_unsampled_span_gives_no_trace_ids() -> None:
    # Old behaviour: with observability.sample_rate below 1 an unsampled
    # query still returned its trace_id, so metadata.trace_id and the MCP
    # and console links named a trace that was never exported.
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

    provider = TracerProvider(sampler=ALWAYS_OFF)
    configure_observability(ObservabilityConfig(enabled=True), tracer_provider=provider)
    try:
        with observability.span("kg_rag.query"):
            assert observability.current_trace_ids() is None
    finally:
        observability.uninstall()


def test_a_sampled_span_gives_its_trace_ids(tracing: Tracing) -> None:
    tracing()
    with observability.span("kg_rag.query"):
        ids = observability.current_trace_ids()
    assert ids is not None and len(ids[0]) == 32


async def test_an_mcp_call_past_its_deadline_in_the_slot_queue_is_a_counted_query(
    tracing: Tracing, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression (rv27): a kg_answer that timed out waiting for a call slot
    # recorded no query metric and carried no id, so the likeliest timeout of
    # an overloaded server never reached QueryFailures or the latency.
    import asyncio

    from unified_kg_rag.application.cli.mcp_tools import KnowledgeGraphTools
    from unified_kg_rag.domain.models.config import McpServerConfig
    from unified_kg_rag.shared import metrics
    from unified_kg_rag.shared.exceptions import QueryTimeoutError, query_failure_ids

    tracing()
    chain = _simple_chain()
    chain.config.search.query_timeout_seconds = 0.3
    sink = _RecordingSink()
    chain.providers.metrics_sink = sink
    real_screen = GraphRAGChain._screen_input

    async def slow_screen(self: GraphRAGChain, *args: Any, **kwargs: Any) -> Any:
        # The first call holds its slot past the others' deadline: it
        # finishes a step that does not stop at the cancellation first.
        until = asyncio.get_running_loop().time() + 0.6
        try:
            await asyncio.sleep(0.6)
        except asyncio.CancelledError:
            await asyncio.sleep(until - asyncio.get_running_loop().time())
            raise
        return await real_screen(self, *args, **kwargs)

    monkeypatch.setattr(GraphRAGChain, "_screen_input", slow_screen)
    tools = KnowledgeGraphTools(chain, settings=McpServerConfig(max_concurrent_calls=1))
    try:
        errors = await asyncio.gather(
            *(
                tools.answer("가젯을 보관하는 곳은?", strategy="simple")
                for _ in range(3)
            ),
            return_exceptions=True,
        )
    finally:
        await chain.aclose()
    assert all(isinstance(e, QueryTimeoutError) for e in errors), errors
    # One timed out in the chain, two waiting for the slot: three queries.
    assert sum("slot" in str(e) for e in errors) == 2
    assert len(sink.emitted) == 3
    for _, values, properties in sink.emitted:
        assert values[metrics.QUERY_FAILED_METRIC] == 1.0
        assert properties["error_class"] == "timeout"
        assert properties["Strategy"] == "simple"
    ids = [query_failure_ids(e) for e in errors]
    assert all(len(i.get("query_id", "")) == 12 and i.get("trace_id") for i in ids)
    assert len({i["query_id"] for i in ids}) == 3
