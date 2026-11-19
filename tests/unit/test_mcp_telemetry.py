# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MCP tools' ``kg_rag.mcp.tool`` span: its arguments only under
``observability.capture_content``, redacted (needs the OpenTelemetry SDK of
the dev group)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from tests.unit.test_mcp_tools import _file_names, _tools  # noqa: F401
from unified_kg_rag.application.observability import configure_observability
from unified_kg_rag.domain.models import ObservabilityConfig
from unified_kg_rag.shared import observability

pytestmark = pytest.mark.unit


async def test_the_tool_span_captures_arguments_only_when_allowed() -> None:
    def spans(**settings: Any) -> list[dict[str, Any]]:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        configure_observability(
            ObservabilityConfig(enabled=True, **settings), tracer_provider=provider
        )
        return exporter  # type: ignore[return-value]

    try:
        exporter = spans()
        await _tools().search("Where is account 1234 stored?")
        (span,) = [s for s in exporter.get_finished_spans() if s.name == "kg_rag.mcp.tool"]  # type: ignore[attr-defined]
        attributes = dict(span.attributes or {})
        assert attributes["mcp.tool.name"] == "kg_search"
        assert "input.value" not in attributes
        observability.uninstall()

        exporter = spans(capture_content=True, redact_patterns=[r"\d{4}"])
        await _tools().search("Where is account 1234 stored?")
        (span,) = [s for s in exporter.get_finished_spans() if s.name == "kg_rag.mcp.tool"]  # type: ignore[attr-defined]
        captured = json.loads(dict(span.attributes or {})["input.value"])
        assert captured["query"] == "Where is account [REDACTED] stored?"
    finally:
        observability.uninstall()


async def test_the_tool_span_names_the_namespace_it_reads() -> None:
    from unified_kg_rag.shared.exceptions import InvalidArgumentError

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    configure_observability(ObservabilityConfig(enabled=True), tracer_provider=provider)
    try:
        tools = _tools(allowed_namespaces=["tenant-b"])
        await tools.search("q")
        await tools.search("q", namespace="tenant-b")
        with pytest.raises(InvalidArgumentError):
            await tools.search("q", namespace="unserved")
        names = [
            dict(s.attributes or {}).get("kg_rag.namespace")
            for s in exporter.get_finished_spans()
            if s.name == "kg_rag.mcp.tool"
        ]
        # A namespace the server does not serve is not recorded.
        assert names == [tools.namespace, "tenant-b", None]
    finally:
        observability.uninstall()


async def test_a_cancelled_call_is_logged_and_spanned_as_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # rv53 low: a client that left was logged as "failed after ...
    # (CancelledError)" and its span ended with the ERROR status.
    import asyncio

    from opentelemetry.trace import StatusCode

    from tests.unit.test_mcp_tools import FakeChain
    from unified_kg_rag.application.cli import mcp_tools
    from unified_kg_rag.application.retrieval.rag_chain import RAGInput, SearchOutput

    lines: list[str] = []
    monkeypatch.setattr(
        mcp_tools.logger, "info", lambda message, *args: lines.append(message % args)
    )
    started = asyncio.Event()

    class SlowChain(FakeChain):
        async def asearch(self, rag_input: RAGInput) -> SearchOutput:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("not reached")

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    configure_observability(ObservabilityConfig(enabled=True), tracer_provider=provider)
    try:
        call = asyncio.create_task(_tools(SlowChain()).search("q"))
        await started.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        (span,) = [
            s for s in exporter.get_finished_spans() if s.name == "kg_rag.mcp.tool"
        ]
    finally:
        observability.uninstall()
    assert span.status.status_code is not StatusCode.ERROR
    attributes = dict(span.attributes or {})
    assert attributes["kg_rag.cancelled"] is True
    assert "error.type" not in attributes
    (line,) = [line for line in lines if "MCP tool kg_search" in line]
    assert "cancelled after" in line and "failed" not in line


async def test_a_cancelled_error_its_task_was_not_cancelled_for_is_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # rv54 low: any CancelledError out of the call was taken for a client
    # that left. One the call raised itself (it awaited a future cancelled
    # elsewhere) while its own task is not being cancelled is a failure.
    import asyncio

    from opentelemetry.trace import StatusCode

    from tests.unit.test_mcp_tools import FakeChain
    from unified_kg_rag.application.cli import mcp_tools
    from unified_kg_rag.application.retrieval.rag_chain import RAGInput, SearchOutput

    lines: list[str] = []
    monkeypatch.setattr(
        mcp_tools.logger, "info", lambda message, *args: lines.append(message % args)
    )

    class CancelledInside(FakeChain):
        async def asearch(self, rag_input: RAGInput) -> SearchOutput:
            future = asyncio.get_running_loop().create_future()
            future.cancel()
            await future
            raise AssertionError("not reached")

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    configure_observability(ObservabilityConfig(enabled=True), tracer_provider=provider)
    try:
        with pytest.raises(asyncio.CancelledError):
            await _tools(CancelledInside()).search("q")
        (span,) = [
            s for s in exporter.get_finished_spans() if s.name == "kg_rag.mcp.tool"
        ]
    finally:
        observability.uninstall()
    assert span.status.status_code is StatusCode.ERROR
    assert "kg_rag.cancelled" not in dict(span.attributes or {})
    (line,) = [line for line in lines if "MCP tool kg_search" in line]
    assert "failed after" in line and "(CancelledError)" in line
