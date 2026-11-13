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
