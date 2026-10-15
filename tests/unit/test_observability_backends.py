# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exporting tracing backends: otlp, langfuse and phoenix (#92 P2).

No network: endpoint and header resolution is a pure function over an
environment mapping; the exporter is either the SDK's in-memory one (passed as
``span_exporter``, which keeps the backend's sampler, batching, vendor mapping
and redaction) or a fake OTLP exporter module that records its constructor
arguments. Covers backend selection, env-only credentials that never reach a
log line, session/user ids, the vendor attribute mappings, redaction of every
exported string, parent-based sampling and the bounded export timeouts.
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import types
from collections.abc import Iterator
from typing import Any
from unittest import mock

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from unified_kg_rag.adapters.observability import otel
from unified_kg_rag.adapters.observability.endpoints import (
    LANGFUSE_OTLP_PATH,
    PHOENIX_DEFAULT_GRPC_ENDPOINT,
    PHOENIX_DEFAULT_HTTP_ENDPOINT,
    resolve_export_settings,
    safe_url,
)
from unified_kg_rag.adapters.observability.span_transforms import (
    Redactor,
    langfuse_attributes,
    openinference_attributes,
)
from unified_kg_rag.domain.models import ObservabilityBackend, ObservabilityConfig
from unified_kg_rag.shared import ConfigurationError, observability

pytestmark = pytest.mark.unit

_PUBLIC = "pk-lf-0000-test-public"
_SECRET = "sk-lf-0000-test-secret"
_PHOENIX_KEY = "phx-test-api-key"
_LANGFUSE_ENV = {
    "LANGFUSE_HOST": "http://localhost:3000/",
    "LANGFUSE_PUBLIC_KEY": _PUBLIC,
    "LANGFUSE_SECRET_KEY": _SECRET,
}


def _config(backend: str, **settings: Any) -> ObservabilityConfig:
    return ObservabilityConfig(enabled=True, backend=backend, **settings)


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    yield
    observability.uninstall()


@pytest.fixture
def no_global_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if a backend built with an injected exporter goes global."""

    def _refuse(provider: Any) -> None:
        raise AssertionError("set_tracer_provider must not be called")

    monkeypatch.setattr(otel.trace, "set_tracer_provider", _refuse)


def _with_env(env: dict[str, str], build: Any, *args: Any, **kwargs: Any) -> Any:
    """``build(*args, **kwargs)`` with exactly ``env`` as the environment."""
    with mock.patch.dict(os.environ, env, clear=True):
        return build(*args, **kwargs)


def _install(
    backend: str, environ: dict[str, str] | None = None, **settings: Any
) -> tuple[InMemorySpanExporter, otel.OTelTracerBackend]:
    exporter = InMemorySpanExporter()
    config = _config(backend, **settings)
    tracer = _with_env(
        _LANGFUSE_ENV if environ is None else environ,
        otel.build_backend,
        config,
        span_exporter=exporter,
    )
    observability.install(
        tracer,
        capture_content=config.capture_content,
        redact_patterns=config.redact_patterns,
    )
    return exporter, tracer


def _flushed(exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    observability.uninstall()  # shuts the provider down, flushing the batch
    return {span.name: span for span in exporter.get_finished_spans()}


# --- endpoint and header resolution -----------------------------------------


def test_otlp_leaves_endpoint_and_headers_to_the_standard_variables() -> None:
    settings = resolve_export_settings(_config("otlp"), {})
    assert (settings.endpoint, settings.headers) == (None, None)
    assert settings.protocol == "http/protobuf"
    assert settings.timeout == 10.0
    assert settings.exporter_kwargs() == {"timeout": 10.0}


def test_export_timeout_from_config_and_env_wins() -> None:
    cfg = _config("otlp", export_timeout_seconds=2.5)
    assert resolve_export_settings(cfg, {}).timeout == 2.5
    env = {"OTEL_EXPORTER_OTLP_TIMEOUT": "4", "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT": "3"}
    assert resolve_export_settings(cfg, env).timeout == 3.0
    with pytest.raises(ConfigurationError, match="positive"):
        resolve_export_settings(cfg, {"OTEL_EXPORTER_OTLP_TIMEOUT": "soon"})
    with pytest.raises(ValueError):
        _config("otlp", export_timeout_seconds=0)
    with pytest.raises(ValueError):
        _config("otlp", export_timeout_seconds=600)


def test_langfuse_builds_endpoint_and_basic_auth_from_env() -> None:
    settings = resolve_export_settings(_config("langfuse"), _LANGFUSE_ENV)
    assert settings.endpoint == "http://localhost:3000" + LANGFUSE_OTLP_PATH
    assert settings.headers is not None
    scheme, token = settings.headers["authorization"].split(" ", 1)
    assert scheme == "Basic"
    assert base64.b64decode(token).decode() == f"{_PUBLIC}:{_SECRET}"
    assert settings.headers["x-langfuse-ingestion-version"] == "4"


def test_langfuse_standard_variables_win() -> None:
    env = {
        **_LANGFUSE_ENV,
        "OTEL_EXPORTER_OTLP_ENDPOINT": "https://collector.example.com",
        "OTEL_EXPORTER_OTLP_HEADERS": "authorization=Basic%20abc",
    }
    settings = resolve_export_settings(_config("langfuse"), env)
    assert (settings.endpoint, settings.headers) == (None, None)


@pytest.mark.parametrize(
    ("env", "match"),
    [
        ({"LANGFUSE_PUBLIC_KEY": _PUBLIC, "LANGFUSE_SECRET_KEY": _SECRET}, "HOST"),
        ({"LANGFUSE_HOST": "http://localhost:3000"}, "LANGFUSE_SECRET_KEY"),
        (
            {"LANGFUSE_HOST": "http://h", "LANGFUSE_PUBLIC_KEY": _PUBLIC},
            "LANGFUSE_SECRET_KEY",
        ),
        ({**_LANGFUSE_ENV, "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc"}, "gRPC"),
    ],
)
def test_langfuse_fails_fast_without_host_keys_or_on_grpc(
    env: dict[str, str], match: str
) -> None:
    with pytest.raises(ConfigurationError, match=match) as excinfo:
        resolve_export_settings(_config("langfuse"), env)
    assert _SECRET not in str(excinfo.value)
    assert _PUBLIC not in str(excinfo.value)


def test_langfuse_missing_host_names_the_full_otlp_endpoint() -> None:
    # The host root alone 404s: the generic variable needs Langfuse's OTLP path.
    env = {"LANGFUSE_PUBLIC_KEY": _PUBLIC, "LANGFUSE_SECRET_KEY": _SECRET}
    with pytest.raises(ConfigurationError, match="/api/public/otel"):
        resolve_export_settings(_config("langfuse"), env)


def test_phoenix_defaults_to_the_local_server_and_adds_bearer_key() -> None:
    cfg = _config("phoenix")
    local = resolve_export_settings(cfg, {})
    assert (local.endpoint, local.headers) == (PHOENIX_DEFAULT_HTTP_ENDPOINT, None)
    grpc = resolve_export_settings(cfg, {"OTEL_EXPORTER_OTLP_PROTOCOL": "grpc"})
    assert grpc.endpoint == PHOENIX_DEFAULT_GRPC_ENDPOINT
    remote = resolve_export_settings(
        cfg,
        {
            "PHOENIX_COLLECTOR_ENDPOINT": "https://phoenix.example.com/",
            "PHOENIX_API_KEY": _PHOENIX_KEY,
        },
    )
    assert remote.endpoint == "https://phoenix.example.com/v1/traces"
    # Lower-case key: gRPC metadata keys must be.
    assert remote.headers == {"authorization": f"Bearer {_PHOENIX_KEY}"}


def test_settings_repr_and_description_hold_no_secret() -> None:
    settings = resolve_export_settings(_config("langfuse"), _LANGFUSE_ENV)
    assert settings.headers is not None
    token = settings.headers["authorization"]
    for text in (repr(settings), settings.describe()):
        assert _SECRET not in text and token not in text and _PUBLIC not in text
    assert "headers=authorization,x-langfuse-ingestion-version" in settings.describe()
    assert (
        safe_url("https://user:pw@collector.example.com:4318/v1/traces?key=s#f")
        == "https://collector.example.com:4318/v1/traces"
    )


# --- backend construction ----------------------------------------------------


class _FakeOTLPExporter(InMemorySpanExporter):
    """Stands in for the OTLP exporter; records how it was constructed."""

    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        type(self).calls.append(kwargs)


@pytest.fixture
def fake_otlp(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    _FakeOTLPExporter.calls = []
    for name in (
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
    ):
        module = types.ModuleType(name)
        module.OTLPSpanExporter = _FakeOTLPExporter  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, module)
    return _FakeOTLPExporter.calls


@pytest.mark.parametrize(
    ("backend", "env", "endpoint", "header_names"),
    [
        ("otlp", {}, None, None),
        (
            "langfuse",
            _LANGFUSE_ENV,
            "http://localhost:3000" + LANGFUSE_OTLP_PATH,
            {"authorization", "x-langfuse-ingestion-version"},
        ),
        (
            "phoenix",
            {"PHOENIX_API_KEY": _PHOENIX_KEY},
            PHOENIX_DEFAULT_HTTP_ENDPOINT,
            {"authorization"},
        ),
    ],
)
def test_backend_builds_a_bounded_otlp_exporter_without_logging_secrets(
    backend: str,
    env: dict[str, str],
    endpoint: str | None,
    header_names: set[str] | None,
    fake_otlp: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(otel.trace, "set_tracer_provider", lambda provider: None)
    with caplog.at_level(logging.DEBUG):
        tracer = _with_env(
            env, otel.build_backend, _config(backend, export_timeout_seconds=3)
        )
    tracer.shutdown()
    (kwargs,) = fake_otlp
    assert kwargs["timeout"] == 3
    assert kwargs.get("endpoint") == endpoint
    assert (set(kwargs["headers"]) if "headers" in kwargs else None) == header_names
    assert f"Observability exporter for backend '{backend}'" in caplog.text
    for secret in (_SECRET, _PUBLIC, _PHOENIX_KEY, "Basic ", "Bearer "):
        assert secret not in caplog.text


def test_reconfiguring_does_not_blame_a_foreign_global_provider(
    fake_otlp: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The first exporting backend becomes the global provider; rebuilding with
    # new settings must not warn that "a global provider is already set" when
    # that global is the package's own. A host's provider still warns.
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    global_provider: list[Any] = [trace.ProxyTracerProvider()]
    monkeypatch.setattr(otel, "_bound_global", [], raising=False)
    monkeypatch.setattr(otel.trace, "get_tracer_provider", lambda: global_provider[0])
    monkeypatch.setattr(
        otel.trace, "set_tracer_provider", lambda p: global_provider.__setitem__(0, p)
    )
    warning = "A global OpenTelemetry tracer provider is already set"
    with caplog.at_level(logging.WARNING):
        for rate in (1.0, 0.5):
            otel.build_backend(_config("otlp", sample_rate=rate)).shutdown()
        assert warning not in caplog.text
        global_provider[0] = TracerProvider()  # a host application's own
        otel.build_backend(_config("otlp")).shutdown()
    assert warning in caplog.text


def test_grpc_protocol_selects_the_grpc_exporter(
    fake_otlp: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class _Grpc(_FakeOTLPExporter):
        def __init__(self, **kwargs: Any) -> None:
            calls.append("grpc")
            super().__init__(**kwargs)

    sys.modules[
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
    ].OTLPSpanExporter = _Grpc  # type: ignore[attr-defined]
    monkeypatch.setattr(otel.trace, "set_tracer_provider", lambda provider: None)
    tracer = _with_env(
        {"OTEL_EXPORTER_OTLP_PROTOCOL": "grpc"}, otel.build_backend, _config("phoenix")
    )
    tracer.shutdown()
    assert calls == ["grpc"]
    assert fake_otlp[0]["endpoint"] == PHOENIX_DEFAULT_GRPC_ENDPOINT


@pytest.mark.parametrize("backend", ["otlp", "langfuse", "phoenix"])
def test_every_exporting_backend_samples_parent_based(
    backend: str, no_global_provider: None
) -> None:
    exporter, tracer = _install(backend, sample_rate=0.25)
    provider = tracer._owned_provider
    description = provider.sampler.get_description()
    assert description.startswith("ParentBased") and "0.25" in description


@pytest.mark.parametrize("backend", ["otlp", "langfuse", "phoenix"])
def test_sample_rate_zero_drops_whole_traces(
    backend: str, no_global_provider: None
) -> None:
    exporter, _ = _install(backend, sample_rate=0.0)
    with observability.span("kg_rag.query"):
        with observability.span("kg_rag.strategy"):
            observability.start_span("chat m").end()
    assert _flushed(exporter) == {}


def test_phoenix_project_and_service_name_resource(no_global_provider: None) -> None:
    _, tracer = _install("phoenix", environ={}, service_name="kg-svc")
    attributes = tracer._owned_provider.resource.attributes
    assert attributes["service.name"] == "kg-svc"
    assert attributes["openinference.project.name"] == "kg-svc"
    _, tracer = _install("phoenix", environ={"PHOENIX_PROJECT_NAME": "proj"})
    assert tracer._owned_provider.resource.attributes["openinference.project.name"] == (
        "proj"
    )
    _, tracer = _install("langfuse")
    assert (
        "openinference.project.name" not in tracer._owned_provider.resource.attributes
    )


# --- session / user ------------------------------------------------------------


def test_bind_identity_tags_every_span_in_scope(no_global_provider: None) -> None:
    exporter, _ = _install("otlp", environ={})
    with observability.bind_identity(session_id="conv-1", user_id="user-7"):
        with observability.span("kg_rag.query"):
            observability.start_span("chat m").end()
    with observability.span("kg_rag.after"):
        pass
    spans = _flushed(exporter)
    for name in ("kg_rag.query", "chat m"):
        assert spans[name].attributes["session.id"] == "conv-1"
        assert spans[name].attributes["user.id"] == "user-7"
    assert "session.id" not in spans["kg_rag.after"].attributes


def test_bind_identity_is_a_noop_when_disabled_or_empty(
    no_global_provider: None,
) -> None:
    disabled = observability.bind_identity(session_id="c", user_id="u")
    assert disabled is observability.bind_identity(session_id="d")
    exporter, _ = _install("otlp", environ={})
    with observability.bind_identity(session_id=None, user_id=None):
        with observability.span("kg_rag.query"):
            pass
    assert "session.id" not in _flushed(exporter)["kg_rag.query"].attributes


# --- vendor attribute mapping -----------------------------------------------------

_CHAT = {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "aws.bedrock",
    "gen_ai.request.model": "model-a",
    "gen_ai.usage.input_tokens": 120,
    "gen_ai.usage.output_tokens": 30,
    "gen_ai.usage.cache_read.input_tokens": 100,
}
_EMBED = {"gen_ai.operation.name": "embeddings", "gen_ai.request.model": "embed-a"}
_RERANK = {"gen_ai.provider.name": "aws.bedrock", "gen_ai.request.model": "rerank-a"}


def test_langfuse_observation_types() -> None:
    def kind(name: str, attributes: dict[str, Any]) -> Any:
        return langfuse_attributes(name, attributes)["langfuse.observation.type"]

    assert kind("chat model-a", _CHAT) == "generation"
    assert kind("embeddings embed-a", _EMBED) == "embedding"
    assert kind("kg_rag.retrieve", {}) == "retriever"
    # A model attribute alone would make Langfuse infer "generation".
    assert kind("kg_rag.rerank", _RERANK) == "retriever"
    assert kind("kg_rag.query", {}) == "span"
    content = langfuse_attributes("chat m", {**_CHAT, "gen_ai.input.messages": "[in]"})
    assert content["langfuse.observation.input"] == "[in]"
    assert "langfuse.observation.output" not in content


def test_openinference_kinds_models_and_token_counts() -> None:
    chat = openinference_attributes("chat model-a", _CHAT)
    assert chat["openinference.span.kind"] == "LLM"
    assert chat["llm.model_name"] == "model-a"
    assert chat["llm.provider"] == "aws"
    local = openinference_attributes(
        "chat model-b", {**_CHAT, "gen_ai.provider.name": "ollama"}
    )
    assert local["llm.provider"] == "ollama"
    assert (
        chat["llm.token_count.prompt"],
        chat["llm.token_count.completion"],
        chat["llm.token_count.total"],
        chat["llm.token_count.prompt_details.cache_read"],
    ) == (120, 30, 150, 100)
    assert "input.value" not in chat  # no content unless captured
    embed = openinference_attributes("embeddings embed-a", _EMBED)
    assert embed == {
        "openinference.span.kind": "EMBEDDING",
        "embedding.model_name": "embed-a",
    }
    assert openinference_attributes("kg_rag.rerank", _RERANK)[
        "openinference.span.kind"
    ] == ("RERANKER")
    assert openinference_attributes("kg_rag.retrieve", {}) == {
        "openinference.span.kind": "RETRIEVER"
    }
    assert openinference_attributes("kg_rag.query", {}) == {
        "openinference.span.kind": "CHAIN"
    }
    out = openinference_attributes("chat m", {**_CHAT, "gen_ai.output.messages": "o"})
    assert (out["output.value"], out["output.mime_type"]) == ("o", "application/json")


@pytest.mark.parametrize(
    ("backend", "key", "value"),
    [
        ("langfuse", "langfuse.observation.type", "generation"),
        ("phoenix", "openinference.span.kind", "LLM"),
        ("otlp", "openinference.span.kind", None),
        ("otlp", "langfuse.observation.type", None),
    ],
)
def test_vendor_attributes_are_added_only_for_that_backend(
    backend: str, key: str, value: str | None, no_global_provider: None
) -> None:
    exporter, _ = _install(backend, environ=_LANGFUSE_ENV)
    with observability.span("chat model-a", _CHAT):
        pass
    span = _flushed(exporter)["chat model-a"]
    assert span.attributes.get(key) == value
    assert span.attributes["gen_ai.usage.input_tokens"] == 120  # kept as recorded


# --- redaction -------------------------------------------------------------------


def test_redactor_handles_strings_lists_and_leaves_other_values() -> None:
    redactor = Redactor([r"\d{3}-\d{4}", "Vendor"])
    assert not Redactor([])
    assert redactor.attributes(
        {"a": "call 555-0100", "b": ["Vendor x", 3], "c": 7, "d": True}
    ) == {"a": "call [REDACTED]", "b": ["[REDACTED] x", 3], "c": 7, "d": True}


@pytest.mark.parametrize("backend", ["otlp", "langfuse", "phoenix"])
def test_redaction_covers_every_exported_string(
    backend: str, no_global_provider: None
) -> None:
    exporter, _ = _install(
        backend,
        redact_patterns=[r"ACCT-\d+"],
        capture_content=True,
    )
    with observability.bind_identity(session_id="sess ACCT-1"):
        with pytest.raises(ValueError):
            with observability.span(
                "chat model-a",
                {
                    **_CHAT,
                    "gen_ai.input.messages": '[{"content": "ACCT-2 balance"}]',
                },
                label="ACCT-3",
                source_ids=["doc-1", "ACCT-4"],
                hits=2,
            ):
                raise ValueError("lookup failed for ACCT-5")
    span = _flushed(exporter)["chat model-a"]
    exported = [str(v) for v in span.attributes.values()]
    exported += [str(v) for e in span.events for v in (e.attributes or {}).values()]
    exported.append(span.status.description or "")
    assert not [text for text in exported if "ACCT-" in text]
    assert span.attributes["kg_rag.label"] == "[REDACTED]"
    assert list(span.attributes["kg_rag.source_ids"]) == ["doc-1", "[REDACTED]"]
    assert span.attributes["session.id"] == "sess [REDACTED]"
    assert span.attributes["kg_rag.hits"] == 2
    assert span.status.status_code == StatusCode.ERROR
    assert "[REDACTED]" in span.events[0].attributes["exception.message"]
    vendor_input = {
        "langfuse": "langfuse.observation.input",
        "phoenix": "input.value",
    }.get(backend)
    if vendor_input:
        assert "[REDACTED] balance" in span.attributes[vendor_input]


@pytest.mark.parametrize("backend", ["otlp", "langfuse", "phoenix", "none"])
def test_content_is_redacted_at_capture_on_every_backend(
    backend: str,
    no_global_provider: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Export-time redaction sees content JSON-encoded, where a newline is the
    # two characters backslash-n, so it cannot be the only layer.
    from unified_kg_rag.application.observability import configure_observability

    build = otel.build_backend

    def _in_memory(config: ObservabilityConfig, tracer_provider: Any = None) -> Any:
        return _with_env(
            _LANGFUSE_ENV, build, config, span_exporter=InMemorySpanExporter()
        )

    monkeypatch.setattr(otel, "build_backend", _in_memory)
    configure_observability(
        _config(backend, capture_content=True, redact_patterns=[r"ACCT-\d+"])
    )
    assert len(observability.content_policy().redact_patterns) == 1


@pytest.mark.parametrize("backend", ["otlp", "langfuse", "phoenix"])
def test_word_boundary_pattern_next_to_a_newline_is_redacted(
    backend: str,
    no_global_provider: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.language_models.fake_chat_models import (
        GenericFakeChatModel,
    )
    from langchain_core.messages import AIMessage, HumanMessage

    from unified_kg_rag.adapters.observability.langchain_callback import (
        GenAITracingCallbackHandler,
    )
    from unified_kg_rag.application.observability import configure_observability

    exporter = InMemorySpanExporter()
    build = otel.build_backend

    def _in_memory(config: ObservabilityConfig, tracer_provider: Any = None) -> Any:
        return _with_env(_LANGFUSE_ENV, build, config, span_exporter=exporter)

    monkeypatch.setattr(otel, "build_backend", _in_memory)
    configure_observability(
        _config(
            backend,
            capture_content=True,
            redact_patterns=[r"\b\d{3}-\d{2}-\d{4}\b"],
        )
    )
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="Noted:\n987-65-4321")]),
        callbacks=[GenAITracingCallbackHandler("vendor.model-v1")],
    )
    model.invoke([HumanMessage(content="Customer record:\n123-45-6789")])
    spans = _flushed(exporter)
    exported = " ".join(
        str(value) for span in spans.values() for value in span.attributes.values()
    )
    assert "123-45-6789" not in exported
    assert "987-65-4321" not in exported
    assert "Customer record:\\n[REDACTED]" in exported


def test_no_patterns_exports_spans_unchanged(no_global_provider: None) -> None:
    exporter, _ = _install("otlp", environ={})
    with observability.span("kg_rag.retrieve", label="ACCT-1"):
        pass
    assert _flushed(exporter)["kg_rag.retrieve"].attributes["kg_rag.label"] == (
        "ACCT-1"
    )


def test_unknown_backend_is_rejected() -> None:
    assert {b.value for b in ObservabilityBackend} == {
        "none",
        "otlp",
        "langfuse",
        "phoenix",
    }
    with pytest.raises(ValueError):
        _config("cloudwatch")
