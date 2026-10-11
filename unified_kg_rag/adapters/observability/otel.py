# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry implementation of the ``shared.observability`` port.

Imported only when ``observability.enabled`` is true; importing it without the
``otel`` extra raises :class:`ConfigurationError` naming the extra.

Backends:

* ``none``: spans go to the process's global tracer provider, which the host
  application (or ``opentelemetry-instrument``) owns. Nothing is installed.
* ``otlp``, ``langfuse``, ``phoenix``: an SDK tracer provider with a
  parent-based ratio sampler (``sample_rate``), a batch span processor and an
  OTLP exporter. The endpoint and headers come from the environment (see
  ``endpoints``: the standard ``OTEL_EXPORTER_OTLP_*`` variables, or the
  ``LANGFUSE_*`` / ``PHOENIX_*`` ones), so no credential is ever part of the
  config file. Spans pass through ``span_transforms.TransformingSpanExporter``
  (vendor attribute names, then ``redact_patterns`` on every string
  attribute) on their way out.

Spans are exported from a background thread. Each export request is bounded by
``export_timeout_seconds`` (or ``OTEL_EXPORTER_OTLP_TIMEOUT``), the batch
queue drops spans rather than block when it is full, and the final flush at
exit waits at most for one bounded export, so a slow or unreachable collector
cannot stall a request.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from enum import Enum
from typing import Any

from unified_kg_rag import __version__
from unified_kg_rag.domain.models.config import (
    ObservabilityBackend,
    ObservabilityConfig,
)
from unified_kg_rag.shared.exceptions import ConfigurationError
from unified_kg_rag.shared.logging import get_logger
from unified_kg_rag.shared.observability import SpanHandle

from .endpoints import ExportSettings, resolve_export_settings

_EXTRA_HINT = (
    "install the 'otel' extra (`uv sync --extra otel`, or "
    "`pip install 'unified-kg-rag-on-aws[otel]'`) or set "
    "observability.enabled: false"
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
except ImportError as e:  # pragma: no cover - exercised via sys.modules patching
    raise ConfigurationError(
        f"observability.enabled is true but OpenTelemetry is not installed: "
        f"{_EXTRA_HINT}"
    ) from e

logger = get_logger(__name__)

INSTRUMENTATION_NAME = "unified_kg_rag"
# The attribute names used here (gen_ai.* included) were checked against
# opentelemetry-semantic-conventions 0.66b1, which ships schema 1.44.0. The
# GenAI conventions are still evolving (now in the semantic-conventions-genai
# repository); the schema URL tells a backend which version to read them as.
SEMCONV_SCHEMA_URL = "https://opentelemetry.io/schemas/1.44.0"

_EXPORTING_BACKENDS = (
    ObservabilityBackend.OTLP,
    ObservabilityBackend.LANGFUSE,
    ObservabilityBackend.PHOENIX,
)
# OpenInference resource attribute Phoenix files traces under.
PHOENIX_PROJECT_ATTRIBUTE = "openinference.project.name"

AttributeValue = (
    str | bool | int | float | list[str] | list[bool] | list[int] | list[float]
)


def to_attribute_value(value: Any) -> AttributeValue | None:
    """``value`` as an OpenTelemetry attribute value; ``None`` drops it.

    Attributes accept primitives and homogeneous lists of them: enums become
    their value, mixed or non-primitive sequences become lists of strings, and
    anything else its ``str``.
    """
    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, Sequence | set | frozenset) and not isinstance(
        value, bytes | bytearray
    ):
        items = [v.value if isinstance(v, Enum) else v for v in value if v is not None]
        if all(isinstance(i, str) for i in items):
            return [str(i) for i in items]
        if all(isinstance(i, bool) for i in items):
            return [bool(i) for i in items]
        if all(isinstance(i, int) and not isinstance(i, bool) for i in items):
            return [int(i) for i in items]
        if all(isinstance(i, int | float) and not isinstance(i, bool) for i in items):
            return [float(i) for i in items]
        return [str(i) for i in items]
    return str(value)


def clean_attributes(attributes: Mapping[str, Any]) -> dict[str, AttributeValue]:
    cleaned: dict[str, AttributeValue] = {}
    for key, value in attributes.items():
        converted = to_attribute_value(value)
        if converted is not None:
            cleaned[key] = converted
    return cleaned


class OTelSpan:
    """``SpanHandle`` over an OpenTelemetry span."""

    __slots__ = ("_span",)

    def __init__(self, span: trace.Span) -> None:
        self._span = span

    def set_attribute(self, key: str, value: Any) -> None:
        converted = to_attribute_value(value)
        if converted is not None:
            self._span.set_attribute(key, converted)

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        cleaned = clean_attributes(attributes)
        if cleaned:
            self._span.set_attributes(cleaned)

    def record_exception(self, exception: BaseException) -> None:
        self._span.record_exception(exception)

    def set_error(self, description: str) -> None:
        self._span.set_status(Status(StatusCode.ERROR, description))

    def end(self) -> None:
        self._span.end()


class OTelTracerBackend:
    """The tracing port over an OpenTelemetry tracer."""

    def __init__(
        self,
        tracer: trace.Tracer,
        owned_provider: Any | None = None,
    ) -> None:
        self._tracer = tracer
        # Shut down with the backend only when this module created it.
        self._owned_provider = owned_provider

    @contextmanager
    def use_span(
        self, name: str, attributes: Mapping[str, Any]
    ) -> Iterator[SpanHandle]:
        # start_as_current_span records an escaping exception and sets the
        # ERROR status before re-raising it.
        with self._tracer.start_as_current_span(
            name, attributes=clean_attributes(attributes)
        ) as otel_span:
            yield OTelSpan(otel_span)

    def start_span(self, name: str, attributes: Mapping[str, Any]) -> SpanHandle:
        return OTelSpan(
            self._tracer.start_span(name, attributes=clean_attributes(attributes))
        )

    def current_span(self) -> SpanHandle:
        return OTelSpan(trace.get_current_span())

    def trace_ids(self) -> tuple[str, str] | None:
        context = trace.get_current_span().get_span_context()
        if not context.is_valid:
            return None
        return format(context.trace_id, "032x"), format(context.span_id, "016x")

    def shutdown(self) -> None:
        if self._owned_provider is not None:
            self._owned_provider.shutdown()


def build_backend(
    config: ObservabilityConfig,
    tracer_provider: Any | None = None,
    *,
    span_exporter: Any | None = None,
    environ: Mapping[str, str] | None = None,
) -> OTelTracerBackend:
    """The tracer backend ``config`` asks for.

    ``tracer_provider`` overrides the backend choice with a caller-owned
    provider (tests use one with an in-memory exporter); it is not shut down
    with the backend, and no vendor mapping or redaction is added to it.
    ``span_exporter`` replaces the OTLP exporter of an exporting backend
    (tests use an in-memory one) while keeping its sampler, batching, vendor
    mapping and redaction; the provider is then not made global.
    ``environ`` defaults to ``os.environ``.
    """
    env = os.environ if environ is None else environ
    owned: Any | None = None
    if tracer_provider is not None:
        provider = tracer_provider
    elif config.backend == ObservabilityBackend.NONE:
        provider = trace.get_tracer_provider()
    elif config.backend in _EXPORTING_BACKENDS:
        provider = owned = _sdk_tracer_provider(config, env, span_exporter)
        if span_exporter is None and isinstance(
            trace.get_tracer_provider(), trace.ProxyTracerProvider
        ):
            # Make it the global provider too, so spans from libraries
            # instrumented through the OTel API join the same traces.
            trace.set_tracer_provider(provider)
        elif span_exporter is None:
            logger.warning(
                "A global OpenTelemetry tracer provider is already set; "
                "observability backend '%s' exports through its own provider. "
                "Use backend 'none' to send spans to the existing one instead.",
                config.backend.value,
            )
    else:  # pragma: no cover - the enum is exhaustive
        raise ConfigurationError(
            f"Unsupported observability.backend '{config.backend}'"
        )
    tracer = provider.get_tracer(
        INSTRUMENTATION_NAME, __version__, schema_url=SEMCONV_SCHEMA_URL
    )
    return OTelTracerBackend(tracer, owned_provider=owned)


def _resource_attributes(
    config: ObservabilityConfig, env: Mapping[str, str]
) -> dict[str, str]:
    from opentelemetry.sdk.resources import SERVICE_NAME

    # OTEL_SERVICE_NAME / OTEL_RESOURCE_ATTRIBUTES (read by Resource.create)
    # win over the config values, as the OpenTelemetry environment
    # conventions specify; an attribute passed here would override them.
    attributes: dict[str, str] = {}
    if not env.get("OTEL_SERVICE_NAME"):
        attributes[SERVICE_NAME] = config.service_name
    if config.backend == ObservabilityBackend.PHOENIX and (
        PHOENIX_PROJECT_ATTRIBUTE not in env.get("OTEL_RESOURCE_ATTRIBUTES", "")
    ):
        attributes[PHOENIX_PROJECT_ATTRIBUTE] = (
            env.get("PHOENIX_PROJECT_NAME") or config.service_name
        )
    return attributes


def _sdk_tracer_provider(
    config: ObservabilityConfig,
    env: Mapping[str, str],
    span_exporter: Any | None,
) -> Any:
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

        from .span_transforms import (
            VENDOR_MAPPERS,
            Redactor,
            TransformingSpanExporter,
        )
    except ImportError as e:
        raise ConfigurationError(
            f"observability.backend '{config.backend.value}' needs the "
            f"OpenTelemetry SDK: {_EXTRA_HINT}"
        ) from e

    settings = resolve_export_settings(config, env)
    exporter = (
        span_exporter if span_exporter is not None else _otlp_exporter(config, settings)
    )
    logger.info(
        "Observability exporter for backend '%s': %s",
        config.backend.value,
        settings.describe(),
    )
    provider = TracerProvider(
        resource=Resource.create(_resource_attributes(config, env)),
        sampler=ParentBased(TraceIdRatioBased(config.sample_rate)),
    )
    exporter = TransformingSpanExporter(
        exporter,
        mapper=VENDOR_MAPPERS.get(config.backend),
        redactor=Redactor(config.redact_patterns),
    )
    timeout_millis = settings.timeout * 1000
    kwargs: dict[str, Any] = {}
    if not env.get("OTEL_BSP_EXPORT_TIMEOUT"):
        # The processor's own bound on one export (30 s by default) follows
        # the exporter's, so neither the batch thread nor the exit flush
        # waits longer than one request is allowed to take.
        kwargs["export_timeout_millis"] = timeout_millis
    provider.add_span_processor(BatchSpanProcessor(exporter, **kwargs))
    return provider


def _otlp_exporter(config: ObservabilityConfig, settings: ExportSettings) -> Any:
    """The OTLP span exporter for the resolved protocol and settings."""
    try:
        if settings.protocol == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter as GrpcExporter,
            )

            return GrpcExporter(**settings.exporter_kwargs())
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HttpExporter,
        )

        return HttpExporter(**settings.exporter_kwargs())
    except ImportError as e:
        raise ConfigurationError(
            f"observability.backend '{config.backend.value}' needs the OTLP "
            f"exporter: {_EXTRA_HINT}"
        ) from e
