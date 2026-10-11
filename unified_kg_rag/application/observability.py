# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install the tracing backend that ``config.observability`` selects.

The composition roots (``DataIngestionPipeline``, ``GraphRAGChain``) call
:func:`configure_observability` when they are built, so the CLIs and library
callers both get tracing from the config alone. With ``enabled: false`` it
returns at once and nothing OpenTelemetry-related is imported.
"""

from __future__ import annotations

import os
from typing import Any

from unified_kg_rag.domain.models.config import ObservabilityConfig
from unified_kg_rag.shared import observability
from unified_kg_rag.shared.logging import get_logger

logger = get_logger(__name__)


def _sdk_disabled() -> bool:
    return os.getenv("OTEL_SDK_DISABLED", "").strip().lower() == "true"


def configure_observability(
    config: ObservabilityConfig, *, tracer_provider: Any | None = None
) -> bool:
    """Install the backend ``config`` asks for; return whether tracing is on.

    A disabled config leaves whatever is installed alone (a host application
    may have installed a backend itself), so building a second chain with the
    default config does not switch tracing off. Re-running with the settings
    already installed is a no-op. ``OTEL_SDK_DISABLED=true`` wins over
    ``enabled: true``, as the OpenTelemetry environment conventions specify.

    ``tracer_provider`` sends spans to a caller-owned OpenTelemetry tracer
    provider instead of the configured backend; later calls with the same
    settings keep it.

    Raises:
        ConfigurationError: tracing is enabled but the ``otel`` extra (or the
            exporter for the selected OTLP protocol) is not installed.
    """
    if not config.enabled:
        return observability.is_enabled()
    if _sdk_disabled():
        logger.info("observability.enabled is ignored: OTEL_SDK_DISABLED=true")
        return observability.is_enabled()

    settings_key = config.model_dump_json()
    if tracer_provider is None and observability.installed_settings() == settings_key:
        return True

    # Imported here: it imports OpenTelemetry, which the disabled path never
    # loads.
    from unified_kg_rag.adapters.observability.otel import build_backend

    backend = build_backend(config, tracer_provider)
    observability.install(
        backend,
        capture_content=config.capture_content,
        redact_patterns=config.redact_patterns,
        settings_key=settings_key,
    )
    logger.info(
        "Observability enabled (backend: '%s', sample_rate: %s, capture_content: %s)",
        config.backend.value,
        config.sample_rate,
        config.capture_content,
    )
    return True
