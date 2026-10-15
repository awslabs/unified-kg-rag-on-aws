# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the OTLP exporter sends spans, per backend, from the environment.

Pure Python (no OpenTelemetry import) so it is testable without the exporter.
Credentials come only from environment variables, which a deployment fills
from Secrets Manager or ECS task secrets; nothing here reads the config file
for them, and :class:`ExportSettings` keeps its headers out of ``repr`` and
:meth:`ExportSettings.describe` so they cannot reach a log line.

Precedence, for every backend: the standard ``OTEL_EXPORTER_OTLP_TRACES_*``
variable, then ``OTEL_EXPORTER_OTLP_*``, then the vendor's variables, then the
vendor's local default. When a standard variable is set the exporter reads it
itself, so the endpoint or headers are left ``None`` here.

* ``otlp``: standard variables only.
* ``langfuse``: ``LANGFUSE_HOST`` (or ``LANGFUSE_BASE_URL``) for the base URL,
  ``LANGFUSE_PUBLIC_KEY`` + ``LANGFUSE_SECRET_KEY`` for the Basic-auth header.
  Both are required (no default host: traces must not go to a hosted service
  nobody chose). Langfuse ingests OTLP over HTTP only.
* ``phoenix``: ``PHOENIX_COLLECTOR_ENDPOINT`` for the base URL (default the
  local server, ``http://localhost:6006`` over HTTP or ``:4317`` over gRPC),
  ``PHOENIX_API_KEY`` for a Bearer header when the server needs one.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

from unified_kg_rag.domain.models.config import (
    ObservabilityBackend,
    ObservabilityConfig,
)
from unified_kg_rag.shared.exceptions import ConfigurationError

OTLP_PROTOCOLS = ("http/protobuf", "grpc")

LANGFUSE_OTLP_PATH = "/api/public/otel/v1/traces"
# Without it Langfuse may take minutes to show directly ingested OTLP data.
LANGFUSE_INGESTION_VERSION_HEADER = ("x-langfuse-ingestion-version", "4")
PHOENIX_DEFAULT_HTTP_ENDPOINT = "http://localhost:6006/v1/traces"
PHOENIX_DEFAULT_GRPC_ENDPOINT = "http://localhost:4317"


@dataclass(frozen=True)
class ExportSettings:
    """Arguments for the OTLP span exporter.

    ``endpoint`` / ``headers`` of ``None`` leave the exporter to read the
    ``OTEL_EXPORTER_OTLP_*`` variables. ``timeout`` is in seconds.
    """

    protocol: str
    timeout: float
    endpoint: str | None = None
    headers: dict[str, str] | None = field(default=None, repr=False)
    endpoint_source: str = "OTEL_EXPORTER_OTLP_*"

    def describe(self) -> str:
        """A log-safe summary: endpoint without credentials, header *names*."""
        endpoint = safe_url(self.endpoint) if self.endpoint else self.endpoint_source
        header_names = ",".join(sorted(self.headers)) if self.headers else "-"
        return (
            f"endpoint={endpoint}, protocol={self.protocol}, "
            f"timeout={self.timeout:g}s, headers={header_names}"
        )

    def exporter_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {"timeout": self.timeout}
        if self.endpoint is not None:
            kwargs["endpoint"] = self.endpoint
        if self.headers is not None:
            kwargs["headers"] = dict(self.headers)
        return kwargs


def safe_url(url: str) -> str:
    """``url`` without userinfo, query or fragment (they may hold a key)."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _env(environ: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = (environ.get(name) or "").strip()
        if value:
            return value
    return None


def resolve_protocol(environ: Mapping[str, str]) -> str:
    protocol = (
        _env(
            environ, "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "OTEL_EXPORTER_OTLP_PROTOCOL"
        )
        or "http/protobuf"
    )
    if protocol not in OTLP_PROTOCOLS:
        raise ConfigurationError(
            f"Unsupported OTLP protocol '{protocol}' (OTEL_EXPORTER_OTLP_PROTOCOL); "
            f"supported: {', '.join(OTLP_PROTOCOLS)}"
        )
    return protocol


def _resolve_timeout(config: ObservabilityConfig, environ: Mapping[str, str]) -> float:
    names = ("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT", "OTEL_EXPORTER_OTLP_TIMEOUT")
    raw = _env(environ, *names)
    if raw is None:
        return config.export_timeout_seconds
    try:
        timeout = float(raw)
    except ValueError:
        timeout = 0.0
    if timeout <= 0:
        raise ConfigurationError(
            f"OTEL_EXPORTER_OTLP_TIMEOUT must be a positive number of seconds, "
            f"got {raw!r}"
        )
    return timeout


def _has_standard(environ: Mapping[str, str], suffix: str) -> bool:
    return (
        _env(
            environ,
            f"OTEL_EXPORTER_OTLP_TRACES_{suffix}",
            f"OTEL_EXPORTER_OTLP_{suffix}",
        )
        is not None
    )


def _join(base: str, path: str) -> str:
    return base.rstrip("/") + path


def resolve_export_settings(
    config: ObservabilityConfig, environ: Mapping[str, str]
) -> ExportSettings:
    """The exporter arguments for ``config.backend`` (see the module docstring).

    Raises:
        ConfigurationError: an unsupported protocol or timeout, gRPC for
            Langfuse, or Langfuse without its host or keys.
    """
    protocol = resolve_protocol(environ)
    timeout = _resolve_timeout(config, environ)
    backend = config.backend
    if backend == ObservabilityBackend.LANGFUSE:
        return _langfuse(protocol, timeout, environ)
    if backend == ObservabilityBackend.PHOENIX:
        return _phoenix(protocol, timeout, environ)
    return ExportSettings(protocol=protocol, timeout=timeout)


def _langfuse(
    protocol: str, timeout: float, environ: Mapping[str, str]
) -> ExportSettings:
    if protocol != "http/protobuf":
        raise ConfigurationError(
            "observability.backend 'langfuse' needs OTLP over HTTP: Langfuse does "
            "not accept gRPC (unset OTEL_EXPORTER_OTLP_PROTOCOL or set it to "
            "http/protobuf)"
        )
    endpoint: str | None = None
    source = "OTEL_EXPORTER_OTLP_*"
    if not _has_standard(environ, "ENDPOINT"):
        host = _env(environ, "LANGFUSE_HOST", "LANGFUSE_BASE_URL")
        if host is None:
            raise ConfigurationError(
                "observability.backend 'langfuse' needs the Langfuse URL: set "
                "LANGFUSE_HOST (e.g. http://localhost:3000 for a local server) "
                "or OTEL_EXPORTER_OTLP_ENDPOINT=<host>/api/public/otel (the "
                "exporter appends /v1/traces)"
            )
        endpoint = _join(host, LANGFUSE_OTLP_PATH)
        source = "LANGFUSE_HOST"
    headers: dict[str, str] | None = None
    if not _has_standard(environ, "HEADERS"):
        public_key = _env(environ, "LANGFUSE_PUBLIC_KEY")
        secret_key = _env(environ, "LANGFUSE_SECRET_KEY")
        if not (public_key and secret_key):
            raise ConfigurationError(
                "observability.backend 'langfuse' needs LANGFUSE_PUBLIC_KEY and "
                "LANGFUSE_SECRET_KEY (from Secrets Manager or task secrets), or "
                "OTEL_EXPORTER_OTLP_HEADERS with the Authorization header"
            )
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        name, value = LANGFUSE_INGESTION_VERSION_HEADER
        headers = {"authorization": f"Basic {token}", name: value}
    return ExportSettings(
        protocol=protocol,
        timeout=timeout,
        endpoint=endpoint,
        headers=headers,
        endpoint_source=source,
    )


def _phoenix(
    protocol: str, timeout: float, environ: Mapping[str, str]
) -> ExportSettings:
    endpoint: str | None = None
    source = "OTEL_EXPORTER_OTLP_*"
    if not _has_standard(environ, "ENDPOINT"):
        base = _env(environ, "PHOENIX_COLLECTOR_ENDPOINT")
        if base is None:
            endpoint = (
                PHOENIX_DEFAULT_GRPC_ENDPOINT
                if protocol == "grpc"
                else PHOENIX_DEFAULT_HTTP_ENDPOINT
            )
            source = "default"
        else:
            # The HTTP exporter takes the full traces URL; gRPC the server root.
            endpoint = base if protocol == "grpc" else _join(base, "/v1/traces")
            source = "PHOENIX_COLLECTOR_ENDPOINT"
    headers: dict[str, str] | None = None
    if not _has_standard(environ, "HEADERS"):
        api_key = _env(environ, "PHOENIX_API_KEY")
        if api_key:
            # Lower-case: gRPC metadata keys must be.
            headers = {"authorization": f"Bearer {api_key}"}
    return ExportSettings(
        protocol=protocol,
        timeout=timeout,
        endpoint=endpoint,
        headers=headers,
        endpoint_source=source,
    )
