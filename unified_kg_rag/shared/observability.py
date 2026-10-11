# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tracing port: spans and span attributes without a tracing dependency.

Code at the pipeline and query choke points opens spans through this module;
it never imports OpenTelemetry itself. Until a backend is installed (see
``application.observability.configure_observability``, which loads the
OpenTelemetry adapter only when ``observability.enabled`` is true) every call
here is a no-op that allocates nothing: ``span`` returns one shared null
context manager and ``set_attributes`` returns at the first check.

Attribute keys: keyword arguments are namespaced as ``kg_rag.<name>``
(``span("kg_rag.retrieve", hits=3)`` records ``kg_rag.hits``); keys in the
``attributes`` mapping are recorded verbatim, for standard names such as the
``gen_ai.*`` semantic conventions. Values are str/bool/int/float or sequences
of them; ``None`` values are dropped by the backend.

Spans carry counts, ids and timings only. Text content (prompts, completions,
retrieved passages) is recorded only through :func:`content_policy`, which is
off unless ``observability.capture_content`` is set.
"""

from __future__ import annotations

import contextlib
import os
import re
import threading
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor

__all__ = [
    "NOOP_SPAN",
    "REDACTED",
    "ContentPolicy",
    "ContextThreadPoolExecutor",
    "SpanHandle",
    "TracerBackend",
    "content_policy",
    "current_trace_ids",
    "install",
    "installed_settings",
    "is_enabled",
    "set_attributes",
    "span",
    "start_span",
    "uninstall",
]

REDACTED = "[REDACTED]"
_NAMESPACE = "kg_rag."


class SpanHandle(Protocol):
    """The slice of a span the instrumented code uses."""

    def set_attribute(self, key: str, value: Any) -> None: ...

    def set_attributes(self, attributes: Mapping[str, Any]) -> None: ...

    def record_exception(self, exception: BaseException) -> None: ...

    def set_error(self, description: str) -> None: ...

    def end(self) -> None: ...


class TracerBackend(Protocol):
    """A tracing implementation (the port an adapter implements)."""

    def use_span(
        self, name: str, attributes: Mapping[str, Any]
    ) -> AbstractContextManager[SpanHandle]:
        """Open a span as the current span; record an escaping exception."""

    def start_span(self, name: str, attributes: Mapping[str, Any]) -> SpanHandle:
        """Start a child of the current span without making it current.

        For spans that start and end in different callbacks (a model call's
        start and end events); the caller must call ``end``.
        """

    def current_span(self) -> SpanHandle:
        """The active span (a non-recording one when there is none)."""

    def trace_ids(self) -> tuple[str, str] | None:
        """``(trace_id, span_id)`` in hex of the active span, if any."""

    def shutdown(self) -> None:
        """Flush and release what the backend owns."""


class _NoOpSpan:
    __slots__ = ()

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass

    def set_error(self, description: str) -> None:
        pass

    def end(self) -> None:
        pass


NOOP_SPAN: SpanHandle = _NoOpSpan()
# nullcontext is reusable and returns its argument from __enter__, so the
# disabled path hands out this one object for every span.
_NOOP_CONTEXT: AbstractContextManager[SpanHandle] = contextlib.nullcontext(NOOP_SPAN)


@dataclass(frozen=True)
class ContentPolicy:
    """Whether span content (prompt/completion text) is captured, and redaction."""

    capture_content: bool = False
    redact_patterns: tuple[re.Pattern[str], ...] = field(default=())

    def redact(self, text: str) -> str:
        for pattern in self.redact_patterns:
            text = pattern.sub(REDACTED, text)
        return text


@dataclass
class _State:
    backend: TracerBackend | None = None
    policy: ContentPolicy = field(default_factory=ContentPolicy)
    # The process that installed the backend. A forked worker inherits the
    # module state but not the exporter's threads, so it must not trace.
    pid: int = 0
    settings_key: Any = None


_state = _State()
_lock = threading.Lock()


def install(
    backend: TracerBackend,
    *,
    capture_content: bool = False,
    redact_patterns: tuple[str, ...] | list[str] = (),
    settings_key: Any = None,
) -> None:
    """Make ``backend`` the process's tracer, replacing (and shutting down) any.

    ``settings_key`` identifies the settings the backend was built from, so a
    caller can skip rebuilding it (see :func:`installed_settings`).
    """
    policy = ContentPolicy(
        capture_content=capture_content,
        redact_patterns=tuple(re.compile(p) for p in redact_patterns),
    )
    with _lock:
        previous = _state.backend
        _state.backend = backend
        _state.policy = policy
        _state.pid = os.getpid()
        _state.settings_key = settings_key
    if previous is not None and previous is not backend:
        previous.shutdown()


def uninstall() -> None:
    """Return to the no-op tracer, shutting the installed backend down."""
    with _lock:
        previous = _state.backend
        _state.backend = None
        _state.policy = ContentPolicy()
        _state.settings_key = None
    if previous is not None:
        previous.shutdown()


def installed_settings() -> Any:
    """The ``settings_key`` of the installed backend (``None`` when disabled)."""
    return _state.settings_key if _active_backend() is not None else None


def _active_backend() -> TracerBackend | None:
    backend = _state.backend
    if backend is None or _state.pid != os.getpid():
        return None
    return backend


def is_enabled() -> bool:
    """Whether a tracing backend is active in this process."""
    return _active_backend() is not None


def _merge(attributes: Mapping[str, Any] | None, kwargs: dict[str, Any]) -> dict:
    merged = dict(attributes) if attributes else {}
    for key, value in kwargs.items():
        merged[_NAMESPACE + key] = value
    return merged


def span(
    name: str, attributes: Mapping[str, Any] | None = None, **kwargs: Any
) -> AbstractContextManager[SpanHandle]:
    """A context manager that runs its block in a span named ``name``.

    Works in sync and async code alike (the active span lives in a
    ``contextvars`` context, so it follows ``await`` and the
    :class:`ContextThreadPoolExecutor` / ``asyncio.to_thread`` hops). Do not
    hold one across a ``yield`` of a generator another task resumes. An
    exception leaving the block is recorded on the span and re-raised.
    """
    backend = _active_backend()
    if backend is None:
        return _NOOP_CONTEXT
    return backend.use_span(name, _merge(attributes, kwargs))


def start_span(
    name: str, attributes: Mapping[str, Any] | None = None, **kwargs: Any
) -> SpanHandle:
    """Start a child span of the current one without activating it.

    The caller ends it (``handle.end()``); for spans opened and closed in
    separate callbacks.
    """
    backend = _active_backend()
    if backend is None:
        return NOOP_SPAN
    return backend.start_span(name, _merge(attributes, kwargs))


def set_attributes(attributes: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
    """Set attributes on the current span (no-op without one)."""
    backend = _active_backend()
    if backend is None:
        return
    backend.current_span().set_attributes(_merge(attributes, kwargs))


def current_trace_ids() -> tuple[str, str] | None:
    """``(trace_id, span_id)`` of the active span, for log correlation."""
    backend = _active_backend()
    if backend is None:
        return None
    return backend.trace_ids()


def content_policy() -> ContentPolicy:
    """The content-capture policy (capture off when tracing is disabled)."""
    if _active_backend() is None:
        return ContentPolicy()
    return _state.policy
