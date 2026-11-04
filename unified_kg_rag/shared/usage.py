# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM usage accounting: model calls, tokens and response-cache hits.

Always on, tracing or not. The model factories attach a callback
(``adapters.observability.usage_callback``) to every chat model they build;
it records each finished call here, into every :func:`usage_scope` open in
the calling context. The ingestion pipeline opens one per stage, the RAG
chain one per query (``RAGOutput.metadata["usage"]``), ``run-eval`` one per
run and one per answer attempt (so a failed answer keeps what it spent);
scopes nest, so a call counts in each enclosing one. A call made where
no scope is open is not recorded: the callback costs one context-variable
read.

Counts are kept per model role (the purpose the model was built for:
``query``, ``ingestion``, ``evaluation``) and per model id. A response the
ingestion response cache served, a chunk-boundary answer the extraction
store served, or a query extraction the RAG chain's extraction cache
served, is a cache hit, not a call: it spends no tokens. A call that failed is not counted (it returned no usage); retries
and calls abandoned on a timeout whose response still arrived are.

The scope is a ``contextvars`` value, so it follows the code into the tasks
and threads it hands its context to (``asyncio`` tasks, ``to_thread``, the
package's ``ContextThreadPoolExecutor``), and concurrent queries, each in its
own task, keep separate counts.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CACHED_RESPONSE_KEY",
    "COUNT_FIELDS",
    "UsageMeter",
    "current_meter",
    "merge_usage",
    "record_cache_hit",
    "record_llm_call",
    "usage_scope",
]

# The counters of every total, role and model, in snapshot order.
COUNT_FIELDS: tuple[str, ...] = (
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "cache_hits",
)

# Set in the ``generation_info`` of a response the LLM response cache served,
# so the usage callback counts it as a cache hit rather than a call.
CACHED_RESPONSE_KEY = "kg_rag_cached_response"

_LOCK = threading.Lock()


@dataclass
class _Counts:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0

    def add(self, other: _Counts) -> None:
        self.llm_calls += other.llm_calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_hits += other.cache_hits

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in COUNT_FIELDS}


@dataclass
class UsageMeter:
    """The usage recorded in one scope (and the scopes it encloses)."""

    parent: UsageMeter | None = None
    total: _Counts = field(default_factory=_Counts)
    by_role: dict[str, _Counts] = field(default_factory=dict)
    by_model: dict[str, _Counts] = field(default_factory=dict)

    def _add(self, role: str, model: str, counts: _Counts) -> None:
        self.total.add(counts)
        self.by_role.setdefault(role, _Counts()).add(counts)
        self.by_model.setdefault(model, _Counts()).add(counts)

    def snapshot(self) -> dict[str, Any]:
        """Plain-data copy: the totals, ``by_role`` and ``by_model``
        (``domain.models.LLMUsage`` validates it)."""
        with _LOCK:
            return {
                **self.total.as_dict(),
                "by_role": {k: v.as_dict() for k, v in sorted(self.by_role.items())},
                "by_model": {k: v.as_dict() for k, v in sorted(self.by_model.items())},
            }


_CURRENT: ContextVar[UsageMeter | None] = ContextVar("llm_usage_meter", default=None)


@contextmanager
def usage_scope() -> Iterator[UsageMeter]:
    """Count the model calls made in this context into a new meter (and the
    enclosing scopes' meters)."""
    meter = UsageMeter(parent=_CURRENT.get())
    token = _CURRENT.set(meter)
    try:
        yield meter
    finally:
        _CURRENT.reset(token)


def current_meter() -> UsageMeter | None:
    """The innermost open scope's meter, or ``None``."""
    return _CURRENT.get()


def _record(role: str, model: str, counts: _Counts) -> None:
    meter = _CURRENT.get()
    if meter is None:
        return
    with _LOCK:
        while meter is not None:
            meter._add(role, model, counts)
            meter = meter.parent


def record_llm_call(
    role: str, model: str, *, input_tokens: int, output_tokens: int
) -> None:
    """Record one finished model call."""
    _record(
        role,
        model,
        _Counts(llm_calls=1, input_tokens=input_tokens, output_tokens=output_tokens),
    )


def record_cache_hit(role: str, model: str) -> None:
    """Record one response served from the LLM response cache."""
    _record(role, model, _Counts(cache_hits=1))


def merge_usage(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """The sum of :meth:`UsageMeter.snapshot` dicts (missing counters are 0)."""
    total = _Counts()
    by: dict[str, dict[str, _Counts]] = {"by_role": {}, "by_model": {}}
    for snapshot in snapshots:
        total.add(_counts(snapshot))
        for key, groups in by.items():
            for name, counts in (snapshot.get(key) or {}).items():
                groups.setdefault(name, _Counts()).add(_counts(counts))
    return {
        **total.as_dict(),
        **{
            key: {name: c.as_dict() for name, c in sorted(groups.items())}
            for key, groups in by.items()
        },
    }


def _counts(data: dict[str, Any]) -> _Counts:
    return _Counts(**{name: int(data.get(name) or 0) for name in COUNT_FIELDS})
