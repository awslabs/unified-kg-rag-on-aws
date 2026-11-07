# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM usage accounting: model calls, tokens and response-cache hits.

Always on, tracing or not. The model factories attach a callback
(``adapters.observability.usage_callback``) to every chat model they build;
it records each call here, into every :func:`usage_scope` open in the
calling context. The ingestion pipeline opens one per stage, the RAG
chain one per query (``RAGOutput.metadata["usage"]``), ``run-eval`` one per
run and one per answer attempt (so a failed answer keeps what it spent);
scopes nest, so a call counts in each enclosing one. A call made where
no scope is open is not recorded: the callback costs one context-variable
read.

Counts are kept per model role (the purpose the model was built for:
``query``, ``ingestion``, ``evaluation``; ``rerank`` for the reranker) and
per model id. A rerank request is no model call: it counts in
``rerank_calls`` and its billed ``rerank_search_units``. A response the
ingestion response cache served, a chunk-boundary answer the extraction
store served, or a query extraction the RAG chain's extraction cache
served, is a cache hit, not a call: it spends no tokens.

A call counts from the moment it starts (:func:`start_llm_call`), at an
estimate of its input tokens, and is settled when it ends: a returned call
by the tokens its response reports, a failed one (it returned no usage) is
taken back out of ``llm_calls`` and counted in ``failed_calls``. A call
that has not ended when a scope is read (one still running in the
background, as an auto query's unread speculative
extraction, or one cancelled in flight by a query deadline) counts with its
estimated input tokens and no output tokens, in ``estimated_calls`` as well
as ``llm_calls``: its request was sent (a cancelled task does not stop a
Bedrock call running on a worker thread), so it is billed. One that ends
later settles in the scopes still open (a ``run-eval`` run's); a snapshot
already taken keeps the estimate. Retries are calls too, but the requests
the SDK resends inside one call (botocore's own retries) are not: they are
counted apart, in ``request_retries``. ``input_tokens`` includes the prompt
cache reads and writes, also counted apart
(``cache_read_input_tokens`` / ``cache_write_input_tokens``) since they are
billed at other rates.

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
    "RERANK_DOCUMENTS_PER_UNIT",
    "LLMCall",
    "UsageMeter",
    "current_meter",
    "merge_usage",
    "metering",
    "record_cache_hit",
    "record_llm_call",
    "record_rerank_call",
    "start_llm_call",
    "usage_scope",
]

# The counters of every total, role and model, in snapshot order.
COUNT_FIELDS: tuple[str, ...] = (
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "cache_hits",
    "estimated_calls",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
    "request_retries",
    "failed_calls",
    "rerank_calls",
    "rerank_search_units",
)

# Documents one rerank search unit covers (Amazon Bedrock bills a rerank
# query per 100 documents; one with more counts as several).
RERANK_DOCUMENTS_PER_UNIT = 100

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
    estimated_calls: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    request_retries: int = 0
    failed_calls: int = 0
    rerank_calls: int = 0
    rerank_search_units: int = 0

    def add(self, other: _Counts) -> None:
        for name in COUNT_FIELDS:
            setattr(self, name, getattr(self, name) + getattr(other, name))

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


@contextmanager
def metering(meter: UsageMeter) -> Iterator[UsageMeter]:
    """Count the model calls made in this context into ``meter`` again (and
    its enclosing scopes' meters): the calls of one piece of work that runs
    interleaved with others, such as one run of a ``run-eval`` comparison
    whose answers alternate with another's, each into its own meter."""
    token = _CURRENT.set(meter)
    try:
        yield meter
    finally:
        _CURRENT.reset(token)


def current_meter() -> UsageMeter | None:
    """The innermost open scope's meter, or ``None``."""
    return _CURRENT.get()


def _record(role: str, model: str, counts: _Counts) -> None:
    _add_to(_CURRENT.get(), role, model, counts)


def _add_to(meter: UsageMeter | None, role: str, model: str, counts: _Counts) -> None:
    with _LOCK:
        while meter is not None:
            meter._add(role, model, counts)
            meter = meter.parent


class LLMCall:
    """One model call counted from its start (:func:`start_llm_call`), until
    it is settled: :meth:`finish`, :meth:`served_from_cache` or :meth:`fail`.

    Until then it counts as an estimated call at its estimated input tokens,
    in the scopes open where it started; settling replaces that estimate
    (once: a second settle does nothing).
    """

    def __init__(
        self, meter: UsageMeter | None, role: str, model: str, estimate: int
    ) -> None:
        self._meter = meter
        self._role = role
        self._model = model
        self._estimate = estimate
        self._settled = False
        _add_to(meter, role, model, self._pending())

    def _pending(self) -> _Counts:
        return _Counts(llm_calls=1, input_tokens=self._estimate, estimated_calls=1)

    def _settle(self, counts: _Counts) -> None:
        with _LOCK:
            if self._settled:
                return
            self._settled = True
        pending = self._pending()
        delta = _Counts(
            **{
                name: getattr(counts, name) - getattr(pending, name)
                for name in COUNT_FIELDS
            }
        )
        _add_to(self._meter, self._role, self._model, delta)

    def finish(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_input_tokens: int = 0,
        cache_write_input_tokens: int = 0,
        request_retries: int = 0,
    ) -> None:
        """The call returned, reporting these tokens (``input_tokens``
        includes the cache reads and writes) after ``request_retries``
        resent requests."""
        self._settle(
            _Counts(
                llm_calls=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                cache_write_input_tokens=cache_write_input_tokens,
                request_retries=request_retries,
            )
        )

    def finish_estimated(self, *, output_tokens: int, request_retries: int = 0) -> None:
        """The call returned reporting no tokens: it stays an estimated call
        at its estimated input tokens, with ``output_tokens`` estimated from
        its response, after ``request_retries`` resent requests."""
        self._settle(
            _Counts(
                llm_calls=1,
                input_tokens=self._estimate,
                output_tokens=output_tokens,
                estimated_calls=1,
                request_retries=request_retries,
            )
        )

    def served_from_cache(self) -> None:
        """The LLM response cache answered: a cache hit, not a call."""
        self._settle(_Counts(cache_hits=1))

    def fail(self, *, request_retries: int = 0) -> None:
        """The call failed and returned no usage: not a call, but a failed
        one (``failed_calls``), after ``request_retries`` resent requests."""
        self._settle(_Counts(failed_calls=1, request_retries=request_retries))


def start_llm_call(role: str, model: str, *, estimated_input_tokens: int) -> LLMCall:
    """Count a model call that is starting, in the open scopes (see
    :class:`LLMCall`)."""
    return LLMCall(_CURRENT.get(), role, model, estimated_input_tokens)


def record_llm_call(
    role: str,
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_write_input_tokens: int = 0,
    request_retries: int = 0,
    estimated: bool = False,
) -> None:
    """Record one finished model call that was not counted at its start
    (``estimated``: its tokens are estimates, not reported)."""
    _record(
        role,
        model,
        _Counts(
            llm_calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
            request_retries=request_retries,
            estimated_calls=int(estimated),
        ),
    )


def record_rerank_call(model: str, *, documents: int) -> None:
    """Record one rerank request of ``documents`` documents, under the
    ``rerank`` role: not a model call (``llm_calls``) but a rerank call and
    its search units (one per ``RERANK_DOCUMENTS_PER_UNIT`` documents),
    which Amazon Bedrock bills per thousand."""
    units = max(1, -(-documents // RERANK_DOCUMENTS_PER_UNIT))
    _record("rerank", model, _Counts(rerank_calls=1, rerank_search_units=units))


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
