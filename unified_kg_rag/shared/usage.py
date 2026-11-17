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
``query``, ``ingestion``, ``evaluation``; ``embedding`` and ``rerank`` for
the embedding model's and reranker's requests) and
per model id. A rerank request is no model call: it counts in
``rerank_calls`` and its billed ``rerank_search_units``. A response the
ingestion response cache served, a chunk-boundary answer the extraction
store served, or a query extraction the RAG chain's extraction cache
served, is a cache hit, not a call: it spends no tokens.

A call counts from the moment it starts (:func:`start_llm_call`), at an
estimate of its input tokens, and is settled when it ends: a returned call
by the tokens its response reports, a failed one (it returned no usage) is
taken back out of ``llm_calls`` and counted in ``failed_calls`` (unless its
response had started: the callback keeps that one's estimate). A call
that has not ended when a scope is read (one still running in the
background, as an auto query's unread speculative
extraction, or one cancelled in flight by a query deadline) counts with its
estimated input tokens and no output tokens, in ``estimated_calls`` as well
as ``llm_calls``: its request was sent (a cancelled task does not stop a
Bedrock call running on a worker thread), so it is billed. One that ends
later settles in the scopes still open (a ``run-eval`` run's); a snapshot
already taken keeps the estimate. Retries are calls too, but the requests
the SDK resends inside one call (botocore's own retries) are not: they are
counted apart, in ``request_retries``. A botocore hook of the Bedrock
model clients (``adapters.aws.bedrock_throttles``: the language, embedding
and rerank models', not CountTokens' or the guardrail's) counts each
request as it comes back, in
``sent_requests``, and those Bedrock throttled in ``throttled_requests``.
``input_tokens`` includes the prompt cache reads and writes, also counted apart
(``cache_read_input_tokens`` / ``cache_write_input_tokens``) since they are
billed at other rates.

A scope may carry limits (:class:`UsageLimits`: model calls, input tokens,
a cost): a call that would take what the scope counted past one is refused
at its start (``start_llm_call`` raises), before its request is sent; a
query's limits (``search.query_max_*``) and an ingestion run's budget
(``processing.max_run_*``) are such limits. :func:`remaining` and
:func:`calls_that_fit` let a step with optional calls (a strategy's
fan-out) start only what fits.

A retry attempts one call again (:class:`RetriedCall`: the query chain's
retry of a transient model error, a stream cut before its answer text
included). The usage counts every attempt, as sent; a run budget does too,
since it bounds what was spent. A query's limits bound the calls the query
plans instead (``UsageLimits(counts_retries=False)``): a retry's attempt
reuses the room its failed attempt held, so a transient fault under a
binding limit is retried rather than refused.

The scope is a ``contextvars`` value, so it follows the code into the tasks
and threads it hands its context to (``asyncio`` tasks, ``to_thread``, the
package's ``ContextThreadPoolExecutor``), and concurrent queries, each in its
own task, keep separate counts.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from unified_kg_rag.shared.exceptions import (
    OptionalCallSkipped,
    UsageLimitExceededError,
)

__all__ = [
    "CACHED_RESPONSE_KEY",
    "COUNT_FIELDS",
    "RERANK_DOCUMENTS_PER_UNIT",
    "LLMCall",
    "Remaining",
    "RetriedCall",
    "UsageLimits",
    "UsageMeter",
    "cache_probed",
    "calls_that_fit",
    "optional_calls",
    "current_meter",
    "merge_usage",
    "metering",
    "record_cache_hit",
    "record_llm_call",
    "record_rerank_call",
    "record_request",
    "remaining",
    "rerank_search_units",
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
    "sent_requests",
    "throttled_requests",
    "failed_calls",
    "rerank_calls",
    "rerank_search_units",
)

# Documents one rerank search unit covers (Amazon Bedrock bills a rerank
# query per 100 documents; one with more counts as several).
RERANK_DOCUMENTS_PER_UNIT = 100
# Tokens one of those documents holds, the query's included: a longer one is
# billed as several documents (Amazon Bedrock pricing, Cohere Rerank 3.5).
RERANK_TOKENS_PER_DOCUMENT = 500

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
    sent_requests: int = 0
    throttled_requests: int = 0
    failed_calls: int = 0
    rerank_calls: int = 0
    rerank_search_units: int = 0

    def add(self, other: _Counts, sign: int = 1) -> None:
        for name in COUNT_FIELDS:
            setattr(self, name, getattr(self, name) + sign * getattr(other, name))

    def minus(self, other: _Counts) -> _Counts:
        counts = _Counts(**self.as_dict())
        counts.add(other, -1)
        return counts

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in COUNT_FIELDS}


# What a limit refusal is built from: the limit's name, its value, and what
# the scope would count with the refused call.
LimitRefusal = Callable[[str, float, float], BaseException]


def _refused(limit: str, value: float, would: float) -> BaseException:
    return UsageLimitExceededError(
        f"A model call would take {limit} to {would:g}, past its limit {value:g}; "
        "it was not sent",
        limit=limit,
        value=value,
    )


@dataclass(frozen=True)
class UsageLimits:
    """Caps on what the model calls counted in one scope may spend
    (``usage_scope(limits=...)``); ``None`` caps nothing.

    A call is checked when it starts, at its estimated input tokens, against
    what the scope counted so far (calls still running at their estimates):
    one that would take ``llm_calls`` past ``max_llm_calls``,
    ``input_tokens`` past ``max_input_tokens``, or ``cost`` past
    ``max_cost`` is refused, raising what ``refuse`` builds from the
    limit's name (``names``), its value and what the scope would count.
    ``cost`` prices the scope's counts by model (each a
    :data:`COUNT_FIELDS` mapping, the refused call's estimate included) and
    may raise itself (a model it has no price for); it runs under the
    module's lock, so it must not read a meter.

    ``counts_retries``: whether every attempt of a retried call
    (:class:`RetriedCall`) counts against the limits, as a run budget's
    must (each was sent and billed), or only its latest, as a query's
    limits do (they bound the calls the query plans: a retry reuses the
    room its failed attempt held). The usage counts every attempt either way.
    """

    max_llm_calls: int | None = None
    max_input_tokens: int | None = None
    max_cost: float | None = None
    # Of max_llm_calls and max_input_tokens, those kept for the scope's last
    # step (a query's answer: its call, and its prompt, question and least
    # context): an optional call (optional_calls) that would take any of
    # them is refused with OptionalCallSkipped.
    reserved_calls: int = 0
    reserved_tokens: int = 0
    counts_retries: bool = True
    cost: Callable[[Mapping[str, Mapping[str, int]]], float] | None = None
    refuse: LimitRefusal = _refused
    # The name each limit is reported by (the setting that sets it); one
    # not named here by its field's name.
    names: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_cost is not None and self.cost is None:
            raise ValueError("UsageLimits.max_cost needs a cost function")

    def _check(
        self,
        total: _Counts,
        by_model: Mapping[str, _Counts],
        model: str,
        call: _Counts,
    ) -> BaseException | None:
        """The refusal of ``call`` (on ``model``) where the limits count
        ``total``, ``by_model`` by model, if any."""
        if (
            self.max_llm_calls is not None
            and total.llm_calls + call.llm_calls > self.max_llm_calls
        ):
            return self.refuse(
                self.names.get("max_llm_calls", "max_llm_calls"),
                self.max_llm_calls,
                total.llm_calls + call.llm_calls,
            )
        if (
            self.max_input_tokens is not None
            and total.input_tokens + call.input_tokens > self.max_input_tokens
        ):
            return self.refuse(
                self.names.get("max_input_tokens", "max_input_tokens"),
                self.max_input_tokens,
                total.input_tokens + call.input_tokens,
            )
        if self.max_cost is not None and self.cost is not None:
            priced = {name: c.as_dict() for name, c in by_model.items()}
            pending = _Counts(**priced.get(model, {}))
            pending.add(call)
            priced[model] = pending.as_dict()
            try:
                cost = self.cost(priced)
            except Exception as e:  # noqa: BLE001 - the refusal names it
                return e
            if cost > self.max_cost:
                return self.refuse(
                    self.names.get("max_cost", "max_cost"), self.max_cost, cost
                )
        return None


@dataclass(frozen=True)
class Remaining:
    """What the limits of the open scopes leave (:func:`remaining`):
    ``None`` where none caps it; and the model-call limit that leaves the
    fewest calls, by its name (the setting that sets it) and value."""

    llm_calls: int | None = None
    input_tokens: int | None = None
    call_limit: str | None = None
    call_limit_value: int | None = None

    @property
    def unlimited(self) -> bool:
        """Whether no open scope caps model calls or input tokens."""
        return self.llm_calls is None and self.input_tokens is None


@dataclass
class UsageMeter:
    """The usage recorded in one scope (and the scopes it encloses)."""

    parent: UsageMeter | None = None
    total: _Counts = field(default_factory=_Counts)
    by_role: dict[str, _Counts] = field(default_factory=dict)
    by_model: dict[str, _Counts] = field(default_factory=dict)
    limits: UsageLimits | None = None
    # Of total and by_model, what the earlier attempts of retried calls
    # count (RetriedCall): limits that count each call once leave it out.
    retried: _Counts = field(default_factory=_Counts)
    retried_by_model: dict[str, _Counts] = field(default_factory=dict)

    def _add(self, role: str, model: str, counts: _Counts) -> None:
        self.total.add(counts)
        self.by_role.setdefault(role, _Counts()).add(counts)
        self.by_model.setdefault(model, _Counts()).add(counts)

    def _add_retried(self, model: str, counts: _Counts) -> None:
        self.retried.add(counts)
        self.retried_by_model.setdefault(model, _Counts()).add(counts)

    def _limited(
        self, superseding: Sequence[LLMCall] = ()
    ) -> tuple[_Counts, dict[str, _Counts]]:
        """What this scope's limits count, in total and by model, once the
        calls of ``superseding`` (earlier attempts a retry's call is about
        to supersede) are, for limits that count a call once, left out
        (the caller holds the module's lock)."""
        total = _Counts(**self.total.as_dict())
        by_model = {name: _Counts(**c.as_dict()) for name, c in self.by_model.items()}
        if self.limits is None or self.limits.counts_retries:
            return total, by_model
        left_out: list[Mapping[str, _Counts]] = [self.retried_by_model]
        left_out += [
            {call._model: call._counted}
            for call in superseding
            # Only from the scopes that counted it. Every attempt of a
            # retried call starts its calls in the meter open where the
            # retry runs, which is the one the package's limited scopes
            # (a query's, a run's) enclose, so this filter drops nothing
            # there; it keeps a limited scope opened inside one attempt from
            # being credited with a call it never counted. By identity:
            # meters are dataclasses, equal when their counts are.
            if any(node is self for node in _chain(call._meter))
        ]
        for counts_by_model in left_out:
            for name, counts in counts_by_model.items():
                total.add(counts, -1)
                by_model.setdefault(name, _Counts()).add(counts, -1)
        return total, by_model

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

# The calls started here that a response cache is still to look up
# (``start_llm_call(probes_cache=True)``), oldest first: a new list each time
# one starts, so the threads and tasks that copied this context keep their own.
_PROBING: ContextVar[list[LLMCall] | None] = ContextVar(
    "llm_usage_probing", default=None
)


@contextmanager
def usage_scope(limits: UsageLimits | None = None) -> Iterator[UsageMeter]:
    """Count the model calls made in this context into a new meter (and the
    enclosing scopes' meters), refusing those past ``limits``."""
    meter = UsageMeter(parent=_CURRENT.get(), limits=limits)
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
        for node in _chain(meter):
            node._add(role, model, counts)


def _chain(meter: UsageMeter | None) -> list[UsageMeter]:
    """``meter`` and its enclosing scopes' meters, innermost first."""
    chain = []
    while meter is not None:
        chain.append(meter)
        meter = meter.parent
    return chain


def _admit(call: LLMCall) -> None:
    """Count ``call`` (starting) in its meter and their parents, unless the
    limits of one of them refuse it: then raise the refusal and count
    nothing. A retry's call supersedes the calls of its earlier attempts
    (:class:`RetriedCall`) for the limits that count a call once."""
    refusal: BaseException | None = None
    optional = _OPTIONAL.get()
    counts = call._pending()
    with _LOCK:
        superseding = call._attempt.superseded() if call._attempt else []
        for node in _chain(call._meter):
            if node.limits is None:
                continue
            total, by_model = node._limited(superseding)
            # An optional call past what the answer keeps is skipped
            # (OptionalCallSkipped) before any limit refuses it: one past a
            # limit itself is past that room too.
            if optional:
                refusal = _optional_refusal(node.limits, total, counts)
            if refusal is None:
                refusal = node.limits._check(total, by_model, call._model, counts)
            if refusal is not None:
                break
        if refusal is None:
            for node in _chain(call._meter):
                node._add(call._role, call._model, counts)
            call._counted = counts
            for earlier in superseding:
                earlier._superseded = True
                for node in _chain(earlier._meter):
                    node._add_retried(earlier._model, earlier._counted)
            if call._attempt is not None:
                call._attempt.admitted(call)
    if refusal is not None:
        raise refusal


def remaining() -> Remaining:
    """What the limits of the scopes open here leave: the fewest calls and
    input tokens any of them still allows (``None`` where none caps it)."""
    calls: int | None = None
    tokens: int | None = None
    call_limit: tuple[str, int] | None = None
    with _LOCK:
        for node in _chain(_CURRENT.get()):
            limits = node.limits
            if limits is None:
                continue
            total, _ = node._limited()
            if limits.max_llm_calls is not None:
                left = max(0, limits.max_llm_calls - total.llm_calls)
                if calls is None or left < calls:
                    calls = left
                    call_limit = (
                        limits.names.get("max_llm_calls", "max_llm_calls"),
                        limits.max_llm_calls,
                    )
            if limits.max_input_tokens is not None:
                left = max(0, limits.max_input_tokens - total.input_tokens)
                tokens = left if tokens is None else min(tokens, left)
    return Remaining(
        llm_calls=calls,
        input_tokens=tokens,
        call_limit=call_limit[0] if call_limit else None,
        call_limit_value=call_limit[1] if call_limit else None,
    )


# Whether the model calls made here are optional (optional_calls).
_OPTIONAL: ContextVar[bool] = ContextVar("llm_usage_optional", default=False)


@contextmanager
def optional_calls() -> Iterator[None]:
    """The model calls made in this context are optional: one that would
    take a call or input tokens the open scopes keep for their last step
    (``UsageLimits.reserved_calls`` / ``reserved_tokens``: a query's
    answer) is refused with
    :class:`OptionalCallSkipped`, which the step making it answers without
    (a query translation or extraction, the structured planner, an output
    repair). The one rule of every optional call."""
    token = _OPTIONAL.set(True)
    try:
        yield
    finally:
        _OPTIONAL.reset(token)


def _optional_refusal(
    limits: UsageLimits, total: _Counts, call: _Counts
) -> BaseException | None:
    """The refusal of an optional ``call`` that would take a call or input
    tokens ``limits`` keep for their scope's last step (none kept: past
    the limit), where they count ``total``, if any."""
    if limits.max_llm_calls is not None:
        room = limits.max_llm_calls - limits.reserved_calls
        if total.llm_calls + call.llm_calls > room:
            name = limits.names.get("max_llm_calls", "max_llm_calls")
            return OptionalCallSkipped(
                (
                    "An optional model call would take one of the "
                    f"{limits.reserved_calls} call(s) {name} "
                    f"{limits.max_llm_calls:g} keeps for the answer"
                    if limits.reserved_calls
                    else "An optional model call would take "
                    f"{name} past {limits.max_llm_calls:g}"
                )
                + "; it was not sent",
                limit=name,
                value=limits.max_llm_calls,
            )
    if limits.max_input_tokens is not None:
        room = limits.max_input_tokens - limits.reserved_tokens
        if total.input_tokens + call.input_tokens > room:
            name = limits.names.get("max_input_tokens", "max_input_tokens")
            return OptionalCallSkipped(
                (
                    "An optional model call would take some of the "
                    f"{limits.reserved_tokens} input tokens {name} "
                    f"{limits.max_input_tokens:g} keeps for the answer"
                    if limits.reserved_tokens
                    else "An optional model call would take "
                    f"{name} past {limits.max_input_tokens:g}"
                )
                + "; it was not sent",
                limit=name,
                value=limits.max_input_tokens,
            )
    return None


def calls_that_fit(
    input_tokens: Sequence[int], *, reserve_calls: int = 0, reserve_tokens: int = 0
) -> int:
    """How many of the leading calls (one each, of ``input_tokens``
    estimated input) the open scopes' limits leave room for, after
    ``reserve_calls`` and ``reserve_tokens`` kept for what must follow
    (the answer). All of them without limits."""
    left = remaining()
    fit = len(input_tokens)
    if left.llm_calls is not None:
        fit = min(fit, max(0, left.llm_calls - reserve_calls))
    if left.input_tokens is not None:
        room = left.input_tokens - reserve_tokens
        used = 0
        for index, cost in enumerate(input_tokens[:fit]):
            used += max(0, cost)
            if used > room:
                fit = index
                break
    return fit


# The attempt of a retried call the model calls made here belong to.
_ATTEMPT: ContextVar[_Attempt | None] = ContextVar("llm_usage_attempt", default=None)


class RetriedCall:
    """One call that a retry may attempt several times (the query chain's
    retry of a transient model error): each attempt is entered with
    ``with attempt:`` for :meth:`attempt`'s value, re-entrant (a stream's
    attempt around each chunk it reads).

    The model calls started in an attempt count as usual, in the usage and
    in every limit. Once a call of a later attempt is admitted, the calls of
    the earlier ones stop counting against the limits that count a call
    once (``UsageLimits(counts_retries=False)``, a query's): the retry
    reuses the room its failed attempt held, whatever that attempt was
    settled at (a stream cut after its response started keeps its
    estimate). Limits that count retries, and the usage, still count them.
    """

    def __init__(self) -> None:
        self._attempts = 0
        # The admitted calls of the latest attempt that made one, by attempt.
        self._calls: list[tuple[int, LLMCall]] = []

    def attempt(self) -> _Attempt:
        """The next attempt."""
        self._attempts += 1
        return _Attempt(self, self._attempts)


class _Attempt:
    """One attempt of a :class:`RetriedCall` (see there)."""

    def __init__(self, call: RetriedCall, number: int) -> None:
        self._call = call
        self._number = number
        self._tokens: list[Token[_Attempt | None]] = []

    def __enter__(self) -> _Attempt:
        self._tokens.append(_ATTEMPT.set(self))
        return self

    def __exit__(self, *exc: object) -> None:
        _ATTEMPT.reset(self._tokens.pop())

    def superseded(self) -> list[LLMCall]:
        """The calls of the earlier attempts a call of this one supersedes
        (the caller holds the module's lock)."""
        return [call for number, call in self._call._calls if number < self._number]

    def admitted(self, call: LLMCall) -> None:
        """``call`` of this attempt was admitted: the earlier attempts'
        calls are superseded (the caller holds the module's lock)."""
        kept = self._call._calls
        kept[:] = [(n, c) for n, c in kept if n >= self._number]
        kept.append((self._number, call))


class LLMCall:
    """One model call counted from its start (:func:`start_llm_call`), until
    it is settled: :meth:`finish`, :meth:`served_from_cache` or :meth:`fail`.

    Until then it counts as an estimated call at its estimated input tokens,
    in the scopes open where it started; settling replaces that estimate
    (once: a second settle does nothing). Started in an attempt of a
    :class:`RetriedCall`, it is one of that attempt's calls.
    """

    def __init__(
        self,
        meter: UsageMeter | None,
        role: str,
        model: str,
        estimate: int,
        *,
        probes_cache: bool = False,
    ) -> None:
        self._meter = meter
        self._role = role
        self._model = model
        self._estimate = estimate
        self._settled = False
        self._attempt = _ATTEMPT.get()
        # What the call counts in its scopes (nothing until admitted), and
        # whether a later attempt of its retried call superseded it.
        self._counted = _Counts()
        self._superseded = False
        if probes_cache:
            # Counted (and checked) once the cache missed (cache_probed): a
            # hit is no call and must not hold a limit's room while in
            # flight.
            _PROBING.set([*(_PROBING.get() or ()), self])
        else:
            self._admit()

    def _admit(self) -> None:
        _admit(self)

    @property
    def estimated_input_tokens(self) -> int:
        """The input tokens estimated at the call's start."""
        return self._estimate

    def _pending(self) -> _Counts:
        return _Counts(llm_calls=1, input_tokens=self._estimate, estimated_calls=1)

    def _settle(self, counts: _Counts) -> None:
        with _LOCK:
            if self._settled:
                return
            self._settled = True
            probing = _PROBING.get()
            if probing and self in probing:
                probing.remove(self)
            # A call the cache was still to look up was never counted: its
            # settle adds what it is (a hit, or a call no lookup preceded).
            delta = counts.minus(self._counted)
            self._counted = counts
            for node in _chain(self._meter):
                node._add(self._role, self._model, delta)
                if self._superseded:
                    node._add_retried(self._model, delta)

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


def start_llm_call(
    role: str, model: str, *, estimated_input_tokens: int, probes_cache: bool = False
) -> LLMCall:
    """Count a model call that is starting, in the open scopes (see
    :class:`LLMCall`). ``probes_cache``: a response cache looks the call up
    before its request is sent (the ingestion response cache), so it is
    counted and checked against the limits only once the cache missed
    (:func:`cache_probed`); a hit is no call.

    Raises:
        BaseException: What a scope's :class:`UsageLimits` raise for a call
            past them (``UsageLimitExceededError`` by default); nothing is
            counted then.
    """
    return LLMCall(
        _CURRENT.get(),
        role,
        model,
        estimated_input_tokens,
        probes_cache=probes_cache,
    )


def cache_probed(*, hit: bool) -> None:
    """A response cache looked up the oldest call started here with
    ``probes_cache``: a miss counts it now, checked against the open
    scopes' limits, before its request is sent; a hit leaves it to settle
    as one (:meth:`LLMCall.served_from_cache`).

    Raises:
        BaseException: The limits' refusal of a missed call, which is then
            counted nowhere.
    """
    probing = _PROBING.get()
    if not probing:
        return
    call = probing.pop(0)
    if hit:
        return
    try:
        call._admit()
    except BaseException:
        call._settled = True
        raise


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


def rerank_search_units(
    query_tokens: int,
    document_tokens: Sequence[int] | None = None,
    *,
    documents: int = 0,
) -> int:
    """The search units Amazon Bedrock bills one rerank request: one per
    ``RERANK_DOCUMENTS_PER_UNIT`` documents, where a document longer than
    ``RERANK_TOKENS_PER_DOCUMENT`` tokens with the query
    (``query_tokens``) counts as one per that many of its own tokens
    (``document_tokens``, one count each). Without the counts,
    ``documents`` documents of one unit each. At least one unit."""
    if document_tokens is None:
        chunks = documents
    else:
        room = max(1, RERANK_TOKENS_PER_DOCUMENT - max(0, query_tokens))
        chunks = sum(max(1, -(-max(0, tokens) // room)) for tokens in document_tokens)
    return max(1, -(-chunks // RERANK_DOCUMENTS_PER_UNIT))


def record_rerank_call(
    model: str, *, documents: int, search_units: int | None = None
) -> None:
    """Record one rerank request of ``documents`` documents, under the
    ``rerank`` role: not a model call (``llm_calls``) but a rerank call and
    its ``search_units`` (:func:`rerank_search_units`; by default one per
    ``RERANK_DOCUMENTS_PER_UNIT`` documents), which Amazon Bedrock bills
    per thousand."""
    units = (
        search_units
        if search_units is not None
        else rerank_search_units(0, documents=documents)
    )
    _record("rerank", model, _Counts(rerank_calls=1, rerank_search_units=units))


def record_request(role: str, model: str, *, throttled: bool = False) -> None:
    """Record one request sent to the provider (an attempt, not a call: the
    SDK's resends of one call count each), and whether it was throttled
    (HTTP 429), whether the SDK resent it or the call failed on it."""
    _record(role, model, _Counts(sent_requests=1, throttled_requests=int(throttled)))


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
