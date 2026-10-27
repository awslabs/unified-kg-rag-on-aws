# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded exponential-backoff retry of transient backend errors.

The caller says which errors are transient (``is_transient``): the language
model factory's ``is_transient_error`` (``transient_error_predicate``), so
validation, access-denied and other client errors fail fast. The bounds come
from one policy, ``aws.bedrock.transient_retry``.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from langchain_core.exceptions import OutputParserException

from unified_kg_rag.domain.models.config import TransientRetryConfig
from unified_kg_rag.domain.models.text_unit import TextUnit
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import LLMOutputTruncatedError

if TYPE_CHECKING:
    from unified_kg_rag.shared.utils.langchain import BatchProcessor

logger = get_logger(__name__)

# Module-level indirection so tests can stub the backoff wait for this module
# only, without catching unrelated threads that call time.sleep.
_sleep = time.sleep
# Backoff jitter is not a security context, but drawing it from the OS
# (CWE-338 scanners flag the seedable module PRNG) costs nothing at retry
# rates and keeps it independent of any random.seed() the caller makes.
_jitter_rng = random.SystemRandom()

T = TypeVar("T")

TransientErrorPredicate = Callable[[BaseException], bool]
ItemErrorPredicate = Callable[[BaseException], bool]

# HTTP statuses that resolve on their own: request timeout, rate limit, 5xx.
_TRANSIENT_HTTP_STATUSES = frozenset({408, 429})
# Transport-level failures of the HTTP clients the local providers use, matched
# by class name so this module imports none of them: httpx's TransportError
# (connect/read/write/pool timeouts, refused or dropped connections) and the
# OpenAI client's APIConnectionError (and its APITimeoutError subclass).
_TRANSPORT_ERROR_NAMES = frozenset({"TransportError", "APIConnectionError"})


def _http_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def looks_transient(exc: BaseException) -> bool:
    """Whether ``exc`` looks like a failure a new attempt can get past.

    Network errors and timeouts (the built-in ones, httpx's and the OpenAI
    client's), and HTTP 408/429/5xx responses — read from ``status_code``
    (``openai.APIStatusError``, ``ollama.ResponseError``) or
    ``response.status_code`` (``httpx.HTTPStatusError``). Any other error,
    including a 4xx such as an unknown model, fails fast.
    """
    if isinstance(exc, ConnectionError | TimeoutError):
        return True
    status = _http_status(exc)
    if status is not None:
        return status in _TRANSIENT_HTTP_STATUSES or 500 <= status < 600
    return any(cls.__name__ in _TRANSPORT_ERROR_NAMES for cls in type(exc).__mro__)


# HTTP statuses of a request the server rejects for what it carries.
_ITEM_HTTP_STATUSES = frozenset({400, 413, 422})
# Rejections of the request's settings, not its text, by an OpenAI-compatible
# server or Ollama: they fail every item alike. A context-length rejection
# counts the requested completion (``max_tokens``) with the prompt ("you
# requested N tokens"), not the prompt alone ("your messages resulted in").
_CONFIGURATION_MESSAGE = re.compile(
    r"invalid model|model\W.{0,80}not (found|exist)|model_not_found|not supported"
    r"|unsupported|does not support|not permitted|unknown (parameter|field|argument)"
    r"|unrecognized|extra (inputs|fields)|extraneous key|cannot both"
    r"|max_(completion_)?tokens.{0,3} is too large|max_tokens\W{0,3}\d+ ?>"
    r"|context length is \d+ tokens\W+however, you requested|api.?key|quota",
    re.IGNORECASE,
)


def looks_like_item_error(exc: BaseException) -> bool:
    """Whether the server rejected the item's own request: HTTP 400/413/422
    (``status_code`` or ``response.status_code``) other than a rejection of
    the model or its settings (an unknown model, an unsupported parameter),
    which fails every item alike. The local factories' ``is_item_error``, and
    the classifier of a factory without one."""
    return _http_status(exc) in _ITEM_HTTP_STATUSES and not (
        _CONFIGURATION_MESSAGE.search(str(exc))
    )


def item_error_predicate(factory: object) -> ItemErrorPredicate:
    """Whether a batch item's last error was the item's own failure, which
    ``indexing.max_document_failures`` counts against its document.

    Positive only: a response that does not parse (``OutputParserException``)
    or was cut off (``LLMOutputTruncatedError``), or an error the factory's
    ``is_item_error`` names (a request the model rejects for its content),
    else :func:`looks_like_item_error`, so a custom backend's HTTP 400/413/422
    counts without naming its error types. Any other error (throttling, 5xx,
    timeouts, connections, credentials, quotas, a bug) says nothing about the
    item and counts for none.
    """
    hook: ItemErrorPredicate | None = getattr(factory, "is_item_error", None)
    named = hook if hook is not None else looks_like_item_error

    def is_item_error(exc: BaseException) -> bool:
        if isinstance(exc, OutputParserException | LLMOutputTruncatedError):
            return True
        return isinstance(exc, Exception) and named(exc)

    return is_item_error


def _cause(exc: BaseException) -> tuple[str, str, int | None]:
    """What tells one cause of a failed item from another: output that does
    not parse or was cut off is one kind, any other error its type, error
    code and HTTP status. Not the message: a server's message can carry
    each request's own figures (an OpenAI-compatible context-length
    rejection names the request's token counts)."""
    if isinstance(exc, OutputParserException | LLMOutputTruncatedError):
        return "output", "", None
    return type(exc).__name__, _error_code(exc), _http_status(exc)


# The text of the known-good request (``known_good_unit``): short, neutral,
# and naming two entities and a relationship, so every extraction prompt
# has something to answer.
_KNOWN_GOOD_TEXT = "Vendor Corp and Buyer Ltd signed a supply agreement in 2020."


def known_good_unit() -> TextUnit:
    """The text unit of the known-good request :func:`item_failed_indices`
    sends when every item failed alike."""
    return TextUnit.model_validate(
        {"id": "known-good-request", "text": _KNOWN_GOOD_TEXT, "document_ids": []}
    )


def item_failed_indices(
    errors: Mapping[int, BaseException],
    units: Sequence[TextUnit],
    is_item_error: ItemErrorPredicate,
    task: str,
    batch_processor: BatchProcessor,
    prepare_inputs: Callable[[list[TextUnit]], list[dict[str, Any]]],
    invoke: Callable[[dict[str, Any]], Any],
    *,
    counted: bool = True,
) -> list[int]:
    """The indices, of the batch of ``units``, whose ``errors`` count
    against their document (``is_item_error``).

    When every item failed and its item errors are of one cause
    (:func:`_cause`: alike in type, error code and HTTP status, or all output
    that does not parse or was cut off; an error that is not an item's own,
    a throttled or timed-out call, says nothing of the others and is left
    out), that is either one cause, the model or a setting
    (a server-side output cap, say), or each document's own. Output errors
    on the units of more than one document count, without a known-good
    request: one is short, so it cannot tell a server-side output cap from
    output of each document's own, and a batch that always fails must not
    stop every run with its deletions. Each run they fail counts once
    toward ``indexing.max_document_failures``, as an error of their own
    does. Otherwise one known-good request
    (:func:`known_good_unit`, made by ``prepare_inputs`` and ``invoke``
    through ``batch_processor``, as the items were) decides: failed too,
    none is counted; answered, the errors are counted. For output errors
    the answer must also parse; for a rejection, being answered is enough.
    ``counted=False`` (claims, which are optional) only words the warnings
    so: their documents are indexed without them, counted for none.

    The request is neither served from nor stored in the run's LLM response
    cache (``uncached_responses``): a response an earlier run cached for it
    would pass for a model that now rejects the setting, and the cache's
    segments stay what the documents' requests made them.
    """
    own = {idx: error for idx, error in sorted(errors.items()) if is_item_error(error)}
    outcome = (
        "counted against their documents"
        if counted
        else "their documents are indexed without them, counted for none"
    )
    first = next(iter(own.values()), None)
    if (
        first is not None
        and len(errors) == len(units)
        and len({_cause(error) for error in own.values()}) == 1
    ):
        documents = {d for idx in own for d in units[idx].document_ids or []}
        if _cause(first)[0] == "output" and len(documents) > 1:
            logger.warning(
                "%s: the output of every item of %d documents failed alike "
                "(%s), possibly under a model or server output cap: taken "
                "for an error of each document's own, %s",
                task,
                len(documents),
                first,
                outcome,
            )
            return list(own)
        # Imported here: the cache module imports the chain factory, which
        # imports this one.
        from unified_kg_rag.adapters.llm.response_cache import uncached_responses

        with uncached_responses():
            passed = batch_processor.answered(
                known_good_unit(),
                prepare_inputs,
                invoke,
                f"{task} (known-good request)",
                parsed=_cause(first)[0] == "output",
            )
        if not passed:
            logger.warning(
                "%s: every item and a known-good request failed alike (%s): "
                "taken for a model or configuration failure, counted against "
                "no document",
                task,
                first,
            )
            return []
        logger.warning(
            "%s: every item failed alike (%s) but a known-good request was "
            "answered: %s",
            task,
            first,
            outcome,
        )
    return list(own)


def transient_error_predicate(factory: object) -> TransientErrorPredicate:
    """The transient-error classifier of a language-model ``factory``.

    The factory's ``is_transient_error`` when it defines one (the Bedrock
    factory classifies botocore errors, the local ones call
    :func:`looks_transient`), else :func:`looks_transient`, so a custom
    backend's network failures are retried without naming its error types.
    """
    predicate: TransientErrorPredicate | None = getattr(
        factory, "is_transient_error", None
    )
    return predicate if predicate is not None else looks_transient


def _error_code(exc: BaseException) -> str:
    # botocore ClientErrors carry the service's error code; name it in the log.
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if code:
            return str(code)
    return type(exc).__name__


def backoff_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    """Exponential backoff with equal jitter for the given 1-based attempt.

    The ceiling doubles per attempt (``base * 2**(attempt-1)``, capped at
    ``max_delay``); the delay is drawn from ``[ceiling/2, ceiling]`` so retries
    from concurrent workers spread out while still backing off meaningfully.
    """
    ceiling: float = min(max_delay, base_delay * 2.0 ** (attempt - 1))
    half = ceiling / 2
    return half + _jitter_rng.uniform(0, half)


def next_transient_retry_delay(
    exc: BaseException,
    *,
    is_transient: TransientErrorPredicate,
    operation: str,
    attempt: int,
    started_at: float,
    policy: TransientRetryConfig,
) -> float | None:
    """Return the backoff before the next attempt, or None to give up on ``exc``."""
    if not is_transient(exc) or attempt >= policy.max_attempts:
        return None
    delay = backoff_delay(attempt, policy.base_delay_seconds, policy.max_delay_seconds)
    if time.monotonic() - started_at + delay > policy.max_total_seconds:
        logger.warning(
            "Giving up on '%s' after %s attempts (%.1fs retry budget exhausted): %s",
            operation,
            attempt,
            policy.max_total_seconds,
            _error_code(exc),
        )
        return None
    logger.warning(
        "Transient error on '%s' (attempt %s/%s): %s. Retrying in %.1fs",
        operation,
        attempt,
        policy.max_attempts,
        _error_code(exc),
        delay,
    )
    return delay


def call_with_transient_retry(
    func: Callable[[], T],
    *,
    is_transient: TransientErrorPredicate,
    operation: str,
    policy: TransientRetryConfig,
) -> T:
    """Call ``func``, retrying transient errors with bounded backoff.

    Non-transient errors propagate immediately. When attempts or the wall-clock
    budget are exhausted, the last transient error is re-raised unchanged so
    callers' existing error handling keeps working.
    """
    started_at = time.monotonic()
    attempt = 1
    while True:
        try:
            return func()
        except Exception as exc:
            delay = next_transient_retry_delay(
                exc,
                is_transient=is_transient,
                operation=operation,
                attempt=attempt,
                started_at=started_at,
                policy=policy,
            )
            if delay is None:
                raise
            _sleep(delay)
            attempt += 1


async def acall_with_transient_retry(
    func: Callable[[], Awaitable[T]],
    *,
    is_transient: TransientErrorPredicate,
    operation: str,
    policy: TransientRetryConfig,
) -> T:
    """Async twin of :func:`call_with_transient_retry` (backs off without blocking)."""
    started_at = time.monotonic()
    attempt = 1
    while True:
        try:
            return await func()
        except Exception as exc:
            delay = next_transient_retry_delay(
                exc,
                is_transient=is_transient,
                operation=operation,
                attempt=attempt,
                started_at=started_at,
                policy=policy,
            )
            if delay is None:
                raise
            await asyncio.sleep(delay)
            attempt += 1
