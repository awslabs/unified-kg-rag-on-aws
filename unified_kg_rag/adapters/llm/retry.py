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
from typing import TypeVar

from langchain_core.exceptions import OutputParserException

from unified_kg_rag.domain.models.config import TransientRetryConfig
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import LLMOutputTruncatedError

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
# server or Ollama: they fail every item alike.
_CONFIGURATION_MESSAGE = re.compile(
    r"invalid model|model\W.{0,80}not (found|exist)|model_not_found|not supported"
    r"|unsupported|does not support|not permitted|unknown (parameter|field|argument)"
    r"|unrecognized|extra (inputs|fields)|extraneous key|cannot both"
    r"|max_tokens.{0,3} is too large|max_tokens\W{0,3}\d+ ?>|api.?key|quota",
    re.IGNORECASE,
)


def looks_like_item_error(exc: BaseException) -> bool:
    """Whether the server rejected the item's own request: HTTP 400/413/422
    (``status_code`` or ``response.status_code``) other than a rejection of
    the model or its settings (an unknown model, an unsupported parameter),
    which fails every item alike. The local factories' ``is_item_error``."""
    return _http_status(exc) in _ITEM_HTTP_STATUSES and not (
        _CONFIGURATION_MESSAGE.search(str(exc))
    )


def item_error_predicate(factory: object) -> ItemErrorPredicate:
    """Whether a batch item's last error was the item's own failure, which
    ``indexing.max_document_failures`` counts against its document.

    Positive only: a response that does not parse (``OutputParserException``)
    or was cut off (``LLMOutputTruncatedError``), or an error the factory's
    ``is_item_error`` names (a request the model rejects for its content).
    Any other error (throttling, 5xx, timeouts, connections, credentials,
    quotas, a bug) says nothing about the item and counts for none; a factory
    without ``is_item_error`` has only the two output failures counted.
    """
    hook: ItemErrorPredicate | None = getattr(factory, "is_item_error", None)

    def is_item_error(exc: BaseException) -> bool:
        if isinstance(exc, OutputParserException | LLMOutputTruncatedError):
            return True
        return hook is not None and isinstance(exc, Exception) and hook(exc)

    return is_item_error


def error_signature(exc: BaseException) -> tuple[str, str, int | None, str]:
    """The type, error code, HTTP status and message of ``exc``: errors alike
    in all four are taken for one cause."""
    return type(exc).__name__, _error_code(exc), _http_status(exc), str(exc)


def item_failed_indices(
    errors: Mapping[int, BaseException],
    document_ids: Sequence[Sequence[str] | None],
    is_item_error: ItemErrorPredicate,
    task: str,
) -> list[int]:
    """The indices, of a batch of items whose documents are ``document_ids``,
    whose ``errors`` count against their document (``is_item_error``).

    Empty when the model rejected every item, of more than one document, with
    the same error (:func:`error_signature`): one cause, a setting it rejects,
    rather than each document's own. A single document that always fails, and
    output that does not parse or was cut off, are still counted.
    """
    documents = {d for ids in document_ids for d in ids or []}
    signatures = {error_signature(error) for error in errors.values()}
    first = next(iter(errors.values()), None)
    if (
        len(documents) > 1
        and len(errors) == len(document_ids)
        and len(signatures) == 1
        and not isinstance(first, OutputParserException | LLMOutputTruncatedError)
    ):
        logger.warning(
            "%s: every item of %s documents failed with the same error (%s): "
            "taken for a model or configuration failure, counted against none "
            "of them",
            task,
            len(documents),
            first,
        )
        return []
    return [idx for idx, error in sorted(errors.items()) if is_item_error(error)]


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
