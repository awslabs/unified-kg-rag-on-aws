# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded exponential-backoff retry of transient backend errors.

The caller says which errors are transient (``is_transient``; for Bedrock,
``adapters.aws.bedrock_retry.is_transient_bedrock_error``), so validation,
access-denied and other client errors fail fast. The bounds come from one
policy, ``aws.bedrock.transient_retry``.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from unified_kg_rag.domain.models.config import TransientRetryConfig
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

# Module-level indirection so tests can stub the backoff wait for this module
# only, without catching unrelated threads that call time.sleep.
_sleep = time.sleep

T = TypeVar("T")

TransientErrorPredicate = Callable[[BaseException], bool]


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
    return half + random.uniform(0, half)  # noqa: S311 - jitter, not crypto


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
