# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Application-level retry for transient Bedrock runtime errors.

botocore's ``standard``/``adaptive`` retry modes only retry throttling, a fixed
set of 5xx codes, and connection-level errors. Bedrock surfaces several
transient model-side failures outside that set — most notably
``ModelErrorException`` (HTTP 424, "The system encountered an unexpected
error") and ``ModelNotReadyException`` — which botocore raises on the first
occurrence. This module adds a bounded exponential-backoff retry on top of the
botocore retries for exactly those transient failures, leaving validation,
access-denied and other client errors to fail fast. The bounds come from one
policy, ``aws.bedrock.transient_retry``.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from botocore import exceptions as boto_exc

from unified_kg_rag.domain.models.config import TransientRetryConfig
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

# Module-level indirection so tests can stub the backoff wait for this module
# only, without catching unrelated threads that call time.sleep.
_sleep = time.sleep

T = TypeVar("T")

# Error codes Bedrock runtime returns for conditions that resolve on their own.
# ThrottlingException / ServiceUnavailableException / InternalServerException
# are normally absorbed by botocore's own retries; they are listed so a burst
# that outlasts botocore's attempts still gets the backoff below.
TRANSIENT_BEDROCK_ERROR_CODES: frozenset[str] = frozenset(
    {
        "ModelErrorException",
        "ModelNotReadyException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ThrottlingException",
    }
)


def is_transient_bedrock_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is a Bedrock/botocore failure worth retrying."""
    if isinstance(exc, boto_exc.ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in TRANSIENT_BEDROCK_ERROR_CODES
    # Connect/read timeouts, endpoint-unreachable and dropped connections.
    return isinstance(exc, boto_exc.ConnectionError | boto_exc.HTTPClientError)


def _error_code(exc: BaseException) -> str:
    if isinstance(exc, boto_exc.ClientError):
        return str(exc.response.get("Error", {}).get("Code", type(exc).__name__))
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
    operation: str,
    attempt: int,
    started_at: float,
    policy: TransientRetryConfig,
) -> float | None:
    """Return the backoff before the next attempt, or None to give up on ``exc``."""
    if not is_transient_bedrock_error(exc) or attempt >= policy.max_attempts:
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
        "Transient Bedrock error on '%s' (attempt %s/%s): %s. Retrying in %.1fs",
        operation,
        attempt,
        policy.max_attempts,
        _error_code(exc),
        delay,
    )
    return delay


def call_with_transient_retry(
    func: Callable[[], T], *, operation: str, policy: TransientRetryConfig
) -> T:
    """Call ``func``, retrying transient Bedrock errors with bounded backoff.

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
    func: Callable[[], Awaitable[T]], *, operation: str, policy: TransientRetryConfig
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
                operation=operation,
                attempt=attempt,
                started_at=started_at,
                policy=policy,
            )
            if delay is None:
                raise
            await asyncio.sleep(delay)
            attempt += 1
