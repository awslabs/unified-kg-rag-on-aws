# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for NeptuneIndexer._execute_with_retries.

This retry-with-exponential-backoff loop is the documented mitigation for the
Neptune ``ConcurrentModificationException`` seen under concurrent index writes.
These tests pin the behaviors that matter: it returns on a later-attempt
success, it re-raises after ``max_attempts`` total attempts, it sleeps once per
failed attempt, and it fails fast on errors a retry cannot fix. The module's
``_sleep``/``_jitter`` seams are patched (never the process-global
``time.sleep``/``random.uniform``, which other threads share) so the tests are
fast and deterministic.
"""

from __future__ import annotations

import json

import aiohttp
import pytest
from gremlin_python.driver.protocol import GremlinServerError

from unified_kg_rag.adapters.aws.neptune import is_transient_neptune_error
from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.unit

_MOD = "unified_kg_rag.adapters.storage.neptune_indexer"


def _neptune_error(code: str) -> GremlinServerError:
    # Neptune puts its JSON error body in the Gremlin status message.
    body = json.dumps({"requestId": "r-1", "code": code, "detailedMessage": "x"})
    return GremlinServerError({"code": 500, "message": body, "attributes": {}})


_CME = _neptune_error("ConcurrentModificationException")


class _FlakyTraversal:
    """A fake GraphTraversal whose ``iterate()`` fails ``fail_times`` times."""

    def __init__(self, fail_times: int, exc: Exception) -> None:
        self._fail_times = fail_times
        self._exc = exc
        self.calls = 0

    def iterate(self) -> None:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc


@pytest.fixture
def indexer(mocker):
    mocker.patch("unified_kg_rag.adapters.storage.neptune_indexer.NeptuneClient")
    return NeptuneIndexer(config=Config())


@pytest.fixture(autouse=True)
def _no_sleep(mocker):
    # Keep retries instant and deterministic.
    mocker.patch(_MOD + "._sleep", return_value=None)
    mocker.patch(
        _MOD + "._jitter",
        return_value=0.0,
    )


def test_returns_after_transient_failures_then_success(indexer, mocker) -> None:
    indexer.neptune_config.max_attempts = 4
    indexer.neptune_config.retry_delay_seconds = 1
    sleep = mocker.patch(_MOD + "._sleep", return_value=None)
    # Fails twice (ConcurrentModificationException), succeeds on the 3rd attempt.
    traversal = _FlakyTraversal(fail_times=2, exc=_CME)
    indexer._execute_with_retries(traversal, "upsert entities")
    assert traversal.calls == 3
    assert sleep.call_count == 2  # one sleep per failed attempt


def test_reraises_after_max_attempts(indexer) -> None:
    indexer.neptune_config.max_attempts = 3
    indexer.neptune_config.retry_delay_seconds = 1
    traversal = _FlakyTraversal(fail_times=99, exc=_CME)
    with pytest.raises(GremlinServerError, match="ConcurrentModification"):
        indexer._execute_with_retries(traversal, "upsert entities")
    assert traversal.calls == 3  # max_attempts counts the first try


def test_non_transient_error_fails_on_first_attempt(indexer, mocker) -> None:
    indexer.neptune_config.max_attempts = 4
    sleep = mocker.patch(_MOD + "._sleep", return_value=None)
    traversal = _FlakyTraversal(
        fail_times=99, exc=_neptune_error("MalformedQueryException")
    )
    with pytest.raises(GremlinServerError, match="MalformedQuery"):
        indexer._execute_with_retries(traversal, "upsert entities")
    assert traversal.calls == 1
    sleep.assert_not_called()


def test_succeeds_on_first_attempt_does_not_sleep(indexer, mocker) -> None:
    indexer.neptune_config.max_attempts = 4
    indexer.neptune_config.retry_delay_seconds = 1
    sleep = mocker.patch(_MOD + "._sleep", return_value=None)
    traversal = _FlakyTraversal(fail_times=0, exc=Exception("never raised"))
    indexer._execute_with_retries(traversal, "upsert entities")
    assert traversal.calls == 1
    sleep.assert_not_called()


def test_single_attempt_raises_immediately(indexer) -> None:
    # max_attempts = 1 -> a single attempt, no retry.
    indexer.neptune_config.max_attempts = 1
    traversal = _FlakyTraversal(fail_times=99, exc=_CME)
    with pytest.raises(GremlinServerError):
        indexer._execute_with_retries(traversal, "upsert entities")
    assert traversal.calls == 1


def test_backoff_is_full_jitter_over_exponential_window(indexer, mocker) -> None:
    indexer.neptune_config.max_attempts = 4
    indexer.neptune_config.retry_delay_seconds = 2
    jitter = mocker.patch(_MOD + "._jitter", side_effect=lambda lo, hi: hi / 2)
    sleep = mocker.patch(_MOD + "._sleep", return_value=None)
    traversal = _FlakyTraversal(fail_times=3, exc=_neptune_error("ThrottlingException"))
    indexer._execute_with_retries(traversal, "upsert entities")
    # Window doubles per attempt: [0, 2], [0, 4], [0, 8]; sleep uses the draw.
    assert [c.args for c in jitter.call_args_list] == [(0, 2), (0, 4), (0, 8)]
    assert [c.args[0] for c in sleep.call_args_list] == [1, 2, 4]


@pytest.mark.parametrize(
    "exc",
    [
        _CME,
        _neptune_error("ThrottlingException"),
        _neptune_error("TooManyRequestsException"),
        _neptune_error("MemoryLimitExceededException"),
        ConnectionResetError("reset by peer"),
        TimeoutError("read timed out"),
        aiohttp.ServerDisconnectedError(),
    ],
)
def test_transient_errors_are_retryable(exc: BaseException) -> None:
    assert is_transient_neptune_error(exc)


@pytest.mark.parametrize(
    "exc",
    [
        _neptune_error("AccessDeniedException"),
        _neptune_error("BadRequestException"),
        _neptune_error("MalformedQueryException"),
        ValueError("bad traversal argument"),
        RuntimeError("unexpected"),
    ],
)
def test_other_errors_are_not_retryable(exc: BaseException) -> None:
    assert not is_transient_neptune_error(exc)
