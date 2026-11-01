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
import random

import aiohttp
import pytest
from gremlin_python.driver.protocol import GremlinServerError

from unified_kg_rag.adapters.aws.neptune import is_permanent_neptune_error
from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer
from unified_kg_rag.domain.models import Config
from unified_kg_rag.ports.indexer import IndexingStats

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


class _ResultTraversal(_FlakyTraversal):
    """A fake traversal whose ``toList()`` fails, then returns ``result``."""

    def __init__(self, fail_times: int, exc: Exception, result: list) -> None:
        super().__init__(fail_times, exc)
        self._result = result

    def toList(self) -> list:  # noqa: N802 - gremlin-python method name
        self.iterate()
        return self._result


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
    indexer.config.retry.graph_store_max_attempts = 4
    indexer.config.retry.base_delay_seconds = 1
    sleep = mocker.patch(_MOD + "._sleep", return_value=None)
    # Fails twice (ConcurrentModificationException), succeeds on the 3rd attempt.
    traversal = _FlakyTraversal(fail_times=2, exc=_CME)
    indexer._execute_with_retries(lambda _g: traversal, "upsert entities")
    assert traversal.calls == 3
    assert sleep.call_count == 2  # one sleep per failed attempt


def test_reraises_after_max_attempts(indexer) -> None:
    indexer.config.retry.graph_store_max_attempts = 3
    indexer.config.retry.base_delay_seconds = 1
    traversal = _FlakyTraversal(fail_times=99, exc=_CME)
    with pytest.raises(GremlinServerError, match="ConcurrentModification"):
        indexer._execute_with_retries(lambda _g: traversal, "upsert entities")
    assert traversal.calls == 3  # max_attempts counts the first try


def test_permanent_error_fails_on_first_attempt(indexer, mocker) -> None:
    indexer.config.retry.graph_store_max_attempts = 4
    sleep = mocker.patch(_MOD + "._sleep", return_value=None)
    traversal = _FlakyTraversal(
        fail_times=99, exc=_neptune_error("MalformedQueryException")
    )
    with pytest.raises(GremlinServerError, match="MalformedQuery"):
        indexer._execute_with_retries(lambda _g: traversal, "upsert entities")
    assert traversal.calls == 1
    sleep.assert_not_called()


def test_succeeds_on_first_attempt_does_not_sleep(indexer, mocker) -> None:
    indexer.config.retry.graph_store_max_attempts = 4
    indexer.config.retry.base_delay_seconds = 1
    sleep = mocker.patch(_MOD + "._sleep", return_value=None)
    traversal = _FlakyTraversal(fail_times=0, exc=Exception("never raised"))
    indexer._execute_with_retries(lambda _g: traversal, "upsert entities")
    assert traversal.calls == 1
    sleep.assert_not_called()


def test_zero_retry_delay_is_accepted_and_retries_at_once(mocker) -> None:
    # base_delay_seconds is ge=0 and documented "0 retries at once"; a
    # validator used to reject 0 whenever max_attempts > 1 (the default 4).
    mocker.patch(_MOD + ".NeptuneClient")
    config = Config(retry={"base_delay_seconds": 0})
    indexer = NeptuneIndexer(config=config)
    jitter = mocker.patch(_MOD + "._jitter", side_effect=lambda lo, hi: hi)
    traversal = _FlakyTraversal(fail_times=2, exc=_CME)
    indexer._execute_with_retries(lambda _g: traversal, "upsert entities")
    assert traversal.calls == 3
    assert [c.args for c in jitter.call_args_list] == [(0, 0), (0, 0)]


def test_single_attempt_raises_immediately(indexer) -> None:
    # max_attempts = 1 -> a single attempt, no retry.
    indexer.config.retry.graph_store_max_attempts = 1
    traversal = _FlakyTraversal(fail_times=99, exc=_CME)
    with pytest.raises(GremlinServerError):
        indexer._execute_with_retries(lambda _g: traversal, "upsert entities")
    assert traversal.calls == 1


def test_backoff_is_full_jitter_over_exponential_window(indexer, mocker) -> None:
    indexer.config.retry.graph_store_max_attempts = 4
    indexer.config.retry.base_delay_seconds = 2
    jitter = mocker.patch(_MOD + "._jitter", side_effect=lambda lo, hi: hi / 2)
    sleep = mocker.patch(_MOD + "._sleep", return_value=None)
    traversal = _FlakyTraversal(fail_times=3, exc=_neptune_error("ThrottlingException"))
    indexer._execute_with_retries(lambda _g: traversal, "upsert entities")
    # Window doubles per attempt: [0, 2], [0, 4], [0, 8]; sleep uses the draw.
    assert [c.args for c in jitter.call_args_list] == [(0, 2), (0, 4), (0, 8)]
    assert [c.args[0] for c in sleep.call_args_list] == [1, 2, 4]


@pytest.mark.parametrize(
    "exc",
    [
        _CME,
        _neptune_error("ThrottlingException"),
        _neptune_error("TimeLimitExceededException"),
        ConnectionResetError("reset by peer"),
        TimeoutError("read timed out"),
        aiohttp.ServerDisconnectedError(),
        RuntimeError("Connection was already closed."),
    ],
)
def test_other_errors_are_retried(exc: BaseException) -> None:
    assert not is_permanent_neptune_error(exc)


@pytest.mark.parametrize(
    "exc",
    [
        _neptune_error("AccessDeniedException"),
        _neptune_error("BadRequestException"),
        _neptune_error("MalformedQueryException"),
        ValueError("bad traversal argument"),
    ],
)
def test_permanent_errors_fail_fast(exc: BaseException) -> None:
    assert is_permanent_neptune_error(exc)


def test_results_mode_returns_the_traversal_results(indexer) -> None:
    indexer.config.retry.graph_store_max_attempts = 3
    traversal = _ResultTraversal(fail_times=1, exc=_CME, result=["r1"])
    assert indexer._execute_with_retries(
        lambda _g: traversal, "edge", results=True
    ) == ["r1"]
    assert traversal.calls == 2


def test_results_mode_returns_empty_when_nothing_matched(indexer) -> None:
    traversal = _ResultTraversal(fail_times=0, exc=_CME, result=[])
    assert (
        indexer._execute_with_retries(lambda _g: traversal, "edge", results=True) == []
    )


# --- a connection the server closed (restart, failover) --------------------
#
# Regression: the retries reused the pooled websocket the server had closed,
# which gremlinpython never reopens, so after a Neptune restart or failover
# every write failed every attempt until the process exited.


class _Connection:
    def __init__(self) -> None:
        self.closed = False

    def is_closed(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True


class _Source:
    """A traversal source bound to one connection (``traversal().withRemote``)."""

    def __init__(self, connection: _Connection) -> None:
        self.connection = connection


def test_a_lost_connection_is_replaced_before_the_retry(monkeypatch) -> None:
    from unified_kg_rag.adapters.aws import neptune as neptune_module
    from unified_kg_rag.adapters.aws.neptune import NeptuneClient

    opened: list[_Connection] = []

    def _connect(self: NeptuneClient) -> _Connection:
        opened.append(_Connection())
        return opened[-1]

    monkeypatch.setattr(NeptuneClient, "_create_connection", _connect)
    monkeypatch.setattr(
        neptune_module,
        "traversal",
        lambda: type("_T", (), {"withRemote": staticmethod(_Source)})(),
    )
    config = Config()
    config.aws.neptune.endpoint = "neptune.example.invalid"
    config.retry.graph_store_max_attempts = 3
    indexer = NeptuneIndexer(config=config)
    lost = RuntimeError("Connection was already closed.")
    built_on: list[_Source] = []

    def build(g: _Source) -> _FlakyTraversal:
        built_on.append(g)
        # Every request over the first connection fails as a closed socket.
        return _FlakyTraversal(99 if g.connection is opened[0] else 0, lost)

    indexer._execute_with_retries(build, "upsert entities")  # type: ignore[arg-type]

    assert len(opened) == 2 and opened[0].closed and not opened[1].closed
    assert [g.connection for g in built_on] == opened


def test_a_failed_batch_the_fallback_indexes_logs_no_error(
    indexer, mocker, caplog
) -> None:
    # Regression: "Failed ... after N attempt(s)" was logged at ERROR for the
    # batch even when the per-item fallback then indexed every item.
    import logging

    def fake_run(client, build, *, results=False):
        if len(build(None)) > 1:
            raise _CME
        return []

    mocker.patch(_MOD + ".run_traversal", side_effect=fake_run)
    indexer.config.retry.graph_store_max_attempts = 1
    stats = IndexingStats(total_items=2)
    with caplog.at_level(logging.WARNING):
        indexer._execute_single_batch(
            ["a", "b"], lambda _g, items: items, "upsert entities", stats
        )
    assert stats.successful_items == 2
    assert [r.levelno for r in caplog.records if "attempt(s)" in r.getMessage()] == [
        logging.WARNING
    ]
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_an_item_the_fallback_cannot_index_logs_an_error(
    indexer, mocker, caplog
) -> None:
    import logging

    mocker.patch(_MOD + ".run_traversal", side_effect=_CME)
    indexer.config.retry.graph_store_max_attempts = 1
    stats = IndexingStats(total_items=2)
    with caplog.at_level(logging.WARNING):
        indexer._execute_single_batch(
            ["a", "b"], lambda _g, items: items, "upsert entities", stats
        )
    assert stats.failed_items == 2
    levels = [r.levelno for r in caplog.records if "attempt(s)" in r.getMessage()]
    assert levels == [logging.WARNING, logging.ERROR, logging.ERROR]


def test_jitter_is_not_the_seedable_module_prng(mocker) -> None:
    # SAST flags random.uniform (CWE-338); the jitter draws from the OS.
    from unified_kg_rag.adapters.storage import neptune_indexer

    mocker.stopall()  # undo the autouse _jitter stub
    assert isinstance(neptune_indexer._jitter.__self__, random.SystemRandom)
