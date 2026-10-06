# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for NeptuneIndexer._execute_with_retries.

This retry-with-exponential-backoff loop is the documented mitigation for the
Neptune ``ConcurrentModificationException`` seen under concurrent index writes.
It was previously untested. These tests pin the behaviors that matter: it
returns on a later-attempt success, it re-raises after exhausting retries, and
it sleeps once per failed attempt (backoff is bounded by max_retries). The module's ``_sleep``/``_jitter``
seams are patched (never the process-global ``time.sleep``/``random.uniform``,
which other threads share) so the tests are fast and deterministic.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.unit

_MOD = "unified_kg_rag.adapters.storage.neptune_indexer"


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
    indexer.neptune_config.max_retries = 3
    indexer.neptune_config.retry_delay_seconds = 1
    sleep = mocker.patch(_MOD + "._sleep", return_value=None)
    # Fails twice (ConcurrentModificationException), succeeds on the 3rd attempt.
    traversal = _FlakyTraversal(
        fail_times=2, exc=Exception("ConcurrentModificationException")
    )
    indexer._execute_with_retries(traversal, "upsert entities")
    assert traversal.calls == 3
    assert sleep.call_count == 2  # one sleep per failed attempt


def test_reraises_after_exhausting_retries(indexer) -> None:
    indexer.neptune_config.max_retries = 2
    indexer.neptune_config.retry_delay_seconds = 1
    # Always fails -> attempts = max_retries + 1, then re-raise.
    traversal = _FlakyTraversal(fail_times=99, exc=RuntimeError("permanent failure"))
    with pytest.raises(RuntimeError, match="permanent failure"):
        indexer._execute_with_retries(traversal, "upsert entities")
    assert traversal.calls == 3  # 1 initial + 2 retries


def test_succeeds_on_first_attempt_does_not_sleep(indexer, mocker) -> None:
    indexer.neptune_config.max_retries = 3
    indexer.neptune_config.retry_delay_seconds = 1
    sleep = mocker.patch(_MOD + "._sleep", return_value=None)
    traversal = _FlakyTraversal(fail_times=0, exc=Exception("never raised"))
    indexer._execute_with_retries(traversal, "upsert entities")
    assert traversal.calls == 1
    sleep.assert_not_called()


def test_zero_retries_raises_immediately(indexer) -> None:
    # max_retries = 0 -> a single attempt, no retry.
    indexer.neptune_config.max_retries = 0
    traversal = _FlakyTraversal(fail_times=99, exc=ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        indexer._execute_with_retries(traversal, "upsert entities")
    assert traversal.calls == 1


def test_backoff_is_full_jitter_over_exponential_window(indexer, mocker) -> None:
    indexer.neptune_config.max_retries = 3
    indexer.neptune_config.retry_delay_seconds = 2
    jitter = mocker.patch(_MOD + "._jitter", side_effect=lambda lo, hi: hi / 2)
    sleep = mocker.patch(_MOD + "._sleep", return_value=None)
    traversal = _FlakyTraversal(fail_times=3, exc=Exception("throttled"))
    indexer._execute_with_retries(traversal, "upsert entities")
    # Window doubles per attempt: [0, 2], [0, 4], [0, 8]; sleep uses the draw.
    assert [c.args for c in jitter.call_args_list] == [(0, 2), (0, 4), (0, 8)]
    assert [c.args[0] for c in sleep.call_args_list] == [1, 2, 4]
