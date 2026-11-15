# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One retry policy, one backoff, one profile per caller (``retry.*``).

Every retrying caller reads its profile of the ``retry`` section and draws
its waits with ``adapters.llm.retry.backoff_delay``. The profiles' defaults
are each caller's values before the section had profiles: these tests pin
them, and the waits each caller makes under them.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from unified_kg_rag.adapters.aws import opensearch as opensearch_mod
from unified_kg_rag.adapters.doc_status import dynamodb as dynamodb_mod
from unified_kg_rag.adapters.llm import retry as llm_retry
from unified_kg_rag.adapters.llm.langchain import BatchProcessor
from unified_kg_rag.adapters.llm.retry import backoff_delay
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import RETRY_PROFILES, RetryPolicy

pytestmark = pytest.mark.unit

# (max_attempts, base, max delay, total budget, jitter) each caller had.
_FORMER = {
    "model": (5, 2.0, 16.0, 60.0, "equal"),
    "ingestion_item": (5, 30.0, 120.0, None, "full"),  # tenacity 30 / 120
    "model_server": (3, 2.0, 16.0, 60.0, "equal"),
    "graph_store": (4, 2.0, None, None, "full"),
    "query_store": (3, 0.5, 4.0, None, "equal"),
    "bulk_item": (5, 1.0, 30.0, None, "equal"),  # 4 resends
    "registry_batch": (8, 0.1, 2.0, None, "none"),  # 0.05 x 2^attempt
}


def _values(policy: RetryPolicy) -> tuple[Any, ...]:
    return (
        policy.max_attempts,
        policy.base_delay_seconds,
        policy.max_delay_seconds,
        policy.max_total_seconds,
        policy.jitter,
    )


@pytest.mark.parametrize("name", sorted(_FORMER))
def test_each_profile_defaults_to_its_callers_former_values(name: str) -> None:
    assert _values(getattr(Config().retry, name)) == _FORMER[name]


def test_every_profile_is_pinned() -> None:
    assert set(RETRY_PROFILES) == set(_FORMER)


def test_a_profile_set_in_part_keeps_its_own_defaults() -> None:
    retry = Config(retry={"graph_store": {"max_attempts": 6}}).retry
    assert _values(retry.graph_store) == (6, 2.0, None, None, "full")
    assert retry.model == RetryPolicy()


def test_a_profile_rejects_an_unknown_key() -> None:
    with pytest.raises(ValueError, match="base_delay"):
        Config(retry={"model": {"base_delay": 1.0}})


@pytest.mark.parametrize(
    ("jitter", "low", "high"),
    [("equal", 0.5, 1.0), ("full", 0.0, 1.0), ("none", 1.0, 1.0)],
)
def test_backoff_draws_under_the_doubling_ceiling(
    jitter: str, low: float, high: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    draws: list[tuple[float, float]] = []

    def uniform(a: float, b: float) -> float:
        draws.append((a, b))
        return b

    monkeypatch.setattr(llm_retry._jitter_rng, "uniform", uniform)
    policy = RetryPolicy(base_delay_seconds=1.0, max_delay_seconds=4.0, jitter=jitter)
    for retry, ceiling in [(1, 1.0), (2, 2.0), (3, 4.0), (4, 4.0)]:
        assert backoff_delay(retry, policy) == ceiling
    if jitter == "none":
        assert draws == []
    else:
        assert draws[0] == (0.0, high - low)


def test_no_upper_bound_keeps_doubling() -> None:
    policy = RetryPolicy(base_delay_seconds=2.0, max_delay_seconds=None)
    assert [policy.ceiling(n) for n in (1, 2, 3, 4)] == [2.0, 4.0, 8.0, 16.0]


def test_the_registry_batch_waits_are_the_former_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # min(2.0, 0.05 * 2**attempt) for attempts 1..7, no jitter.
    slept: list[float] = []
    monkeypatch.setattr(dynamodb_mod, "_sleep", slept.append)
    policy = Config().retry.registry_batch
    for attempt in range(policy.max_attempts):
        dynamodb_mod._backoff_before(attempt, policy)
    assert slept == [0.1, 0.2, 0.4, 0.8, 1.6, 2.0, 2.0]


def test_the_ingestion_item_wait_is_drawn_under_its_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # tenacity's wait_random_exponential(30, 120) drew from [0, 30 x 2^(n-1)]
    # capped at 120; the batch processor draws the same through the shared
    # backoff.
    monkeypatch.setattr(llm_retry._jitter_rng, "uniform", lambda a, b: b)
    slept: list[float] = []
    processor = BatchProcessor(retry=Config().retry.ingestion_item)
    calls = {"n": 0}
    # The waits go to the batch's abandon event (its wait is the sleep).
    never_abandoned = MagicMock(is_set=lambda: False, wait=slept.append)

    @processor._create_retry_decorator("op", never_abandoned)
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 5:
            raise TimeoutError("hung")
        return "ok"

    assert flaky() == "ok"
    assert slept == [30.0, 60.0, 120.0, 120.0]


def test_an_unparseable_answer_is_asked_again_without_the_backoff() -> None:
    # The backoff waits out a throttled quota or a stalled model; an output
    # that does not parse is the model's answer, which a new call may fix at
    # once: it waited 30 to 120 s like a throttle.
    from langchain_core.exceptions import OutputParserException

    slept: list[float] = []
    processor = BatchProcessor(retry=Config().retry.ingestion_item)
    calls = {"n": 0}
    never_abandoned = MagicMock(is_set=lambda: False, wait=slept.append)

    @processor._create_retry_decorator("op", never_abandoned)
    def unparseable_twice() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise OutputParserException("no JSON object")
        return "ok"

    assert unparseable_twice() == "ok"
    assert slept == [0.0, 0.0]


async def test_an_async_unparseable_answer_is_asked_again_without_the_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from langchain_core.exceptions import OutputParserException

    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def recorded(delay: float, *args: Any, **kwargs: Any) -> None:
        slept.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", recorded)
    processor = BatchProcessor(retry=Config().retry.ingestion_item)
    calls = {"n": 0}

    @processor._create_retry_decorator("op")
    async def unparseable_once() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise OutputParserException("no JSON object")
        return "ok"

    assert await unparseable_once() == "ok"
    assert slept == [0.0]


def test_the_ingestion_item_budget_stops_the_retries() -> None:
    processor = BatchProcessor(
        retry=RetryPolicy(
            max_attempts=5, base_delay_seconds=10.0, max_total_seconds=1.0
        )
    )
    calls = {"n": 0}

    @processor._create_retry_decorator("op")
    def failing() -> None:
        calls["n"] += 1
        raise TimeoutError("hung")

    with pytest.raises(TimeoutError):
        failing()
    assert calls["n"] == 1  # the first wait would cross the 1 s budget


def test_bulk_items_are_resent_under_the_bulk_item_profile(mocker) -> None:
    config = Config(retry={"bulk_item": {"max_attempts": 2}})
    client = opensearch_mod.OpenSearchClient.__new__(opensearch_mod.OpenSearchClient)
    client._client = MagicMock()
    client.config = config
    sent: list[int] = []

    def rejected(client: Any, actions: list[dict[str, Any]], **_: Any) -> Any:
        sent.append(len(actions))
        for action in actions:
            yield False, {"index": {"_id": action["_id"], "status": 429}}

    mocker.patch.object(opensearch_mod, "streaming_bulk", side_effect=rejected)
    sleep = mocker.patch.object(opensearch_mod, "_sleep")
    client._bulk_with_item_retries([{"_id": "a"}], lambda ok, _: ok)
    assert sent == [1, 1]  # max_attempts rounds in all
    assert sleep.call_count == 1
