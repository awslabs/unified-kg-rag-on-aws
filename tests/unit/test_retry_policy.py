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
from unified_kg_rag.domain.models.config import (
    RETRY_PROFILES,
    ClientRetryPolicy,
    RetryPolicy,
)

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
    assert retry.model == ClientRetryPolicy(**RETRY_PROFILES["model"])


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
    resends = dynamodb_mod._Resends(Config().retry.registry_batch)
    while resends.wait():
        pass
    assert slept == [0.1, 0.2, 0.4, 0.8, 1.6, 2.0, 2.0]
    assert resends.requests == 8


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


# --- the service clients' own attempts, stacked under the profiles ------------


def test_the_client_attempts_default_to_the_clients_former_counts() -> None:
    # botocore's max_attempts 5 (6 requests) on every non-query Bedrock
    # client, opensearch-py's max_retries 5 on the indexing client.
    retry = Config().retry
    assert retry.model.client_max_attempts == 6
    assert retry.bulk_item.client_max_attempts == 6
    # The other profiles have no such key: their callers' clients make one
    # request per attempt (the configuration reference lists none).
    for name in RETRY_PROFILES.keys() - {"model", "bulk_item"}:
        assert "client_max_attempts" not in type(getattr(retry, name)).model_fields


@pytest.mark.parametrize(
    "factory_name",
    [
        "BedrockLanguageModelFactory",
        "BedrockEmbeddingModelFactory",
        "BedrockRerankModelFactory",
    ],
)
def test_a_bedrock_client_makes_the_model_profiles_requests(factory_name: str) -> None:
    from unified_kg_rag.adapters.aws import bedrock as bedrock_mod

    factory_cls = getattr(bedrock_mod, factory_name)
    factory = factory_cls.__new__(factory_cls)
    factory.config = Config(retry={"model": {"client_max_attempts": 3}})
    retries = factory._boto_config(read_timeout=300).retries
    assert retries["total_max_attempts"] == 3
    assert "max_attempts" not in retries
    # A query client keeps search.model_max_attempts.
    query = factory._boto_config(read_timeout=30, total_max_attempts=2).retries
    assert query["total_max_attempts"] == 2


def test_the_opensearch_indexing_client_makes_the_bulk_item_profiles_requests() -> None:
    client = opensearch_mod.OpenSearchClient.__new__(opensearch_mod.OpenSearchClient)
    client.config = Config(
        aws={"opensearch": {"endpoint": "search.example"}},
        retry={"bulk_item": {"client_max_attempts": 2}},
    )
    client.opensearch_config = client.config.aws.opensearch
    client._get_auth = lambda async_mode: None  # type: ignore[method-assign]
    assert client._get_base_connection_params(async_mode=False)["max_retries"] == 1
    # The query client's attempts are retry.query_store's own.
    assert client._get_base_connection_params(async_mode=True)["max_retries"] == 0


@pytest.mark.parametrize("profile", ["ingestion_item", "query_store", "graph_store"])
def test_a_profile_whose_client_retries_nothing_refuses_client_attempts(
    profile: str,
) -> None:
    with pytest.raises(ValueError, match=f"retry.{profile}.client_max_attempts"):
        Config(retry={profile: {"client_max_attempts": 3}})
    # Assigned too, and from a policy that has the key.
    config = Config()
    with pytest.raises(ValueError, match=f"retry.{profile}.client_max_attempts"):
        setattr(config.retry, profile, config.retry.model)


# --- a refused call is never retried -------------------------------------------


def _refusals() -> list[Exception]:
    from unified_kg_rag.shared.exceptions import (
        OptionalCallSkipped,
        UsageLimitExceededError,
    )

    return [
        UsageLimitExceededError("past the limit", limit="llm_calls", value=1),
        OptionalCallSkipped("kept for the answer", limit="llm_calls", value=1),
    ]


@pytest.mark.parametrize("refusal", _refusals(), ids=type)
@pytest.mark.parametrize("predicate", [None, "everything"])
def test_a_batch_item_refused_by_the_usage_limits_is_not_called_again(
    refusal: Exception, predicate: str | None
) -> None:
    # A refusal is the query's or the run's limit, not a fault: a new call is
    # refused again (after the backoff), whatever the factory calls transient.
    processor = BatchProcessor(
        retry=Config().retry.ingestion_item,
        is_transient_error=None if predicate is None else (lambda _exc: True),
    )
    calls = {"n": 0}

    @processor._create_retry_decorator("op")
    def refused() -> None:
        calls["n"] += 1
        raise refusal

    with pytest.raises(type(refusal)):
        refused()
    assert calls["n"] == 1


@pytest.mark.parametrize("refusal", _refusals(), ids=type)
async def test_a_refused_call_takes_no_transient_retry(refusal: Exception) -> None:
    calls = {"n": 0}

    async def refused() -> None:
        calls["n"] += 1
        raise refusal

    with pytest.raises(type(refusal)):
        await llm_retry.acall_with_transient_retry(
            refused,
            is_transient=lambda _exc: True,
            operation="op",
            policy=Config().retry.model,
        )
    assert calls["n"] == 1


# --- every profile's wall-clock budget is honoured -----------------------------


def test_the_graph_store_budget_stops_the_indexer_retries(mocker) -> None:
    from unified_kg_rag.adapters.storage import neptune_indexer as indexer_mod

    mocker.patch.object(indexer_mod, "NeptuneClient")
    indexer = indexer_mod.NeptuneIndexer(
        config=Config(
            retry={
                "graph_store": {
                    "base_delay_seconds": 2.0,
                    "jitter": "none",
                    "max_total_seconds": 1.0,
                }
            }
        )
    )
    sleep = mocker.patch.object(indexer_mod, "_sleep")
    calls = {"n": 0}

    def failing(*_args: Any, **_kwargs: Any) -> None:
        calls["n"] += 1
        raise ConnectionError("lost")

    mocker.patch.object(indexer_mod, "run_traversal", side_effect=failing)
    with pytest.raises(ConnectionError):
        indexer._execute_with_retries(lambda _g: None, "upsert", fenced=False)
    assert calls["n"] == 1  # the first wait, 2 s, would cross the 1 s budget
    assert sleep.call_count == 0


def test_the_registry_batch_budget_stops_the_resends(mocker) -> None:
    from unified_kg_rag.shared.exceptions import DocStatusRegistryError

    store = dynamodb_mod.DynamoDBDocStatusStore(
        Config(retry={"registry_batch": {"max_total_seconds": 0.05}}),
        boto_session=mocker.MagicMock(),
    )
    client = mocker.MagicMock()
    store._client = client
    client.batch_get_item.side_effect = lambda RequestItems: {
        "Responses": {},
        "UnprocessedKeys": RequestItems,
    }
    sleep = mocker.patch.object(dynamodb_mod, "_sleep")
    with pytest.raises(DocStatusRegistryError, match="after 1 BatchGetItem"):
        store._batch_get(["ns#a"])
    assert client.batch_get_item.call_count == 1  # 0.1 s would cross 0.05 s
    assert sleep.call_count == 0


def test_the_bulk_item_budget_stops_the_resends(mocker) -> None:
    config = Config(retry={"bulk_item": {"max_total_seconds": 0.5}})
    client = opensearch_mod.OpenSearchClient.__new__(opensearch_mod.OpenSearchClient)
    client._client = MagicMock()
    client.config = config
    sent: list[int] = []

    def rejected(client: Any, actions: list[dict[str, Any]], **_: Any) -> Any:
        sent.append(len(actions))
        for action in actions:
            yield False, {"index": {"_id": action["_id"], "status": 429}}

    mocker.patch.object(opensearch_mod, "streaming_bulk", side_effect=rejected)
    mocker.patch.object(llm_retry._jitter_rng, "uniform", lambda a, b: b)
    sleep = mocker.patch.object(opensearch_mod, "_sleep")
    client._bulk_with_item_retries([{"_id": "a"}], lambda ok, _: ok)
    assert sent == [1]  # the first wait, 1 s, would cross the 0.5 s budget
    assert sleep.call_count == 0


# --- a profile given as a RetryPolicy keeps the profile's own defaults --------


def test_a_policy_object_keeps_its_profiles_defaults() -> None:
    # It replaced the ingestion item's 30-120 s full-jitter backoff with
    # RetryPolicy's 2-16 s equal-jitter one and a 60 s budget.
    retry = Config(retry={"ingestion_item": RetryPolicy(max_attempts=3)}).retry
    assert _values(retry.ingestion_item) == (3, 30.0, 120.0, None, "full")


def test_an_assigned_policy_object_keeps_its_profiles_defaults() -> None:
    config = Config()
    config.retry.ingestion_item = RetryPolicy(max_attempts=3)
    assert _values(config.retry.ingestion_item) == (3, 30.0, 120.0, None, "full")
    config.retry.graph_store = {"max_attempts": 6}  # type: ignore[assignment]
    assert _values(config.retry.graph_store) == (6, 2.0, None, None, "full")


def test_a_policy_object_beside_a_deprecated_key_is_its_replacement() -> None:
    with pytest.raises(ValueError, match="retry.max_attempts and retry.model"):
        Config(retry={"max_attempts": 7, "model": RetryPolicy(max_attempts=3)})
    retry = Config(
        retry={"max_attempts": 7, "model": RetryPolicy(base_delay_seconds=1.0)}
    ).retry
    assert (retry.model.max_attempts, retry.model.base_delay_seconds) == (7, 1.0)
