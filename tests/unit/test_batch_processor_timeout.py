# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wall-clock timeout behavior for BatchProcessor.

A hung Bedrock Converse call (open socket, no completion) is not caught by
botocore's byte-gap read_timeout and would block the single Fargate worker for
a whole stage (observed in claim_extraction). BatchProcessor runs every item
as its own call under a wall-clock timeout, so a hung item aborts and is
retried alone while the other items keep their results.

A "hung" call here blocks on an Event that the ``release_hung_calls`` fixture
sets at teardown. A timed-out call is abandoned, not cancelled, and executor
threads are joined at interpreter exit, so a plain ``time.sleep`` would keep
the test process alive for the whole sleep after the suite finished.
"""

import asyncio
import logging
import random
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator

import pytest
from langchain_core.runnables import RunnableLambda

from tests.fixtures.held_timeouts import HeldCallTimeouts
from unified_kg_rag.adapters.aws.bedrock import BaseBedrockModelFactory
from unified_kg_rag.adapters.llm import langchain as langchain_mod
from unified_kg_rag.adapters.llm.langchain import BATCH_ITEM_FAILED, BatchProcessor
from unified_kg_rag.domain.models.config import RetryPolicy
from unified_kg_rag.shared.exceptions import ModelCallStillRunningError

pytestmark = pytest.mark.unit


@pytest.fixture
def release_hung_calls() -> Iterator[threading.Event]:
    release = threading.Event()
    yield release
    release.set()


@pytest.fixture
def held_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[HeldCallTimeouts]:
    """Call timeouts that expire only for the calls that hang: the others
    never time out, so the call counts hold on a loaded runner."""
    held = HeldCallTimeouts(monkeypatch)
    yield held
    held.release.set()


def test_run_with_timeout_aborts_hung_call(release_hung_calls) -> None:
    with pytest.raises(TimeoutError, match="call timeout"):
        BatchProcessor._run_with_timeout(release_hung_calls.wait, 1, "hung")


def test_run_with_timeout_reraises_a_timeout_the_call_raised(caplog) -> None:
    # TimeoutError is concurrent.futures.TimeoutError, so a
    # call that raised one (a socket timeout, say) read as the call timeout:
    # a misleading warning and a new error naming the wrong limit.
    def times_out() -> None:
        raise TimeoutError("read timed out")

    with pytest.raises(TimeoutError, match="^read timed out$"):
        BatchProcessor._run_with_timeout(times_out, 5, "own")
    assert "call timeout" not in caplog.text


def test_run_with_timeout_returns_fast_result() -> None:
    assert BatchProcessor._run_with_timeout(lambda: 42, 5, "fast") == 42


def test_run_with_timeout_zero_disables() -> None:
    # 0 means "no timeout" — run directly.
    assert BatchProcessor._run_with_timeout(lambda: "ok", 0, "nolimit") == "ok"


class _SlowFirstCall:
    """Per-item call whose first call for item ``slow`` hangs (``hang``)."""

    def __init__(self, slow: int, hang: Callable[[], object]) -> None:
        self.slow = slow
        self.hang = hang
        self.calls: Counter[int] = Counter()
        self._lock = threading.Lock()

    def __call__(self, item: dict[str, int]) -> dict[str, int]:
        with self._lock:
            self.calls[item["v"]] += 1
            first = self.calls[item["v"]] == 1
        if item["v"] == self.slow and first:
            self.hang()
        return {"echo": item["v"]}


def test_one_slow_item_times_out_alone(held_calls: HeldCallTimeouts) -> None:
    # Each item has its own timeout: the finished items keep their results
    # and only the slow one is called again.
    fake = _SlowFirstCall(slow=3, hang=held_calls.hang)
    bp = BatchProcessor(
        call_timeout_seconds=60,
        retry=RetryPolicy(base_delay_seconds=1.0, max_delay_seconds=0),
    )
    results = bp.execute_with_fallback(
        items_to_process=list(range(10)),
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=fake,
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": i} for i in range(10)]
    assert fake.calls[3] == 2
    assert sum(fake.calls.values()) == 11


def test_item_slow_on_every_attempt_fails_in_place(
    held_calls: HeldCallTimeouts,
) -> None:
    # Each of the max_attempts calls times out; only that position is
    # marked failed.
    calls: Counter[int] = Counter()
    lock = threading.Lock()

    def call(item: dict[str, int]) -> dict[str, int]:
        with lock:
            calls[item["v"]] += 1
        if item["v"] == 1:
            held_calls.hang()
        return {"echo": item["v"]}

    bp = BatchProcessor(
        call_timeout_seconds=60,
        retry=RetryPolicy(max_attempts=2, base_delay_seconds=1.0, max_delay_seconds=0),
    )
    results = bp.execute_with_fallback(
        items_to_process=[0, 1, 2],
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=call,
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": 0}, BATCH_ITEM_FAILED, {"echo": 2}]
    assert calls == Counter({0: 1, 1: 2, 2: 1})


def test_items_run_concurrently() -> None:
    # The items are in flight together, up to max_concurrency.
    barrier = threading.Barrier(3, timeout=10)

    def call(item: dict[str, int]) -> dict[str, int]:
        barrier.wait()
        return {"echo": item["v"]}

    bp = BatchProcessor(max_concurrency=3, call_timeout_seconds=0)
    results = bp.execute_with_fallback(
        items_to_process=[1, 2, 3],
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=call,
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": 1}, {"echo": 2}, {"echo": 3}]


def test_results_preserve_order_when_items_finish_out_of_order() -> None:
    # Each item waits for the next one to finish, so they complete strictly in
    # reverse submission order (the last item has nothing to wait for).
    bp = BatchProcessor(max_concurrency=4, call_timeout_seconds=0)
    n = 4
    done = [threading.Event() for _ in range(n)]

    def call(item: dict[str, int]) -> dict[str, int]:
        v = item["v"]
        if v + 1 < n:
            assert done[v + 1].wait(timeout=10), "items did not run concurrently"
        done[v].set()
        return {"echo": v}

    results = bp.execute_with_fallback(
        items_to_process=list(range(n)),
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=call,
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": i} for i in range(n)]


class _InFlight:
    """Counts the calls in flight and the peak."""

    def __init__(self) -> None:
        self.now = 0
        self.peak = 0
        self._lock = threading.Lock()

    def __enter__(self) -> None:
        with self._lock:
            self.now += 1
            self.peak = max(self.peak, self.now)

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            self.now -= 1


def test_peak_in_flight_is_max_concurrency_under_random_latency() -> None:
    # One pool of max_concurrency slots: never more calls in flight (chunks
    # run in parallel used to reach max_concurrency x chunk_concurrency), and
    # a slot is refilled as soon as it frees up, so the peak is reached.
    rng = random.Random(0)
    latencies = [rng.uniform(0, 0.02) for _ in range(60)]
    in_flight = _InFlight()

    def call(item: dict[str, int]) -> int:
        with in_flight:
            time.sleep(latencies[item["v"]])
        return item["v"]

    bp = BatchProcessor(max_concurrency=4, call_timeout_seconds=0)
    results = bp.execute_with_fallback(
        items_to_process=list(range(60)),
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=call,
        task_name="t",
        show_progress=False,
    )
    assert results == list(range(60))
    assert in_flight.peak == 4


def test_slow_item_does_not_hold_back_the_other_slots() -> None:
    # Item 0 finishes only after every other item did: with one slot held by
    # it, the other slot must run all of them, and no extra slot opens to make
    # up for the held one (chunks used to run in parallel, each waiting for
    # its slowest item).
    others_done = threading.Event()
    finished: list[int] = []
    lock = threading.Lock()
    in_flight = _InFlight()
    n = 30

    def call(item: dict[str, int]) -> int:
        with in_flight:
            if item["v"] == 0:
                assert others_done.wait(timeout=10), "slow item blocked the others"
            else:
                time.sleep(0.002)
                with lock:
                    finished.append(item["v"])
                    if len(finished) == n - 1:
                        others_done.set()
        return item["v"]

    bp = BatchProcessor(max_concurrency=2, call_timeout_seconds=0)
    results = bp.execute_with_fallback(
        items_to_process=list(range(n)),
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=call,
        task_name="t",
        show_progress=False,
    )
    assert results == list(range(n))
    assert in_flight.peak == 2


async def test_async_slow_item_does_not_hold_back_the_other_slots() -> None:
    others_done = asyncio.Event()
    finished: list[int] = []
    in_flight = _InFlight()
    n = 30

    async def call(item: dict[str, int]) -> int:
        with in_flight:
            if item["v"] == 0:
                await asyncio.wait_for(others_done.wait(), timeout=10)
            else:
                await asyncio.sleep(0.001)
                finished.append(item["v"])
                if len(finished) == n - 1:
                    others_done.set()
        return item["v"]

    runnable = RunnableLambda(call)
    bp = BatchProcessor(max_concurrency=2, retry=RetryPolicy(max_attempts=1))
    results = await bp.aexecute_with_fallback(
        items_to_process=list(range(n)),
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=runnable.ainvoke,
        task_name="t",
        show_progress=False,
    )
    assert results == list(range(n))
    assert in_flight.peak == 2


def test_the_call_timeout_is_the_factory_s_request_timeout() -> None:
    # It was a fixed 300 s that no caller set: a local server given 1,200 s
    # (providers.llm.timeout_seconds) had its long extraction calls
    # abandoned at 300 s and sent again while it kept generating the first
    # (ARCH baseline).
    from tests.fixtures.fakes.boto import FakeBotoSession
    from unified_kg_rag.adapters.aws.bedrock import BedrockLanguageModelFactory
    from unified_kg_rag.adapters.llm.retry import (
        DEFAULT_CALL_TIMEOUT_SECONDS,
        call_timeout_for,
        transient_error_predicate,
    )
    from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory
    from unified_kg_rag.domain.models import Config
    from unified_kg_rag.domain.models.config import ModelPurpose

    local = Config.model_validate(
        {"backend": "local", "providers": {"llm": {"timeout_seconds": 1200}}}
    )
    ollama = OllamaLanguageModelFactory(local)
    processor = BatchProcessor.from_config(local, ollama)
    assert processor.call_timeout_seconds == 1200
    assert processor.is_transient_error is transient_error_predicate(ollama)
    # The query calls' own (capped) timeout is not the ingestion one.
    assert call_timeout_for(ollama, ModelPurpose.QUERY) == (
        local.search.model_timeout_seconds
    )
    bedrock = BedrockLanguageModelFactory(Config(), boto_session=FakeBotoSession())
    assert BatchProcessor.from_config(Config(), bedrock).call_timeout_seconds == (
        BaseBedrockModelFactory.BOTO_READ_TIMEOUT
    )
    # A factory that names none keeps the default; an explicit one wins.
    assert BatchProcessor.from_config(Config(), object()).call_timeout_seconds == (
        DEFAULT_CALL_TIMEOUT_SECONDS
    )
    assert (
        BatchProcessor.from_config(
            local, ollama, call_timeout_seconds=5
        ).call_timeout_seconds
        == 5
    )


def test_an_abandoned_call_is_not_resent_while_it_runs(
    held_calls: HeldCallTimeouts,
) -> None:
    # Its retry started at once beside it: the server generated both.
    events: list[str] = []
    lock = threading.Lock()

    def call(item: dict[str, int]) -> dict[str, int]:
        with lock:
            events.append("start")
            first = events.count("start") == 1
        if first:
            held_calls.hang()
            with lock:
                events.append("abandoned call ended")
        return {"echo": item["v"]}

    bp = BatchProcessor(
        call_timeout_seconds=60,
        retry=RetryPolicy(base_delay_seconds=1.0, max_delay_seconds=0),
    )
    results = bp.execute_with_fallback(
        items_to_process=[1],
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=call,
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": 1}]
    assert events == ["start", "abandoned call ended", "start"]


def test_an_abandoned_call_sends_no_further_request() -> None:
    # The model clients' stop points (Bedrock's before-send and needs-retry
    # hooks, the local clients' request hooks) refuse every request of a
    # call once it is abandoned: a client's own resend after its read
    # timeout would otherwise run on beside the item's retry.
    from unified_kg_rag.adapters.local import models as local_models
    from unified_kg_rag.shared.query_stop import query_stopped

    abandoned = threading.Event()
    seen: list[object] = []

    def call() -> None:
        abandoned.wait(30)
        seen.append(query_stopped())
        try:
            local_models._stop_before_sync_request(None)
        except asyncio.CancelledError:
            seen.append("refused")

    calls = langchain_mod._ItemCalls()
    with pytest.raises(TimeoutError, match="call timeout"):
        BatchProcessor._run_with_timeout(call, 0.2, "hung", calls)
    abandoned.set()
    assert calls.abandoned is not None
    calls.abandoned.result(timeout=30)
    assert seen == [True, "refused"]


def test_an_abandoned_call_still_running_fails_its_item_unresent(
    release_hung_calls: threading.Event,
) -> None:
    # A call that outlasts one more call timeout (a server dribbling bytes
    # past every read timeout) is not sent again beside it: the item fails.
    calls: Counter[int] = Counter()
    errors: dict[int, BaseException] = {}

    def call(item: dict[str, int]) -> dict[str, int]:
        calls[item["v"]] += 1
        release_hung_calls.wait()
        return {"echo": item["v"]}

    bp = BatchProcessor(
        call_timeout_seconds=0.2,
        retry=RetryPolicy(max_attempts=3, base_delay_seconds=1.0, max_delay_seconds=0),
    )
    results = bp.execute_with_fallback(
        items_to_process=[7],
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=call,
        task_name="t",
        show_progress=False,
        errors=errors,
    )
    assert results == [BATCH_ITEM_FAILED]
    assert calls == Counter({7: 1})
    assert isinstance(errors[0], ModelCallStillRunningError)


def test_cancelling_an_async_batch_cancels_its_items() -> None:
    # Before, the items kept running (and a cancelled one was retried) after
    # the caller was cancelled: a timed-out evaluation went on answering the
    # whole dataset in the background.
    calls = Counter()

    async def main() -> None:
        both_started = asyncio.Event()
        never = asyncio.Event()

        async def work(item: dict) -> int:
            calls["started"] += 1
            if calls["started"] == 2:
                both_started.set()
            await never.wait()  # only a cancellation ends it
            calls["finished"] += 1
            return item["i"]

        processor = BatchProcessor(max_concurrency=2)
        task = asyncio.create_task(
            processor.aexecute_with_fallback(
                list(range(10)),
                lambda batch: [{"i": i} for i in batch],
                work,
                "T",
                show_progress=False,
            )
        )
        # Cancelled once both slots run an item, not after a guessed delay.
        await asyncio.wait_for(both_started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # A surviving item would still be waiting: none is.
        assert len(asyncio.all_tasks()) == 1  # only main

    asyncio.run(main())
    assert calls == Counter(started=2)


async def test_an_async_item_that_failed_names_its_ids_and_keeps_its_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Regression (rv29 logging #3): run-eval's "Item failed every attempt in
    # 'Answer Generation'" line named no query, and its results only "no
    # result": the error was dropped with the failed item.
    async def call(item: dict[str, str]) -> str:
        if item["id"] == "q2":
            raise ValueError("Quokkasecret reply")
        return item["id"]

    errors: dict[int, BaseException] = {}
    bp = BatchProcessor(max_concurrency=2, retry=RetryPolicy(max_attempts=1))
    with caplog.at_level("INFO"):
        results = await bp.aexecute_with_fallback(
            items_to_process=["q1", "q2"],
            prepare_inputs_func=lambda ids: [{"id": i} for i in ids],
            sequential_func=RunnableLambda(call).ainvoke,
            task_name="Answer Generation",
            show_progress=False,
            errors=errors,
            item_ids=lambda item: {"query_id": item["id"]},
        )
    assert results == ["q1", BATCH_ITEM_FAILED]
    assert list(errors) == [1] and isinstance(errors[1], ValueError)
    (failed,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert failed.query_id == "q2"  # type: ignore[attr-defined]
    assert "Quokkasecret" not in failed.getMessage()


@pytest.mark.parametrize("order", ["abandoned first", "bound first"])
def test_an_abandoned_attempt_stops_its_call_whenever_it_binds(order: str) -> None:
    # The caller may abandon a call before its thread bound its stop.
    from unified_kg_rag.shared.query_stop import QueryStop, StopCause

    attempt, stop = langchain_mod._Attempt(), QueryStop()
    if order == "abandoned first":
        attempt.abandon()
        attempt.bind(stop)
    else:
        attempt.bind(stop)
        assert not stop.is_set()
        attempt.abandon()
    assert stop.is_set() and stop.cause is StopCause.DEADLINE
