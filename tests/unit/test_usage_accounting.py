# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Always-on LLM usage accounting (shared.usage + the usage callback)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tests.fixtures.botocore_backoff import stub_botocore_backoff
from tests.fixtures.fakes.model_servers import FakeOllama, ModelRejects
from unified_kg_rag.adapters.llm.response_cache import (
    CachingLLMFactory,
    LLMResponseCache,
)
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory
from unified_kg_rag.adapters.observability.usage_callback import UsageCallbackHandler
from unified_kg_rag.domain.models import Config, LLMUsage, ModelPurpose
from unified_kg_rag.shared import observability, usage
from unified_kg_rag.shared.exceptions import UsageLimitExceededError

pytestmark = pytest.mark.unit


def _respond(system: str, human: str) -> str:
    if "reject" in human:
        raise ModelRejects()
    return f"reply to {human} with a few words"


def _factory(server: FakeOllama) -> OllamaLanguageModelFactory:
    config = Config(
        providers={
            "llm": {"kind": "ollama", "base_url": "http://model-server.test"},
            "embedding": {"kind": "ollama", "base_url": "http://model-server.test"},
        },
        models={"default": "chat-a", "embedding": {"model_id": "e", "dimension": 4}},
    )
    return OllamaLanguageModelFactory(config, transport=httpx.MockTransport(server))


def test_a_fake_model_run_reports_the_servers_calls_and_tokens() -> None:
    # The acceptance check: N chat calls (sync, and concurrent async ones)
    # with tracing off report exactly N calls and the server's token counts.
    assert not observability.is_enabled()
    server = FakeOllama(_respond, "unused", 4)
    model = _factory(server).get_model("chat-a", model_purpose=ModelPurpose.QUERY)

    async def _many() -> None:
        await asyncio.gather(
            *(
                model.ainvoke([HumanMessage(content=f"q{i} " * (i + 1))])
                for i in range(5)
            )
        )

    with usage.usage_scope() as meter:
        model.invoke([HumanMessage(content="one two three")])
        model.invoke([HumanMessage(content="four")])
        asyncio.run(_many())
    recorded = LLMUsage.model_validate(meter.snapshot())
    assert len(server.chat_models) == 7
    assert recorded.llm_calls == 7
    assert (recorded.input_tokens, recorded.output_tokens) == (
        server.input_tokens,
        server.output_tokens,
    )
    assert server.input_tokens > 7 and server.output_tokens > 7
    assert recorded.by_role["query"].llm_calls == 7
    assert recorded.by_model["chat-a"].input_tokens == server.input_tokens
    assert recorded.cache_hits == 0


def test_concurrent_scopes_count_apart_and_nest() -> None:
    server = FakeOllama(_respond, "unused", 4)
    model = _factory(server).get_model("chat-a", model_purpose=ModelPurpose.QUERY)

    async def _query(n: int) -> dict[str, Any]:
        with usage.usage_scope() as meter:
            for _ in range(n):
                await model.ainvoke([HumanMessage(content="hello")])
            return meter.snapshot()

    async def _run() -> list[dict[str, Any]]:
        return list(await asyncio.gather(_query(1), _query(2), _query(3)))

    with usage.usage_scope() as outer:
        per_query = asyncio.run(_run())
    assert [q["llm_calls"] for q in per_query] == [1, 2, 3]
    # A call counts in every enclosing scope.
    assert outer.snapshot()["llm_calls"] == 6
    assert outer.snapshot() == usage.merge_usage(per_query)


def test_no_scope_no_record_and_a_failed_call_is_not_a_call() -> None:
    server = FakeOllama(_respond, "unused", 4)
    model = _factory(server).get_model("chat-a", model_purpose=ModelPurpose.QUERY)
    model.invoke([HumanMessage(content="outside")])
    assert usage.current_meter() is None
    with usage.usage_scope() as meter:
        with pytest.raises(Exception, match="input rejected"):
            model.invoke([HumanMessage(content="reject this")])
    assert meter.snapshot()["llm_calls"] == 0


def test_a_cached_response_is_a_cache_hit_not_a_call(tmp_path: Path) -> None:
    server = FakeOllama(_respond, "unused", 4)

    def _model(cache: LLMResponseCache) -> Any:
        return CachingLLMFactory(_factory(server), cache).get_model(
            "chat-a", model_purpose=ModelPurpose.INGESTION
        )

    first = LLMResponseCache(flush_every=100)
    first.open(tmp_path)
    with usage.usage_scope() as run_one:
        _model(first).invoke([HumanMessage(content="extract")])
    first.close()
    second = LLMResponseCache(flush_every=100)
    second.open(tmp_path)
    with usage.usage_scope() as run_two:
        _model(second).invoke([HumanMessage(content="extract")])
    second.close()
    assert len(server.chat_models) == 1
    assert run_one.snapshot()["llm_calls"] == 1
    replayed = run_two.snapshot()
    assert (replayed["llm_calls"], replayed["input_tokens"]) == (0, 0)
    assert replayed["cache_hits"] == 1
    assert replayed["by_role"] == {
        "ingestion": {
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_hits": 1,
            "estimated_calls": 0,
            "cache_read_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "request_retries": 0,
            "sent_requests": 0,
            "throttled_requests": 0,
            "failed_calls": 0,
            "rerank_calls": 0,
            "rerank_search_units": 0,
            "embedding_calls": 0,
            "embedding_input_tokens": 0,
            "embedding_estimated_calls": 0,
        }
    }


def _result(**message: Any) -> LLMResult:
    return LLMResult(generations=[[ChatGeneration(message=AIMessage(**message))]])


def test_the_handler_reads_both_usage_shapes_and_early_stream_ends() -> None:
    handler = UsageCallbackHandler("vendor.model-v1", role="evaluation")
    with usage.usage_scope() as meter:
        handler.on_llm_end(
            _result(
                content="x",
                usage_metadata={
                    "input_tokens": 12,
                    "output_tokens": 3,
                    "total_tokens": 15,
                },
            ),
            run_id=uuid4(),
        )
        # ChatBedrock (InvokeModel) reports the totals in llm_output.
        handler.on_llm_end(
            LLMResult(
                generations=[[ChatGeneration(message=AIMessage(content="y"))]],
                llm_output={"usage": {"prompt_tokens": 5, "completion_tokens": 2}},
            ),
            run_id=uuid4(),
        )
        # A stream the consumer stopped reading was a call; a failure is not.
        handler.on_llm_error(
            GeneratorExit(),
            run_id=uuid4(),
            response=_result(
                content="z",
                usage_metadata={
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            ),
        )
        handler.on_llm_error(RuntimeError("throttled"), run_id=uuid4())
    snapshot = meter.snapshot()
    assert (snapshot["llm_calls"], snapshot["input_tokens"]) == (3, 18)
    assert snapshot["output_tokens"] == 6
    assert list(snapshot["by_model"]) == ["vendor.model-v1"]
    assert list(snapshot["by_role"]) == ["evaluation"]


def _converse_result(
    details: dict[str, int], *, input_tokens: int, retries: int = 0
) -> LLMResult:
    # The shape langchain-aws's Converse parser gives: input_tokens includes
    # the cache reads and writes, and the boto response's metadata rides along.
    return _result(
        content="x",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": 7,
            "total_tokens": input_tokens + 7,
            "input_token_details": details,
        },
        response_metadata={"ResponseMetadata": {"RetryAttempts": retries}},
    )


def test_the_handler_counts_cache_tokens_retries_and_failed_calls() -> None:
    # Cache reads were counted as plain input (priced at the input rate) and
    # botocore's retries and failed calls not at all, so the meter fell
    # short of Bedrock's invocation count.
    from botocore.exceptions import ClientError

    handler = UsageCallbackHandler("vendor.model-v1", role="ingestion")
    with usage.usage_scope() as meter:
        handler.on_llm_end(
            _converse_result(
                {"cache_read": 800, "cache_creation": 50}, input_tokens=1000, retries=2
            ),
            run_id=uuid4(),
        )
        # With a per-TTL cacheDetails breakdown the parser zeroes
        # cache_creation and reports the writes per TTL.
        handler.on_llm_end(
            _converse_result(
                {
                    "cache_read": 0,
                    "cache_creation": 0,
                    "ephemeral_5m_input_tokens": 30,
                    "ephemeral_1h_input_tokens": 20,
                },
                input_tokens=100,
            ),
            run_id=uuid4(),
        )
        failed = uuid4()
        handler.on_chat_model_start({}, [[HumanMessage(content="q")]], run_id=failed)
        error = ClientError(
            {
                "Error": {"Code": "ServiceUnavailableException", "Message": "busy"},
                "ResponseMetadata": {"RetryAttempts": 5},
            },
            "Converse",
        )
        handler.on_llm_error(error, run_id=failed)
    snapshot = meter.snapshot()
    assert (snapshot["llm_calls"], snapshot["input_tokens"]) == (2, 1100)
    assert snapshot["cache_read_input_tokens"] == 800
    assert snapshot["cache_write_input_tokens"] == 100
    assert snapshot["request_retries"] == 7
    assert snapshot["failed_calls"] == 1
    assert snapshot["estimated_calls"] == 0
    model = LLMUsage.model_validate(snapshot).by_model["vendor.model-v1"]
    assert (model.cache_read_input_tokens, model.failed_calls) == (800, 1)


def test_bedrock_models_get_the_usage_handler_with_tracing_off(mocker) -> None:
    from tests.fixtures.fakes.boto import FakeBotoSession
    from unified_kg_rag.adapters.aws import bedrock as bedrock_mod

    assert not observability.is_enabled()
    factory = bedrock_mod.BedrockLanguageModelFactory(
        Config(), boto_session=FakeBotoSession()  # type: ignore[arg-type]
    )
    mocker.patch.object(
        bedrock_mod.BedrockCrossRegionModelHelper,
        "get_cross_region_model_id",
        return_value="global.vendor.model-v1",
    )
    captured: dict[str, Any] = {}

    class _FakeChat:
        def __new__(cls, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return "model"

    mocker.patch.object(bedrock_mod, "ChatBedrockConverse", _FakeChat)
    mocker.patch.object(bedrock_mod, "ChatBedrock", _FakeChat)
    factory.get_model(
        Config().model_id(_answer_role()), model_purpose=ModelPurpose.INGESTION
    )
    (handler,) = captured["callbacks"]
    assert isinstance(handler, UsageCallbackHandler)
    assert (handler.model_id, handler.role) == (captured["model_id"], "ingestion")
    # The id a query extraction cache hit is recorded under.
    assert factory.resolved_model_id(Config().model_id(_answer_role())) == (
        handler.model_id
    )


def _answer_role() -> Any:
    from unified_kg_rag.domain.models import ModelRole

    return ModelRole.ANSWER_GENERATION


def test_a_query_extraction_from_the_chains_cache_is_a_cache_hit() -> None:
    from unified_kg_rag.application.retrieval.rag_chain import (
        GraphRAGChain,
        _ExtractionCache,
    )

    chain = GraphRAGChain.__new__(GraphRAGChain)
    chain._extractions = _ExtractionCache(4)
    key = ("EntityExtractionPrompt", "model-x", "templates", "who?", "en")
    with usage.usage_scope() as meter:
        assert chain._cached_extraction(key) is None  # a miss records nothing
        chain._cache_extraction(key, ("Depot",))
        assert chain._cached_extraction(key) == ("Depot",)
    snapshot = meter.snapshot()
    assert (snapshot["llm_calls"], snapshot["cache_hits"]) == (0, 1)
    assert snapshot["by_role"]["query"]["cache_hits"] == 1
    assert snapshot["by_model"]["model-x"]["cache_hits"] == 1


def test_a_call_counts_from_its_start_until_it_settles() -> None:
    handler = UsageCallbackHandler("vendor.model-v1", role="query")
    text = "one two three four five six seven eight"
    prompt = [[HumanMessage(content=text)]]
    estimate = estimate_token_count(text)
    with usage.usage_scope() as meter:
        returned, failed, cancelled = uuid4(), uuid4(), uuid4()
        for run_id in (returned, failed, cancelled):
            handler.on_chat_model_start({}, prompt, run_id=run_id)
        # In flight: counted at an estimate, as cancelled until it settles.
        in_flight = meter.snapshot()
        handler.on_llm_end(
            _result(
                content="x",
                usage_metadata={
                    "input_tokens": 40,
                    "output_tokens": 3,
                    "total_tokens": 43,
                },
            ),
            run_id=returned,
        )
        handler.on_llm_error(RuntimeError("throttled"), run_id=failed)
        handler.on_llm_error(asyncio.CancelledError(), run_id=cancelled)
    assert (in_flight["llm_calls"], in_flight["estimated_calls"]) == (3, 3)
    assert in_flight["input_tokens"] == 3 * estimate
    settled = LLMUsage.model_validate(meter.snapshot())
    # The returned call by its response, the failed one gone, the cancelled
    # one at its estimate.
    assert (settled.llm_calls, settled.estimated_calls) == (2, 1)
    assert (settled.input_tokens, settled.output_tokens) == (40 + estimate, 3)
    assert settled.by_model["vendor.model-v1"].estimated_calls == 1


def test_a_cancelled_worker_thread_call_is_counted() -> None:
    # How ChatBedrockConverse runs ainvoke: the request runs on a worker
    # thread, which a cancelled caller does not stop; LangChain reports no
    # end or error for it.
    import threading
    import time

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatResult

    finished = threading.Event()

    class _ThreadedChat(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "threaded"

        def _generate(self, messages: Any, *a: Any, **k: Any) -> ChatResult:
            time.sleep(0.2)
            finished.set()
            message = AIMessage(
                content="a",
                usage_metadata={
                    "input_tokens": 500,
                    "output_tokens": 40,
                    "total_tokens": 540,
                },
            )
            return ChatResult(generations=[ChatGeneration(message=message)])

    handler = UsageCallbackHandler("m", role="query")
    model = _ThreadedChat(callbacks=[handler])

    async def _run() -> None:
        task = asyncio.ensure_future(model.ainvoke("what does the Vendor ship"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with usage.usage_scope() as meter:
        asyncio.run(_run())
    assert finished.wait(5)
    snapshot = meter.snapshot()
    assert (snapshot["llm_calls"], snapshot["estimated_calls"]) == (1, 1)
    assert 0 < snapshot["input_tokens"] < 500 and snapshot["output_tokens"] == 0
    # Never to settle, the call was let go when its task ended.
    assert not handler._calls and not handler._tasks


async def test_a_settled_call_leaves_its_task_and_settles_once() -> None:
    handler = UsageCallbackHandler("m", role="query")
    task = asyncio.current_task()
    assert task is not None
    run_id = uuid4()
    with usage.usage_scope() as meter:
        handler.on_chat_model_start({}, [[HumanMessage(content="hi")]], run_id=run_id)
        call = handler._calls[run_id].call
        handler.on_llm_end(
            _result(
                content="x",
                usage_metadata={
                    "input_tokens": 3,
                    "output_tokens": 1,
                    "total_tokens": 4,
                },
            ),
            run_id=run_id,
        )
        # A second settle (a late end of the same call) changes nothing.
        call.finish(input_tokens=900, output_tokens=90)
        call.fail()
    assert not handler._calls and not handler._tasks
    assert handler._ended not in [getattr(cb, "func", None) for cb in _callbacks(task)]
    snapshot = meter.snapshot()
    assert (snapshot["llm_calls"], snapshot["failed_calls"]) == (1, 0)
    assert snapshot["input_tokens"] < 900 and snapshot["output_tokens"] < 90


def _callbacks(task: asyncio.Future[Any]) -> list[Any]:
    """The done callbacks of ``task`` (CPython keeps them on the task)."""
    return [callback for callback, _ in getattr(task, "_callbacks", None) or []]


def test_an_extraction_cache_hit_is_recorded_under_the_resolved_model_id() -> None:
    from types import SimpleNamespace

    from unified_kg_rag.application.retrieval.rag_chain import (
        GraphRAGChain,
        _ExtractionCache,
    )

    chain = GraphRAGChain.__new__(GraphRAGChain)
    chain._extractions = _ExtractionCache(4)
    # The Bedrock factory calls a model by its inference profile, and its
    # usage callback records the calls under that id.
    chain.factory = SimpleNamespace(  # type: ignore[assignment]
        resolved_model_id=lambda model_id: f"global.{model_id}"
    )
    key = ("EntityExtractionPrompt", "model-x", "templates", "who?", "en")
    chain._cache_extraction(key, ("Depot",))
    with usage.usage_scope() as meter:
        assert chain._cached_extraction(key) == ("Depot",)
    assert list(meter.snapshot()["by_model"]) == ["global.model-x"]


def test_calls_that_never_settle_are_let_go(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(UsageCallbackHandler, "MAX_UNSETTLED", 2)
    handler = UsageCallbackHandler("m", role="query")
    with usage.usage_scope() as meter:
        for _ in range(5):
            handler.on_chat_model_start(
                {}, [[HumanMessage(content="hello")]], run_id=uuid4()
            )
    assert len(handler._calls) == 2
    assert meter.snapshot()["estimated_calls"] == 5


async def test_a_call_whose_model_reports_no_usage_keeps_its_estimate() -> None:
    # A model or server that returns no token usage was settled at 0 tokens
    # and 0 estimated calls, so the call read as free and measured.
    from langchain_core.language_models.fake_chat_models import (
        GenericFakeChatModel,
    )

    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="an answer of several words")]),
        callbacks=[UsageCallbackHandler("local-model", role="query")],
    )
    prompt = "a fairly long prompt " * 200
    with usage.usage_scope() as meter:
        await model.ainvoke(prompt)
    snapshot = meter.snapshot()
    assert snapshot["llm_calls"] == 1
    assert snapshot["estimated_calls"] == 1
    assert snapshot["input_tokens"] == estimate_token_count(prompt) > 0


@pytest.mark.parametrize("started", [True, False])
def test_a_response_without_usage_keeps_its_retries_and_estimates_its_output(
    started: bool,
) -> None:
    # A response that reported no tokens dropped the requests botocore
    # resent inside the call, and its output read as 0 tokens.
    handler = UsageCallbackHandler("m", role="query")
    run_id = uuid4()
    answer = "an answer of several words about the Vendor"
    prompt = "a fairly long prompt " * 20
    with usage.usage_scope() as meter:
        if started:
            handler.on_chat_model_start(
                {}, [[HumanMessage(content=prompt)]], run_id=run_id
            )
        handler.on_llm_end(
            _result(
                content=answer,
                response_metadata={"ResponseMetadata": {"RetryAttempts": 2}},
            ),
            run_id=run_id,
        )
    snapshot = meter.snapshot()
    assert (snapshot["llm_calls"], snapshot["estimated_calls"]) == (1, 1)
    assert snapshot["request_retries"] == 2
    assert snapshot["output_tokens"] == estimate_token_count(answer) > 0
    assert snapshot["input_tokens"] == (estimate_token_count(prompt) if started else 0)


def _calls(handler: UsageCallbackHandler, prompt: str, reported: int, n: int) -> None:
    for _ in range(n):
        run_id = uuid4()
        handler.on_llm_start({}, [prompt], run_id=run_id)
        handler.on_llm_end(
            _result(
                content="x",
                usage_metadata={
                    "input_tokens": reported,
                    "output_tokens": 1,
                    "total_tokens": reported + 1,
                },
            ),
            run_id=run_id,
        )


@pytest.mark.parametrize(("factor", "effect"), [(3, "overflow"), (0.3, "unused")])
def test_a_model_far_from_the_estimate_is_warned_of_once(
    factor: float,
    effect: str,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unified_kg_rag.shared import logging as shared_logging

    monkeypatch.setattr(shared_logging, "_warned_once", set())
    prompt = "आपूर्तिकर्ता ने दो जून को गोदाम में लिडार इकाइयाँ पहुँचाईं। " * 20
    estimate = estimate_token_count(prompt)
    handler = UsageCallbackHandler("vendor.model-v1", role="query")
    with caplog.at_level("WARNING"), usage.usage_scope():
        _calls(handler, prompt, int(estimate * factor), 19)
        assert "local token estimate" not in caplog.text  # too few calls yet
        _calls(handler, prompt, int(estimate * factor), 10)
    assert caplog.text.count("local token estimate") == 1
    assert effect in caplog.text and "vendor.model-v1" in caplog.text


def test_a_model_near_the_estimate_is_not_warned_of(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unified_kg_rag.shared import logging as shared_logging

    monkeypatch.setattr(shared_logging, "_warned_once", set())
    prompt = "The vendor delivered the units to the warehouse. " * 20
    handler = UsageCallbackHandler("vendor.model-v1", role="query")
    with caplog.at_level("WARNING"), usage.usage_scope():
        _calls(handler, prompt, int(estimate_token_count(prompt) * 1.3), 40)
    assert "local token estimate" not in caplog.text


class _Raw:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def stream(self, **_: Any) -> Any:
        yield self._body


def test_throttled_bedrock_requests_count_per_attempt_and_role(monkeypatch) -> None:
    # Old behaviour: botocore resent each throttled request inside the call,
    # so usage showed request_retries with no cause and nothing told this
    # deployment's throttling from a read timeout's.
    import json

    import boto3
    from botocore.awsrequest import AWSResponse

    from unified_kg_rag.adapters.aws.bedrock import BedrockLanguageModelFactory

    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    stub_botocore_backoff(monkeypatch)
    session = boto3.Session(
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        region_name="us-east-1",
    )
    factory = BedrockLanguageModelFactory(Config(), boto_session=session)
    answer = {
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        "metrics": {"latencyMs": 1},
    }

    def responder(throttles: int) -> Any:
        sent: list[int] = []

        def before_send(request: Any, **_: Any) -> AWSResponse:
            sent.append(1)
            if len(sent) <= throttles:
                return AWSResponse(
                    request.url,
                    429,
                    {"x-amzn-ErrorType": "ThrottlingException"},
                    _Raw(b'{"message": "Too many requests"}'),
                )
            return AWSResponse(request.url, 200, {}, _Raw(json.dumps(answer).encode()))

        return before_send

    message = [{"role": "user", "content": [{"text": "hi"}]}]
    for purpose, throttles in (
        (ModelPurpose.INGESTION, 2),
        (ModelPurpose.EVALUATION, 1),
        (ModelPurpose.QUERY, 0),
    ):
        client = factory._client_for(purpose)
        client.meta.events.register("before-send.bedrock-runtime", responder(throttles))
        with usage.usage_scope() as meter:
            client.converse(modelId="vendor.model-a", messages=message)
        snapshot = meter.snapshot()
        assert snapshot["throttled_requests"] == throttles
        # Every attempt is a request sent, throttled or not.
        assert snapshot["sent_requests"] == throttles + 1
        if throttles:
            assert snapshot["by_role"] == {
                purpose.value: {**snapshot["by_role"][purpose.value]}
            }
            assert snapshot["by_model"]["vendor.model-a"]["throttled_requests"] == (
                throttles
            )
    # One that fails on its last throttled attempt counts every attempt.
    client = factory._client_for(ModelPurpose.INGESTION)
    client.meta.events.register_first("before-send.bedrock-runtime", responder(99))
    with usage.usage_scope() as meter, pytest.raises(Exception, match="Throttling"):
        client.converse(modelId="vendor.model-a", messages=message)
    attempts = factory.config.retry.model.client_max_attempts
    assert meter.snapshot()["throttled_requests"] == attempts
    assert meter.snapshot()["sent_requests"] == attempts
    # Validated as part of the usage record.
    assert LLMUsage.model_validate(meter.snapshot()).throttled_requests == 6


def test_embedding_and_rerank_requests_count_like_the_chat_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression (rv25): only the language model clients were counted, so an
    # embedding-heavy stage (indexing) reported 0 throttled requests of 0
    # sent although Bedrock throttled them, and the throttling alarm, which
    # reads this deployment's share, never saw it.
    import json

    import boto3
    from botocore.awsrequest import AWSResponse

    from unified_kg_rag.adapters.aws.bedrock import (
        BedrockEmbeddingModelFactory,
        BedrockRerankModelFactory,
    )

    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    stub_botocore_backoff(monkeypatch)
    session = boto3.Session(
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        region_name="us-east-1",
    )

    def responder(throttles: int, answer: dict[str, Any]) -> Any:
        sent: list[int] = []

        def before_send(request: Any, **_: Any) -> AWSResponse:
            sent.append(1)
            if len(sent) <= throttles:
                return AWSResponse(
                    request.url,
                    429,
                    {"x-amzn-ErrorType": "ThrottlingException"},
                    _Raw(b'{"message": "Too many requests"}'),
                )
            return AWSResponse(request.url, 200, {}, _Raw(json.dumps(answer).encode()))

        return before_send

    embedding = BedrockEmbeddingModelFactory(Config(), boto_session=session)
    client = embedding._client_for(ModelPurpose.INGESTION)
    client.meta.events.register(
        "before-send.bedrock-runtime",
        responder(2, {"embedding": [0.1], "inputTextTokenCount": 1}),
    )
    with usage.usage_scope() as meter:
        client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({"inputText": "hi"}),
        )
    snapshot = meter.snapshot()
    assert (snapshot["sent_requests"], snapshot["throttled_requests"]) == (3, 2)
    assert snapshot["by_role"]["embedding"]["throttled_requests"] == 2
    assert snapshot["by_model"]["amazon.titan-embed-text-v2:0"]["sent_requests"] == 3

    rerank = BedrockRerankModelFactory(Config(), boto_session=session)
    client = rerank._client_for(ModelPurpose.QUERY)
    client.meta.events.register(
        "before-send.bedrock-agent-runtime",
        # A query client: search.model_max_attempts (2) requests at most.
        responder(1, {"results": []}),
    )
    model_arn = "arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"
    with usage.usage_scope() as meter:
        client.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": "q"}}],
            sources=[
                {
                    "type": "INLINE",
                    "inlineDocumentSource": {
                        "type": "TEXT",
                        "textDocument": {"text": "d"},
                    },
                }
            ],
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {"modelArn": model_arn}
                },
            },
        )
    snapshot = meter.snapshot()
    assert (snapshot["sent_requests"], snapshot["throttled_requests"]) == (2, 1)
    # Under the role and model id the rerank calls are recorded under.
    assert snapshot["by_role"]["rerank"]["throttled_requests"] == 1
    assert snapshot["by_model"]["cohere.rerank-v3-5:0"]["sent_requests"] == 2


def test_a_factory_builds_only_the_clients_its_calls_use() -> None:
    # Old behaviour: every factory built the ingestion client eagerly, so a
    # query process (run-rag, run-mcp-server) paid for one it never used.
    import boto3

    from unified_kg_rag.adapters.aws.bedrock import BedrockLanguageModelFactory

    session = boto3.Session(
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        region_name="us-east-1",
    )
    factory = BedrockLanguageModelFactory(Config(), boto_session=session)
    assert factory._purpose_clients == {}
    factory._client_for(ModelPurpose.QUERY)
    assert set(factory._purpose_clients) == {ModelPurpose.QUERY}


async def test_a_call_started_in_a_task_that_ended_settles_once() -> None:
    # Regression (W2 review S4): LangChain starts a streamed model call in a
    # short-lived task and ends it in the task that read the stream. The
    # call was let go when the first task ended, so its end found nothing
    # and counted a second call (the estimate of the first stayed).
    handler = UsageCallbackHandler("m", role="query")
    run_id = uuid4()
    with usage.usage_scope() as meter:

        async def start() -> None:
            handler.on_chat_model_start(
                {}, [[HumanMessage(content="what does the Vendor ship")]], run_id=run_id
            )

        await asyncio.ensure_future(start())
        await asyncio.sleep(0)  # the task's done callbacks have run
        handler.on_llm_end(
            _result(
                content="x",
                usage_metadata={
                    "input_tokens": 7,
                    "output_tokens": 2,
                    "total_tokens": 9,
                },
            ),
            run_id=run_id,
        )
    snapshot = meter.snapshot()
    assert (snapshot["llm_calls"], snapshot["estimated_calls"]) == (1, 0)
    assert (snapshot["input_tokens"], snapshot["output_tokens"]) == (7, 2)


class _CutModel:
    """A chat model whose response fails (``error``, by default a
    ``RuntimeError``) after ``chunks`` streamed chunks of ``text`` (0:
    before any), streamed or invoked."""

    @staticmethod
    def build(
        chunks: int,
        handler: UsageCallbackHandler,
        *,
        text: str = "",
        error: Callable[[], BaseException] = lambda: RuntimeError("throttled"),
    ) -> Any:
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk, ChatResult

        class Cut(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "cut"

            def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
                raise error()

            def _stream(self, *args: Any, **kwargs: Any) -> Any:
                for _ in range(chunks):
                    yield ChatGenerationChunk(message=AIMessageChunk(content=text))
                raise error()

        return Cut(callbacks=[handler])


@pytest.mark.parametrize("chunks", [0, 1])
def test_a_failed_call_counts_its_estimate_once_its_response_started(
    chunks: int,
) -> None:
    # rv45 query low: a retried Bedrock stream that failed after its first
    # chunks counted 0 tokens, so the attempts its retry spent were outside
    # limits.max_input_tokens. One rule for a stream and an invoke: a call
    # whose response started (a chunk arrived: the model read the prompt)
    # keeps its estimate; one whose response never started is a failed call
    # (this model's invoke streams nothing; ChatOllama's streams inside the
    # call, so a drop mid-response keeps its estimate there too).
    handler = UsageCallbackHandler("vendor.model-v1", role="query")
    prompt = "a fairly long prompt " * 50
    estimate = estimate_token_count(prompt)
    model = _CutModel.build(chunks, handler)
    with usage.usage_scope() as streamed:
        with pytest.raises(RuntimeError):
            list(model.stream(prompt))
    with usage.usage_scope() as invoked:
        with pytest.raises(RuntimeError):
            model.invoke(prompt)
    got = LLMUsage.model_validate(streamed.snapshot())
    if chunks:
        assert (got.llm_calls, got.estimated_calls, got.failed_calls) == (1, 1, 0)
        assert got.input_tokens == estimate
    else:
        assert (got.llm_calls, got.failed_calls, got.input_tokens) == (0, 1, 0)
    failed = LLMUsage.model_validate(invoked.snapshot())
    assert (failed.llm_calls, failed.failed_calls, failed.input_tokens) == (0, 1, 0)
    assert not handler._calls


def test_a_failed_stream_counts_the_output_it_sent_and_the_sdks_resends() -> None:
    # rv46 test low: a failed stream's output estimate and its request
    # retries were each settable to 0 without a test failing.
    from botocore.exceptions import EventStreamError

    def error() -> BaseException:
        return EventStreamError(
            {
                "Error": {"Code": "throttlingException", "Message": "slow down"},
                "ResponseMetadata": {"RetryAttempts": 2},
            },
            "ConverseStream",
        )

    handler = UsageCallbackHandler("vendor.model-v1", role="query")
    text = "Vendor ships parts "
    model = _CutModel.build(3, handler, text=text, error=error)
    with usage.usage_scope() as meter:
        with pytest.raises(EventStreamError):
            list(model.stream("what does the Vendor ship"))
    got = LLMUsage.model_validate(meter.snapshot())
    assert (got.llm_calls, got.estimated_calls) == (1, 1)
    assert got.output_tokens == estimate_token_count(text * 3) > 0
    assert got.request_retries == 2


async def test_a_long_lived_handler_holds_no_call_it_cannot_settle() -> None:
    # rv46 test low: tokens of a call the handler does not hold (made
    # outside a scope, or after it settled), and a streamed call whose task
    # was cancelled, left their run ids in a set that only grew.
    handler = UsageCallbackHandler("m", role="query")
    for _ in range(3):
        handler.on_llm_new_token("x", run_id=uuid4())
    settled = uuid4()
    with usage.usage_scope():
        handler.on_chat_model_start({}, [[HumanMessage(content="q")]], run_id=settled)
        handler.on_llm_end(_result(content="x"), run_id=settled)
        handler.on_llm_new_token("late", run_id=settled)
        cancelled = uuid4()

        async def start() -> None:
            handler.on_chat_model_start(
                {}, [[HumanMessage(content="q")]], run_id=cancelled
            )
            handler.on_llm_new_token("a", run_id=cancelled)
            await asyncio.sleep(10)

        task = asyncio.ensure_future(start())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
    assert not handler._calls and not handler._tasks


class _DropOnce:
    """A chat model whose first ``faults`` responses stream two chunks and
    drop (``ConnectionResetError``), then answer: through ``stream`` and
    ``astream``, and through ``invoke``/``ainvoke``, which stream inside
    the call as ``ChatOllama``'s do (``on_llm_new_token``)."""

    ANSWER = "Vendor ships parts to Buyer."

    @staticmethod
    def build(faults: int, handler: UsageCallbackHandler) -> Any:
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk, ChatResult

        state = {"attempts": 0}

        def chunks() -> Any:
            state["attempts"] += 1
            for _ in range(2):
                yield ChatGenerationChunk(message=AIMessageChunk(content=""))
            if state["attempts"] <= faults:
                raise ConnectionResetError("connection dropped mid-stream")
            yield ChatGenerationChunk(message=AIMessageChunk(content=_DropOnce.ANSWER))

        class Drop(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "drop-once"

            def _generate(
                self,
                messages: Any,
                stop: Any = None,
                run_manager: Any = None,
                **kw: Any,
            ) -> ChatResult:
                text = ""
                for chunk in chunks():
                    if run_manager is not None:
                        run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    text += chunk.text
                return ChatResult(generations=[ChatGeneration(message=AIMessage(text))])

            def _stream(self, *args: Any, **kwargs: Any) -> Any:
                yield from chunks()

        model = Drop(callbacks=[handler])
        return model, state


def _retried(model: Any) -> Any:
    from unified_kg_rag.adapters.llm.chain_factory import with_transient_retry
    from unified_kg_rag.domain.models.config import RetryPolicy

    return with_transient_retry(
        model,
        operation="answer",
        retry=RetryPolicy(
            max_attempts=3, base_delay_seconds=0.0, max_delay_seconds=0.0
        ),
        is_transient=lambda e: isinstance(e, ConnectionError),
    )


async def _answer(chain: Any, how: str, prompt: str) -> str:
    if how == "invoke":
        return str(chain.invoke(prompt).content)
    if how == "ainvoke":
        return str((await chain.ainvoke(prompt)).content)
    if how == "stream":
        return "".join(c.text for c in chain.stream(prompt))
    return "".join([c.text async for c in chain.astream(prompt)])


_PATHS = ["invoke", "ainvoke", "stream", "astream"]
_PROMPT = "a fairly long prompt about the Vendor and the Buyer " * 20


def _query_limits(**kwargs: Any) -> usage.UsageLimits:
    # A query's (search.query_max_*): they bound the calls it plans.
    return usage.UsageLimits(counts_retries=False, **kwargs)


@pytest.mark.parametrize("how", _PATHS)
@pytest.mark.parametrize("limit", ["calls", "tokens"])
@pytest.mark.parametrize(("faults", "spare"), [(1, 0), (2, 0), (2, 1)])
async def test_a_retry_within_a_query_limit_reuses_its_failed_attempts_room(
    how: str, limit: str, faults: int, spare: int
) -> None:
    # Regression (rv46 medium, from #794): a stream that dropped after its
    # response started keeps its estimate, so under a binding query limit
    # (the answer packed up to query_max_input_tokens, or the one call
    # query_max_model_calls keeps for it) its retry was refused and a
    # transient fault failed the query as limit_exceeded. A query's limits
    # count each logical call once; the usage still counts each attempt.
    handler = UsageCallbackHandler("vendor.model-v1", role="query")
    model, state = _DropOnce.build(faults, handler)
    estimate = estimate_token_count(_PROMPT)
    limits = (
        _query_limits(max_llm_calls=1 + spare)
        if limit == "calls"
        else _query_limits(max_input_tokens=(1 + spare) * estimate)
    )
    with usage.usage_scope(limits=limits) as meter:
        assert await _answer(_retried(model), how, _PROMPT) == _DropOnce.ANSWER
        left = usage.remaining()
    assert state["attempts"] == faults + 1
    got = LLMUsage.model_validate(meter.snapshot())
    # Every attempt was sent: each is a call, all estimated (no usage
    # reported), the dropped ones at their prompt's estimate.
    attempts = faults + 1
    assert (got.llm_calls, got.estimated_calls, got.failed_calls) == (
        attempts,
        attempts,
        0,
    )
    assert got.input_tokens == attempts * estimate
    # What the limit counted: the one logical call, the rest still room.
    assert (left.llm_calls, left.input_tokens) == (
        (spare, None) if limit == "calls" else (None, spare * estimate)
    )


def test_an_attempt_superseded_in_flight_leaves_the_limits_when_it_settles() -> None:
    # A retry may start before its failed attempt's call is settled (its
    # end comes later): what that call settles at stays out of the limits
    # that count a call once, whatever it is.
    limits = _query_limits(max_llm_calls=3, max_input_tokens=100)
    with usage.usage_scope(limits=limits) as meter:
        retried = usage.RetriedCall()
        with retried.attempt():
            first = usage.start_llm_call("query", "m", estimated_input_tokens=30)
        with retried.attempt():
            second = usage.start_llm_call("query", "m", estimated_input_tokens=30)
        assert usage.remaining() == usage.Remaining(
            llm_calls=2,
            input_tokens=70,
            call_limit="max_llm_calls",
            call_limit_value=3,
        )
        first.finish(input_tokens=45, output_tokens=5)
        second.fail()
        left = usage.remaining()
    assert (left.llm_calls, left.input_tokens) == (3, 100)
    got = meter.snapshot()
    assert (got["llm_calls"], got["input_tokens"], got["failed_calls"]) == (1, 45, 1)


@pytest.mark.parametrize("how", _PATHS)
async def test_a_fault_on_every_attempt_within_a_query_limit_is_the_models_error(
    how: str,
) -> None:
    handler = UsageCallbackHandler("vendor.model-v1", role="query")
    model, state = _DropOnce.build(99, handler)
    limits = _query_limits(
        max_llm_calls=1, max_input_tokens=estimate_token_count(_PROMPT)
    )
    with usage.usage_scope(limits=limits) as meter:
        with pytest.raises(ConnectionResetError):
            await _answer(_retried(model), how, _PROMPT)
    assert state["attempts"] == 3
    got = LLMUsage.model_validate(meter.snapshot())
    assert (got.llm_calls, got.estimated_calls) == (3, 3)


@pytest.mark.parametrize("how", _PATHS)
async def test_a_run_budget_counts_every_attempt_of_a_retried_call(how: str) -> None:
    # An ingestion run's budget (processing.max_run_*) bounds what was sent
    # and billed: a retry's attempts each take a call of it.
    handler = UsageCallbackHandler("vendor.model-v1", role="ingestion")
    model, state = _DropOnce.build(1, handler)
    with usage.usage_scope(limits=usage.UsageLimits(max_llm_calls=1)) as meter:
        with pytest.raises(UsageLimitExceededError):
            await _answer(_retried(model), how, _PROMPT)
    assert state["attempts"] == 1
    assert meter.snapshot()["llm_calls"] == 1


@pytest.mark.parametrize("how", _PATHS)
async def test_a_retried_call_without_limits_counts_each_attempt(how: str) -> None:
    handler = UsageCallbackHandler("vendor.model-v1", role="query")
    model, state = _DropOnce.build(1, handler)
    with usage.usage_scope() as meter:
        assert await _answer(_retried(model), how, _PROMPT) == _DropOnce.ANSWER
    assert state["attempts"] == 2
    got = LLMUsage.model_validate(meter.snapshot())
    assert (got.llm_calls, got.estimated_calls, got.failed_calls) == (2, 2, 0)
    assert got.input_tokens == 2 * estimate_token_count(_PROMPT)


class _Fixer:
    """A factory whose fixing model, metered by ``handler``, answers the
    repair with ``{"answer": "Vendor"}``."""

    def __init__(self, handler: UsageCallbackHandler) -> None:
        self.handler = handler
        self.calls = 0

    def get_model(self, **kwargs: Any) -> Any:
        from langchain_core.language_models import BaseChatModel
        from langchain_core.outputs import ChatResult

        fixer = self

        class Fix(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "fixer"

            def _generate(self, *args: Any, **kw: Any) -> ChatResult:
                fixer.calls += 1
                message = AIMessage('{"answer": "Vendor"}')
                return ChatResult(generations=[ChatGeneration(message=message)])

        return Fix(callbacks=[self.handler])


@pytest.mark.parametrize("how", ["invoke", "ainvoke"])
@pytest.mark.parametrize("faults", [0, 1])
async def test_an_answer_and_its_repair_in_one_attempt_count_as_two_calls(
    how: str, faults: int
) -> None:
    # rv47 test low: the output repair runs inside the retried chain, in the
    # attempt that answered. It is a call of its own there (it supersedes
    # no call of its attempt, so a third call past max_llm_calls=2 is
    # refused), and an optional one whose room is read without the retried
    # attempts (the answer's dropped first attempt does not skip it).
    from unified_kg_rag.adapters.llm.chain_factory import create_json_output_parser

    handler = UsageCallbackHandler("vendor.model-v1", role="query")
    model, state = _DropOnce.build(faults, handler)
    fixer = _Fixer(handler)
    parser = create_json_output_parser(
        fixer, "fixer", answer_keys=("answer",)  # type: ignore[arg-type]
    )
    chain = _retried(model | parser)
    with usage.usage_scope(limits=_query_limits(max_llm_calls=2)) as meter:
        if how == "invoke":
            answer = chain.invoke(_PROMPT)
        else:
            answer = await chain.ainvoke(_PROMPT)
        assert usage.remaining().llm_calls == 0
        with pytest.raises(UsageLimitExceededError):
            usage.start_llm_call("query", "m", estimated_input_tokens=1)
    assert answer == {"answer": "Vendor"}
    assert (state["attempts"], fixer.calls) == (faults + 1, 1)
    assert meter.snapshot()["llm_calls"] == faults + 2


async def test_a_stream_closed_early_settles_its_call_at_once() -> None:
    # rv47 test low: a caller that stops reading a retried stream (a client
    # disconnect, a consumer that takes the first chunks) closes the model's
    # stream too, so its call settles then, not when the stream is
    # collected (an async generator only on a later turn of the loop).
    handler = UsageCallbackHandler("vendor.model-v1", role="query")
    model, _ = _DropOnce.build(0, handler)
    with usage.usage_scope():
        chunks = _retried(model).stream(_PROMPT)
        next(chunks)
        assert handler._calls
        chunks.close()
        assert not handler._calls
        streamed = _retried(model).astream(_PROMPT)
        await anext(streamed)
        assert handler._calls
        await streamed.aclose()
        assert not handler._calls


async def test_an_attempts_reader_closes_its_model_stream_when_closed() -> None:
    # The model's stream may be held elsewhere too (so not collected when
    # the reader is): closing the reader closes it all the same.
    from unified_kg_rag.adapters.llm.chain_factory import _ain_attempt, _in_attempt

    closed: list[str] = []

    def source() -> Any:
        try:
            yield 1
            yield 2
        finally:
            closed.append("sync")

    async def asource() -> Any:
        try:
            yield 1
            yield 2
        finally:
            closed.append("async")

    attempt = usage.RetriedCall().attempt()
    held = source()
    reader = _in_attempt(held, attempt)
    assert next(reader) == 1
    reader.close()
    aheld = asource()
    areader = _ain_attempt(aheld, attempt)
    assert await anext(areader) == 1
    await areader.aclose()
    assert closed == ["sync", "async"]


async def test_a_local_embedding_request_is_counted_with_estimated_tokens() -> None:
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
    from unified_kg_rag.adapters.local.embeddings import TracedEmbeddings

    model = TracedEmbeddings(
        DeterministicFakeEmbedding(size=4),
        model_id="synthetic-embed",
        provider_name="ollama",
        max_input_tokens=512,
    )
    texts = ["Vendor ships widgets.", "Depot stores them."]
    model.embed_documents(texts)  # outside a scope: not recorded
    with usage.usage_scope() as meter:
        model.embed_documents(texts)
        model.embed_query("Who ships?")
        await model.aembed_documents(texts)
        await model.aembed_query("Who stores?")
    counts = meter.snapshot()["by_model"]["synthetic-embed"]
    expected = 2 * sum(map(estimate_token_count, texts)) + sum(
        map(estimate_token_count, ["Who ships?", "Who stores?"])
    )
    assert counts["embedding_calls"] == counts["embedding_estimated_calls"] == 4
    assert counts["embedding_input_tokens"] == expected
    assert counts["llm_calls"] == 0
    assert meter.snapshot()["by_role"]["embedding"]["embedding_calls"] == 4
