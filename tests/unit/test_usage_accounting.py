# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Always-on LLM usage accounting (shared.usage + the usage callback)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tests.fixtures.fakes.model_servers import FakeOllama, ModelRejects
from unified_kg_rag.adapters.llm.response_cache import (
    CachingLLMFactory,
    LLMResponseCache,
)
from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory
from unified_kg_rag.adapters.observability.usage_callback import UsageCallbackHandler
from unified_kg_rag.domain.models import Config, LLMUsage, ModelPurpose
from unified_kg_rag.shared import observability, usage

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

    first = LLMResponseCache()
    first.open(tmp_path)
    with usage.usage_scope() as run_one:
        _model(first).invoke([HumanMessage(content="extract")])
    first.close()
    second = LLMResponseCache()
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
