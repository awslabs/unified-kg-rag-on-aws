# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured-output pilot for LightRAG keyword extraction (AWS-free).

``search.lightrag_search.structured_keyword_output`` asks the model for a
JSON-schema constrained answer (``with_structured_output(method=
"json_schema")``). Each provider class is exercised with its real LangChain
chat model and a fake transport, and the request is checked for the native
response format: Converse ``outputConfig.textFormat`` (a recording boto client
stand-in), Ollama ``format`` and OpenAI ``response_format`` (``httpx``
mock transports). Unsupported models and failed calls fall back to the text
parser.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from tests.fixtures.fakes.boto import FakeBotoSession
from unified_kg_rag.adapters.aws.bedrock import (
    BedrockCrossRegionModelHelper,
    BedrockLanguageModelFactory,
)
from unified_kg_rag.adapters.llm.chain_factory import (
    setup_structured_chain,
    structured_output_unsupported_reason,
)
from unified_kg_rag.adapters.local.models import (
    OllamaLanguageModelFactory,
    OpenAICompatibleLanguageModelFactory,
)
from unified_kg_rag.application.retrieval import rag_chain as rag_chain_mod
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import Config, LanguageModelId
from unified_kg_rag.domain.prompts.retrieval import KeywordsExtractionPrompt

pytestmark = pytest.mark.unit

SCHEMA = GraphRAGChain._KEYWORD_SCHEMA
ANSWER = {"high_level_keywords": ["supply"], "low_level_keywords": ["Vendor"]}
INPUTS = {"query": "Who supplies Buyer?", "target_language": "English"}


# --- Bedrock ------------------------------------------------------------------


class _RecordingRuntime:
    """bedrock-runtime stand-in: records each Converse request."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": json.dumps(ANSWER)}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "metrics": {"latencyMs": 1},
        }


class _RuntimeSession(FakeBotoSession):
    def __init__(self, runtime: _RecordingRuntime) -> None:
        super().__init__()
        self.runtime = runtime

    def client(self, service_name: str, **kwargs: Any) -> Any:
        return self.runtime if service_name == "bedrock-runtime" else object()


def _bedrock_factory(mocker: Any, resolved_id: str) -> tuple[Any, _RecordingRuntime]:
    mocker.patch.object(
        BedrockCrossRegionModelHelper,
        "get_cross_region_model_id",
        return_value=resolved_id,
    )
    runtime = _RecordingRuntime()
    factory = BedrockLanguageModelFactory(
        Config(), boto_session=_RuntimeSession(runtime)  # type: ignore[arg-type]
    )
    return factory, runtime


async def test_converse_sends_output_config_text_format(mocker: Any) -> None:
    # Claude Sonnet 4.5 without thinking carries no effort output_config.
    model = LanguageModelId.CLAUDE_V4_5_SONNET
    factory, runtime = _bedrock_factory(mocker, f"us.{model.value}")
    chain = setup_structured_chain(factory, model, KeywordsExtractionPrompt, SCHEMA)
    assert chain is not None
    assert await chain.ainvoke(INPUTS) == ANSWER
    (request,) = runtime.requests
    text_format = request["outputConfig"]["textFormat"]
    assert text_format["type"] == "json_schema"
    sent = json.loads(text_format["structure"]["jsonSchema"]["schema"])
    assert sent["required"] == ["high_level_keywords", "low_level_keywords"]
    assert sent["additionalProperties"] is False
    assert "output_config" not in (request.get("additionalModelRequestFields") or {})


def test_converse_with_effort_output_config_is_left_out(mocker: Any) -> None:
    # Claude 5.5 always sends output_config.effort; the combination with
    # outputConfig.textFormat is unverified, so the text parser stays.
    model = LanguageModelId.CLAUDE_V5_5_HAIKU
    factory, _ = _bedrock_factory(mocker, f"global.{model.value}")
    llm = factory.get_model(model)
    assert "output_config" in llm.additional_model_request_fields
    reason = structured_output_unsupported_reason(llm, factory.get_model_info(model))
    assert reason is not None and "output_config" in reason
    assert (
        setup_structured_chain(factory, model, KeywordsExtractionPrompt, SCHEMA) is None
    )


def test_converse_model_without_structured_output_is_left_out(mocker: Any) -> None:
    model = LanguageModelId.CLAUDE_V3_7_SONNET
    factory, _ = _bedrock_factory(mocker, f"us.{model.value}")
    llm = factory.get_model(model)
    assert type(llm).__name__ == "ChatBedrockConverse"
    assert "structured-output" in str(
        structured_output_unsupported_reason(llm, factory.get_model_info(model))
    )


def test_invoke_model_chat_bedrock_is_left_out(mocker: Any) -> None:
    # ChatBedrock has only tool-based structured output.
    model = LanguageModelId.CLAUDE_V3_7_SONNET
    factory, _ = _bedrock_factory(mocker, model.value)
    llm = factory.get_model(model)
    assert type(llm).__name__ == "ChatBedrock"
    assert "ChatBedrock" in str(
        structured_output_unsupported_reason(llm, factory.get_model_info(model))
    )


# --- local servers --------------------------------------------------------------


def _local_config(kind: str, base_url: str) -> Config:
    return Config.model_validate(
        {
            "providers": {"llm": {"kind": kind, "base_url": base_url}},
            "models": {"default": "local-model"},
        }
    )


async def test_ollama_sends_the_schema_as_format() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "created_at": "2026-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": json.dumps(ANSWER)},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
        )

    config = _local_config("ollama", "http://ollama.test:11434")
    factory = OllamaLanguageModelFactory(config, transport=httpx.MockTransport(handler))
    chain = setup_structured_chain(
        factory, "local-model", KeywordsExtractionPrompt, SCHEMA
    )
    assert chain is not None
    assert await chain.ainvoke(INPUTS) == ANSWER
    (body,) = requests
    assert body["format"]["required"] == SCHEMA["required"]
    assert body["format"]["properties"] == SCHEMA["properties"]


async def test_openai_compatible_sends_a_json_schema_response_format() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(ANSWER),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    config = _local_config("openai_compatible", "http://vllm.test/v1")
    factory = OpenAICompatibleLanguageModelFactory(
        config, transport=httpx.MockTransport(handler)
    )
    chain = setup_structured_chain(
        factory, "local-model", KeywordsExtractionPrompt, SCHEMA
    )
    assert chain is not None
    assert await chain.ainvoke(INPUTS) == ANSWER
    (body,) = requests
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"]["required"] == SCHEMA["required"]


# --- the keyword step -----------------------------------------------------------


class _Chain:
    def __init__(self, output: Any = None, error: Exception | None = None) -> None:
        self.output, self.error, self.calls = output, error, 0

    async def ainvoke(self, _inputs: dict, config: Any = None) -> Any:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.output


@pytest.fixture
def chain(config: Config) -> GraphRAGChain:
    return GraphRAGChain(config=config)


async def test_flag_off_never_builds_a_structured_chain(
    chain: GraphRAGChain, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds: list[Any] = []
    monkeypatch.setattr(
        rag_chain_mod, "setup_structured_chain", lambda **k: builds.append(k)
    )
    text = _Chain(json.dumps(ANSWER))
    monkeypatch.setattr(chain, "_get_chain_for_prompt", lambda *a, **k: text)
    assert await chain._extract_dual_keywords("q", "English") == (
        ["supply"],
        ["Vendor"],
    )
    assert builds == []


async def test_flag_on_uses_the_structured_answer(
    chain: GraphRAGChain, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain.config.search.lightrag_search.structured_keyword_output = True
    structured = _Chain(ANSWER)
    builds: list[Any] = []

    def build(**kwargs: Any) -> Any:
        builds.append(kwargs)
        return structured

    monkeypatch.setattr(rag_chain_mod, "setup_structured_chain", build)
    text = _Chain(error=AssertionError("text parser called"))
    monkeypatch.setattr(chain, "_get_chain_for_prompt", lambda *a, **k: text)
    for _ in range(2):
        assert await chain._extract_dual_keywords("q", "English") == (
            ["supply"],
            ["Vendor"],
        )
    assert len(builds) == 1 and builds[0]["schema"] is SCHEMA
    assert structured.calls == 2 and text.calls == 0


@pytest.mark.parametrize(
    "structured",
    [
        pytest.param(None, id="unsupported-model"),
        pytest.param(_Chain(error=RuntimeError("ValidationException")), id="fails"),
        pytest.param(_Chain(["not", "an", "object"]), id="not-an-object"),
        pytest.param(_Chain({"high_level_keywords": "x"}), id="bad-payload"),
    ],
)
async def test_flag_on_falls_back_to_the_text_parser(
    chain: GraphRAGChain, monkeypatch: pytest.MonkeyPatch, structured: Any
) -> None:
    chain.config.search.lightrag_search.structured_keyword_output = True
    monkeypatch.setattr(rag_chain_mod, "setup_structured_chain", lambda **_: structured)
    text = _Chain(json.dumps(ANSWER))
    monkeypatch.setattr(chain, "_get_chain_for_prompt", lambda *a, **k: text)
    assert await chain._extract_dual_keywords("q", "English") == (
        ["supply"],
        ["Vendor"],
    )
    assert text.calls == 1
