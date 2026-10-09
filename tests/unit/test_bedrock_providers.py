# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for provider-aware Bedrock request shaping (AWS-free).

Covers the Claude 4.6-5.5 and OpenAI GPT capability records and the provider
routing built on them: GPT goes through Converse with ``reasoning.effort`` and
no Anthropic-only fields, Anthropic adaptive thinking carries ``effort`` in
``output_config``, explicit prompt-cache markers follow
``supports_prompt_caching``, and CountTokens is skipped for models that do not
support it. No boto client is invoked; factories use a stub session.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

from unified_kg_rag.adapters.aws import bedrock as bedrock_mod
from unified_kg_rag.adapters.aws.bedrock import (
    BedrockCrossRegionModelHelper,
    BedrockLanguageModelFactory,
)
from unified_kg_rag.adapters.aws.bedrock_models import get_language_model_info
from unified_kg_rag.adapters.aws.chain_factory import setup_chain
from unified_kg_rag.adapters.aws.token_counter import BedrockTokenCounter
from unified_kg_rag.domain.models import Config, LanguageModelId
from unified_kg_rag.domain.prompts.graph_extraction import GraphRefinementPrompt
from unified_kg_rag.shared import LanguageModelError

pytestmark = pytest.mark.unit

GPT_MODELS = [m for m in LanguageModelId if m.value.startswith("openai.")]
NEW_CLAUDE_MODELS = [
    LanguageModelId.CLAUDE_V5_5_SONNET,
    LanguageModelId.CLAUDE_V5_5_OPUS,
    LanguageModelId.CLAUDE_V5_5_HAIKU,
    LanguageModelId.CLAUDE_V4_8_OPUS,
    LanguageModelId.CLAUDE_V4_7_OPUS,
    LanguageModelId.CLAUDE_V4_6_OPUS,
    LanguageModelId.CLAUDE_V4_6_SONNET,
]


class _FakeSession:
    profile_name = "default"

    def client(self, service_name: str, **kwargs: Any) -> Any:
        return object()

    def get_credentials(self) -> Any:
        return None


def _factory(config: Config | None = None) -> BedrockLanguageModelFactory:
    return BedrockLanguageModelFactory(
        config or Config(), boto_session=_FakeSession()  # type: ignore[arg-type]
    )


def _info(model_id: LanguageModelId) -> Any:
    info = get_language_model_info(model_id)
    assert info is not None, model_id
    return info


# --- capability table ------------------------------------------------------


def test_gpt_catalog_is_proprietary_only() -> None:
    expected = {
        "openai.gpt-5.4",
        "openai.gpt-5.5",
        "openai.gpt-5.6-luna",
        "openai.gpt-5.6-sol",
        "openai.gpt-5.6-terra",
        "openai.gpt-6-astra",
        "openai.gpt-6-luna",
        "openai.gpt-6-sol",
        "openai.gpt-6.1-sol",
    }
    assert {m.value for m in GPT_MODELS} == expected
    assert not any("gpt-oss" in m.value for m in LanguageModelId)


def test_provider_matches_model_id_prefix() -> None:
    for model_id in LanguageModelId:
        info = _info(model_id)
        assert model_id.value.startswith(f"{info.provider}."), model_id


@pytest.mark.parametrize("model_id", GPT_MODELS)
def test_gpt_capabilities(model_id: LanguageModelId) -> None:
    info = _info(model_id)
    assert info.provider == "openai"
    assert info.requires_inference_profile is True
    assert info.supports_prompt_caching is False
    assert info.supports_count_tokens is False
    assert info.supports_sampling_params is False
    assert info.always_reasons is True
    assert info.uses_adaptive_thinking is False
    assert info.context_window_size >= 1000000
    assert info.max_output_tokens >= 128000
    # Live Bedrock lists exactly these levels (plus 'none') for every GPT model.
    assert info.supported_efforts == {"low", "medium", "high", "xhigh", "max"}


@pytest.mark.parametrize("model_id", NEW_CLAUDE_MODELS)
def test_new_claude_capabilities(model_id: LanguageModelId) -> None:
    info = _info(model_id)
    assert info.provider == "anthropic"
    assert info.requires_inference_profile is True
    assert info.supports_prompt_caching is True
    assert info.uses_adaptive_thinking is True
    assert info.native_1m_context_window is True
    assert info.context_window_size == 1000000


def test_claude_4_6_keeps_opt_in_thinking_and_sampling() -> None:
    for model_id in (
        LanguageModelId.CLAUDE_V4_6_OPUS,
        LanguageModelId.CLAUDE_V4_6_SONNET,
    ):
        info = _info(model_id)
        assert info.adaptive_thinking_only is False
        assert info.always_reasons is False
        assert info.supports_sampling_params is True
        # Bedrock rejects 'xhigh' on Claude 4.6 (both Opus and Sonnet).
        assert info.supported_efforts == {"low", "medium", "high", "max"}
    assert _info(LanguageModelId.CLAUDE_V4_6_SONNET).max_output_tokens == 64000


def test_fable_models_are_not_offered() -> None:
    # Fable 5 / 5.1 reject accounts on the default data-retention mode.
    assert not any("fable" in m.value for m in LanguageModelId)


# --- inference-profile resolution ----------------------------------------


class _FakeBedrockClient:
    def __init__(self, profiles: set[str]) -> None:
        self._profiles = profiles

    def list_inference_profiles(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "inferenceProfileSummaries": [
                {"inferenceProfileId": p} for p in sorted(self._profiles)
            ]
        }


class _ProfileSession:
    def __init__(self, profiles: set[str]) -> None:
        self._client = _FakeBedrockClient(profiles)

    def client(self, service_name: str, **kwargs: Any) -> Any:
        return self._client


@pytest.fixture(autouse=True)
def _clear_profile_cache() -> Any:
    BedrockCrossRegionModelHelper._profiles_by_region.clear()
    yield
    BedrockCrossRegionModelHelper._profiles_by_region.clear()


def test_gpt_ids_build_us_and_global_profiles() -> None:
    build = BedrockCrossRegionModelHelper._build_cross_region_model_id
    assert build(LanguageModelId.GPT_V6_1_SOL, "us-west-2") == "us.openai.gpt-6.1-sol"
    assert (
        build(LanguageModelId.GPT_V6_1_SOL, "us-west-2", is_global=True)
        == "global.openai.gpt-6.1-sol"
    )


@pytest.mark.parametrize("enable_global", [True, False])
def test_gpt_id_resolves_to_available_profile(enable_global: bool) -> None:
    model_id = LanguageModelId.GPT_V6_SOL
    session = _ProfileSession({f"global.{model_id.value}", f"us.{model_id.value}"})
    resolved = BedrockCrossRegionModelHelper.get_cross_region_model_id(
        session,  # type: ignore[arg-type]
        model_id,
        "us-west-2",
        enable_global_profile=enable_global,
    )
    prefix = "global" if enable_global else "us"
    assert resolved == f"{prefix}.{model_id.value}"


def test_gpt_get_model_uses_converse(mocker) -> None:
    factory = _factory()
    mocker.patch.object(
        bedrock_mod.BedrockCrossRegionModelHelper,
        "get_cross_region_model_id",
        return_value="global.openai.gpt-6-sol",
    )
    captured: dict[str, Any] = {}

    class _FakeConverse:
        def __new__(cls, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return "converse"

    class _FakeInvoke:
        def __new__(cls, **kwargs: Any) -> Any:
            raise AssertionError("GPT must not use the InvokeModel client")

    mocker.patch.object(bedrock_mod, "ChatBedrockConverse", _FakeConverse)
    mocker.patch.object(bedrock_mod, "ChatBedrock", _FakeInvoke)
    assert factory.get_model(LanguageModelId.GPT_V6_SOL) == "converse"
    assert captured["model_id"] == "global.openai.gpt-6-sol"


def test_gpt_get_model_fails_fast_without_profile(mocker) -> None:
    factory = _factory()
    mocker.patch.object(
        bedrock_mod.BedrockCrossRegionModelHelper,
        "get_cross_region_model_id",
        return_value=LanguageModelId.GPT_V6_SOL.value,
    )
    with pytest.raises(LanguageModelError, match="cross-region inference profile"):
        factory.get_model(LanguageModelId.GPT_V6_SOL)


# --- request shaping --------------------------------------------------------


_ANTHROPIC_ONLY_FIELDS = ("thinking", "output_config", "anthropic_beta")


@pytest.mark.parametrize("model_id", GPT_MODELS)
def test_gpt_request_has_reasoning_effort_object_and_no_anthropic_fields(
    model_id: LanguageModelId,
) -> None:
    config = Config()
    config.aws.bedrock.enable_1m_context = True
    config.aws.bedrock.default_effort = "medium"
    factory = _factory(config)
    cfg = factory._build_model_config(_info(model_id), f"us.{model_id.value}", True)
    fields = cfg["additional_model_request_fields"]
    assert fields == {"reasoning": {"effort": "medium"}}
    for key in _ANTHROPIC_ONLY_FIELDS:
        assert key not in fields
    assert "stop_sequences" not in cfg
    assert "temperature" not in cfg
    assert "top_k" not in cfg


def test_gpt_reasoning_effort_per_call_override() -> None:
    factory = _factory()
    info = _info(LanguageModelId.GPT_V6_LUNA)
    assert factory._build_thinking_config(info, effort="low") == {
        "reasoning": {"effort": "low"}
    }


def test_claude_request_keeps_anthropic_stop_sequence() -> None:
    factory = _factory()
    info = _info(LanguageModelId.CLAUDE_V5_5_SONNET)
    cfg = factory._build_model_config(info, "us.anthropic.claude-sonnet-5-5", True)
    assert cfg["stop_sequences"] == ["\n\nHuman:"]


def test_claude_5_5_gets_adaptive_thinking_by_default() -> None:
    factory = _factory()
    for model_id in (
        LanguageModelId.CLAUDE_V5_5_SONNET,
        LanguageModelId.CLAUDE_V5_5_OPUS,
    ):
        cfg = factory._build_model_config(_info(model_id), f"us.{model_id.value}", True)
        fields = cfg["additional_model_request_fields"]
        assert fields["thinking"] == {"type": "adaptive"}
        assert fields["output_config"] == {"effort": "high"}
        assert "reasoning" not in fields
        assert "anthropic_beta" not in fields
        assert "temperature" not in cfg


def test_claude_4_6_thinking_is_opt_in_and_adaptive() -> None:
    factory = _factory()
    info = _info(LanguageModelId.CLAUDE_V4_6_SONNET)
    off = factory._build_model_config(info, "us.anthropic.claude-sonnet-4-6", True)
    assert "additional_model_request_fields" not in off
    assert off["temperature"] == factory.DEFAULT_TEMPERATURE

    on = factory._build_model_config(
        info, "us.anthropic.claude-sonnet-4-6", True, enable_thinking=True
    )
    fields = on["additional_model_request_fields"]
    assert fields["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in fields["thinking"]
    assert fields["output_config"] == {"effort": "high"}
    assert on["temperature"] == 1.0


def test_undocumented_effort_level_fails_fast() -> None:
    factory = _factory()
    for model_id in (
        LanguageModelId.CLAUDE_V4_6_SONNET,
        LanguageModelId.CLAUDE_V4_6_OPUS,
    ):
        with pytest.raises(LanguageModelError, match="not supported by this model"):
            factory._build_thinking_config(_info(model_id), effort="xhigh")
    out = factory._build_thinking_config(_info(LanguageModelId.GPT_V5_5), effort="max")
    assert out == {"reasoning": {"effort": "max"}}


def test_guardrail_applies_to_gpt_converse_request() -> None:
    config = Config()
    config.aws.bedrock.guardrail.identifier = "gid-1"
    factory = _factory(config)
    cfg = factory._build_model_config(
        _info(LanguageModelId.GPT_V6_SOL), "us.openai.gpt-6-sol", True
    )
    assert cfg["guardrail_config"]["guardrailIdentifier"] == "gid-1"


# --- prompt caching -------------------------------------------------------


class _StubFactory:
    def get_model(self, model_id: LanguageModelId, **kwargs: Any) -> Any:
        return FakeListChatModel(responses=["ok"])

    def get_model_info(self, model_id: LanguageModelId) -> Any:
        return get_language_model_info(model_id)


def _system_message(model_id: LanguageModelId) -> Any:
    chain = setup_chain(
        _StubFactory(),  # type: ignore[arg-type]
        model_id,
        GraphRefinementPrompt,
        StrOutputParser(),
    )
    prompt = chain.first  # type: ignore[attr-defined]
    values = dict.fromkeys(prompt.input_variables, "x")
    return prompt.format_messages(**values)[0]


def _has_cache_marker(message: SystemMessage) -> bool:
    content = message.content
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("cache_control") == {"type": "ephemeral"}
        for block in content
    )


def test_prompt_cache_marker_only_for_caching_models() -> None:
    cached = _system_message(LanguageModelId.CLAUDE_V5_5_SONNET)
    assert isinstance(cached, SystemMessage)
    assert _has_cache_marker(cached)

    for model_id in GPT_MODELS:
        plain = _system_message(model_id)
        assert isinstance(plain, SystemMessage), model_id
        assert not _has_cache_marker(plain), model_id


# --- token counting -------------------------------------------------------


class _ExplodingClient:
    def count_tokens(self, **kwargs: Any) -> Any:
        raise AssertionError("CountTokens must not be called")


def test_token_counter_skips_api_when_unsupported() -> None:
    counter = BedrockTokenCounter(
        model_id=LanguageModelId.GPT_V6_SOL.value,
        client=_ExplodingClient(),
        api_supported=False,
    )
    assert counter.count_tokens("one two three four") >= 4
    truncated, count = counter.truncate_to_token_limit("word " * 200, 50)
    assert count <= 50 and len(truncated) < len("word " * 200)


def test_token_manager_disables_count_tokens_for_gpt(mocker) -> None:
    from unified_kg_rag.adapters import providers as providers_module
    from unified_kg_rag.adapters.retrieval import token_manager as tm_module

    counter_cls = mocker.patch.object(providers_module, "BedrockTokenCounter")
    config = Config()
    config.search.answer_generation_model_id = LanguageModelId.GPT_V6_1_SOL
    tm_module.TokenManager(config, boto_session=mocker.MagicMock())
    assert counter_cls.call_args.kwargs["api_supported"] is False
    # No CountTokens client is built for a model that cannot use it.
    assert counter_cls.call_args.kwargs["client"] is None


# --- defaults ---------------------------------------------------------------


def _language_model_defaults(model: BaseModel, path: str = "") -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name in type(model).model_fields:
        value = getattr(model, name)
        here = f"{path}.{name}" if path else name
        if name.endswith("_model_id") and here.split(".")[0] != "aws":
            if "embedding" not in name and "rerank" not in name:
                found[here] = value
        elif isinstance(value, BaseModel):
            found.update(_language_model_defaults(value, here))
        elif isinstance(value, Enum):
            continue
    return found


def test_defaults_use_sonnet_5_5_and_keep_haiku() -> None:
    defaults = _language_model_defaults(Config())
    assert defaults, "expected language-model defaults in Config"
    values = set(defaults.values())
    assert LanguageModelId.CLAUDE_V5_SONNET not in values
    assert values <= {
        LanguageModelId.CLAUDE_V5_5_SONNET,
        LanguageModelId.CLAUDE_V5_5_HAIKU,
    }
    for key in (
        "search.answer_generation_model_id",
        "evaluation.evaluation_model_id",
        "fixing.fixing_model_id",
    ):
        assert defaults[key] == LanguageModelId.CLAUDE_V5_5_SONNET, key
