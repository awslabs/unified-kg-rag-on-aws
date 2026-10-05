# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default output cap on LLM requests (AWS-free).

Bedrock reserves input + max_tokens against the tokens-per-minute quota when a
request starts, so a request must not ask for the model maximum by default.
The cap is ``aws.bedrock.default_max_output_tokens``, raised to the prompt's
``min_output_tokens`` floor and clamped to the model maximum.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.output_parsers import StrOutputParser

from unified_kg_rag.adapters.aws.bedrock import (
    BedrockLanguageModelFactory,
    effective_max_output_tokens,
    get_language_model_info,
)
from unified_kg_rag.adapters.aws.chain_factory import (
    create_robust_xml_output_parser,
    setup_chain,
)
from unified_kg_rag.domain.models import Config, LanguageModelId, ModelPurpose
from unified_kg_rag.domain.prompts import (
    AnswerGenerationPrompt,
    GraphExtractionPrompt,
)
from unified_kg_rag.domain.prompts.data_processing import TextTranslationPrompt

pytestmark = pytest.mark.unit

SONNET = LanguageModelId.CLAUDE_V5_5_SONNET
HAIKU = LanguageModelId.CLAUDE_V4_5_HAIKU


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


def _max_tokens(cfg: dict[str, Any]) -> int:
    return int(cfg.get("max_tokens") or cfg["model_kwargs"]["max_tokens"])


def test_default_cap_replaces_the_model_maximum() -> None:
    factory = _factory()
    info = factory.get_model_info(SONNET)
    cfg = factory._build_model_config(info, f"us.{SONNET.value}", True)
    assert info.max_output_tokens == 128000
    assert _max_tokens(cfg) == 16384


def test_cap_applies_on_the_invoke_model_path() -> None:
    factory = _factory()
    info = factory.get_model_info(LanguageModelId.CLAUDE_V3_5_SONNET_V2)
    cfg = factory._build_model_config(info, "anthropic.claude-x", False)
    # Claude 3.5 Sonnet v2 tops out below the cap, so it gets its maximum.
    assert cfg["model_kwargs"]["max_tokens"] == 8192


def test_prompt_floor_raises_the_cap_up_to_the_model_maximum() -> None:
    factory = _factory()
    sonnet = factory.get_model_info(SONNET)
    haiku = factory.get_model_info(HAIKU)
    floor = GraphExtractionPrompt.min_output_tokens
    cfg = factory._build_model_config(
        sonnet, f"us.{SONNET.value}", True, min_output_tokens=floor
    )
    assert _max_tokens(cfg) == floor == 32768
    cfg = factory._build_model_config(
        haiku,
        f"us.{HAIKU.value}",
        True,
        min_output_tokens=TextTranslationPrompt.min_output_tokens,
    )
    assert _max_tokens(cfg) == haiku.max_output_tokens == 64000


def test_explicit_max_tokens_wins() -> None:
    factory = _factory()
    info = factory.get_model_info(SONNET)
    cfg = factory._build_model_config(
        info, f"us.{SONNET.value}", True, max_tokens=500, min_output_tokens=32768
    )
    assert _max_tokens(cfg) == 500


def test_null_cap_restores_the_model_maximum() -> None:
    config = Config(aws={"bedrock": {"default_max_output_tokens": None}})
    info = get_language_model_info(SONNET)
    assert effective_max_output_tokens(info, config.aws.bedrock) == 128000


def test_configured_cap_above_the_floor_wins() -> None:
    config = Config(aws={"bedrock": {"default_max_output_tokens": 50000}})
    info = get_language_model_info(SONNET)
    assert effective_max_output_tokens(info, config.aws.bedrock, 32768) == 50000


def test_short_output_prompts_take_the_default_cap() -> None:
    assert AnswerGenerationPrompt.min_output_tokens == 0


class _RecordingFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_model(self, model_id: str, **kwargs: Any) -> Any:
        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )

        self.calls.append(kwargs)
        return FakeListChatModel(responses=["ok"])

    def get_model_info(self, model_id: str) -> Any:
        return None


def test_setup_chain_forwards_the_prompt_floor() -> None:
    factory = _RecordingFactory()
    setup_chain(factory, SONNET, GraphExtractionPrompt, StrOutputParser())
    assert factory.calls[0]["min_output_tokens"] == 32768


def test_output_fixer_gets_the_repaired_prompts_floor() -> None:
    factory = _RecordingFactory()
    create_robust_xml_output_parser(
        factory=factory,
        enable_output_fixing=True,
        output_fixing_model_id=SONNET,
        model_purpose=ModelPurpose.INGESTION,
        min_output_tokens=GraphExtractionPrompt.min_output_tokens,
    )
    assert factory.calls[0]["min_output_tokens"] == 32768
