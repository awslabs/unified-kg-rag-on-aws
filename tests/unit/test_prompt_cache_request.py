# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt-cache markers on the rendered Bedrock request (AWS-free).

Checks what langchain-aws actually sends, not just the LangChain message: the
Converse path (every inference profile) drops an Anthropic ``cache_control``
key, so it must carry a native ``cachePoint`` system block, while the
InvokeModel path keeps ``cache_control``. The chains are built by the real
factory with the real chat-model classes and a stub boto client.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_aws.chat_models.bedrock import _format_anthropic_messages
from langchain_aws.chat_models.bedrock_converse import _messages_to_bedrock
from langchain_core.output_parsers import StrOutputParser

from unified_kg_rag.adapters.aws.bedrock import (
    BedrockCrossRegionModelHelper,
    BedrockLanguageModelFactory,
)
from unified_kg_rag.adapters.aws.chain_factory import setup_chain
from unified_kg_rag.adapters.aws.token_counter import estimate_token_count
from unified_kg_rag.domain.models import Config, LanguageModelId, ModelPurpose
from unified_kg_rag.domain.prompts import BasePrompt
from unified_kg_rag.domain.prompts.data_processing import (
    DescriptionSummarizationPrompt,
)
from unified_kg_rag.domain.prompts.graph_extraction import GraphRefinementPrompt

pytestmark = pytest.mark.unit

_CACHE_POINT = {"cachePoint": {"type": "default"}}


class _FakeSession:
    profile_name = "default"

    def client(self, service_name: str, **kwargs: Any) -> Any:
        return object()

    def get_credentials(self) -> Any:
        return None


def _messages(
    mocker: Any, model_id: str, resolved_id: str, prompt_cls: type[BasePrompt]
) -> tuple[Any, list[Any]]:
    mocker.patch.object(
        BedrockCrossRegionModelHelper,
        "get_cross_region_model_id",
        return_value=resolved_id,
    )
    factory = BedrockLanguageModelFactory(
        Config(), boto_session=_FakeSession()  # type: ignore[arg-type]
    )
    # INGESTION: no query retry wrapper, so the chain is the bare sequence.
    chain = setup_chain(
        factory,
        model_id,
        prompt_cls,
        StrOutputParser(),
        model_purpose=ModelPurpose.INGESTION,
    )
    prompt, llm = chain.first, chain.middle[0]  # type: ignore[attr-defined]
    values = dict.fromkeys(prompt.input_variables, "synthetic")
    return llm, prompt.format_messages(**values)


def test_long_system_prompt_is_in_range_for_the_tests() -> None:
    long_tokens = estimate_token_count(GraphRefinementPrompt.system_prompt_template)
    short_tokens = estimate_token_count(
        DescriptionSummarizationPrompt.system_prompt_template
    )
    assert short_tokens < 512 < 1024 < long_tokens < 4096


def test_converse_request_carries_a_cache_point(mocker: Any) -> None:
    model = LanguageModelId.CLAUDE_V5_5_SONNET
    llm, messages = _messages(
        mocker, model, f"global.{model.value}", GraphRefinementPrompt
    )
    assert type(llm).__name__ == "ChatBedrockConverse"
    _, system = _messages_to_bedrock(messages)
    assert system[-1] == _CACHE_POINT
    assert len(system) == 2 and "text" in system[0]


def test_invoke_model_request_carries_cache_control(mocker: Any) -> None:
    model = LanguageModelId.CLAUDE_V3_7_SONNET
    llm, messages = _messages(mocker, model, model.value, GraphRefinementPrompt)
    assert type(llm).__name__ == "ChatBedrock"
    system, _ = _format_anthropic_messages(messages)
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_short_system_prompt_gets_no_marker(mocker: Any) -> None:
    model = LanguageModelId.CLAUDE_V5_5_SONNET
    _, messages = _messages(
        mocker, model, f"global.{model.value}", DescriptionSummarizationPrompt
    )
    _, system = _messages_to_bedrock(messages)
    assert _CACHE_POINT not in system


def test_model_minimum_gates_the_marker(mocker: Any) -> None:
    # Haiku 4.5 needs 4096 tokens per checkpoint; the long prompt is below it.
    model = LanguageModelId.CLAUDE_V4_5_HAIKU
    _, messages = _messages(
        mocker, model, f"global.{model.value}", GraphRefinementPrompt
    )
    _, system = _messages_to_bedrock(messages)
    assert _CACHE_POINT not in system


def test_gpt_request_has_no_cache_marker(mocker: Any) -> None:
    model = LanguageModelId.GPT_V6_SOL
    _, messages = _messages(
        mocker, model, f"global.{model.value}", GraphRefinementPrompt
    )
    _, system = _messages_to_bedrock(messages)
    assert system == [{"text": system[0]["text"]}]
