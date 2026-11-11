# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``RAGInput.effort``: one query's answer reasoning effort, and the
deprecated ``enable_thinking`` it replaces."""

from __future__ import annotations

import itertools
import warnings
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from tests.unit.test_bedrock_capabilities import (  # noqa: F401 - the fixture
    _lang_factory,
    _resolve_to,
    recorded_chat,
)
from tests.unit.test_rag_chain_streaming import (
    _ANSWER,
    _input,
    _make_chain,
    _StreamingModelFactory,
)
from unified_kg_rag.application.retrieval.rag_chain import RAGInput
from unified_kg_rag.domain.models import Config, LanguageModelId, ModelRole
from unified_kg_rag.domain.models.config import EFFORT_THINKING_TOKENS
from unified_kg_rag.domain.models.evaluation import EvaluationResult
from unified_kg_rag.ports.model_factory import CEILING_ANSWER_TOKENS
from unified_kg_rag.shared.exceptions import ToolArgumentError

pytestmark = pytest.mark.unit


class _RecordingFactory(_StreamingModelFactory):
    """Records the keyword arguments of each model it builds."""

    def __init__(self) -> None:
        super().__init__()
        self.kwargs: list[dict[str, Any]] = []

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        self.get_model_calls += 1
        self.kwargs.append(kwargs)
        return GenericFakeChatModel(
            messages=itertools.repeat(AIMessage(content=_ANSWER))
        )


async def _answer_kwargs(config: Config, rag_input: RAGInput) -> dict[str, Any]:
    factory = _RecordingFactory()
    chain, _, _ = _make_chain(config, factory=factory)
    output = await chain.ainvoke(rag_input)
    assert output.answer == _ANSWER
    (kwargs,) = factory.kwargs  # the answer is the only model call
    return kwargs


async def test_an_effort_turns_the_answer_s_reasoning_on_at_it(config: Config) -> None:
    kwargs = await _answer_kwargs(config, _input(effort="low"))
    assert kwargs["enable_thinking"] is True and kwargs["effort"] == "low"


async def test_without_an_effort_the_answer_is_built_as_before(config: Config) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        kwargs = await _answer_kwargs(config, _input())
    assert kwargs["enable_thinking"] is False and "effort" not in kwargs


async def test_enable_thinking_is_a_deprecated_alias(config: Config) -> None:
    with pytest.warns(DeprecationWarning, match="RAGInput.effort"):
        rag_input = _input(enable_thinking=True)
    kwargs = await _answer_kwargs(config, rag_input)
    # Thinking on at the tier's configured effort: none is sent per call.
    assert kwargs["enable_thinking"] is True and "effort" not in kwargs
    # An effort wins over it.
    with pytest.warns(DeprecationWarning):
        both = _input(enable_thinking=True, effort="high")
    assert (await _answer_kwargs(config, both))["effort"] == "high"


def test_an_unknown_effort_is_refused_before_any_call() -> None:
    with pytest.raises(ToolArgumentError, match="effort"):
        RAGInput.checked(query="q", effort="extreme")


def test_bedrock_reasons_at_the_query_s_effort_with_room_for_it(
    monkeypatch, recorded_chat  # noqa: F811 - the fixture
) -> None:
    # Opus 4.6 thinks only when asked: the query's effort asks, and its
    # output ceiling leaves that level's reasoning room.
    opus = LanguageModelId.CLAUDE_V4_6_OPUS.value
    _resolve_to(monkeypatch, f"global.{opus}")
    config = Config.model_validate({"models": {"default": opus}})
    factory = _lang_factory(config)
    model_id = config.model_id(ModelRole.ANSWER_GENERATION)
    sent = []
    for effort in ("low", "max"):
        factory.get_model(
            model_id, enable_thinking=True, effort=effort, output_ceiling=500
        )
        _, kwargs = recorded_chat[-1]
        fields = kwargs["additional_model_request_fields"]
        assert fields["thinking"] == {"type": "adaptive"}
        assert fields["output_config"] == {"effort": effort}
        sent.append(kwargs["max_tokens"])
    # Each level's reasoning budget plus the answer's floor, under the
    # model's own output cap (ceiled_output_tokens).
    low, high = sent
    assert low == CEILING_ANSWER_TOKENS + EFFORT_THINKING_TOKENS["low"]
    assert high > low


def test_evaluation_results_record_no_thinking_flag() -> None:
    # It was always false: no query recorded it. A saved result that holds
    # it still loads.
    assert "enable_thinking" not in EvaluationResult.model_fields
    saved = EvaluationResult.model_validate(
        {
            "query_id": "q1",
            "question": "q",
            "generated_answer": "a",
            "ground_truth": "a",
            "enable_thinking": False,
        }
    )
    assert saved.query_id == "q1"
    described = EvaluationResult.model_fields["search_strategy"].description or ""
    assert "ppr" in described and "iterative" in described


class _ModelInfoFactory(_RecordingFactory):
    """A factory whose answer model has the capability record ``info``."""

    def __init__(self, **info: Any) -> None:
        super().__init__()
        self.info = info

    def get_model_info(self, model_id: Any) -> Any:
        from unified_kg_rag.ports.model_factory import LanguageModelInfo

        return LanguageModelInfo(
            context_window_size=32000, max_output_tokens=4096, **self.info
        )


@pytest.mark.parametrize(
    ("info", "effort", "match"),
    [
        # A local server's model, or a Bedrock model that does not reason:
        # an effort would be dropped without a word.
        ({"provider": "other", "supports_thinking": False}, "low", "does not reason"),
        # A level the model does not take: a caller's error, not the
        # model factory's LanguageModelError once retrieval ran.
        (
            {
                "provider": "openai",
                "supports_thinking": True,
                "supported_efforts": frozenset({"low", "medium", "high"}),
            },
            "max",
            "low, medium, high",
        ),
    ],
)
async def test_an_effort_the_answer_model_cannot_apply_is_refused_first(
    config: Config, info: dict[str, Any], effort: str, match: str
) -> None:
    from unified_kg_rag.shared.exceptions import ErrorClass, classify

    factory = _ModelInfoFactory(**info)
    chain, _, _ = _make_chain(config, factory=factory)
    with pytest.raises(ToolArgumentError, match=match) as raised:
        await chain.ainvoke(_input(effort=effort))
    assert "effort" in str(raised.value)
    assert classify(raised.value) is ErrorClass.INVALID_ARGUMENT
    assert factory.kwargs == []  # no model was built, none was called
    # Without one the same model answers.
    assert (await chain.ainvoke(_input())).answer == _ANSWER


def test_an_older_claude_thinks_on_the_effort_s_budget(
    monkeypatch, recorded_chat  # noqa: F811 - the fixture
) -> None:
    # Claude 4.5 takes a manual budget_tokens, no effort: the query's level
    # sets that budget, with max_tokens above it (Bedrock refuses a budget
    # that is not below max_tokens).
    sonnet = LanguageModelId.CLAUDE_V4_5_SONNET.value
    _resolve_to(monkeypatch, f"global.{sonnet}")
    config = Config.model_validate({"models": {"default": sonnet}})
    factory = _lang_factory(config)
    model_id = config.model_id(ModelRole.ANSWER_GENERATION)
    for effort in ("low", "max"):
        factory.get_model(model_id, enable_thinking=True, effort=effort)
        _, kwargs = recorded_chat[-1]
        thinking = kwargs["additional_model_request_fields"]["thinking"]
        budget = EFFORT_THINKING_TOKENS[effort]
        assert thinking == {"type": "enabled", "budget_tokens": budget}
        assert kwargs["max_tokens"] >= budget + CEILING_ANSWER_TOKENS
    # Without a per-call effort the budget is the factory's default, as before.
    factory.get_model(model_id, enable_thinking=True)
    _, kwargs = recorded_chat[-1]
    assert kwargs["additional_model_request_fields"]["thinking"]["budget_tokens"] == (
        factory.DEFAULT_THINKING_BUDGET_TOKENS
    )
