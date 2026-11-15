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


@pytest.mark.parametrize(
    "model",
    [
        LanguageModelId.CLAUDE_V4_6_SONNET,  # adaptive thinking, opt-in
        LanguageModelId.CLAUDE_V5_5_SONNET,  # adaptive, always on
        LanguageModelId.GPT_V5_5,  # reasoning.effort, always on
        LanguageModelId.CLAUDE_V4_5_SONNET,  # a manual budget
    ],
)
def test_every_reasoning_family_leaves_room_for_the_query_s_effort(
    monkeypatch, recorded_chat, model  # noqa: F811 - the fixture
) -> None:
    # Regression (W2 review E3): only older Claude raised max_tokens for a
    # query's effort; at max (32,768 reasoning tokens) an adaptive Claude or
    # GPT answer kept the 16,384 default and could be cut off after the
    # retrieval ran. One rule: a request's reasoning plus the answer's
    # floor, under the model's own maximum.
    _resolve_to(monkeypatch, f"global.{model.value}")
    config = Config.model_validate({"models": {"default": model.value}})
    factory = _lang_factory(config)
    model_id = config.model_id(ModelRole.ANSWER_GENERATION)
    factory.get_model(model_id, enable_thinking=True, effort="max")
    _, kwargs = recorded_chat[-1]
    assert kwargs["max_tokens"] >= EFFORT_THINKING_TOKENS["max"] + CEILING_ANSWER_TOKENS
    # At the tier's effort (high) the default cap stands, as before.
    factory.get_model(model_id, enable_thinking=True)
    _, kwargs = recorded_chat[-1]
    assert kwargs["max_tokens"] == config.models.default_max_output_tokens


def test_the_deprecation_is_shown_under_the_default_warning_filters() -> None:
    # Regression (W2 review E4): stacklevel=2 inside the pydantic validator
    # named a pydantic frame, which Python's default filters hide: a caller
    # setting enable_thinking saw nothing. The warning names the caller's
    # line, so the default filters show it (as from __main__).
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    code = (
        "from unified_kg_rag import RAGInput\n"
        "RAGInput(query='q', enable_thinking=True)\n"
        "RAGInput.model_validate({'query': 'q', 'enable_thinking': True})\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    shown = [line for line in done.stderr.splitlines() if "DeprecationWarning" in line]
    assert len(shown) == 2 and all(
        line.startswith("<string>:") for line in shown
    ), done.stderr


async def test_a_search_takes_an_effort_it_never_spends(config: Config) -> None:
    # W2 review E6: a search makes no answer, so an effort the answer model
    # cannot apply is no reason to refuse it.
    factory = _ModelInfoFactory(provider="other", supports_thinking=False)
    chain, _, _ = _make_chain(config, factory=factory)
    found = await chain.asearch(_input(effort="low"))
    assert found.search_results.results


def test_the_deprecation_is_shown_through_a_symlinked_install(tmp_path) -> None:
    # W2 re-check low 1: the package and pydantic paths were compared
    # resolved against the frames' unresolved file names, so through a
    # symlinked venv or checkout no frame matched and the warning named
    # a pydantic frame again, hidden by the default filters.
    import os
    import subprocess
    import sys
    from pathlib import Path

    import pydantic

    import unified_kg_rag

    linked = tmp_path / "site"
    linked.mkdir()
    for module in (unified_kg_rag, pydantic):
        package = Path(module.__file__).resolve().parent
        (linked / package.name).symlink_to(package, target_is_directory=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    env["PYTHONPATH"] = str(linked)
    code = (
        "import pydantic, unified_kg_rag\n"
        f"assert unified_kg_rag.__file__.startswith({str(linked)!r})\n"
        "from unified_kg_rag import RAGInput\n"
        "RAGInput(query='q', enable_thinking=True)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    shown = [line for line in done.stderr.splitlines() if "DeprecationWarning" in line]
    assert len(shown) == 1 and shown[0].startswith("<string>:"), done.stderr
