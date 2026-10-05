# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""System-prompt rendering with and without Bedrock prompt caching.

The prompt-cache path wraps the system prompt in a content block carrying a
``cache_control`` marker. That block must still be a template: system-side
placeholders ({entity_types}, {target_language}, ...) have to be substituted and
escaped braces rendered, exactly as in the non-cache path.
"""

from __future__ import annotations

import string
from typing import Any

import pytest
from langchain_core.messages import BaseMessage

import unified_kg_rag.domain.prompts as prompts_pkg
from unified_kg_rag.adapters.aws.chain_factory import _build_chat_prompt
from unified_kg_rag.domain.prompts import BasePrompt

pytestmark = pytest.mark.unit

_CACHE_CONTROL = {"type": "ephemeral"}


def _all_prompt_classes() -> list[type[BasePrompt]]:
    classes = [getattr(prompts_pkg, name) for name in prompts_pkg.__all__]
    return [
        cls
        for cls in classes
        if isinstance(cls, type)
        and issubclass(cls, BasePrompt)
        and cls is not BasePrompt
    ]


def _system_placeholders(template: str) -> set[str]:
    return {
        field
        for _, field, _, _ in string.Formatter().parse(template)
        if field  # None for literal-only segments, "" never used by our prompts
    }


_PROMPTS_WITH_SYSTEM_PLACEHOLDERS = [
    cls
    for cls in _all_prompt_classes()
    if _system_placeholders(cls.resolve().system_prompt_template)
]


def _render_system(cls: type[BasePrompt], enable_prompt_cache: bool) -> BaseMessage:
    prompt = _build_chat_prompt(cls.resolve(), enable_prompt_cache)
    values: dict[str, Any] = {
        name: f"<value-of-{name}>" for name in prompt.input_variables
    }
    return prompt.format_messages(**values)[0]


def _system_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    assert len(content) == 1
    block = content[0]
    assert isinstance(block, dict)
    return str(block["text"])


def test_enumeration_covers_system_placeholder_prompts() -> None:
    # Guards the parametrization below against silently testing nothing.
    names = {cls.__name__ for cls in _PROMPTS_WITH_SYSTEM_PLACEHOLDERS}
    assert {"GraphExtractionPrompt", "GlobalMapPrompt"} <= names


@pytest.mark.parametrize(
    "prompt_cls", _PROMPTS_WITH_SYSTEM_PLACEHOLDERS, ids=lambda c: c.__name__
)
def test_cache_path_substitutes_system_placeholders(
    prompt_cls: type[BasePrompt],
) -> None:
    placeholders = _system_placeholders(prompt_cls.resolve().system_prompt_template)

    cached = _render_system(prompt_cls, enable_prompt_cache=True)
    cached_text = _system_text(cached)

    for name in placeholders:
        assert f"{{{name}}}" not in cached_text
        assert f"<value-of-{name}>" in cached_text

    assert isinstance(cached.content, list)
    block = cached.content[0]
    assert isinstance(block, dict)
    assert block["cache_control"] == _CACHE_CONTROL


@pytest.mark.parametrize("prompt_cls", _all_prompt_classes(), ids=lambda c: c.__name__)
def test_cache_and_non_cache_render_identical_system_text(
    prompt_cls: type[BasePrompt],
) -> None:
    resolved = prompt_cls.resolve()
    cached_prompt = _build_chat_prompt(resolved, enable_prompt_cache=True)
    plain_prompt = _build_chat_prompt(resolved, enable_prompt_cache=False)
    assert sorted(cached_prompt.input_variables) == sorted(plain_prompt.input_variables)

    cached_text = _system_text(_render_system(prompt_cls, enable_prompt_cache=True))
    plain_text = _system_text(_render_system(prompt_cls, enable_prompt_cache=False))
    assert cached_text == plain_text


def test_cache_path_renders_escaped_braces_as_single_braces() -> None:
    from unified_kg_rag.domain.prompts.base import ResolvedPrompt

    resolved = ResolvedPrompt(
        system_prompt_template='Use {language}. Reply as {{"key": "value"}}.',
        human_prompt_template="{question}",
        input_variables=["language", "question"],
    )
    prompt = _build_chat_prompt(resolved, enable_prompt_cache=True)
    system = prompt.format_messages(language="English", question="q")[0]

    assert _system_text(system) == 'Use English. Reply as {"key": "value"}.'
    assert isinstance(system.content, list)
    block = system.content[0]
    assert isinstance(block, dict)
    assert block["cache_control"] == _CACHE_CONTROL
