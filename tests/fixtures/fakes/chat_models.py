# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic, model-free chat provider (``LLMFactoryPort``).

``ScriptedLLMFactory`` hands out a chat model that answers every prompt with
``respond(system_prompt, human_prompt)``, so a test can drive the ingestion
chains (``prompt | llm | parser``) with valid, fixed LLM output.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from unified_kg_rag.adapters.aws.bedrock_models import get_language_model_info
from unified_kg_rag.ports.model_factory import LanguageModelInfo

Responder = Callable[[str, str], str]


def fake_model_info(model_id: Any) -> LanguageModelInfo:
    """A fake LLM factory's capability record: the catalog's for ``model_id``."""
    return get_language_model_info(str(model_id))


class ScriptedChatModel(BaseChatModel):
    respond: Responder

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = "".join(str(m.content) for m in messages if m.type == "system")
        human = "".join(str(m.content) for m in messages if m.type == "human")
        reply = AIMessage(content=self.respond(system, human))
        return ChatResult(generations=[ChatGeneration(message=reply)])


class ScriptedLLMFactory:
    def __init__(self, respond: Responder) -> None:
        self.respond = respond

    def get_model(self, model_id: Any, **kwargs: Any) -> ScriptedChatModel:
        return ScriptedChatModel(respond=self.respond)

    def get_model_info(self, model_id: Any) -> LanguageModelInfo:
        return fake_model_info(model_id)


# The opening words of the prompts CountingResponder tells apart.
SUMMARY_TREE_PROMPT = "You write the summaries of a summary tree"
_PROMPT_KINDS = (
    ("knowledge graph extraction expert", "extraction"),
    ("community analysis", "report"),
)


class CountingResponder:
    """``respond`` with its calls counted by prompt kind: ``extraction``,
    ``report``, ``summary`` (a summary tree's, answered by ``summarize(human)``
    when given) and ``other``."""

    def __init__(
        self, respond: Responder, summarize: Callable[[str], str] | None = None
    ) -> None:
        self.respond = respond
        self.summarize = summarize
        self.kinds: Counter[str] = Counter()
        self.tree_prompts: list[str] = []

    def __call__(self, system: str, human: str) -> str:
        if SUMMARY_TREE_PROMPT in system:
            self.kinds["summary"] += 1
            self.tree_prompts.append(system + "\n" + human)
            if self.summarize is not None:
                return self.summarize(human)
            return self.respond(system, human)
        kind = next((k for marker, k in _PROMPT_KINDS if marker in system), "other")
        self.kinds[kind] += 1
        return self.respond(system, human)

    def since(self, before: Counter[str]) -> Counter[str]:
        """The calls made after ``before`` (a copy of ``kinds``)."""
        return self.kinds - before
