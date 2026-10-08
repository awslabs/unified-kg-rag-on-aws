# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic, model-free chat provider (``LLMFactoryPort``).

``ScriptedLLMFactory`` hands out a chat model that answers every prompt with
``respond(system_prompt, human_prompt)``, so a test can drive the ingestion
chains (``prompt | llm | parser``) with valid, fixed LLM output.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

Responder = Callable[[str, str], str]


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

    def get_model_info(self, model_id: Any) -> None:
        return None
