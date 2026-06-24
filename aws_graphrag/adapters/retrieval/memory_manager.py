# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import threading
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import boto3
from langchain.memory.chat_memory import BaseChatMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from pydantic import Field

from aws_graphrag.adapters.aws import BedrockLanguageModelFactory
from aws_graphrag.adapters.aws.chain_factory import setup_chain
from aws_graphrag.domain.models import Config, ConversationContext, MessageRole
from aws_graphrag.domain.prompts import EntityExtractionPrompt
from aws_graphrag.shared import get_config, get_logger

logger = get_logger(__name__)


class GraphRAGChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        config: Config,
        conversation_id: str,
        max_messages: int = 20,
        ttl_hours: int = 24,
        boto_session: boto3.Session | None = None,
        n_entities: int = 5,
    ):
        self.config = config
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name
        )
        self.conversation_id = conversation_id
        self.max_messages = max_messages
        self.ttl = timedelta(hours=ttl_hours)
        self.n_entities = n_entities
        self._messages: list[BaseMessage] = []
        self._context = ConversationContext()
        self.updated_at = datetime.now()

        factory = BedrockLanguageModelFactory(
            config=self.config, boto_session=self.boto_session
        )
        self.entity_extractor = setup_chain(
            model_id=self.config.search.entity_extraction_model_id,
            factory=factory,
            prompt_class=EntityExtractionPrompt,
            parser=CommaSeparatedListOutputParser(),
        )

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]
        self._update_context(message)
        self.updated_at = datetime.now()

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        for message in messages:
            self.add_message(message)

    def clear(self) -> None:
        self._messages.clear()
        self._context = ConversationContext()
        self.updated_at = datetime.now()

    # LangChain's BaseChatMessageHistory types `messages` as a writeable
    # attribute; we expose it read-only and mutate via _messages.
    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        return self._messages

    def _update_context(self, message: BaseMessage) -> None:
        if not isinstance(message, HumanMessage):
            return

        try:
            result = self.entity_extractor.invoke(
                {
                    "query": message.content,
                    "target_language": self.config.processing.translation.target_language,
                }
            )
            # CommaSeparatedListOutputParser yields a list[str]; the previous
            # result.get("entities") (dict access) always returned [], so
            # conversation entities were never populated. Accept the list shape
            # (and tolerate a dict {"entities": [...]} just in case).
            if isinstance(result, dict):
                raw_entities = result.get("entities", [])
            else:
                raw_entities = result or []

            entity_names = [
                str(entity).strip() for entity in raw_entities if str(entity).strip()
            ]

            if entity_names:
                current_entities = set(self._context.mentioned_entities)
                current_entities.update(entity_names)
                self._context.mentioned_entities = sorted(current_entities)
                self._context.focused_entities = entity_names[: self.n_entities]

        except Exception as e:
            logger.warning(
                "Entity extraction failed for conversation '%s': %s",
                self.conversation_id,
                e,
            )

    def get_context_summary(self) -> str:
        parts = []

        if self._context.mentioned_entities:
            entities_str = ", ".join(
                self._context.mentioned_entities[: self.n_entities]
            )
            parts.append(f"Mentioned: {entities_str}")

        if self._context.focused_entities:
            parts.append(f"Focused on: {', '.join(self._context.focused_entities)}")

        return " | ".join(parts) or "New Conversation"

    def get_relevant_entities(self) -> list[str]:
        return self._context.mentioned_entities[: self.n_entities]


class GraphRAGConversationBufferMemory(BaseChatMemory):
    chat_memory: GraphRAGChatMessageHistory = Field(
        default_factory=lambda: GraphRAGChatMessageHistory(
            config=get_config(), conversation_id="default"
        ),
        description="The underlying chat message history that stores conversation messages and extracts entities",
    )
    return_messages: bool = Field(
        default=False,
        description="Whether to return messages as BaseMessage objects or as a formatted string",
    )
    include_entity_context: bool = Field(
        default=True,
        description="Whether to include entity context (relevant entities and conversation summary) in memory variables",
    )
    human_prefix: str = Field(
        default="Human",
        description="Prefix to use for human messages when formatting as string",
    )
    ai_prefix: str = Field(
        default="AI",
        description="Prefix to use for AI messages when formatting as string",
    )
    memory_key: str = Field(
        default="history",
        description="The key name to use for the conversation history in memory variables",
    )

    @property
    def buffer(self) -> list[BaseMessage] | str:
        messages = self.chat_memory.messages
        return (
            messages
            if self.return_messages
            else get_buffer_string(
                messages,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        )

    @property
    def memory_variables(self) -> list[str]:
        base = [self.memory_key]
        if self.include_entity_context:
            base.extend(["relevant_entities", "conversation_context"])
        return base

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        mem_vars: dict[str, Any] = {self.memory_key: self.buffer}

        if self.include_entity_context:
            mem_vars["relevant_entities"] = self.chat_memory.get_relevant_entities()
            mem_vars["conversation_context"] = self.chat_memory.get_context_summary()

        return mem_vars

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        input_str = self._get_input(inputs)
        output_str = self._get_output(outputs)
        self.chat_memory.add_messages(
            [HumanMessage(content=input_str), AIMessage(content=output_str)]
        )

    def clear(self) -> None:
        self.chat_memory.clear()

    @staticmethod
    def _get_input(inputs: dict[str, Any]) -> str:
        return str(inputs.get("input") or inputs.get("query") or inputs)

    @staticmethod
    def _get_output(outputs: dict[str, Any]) -> str:
        return str(
            outputs.get("output")
            or outputs.get("answer")
            or outputs.get("text")
            or outputs
        )


class MemoryManager:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._memories: dict[str, GraphRAGChatMessageHistory] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_memory(self, conv_id: str) -> GraphRAGChatMessageHistory:
        async with self._lock:
            if memory := self._memories.get(conv_id):
                return memory

            if len(self._memories) >= self.config.memory.max_conversations:
                await self._cleanup_oldest_unsafe(1)

            memory = GraphRAGChatMessageHistory(
                config=self.config,
                conversation_id=conv_id,
                max_messages=self.config.memory.max_messages_per_conversation,
                ttl_hours=self.config.memory.max_conversation_age_hours,
            )
            self._memories[conv_id] = memory
            return memory

    async def get_langchain_memory(
        self, conv_id: str, **kwargs: Any
    ) -> GraphRAGConversationBufferMemory:
        history = await self.get_or_create_memory(conv_id)
        return GraphRAGConversationBufferMemory(chat_memory=history, **kwargs)

    async def add_message(self, conv_id: str, role: MessageRole, content: str) -> None:
        memory = await self.get_or_create_memory(conv_id)
        message_map = {
            MessageRole.USER: HumanMessage,
            MessageRole.ASSISTANT: AIMessage,
            MessageRole.SYSTEM: SystemMessage,
        }
        message = message_map.get(role, SystemMessage)(content=content)

        async with self._lock:
            memory.add_message(message)

    async def _cleanup_oldest_unsafe(self, count: int) -> None:
        if count <= 0:
            return

        sorted_convs = sorted(
            self._memories.items(), key=lambda item: item[1].updated_at
        )
        to_remove = [item[0] for item in sorted_convs[:count]]

        for cid in to_remove:
            del self._memories[cid]

        if to_remove:
            logger.info(
                "Removed %s oldest conversations to maintain capacity", len(to_remove)
            )


_memory_manager: MemoryManager | None = None
_manager_lock = threading.Lock()


def get_memory_manager() -> MemoryManager:
    global _memory_manager

    if _memory_manager is None:
        with _manager_lock:
            if _memory_manager is None:
                _memory_manager = MemoryManager(config=get_config())
    return _memory_manager
