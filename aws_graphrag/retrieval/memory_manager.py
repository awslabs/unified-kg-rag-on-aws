import asyncio
import threading
from datetime import datetime, timedelta
from typing import Any

import boto3
from langchain.memory.chat_memory import BaseChatMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from pydantic import Field

from aws_graphrag.aws import BedrockLanguageModelFactory
from aws_graphrag.core import get_config, get_logger
from aws_graphrag.models import Config, ConversationContext, MessageRole
from aws_graphrag.prompts import EntityExtractionPrompt
from aws_graphrag.utils import setup_chain

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

    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]
        self._update_context(message)
        self.updated_at = datetime.now()

    def add_messages(self, messages: list[BaseMessage]) -> None:
        for message in messages:
            self.add_message(message)

    def clear(self) -> None:
        self._messages.clear()
        self._context = ConversationContext()
        self.updated_at = datetime.now()

    def is_expired(self) -> bool:
        return datetime.now() > self.updated_at + self.ttl

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
            entities = result.get("entities", []) if isinstance(result, dict) else []

            if not entities:
                return

            entity_names = [
                entity.get("name", "").strip()
                for entity in entities
                if isinstance(entity, dict) and entity.get("name", "").strip()
            ]

            if entity_names:
                current_entities = set(self._context.mentioned_entities)
                current_entities.update(entity_names)
                self._context.mentioned_entities = sorted(current_entities)
                self._context.focused_entities = entity_names[: self.n_entities]

        except Exception as e:
            logger.warning(
                f"Entity extraction failed for conversation '{self.conversation_id}': {e}"
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
        return (
            self.chat_memory.messages
            if self.return_messages
            else get_buffer_string(
                self.chat_memory.messages,
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
        self._last_cleanup = datetime.now()

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

    async def add_message(self, conv_id: str, role: MessageRole, content: str):
        memory = await self.get_or_create_memory(conv_id)
        message_map = {
            MessageRole.USER: HumanMessage,
            MessageRole.ASSISTANT: AIMessage,
            MessageRole.SYSTEM: SystemMessage,
        }
        message = message_map.get(role, SystemMessage)(content=content)

        async with self._lock:
            memory.add_message(message)

    async def clear_conversation(self, conv_id: str) -> bool:
        async with self._lock:
            if conv_id in self._memories:
                del self._memories[conv_id]
                return True
            return False

    async def _cleanup_expired_unsafe(self) -> int:
        to_remove = [cid for cid, mem in self._memories.items() if mem.is_expired()]
        for cid in to_remove:
            del self._memories[cid]
        return len(to_remove)

    async def _cleanup_oldest_unsafe(self, count: int):
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
                f"Removed {len(to_remove)} oldest conversations to maintain capacity"
            )

    async def periodic_cleanup_task(self):
        while True:
            await asyncio.sleep(self.config.memory.cleanup_interval_hours * 3600)
            try:
                async with self._lock:
                    num_removed = await self._cleanup_expired_unsafe()
                    self._last_cleanup = datetime.now()
                    if num_removed > 0:
                        logger.info(f"Cleaned up {num_removed} expired conversations")
            except Exception as e:
                logger.error(f"Memory cleanup failed: {e}")

    async def get_stats(self) -> dict[str, Any]:
        async with self._lock:
            num_convs = len(self._memories)
            total_msgs = sum(len(mem.messages) for mem in self._memories.values())
            return {
                "total_conversations": num_convs,
                "total_messages": total_msgs,
                "avg_messages_per_conv": total_msgs / num_convs if num_convs else 0,
                "last_cleanup": self._last_cleanup.isoformat(),
            }


_memory_manager: MemoryManager | None = None
_manager_lock = threading.Lock()


def get_memory_manager() -> MemoryManager:
    global _memory_manager

    if _memory_manager is None:
        with _manager_lock:
            if _memory_manager is None:
                _memory_manager = MemoryManager(config=get_config())
    return _memory_manager


def start_memory_cleanup_task() -> asyncio.Task | None:
    manager = get_memory_manager()
    if manager.config.memory.auto_cleanup:
        logger.info("Starting periodic memory cleanup task")
        return asyncio.create_task(manager.periodic_cleanup_task())
    return None
