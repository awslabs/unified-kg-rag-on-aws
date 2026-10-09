# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import threading
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import boto3
from langchain_classic.memory.chat_memory import BaseChatMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.runnables import Runnable
from pydantic import Field

from unified_kg_rag.adapters.aws.chain_factory import setup_chain
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import Config, ConversationContext, MessageRole
from unified_kg_rag.domain.prompts import EntityExtractionPrompt
from unified_kg_rag.shared import get_config, get_logger

logger = get_logger(__name__)


def build_entity_extractor(config: Config, providers: Providers) -> Runnable:
    """The conversation entity-extraction chain (shared by conversations)."""
    return setup_chain(
        model_id=config.search.entity_extraction_model_id,
        factory=providers.llm_factory,
        prompt_class=EntityExtractionPrompt,
        parser=CommaSeparatedListOutputParser(),
        custom_prompts=config.custom_prompts,
    )


class GraphRAGChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        config: Config,
        conversation_id: str,
        max_messages: int = 20,
        ttl_hours: int = 24,
        boto_session: boto3.Session | None = None,
        n_entities: int = 5,
        *,
        providers: Providers | None = None,
        entity_extractor: Runnable | None = None,
    ):
        self.config = config
        providers = Providers.resolve(config, providers, boto_session)
        self.boto_session = providers.boto_session
        self.conversation_id = conversation_id
        self.max_messages = max_messages
        self.ttl = timedelta(hours=ttl_hours)
        self.n_entities = n_entities
        self._messages: list[BaseMessage] = []
        self._context = ConversationContext()
        self.updated_at = datetime.now()

        # MemoryManager passes one shared extractor to every conversation.
        self.entity_extractor = entity_extractor or build_entity_extractor(
            self.config, providers
        )

    def is_expired(self, now: datetime) -> bool:
        """Whether the conversation has been idle for longer than its TTL."""
        return now - self.updated_at > self.ttl

    def add_message(self, message: BaseMessage) -> None:
        self.append_message(message)
        self._update_context(message)
        self.updated_at = datetime.now()

    def append_message(self, message: BaseMessage) -> None:
        """Append + trim only, without the blocking entity-extraction LLM call.

        Split out so an async caller can hold its lock for just the cheap state
        mutation and run _update_context (a blocking Bedrock invoke) off the
        event loop, outside the lock. See MemoryManager.add_message.
        """
        self._messages.append(message)
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]
        self.updated_at = datetime.now()

    def update_context(self, message: BaseMessage) -> None:
        """Public wrapper for the blocking entity-extraction step."""
        self._update_context(message)

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
            self.record_entities(raw_entities)

        except Exception as e:
            logger.warning(
                "Entity extraction failed for conversation '%s': %s",
                self.conversation_id,
                e,
            )

    def record_entities(self, entities: Sequence[Any]) -> None:
        """Merge entity names already extracted for a user message."""
        entity_names = [
            str(entity).strip() for entity in entities if str(entity).strip()
        ]
        if entity_names:
            current_entities = set(self._context.mentioned_entities)
            current_entities.update(entity_names)
            self._context.mentioned_entities = sorted(current_entities)
            self._context.focused_entities = entity_names[: self.n_entities]

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
    def __init__(self, config: Config, *, providers: Providers | None = None) -> None:
        self.config = config
        # One provider bundle for every conversation: each history reuses the
        # same LLM factory instead of building a Bedrock client per conversation.
        self.providers = Providers.resolve(config, providers)
        self._entity_extractor: Runnable | None = None
        self._memories: dict[str, GraphRAGChatMessageHistory] = {}
        # A threading lock, not asyncio.Lock: the process-wide manager is shared
        # by chains on different threads and event loops (an asyncio.Lock binds
        # to one loop and does not exclude across threads). Every section it
        # guards is a short, non-awaiting state mutation.
        self._lock = threading.Lock()

    def _shared_entity_extractor(self) -> Runnable:
        if self._entity_extractor is None:
            self._entity_extractor = build_entity_extractor(self.config, self.providers)
        return self._entity_extractor

    async def get_or_create_memory(self, conv_id: str) -> GraphRAGChatMessageHistory:
        # Built once, outside the lock: a new conversation then costs no
        # model construction while other queries wait on the lock.
        extractor = self._shared_entity_extractor()
        with self._lock:
            # Conversations idle past memory.max_conversation_age_hours are
            # dropped (a returning one starts over). A sweep over at most
            # max_conversations entries per lookup.
            now = datetime.now()
            for cid in [c for c, m in self._memories.items() if m.is_expired(now)]:
                del self._memories[cid]
            if (memory := self._memories.get(conv_id)) is not None:
                return memory

            if len(self._memories) >= self.config.memory.max_conversations:
                self._cleanup_oldest_unsafe(1)

            memory = GraphRAGChatMessageHistory(
                config=self.config,
                conversation_id=conv_id,
                max_messages=self.config.memory.max_messages_per_conversation,
                ttl_hours=self.config.memory.max_conversation_age_hours,
                providers=self.providers,
                entity_extractor=extractor,
            )
            self._memories[conv_id] = memory
            return memory

    async def get_langchain_memory(
        self, conv_id: str, **kwargs: Any
    ) -> GraphRAGConversationBufferMemory:
        history = await self.get_or_create_memory(conv_id)
        return GraphRAGConversationBufferMemory(chat_memory=history, **kwargs)

    async def add_message(
        self,
        conv_id: str,
        role: MessageRole,
        content: str,
        *,
        entities: Sequence[str] | None = None,
    ) -> None:
        """Append a message; for a user message, update the entity context.

        ``entities`` are the names already extracted from this user message
        (the chain's query step extracts them with the same prompt); when given,
        they are recorded directly instead of paying a second LLM call.
        """
        memory = await self.get_or_create_memory(conv_id)
        message_map = {
            MessageRole.USER: HumanMessage,
            MessageRole.ASSISTANT: AIMessage,
            MessageRole.SYSTEM: SystemMessage,
        }
        message = message_map.get(role, SystemMessage)(content=content)

        # Hold the lock only for the cheap append/trim state mutation. The
        # entity-extraction step is a blocking Bedrock LLM invoke (~1-3s); doing
        # it under the lock (and on the event-loop thread) serialized every
        # concurrent query sharing this manager (e.g. under abatch). Run it off
        # the loop, outside the lock. It only mutates ConversationContext, which
        # is read via get_context_summary at the start of the next turn.
        with self._lock:
            memory.append_message(message)

        if not isinstance(message, HumanMessage):
            return
        if entities is not None:
            memory.record_entities(entities)
        else:
            await asyncio.to_thread(memory.update_context, message)

    async def add_turn(
        self,
        conv_id: str,
        user_content: str,
        assistant_content: str,
        *,
        entities: Sequence[str] | None = None,
    ) -> None:
        """Append a user message and its answer as one unit.

        Two ``add_message`` calls let a concurrent turn on the same
        conversation land between them (user, user, answer, answer). Both
        messages are appended under one lock hold; the user message's entity
        context is updated afterwards, as in ``add_message``.
        """
        memory = await self.get_or_create_memory(conv_id)
        user_message = HumanMessage(content=user_content)
        with self._lock:
            memory.append_message(user_message)
            memory.append_message(AIMessage(content=assistant_content))

        if entities is not None:
            memory.record_entities(entities)
        else:
            await asyncio.to_thread(memory.update_context, user_message)

    def _cleanup_oldest_unsafe(self, count: int) -> None:
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
_mismatch_logged = False


def _memory_fingerprint(config: Config) -> tuple[str, str, str]:
    """The config parts conversation memory is built from."""
    return (
        config.memory.model_dump_json(),
        str(config.search.entity_extraction_model_id),
        config.custom_prompts.model_dump_json(),
    )


def get_memory_manager(
    config: Config | None = None, providers: Providers | None = None
) -> MemoryManager:
    """The process-wide conversation memory, shared by every default chain.

    Created on first use from the first caller's ``config`` and ``providers``
    (``get_config()`` and a default bundle when omitted), so conversations
    survive across chain instances, e.g. a chain built per request. A later
    caller whose memory-relevant config (``memory`` limits, entity-extraction
    model, ``custom_prompts``) differs keeps using the shared manager; the
    mismatch is logged once. Pass a chain its own ``MemoryManager`` to isolate
    it instead.
    """
    global _memory_manager, _mismatch_logged

    if _memory_manager is None:
        with _manager_lock:
            if _memory_manager is None:
                _memory_manager = MemoryManager(
                    config=config or get_config(), providers=providers
                )
                return _memory_manager
    if (
        config is not None
        and not _mismatch_logged
        and _memory_fingerprint(config) != _memory_fingerprint(_memory_manager.config)
    ):
        _mismatch_logged = True
        logger.warning(
            "The shared conversation memory was created from a different memory "
            "config than this chain's; it keeps the first config. Pass "
            "GraphRAGChain(memory_manager=MemoryManager(config)) to isolate a chain."
        )
    return _memory_manager
