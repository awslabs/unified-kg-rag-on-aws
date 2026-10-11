# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import threading
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import boto3
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.runnables import Runnable

from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import Config, language_name
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
        # The n_entities most recently mentioned entities, newest first.
        self._recent_entities: list[str] = []
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
        event loop, outside the lock. See MemoryManager.add_turn.
        """
        self._messages.append(message)
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]
        self.updated_at = datetime.now()

    def update_context(self, message: BaseMessage) -> None:
        """Public wrapper for the blocking entity-extraction step."""
        self._update_context(message)

    def clear(self) -> None:
        self._messages.clear()
        self._recent_entities = []
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
                    "target_language": language_name(
                        self.config.processing.translation.target_language
                    ),
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
        """Record the entity names already extracted for a user message.

        The newest message's entities come first (in extraction order), ahead
        of earlier ones, so a follow-up is focused on what was just discussed.
        """
        names = [str(entity).strip() for entity in entities if str(entity).strip()]
        merged = dict.fromkeys(names + self._recent_entities)
        self._recent_entities = list(merged)[: self.n_entities]

    def get_relevant_entities(self) -> list[str]:
        """The most recently mentioned entities, newest first."""
        return list(self._recent_entities)

    def get_buffer_string(self) -> str:
        """The conversation so far as ``Human: ...`` / ``AI: ...`` lines."""
        return get_buffer_string(self._messages)


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

    async def load_context(self, conv_id: str) -> tuple[str, list[str]]:
        """The conversation's history text and its most recent entities."""
        memory = await self.get_or_create_memory(conv_id)
        with self._lock:
            return memory.get_buffer_string(), memory.get_relevant_entities()

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
        context is updated afterwards, outside the lock: entity extraction is
        a blocking LLM call (~1-3s), so it runs off the event loop and does
        not serialize concurrent queries sharing this manager.
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
