# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import threading
from datetime import datetime, timedelta

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.utils import get_buffer_string

from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import get_config, get_logger

logger = get_logger(__name__)

# Each earlier answer is cut to this many characters in the history the
# follow-up rewrite reads: the rewrite needs what a turn was about (the names
# an answer opens with), not every answer in full on every turn.
HISTORY_ANSWER_CHARS = 1000


class GraphRAGChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        conversation_id: str,
        max_messages: int = 20,
        ttl_hours: int = 24,
    ):
        self.conversation_id = conversation_id
        self.max_messages = max_messages
        self.ttl = timedelta(hours=ttl_hours)
        self._messages: list[BaseMessage] = []
        self.updated_at = datetime.now()

    def is_expired(self, now: datetime) -> bool:
        """Whether the conversation has been idle for longer than its TTL."""
        return now - self.updated_at > self.ttl

    def add_message(self, message: BaseMessage) -> None:
        """Append a message, keeping only the ``max_messages`` most recent."""
        self._messages.append(message)
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]
        self.updated_at = datetime.now()

    def clear(self) -> None:
        self._messages.clear()
        self.updated_at = datetime.now()

    # LangChain's BaseChatMessageHistory types `messages` as a writeable
    # attribute; we expose it read-only and mutate via _messages.
    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        return self._messages

    def get_buffer_string(self, max_answer_chars: int | None = None) -> str:
        """The conversation so far as ``Human: ...`` / ``AI: ...`` lines.

        ``max_answer_chars`` cuts each answer (not the user messages) to that
        many characters, marking the cut with an ellipsis.
        """
        messages = self._messages
        if max_answer_chars is not None:
            messages = [
                (
                    AIMessage(content=f"{m.content[:max_answer_chars]}...")
                    if isinstance(m, AIMessage)
                    and isinstance(m.content, str)
                    and len(m.content) > max_answer_chars
                    else m
                )
                for m in messages
            ]
        return get_buffer_string(messages)


class MemoryManager:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._memories: dict[str, GraphRAGChatMessageHistory] = {}
        # A threading lock, not asyncio.Lock: the process-wide manager is shared
        # by chains on different threads and event loops (an asyncio.Lock binds
        # to one loop and does not exclude across threads). Every section it
        # guards is a short, non-awaiting state mutation.
        self._lock = threading.Lock()

    async def get_or_create_memory(self, conv_id: str) -> GraphRAGChatMessageHistory:
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
                conversation_id=conv_id,
                max_messages=self.config.memory.max_messages_per_conversation,
                ttl_hours=self.config.memory.max_conversation_age_hours,
            )
            self._memories[conv_id] = memory
            return memory

    async def load_history(self, conv_id: str) -> str:
        """The conversation's history text, each answer cut to
        ``HISTORY_ANSWER_CHARS`` (the turn count is bounded by
        ``memory.max_messages_per_conversation``)."""
        memory = await self.get_or_create_memory(conv_id)
        with self._lock:
            return memory.get_buffer_string(max_answer_chars=HISTORY_ANSWER_CHARS)

    async def add_turn(
        self, conv_id: str, user_content: str, assistant_content: str
    ) -> None:
        """Append a user message and its answer as one unit.

        Two unlocked ``add_message`` calls would let a concurrent turn on the
        same conversation land between them (user, user, answer, answer), so
        both messages are appended under one lock hold.
        """
        memory = await self.get_or_create_memory(conv_id)
        with self._lock:
            memory.add_message(HumanMessage(content=user_content))
            memory.add_message(AIMessage(content=assistant_content))

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


def _memory_fingerprint(config: Config) -> str:
    """The config part conversation memory is built from."""
    return config.memory.model_dump_json()


def get_memory_manager(config: Config | None = None) -> MemoryManager:
    """The process-wide conversation memory, shared by every default chain.

    Created on first use from the first caller's ``config`` (``get_config()``
    when omitted), so conversations survive across chain instances, e.g. a
    chain built per request. A later caller whose ``memory`` limits differ
    keeps using the shared manager; the mismatch is logged once. Pass a chain
    its own ``MemoryManager`` to isolate it instead.
    """
    global _memory_manager, _mismatch_logged

    if _memory_manager is None:
        with _manager_lock:
            if _memory_manager is None:
                _memory_manager = MemoryManager(config=config or get_config())
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
