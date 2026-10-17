# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the conversation-memory adapter: message buffering,
trimming, the history text the follow-up rewrite reads, and the async
``MemoryManager`` capacity/eviction logic.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from unified_kg_rag.adapters.retrieval import memory_manager as mm
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.unit


@pytest.fixture
def patched_history():
    """A factory for histories of one conversation."""

    def make(**kwargs):
        return mm.GraphRAGChatMessageHistory(conversation_id="conv-1", **kwargs)

    return make


def _key(conversation_id: str) -> mm.ConversationKey:
    return mm.ConversationKey("default", None, conversation_id)


# --- GraphRAGChatMessageHistory ------------------------------------------


def test_add_message_buffers_and_updates_timestamp(patched_history) -> None:
    history = patched_history()
    before = history.updated_at
    history.add_message(HumanMessage(content="hi"))
    assert history.messages == [HumanMessage(content="hi")]
    assert history.updated_at >= before


def test_add_message_trims_to_max(patched_history) -> None:
    history = patched_history(max_messages=3)
    for i in range(5):
        history.add_message(AIMessage(content=str(i)))
    # Only the last 3 survive.
    assert [m.content for m in history.messages] == ["2", "3", "4"]


def test_clear_resets_messages(patched_history) -> None:
    history = patched_history()
    history.add_message(HumanMessage(content="Alice"))
    history.clear()
    assert history.messages == []


def test_buffer_string_formats_the_conversation(patched_history) -> None:
    history = patched_history()
    history.add_message(HumanMessage(content="hi"))
    history.add_message(AIMessage(content="hello"))
    assert history.get_buffer_string() == "Human: hi\nAI: hello"


def test_buffer_string_cuts_long_answers_only(patched_history) -> None:
    history = patched_history()
    history.add_message(HumanMessage(content="q" * 50))
    history.add_message(AIMessage(content="a" * 50))
    assert history.get_buffer_string(max_answer_chars=10) == (
        f"Human: {'q' * 50}\nAI: {'a' * 10}..."
    )
    assert history.messages[1].content == "a" * 50  # stored answer untouched


# --- MemoryManager (async) -----------------------------------------------


def test_get_or_create_memory_caches_instance(patched_history) -> None:
    import asyncio

    manager = mm.MemoryManager(config=Config())
    h1 = asyncio.run(manager.get_or_create_memory(_key("c1")))
    h2 = asyncio.run(manager.get_or_create_memory(_key("c1")))
    assert h1 is h2


def test_get_or_create_memory_evicts_oldest_at_capacity(patched_history) -> None:
    import asyncio

    cfg = Config()
    cfg.memory.max_conversations = 2

    manager = mm.MemoryManager(config=cfg)

    async def scenario():
        a = await manager.get_or_create_memory(_key("a"))
        # Make "a" the oldest by back-dating it.
        a.updated_at = datetime.now() - timedelta(hours=1)
        await manager.get_or_create_memory(_key("b"))
        # Third insertion is at capacity -> evicts oldest ("a").
        await manager.get_or_create_memory(_key("c"))
        return {key.conversation_id for key in manager._memories}

    keys = asyncio.run(scenario())
    assert keys == {"b", "c"}


async def test_get_or_create_memory_expires_idle_conversations(
    patched_history,
) -> None:
    # memory.max_conversation_age_hours: a conversation idle for longer starts
    # over, and other expired conversations are dropped rather than kept until
    # the capacity limit evicts them.
    cfg = Config()
    cfg.memory.max_conversation_age_hours = 2
    manager = mm.MemoryManager(config=cfg)
    stale = await manager.get_or_create_memory(_key("stale"))
    stale.add_message(HumanMessage(content="old question"))
    idle = await manager.get_or_create_memory(_key("idle"))
    fresh = await manager.get_or_create_memory(_key("fresh"))
    expired_at = datetime.now() - timedelta(hours=3)
    stale.updated_at = idle.updated_at = expired_at

    again = await manager.get_or_create_memory(_key("stale"))

    assert again is not stale and again.messages == []
    assert {key.conversation_id for key in manager._memories} == {"stale", "fresh"}
    assert manager._memories[_key("fresh")] is fresh


def test_cleanup_oldest_no_op_for_nonpositive(patched_history) -> None:
    import asyncio

    manager = mm.MemoryManager(config=Config())

    async def scenario():
        await manager.get_or_create_memory(_key("a"))
        manager._cleanup_oldest_unsafe(0)
        return [key.conversation_id for key in manager._memories]

    assert asyncio.run(scenario()) == ["a"]


async def test_load_history_returns_the_bounded_history(patched_history) -> None:
    cfg = Config()
    cfg.memory.max_messages_per_conversation = 2
    manager = mm.MemoryManager(config=cfg)
    await manager.add_turn(_key("c1"), "first", "old answer")
    long_answer = "x" * (mm.HISTORY_ANSWER_CHARS + 50)
    await manager.add_turn(_key("c1"), "about Vendor", long_answer)

    history = await manager.load_history(_key("c1"))

    # The last turn only, its answer cut for the rewrite.
    assert history == (f"Human: about Vendor\nAI: {'x' * mm.HISTORY_ANSWER_CHARS}...")


async def test_conversations_are_keyed_by_namespace_user_and_id(
    patched_history,
) -> None:
    # The same conversation id in another tenant (index suffix) or for another
    # user is another conversation.
    manager = mm.MemoryManager(config=Config())
    await manager.add_turn(mm.ConversationKey("tenant-a", "u1", "c1"), "q", "a")

    assert await manager.load_history(mm.ConversationKey("tenant-b", "u1", "c1")) == ""
    assert await manager.load_history(mm.ConversationKey("tenant-a", "u2", "c1")) == ""
    assert await manager.load_history(mm.ConversationKey("tenant-a", None, "c1")) == ""
    assert (
        await manager.load_history(mm.ConversationKey("tenant-a", "u1", "c1"))
        == "Human: q\nAI: a"
    )


def test_concurrent_turns_keep_each_answer_after_its_question(patched_history) -> None:
    import asyncio

    manager = mm.MemoryManager(config=Config())

    async def scenario():
        await asyncio.gather(
            *(manager.add_turn(_key("c1"), f"q{i}", f"a{i}") for i in range(5))
        )
        history = await manager.get_or_create_memory(_key("c1"))
        return [m.content for m in history.messages]

    contents = asyncio.run(scenario())
    assert len(contents) == 10
    for question, answer in zip(contents[::2], contents[1::2], strict=True):
        assert question.startswith("q") and answer == "a" + question[1:]


def test_turns_from_threads_on_separate_loops_stay_atomic(
    patched_history, mocker
) -> None:
    # The process-wide manager serves chains on different threads/loops (the
    # sync API's loop thread and the caller's loop). Its lock must exclude
    # across threads and must not bind to one event loop.
    import asyncio
    import threading
    import time

    original = mm.GraphRAGChatMessageHistory.add_message

    def slow_append(self, message):
        time.sleep(0.002)  # widen the critical section so threads contend
        original(self, message)

    mocker.patch.object(mm.GraphRAGChatMessageHistory, "add_message", slow_append)
    manager = mm.MemoryManager(config=Config())
    errors: list[BaseException] = []

    def worker(tag: str) -> None:
        async def turns() -> None:
            await asyncio.gather(
                *(
                    manager.add_turn(_key("c1"), f"q{tag}{i}", f"a{tag}{i}")
                    for i in range(5)
                )
            )

        try:
            asyncio.run(turns())
        except BaseException as e:  # noqa: BLE001 - surfaced by the assert
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,), daemon=True) for t in "xy"]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    contents = [m.content for m in manager._memories[_key("c1")].messages]
    assert len(contents) == 20
    for question, answer in zip(contents[::2], contents[1::2], strict=True):
        assert question.startswith("q") and answer == "a" + question[1:]
