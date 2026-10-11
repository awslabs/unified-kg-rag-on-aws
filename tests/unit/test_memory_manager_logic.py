# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for the conversation-memory adapter.

``GraphRAGChatMessageHistory.__init__`` obtains a boto3 Session and a
``BedrockLanguageModelFactory`` (through its ``Providers`` bundle) and builds an
entity-extraction chain. All three are
patched out so the message buffering, entity-context tracking, trimming, and the
async ``MemoryManager`` capacity/eviction logic can be exercised without AWS or a
network.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from unified_kg_rag.adapters import providers as providers_module
from unified_kg_rag.adapters.retrieval import memory_manager as mm
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.unit


@pytest.fixture
def patched_history(mocker):
    """Patch the AWS-coupled bits of GraphRAGChatMessageHistory.

    Returns a factory that builds a history whose entity extractor is a stub
    invokable (default: returns a comma-separated entity list shape).
    """
    mocker.patch.object(mm.boto3, "Session", return_value=object())
    mocker.patch.object(
        providers_module, "BedrockLanguageModelFactory", return_value=object()
    )

    extractor = mocker.MagicMock()
    extractor.invoke.return_value = ["Alice", "Bob"]
    mocker.patch.object(mm, "setup_chain", return_value=extractor)

    def make(**kwargs):
        history = mm.GraphRAGChatMessageHistory(
            config=Config(), conversation_id="conv-1", **kwargs
        )
        return history

    make.extractor = extractor  # type: ignore[attr-defined]
    return make


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


def test_human_message_populates_entity_context(patched_history) -> None:
    history = patched_history()
    history.add_message(HumanMessage(content="Tell me about Alice and Bob"))
    assert history.get_relevant_entities() == ["Alice", "Bob"]


def test_relevant_entities_are_the_most_recent(patched_history) -> None:
    # Entities are kept newest first, so a follow-up focuses on what was just
    # discussed, not on whichever names sort first alphabetically.
    history = patched_history(n_entities=3)
    history.record_entities(["Alpha", "Beta"])
    history.record_entities(["Zulu", "Alpha"])
    assert history.get_relevant_entities() == ["Zulu", "Alpha", "Beta"]
    history.record_entities(["Yankee", "Xray"])
    assert history.get_relevant_entities() == ["Yankee", "Xray", "Zulu"]


def test_ai_message_does_not_trigger_extraction(patched_history) -> None:
    history = patched_history()
    history.add_message(AIMessage(content="some answer"))
    # No HumanMessage -> extractor never invoked, no entities.
    patched_history.extractor.invoke.assert_not_called()
    assert history.get_relevant_entities() == []


def test_entity_extraction_accepts_dict_shape(patched_history) -> None:
    patched_history.extractor.invoke.return_value = {"entities": ["Carol"]}
    history = patched_history()
    history.add_message(HumanMessage(content="who is Carol"))
    assert history.get_relevant_entities() == ["Carol"]


def test_entity_extraction_failure_is_swallowed(patched_history) -> None:
    patched_history.extractor.invoke.side_effect = RuntimeError("bedrock down")
    history = patched_history()
    # Must not raise; entities just stay empty.
    history.add_message(HumanMessage(content="anything"))
    assert history.get_relevant_entities() == []


def test_n_entities_limits_relevant(patched_history) -> None:
    patched_history.extractor.invoke.return_value = ["A", "B", "C", "D", "E", "F"]
    history = patched_history(n_entities=2)
    history.add_message(HumanMessage(content="many entities"))
    assert history.get_relevant_entities() == ["A", "B"]


def test_clear_resets_messages_and_context(patched_history) -> None:
    history = patched_history()
    history.add_message(HumanMessage(content="Alice"))
    history.clear()
    assert history.messages == []
    assert history.get_relevant_entities() == []


def test_buffer_string_formats_the_conversation(patched_history) -> None:
    history = patched_history()
    history.add_message(HumanMessage(content="hi"))
    history.add_message(AIMessage(content="hello"))
    assert history.get_buffer_string() == "Human: hi\nAI: hello"


# --- MemoryManager (async) -----------------------------------------------


def test_get_or_create_memory_caches_instance(patched_history) -> None:
    import asyncio

    manager = mm.MemoryManager(config=Config())
    h1 = asyncio.run(manager.get_or_create_memory("c1"))
    h2 = asyncio.run(manager.get_or_create_memory("c1"))
    assert h1 is h2


def test_get_or_create_memory_evicts_oldest_at_capacity(patched_history) -> None:
    import asyncio

    cfg = Config()
    cfg.memory.max_conversations = 2

    manager = mm.MemoryManager(config=cfg)

    async def scenario():
        a = await manager.get_or_create_memory("a")
        # Make "a" the oldest by back-dating it.
        a.updated_at = datetime.now() - timedelta(hours=1)
        await manager.get_or_create_memory("b")
        # Third insertion is at capacity -> evicts oldest ("a").
        await manager.get_or_create_memory("c")
        return set(manager._memories.keys())

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
    stale = await manager.get_or_create_memory("stale")
    stale.add_message(HumanMessage(content="old question"))
    idle = await manager.get_or_create_memory("idle")
    fresh = await manager.get_or_create_memory("fresh")
    expired_at = datetime.now() - timedelta(hours=3)
    stale.updated_at = idle.updated_at = expired_at

    again = await manager.get_or_create_memory("stale")

    assert again is not stale and again.messages == []
    assert set(manager._memories) == {"stale", "fresh"}
    assert manager._memories["fresh"] is fresh


def test_cleanup_oldest_no_op_for_nonpositive(patched_history) -> None:
    import asyncio

    manager = mm.MemoryManager(config=Config())

    async def scenario():
        await manager.get_or_create_memory("a")
        manager._cleanup_oldest_unsafe(0)
        return list(manager._memories.keys())

    assert asyncio.run(scenario()) == ["a"]


async def test_load_context_returns_history_and_entities(patched_history) -> None:
    manager = mm.MemoryManager(config=Config())
    await manager.add_turn("c1", "about Vendor", "an answer", entities=["Vendor"])

    history, entities = await manager.load_context("c1")

    assert history == "Human: about Vendor\nAI: an answer"
    assert entities == ["Vendor"]


def test_get_memory_manager_is_singleton(patched_history, mocker) -> None:
    # Reset the module-level singleton so the test is deterministic.
    mocker.patch.object(mm, "_memory_manager", None)
    mocker.patch.object(mm, "get_config", return_value=Config())
    m1 = mm.get_memory_manager()
    m2 = mm.get_memory_manager()
    assert m1 is m2


# --- shared extractor / reuse of query-step entities ---------------------


def test_conversations_share_one_entity_extractor(patched_history, mocker) -> None:
    import asyncio

    setup = mm.setup_chain
    manager = mm.MemoryManager(config=Config())

    async def scenario():
        a = await manager.get_or_create_memory("a")
        b = await manager.get_or_create_memory("b")
        return a, b

    a, b = asyncio.run(scenario())
    assert a.entity_extractor is b.entity_extractor
    assert setup.call_count == 1  # type: ignore[attr-defined]


def test_add_turn_with_known_entities_skips_the_llm(patched_history) -> None:
    import asyncio

    manager = mm.MemoryManager(config=Config())

    async def scenario():
        await manager.add_turn("c1", "about Vendor", "a", entities=["Vendor", " "])
        return await manager.get_or_create_memory("c1")

    history = asyncio.run(scenario())
    patched_history.extractor.invoke.assert_not_called()
    assert history.get_relevant_entities() == ["Vendor"]


def test_concurrent_turns_keep_each_answer_after_its_question(patched_history) -> None:
    import asyncio

    manager = mm.MemoryManager(config=Config())

    async def scenario():
        # No pre-extracted entities: each turn's entity update runs in a thread,
        # the await that let another turn interleave between two add_message
        # calls.
        await asyncio.gather(
            *(manager.add_turn("c1", f"q{i}", f"a{i}") for i in range(5))
        )
        history = await manager.get_or_create_memory("c1")
        return [m.content for m in history.messages]

    contents = asyncio.run(scenario())
    assert len(contents) == 10
    for question, answer in zip(contents[::2], contents[1::2], strict=True):
        assert question.startswith("q") and answer == "a" + question[1:]


def test_add_turn_records_given_entities_without_extraction(patched_history) -> None:
    import asyncio

    manager = mm.MemoryManager(config=Config())

    async def scenario():
        await manager.add_turn("c1", "q", "a", entities=["Vendor"])
        return await manager.get_or_create_memory("c1")

    history = asyncio.run(scenario())
    assert [type(m).__name__ for m in history.messages] == ["HumanMessage", "AIMessage"]
    assert "Vendor" in history.get_relevant_entities()
    patched_history.extractor.invoke.assert_not_called()


def test_turns_from_threads_on_separate_loops_stay_atomic(
    patched_history, mocker
) -> None:
    # The process-wide manager serves chains on different threads/loops (the
    # sync API's loop thread and the caller's loop). Its lock must exclude
    # across threads and must not bind to one event loop.
    import asyncio
    import threading
    import time

    original = mm.GraphRAGChatMessageHistory.append_message

    def slow_append(self, message):
        time.sleep(0.002)  # widen the critical section so threads contend
        original(self, message)

    mocker.patch.object(mm.GraphRAGChatMessageHistory, "append_message", slow_append)
    manager = mm.MemoryManager(config=Config())
    errors: list[BaseException] = []

    def worker(tag: str) -> None:
        async def turns() -> None:
            await asyncio.gather(
                *(
                    manager.add_turn("c1", f"q{tag}{i}", f"a{tag}{i}", entities=[])
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
    contents = [m.content for m in manager._memories["c1"].messages]
    assert len(contents) == 20
    for question, answer in zip(contents[::2], contents[1::2], strict=True):
        assert question.startswith("q") and answer == "a" + question[1:]
