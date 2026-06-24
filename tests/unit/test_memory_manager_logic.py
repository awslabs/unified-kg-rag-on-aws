# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for the conversation-memory adapter.

``GraphRAGChatMessageHistory.__init__`` constructs a boto3 Session, a
``BedrockLanguageModelFactory`` and an entity-extraction chain. All three are
patched out so the message buffering, entity-context tracking, trimming, the
LangChain ``GraphRAGConversationBufferMemory`` glue, and the async
``MemoryManager`` capacity/eviction logic can be exercised without AWS or a
network.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aws_graphrag.adapters.retrieval import memory_manager as mm
from aws_graphrag.domain.models import Config, MessageRole

pytestmark = pytest.mark.unit


@pytest.fixture
def patched_history(mocker):
    """Patch the AWS-coupled bits of GraphRAGChatMessageHistory.

    Returns a factory that builds a history whose entity extractor is a stub
    invokable (default: returns a comma-separated entity list shape).
    """
    mocker.patch.object(mm.boto3, "Session", return_value=object())
    mocker.patch.object(mm, "BedrockLanguageModelFactory", return_value=object())

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


def test_add_messages_appends_all(patched_history) -> None:
    history = patched_history()
    history.add_messages([HumanMessage(content="a"), AIMessage(content="b")])
    assert [m.content for m in history.messages] == ["a", "b"]


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
    summary = history.get_context_summary()
    assert "Alice" in summary and "Bob" in summary
    assert "Focused on" in summary


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


def test_n_entities_limits_focused_and_relevant(patched_history) -> None:
    patched_history.extractor.invoke.return_value = ["A", "B", "C", "D", "E", "F"]
    history = patched_history(n_entities=2)
    history.add_message(HumanMessage(content="many entities"))
    # focused_entities capped at n_entities.
    assert len(history._context.focused_entities) == 2
    assert history.get_relevant_entities() == ["A", "B"]


def test_clear_resets_messages_and_context(patched_history) -> None:
    history = patched_history()
    history.add_message(HumanMessage(content="Alice"))
    history.clear()
    assert history.messages == []
    assert history.get_relevant_entities() == []
    assert history.get_context_summary() == "New Conversation"


def test_get_context_summary_empty_is_new_conversation(patched_history) -> None:
    history = patched_history()
    assert history.get_context_summary() == "New Conversation"


# --- GraphRAGConversationBufferMemory ------------------------------------


def test_buffer_returns_formatted_string_by_default(patched_history) -> None:
    history = patched_history()
    history.add_messages([HumanMessage(content="hi"), AIMessage(content="hello")])
    memory = mm.GraphRAGConversationBufferMemory(chat_memory=history)
    buf = memory.buffer
    assert isinstance(buf, str)
    assert "Human: hi" in buf
    assert "AI: hello" in buf


def test_buffer_returns_messages_when_return_messages(patched_history) -> None:
    history = patched_history()
    history.add_message(HumanMessage(content="hi"))
    memory = mm.GraphRAGConversationBufferMemory(
        chat_memory=history, return_messages=True
    )
    assert memory.buffer == [HumanMessage(content="hi")]


def test_memory_variables_include_entity_context_keys(patched_history) -> None:
    history = patched_history()
    memory = mm.GraphRAGConversationBufferMemory(chat_memory=history)
    assert memory.memory_variables == [
        "history",
        "relevant_entities",
        "conversation_context",
    ]


def test_memory_variables_omit_entity_context_when_disabled(patched_history) -> None:
    history = patched_history()
    memory = mm.GraphRAGConversationBufferMemory(
        chat_memory=history, include_entity_context=False
    )
    assert memory.memory_variables == ["history"]


def test_load_memory_variables_populates_entities(patched_history) -> None:
    history = patched_history()
    history.add_message(HumanMessage(content="about Alice and Bob"))
    memory = mm.GraphRAGConversationBufferMemory(chat_memory=history)
    mem = memory.load_memory_variables({})
    assert mem["relevant_entities"] == ["Alice", "Bob"]
    assert "Alice" in mem["conversation_context"]
    assert "history" in mem


def test_save_context_adds_human_and_ai_messages(patched_history) -> None:
    history = patched_history()
    memory = mm.GraphRAGConversationBufferMemory(chat_memory=history)
    memory.save_context({"input": "q"}, {"output": "a"})
    kinds = [type(m).__name__ for m in history.messages]
    assert kinds == ["HumanMessage", "AIMessage"]


def test_memory_clear_delegates_to_history(patched_history) -> None:
    history = patched_history()
    history.add_message(HumanMessage(content="x"))
    memory = mm.GraphRAGConversationBufferMemory(chat_memory=history)
    memory.clear()
    assert history.messages == []


def test_get_input_prefers_input_then_query() -> None:
    assert mm.GraphRAGConversationBufferMemory._get_input({"input": "i"}) == "i"
    assert mm.GraphRAGConversationBufferMemory._get_input({"query": "q"}) == "q"
    # Falls back to str(dict) when neither present.
    assert "k" in mm.GraphRAGConversationBufferMemory._get_input({"k": "v"})


def test_get_output_prefers_output_answer_text() -> None:
    g = mm.GraphRAGConversationBufferMemory
    assert g._get_output({"output": "o"}) == "o"
    assert g._get_output({"answer": "a"}) == "a"
    assert g._get_output({"text": "t"}) == "t"


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


def test_add_message_maps_roles(patched_history) -> None:
    import asyncio

    manager = mm.MemoryManager(config=Config())

    async def scenario():
        await manager.add_message("c1", MessageRole.USER, "u")
        await manager.add_message("c1", MessageRole.ASSISTANT, "a")
        await manager.add_message("c1", MessageRole.SYSTEM, "s")
        history = await manager.get_or_create_memory("c1")
        return [type(m).__name__ for m in history.messages]

    kinds = asyncio.run(scenario())
    assert kinds == ["HumanMessage", "AIMessage", "SystemMessage"]


def test_cleanup_oldest_no_op_for_nonpositive(patched_history) -> None:
    import asyncio

    manager = mm.MemoryManager(config=Config())

    async def scenario():
        await manager.get_or_create_memory("a")
        await manager._cleanup_oldest_unsafe(0)
        return list(manager._memories.keys())

    assert asyncio.run(scenario()) == ["a"]


def test_get_langchain_memory_wraps_history(patched_history) -> None:
    import asyncio

    manager = mm.MemoryManager(config=Config())
    memory = asyncio.run(manager.get_langchain_memory("c1", return_messages=True))
    assert isinstance(memory, mm.GraphRAGConversationBufferMemory)
    assert memory.return_messages is True


def test_get_memory_manager_is_singleton(patched_history, mocker) -> None:
    # Reset the module-level singleton so the test is deterministic.
    mocker.patch.object(mm, "_memory_manager", None)
    mocker.patch.object(mm, "get_config", return_value=Config())
    m1 = mm.get_memory_manager()
    m2 = mm.get_memory_manager()
    assert m1 is m2
