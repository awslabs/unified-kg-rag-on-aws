# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""GraphRAGChain.stream / astream: real token streaming of the answer only.

The chain runs retrieval + context building once, then streams the answer LLM.
A fake model factory returns a ``GenericFakeChatModel`` that emits the answer
word by word, so streaming is observable without Bedrock. Retrievers are
injected via ``retriever_builders``; memory is a recording fake.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from unified_kg_rag.application.retrieval.rag_chain import (
    DEFAULT_ERROR_MESSAGE,
    ChainMode,
    GraphRAGChain,
    RAGInput,
)
from unified_kg_rag.domain.models import (
    Config,
    MessageRole,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)

pytestmark = pytest.mark.unit

_ANSWER = "Vendor supplies Buyer with parts under the agreement."
_NO_DATA_FRAGMENT = "could not find relevant information"


class _FailingChatModel(GenericFakeChatModel):
    """Emits a few chunks, then fails mid-stream."""

    async def _astream(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        count = 0
        async for chunk in super()._astream(*args, **kwargs):
            if count == 2:
                raise RuntimeError("synthetic model failure")
            count += 1
            yield chunk


class _StreamingModelFactory:
    """LLMFactoryPort whose models stream ``_ANSWER`` in word-sized chunks."""

    def __init__(self, *, fail_mid_stream: bool = False) -> None:
        self.fail_mid_stream = fail_mid_stream
        self.get_model_calls = 0

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        self.get_model_calls += 1
        cls = _FailingChatModel if self.fail_mid_stream else GenericFakeChatModel
        return cls(messages=iter([AIMessage(content=_ANSWER)]))

    def get_model_info(self, model_id: Any) -> Any:
        return None


class _FakeRetriever:
    def __init__(self, *, empty: bool = False) -> None:
        self.empty = empty

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        if self.empty:
            return []
        return [
            RetrievalResult(
                content="Vendor and Buyer signed the supply agreement.",
                score=0.9,
                source="doc-1",
                retriever_type="document",
                metadata={"id": "doc-1", "text_unit_ids": []},
            )
        ]


class _FakeMemory:
    def load_memory_variables(self, _inputs: dict[str, Any]) -> dict[str, Any]:
        return {"history": "", "relevant_entities": []}


class _RecordingMemoryManager:
    def __init__(self) -> None:
        self.messages: list[tuple[str, MessageRole, str]] = []

    async def get_langchain_memory(self, conv_id: str, **_: Any) -> _FakeMemory:
        return _FakeMemory()

    async def add_message(self, conv_id: str, role: MessageRole, content: str) -> None:
        self.messages.append((conv_id, role, content))


def _make_chain(
    config: Config,
    *,
    factory: _StreamingModelFactory | None = None,
    empty: bool = False,
    retrieval_error: bool = False,
    ignore_errors: bool = False,
    mode: ChainMode = ChainMode.RAG,
) -> tuple[GraphRAGChain, _StreamingModelFactory, _RecordingMemoryManager]:
    config.processing.ignore_errors = ignore_errors
    factory = factory or _StreamingModelFactory()
    retriever = _FakeRetriever(empty=empty)

    def _build() -> _FakeRetriever:
        # Strategies swallow per-retriever query errors, so a failing backend
        # is simulated at retriever construction, which surfaces to the chain.
        if retrieval_error:
            raise RuntimeError("synthetic retrieval failure")
        return retriever

    chain = GraphRAGChain(
        config=config,
        mode=mode,
        model_factory=factory,
        retriever_builders={
            RetrieverRole.DOCUMENT: _build,
            RetrieverRole.GRAPH: _build,
        },
    )
    chain.token_manager.count_tokens = lambda text: len((text or "").split())
    memory = _RecordingMemoryManager()
    chain.memory_manager = memory  # type: ignore[assignment]
    return chain, factory, memory


def _input(**overrides: Any) -> RAGInput:
    params: dict[str, Any] = {
        "query": "What does Vendor supply to Buyer?",
        "search_strategy": SearchStrategy.SIMPLE,
        "enable_query_processing": False,
    }
    params.update(overrides)
    return RAGInput(**params)


async def test_astream_yields_multiple_chunks_forming_the_full_answer(
    config: Config,
) -> None:
    chain, factory, _ = _make_chain(config)
    chunks = [chunk async for chunk in chain.astream(_input())]

    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)
    assert "".join(chunks) == _ANSWER
    assert factory.get_model_calls == 1


async def test_astream_matches_ainvoke_answer(config: Config) -> None:
    chain, _, _ = _make_chain(config)
    streamed = "".join([c async for c in chain.astream(_input())])
    invoked = await chain.ainvoke(_input())
    assert streamed == invoked.answer  # type: ignore[union-attr]


async def test_astream_empty_context_yields_no_data_answer_without_llm(
    config: Config,
) -> None:
    chain, factory, _ = _make_chain(config, empty=True)
    chunks = [chunk async for chunk in chain.astream(_input())]

    assert len(chunks) == 1
    assert _NO_DATA_FRAGMENT in chunks[0].lower()
    assert factory.get_model_calls == 0


async def test_astream_saves_full_answer_to_memory(config: Config) -> None:
    chain, _, memory = _make_chain(config)
    chunks = [
        c
        async for c in chain.astream(
            _input(use_memory=True, conversation_id="conv-stream-1")
        )
    ]

    assert "".join(chunks) == _ANSWER
    assert memory.messages == [
        ("conv-stream-1", MessageRole.USER, "What does Vendor supply to Buyer?"),
        ("conv-stream-1", MessageRole.ASSISTANT, _ANSWER),
    ]


async def test_astream_without_memory_does_not_save(config: Config) -> None:
    chain, _, memory = _make_chain(config)
    _ = [c async for c in chain.astream(_input())]
    assert memory.messages == []


async def test_astream_retrieval_error_propagates_by_default(config: Config) -> None:
    chain, factory, memory = _make_chain(config, retrieval_error=True)
    with pytest.raises(RuntimeError, match="synthetic retrieval failure"):
        _ = [c async for c in chain.astream(_input())]
    assert factory.get_model_calls == 0
    assert memory.messages == []


async def test_astream_retrieval_error_degrades_with_ignore_errors(
    config: Config,
) -> None:
    chain, _, memory = _make_chain(config, retrieval_error=True, ignore_errors=True)
    chunks = [c async for c in chain.astream(_input())]
    assert chunks == [DEFAULT_ERROR_MESSAGE]
    assert memory.messages == []


async def test_astream_mid_stream_llm_error_propagates_by_default(
    config: Config,
) -> None:
    chain, _, memory = _make_chain(
        config, factory=_StreamingModelFactory(fail_mid_stream=True)
    )
    received: list[str] = []
    with pytest.raises(RuntimeError, match="synthetic model failure"):
        async for chunk in chain.astream(_input()):
            received.append(chunk)
    assert received  # partial output was delivered before the failure
    assert memory.messages == []


async def test_astream_mid_stream_llm_error_appends_notice_with_ignore_errors(
    config: Config,
) -> None:
    chain, _, memory = _make_chain(
        config,
        factory=_StreamingModelFactory(fail_mid_stream=True),
        ignore_errors=True,
    )
    chunks = [c async for c in chain.astream(_input())]
    assert len(chunks) > 1
    assert chunks[-1].endswith(DEFAULT_ERROR_MESSAGE)
    assert memory.messages == []


async def test_astream_search_mode_yields_nothing(config: Config) -> None:
    chain, _, _ = _make_chain(config, mode=ChainMode.SEARCH)
    assert [c async for c in chain.astream(_input())] == []


def test_stream_sync_yields_multiple_chunks(config: Config) -> None:
    chain, _, _ = _make_chain(config)
    chunks = list(chain.stream(_input()))
    assert len(chunks) > 1
    assert "".join(chunks) == _ANSWER


def test_stream_sync_empty_context_yields_no_data_answer(config: Config) -> None:
    chain, factory, _ = _make_chain(config, empty=True)
    chunks = list(chain.stream(_input()))
    assert len(chunks) == 1
    assert _NO_DATA_FRAGMENT in chunks[0].lower()
    assert factory.get_model_calls == 0


def test_stream_sync_error_propagates(config: Config) -> None:
    chain, _, _ = _make_chain(config, retrieval_error=True)
    with pytest.raises(RuntimeError, match="synthetic retrieval failure"):
        list(chain.stream(_input()))


def test_stream_sync_early_break_closes_cleanly(config: Config) -> None:
    chain, _, memory = _make_chain(config)
    gen = chain.stream(_input(use_memory=True, conversation_id="conv-stream-2"))
    first = next(gen)
    gen.close()
    assert first
    # The answer was abandoned, so nothing is persisted as a completed turn.
    assert memory.messages == []


async def test_stream_sync_works_inside_running_event_loop(config: Config) -> None:
    # Sync stream called from a thread that already runs a loop must not raise
    # "asyncio.run() cannot be called from a running event loop".
    asyncio.get_running_loop()
    chain, _, _ = _make_chain(config)
    chunks = list(chain.stream(_input()))
    assert "".join(chunks) == _ANSWER
