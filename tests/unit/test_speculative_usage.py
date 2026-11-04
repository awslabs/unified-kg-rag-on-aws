# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A query's usage counts every model call it sent (AWS-free).

Regression: ``auto`` cancelled the speculative extraction its route did not
read, and the usage callback counted only calls that returned. A cancelled
task does not stop a request already sent (a Bedrock call runs on a worker
thread to the end), so the call was billed but missing from
``RAGOutput.metadata["usage"]``: offline, 12 of 20 calls of auto queries
routed to global. A call now counts from its start, at an estimate of its
input tokens until it answers (``estimated_calls``), so the query returns
without waiting for that extraction and leaves it running in the
background, to fill the extraction cache, until it answers, the query's
deadline passes or the chain closes. A call cancelled in flight counts the
same way.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import time
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from tests.fixtures.fakes.chat_models import fake_model_info
from unified_kg_rag.adapters.local.token_counter import local_token_counter_factory
from unified_kg_rag.adapters.observability.usage_callback import UsageCallbackHandler
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.application.retrieval.rag_chain import (
    DEFAULT_ERROR_MESSAGE,
    GraphRAGChain,
    RAGInput,
)
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.shared.usage import usage_scope

pytestmark = pytest.mark.unit

_KEYWORDS = '{"high_level_keywords": ["supply"], "low_level_keywords": ["Vendor"]}'


class _ThreadedChat(BaseChatModel):
    """Answers by prompt kind on a worker thread, as ChatBedrockConverse
    runs ``ainvoke``: a cancelled caller leaves the request running to the
    end. ``delays`` holds the seconds a kind takes; ``served`` the calls
    that completed at the "server"."""

    route: str = "simple"
    delays: dict[str, float] = Field(default_factory=dict)
    served: list[str] = Field(default_factory=list)
    cancelled: list[str] = Field(default_factory=list)
    # False: answer on the event loop instead (a cancel stops the request).
    threaded: bool = True
    lock: Any = Field(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "threaded"

    @staticmethod
    def kind(messages: list[BaseMessage]) -> str:
        system = "".join(str(m.content) for m in messages if m.type == "system")
        if "search strategy selector" in system:
            return "router"
        if "keyword extractor" in system:
            return "keywords"
        return "answer"

    def _generate(self, messages: list[BaseMessage], *a: Any, **k: Any) -> ChatResult:
        kind = self.kind(messages)
        time.sleep(self.delays.get(kind, 0.0))
        return self._reply(kind)

    async def _agenerate(
        self, messages: list[BaseMessage], *a: Any, **k: Any
    ) -> ChatResult:
        if self.threaded:
            return await super()._agenerate(messages, *a, **k)
        kind = self.kind(messages)
        try:
            await asyncio.sleep(self.delays.get(kind, 0.0))
        except asyncio.CancelledError:
            self.cancelled.append(kind)
            raise
        return self._reply(kind)

    def _reply(self, kind: str) -> ChatResult:
        with self.lock:
            self.served.append(kind)
        content = {"router": self.route, "keywords": _KEYWORDS}.get(
            kind, "Vendor supplies Buyer."
        )
        message = AIMessage(
            content=content,
            usage_metadata={"input_tokens": 50, "output_tokens": 5, "total_tokens": 55},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class _Factory:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def get_model(self, model_id: Any, **kwargs: Any) -> BaseChatModel:
        return self.model

    def get_model_info(self, model_id: Any) -> Any:
        return fake_model_info(model_id)


class _Retriever:
    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                content="Vendor and Buyer signed the supply agreement.",
                score=0.9,
                source="supply.txt",
                retriever_type="document",
                metadata={"id": "doc-1", "text_unit_ids": []},
            )
        ]


def _chain(model: _ThreadedChat, **search: Any) -> GraphRAGChain:
    config = Config()
    config.search.auto_routable_strategies = [SearchStrategy.SIMPLE, SearchStrategy.MIX]
    config.search.query_extraction_cache_size = 0
    # No rerank model: nothing here may build an AWS client.
    config.search.reranking.enabled = False
    for key, value in search.items():
        setattr(config.search, key, value)
    translation = config.processing.translation
    translation.source_language = translation.target_language
    model.callbacks = [UsageCallbackHandler("m", role="query")]
    retriever = _Retriever()
    providers = Providers(
        config,
        llm_factory=_Factory(model),
        token_counter_factory=local_token_counter_factory(config),
        retriever_builders={RetrieverRole.DOCUMENT: lambda: retriever},
    )
    return GraphRAGChain(config=config, providers=providers)


def _auto() -> RAGInput:
    return RAGInput(query="What does Vendor supply?", search_strategy="auto")


async def _until(check: Any, within: float = 10.0) -> None:
    """Wait (bounded) until ``check()`` holds."""
    deadline = time.monotonic() + within
    while not check():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


async def test_the_query_does_not_wait_for_the_unread_extraction() -> None:
    # Routed to simple, the keyword extraction mix would read runs on in the
    # background: the query returns at the answer, counting it from its start
    # at an estimate.
    model = _ThreadedChat(delays={"keywords": 1.0})
    chain = _chain(model)
    started = time.monotonic()
    out = await chain.ainvoke(_auto())
    assert time.monotonic() - started < 0.8
    assert out.metadata["search_strategy"] == "simple"
    usage = out.metadata["usage"]
    assert (usage["llm_calls"], usage["estimated_calls"]) == (3, 1)
    assert usage["output_tokens"] == 10 and usage["input_tokens"] > 100
    # Never cancelled: the request completes at the server.
    await _until(lambda: "keywords" in model.served)
    await _until(lambda: not chain._background_extractions._snapshot())


async def test_one_that_answers_before_the_snapshot_has_its_real_tokens() -> None:
    model = _ThreadedChat(delays={"answer": 0.3})
    chain = _chain(model)
    out = await chain.ainvoke(_auto())
    assert sorted(model.served) == ["answer", "keywords", "router"]
    usage = out.metadata["usage"]
    assert (usage["llm_calls"], usage["estimated_calls"]) == (3, 0)
    assert (usage["input_tokens"], usage["output_tokens"]) == (150, 15)


async def test_one_that_answers_later_settles_in_the_open_scopes() -> None:
    model = _ThreadedChat(delays={"keywords": 0.3})
    chain = _chain(model)
    with usage_scope() as run:
        out = await chain.ainvoke(_auto())
        await _until(lambda: not chain._background_extractions._snapshot())
    assert out.metadata["usage"]["estimated_calls"] == 1
    totals = run.snapshot()
    assert (totals["llm_calls"], totals["estimated_calls"]) == (3, 0)
    assert (totals["input_tokens"], totals["output_tokens"]) == (150, 15)


async def test_the_unread_extraction_fills_the_cache() -> None:
    model = _ThreadedChat(delays={"keywords": 0.2})
    chain = _chain(model, query_extraction_cache_size=8)
    await chain.ainvoke(_auto())
    await _until(lambda: not chain._background_extractions._snapshot())
    model.route = "mix"
    out = await chain.ainvoke(_auto())
    assert model.served.count("keywords") == 1
    assert out.metadata["usage"]["cache_hits"] == 1


async def test_a_stream_leaves_the_unread_extraction_running() -> None:
    model = _ThreadedChat(delays={"keywords": 0.5})
    chain = _chain(model, query_extraction_cache_size=8)
    chunks = [chunk async for chunk in chain.astream(_auto())]
    assert "".join(chunks)
    assert "keywords" not in model.served
    await _until(lambda: not chain._background_extractions._snapshot())
    model.route = "mix"
    await chain.ainvoke(_auto())
    assert model.served.count("keywords") == 1


async def test_the_query_deadline_cancels_the_background_extraction() -> None:
    model = _ThreadedChat(delays={"keywords": 30.0}, threaded=False)
    chain = _chain(model, query_timeout_seconds=0.6)
    started = time.monotonic()
    out = await chain.ainvoke(_auto())
    assert time.monotonic() - started < 0.5
    assert out.answer != DEFAULT_ERROR_MESSAGE
    assert out.metadata["usage"]["estimated_calls"] == 1
    await _until(lambda: "keywords" in model.cancelled, within=5)
    assert time.monotonic() - started < 5
    await _until(lambda: not chain._background_extractions._snapshot())


async def test_aclose_cancels_the_background_extraction() -> None:
    model = _ThreadedChat(delays={"keywords": 30.0}, threaded=False)
    chain = _chain(model, query_timeout_seconds=None)
    await chain.ainvoke(_auto())
    assert chain._background_extractions._snapshot()
    await chain.aclose()
    assert model.cancelled == ["keywords"]
    assert not chain._background_extractions._snapshot()


def test_close_ends_the_sync_paths_background_extraction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _ThreadedChat(delays={"keywords": 30.0}, threaded=False)
    chain = _chain(model, query_timeout_seconds=None)
    with caplog.at_level("DEBUG", logger="asyncio"):
        out = chain.invoke(_auto())
        assert out.metadata["usage"]["estimated_calls"] == 1
        assert chain._background_extractions._snapshot()
        started = time.monotonic()
        chain.close()
        assert time.monotonic() - started < 5
        gc.collect()
    assert model.cancelled == ["keywords"]
    assert not chain._background_extractions._snapshot()
    assert "destroyed but it is pending" not in caplog.text
    assert "exception was never retrieved" not in caplog.text


async def test_a_call_the_deadline_cancels_is_counted() -> None:
    # The answer call is cut by the deadline: under ignore_errors the query
    # returns the error answer, whose usage still holds the cut call.
    model = _ThreadedChat(delays={"answer": 3.0})
    chain = _chain(model, query_timeout_seconds=0.6)
    chain.ignore_errors = True
    out = await chain.ainvoke(_auto())
    assert out.answer == DEFAULT_ERROR_MESSAGE
    usage = out.metadata["usage"]
    assert (usage["llm_calls"], usage["estimated_calls"]) == (3, 1)
