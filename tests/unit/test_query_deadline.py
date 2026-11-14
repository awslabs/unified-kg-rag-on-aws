# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A query ends at its deadline, and query-time model clients time out early.

Regression: queries shared the ingestion Bedrock client (330 s read timeout,
6 requests per call) and the chain had no overall deadline, so one stalled
model call held a query for about 33 minutes. ``search.query_timeout_seconds``
now bounds ``ainvoke``/``invoke``/``asearch`` and a stream (with a first-chunk
limit of ``search.model_timeout_seconds``), raising ``QueryTimeoutError``; the
query's model clients get ``search.model_timeout_seconds`` and
``search.model_max_attempts``.

The stall tests drive a real Bedrock client against a local server that
accepts connections and never answers.
"""

from __future__ import annotations

import asyncio
import re
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from botocore.httpsession import URLLib3Session
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from tests.fixtures.fakes.chat_models import fake_model_info
from tests.fixtures.held_timeouts import HeldTimeouts
from unified_kg_rag import QueryTimeoutError
from unified_kg_rag.adapters.aws.bedrock import (
    BedrockEmbeddingModelFactory,
    BedrockLanguageModelFactory,
    BedrockRerankModelFactory,
)
from unified_kg_rag.adapters.llm.retry import next_transient_retry_delay, retry_deadline
from unified_kg_rag.adapters.local.token_counter import local_token_counter_factory
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.application.cli.run_rag_chain import RAGChainRunner
from unified_kg_rag.application.retrieval.rag_chain import (
    DEFAULT_ERROR_MESSAGE,
    GraphRAGChain,
    RAGInput,
    RAGOutput,
    caller_deadline,
)
from unified_kg_rag.domain.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationQuery,
    EvaluatorType,
    ModelPurpose,
    ModelRole,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
    SectionType,
)
from unified_kg_rag.domain.models.config import TransientRetryConfig
from unified_kg_rag.evaluation import EvaluationManager

pytestmark = pytest.mark.unit

_ANSWER = "Vendor supplies Buyer with parts."
# Captured at import, before the autouse fixture blocks botocore's transport:
# the stall tests send to a local server only.
_REAL_SEND = URLLib3Session.send


class _SlowChatModel(BaseChatModel):
    """Answers ``_ANSWER`` after ``first_delay`` s (``stall_on``: only for a
    question containing it), streaming a word per ``chunk_delay`` s."""

    first_delay: float = 0.0
    chunk_delay: float = 0.0
    stall_on: str | None = None
    started: list[bool] = Field(default_factory=list)
    stalled: list[bool] = Field(default_factory=list)
    answered: list[bool] = Field(default_factory=list)
    cancelled: list[bool] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "slow"

    def _delay_for(self, messages: list[BaseMessage]) -> float:
        if self.stall_on is None:
            return self.first_delay
        text = "".join(str(m.content) for m in messages)
        return self.first_delay if self.stall_on in text else 0.0

    def _generate(self, messages: list[BaseMessage], *a: Any, **k: Any) -> ChatResult:
        raise AssertionError("the chain calls the model asynchronously")

    async def _agenerate(
        self, messages: list[BaseMessage], *a: Any, **k: Any
    ) -> ChatResult:
        self.started.append(True)
        delay = self._delay_for(messages)
        if delay:
            self.stalled.append(True)
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            self.cancelled.append(True)
            raise
        self.answered.append(True)
        reply = AIMessage(content=_ANSWER)
        return ChatResult(generations=[ChatGeneration(message=reply)])

    async def _astream(  # type: ignore[override]
        self, messages: list[BaseMessage], *a: Any, **k: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.started.append(True)
        try:
            await asyncio.sleep(self._delay_for(messages))
            for word in _ANSWER.split(" "):
                yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
                await asyncio.sleep(self.chunk_delay)
        except asyncio.CancelledError:
            self.cancelled.append(True)
            raise


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
                retriever_type=SectionType.TEXT,
                metadata={"id": "doc-1", "text_unit_ids": []},
            )
        ]


def _config(**search: Any) -> Config:
    config = Config()
    for key, value in search.items():
        setattr(config.search, key, value)
    return config


def _chain(config: Config, llm_factory: Any) -> GraphRAGChain:
    retriever = _Retriever()
    providers = Providers(
        config,
        llm_factory=llm_factory,
        # No CountTokens round trip (it would go to the stall server too).
        token_counter_factory=local_token_counter_factory(config),
        retriever_builders={RetrieverRole.DOCUMENT: lambda: retriever},
    )
    return GraphRAGChain(config=config, providers=providers)


def _query(question: str = "What does Vendor supply?") -> RAGInput:
    return RAGInput(
        query=question,
        search_strategy=SearchStrategy.SIMPLE,
        enable_query_processing=False,
    )


async def _until(flag: list[bool], within: float = 30.0) -> None:
    """Wait (bounded) until ``flag`` holds an entry."""
    deadline = time.monotonic() + within
    while not flag:
        assert time.monotonic() < deadline, "timed out waiting for the fake model"
        await asyncio.sleep(0.01)


# --- the deadline --------------------------------------------------------------


async def test_a_normal_query_is_unchanged_under_the_default_deadline() -> None:
    config = Config()
    assert config.search.query_timeout_seconds == 180
    assert config.search.model_timeout_seconds == 90
    assert config.search.model_max_attempts == 2
    chain = _chain(config, _Factory(_SlowChatModel(first_delay=0.05)))
    out = await chain.ainvoke(_query())
    assert isinstance(out, RAGOutput)
    assert out.answer == _ANSWER and "error" not in out.metadata
    chunks = [c async for c in chain.astream(_query())]
    assert "".join(chunks).strip() == _ANSWER


async def test_ainvoke_past_the_deadline_raises_and_cancels_the_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _SlowChatModel(first_delay=600)
    chain = _chain(_config(query_timeout_seconds=300), _Factory(model))
    timeouts = HeldTimeouts(monkeypatch)
    query = asyncio.ensure_future(chain.ainvoke(_query()))
    # The deadline expires only once the model call is in flight.
    await _until(model.started)
    await timeouts.wait_open()
    timeouts.expire()
    with pytest.raises(QueryTimeoutError) as raised:
        await asyncio.wait_for(query, timeout=30)
    assert raised.value.timeout_seconds == 300
    assert isinstance(raised.value, TimeoutError)
    assert "search.query_timeout_seconds" in str(raised.value)
    await _until(model.cancelled)
    assert model.cancelled == [True]


async def test_a_callers_earlier_deadline_states_the_budget_the_query_had() -> None:
    # Regression (rv27 algo-ret low): under an MCP call's deadline the
    # error said "within 300 s (search.query_timeout_seconds)" for a query
    # that had well under a second after its wait for a call slot.
    chain = _chain(
        _config(query_timeout_seconds=300), _Factory(_SlowChatModel(first_delay=60))
    )
    with caller_deadline(time.monotonic() + 0.3):
        with pytest.raises(QueryTimeoutError) as raised:
            await asyncio.wait_for(chain.ainvoke(_query()), timeout=30)
    assert raised.value.timeout_seconds <= 0.3
    assert "the caller's deadline" in str(raised.value)
    # The budget left, rounded to hundredths.
    assert re.search(r"within 0(\.\d{1,2})? s", str(raised.value))
    assert "300" not in str(raised.value)


async def test_asearch_is_bounded_too() -> None:
    chain = _chain(_config(query_timeout_seconds=0.3), _Factory(_SlowChatModel()))

    class _StalledSearch:
        async def ainvoke(self, *_a: Any, **_k: Any) -> Any:
            await asyncio.sleep(60)

    chain._search_chain = _StalledSearch()  # type: ignore[assignment]
    with pytest.raises(QueryTimeoutError):
        await chain.asearch(_query())


def test_sync_invoke_past_the_deadline_raises() -> None:
    chain = _chain(
        _config(query_timeout_seconds=0.5), _Factory(_SlowChatModel(first_delay=60))
    )
    try:
        started = time.monotonic()
        with pytest.raises(QueryTimeoutError):
            chain.invoke(_query())
        assert time.monotonic() - started < 0.5 + 5
    finally:
        chain.close()


async def test_ignore_errors_returns_the_error_answer_on_a_timeout() -> None:
    config = _config(query_timeout_seconds=0.3)
    config.processing.ignore_errors = True
    chain = _chain(config, _Factory(_SlowChatModel(first_delay=60)))
    out = await chain.ainvoke(_query())
    assert isinstance(out, RAGOutput)
    assert out.answer == DEFAULT_ERROR_MESSAGE and out.metadata["error"] is True
    assert "search.query_timeout_seconds" in out.search_results.metadata["error"]


class _StalledRecordRetriever(_Retriever):
    """A store whose build-record read stalls (``stall``) or fails."""

    def __init__(self, *, stall: float = 0.0, error: Exception | None = None) -> None:
        self.stall = stall
        self.error = error
        self.reads = 0

    async def aread_build_record(self, suffix: str | None) -> dict[str, Any] | None:
        self.reads += 1
        await asyncio.sleep(self.stall)
        if self.error is not None:
            raise self.error
        return None


def _chain_on(config: Config, retriever: _Retriever) -> GraphRAGChain:
    providers = Providers(
        config,
        llm_factory=_Factory(_SlowChatModel()),
        token_counter_factory=local_token_counter_factory(config),
        retriever_builders={RetrieverRole.DOCUMENT: lambda: retriever},
    )
    return GraphRAGChain(config=config, providers=providers)


@pytest.mark.parametrize("stream", [False, True])
async def test_the_deadline_covers_a_new_chains_methodology_check(
    stream: bool,
) -> None:
    # Regression (rv24): a chain's first query read the namespace's build
    # record (the methodology check) before its deadline began, so a stalled
    # store held run-rag and a new console or MCP chain for 2 x store_timeout
    # x attempts before the 180 s deadline even started.
    chain = _chain_on(
        _config(query_timeout_seconds=0.3), _StalledRecordRetriever(stall=60)
    )
    started = time.monotonic()
    with pytest.raises(QueryTimeoutError):
        if stream:
            await asyncio.wait_for(_drain(chain.astream(_query())), timeout=10)
        else:
            await asyncio.wait_for(chain.ainvoke(_query()), timeout=10)
    assert time.monotonic() - started < 5


async def _drain(chunks: AsyncIterator[str]) -> None:
    async for _ in chunks:
        pass


async def test_a_failed_build_record_read_is_not_repeated_within_a_query() -> None:
    # The methodology check and the held artifacts read the record in turn:
    # a failed read (a store timeout, retried by its client) is kept for the
    # rest of the query instead of being paid twice.
    from unified_kg_rag.shared.exceptions import StoreTimeoutError

    retriever = _StalledRecordRetriever(error=StoreTimeoutError("stalled"))
    config = _config()
    config.providers.rerank.kind = "none"
    chain = _chain_on(config, retriever)
    await chain.ainvoke(_query())
    assert retriever.reads == 1
    # Not kept past the query: the next one reads it again.
    await chain.ainvoke(_query())
    assert retriever.reads == 2


async def test_null_disables_both_deadlines() -> None:
    config = _config(query_timeout_seconds=None, model_timeout_seconds=0.05)
    chain = _chain(config, _Factory(_SlowChatModel(first_delay=0.3)))
    assert (await chain.ainvoke(_query())).answer == _ANSWER
    chunks = [c async for c in chain.astream(_query())]
    assert "".join(chunks).strip() == _ANSWER


async def test_a_stream_without_a_first_chunk_fails_at_the_model_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _SlowChatModel(first_delay=600)
    config = _config(query_timeout_seconds=600, model_timeout_seconds=300)
    chain = _chain(config, _Factory(model))
    timeouts = HeldTimeouts(monkeypatch)

    async def consume() -> None:
        async for _ in chain.astream(_query()):
            pass

    stream = asyncio.ensure_future(consume())
    # The first-chunk limit (the earlier of the two) expires once the stream's
    # model call is in flight; the query deadline is not reached.
    await _until(model.started)
    await timeouts.wait_open()
    timeouts.expire()
    with pytest.raises(QueryTimeoutError) as raised:
        await asyncio.wait_for(stream, timeout=30)
    assert raised.value.timeout_seconds == 300
    assert "first chunk" in str(raised.value)
    await _until(model.cancelled)
    assert model.cancelled == [True]


async def test_a_stream_that_stalls_after_its_first_chunk_ends_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _SlowChatModel(chunk_delay=600)
    config = _config(query_timeout_seconds=600, model_timeout_seconds=300)
    chain = _chain(config, _Factory(model))
    timeouts = HeldTimeouts(monkeypatch)
    received: list[str] = []

    async def consume() -> None:
        async for chunk in chain.astream(_query()):
            received.append(chunk)

    stream = asyncio.ensure_future(consume())
    # The query deadline expires once the first chunk is delivered (the
    # stream then waits on the second under the query deadline alone).
    await _until(received)  # type: ignore[arg-type]
    # The scope of the wait for the next chunk: the query deadline (600 s),
    # not the first-chunk limit (300 s).
    await timeouts.wait_open(beyond=400)
    timeouts.expire(beyond=400)
    with pytest.raises(QueryTimeoutError) as raised:
        await asyncio.wait_for(stream, timeout=30)
    assert received == ["Vendor "]
    assert raised.value.timeout_seconds == 600


def test_sync_stream_past_the_first_chunk_limit_raises() -> None:
    config = _config(query_timeout_seconds=30, model_timeout_seconds=0.3)
    chain = _chain(config, _Factory(_SlowChatModel(first_delay=60)))
    try:
        with pytest.raises(QueryTimeoutError):
            list(chain.stream(_query()))
    finally:
        chain.close()


async def test_run_eval_counts_a_timed_out_query_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(query_timeout_seconds=300)
    config.evaluation.enabled_evaluators = [EvaluatorType.RETRIEVAL]
    model = _SlowChatModel(first_delay=600, stall_on="stall")
    timeouts = HeldTimeouts(monkeypatch)
    manager = EvaluationManager(config, rag_chain=_chain(config, _Factory(model)))
    metadata = {"search_strategy": "simple", "enable_query_processing": False}
    queries = [
        EvaluationQuery(query_id="ok", question="What is supplied?", metadata=metadata),
        EvaluationQuery(query_id="slow", question="stall here", metadata=metadata),
    ]
    truths = [
        EvaluationGroundTruth(
            query_id=q.query_id, ground_truth="", reference_sources=["supply.txt"]
        )
        for q in queries
    ]
    run = asyncio.ensure_future(
        manager.evaluate_dataset(queries, truths, show_progress=False)
    )
    # Only the stalled query's deadline expires, once the other is answered.
    await _until(model.stalled)
    await _until(model.answered)
    await timeouts.wait_open(count=1)  # the answered query's scope has closed
    timeouts.expire()
    results, _, summary = await asyncio.wait_for(run, timeout=60)
    by_id = {r.query_id: r for r in results}
    assert by_id["slow"].error and not by_id["ok"].error
    assert summary.failed_evaluations == 1 and summary.successful_evaluations == 1


async def test_run_rag_reports_a_timeout_without_a_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = RAGChainRunner.__new__(RAGChainRunner)

    class _Args:
        mode = "rag"
        interactive = False
        search_strategy = "simple"

    class _TimedOut:
        async def ainvoke(self, _input: Any) -> Any:
            raise QueryTimeoutError("The query did not finish", timeout_seconds=1)

    runner.args = _Args()  # type: ignore[assignment]
    runner.rag_chain = _TimedOut()  # type: ignore[assignment]
    result = await runner._run_query(_query())
    assert result["success"] is False  # run() exits 1 on it
    assert "did not finish" in result["error"]
    assert not any(record.exc_info for record in caplog.records)


def test_no_transient_retry_starts_past_the_query_deadline() -> None:
    policy = TransientRetryConfig(base_delay_seconds=0.0, max_total_seconds=600)
    error = ConnectionError("stalled")

    def delay() -> float | None:
        return next_transient_retry_delay(
            error,
            is_transient=lambda _e: True,
            operation="answer",
            attempt=1,
            started_at=time.monotonic(),
            policy=policy,
        )

    assert delay() is not None
    with retry_deadline(time.monotonic() - 1):
        assert delay() is None
    with retry_deadline(time.monotonic() + 60):
        assert delay() is not None


# --- query-time model clients ----------------------------------------------------


def _client_settings(client: Any) -> tuple[float, dict[str, Any]]:
    config = client.meta.config
    return config.read_timeout, dict(config.retries)


def test_query_models_get_the_short_client_and_ingestion_keeps_its_own() -> None:
    config = Config()
    llm = BedrockLanguageModelFactory(config)
    query = llm.get_model(
        config.model_id(ModelRole.ANSWER_GENERATION), model_purpose=ModelPurpose.QUERY
    )
    ingestion = llm.get_model(
        config.model_id(ModelRole.ANSWER_GENERATION),
        model_purpose=ModelPurpose.INGESTION,
    )
    judge = llm.get_model(
        config.model_id(ModelRole.ANSWER_GENERATION),
        model_purpose=ModelPurpose.EVALUATION,
    )
    assert _client_settings(query.client) == (
        90,
        {"total_max_attempts": 2, "mode": "adaptive"},
    )
    for model in (ingestion, judge):
        assert _client_settings(model.client) == (
            330,
            {"total_max_attempts": 6, "mode": "adaptive"},
        )

    embeddings = BedrockEmbeddingModelFactory(config)
    query_embed = embeddings.get_model(
        config.embedding_model_id, model_purpose=ModelPurpose.QUERY
    )
    ingest_embed = embeddings.get_model(
        config.embedding_model_id, model_purpose=ModelPurpose.INGESTION
    )
    assert _client_settings(query_embed.client)[0] == 90
    assert _client_settings(ingest_embed.client)[0] == 330

    rerank = BedrockRerankModelFactory(config).get_model(config.rerank_model_id)
    assert _client_settings(rerank.client)[0] == 90


def test_query_client_follows_the_search_settings() -> None:
    config = _config(model_timeout_seconds=15, model_max_attempts=3)
    llm = BedrockLanguageModelFactory(config).get_model(
        config.model_id(ModelRole.ANSWER_GENERATION)
    )
    assert _client_settings(llm.client) == (
        15,
        {"total_max_attempts": 3, "mode": "adaptive"},
    )


def test_local_query_models_get_the_capped_request_timeout() -> None:
    pytest.importorskip("langchain_openai")
    pytest.importorskip("langchain_ollama")
    from unified_kg_rag.adapters.local.embeddings import (
        OllamaEmbeddingFactory,
        OpenAICompatibleEmbeddingFactory,
    )
    from unified_kg_rag.adapters.local.models import (
        OllamaLanguageModelFactory,
        OpenAICompatibleLanguageModelFactory,
    )

    config = _config(model_timeout_seconds=20)
    config.providers.llm.timeout_seconds = 300
    config.providers.embedding.timeout_seconds = 300
    openai = OpenAICompatibleLanguageModelFactory(config)
    assert openai.get_model("m", model_purpose=ModelPurpose.QUERY).request_timeout == 20
    assert (
        openai.get_model("m", model_purpose=ModelPurpose.INGESTION).request_timeout
        == 300
    )
    ollama = OllamaLanguageModelFactory(config)
    query = ollama.get_model("m", model_purpose=ModelPurpose.QUERY)
    ingestion = ollama.get_model("m", model_purpose=ModelPurpose.INGESTION)
    assert query.client_kwargs["timeout"] == 20
    assert ingestion.client_kwargs["timeout"] == 300

    embed = OpenAICompatibleEmbeddingFactory(config)
    assert (
        embed.get_model("e", model_purpose=ModelPurpose.QUERY).inner.request_timeout
        == 20
    )
    assert (
        embed.get_model("e", model_purpose=ModelPurpose.INGESTION).inner.request_timeout
        == 300
    )
    ollama_embed = OllamaEmbeddingFactory(config)
    assert (
        ollama_embed.get_model(
            "e", model_purpose=ModelPurpose.QUERY
        ).inner.client_kwargs["timeout"]
        == 20
    )


# --- a stalled Bedrock endpoint ----------------------------------------------------


class _StallServer:
    """Accepts every connection, reads nothing and never answers."""

    def __init__(self) -> None:
        self._server = socket.socket()
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(64)
        self.port = self._server.getsockname()[1]
        self.connections: list[socket.socket] = []
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        while True:
            try:
                connection, _ = self._server.accept()
            except OSError:
                return
            self.connections.append(connection)

    def close(self) -> None:
        self._server.close()
        for connection in self.connections:
            connection.close()


@pytest.fixture
def stall_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[_StallServer]:
    server = _StallServer()

    def local_only_send(self: URLLib3Session, request: Any) -> Any:
        assert request.url.startswith(f"http://127.0.0.1:{server.port}/")
        return _REAL_SEND(self, request)

    monkeypatch.setattr(URLLib3Session, "send", local_only_send)
    monkeypatch.setenv(
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME", f"http://127.0.0.1:{server.port}"
    )
    yield server
    server.close()


def _stall_config() -> Config:
    config = _config(
        query_timeout_seconds=5.0, model_timeout_seconds=1.0, model_max_attempts=2
    )
    # Retry at once, so the query-chain retry runs into the deadline.
    config.retry.base_delay_seconds = 0.0
    return config


def _assert_bounded(server: _StallServer, started: float, deadline: float) -> None:
    # Timing is the point here (a real socket stalls): the bounds leave
    # generous headroom for a loaded runner.
    assert time.monotonic() - started < deadline + 5
    # Up to the deadline: 1 s per request, so at most 5, and the one in
    # flight on the worker thread then makes its second attempt (sent within
    # 1 s, ended within 2 s); no retry starts after it.
    time.sleep(4.0)
    sent = len(server.connections)
    assert 1 <= sent <= 6
    time.sleep(2.0)
    assert len(server.connections) == sent


@pytest.mark.slow
async def test_a_stalled_bedrock_answer_fails_at_the_deadline(
    stall_server: _StallServer,
) -> None:
    config = _stall_config()
    chain = _chain(config, BedrockLanguageModelFactory(config))
    started = time.monotonic()
    with pytest.raises(QueryTimeoutError):
        await chain.ainvoke(_query())
    _assert_bounded(stall_server, started, 5.0)


@pytest.mark.slow
def test_a_stalled_bedrock_answer_fails_at_the_deadline_on_the_sync_path(
    stall_server: _StallServer,
) -> None:
    config = _stall_config()
    chain = _chain(config, BedrockLanguageModelFactory(config))
    try:
        started = time.monotonic()
        with pytest.raises(QueryTimeoutError):
            chain.invoke(_query())
        _assert_bounded(stall_server, started, 5.0)
    finally:
        chain.close()


def test_an_abandoned_executor_does_not_hold_asyncio_run() -> None:
    from concurrent.futures import ThreadPoolExecutor

    from unified_kg_rag.shared.utils.event_loop import abandon_default_executor

    running, release, ended = threading.Event(), threading.Event(), threading.Event()

    def left_behind() -> None:
        running.set()
        release.wait(30)
        ended.set()

    async def main() -> None:
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=2)
        loop.set_default_executor(executor)
        # A call a timed-out query left behind.
        loop.run_in_executor(None, left_behind)
        assert await asyncio.to_thread(running.wait, 30)
        abandon_default_executor(loop, executor)

    try:
        asyncio.run(main())
        # asyncio.run returned while the call still runs: it did not wait.
        assert not ended.is_set()
    finally:
        release.set()


@pytest.mark.parametrize("stream", [False, True])
async def test_ignore_errors_answers_a_timeout_during_the_admission(
    stream: bool,
) -> None:
    # Regression (rv25): a deadline passing during the admission (the
    # build-record read) raised QueryTimeoutError under ignore_errors, where
    # every later timeout gives the error answer (the console and MCP got an
    # exception).
    config = _config(query_timeout_seconds=0.3)
    config.processing.ignore_errors = True
    chain = _chain_on(config, _StalledRecordRetriever(stall=60))
    started = time.monotonic()
    if stream:
        chunks = [c async for c in chain.astream(_query())]
        assert "".join(chunks) == DEFAULT_ERROR_MESSAGE
    else:
        out = await asyncio.wait_for(chain.ainvoke(_query()), timeout=10)
        assert isinstance(out, RAGOutput)
        assert out.answer == DEFAULT_ERROR_MESSAGE and out.metadata["error"] is True
        assert "search.query_timeout_seconds" in out.search_results.metadata["error"]
    assert time.monotonic() - started < 5


def test_a_caller_deadline_bounds_a_query_without_its_own_timeout() -> None:
    """With ``search.query_timeout_seconds`` unset, a caller's deadline
    (``caller_deadline``, the MCP server's) is the query's, its seconds
    what is left of it; an earlier one of the query's own wins."""
    from unified_kg_rag.application.retrieval.rag_chain import (
        _QueryDeadline,
        caller_deadline,
    )

    at = time.monotonic() + 30
    with caller_deadline(at):
        unbounded = _QueryDeadline(None)
        own = _QueryDeadline(5.0)
    assert unbounded.at == at and unbounded.seconds is not None
    assert 0 < unbounded.seconds <= 30
    assert own.seconds == 5.0 and own.at is not None and own.at < at
    assert _QueryDeadline(None).at is None
