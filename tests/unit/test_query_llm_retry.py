# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the transient-error retry on query-time LLM chains.

AWS-free: chains are built by ``setup_chain`` from a Bedrock factory stand-in
(no boto client) whose model raises scripted botocore ``ClientError`` s shaped
like Bedrock's HTTP 424 ``ModelErrorException`` before answering. Backoff
delays are configured to zero so no test waits.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from botocore.exceptions import ClientError, EventStreamError
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessageChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from urllib3 import exceptions as urllib3_exc

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from tests.fixtures.fakes.chat_models import fake_model_info
from tests.fixtures.fakes.retrievers import StaticRetriever, hit
from unified_kg_rag import AnswerChunk
from unified_kg_rag.adapters.aws.bedrock import BedrockLanguageModelFactory
from unified_kg_rag.adapters.llm.chain_factory import (
    TransientRetryRunnable,
    setup_chain,
    with_transient_retry,
)
from unified_kg_rag.adapters.llm.langchain import BATCH_ITEM_FAILED, BatchProcessor
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.models import (
    Config,
    LanguageModelId,
    ModelPurpose,
    RetrieverRole,
    SearchStrategy,
)
from unified_kg_rag.domain.models.config import RetryConfig, RetryPolicy
from unified_kg_rag.domain.prompts import StrategySelectionPrompt

pytestmark = pytest.mark.unit

_ANSWER = "Vendor ships parts to Buyer every quarter."
_NO_WAIT = RetryPolicy(max_attempts=3, base_delay_seconds=0.0, max_delay_seconds=0.0)


def _client_error(code: str, status: int) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": "synthetic failure"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "Converse",
    )


def _model_error() -> ClientError:
    return _client_error("ModelErrorException", 424)


def _validation_error() -> ClientError:
    return _client_error("ValidationException", 400)


class _ScriptedModel(Runnable[Any, str]):
    """Fake chat model: raises ``errors`` in order, then returns ``answer``.

    Streaming yields the answer word by word and can fail after a number of
    chunks, to exercise the no-retry-after-output rule. A stream first yields
    ``reasoning`` (chunks without answer text), then raises the next of
    ``after_reasoning`` if any is left.
    """

    def __init__(
        self,
        errors: list[BaseException] | None = None,
        *,
        answer: str = _ANSWER,
        fail_stream_after: int | None = None,
        reasoning: list[Any] | None = None,
        after_reasoning: list[BaseException] | None = None,
    ) -> None:
        self.errors = list(errors or [])
        self.answer = answer
        self.fail_stream_after = fail_stream_after
        self.reasoning = list(reasoning or [])
        self.after_reasoning = list(after_reasoning or [])
        self.calls = 0

    def _next(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.answer

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> str:
        return self._next()

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> str:
        return self._next()

    def stream(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Iterator[Any]:
        answer = self._next()
        yield from self.reasoning
        if self.after_reasoning:
            raise self.after_reasoning.pop(0)
        for i, word in enumerate(answer.split(" ")):
            if self.fail_stream_after is not None and i >= self.fail_stream_after:
                raise _model_error()
            yield word + " "

    async def astream(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        for chunk in self.stream(input, config, **kwargs):
            yield chunk


class _FakeBedrockFactory(BedrockLanguageModelFactory):
    """Bedrock factory without a boto client, returning one scripted model."""

    def __init__(self, model: _ScriptedModel, config: Config) -> None:
        self.config = config
        self.model = model

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        return self.model

    def get_model_info(self, model_id: Any) -> Any:
        return fake_model_info(model_id)


def _config(retry: RetryPolicy = _NO_WAIT) -> Config:
    config = Config()
    config.retry = RetryConfig(model=retry)
    return config


def _query_chain(
    model: _ScriptedModel,
    retry: RetryPolicy = _NO_WAIT,
    purpose: ModelPurpose = ModelPurpose.QUERY,
) -> Runnable:
    return setup_chain(
        factory=_FakeBedrockFactory(model, _config(retry)),
        model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
        prompt_class=StrategySelectionPrompt,
        parser=StrOutputParser(),
        model_purpose=purpose,
    )


_QUERY = {
    "query": "Which parts does Vendor ship?",
    "strategies": "local, mix",
    "strategy_descriptions": "1. LOCAL SEARCH\n\n2. MIX SEARCH",
}


# --- wrapper behaviour ----------------------------------------------------


def test_transient_model_error_is_retried_sync() -> None:
    model = _ScriptedModel([_model_error()])
    assert _query_chain(model).invoke(_QUERY) == _ANSWER
    assert model.calls == 2


async def test_transient_model_error_is_retried_async() -> None:
    model = _ScriptedModel([_model_error()])
    assert await _query_chain(model).ainvoke(_QUERY) == _ANSWER
    assert model.calls == 2


async def test_non_transient_error_propagates_after_one_call() -> None:
    error = _validation_error()
    model = _ScriptedModel([error])
    with pytest.raises(ClientError) as excinfo:
        await _query_chain(model).ainvoke(_QUERY)
    assert excinfo.value is error
    assert model.calls == 1


async def test_exhausted_retries_reraise_last_transient_error() -> None:
    errors = [_model_error() for _ in range(5)]
    model = _ScriptedModel(list(errors))
    with pytest.raises(ClientError) as excinfo:
        await _query_chain(model).ainvoke(_QUERY)
    assert model.calls == _NO_WAIT.max_attempts
    assert excinfo.value is errors[_NO_WAIT.max_attempts - 1]


async def test_retry_budget_bounds_attempts() -> None:
    # A backoff that would cross the wall-clock budget is not taken.
    retry = RetryPolicy(
        max_attempts=5,
        base_delay_seconds=30.0,
        max_delay_seconds=30.0,
        max_total_seconds=1.0,
    )
    model = _ScriptedModel([_model_error()])
    with pytest.raises(ClientError):
        await _query_chain(model, retry).ainvoke(_QUERY)
    assert model.calls == 1


def test_batch_retries_each_input_independently() -> None:
    model = _ScriptedModel([_model_error()])
    assert _query_chain(model).batch([_QUERY, _QUERY]) == [_ANSWER, _ANSWER]
    assert model.calls == 3


async def test_stream_retries_before_first_chunk() -> None:
    model = _ScriptedModel([_model_error()])
    chunks = [c async for c in _query_chain(model).astream(_QUERY)]
    assert "".join(chunks).strip() == _ANSWER
    assert model.calls == 2


async def test_stream_does_not_retry_after_output_was_emitted() -> None:
    model = _ScriptedModel(fail_stream_after=2)
    chunks: list[str] = []
    with pytest.raises(ClientError):
        async for chunk in _query_chain(model).astream(_QUERY):
            chunks.append(chunk)
    assert len(chunks) == 2
    assert model.calls == 1


def test_sync_stream_does_not_retry_after_output_was_emitted() -> None:
    model = _ScriptedModel(fail_stream_after=1)
    with pytest.raises(ClientError):
        list(_query_chain(model).stream(_QUERY))
    assert model.calls == 1


# Chunks a stream sends before any answer text: they reach the caller
# emptied (or empty), so a retry after them repeats no output.
_TEXTLESS_STARTS = {
    # Ollama's native thinking: the reasoning is not in the content.
    "native": [AIMessageChunk(content="", additional_kwargs={"reasoning_content": "a"})]
    * 3,
    # A reasoning block in the answer, emptied by the reasoning stripper.
    "inline": ["<think>", "a ", "</think>\n"],
    # A server's empty first chunk.
    "empty": [""],
}


@pytest.mark.parametrize("start", sorted(_TEXTLESS_STARTS))
async def test_stream_retries_after_chunks_without_answer_text(start: str) -> None:
    # Regression (rv43 query medium): reasoning chunks pass on emptied (so a
    # stream's first-chunk window ends on them), and the retry took them for
    # output: a transient error while the model reasoned failed the stream,
    # though ainvoke retried it.
    model = _ScriptedModel(
        reasoning=_TEXTLESS_STARTS[start], after_reasoning=[_model_error()]
    )
    chunks = [c async for c in _query_chain(model).astream(_QUERY)]
    assert "".join(chunks).strip() == _ANSWER
    assert model.calls == 2


@pytest.mark.parametrize("start", sorted(_TEXTLESS_STARTS))
def test_sync_stream_retries_after_chunks_without_answer_text(start: str) -> None:
    model = _ScriptedModel(
        reasoning=_TEXTLESS_STARTS[start], after_reasoning=[_model_error()]
    )
    assert "".join(_query_chain(model).stream(_QUERY)).strip() == _ANSWER
    assert model.calls == 2


def test_retry_disabled_returns_plain_chain() -> None:
    chain = RunnableLambda(lambda x: x)
    assert with_transient_retry(chain, operation="op", retry=None) is chain
    off = RetryPolicy(max_attempts=1)
    assert with_transient_retry(chain, operation="op", retry=off) is chain
    assert not isinstance(_query_chain(_ScriptedModel(), off), TransientRetryRunnable)


@pytest.mark.parametrize(
    ("purpose", "wrapped"),
    [
        (ModelPurpose.QUERY, True),
        (ModelPurpose.INGESTION, False),
        (ModelPurpose.EVALUATION, False),
    ],
)
def test_retry_is_derived_from_purpose(purpose: ModelPurpose, wrapped: bool) -> None:
    chain = _query_chain(_ScriptedModel(), purpose=purpose)
    assert isinstance(chain, TransientRetryRunnable) is wrapped


def test_non_bedrock_factory_gets_no_chain_retry() -> None:
    class _PortOnlyFactory:
        def get_model(self, model_id: Any, **kwargs: Any) -> Any:
            return _ScriptedModel()

        def get_model_info(self, model_id: Any) -> Any:
            return None

    chain = setup_chain(
        factory=_PortOnlyFactory(),
        model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
        prompt_class=StrategySelectionPrompt,
        parser=StrOutputParser(),
    )
    assert not isinstance(chain, TransientRetryRunnable)


# --- end-to-end through GraphRAGChain -------------------------------------


def _rag_chain(model: _ScriptedModel, config: Config | None = None) -> GraphRAGChain:
    config = config or _config()
    retriever = StaticRetriever(
        [hit("Vendor ships parts to Buyer under the supply agreement.")]
    )
    chain = GraphRAGChain(
        config=config,
        model_factory=_FakeBedrockFactory(model, config),
        retriever_builders={
            RetrieverRole.DOCUMENT: lambda: retriever,
            RetrieverRole.GRAPH: lambda: retriever,
        },
    )
    chain.token_manager.count_tokens = lambda text: len((text or "").split())
    return chain


_RAG_INPUT = RAGInput(
    query="Which parts does Vendor ship?",
    search_strategy=SearchStrategy.SIMPLE,
    enable_query_processing=False,
)


async def test_rag_query_survives_one_transient_model_error() -> None:
    model = _ScriptedModel([_model_error()])
    out = await _rag_chain(model).ainvoke(_RAG_INPUT)
    assert isinstance(out, RAGOutput)
    assert out.answer == _ANSWER
    assert model.calls == 2


async def test_rag_query_non_transient_error_fails_after_one_call() -> None:
    model = _ScriptedModel([_validation_error()])
    with pytest.raises(ClientError):
        await _rag_chain(model).ainvoke(_RAG_INPUT)
    assert model.calls == 1


# --- ingestion path is not double-wrapped ---------------------------------


def test_batch_processor_chain_is_not_double_retried() -> None:
    # Ingestion chains get no chain-level retry; BatchProcessor's tenacity
    # retry is the only one, so max_attempts N => N model calls in total (not
    # 3 per attempt).
    model = _ScriptedModel([_model_error() for _ in range(10)])
    chain = _query_chain(model, purpose=ModelPurpose.INGESTION)
    processor = BatchProcessor(
        retry=RetryPolicy(max_attempts=2, base_delay_seconds=1.0, max_delay_seconds=0),
    )

    results = processor.execute_with_fallback(
        items_to_process=["item"],
        prepare_inputs_func=lambda items: [_QUERY for _ in items],
        sequential_func=chain.invoke,
        task_name="synthetic_ingestion",
        show_progress=False,
    )
    assert model.calls == 2
    assert results == [BATCH_ITEM_FAILED]


class _RootRunNames(BaseCallbackHandler):
    def __init__(self) -> None:
        self.names: list[str | None] = []

    def on_chain_start(
        self, serialized: Any, inputs: Any, *, parent_run_id: Any = None, **kw: Any
    ) -> None:
        if parent_run_id is None:
            self.names.append(kw.get("name"))


@pytest.mark.parametrize("purpose", [ModelPurpose.QUERY, ModelPurpose.INGESTION])
async def test_chain_run_is_named_after_its_prompt(purpose: ModelPurpose) -> None:
    names = _RootRunNames()
    chain = _query_chain(_ScriptedModel(), purpose=purpose)
    await chain.ainvoke(_QUERY, {"callbacks": [names]})
    assert names.names == ["StrategySelectionPrompt"]


# --- a stream's first-chunk window bounds its retries -----------------------

# The window, the backoff and the query deadline (and the retry budget) sit
# an order of magnitude apart: the backoff crosses the window but neither
# the deadline nor the budget, so only the window can refuse the retry; and
# the window is far beyond any stall a loaded runner puts between the
# stream's start and its model call (a 1 s window expired there on CI,
# before the model was reached).
_WINDOW_SECONDS = 60.0
_LONG_BACKOFF = RetryPolicy(
    max_attempts=3,
    base_delay_seconds=600.0,
    max_delay_seconds=600.0,
    max_total_seconds=6000.0,
)


async def _consume(chain: GraphRAGChain, surface: str) -> None:
    if surface == "astream":
        async for _ in chain.astream(_RAG_INPUT):
            pass
    else:
        async for _ in chain.aquery_events(_RAG_INPUT):
            pass


@pytest.mark.parametrize("surface", ["astream", "aquery_events"])
async def test_a_retry_past_the_first_chunk_window_raises_the_model_error(
    surface: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression (rv33): the first-chunk window moved only the asyncio
    # timeout, so a backoff that crossed it was taken and the stream failed
    # as a timeout. No retry starts that cannot reach a first chunk in time:
    # the model's error is raised, at once, as unstreamed.
    real_sleep = asyncio.sleep
    backoffs: list[float] = []

    async def sleep(delay: float, *args: Any, **kwargs: Any) -> Any:
        # A backoff taken is recorded and skipped: a retry the window should
        # have refused then answers at once instead of waiting it out.
        if delay > 0:
            backoffs.append(delay)
            delay = 0
        return await real_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    config = _config(_LONG_BACKOFF)
    config.search.query_timeout_seconds = 6000
    config.search.model_timeout_seconds = _WINDOW_SECONDS
    config.providers.rerank.kind = "none"
    model = _ScriptedModel([_model_error()])
    chain = _rag_chain(model, config)
    with caplog.at_level("WARNING"), pytest.raises(ClientError):
        await asyncio.wait_for(_consume(chain, surface), timeout=30)
    assert (model.calls, backoffs) == (1, [])
    # The give-up names the window's setting, not the query deadline (rv34).
    (gave_up,) = [
        r.getMessage() for r in caplog.records if "Giving up" in r.getMessage()
    ]
    assert "first-chunk window (search.model_timeout_seconds)" in gave_up
    assert "query deadline" not in gave_up


@pytest.mark.parametrize("surface", ["astream", "aquery_events"])
@pytest.mark.parametrize("start", sorted(_TEXTLESS_STARTS))
async def test_a_retry_after_reasoning_is_not_bounded_by_the_window(
    surface: str, start: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression (rv43 query medium): the model's first chunk, its reasoning,
    # ends the first-chunk window, but LangChain runs the stream in a copy of
    # the context the window was open in, so the window's retry deadline
    # outlived it there: a transient error while the model reasoned past the
    # window was not retried. The query deadline alone now bounds that retry.
    real_sleep = asyncio.sleep
    backoffs: list[float] = []

    async def sleep(delay: float, *args: Any, **kwargs: Any) -> Any:
        if delay > 0:
            backoffs.append(delay)
            delay = 0
        return await real_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    config = _config(_LONG_BACKOFF)
    config.search.query_timeout_seconds = 6000
    config.search.model_timeout_seconds = _WINDOW_SECONDS
    config.providers.rerank.kind = "none"
    model = _ScriptedModel(
        reasoning=_TEXTLESS_STARTS[start], after_reasoning=[_model_error()]
    )
    chain = _rag_chain(model, config)
    if surface == "astream":
        answer = "".join(
            [c async for c in chain.astream(_RAG_INPUT.model_copy(deep=True))]
        )
    else:
        answer = "".join(
            [
                event.text
                async for event in chain.aquery_events(_RAG_INPUT.model_copy(deep=True))
                if isinstance(event, AnswerChunk)
            ]
        )
    assert answer.strip() == _ANSWER
    assert (model.calls, len(backoffs)) == (2, 1)


# --- an output repair's failure does not re-run the answered call -----------


class _FailingFixerFactory:
    """A factory whose fixing model fails with a transient error."""

    is_transient_error = staticmethod(BedrockLanguageModelFactory.is_transient_error)

    def __init__(self) -> None:
        self.calls = 0

    def get_model(self, **kwargs: object) -> object:
        def fail(_: object) -> object:
            self.calls += 1
            raise _model_error()

        return RunnableLambda(fail)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_a_failed_repair_leaves_the_answer_unrepaired(asynchronous: bool) -> None:
    # rv33: the JSON parser runs inside the transient retry, so a transient
    # error of the fixing model re-ran the whole chain, paying again for
    # the model call that had answered (and, under a call limit, taking the
    # answer's). A repair is optional: its failure leaves the answer as it
    # is, unreadable.
    from langchain_core.exceptions import OutputParserException

    from unified_kg_rag.adapters.llm.chain_factory import create_json_output_parser

    model = _ScriptedModel(answer="no JSON here")
    fixer = _FailingFixerFactory()
    chain = setup_chain(
        factory=_FakeBedrockFactory(model, _config()),
        model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
        prompt_class=StrategySelectionPrompt,
        parser=create_json_output_parser(
            fixer, "fixer", answer_keys=("strategy",)  # type: ignore[arg-type]
        ),
    )
    assert isinstance(chain, TransientRetryRunnable)
    with pytest.raises(OutputParserException) as raised:
        if asynchronous:
            await chain.ainvoke(_QUERY)
        else:
            chain.invoke(_QUERY)
    assert (model.calls, fixer.calls) == (1, 1)
    assert isinstance(raised.value.__cause__, ClientError)


# --- a Bedrock stream's own failures are transient --------------------------


def _event_stream_error(code: str) -> EventStreamError:
    # botocore codes an event stream's error by the event's member name.
    return EventStreamError(
        {"Error": {"Code": code, "Message": "synthetic failure"}}, "ConverseStream"
    )


def _bedrock_reasoning() -> list[AIMessageChunk]:
    # ChatBedrockConverse's reasoning: content blocks without answer text.
    block = {"type": "reasoning_content", "reasoning_content": {"text": "a "}}
    return [AIMessageChunk(content=[dict(block, index=0)]) for _ in range(3)]


# What a stream raises after its response started, none of which botocore
# retries: an error event (ConverseStream's and InvokeModelWithResponseStream's
# members), or urllib3's own error while the events are read.
_STREAM_FAILURES = {
    "throttling": lambda: _event_stream_error("throttlingException"),
    "internal": lambda: _event_stream_error("internalServerException"),
    "unavailable": lambda: _event_stream_error("serviceUnavailableException"),
    "stream-error": lambda: _event_stream_error("modelStreamErrorException"),
    "model-timeout": lambda: _event_stream_error("modelTimeoutException"),
    "dropped": lambda: urllib3_exc.ProtocolError("Connection broken"),
    "stalled": lambda: urllib3_exc.ReadTimeoutError(None, "/", "Read timed out."),
    # langchain-aws's for a stream that ended without its messageStop event.
    "incomplete": lambda: ConnectionError(
        "Incomplete Bedrock response stream: missing messageStop event."
    ),
}


@pytest.mark.parametrize("failure", sorted(_STREAM_FAILURES))
@pytest.mark.parametrize("surface", ["astream", "aquery_events"])
async def test_a_bedrock_stream_failing_after_reasoning_is_retried(
    failure: str, surface: str
) -> None:
    # Regression (rv44 query medium): an error event's code is camelCase
    # (throttlingException), and urllib3's errors reach the caller as they
    # are; neither was transient, so a stream that failed while Claude
    # reasoned was never retried, though ainvoke's call was.
    config = _config()
    config.providers.rerank.kind = "none"
    model = _ScriptedModel(
        reasoning=_bedrock_reasoning(),
        after_reasoning=[_STREAM_FAILURES[failure]()],
    )
    chain = _rag_chain(model, config)
    if surface == "astream":
        answer = "".join([c async for c in chain.astream(_RAG_INPUT)])
    else:
        answer = "".join(
            [
                event.text
                async for event in chain.aquery_events(_RAG_INPUT)
                if isinstance(event, AnswerChunk)
            ]
        )
    assert answer.strip() == _ANSWER
    assert model.calls == 2


@pytest.mark.parametrize("failure", ["throttling", "incomplete"])
async def test_a_bedrock_stream_failing_after_answer_text_is_not_retried(
    failure: str,
) -> None:
    # The text reached the caller: a retry would repeat it.
    config = _config()
    config.providers.rerank.kind = "none"
    text = AIMessageChunk(content=[{"type": "text", "text": "PARTIAL ", "index": 1}])
    error = _STREAM_FAILURES[failure]()
    model = _ScriptedModel(
        reasoning=[*_bedrock_reasoning(), text], after_reasoning=[error]
    )
    chunks: list[str] = []
    # A model client's error the package does not name is raised as it is,
    # else as the LanguageModelError raised from it.
    with pytest.raises(Exception) as raised:
        async for chunk in _rag_chain(model, config).astream(_RAG_INPUT):
            chunks.append(chunk)
    assert error in (raised.value, raised.value.__cause__)
    assert "".join(chunks).count("PARTIAL") == 1
    assert model.calls == 1


@pytest.mark.parametrize(
    "code", ["validationException", "accessDeniedException", "ValidationException"]
)
async def test_a_bedrock_stream_s_client_error_event_is_not_retried(code: str) -> None:
    error = _event_stream_error(code)
    model = _ScriptedModel(reasoning=_bedrock_reasoning(), after_reasoning=[error])
    with pytest.raises(EventStreamError) as raised:
        async for _ in _query_chain(model).astream(_QUERY):
            pass
    assert raised.value is error
    assert model.calls == 1


# --- what counts as output a retry would repeat -----------------------------


class _StreamThenFail(Runnable[Any, Any]):
    """Streams ``chunks``, then fails with a transient error, once."""

    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self.calls = 0

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    async def astream(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        self.calls += 1
        for chunk in self.chunks:
            yield chunk
        if self.calls == 1:
            raise _model_error()


@pytest.mark.parametrize(
    "chunk",
    [
        # A structured parser's value: content that is not text.
        {"strategy": "local"},
        # Whitespace is text the caller received.
        " ",
        AIMessageChunk(content="\n"),
    ],
    ids=["structured", "whitespace", "whitespace-message"],
)
async def test_a_stream_is_not_retried_after_any_output(chunk: Any) -> None:
    bound = _StreamThenFail([chunk])
    chain = with_transient_retry(
        bound,
        operation="op",
        retry=_NO_WAIT,
        is_transient=BedrockLanguageModelFactory.is_transient_error,
    )
    with pytest.raises(ClientError):
        async for _ in chain.astream(_QUERY):
            pass
    assert bound.calls == 1


@pytest.mark.parametrize(
    "chunk", ["", AIMessageChunk(content="")], ids=["empty", "empty-message"]
)
async def test_a_stream_is_retried_after_empty_chunks(chunk: Any) -> None:
    bound = _StreamThenFail([chunk])
    chain = with_transient_retry(
        bound,
        operation="op",
        retry=_NO_WAIT,
        is_transient=BedrockLanguageModelFactory.is_transient_error,
    )
    assert [c async for c in chain.astream(_QUERY)] == [chunk, chunk]
    assert bound.calls == 2
