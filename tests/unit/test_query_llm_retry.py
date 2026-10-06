# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the transient-error retry on query-time LLM chains.

AWS-free: chains are built by ``setup_chain`` from a Bedrock factory stand-in
(no boto client) whose model raises scripted botocore ``ClientError`` s shaped
like Bedrock's HTTP 424 ``ModelErrorException`` before answering. Backoff
delays are configured to zero so no test waits.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from botocore.exceptions import ClientError
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from unified_kg_rag.adapters.aws.bedrock import BedrockLanguageModelFactory
from unified_kg_rag.adapters.aws.chain_factory import (
    TransientRetryRunnable,
    setup_chain,
    with_transient_retry,
)
from unified_kg_rag.application.retrieval.rag_chain import (
    ChainMode,
    GraphRAGChain,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.models import (
    Config,
    LanguageModelId,
    ModelPurpose,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.domain.models.config import TransientRetryConfig
from unified_kg_rag.domain.prompts import StrategySelectionPrompt
from unified_kg_rag.shared.utils.langchain import BATCH_ITEM_FAILED, BatchProcessor

pytestmark = pytest.mark.unit

_ANSWER = "Vendor ships parts to Buyer every quarter."
_NO_WAIT = TransientRetryConfig(
    max_attempts=3, base_delay_seconds=0.0, max_delay_seconds=0.0
)


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
    chunks, to exercise the no-retry-after-output rule.
    """

    def __init__(
        self,
        errors: list[BaseException] | None = None,
        *,
        answer: str = _ANSWER,
        fail_stream_after: int | None = None,
    ) -> None:
        self.errors = list(errors or [])
        self.answer = answer
        self.fail_stream_after = fail_stream_after
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
    ) -> Iterator[str]:
        answer = self._next()
        for i, word in enumerate(answer.split(" ")):
            if self.fail_stream_after is not None and i >= self.fail_stream_after:
                raise _model_error()
            yield word + " "

    async def astream(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
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
        return None


def _config(retry: TransientRetryConfig = _NO_WAIT) -> Config:
    config = Config()
    config.aws.bedrock.transient_retry = retry
    return config


def _query_chain(
    model: _ScriptedModel,
    retry: TransientRetryConfig = _NO_WAIT,
    purpose: ModelPurpose = ModelPurpose.QUERY,
) -> Runnable:
    return setup_chain(
        factory=_FakeBedrockFactory(model, _config(retry)),
        model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
        prompt_class=StrategySelectionPrompt,
        parser=StrOutputParser(),
        model_purpose=purpose,
    )


_QUERY = {"query": "Which parts does Vendor ship?", "strategies": "local, mix"}


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
    retry = TransientRetryConfig(
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


def test_retry_disabled_returns_plain_chain() -> None:
    chain = RunnableLambda(lambda x: x)
    assert with_transient_retry(chain, operation="op", retry=None) is chain
    off = TransientRetryConfig(max_attempts=1)
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


class _FakeRetriever:
    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                content="Vendor ships parts to Buyer under the supply agreement.",
                score=0.9,
                source="doc-1",
                retriever_type="document",
                metadata={"id": "doc-1", "text_unit_ids": []},
            )
        ]


def _rag_chain(model: _ScriptedModel) -> GraphRAGChain:
    config = _config()
    retriever = _FakeRetriever()
    chain = GraphRAGChain(
        config=config,
        mode=ChainMode.RAG,
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
    # retry is the only one, so N attempts => N model calls (not N * 3).
    model = _ScriptedModel([_model_error() for _ in range(10)])
    chain = _query_chain(model, purpose=ModelPurpose.INGESTION)
    processor = BatchProcessor(
        batch_size=1, max_attempts=2, retry_multiplier=1.0, retry_max_wait=0
    )

    def batch_func(inputs: list[dict[str, Any]], **_: Any) -> list[Any]:
        raise RuntimeError("force the per-item sequential path")

    results = processor.execute_with_fallback(
        items_to_process=["item"],
        prepare_inputs_func=lambda items: [_QUERY for _ in items],
        batch_func=batch_func,
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
