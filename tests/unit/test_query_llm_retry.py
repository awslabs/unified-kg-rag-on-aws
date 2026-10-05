# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the transient-error retry on query-time LLM chains.

AWS-free: chains are built by ``setup_chain`` from a fake model factory whose
model raises scripted botocore ``ClientError`` s shaped like Bedrock's HTTP 424
``ModelErrorException`` before answering. Backoff delays are configured to zero
so no test waits.
"""

from __future__ import annotations

import ast
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
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
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.domain.models.config import QueryLLMRetryConfig
from unified_kg_rag.domain.prompts import StrategySelectionPrompt
from unified_kg_rag.shared.utils.langchain import BatchProcessor

pytestmark = pytest.mark.unit

_ANSWER = "Vendor ships parts to Buyer every quarter."
_NO_WAIT = QueryLLMRetryConfig(
    max_attempts=3, base_delay_seconds=0.0, max_delay_seconds=0.0
)
_PACKAGE = Path(__file__).resolve().parents[2] / "unified_kg_rag"


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


class _FakeFactory:
    """LLMFactoryPort returning one shared scripted model."""

    def __init__(self, model: _ScriptedModel) -> None:
        self.model = model

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        return self.model

    def get_model_info(self, model_id: Any) -> Any:
        return None


def _query_chain(
    model: _ScriptedModel, retry: QueryLLMRetryConfig | None = _NO_WAIT
) -> Runnable:
    return setup_chain(
        factory=_FakeFactory(model),
        model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
        prompt_class=StrategySelectionPrompt,
        parser=StrOutputParser(),
        retry=retry,
    )


_QUERY = {"query": "Which parts does Vendor ship?"}


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
    retry = QueryLLMRetryConfig(
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
    off = QueryLLMRetryConfig(max_attempts=1)
    assert with_transient_retry(chain, operation="op", retry=off) is chain
    assert not isinstance(_query_chain(_ScriptedModel(), None), TransientRetryRunnable)


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
    config = Config()
    config.search.llm_retry = _NO_WAIT
    retriever = _FakeRetriever()
    chain = GraphRAGChain(
        config=config,
        mode=ChainMode.RAG,
        model_factory=_FakeFactory(model),
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
    # Ingestion chains are built without ``retry``; BatchProcessor's tenacity
    # retry is the only one, so N attempts => N model calls (not N * 3).
    model = _ScriptedModel([_model_error() for _ in range(10)])
    chain = _query_chain(model, retry=None)
    processor = BatchProcessor(
        batch_size=1, max_retries=2, retry_multiplier=1.0, retry_max_wait=0
    )

    def batch_func(inputs: list[dict[str, Any]], config: Any = None) -> list[Any]:
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
    assert results == [{}]


def _setup_chain_calls(path: Path) -> list[tuple[str, ast.Call]]:
    """Return (assigned attribute name, call) for each ``setup_chain`` call."""
    tree = ast.parse(path.read_text())
    calls: list[tuple[str, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign | ast.AnnAssign | ast.Return):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "setup_chain"
        ):
            continue
        target = ""
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if targets and isinstance(targets[0], ast.Attribute | ast.Name):
            first = targets[0]
            target = first.attr if isinstance(first, ast.Attribute) else first.id
        calls.append((target, value))
    return calls


def _has_retry(call: ast.Call) -> bool:
    return any(kw.arg == "retry" for kw in call.keywords)


@pytest.mark.parametrize(
    "path", sorted((_PACKAGE / "adapters" / "ingestion").glob("*.py")), ids=str
)
def test_ingestion_chains_do_not_add_query_retry(path: Path) -> None:
    for target, call in _setup_chain_calls(path):
        assert not _has_retry(call), f"{path.name}:{target} stacks on BatchProcessor"


@pytest.mark.parametrize(
    ("relative", "unwrapped"),
    [
        ("application/retrieval/rag_chain.py", set()),
        ("adapters/retrieval/memory_manager.py", set()),
        ("adapters/search_strategies/drift_search.py", set()),
        # map_rater runs under BatchProcessor, which already retries it.
        ("adapters/search_strategies/global_search.py", {"map_rater"}),
    ],
)
def test_query_chains_use_transient_retry(relative: str, unwrapped: set[str]) -> None:
    calls = _setup_chain_calls(_PACKAGE / relative)
    assert calls
    for target, call in calls:
        assert _has_retry(call) is (target not in unwrapped), f"{relative}:{target}"
