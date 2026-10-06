# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reranking on a shared model is safe for concurrent queries (AWS-free).

Regression: the scorer (and the Bedrock rerank wrapper) narrowed ``top_n`` by
mutating the shared rerank model and restoring it afterwards. With strategies
reused across queries and fusion running in worker threads, one query's limit
could leak into another's rerank. ``top_n`` is now applied to a per-call copy.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from pydantic import BaseModel

from unified_kg_rag.adapters.aws.bedrock import BedrockRerankWrapper
from unified_kg_rag.adapters.retrieval.hybrid_scorer import HybridScorer, _with_top_n
from unified_kg_rag.domain.models import Config, RetrievalResult

pytestmark = pytest.mark.unit


class _RecordingReranker(BaseModel):
    """Pydantic stand-in for a LangChain document compressor."""

    top_n: int | None = 10
    seen: list[int | None] = []

    def compress_documents(self, documents: Any, query: str) -> list[Document]:
        # Overlap the two calls so a shared mutation would be observed.
        time.sleep(0.05 if query == "slow" else 0.0)
        self.seen.append(self.top_n)
        limit = self.top_n or len(documents)
        return [
            Document(page_content=d.page_content, metadata={**d.metadata})
            for d in documents[:limit]
        ]


def _results(prefix: str, n: int) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            content=f"{prefix} {i}",
            score=1.0 - i / 100,
            source=f"{prefix}-{i}",
            retriever_type="text",
        )
        for i in range(n)
    ]


def _scorer(model: Any) -> HybridScorer:
    config = Config()
    config.search.reranking.enabled = False
    config.search.fusion.diversity_lambda = 1.0
    scorer = HybridScorer(config)
    scorer.rerank_model = model
    return scorer


def test_with_top_n_copies_instead_of_mutating() -> None:
    model = _RecordingReranker(top_n=10)
    narrowed = _with_top_n(model, 3)
    assert narrowed is not model
    assert narrowed.top_n == 3 and model.top_n == 10
    assert _with_top_n(model, 10) is model

    class _Plain:
        top_n = 10

    plain = _Plain()
    clone = _with_top_n(plain, 4)
    assert clone.top_n == 4 and plain.top_n == 10


async def test_concurrent_reranks_do_not_share_top_n() -> None:
    model = _RecordingReranker(top_n=10, seen=[])
    scorer = _scorer(model)

    slow, fast = await asyncio.gather(
        asyncio.to_thread(
            scorer.fuse_and_rerank_results,
            {"a": _results("slow", 3)},
            top_k=10,
            query="slow",
        ),
        asyncio.to_thread(
            scorer.fuse_and_rerank_results,
            {"a": _results("fast", 7)},
            top_k=10,
            query="fast",
        ),
    )

    assert len(slow) == 3 and len(fast) == 7
    assert sorted(model.seen, key=lambda n: n or 0) == [3, 7]
    assert model.top_n == 10  # the shared model was never narrowed


async def test_strategy_fusion_runs_off_the_event_loop() -> None:
    from unified_kg_rag.adapters.providers import Providers
    from unified_kg_rag.adapters.search_strategies import SimpleSearchStrategy

    config = Config()
    config.search.reranking.enabled = False
    providers = Providers(
        config,
        boto_session=MagicMock(),
        token_counter_factory=lambda *_, **__: MagicMock(),
    )
    strategy = SimpleSearchStrategy(config=config, retrievers={}, providers=providers)
    seen: list[threading.Thread] = []

    def fuse(*args: Any, **kwargs: Any) -> list[RetrievalResult]:
        seen.append(threading.current_thread())
        return []

    strategy.hybrid_scorer.fuse_and_rerank_results = fuse  # type: ignore[method-assign]
    await strategy._fuse_and_rerank({}, top_k=1)
    assert seen and seen[0] is not threading.current_thread()


def test_bedrock_rerank_wrapper_clamps_on_a_copy(mocker) -> None:
    from langchain_aws import BedrockRerank

    seen: list[int | None] = []

    def fake_compress(self: Any, documents: Any, query: str, callbacks: Any = None):
        seen.append(self.top_n)
        return list(documents)

    mocker.patch.object(BedrockRerank, "compress_documents", fake_compress)
    wrapper = BedrockRerankWrapper(
        model_arn="arn:aws:bedrock:us-east-1::foundation-model/synthetic-rerank",
        top_n=5,
        region_name="us-east-1",
        client=MagicMock(),
    )

    out = wrapper.compress_documents(
        [Document(page_content="a"), Document(page_content="b")], "q"
    )

    assert len(out) == 2
    assert seen == [2]
    assert wrapper.top_n == 5
