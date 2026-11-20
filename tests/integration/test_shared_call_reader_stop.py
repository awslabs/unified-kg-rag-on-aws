# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One reader's stop ends no other reader's query.

Concurrent queries share their query embeddings (an in-flight cache,
``single_flight``): two questions that name the same entity embed it once.
On the embedded stores with the Ollama adapters, each embedding is refused
once (a 503, then a retry's backoff) and half the readers stop during it:
every reader that keeps reading gets its answer, whichever query started
the embedding it waits for.

The readers that stop start first, so they start the shared embeddings, and
stop once every query waits for one (cues, not clocks): earlier, a query
that has not joined it yet would start its own.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.fixtures.fakes.rerank import FakeCrossEncoder
from tests.integration import test_embedded_e2e as embedded
from unified_kg_rag import QueryResult
from unified_kg_rag.adapters.local.embeddings import OllamaEmbeddingFactory
from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory
from unified_kg_rag.adapters.local.rerank import CrossEncoderRerankFactory
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.bootstrap import build_providers
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, RAGInput
from unified_kg_rag.domain.models import Config, PipelineStageStatus, SearchStrategy
from unified_kg_rag.shared import query_stop
from unified_kg_rag.shared.utils import concurrency

pytestmark = pytest.mark.integration

# A hang backstop (seconds), far past anything a passing run waits.
_BACKSTOP = 60.0


class _RefusingEachEmbeddingOnce:
    """The model server, refusing the first request for each embedding
    input once ``refusing`` (after ingestion)."""

    def __init__(self) -> None:
        self.server = FakeOllama(embedded._respond, embedded.ANSWER, embedded.DIM)
        self.refusing = False
        self.seen: set[str] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.refusing and request.url.path == "/api/embed":
            key = json.dumps(json.loads(request.content)["input"])
            if key not in self.seen:
                self.seen.add(key)
                return httpx.Response(503, json={"error": "busy"})
        return self.server(request)


def _waiting(
    monkeypatch: pytest.MonkeyPatch, readers: int
) -> tuple[asyncio.Event, asyncio.Event]:
    """Set once a query waits for a shared call, and once ``readers``
    queries (told apart by their stops; a query's sub-searches share its
    calls too) wait for one."""
    started, joined = asyncio.Event(), asyncio.Event()
    queries: dict[int, set[int]] = {}
    shared = concurrency.await_shared

    async def counted(future: asyncio.Future[Any]) -> Any:
        waiting = queries.setdefault(id(future), set())
        waiting.add(id(query_stop._QUERY_STOP.get()))
        started.set()
        if len(waiting) >= readers:
            joined.set()
        return await shared(future)

    monkeypatch.setattr(concurrency, "await_shared", counted)
    return started, joined


@pytest.fixture(scope="module")
def ingested(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Config, Providers, _RefusingEachEmbeddingOnce]]:
    tmp = tmp_path_factory.mktemp("shared-stop")
    (tmp / "stores").mkdir()
    config = embedded._config(tmp / "stores", tmp / "viz")
    # Every refused embedding waits 2 s before its retry: the stops land
    # in that wait.
    policy = config.retry.model_server
    policy.base_delay_seconds = policy.max_delay_seconds = 2.0
    policy.jitter = "none"
    server = _RefusingEachEmbeddingOnce()
    transport = httpx.MockTransport(server)
    encoder = FakeCrossEncoder("synthetic-reranker")
    providers = build_providers(
        config,
        llm_factory=OllamaLanguageModelFactory(config, transport=transport),
        embedding_factory=OllamaEmbeddingFactory(config, transport=transport),
        rerank_factory=CrossEncoderRerankFactory(
            config, loader=lambda *_, **__: encoder
        ),
    )
    corpus = tmp / "corpus"
    corpus.mkdir()
    (corpus / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (corpus / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    context = embedded._ingest(config, providers, corpus, tmp / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, embedded._errors(context)
    yield config, providers, server


@pytest.mark.parametrize(
    ("readers", "questions"),
    [(2, 1), (16, 4), (16, 16)],
    ids=["one-question", "four-questions", "distinct-questions"],
)
async def test_a_stopped_reader_ends_no_other_readers_query(
    ingested: tuple[Config, Providers, _RefusingEachEmbeddingOnce],
    readers: int,
    questions: int,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, providers, server = ingested
    case = request.node.callspec.id
    # A chain of its own: no embedding cached by an earlier case.
    chain = GraphRAGChain(config, providers=providers)
    server.seen.clear()
    server.refusing = True
    started, joined = _waiting(monkeypatch, readers)

    async def read(question: str) -> Any:
        async for event in chain.aquery_events(
            RAGInput(query=question, search_strategy=SearchStrategy.LOCAL)
        ):
            if isinstance(event, QueryResult):
                return event.output
        return None

    async def reader(i: int) -> Any:
        if i % 2:
            # After a query that stops started the shared calls: theirs.
            await started.wait()
        task = asyncio.ensure_future(
            read(f"Who stores the gadgets for Vendor? #{case}-{i % questions}")
        )
        if i % 2:
            return await task
        # Stopped while the shared embeddings are refused and retried.
        await joined.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return "stopped"

    try:
        async with asyncio.timeout(_BACKSTOP):
            outcomes = await asyncio.gather(
                *(reader(i) for i in range(readers)), return_exceptions=True
            )
    finally:
        server.refusing = False
        await chain.aclose()
    # The shared embeddings (the entities every question names) were
    # refused and retried.
    assert {'["Depot"]', '["Vendor"]'} <= server.seen
    reading = outcomes[1::2]
    assert all(not isinstance(o, BaseException) for o in reading), reading
    assert all(o is not None and not o.metadata.get("error") for o in reading)
    assert all(o.answer == embedded.ANSWER for o in reading)
    assert outcomes[0::2] == ["stopped"] * (readers // 2)
