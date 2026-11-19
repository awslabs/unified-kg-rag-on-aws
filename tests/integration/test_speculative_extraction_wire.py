# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""auto's speculative extraction its route did not read fills the
extraction cache over real HTTP, as it outlives its query.

Regression: the query's model-response scope closed the extraction's
response as the query ended (it is kept running on purpose), and closed
every response it opened afterwards as its headers arrived: the extraction
failed with ``ReadError``, was retried and failed again, and the next query
that reads it extracted it once more. Over each provider's wire (Ollama:
the console demo's fake model server; an OpenAI-compatible server), the
extraction's body arrives a while after its headers, past the end of the
query that started it: the server must see one extraction request, and
the next ``mix`` query must take it from the cache.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from tests.fixtures.console_demo import load
from tests.unit.test_speculative_usage import _Retriever
from unified_kg_rag.adapters.local.models import (
    OllamaLanguageModelFactory,
    OpenAICompatibleLanguageModelFactory,
)
from unified_kg_rag.adapters.local.token_counter import local_token_counter_factory
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, RAGInput
from unified_kg_rag.domain.models import Config, RetrieverRole, SearchStrategy

pytestmark = [pytest.mark.integration]

# Seconds the extraction's body arrives after its headers: past the end of
# the query that started it.
BODY_DELAY = 1.0
_KEYWORDS = "expert keyword extractor"


class _Wire:
    """The loopback server's state: the route its router answers, and the
    keyword-extraction requests it got and answered to their end."""

    def __init__(self) -> None:
        self.route = "simple"
        self.extractions = 0
        self.answered = 0
        self.lock = threading.Lock()

    def count(self, name: str) -> None:
        with self.lock:
            setattr(self, name, getattr(self, name) + 1)


def _fake_models(monkeypatch: pytest.MonkeyPatch, wire: _Wire) -> Any:
    fake = load("fake_models")
    monkeypatch.setattr(fake, "route", lambda _question: wire.route)
    return fake


def _system(body: dict[str, Any]) -> str:
    return "".join(
        str(m.get("content", ""))
        for m in body.get("messages") or []
        if m.get("role") == "system"
    )


def _ollama(monkeypatch: pytest.MonkeyPatch, wire: _Wire) -> tuple[Any, str]:
    """The console demo's fake Ollama server; an extraction's body (its one
    NDJSON line) comes ``BODY_DELAY`` s after its headers."""
    fake = _fake_models(monkeypatch, wire)
    server, models = fake.serve(chunk_delay=0.0)
    chat = models.chat

    def delayed(body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        extraction = _KEYWORDS in _system(body)
        if extraction:
            wire.count("extractions")
            time.sleep(BODY_DELAY)
        yield from chat(body)
        if extraction:
            wire.count("answered")

    models.chat = delayed
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _openai(monkeypatch: pytest.MonkeyPatch, wire: _Wire) -> tuple[Any, str]:
    """An OpenAI-compatible server answering as the fake models do,
    unstreamed; an extraction's body comes ``BODY_DELAY`` s after its
    headers."""
    fake = _fake_models(monkeypatch, wire)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return None

        def do_POST(self) -> None:  # noqa: N802
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = json.loads(raw or b"{}")
            system = _system(body)
            human = "".join(
                str(m.get("content", ""))
                for m in body.get("messages") or []
                if m.get("role") == "user"
            )
            reply, _ = fake.respond(system, human)
            if body.get("stream"):
                self._stream(reply)
                return
            data = json.dumps(
                {
                    "id": "x",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model") or "fake",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": reply},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 5,
                        "total_tokens": 10,
                    },
                }
            ).encode()
            extraction = _KEYWORDS in system
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.flush()
            if extraction:
                wire.count("extractions")
                time.sleep(BODY_DELAY)
            try:
                self.wfile.write(data)
                self.wfile.flush()
            except OSError:
                return
            if extraction:
                wire.count("answered")

        def _stream(self, reply: str) -> None:
            """A streamed answer (the answer call's), in one SSE chunk."""
            chunk = {
                "id": "x",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "fake",
                "choices": [
                    {"index": 0, "delta": {"content": reply}, "finish_reason": "stop"}
                ],
            }
            data = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


def _chain(provider: str, url: str) -> GraphRAGChain:
    config = Config()
    config.search.auto_routable_strategies = [SearchStrategy.SIMPLE, SearchStrategy.MIX]
    # No rerank model: nothing here may build an AWS client.
    config.providers.rerank.kind = "none"
    for endpoint in (config.providers.llm, config.providers.embedding):
        endpoint.kind = "ollama" if provider == "ollama" else "openai_compatible"
        endpoint.base_url = url
    translation = config.processing.translation
    translation.source_language = translation.target_language
    factory = (
        OllamaLanguageModelFactory
        if provider == "ollama"
        else OpenAICompatibleLanguageModelFactory
    )
    retriever = _Retriever()
    providers = Providers(
        config,
        llm_factory=factory(config),
        token_counter_factory=local_token_counter_factory(config),
        retriever_builders={
            RetrieverRole.DOCUMENT: lambda: retriever,
            RetrieverRole.GRAPH: lambda: retriever,
        },
    )
    return GraphRAGChain(config=config, providers=providers)


async def _settled(chain: GraphRAGChain, within: float = 30.0) -> None:
    deadline = time.monotonic() + within
    while chain._background_extractions._snapshot():
        assert time.monotonic() < deadline, "the extraction never ended"
        await asyncio.sleep(0.05)


@pytest.mark.parametrize("events", [False, True], ids=["ainvoke", "events"])
@pytest.mark.parametrize("provider", ["ollama", "openai"])
async def test_the_unread_extraction_fills_the_cache_over_http(
    monkeypatch: pytest.MonkeyPatch, provider: str, events: bool
) -> None:
    wire = _Wire()
    server, url = (_ollama if provider == "ollama" else _openai)(monkeypatch, wire)
    chain = _chain(provider, url)
    query = RAGInput(query="What does Vendor supply?", search_strategy="auto")
    try:
        if events:
            _ = [event async for event in chain.aquery_events(query)]
        else:
            out = await chain.ainvoke(query)
            assert out.metadata["search_strategy"] == "simple"
        # The query did not wait for the extraction's body.
        assert wire.answered == 0
        await _settled(chain)
        assert (wire.extractions, wire.answered) == (1, 1)
        wire.route = "mix"
        out = await chain.ainvoke(query)
        assert out.metadata["search_strategy"] == "mix"
        assert wire.extractions == 1
        assert out.metadata["usage"]["cache_hits"] == 1
    finally:
        await chain.aclose()
        server.shutdown()
