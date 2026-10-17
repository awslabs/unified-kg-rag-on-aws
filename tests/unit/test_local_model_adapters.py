# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local model adapters against fake HTTP servers (no network, no models).

The OpenAI-compatible and Ollama factories are driven through an
``httpx.MockTransport`` that answers like the real servers, so the tests see
the exact requests the adapters send (model, limits, num_ctx) and parse real
response shapes. The cross-encoder is a fake loader; tiktoken is not loaded.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from unified_kg_rag.adapters.llm import retry as retry_module
from unified_kg_rag.adapters.llm.chain_factory import (
    TransientRetryRunnable,
    setup_chain,
)
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.local.embeddings import (
    OllamaEmbeddingFactory,
    OpenAICompatibleEmbeddingFactory,
    truncate_to_estimate,
)
from unified_kg_rag.adapters.local.models import (
    PLACEHOLDER_API_KEY,
    OllamaLanguageModelFactory,
    OpenAICompatibleLanguageModelFactory,
    PerLoopAsyncClient,
    import_optional,
)
from unified_kg_rag.adapters.local.rerank import (
    CrossEncoderReranker,
    CrossEncoderRerankFactory,
    NoRerankFactory,
)
from unified_kg_rag.adapters.local.token_counter import (
    LocalTokenCounter,
    local_token_counter_factory,
)
from unified_kg_rag.domain.models import Config, ModelPurpose
from unified_kg_rag.domain.prompts import AnswerGenerationPrompt, GraphExtractionPrompt
from unified_kg_rag.ports import (
    EmbeddingFactoryPort,
    LLMFactoryPort,
    RerankFactoryPort,
    TokenCounterPort,
)
from unified_kg_rag.shared import ConfigurationError, EmbeddingModelError
from unified_kg_rag.shared.exceptions import LLMOutputTruncatedError
from unified_kg_rag.shared.utils.langchain import RobustXMLOutputParser

pytestmark = pytest.mark.unit

DIM = 8


def _config(kind: str, **llm: Any) -> Config:
    base_url = (
        "http://model-server.test/v1"
        if kind == "openai_compatible"
        else "http://model-server.test"
    )
    return Config(
        providers={
            "llm": {
                "kind": kind,
                "base_url": base_url,
                **llm,
            },
            "embedding": {
                "kind": kind,
                "base_url": base_url,
                "max_input_tokens": 16,
            },
        },
        models={
            "default": "synthetic-chat",
            "embedding": {"model_id": "synthetic-embed", "dimension": DIM},
        },
    )


class FakeServer:
    """Records requests and answers like an OpenAI-compatible or Ollama server."""

    def __init__(
        self, reply: str = "Vendor ships to Buyer.", stop: str = "stop"
    ) -> None:
        self.reply = reply
        # finish_reason / done_reason; "length" = cut at the output limit.
        self.stop = stop
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.requests.append((request.url.path, body))
        path = request.url.path
        if path.endswith("/chat/completions"):
            return httpx.Response(200, json=self._openai_chat(body))
        if path.endswith("/embeddings"):
            inputs = body["input"]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": body["model"],
                    "data": [
                        {"object": "embedding", "index": i, "embedding": _vector(i)}
                        for i in range(len(inputs))
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        if path == "/api/chat":
            return httpx.Response(200, content=self._ollama_chat(body))
        if path == "/api/embed":
            inputs = body["input"]
            inputs = inputs if isinstance(inputs, list) else [inputs]
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "embeddings": [_vector(i) for i in range(len(inputs))],
                },
            )
        return httpx.Response(404, json={"error": f"unexpected path {path}"})

    def _openai_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.reply},
                    "finish_reason": self.stop,
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        }

    def _ollama_chat(self, body: dict[str, Any]) -> bytes:
        chunk = {
            "model": body["model"],
            "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": self.reply},
            "done": True,
            "done_reason": self.stop,
            "prompt_eval_count": 5,
            "eval_count": 4,
        }
        return (json.dumps(chunk) + "\n").encode()


def _vector(i: int) -> list[float]:
    return [float(i + 1)] + [0.0] * (DIM - 1)


class FailingServer(FakeServer):
    """Answers the first ``failures`` requests with ``status``, then normally."""

    def __init__(self, status: int, failures: int = 1) -> None:
        super().__init__()
        self.status = status
        self.failures = failures
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls <= self.failures:
            # retry-after-ms keeps the OpenAI client's own backoff short.
            return httpx.Response(
                self.status,
                json={"error": "synthetic failure"},
                headers={"retry-after-ms": "1"},
            )
        return super().__call__(request)


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retry_module, "backoff_delay", lambda *args: 0.0)


# --- language models --------------------------------------------------------


@pytest.mark.parametrize(
    "factory_cls",
    [OpenAICompatibleLanguageModelFactory, OllamaLanguageModelFactory],
)
def test_factories_conform_to_the_ports(factory_cls: type) -> None:
    config = _config("ollama")
    assert isinstance(factory_cls(config), LLMFactoryPort)
    assert isinstance(OllamaEmbeddingFactory(config), EmbeddingFactoryPort)
    assert isinstance(NoRerankFactory(config), RerankFactoryPort)
    assert isinstance(LocalTokenCounter("m"), TokenCounterPort)


def test_openai_compatible_chat_round_trip() -> None:
    server = FakeServer()
    config = _config("openai_compatible", max_output_tokens=512, temperature=0.2)
    factory = OpenAICompatibleLanguageModelFactory(
        config, transport=httpx.MockTransport(server)
    )

    # Bedrock-only kwargs that call sites pass are accepted and ignored.
    model = factory.get_model(
        "synthetic-chat",
        model_purpose=ModelPurpose.INGESTION,
        min_output_tokens=100000,
        enable_thinking=True,
        model_tier="fast",
        top_k=5,
    )
    reply = model.invoke([HumanMessage(content="Who ships?")])

    assert reply.content == server.reply
    path, body = server.requests[-1]
    assert path == "/v1/chat/completions"
    assert body["model"] == "synthetic-chat"
    # The prompt's output floor is clamped to the configured ceiling.
    assert body.get("max_tokens", body.get("max_completion_tokens")) == 512
    assert body["temperature"] == 0.2


async def test_openai_compatible_chat_async() -> None:
    server = FakeServer()
    factory = OpenAICompatibleLanguageModelFactory(
        _config("openai_compatible"), transport=httpx.MockTransport(server)
    )
    reply = await factory.get_model("synthetic-chat").ainvoke("hello")
    assert reply.content == server.reply


def test_ollama_chat_round_trip_sends_the_context_window() -> None:
    server = FakeServer()
    config = _config("ollama", context_window_size=16384, max_output_tokens=2048)
    factory = OllamaLanguageModelFactory(config, transport=httpx.MockTransport(server))

    reply = factory.get_model("synthetic-chat", max_tokens=300).invoke("hello")

    assert reply.content == server.reply
    path, body = server.requests[-1]
    assert path == "/api/chat"
    assert body["model"] == "synthetic-chat"
    assert body["options"]["num_ctx"] == 16384
    assert body["options"]["num_predict"] == 300
    assert body["options"]["temperature"] == 0.0


_ANSWER_INPUT = {"query": "Who ships?", "context": "Vendor ships to Buyer."}


def _query_chain(factory: Any, purpose: ModelPurpose = ModelPurpose.QUERY) -> Any:
    return setup_chain(
        factory=factory,
        model_id="synthetic-chat",
        prompt_class=AnswerGenerationPrompt,
        parser=StrOutputParser(),
        model_purpose=purpose,
    )


@pytest.mark.usefixtures("no_backoff")
async def test_ollama_query_chain_retries_a_transient_error() -> None:
    server = FailingServer(503)
    factory = OllamaLanguageModelFactory(
        _config("ollama"), transport=httpx.MockTransport(server)
    )
    chain = _query_chain(factory)

    assert isinstance(chain, TransientRetryRunnable)
    assert chain.retry.max_attempts == 3
    assert await chain.ainvoke(_ANSWER_INPUT) == server.reply
    assert server.calls == 2


@pytest.mark.usefixtures("no_backoff")
def test_ollama_query_chain_retries_a_transient_error_sync() -> None:
    server = FailingServer(503)
    factory = OllamaLanguageModelFactory(
        _config("ollama"), transport=httpx.MockTransport(server)
    )
    assert _query_chain(factory).invoke(_ANSWER_INPUT) == server.reply
    assert server.calls == 2


@pytest.mark.usefixtures("no_backoff")
async def test_ollama_query_chain_does_not_retry_a_client_error() -> None:
    server = FailingServer(400)
    factory = OllamaLanguageModelFactory(
        _config("ollama"), transport=httpx.MockTransport(server)
    )
    with pytest.raises(Exception, match="synthetic failure"):
        await _query_chain(factory).ainvoke(_ANSWER_INPUT)
    assert server.calls == 1


@pytest.mark.usefixtures("no_backoff")
async def test_ollama_query_retry_follows_max_attempts() -> None:
    server = FailingServer(503, failures=10)
    factory = OllamaLanguageModelFactory(
        _config("ollama", max_attempts=2), transport=httpx.MockTransport(server)
    )
    with pytest.raises(Exception, match="synthetic failure"):
        await _query_chain(factory).ainvoke(_ANSWER_INPUT)
    assert server.calls == 2
    # max_attempts 1 turns the retry off.
    off = OllamaLanguageModelFactory(_config("ollama", max_attempts=1))
    assert not isinstance(_query_chain(off), TransientRetryRunnable)


def test_ollama_ingestion_chain_keeps_the_batch_retry_only() -> None:
    factory = OllamaLanguageModelFactory(_config("ollama"))
    chain = _query_chain(factory, ModelPurpose.INGESTION)
    assert not isinstance(chain, TransientRetryRunnable)


async def test_openai_compatible_query_is_retried_by_its_client_only() -> None:
    # The OpenAI client retries (max_attempts - 1 retries), so the chain gets
    # no second retry layer that would multiply the attempts.
    server = FailingServer(503, failures=10)
    factory = OpenAICompatibleLanguageModelFactory(
        _config("openai_compatible", max_attempts=2),
        transport=httpx.MockTransport(server),
    )
    chain = _query_chain(factory)

    assert not isinstance(chain, TransientRetryRunnable)
    with pytest.raises(Exception, match="synthetic failure"):
        await chain.ainvoke(_ANSWER_INPUT)
    assert server.calls == 2

    recovering = FailingServer(503)
    factory = OpenAICompatibleLanguageModelFactory(
        _config("openai_compatible"), transport=httpx.MockTransport(recovering)
    )
    assert await _query_chain(factory).ainvoke(_ANSWER_INPUT) == recovering.reply
    assert recovering.calls == 2


@pytest.mark.parametrize(
    ("kind", "factory_cls"),
    [
        ("openai_compatible", OpenAICompatibleLanguageModelFactory),
        ("ollama", OllamaLanguageModelFactory),
    ],
)
def test_response_cut_at_the_output_limit_fails_the_chain(kind, factory_cls) -> None:
    # Both servers report a cut answer as "length", not Bedrock's
    # "max_tokens"; the partial answer must not be parsed as a success.
    server = FakeServer(reply="<entities><entity><name>Vendor", stop="length")
    factory = factory_cls(_config(kind), transport=httpx.MockTransport(server))
    chain = setup_chain(
        factory=factory,
        model_id="synthetic-chat",
        prompt_class=GraphExtractionPrompt,
        parser=RobustXMLOutputParser(),
        model_purpose=ModelPurpose.INGESTION,
    )
    with pytest.raises(LLMOutputTruncatedError):
        chain.invoke(
            {
                "input_text": "Vendor ships parts to Buyer.",
                "max_entities_per_chunk": 10,
                "max_relationships_per_chunk": 10,
                "entity_types": "ORG",
                "target_language": "en",
            }
        )


def test_model_info_comes_from_config() -> None:
    config = _config("ollama", context_window_size=8192, max_output_tokens=1024)
    info = OllamaLanguageModelFactory(config).get_model_info("anything")
    assert (info.context_window_size, info.max_output_tokens) == (8192, 1024)
    assert info.supports_prompt_caching is False
    assert info.supports_count_tokens is False
    assert info.supports_sampling_params is True


def test_api_key_placeholder_is_sent_when_unset() -> None:
    server = FakeServer()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization", ""))
        return server(request)

    factory = OpenAICompatibleLanguageModelFactory(
        _config("openai_compatible"), transport=httpx.MockTransport(handler)
    )
    factory.get_model("synthetic-chat").invoke("hello")
    assert seen == [f"Bearer {PLACEHOLDER_API_KEY}"]


def test_tracing_callback_is_attached_when_observability_is_on(mocker) -> None:
    from unified_kg_rag.shared import observability

    mocker.patch.object(observability, "is_enabled", return_value=True)
    factory = OllamaLanguageModelFactory(_config("ollama"))
    model = factory.get_model("synthetic-chat", model_purpose=ModelPurpose.QUERY)

    from unified_kg_rag.adapters.observability.langchain_callback import (
        GenAITracingCallbackHandler,
    )

    handlers = [
        c for c in model.callbacks or [] if isinstance(c, GenAITracingCallbackHandler)
    ]
    assert len(handlers) == 1
    assert handlers[0].provider_name == "ollama"
    assert handlers[0].purpose == "query"


def test_no_tracing_callback_when_observability_is_off() -> None:
    model = OllamaLanguageModelFactory(_config("ollama")).get_model("m")
    assert not model.callbacks


def test_missing_extra_names_the_remedy(mocker) -> None:
    mocker.patch(
        "unified_kg_rag.adapters.local.models.importlib.import_module",
        side_effect=ImportError("no module"),
    )
    with pytest.raises(ConfigurationError, match="--extra local"):
        import_optional("langchain_ollama", "ollama")


# --- embeddings -------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory_cls", "kind", "path"),
    [
        (OpenAICompatibleEmbeddingFactory, "openai_compatible", "/v1/embeddings"),
        (OllamaEmbeddingFactory, "ollama", "/api/embed"),
    ],
)
def test_embeddings_round_trip(factory_cls: type, kind: str, path: str) -> None:
    server = FakeServer()
    factory = factory_cls(_config(kind), transport=httpx.MockTransport(server))

    assert factory.get_model_info("synthetic-embed").dimensions == DIM
    model = factory.get_model("synthetic-embed", dimensions=DIM)
    long_text = "word " * 100
    vectors = model.embed_documents(["Vendor", long_text])
    query = model.embed_query("Buyer")

    assert len(vectors) == 2 and all(len(v) == DIM for v in vectors)
    assert len(query) == DIM
    sent_path, body = server.requests[0]
    assert sent_path == path
    assert body["model"] == "synthetic-embed"
    sent = body["input"]
    # Plain text (no token ids), truncated to max_input_tokens.
    assert sent[0] == "Vendor"
    assert estimate_token_count(sent[1]) <= 16


async def test_embeddings_async() -> None:
    server = FakeServer()
    factory = OllamaEmbeddingFactory(
        _config("ollama"), transport=httpx.MockTransport(server)
    )
    model = factory.get_model("synthetic-embed")
    assert len(await model.aembed_query("Vendor")) == DIM
    assert len(await model.aembed_documents(["a", "b"])) == 2
    assert await model.aembed_documents([]) == []


@pytest.mark.usefixtures("no_backoff")
async def test_ollama_embeddings_retry_a_transient_error() -> None:
    server = FailingServer(503)
    factory = OllamaEmbeddingFactory(
        _config("ollama"), transport=httpx.MockTransport(server)
    )
    assert len(await factory.get_model("synthetic-embed").aembed_query("a")) == DIM
    assert server.calls == 2

    server = FailingServer(503)
    factory = OllamaEmbeddingFactory(
        _config("ollama"), transport=httpx.MockTransport(server)
    )
    assert len(factory.get_model("synthetic-embed").embed_documents(["a"])) == 1
    assert server.calls == 2


@pytest.mark.usefixtures("no_backoff")
def test_ollama_embeddings_do_not_retry_a_client_error() -> None:
    server = FailingServer(400)
    factory = OllamaEmbeddingFactory(
        _config("ollama"), transport=httpx.MockTransport(server)
    )
    with pytest.raises(Exception, match="synthetic failure"):
        factory.get_model("synthetic-embed").embed_query("a")
    assert server.calls == 1


def test_openai_compatible_embeddings_leave_retries_to_the_client() -> None:
    factory = OpenAICompatibleEmbeddingFactory(_config("openai_compatible"))
    assert factory.get_model("synthetic-embed").retry is None


def test_embedding_dimension_mismatch_fails() -> None:
    factory = OllamaEmbeddingFactory(_config("ollama"))
    with pytest.raises(EmbeddingModelError, match="models.embedding.dimension"):
        factory.get_model("synthetic-embed", dimensions=DIM * 2)


def test_truncate_to_estimate_keeps_the_longest_fitting_prefix() -> None:
    text = "alpha beta gamma delta " * 20
    cut = truncate_to_estimate(text, 10)
    assert estimate_token_count(cut) <= 10
    assert estimate_token_count(text[: len(cut) + 1]) > 10
    assert truncate_to_estimate("short", 10) == "short"


# --- rerank -----------------------------------------------------------------


class FakeCrossEncoder:
    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.kwargs = kwargs
        self.calls = 0

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        self.calls += 1
        query = pairs[0][0].lower().split()
        return [
            sum(word in passage.lower() for word in query) / len(query)
            for _, passage in pairs
        ]


def _rerank_config() -> Config:
    return Config(
        providers={"rerank": {"kind": "cross_encoder"}},
        models={"rerank": "synthetic-ce"},
    )


def test_cross_encoder_reranks_and_limits_to_top_n() -> None:
    loaded: list[FakeCrossEncoder] = []

    def loader(model_id: str, **kwargs: Any) -> FakeCrossEncoder:
        loaded.append(FakeCrossEncoder(model_id, **kwargs))
        return loaded[-1]

    factory = CrossEncoderRerankFactory(_rerank_config(), loader=loader)
    reranker = factory.get_model("synthetic-ce", top_k=2)
    assert isinstance(reranker, CrossEncoderReranker)
    docs = [
        Document(page_content="Buyer pays", metadata={"key": "a"}),
        Document(page_content="Vendor ships goods to Buyer", metadata={"key": "b"}),
        Document(page_content="unrelated", metadata={"key": "c"}),
    ]

    out = reranker.compress_documents(docs, "vendor ships buyer")

    assert [d.metadata["key"] for d in out] == ["b", "a"]
    assert out[0].metadata["relevance_score"] == pytest.approx(1.0)
    # One model load per factory, shared by per-query copies.
    copy = reranker.model_copy(update={"top_n": 1})
    assert len(copy.compress_documents(docs, "vendor")) == 1
    factory.get_model("synthetic-ce")
    assert len(loaded) == 1
    assert loaded[0].kwargs["max_length"] == 512


def test_cross_encoder_without_the_extra_degrades_to_no_rerank(mocker) -> None:
    mocker.patch(
        "unified_kg_rag.adapters.local.rerank.import_optional",
        side_effect=ConfigurationError("needs sentence_transformers"),
    )
    factory = CrossEncoderRerankFactory(_rerank_config())
    assert factory.get_model("synthetic-ce") is None


def test_no_rerank_factory_returns_no_model() -> None:
    assert NoRerankFactory(Config()).get_model("anything", top_k=3) is None


def test_hybrid_scorer_keeps_the_fused_order_without_a_reranker() -> None:
    from unified_kg_rag.adapters.retrieval.hybrid_scorer import HybridScorer
    from unified_kg_rag.domain.models import RetrievalResult

    config = Config(providers={"rerank": {"kind": "none"}})
    scorer = HybridScorer(config, rerank_factory=NoRerankFactory(config))
    assert scorer.rerank_model is None
    results = {
        "text": [
            RetrievalResult(
                chunk_id="t1", content="Vendor", score=0.9, retriever_type="text"
            ),
            RetrievalResult(
                chunk_id="t2", content="Buyer", score=0.5, retriever_type="text"
            ),
        ]
    }
    fused = scorer.fuse_and_rerank_results(results, top_k=2, query="Vendor")
    assert [r.chunk_id for r in fused] == ["t1", "t2"]


# --- token counter ----------------------------------------------------------


def test_estimate_counter_matches_the_estimate_and_truncates() -> None:
    counter = LocalTokenCounter("m")
    text = "Vendor ships goods to Buyer " * 30
    assert counter.count_tokens(text) == estimate_token_count(text)
    truncated, count = counter.truncate_to_token_limit(text, 20)
    assert count <= 20 and text.startswith(truncated)
    assert counter.truncate_to_token_limit("short", 20) == ("short", 1)
    assert counter.count_tokens("") == 0


def test_tiktoken_counter_uses_the_encoding(mocker) -> None:
    class FakeEncoding:
        def encode(self, text: str, **kwargs: Any) -> list[str]:
            return list(text)

        def decode(self, tokens: list[str]) -> str:
            return "".join(tokens)

    from unified_kg_rag.adapters.local import token_counter as module

    mocker.patch.object(module, "load_encoding", return_value=FakeEncoding())
    counter = LocalTokenCounter("m", tokenizer="tiktoken")
    assert counter.count_tokens("abcdef") == 6
    assert counter.truncate_to_token_limit("abcdef", 4) == ("abcd", 4)


def test_unavailable_tiktoken_falls_back_to_the_estimate(mocker) -> None:
    import builtins

    from unified_kg_rag.adapters.local import token_counter as module

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "tiktoken":
            raise ImportError("no tiktoken")
        return real_import(name, *args, **kwargs)

    mocker.patch.object(module, "_encodings", {})
    mocker.patch.object(module, "_failed", set())
    mocker.patch("builtins.__import__", side_effect=fake_import)
    counter = LocalTokenCounter("m", tokenizer="tiktoken", encoding_name="synthetic")
    assert counter.count_tokens("Vendor ships") == estimate_token_count("Vendor ships")


def test_token_counter_factory_follows_the_llm_settings() -> None:
    config = Config(
        providers={"llm": {"kind": "ollama", "tokenizer": "estimate"}},
        models={"default": "m"},
    )
    counter = local_token_counter_factory(config)(
        "m", cache_maxsize=8, api_supported=True
    )
    assert isinstance(counter, LocalTokenCounter)
    assert counter.count_tokens("Vendor") == 1


# --- event loops ---------------------------------------------------------------


class _HTTPFakeServer:
    """``FakeServer`` over real HTTP/1.1 keep-alive sockets on localhost.

    ``httpx.MockTransport`` opens no connection, so it cannot show a pooled
    connection bound to a closed event loop; this server can.
    """

    def __init__(self, server: FakeServer) -> None:
        fake = server

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length") or 0)
                request = httpx.Request(
                    "POST",
                    f"http://localhost{self.path}",
                    content=self.rfile.read(length),
                )
                response = fake(request)
                body = response.content
                self.send_response(response.status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: Any) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> _HTTPFakeServer:
        self.thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _http_config(kind: str, url: str) -> Config:
    config = _config(kind)
    base_url = f"{url}/v1" if kind == "openai_compatible" else url
    config.providers.llm.base_url = base_url
    config.providers.embedding.base_url = base_url
    return config


@pytest.mark.parametrize(
    ("kind", "factory_cls"),
    [
        ("ollama", OllamaLanguageModelFactory),
        ("openai_compatible", OpenAICompatibleLanguageModelFactory),
    ],
)
def test_one_chat_model_serves_successive_event_loops(
    kind: str, factory_cls: type
) -> None:
    # A query chain is built once and reused, and asyncio.run gives every
    # call a new loop: a client whose pooled connection belongs to the first
    # loop fails the second call with "Event loop is closed".
    with _HTTPFakeServer(FakeServer()) as server:
        model = factory_cls(_http_config(kind, server.url)).get_model("synthetic-chat")
        for _ in range(3):
            reply = asyncio.run(model.ainvoke([HumanMessage(content="Who ships?")]))
            assert reply.content == "Vendor ships to Buyer."
        if kind != "ollama":
            return  # FakeServer answers OpenAI streams without SSE framing.

        async def stream() -> str:
            return "".join(
                [
                    str(chunk.content)
                    async for chunk in model.astream([HumanMessage(content="Who?")])
                ]
            )

        assert asyncio.run(stream()) == "Vendor ships to Buyer."


@pytest.mark.parametrize(
    ("kind", "factory_cls"),
    [
        ("ollama", OllamaEmbeddingFactory),
        ("openai_compatible", OpenAICompatibleEmbeddingFactory),
    ],
)
def test_one_embedding_model_serves_successive_event_loops(
    kind: str, factory_cls: type
) -> None:
    with _HTTPFakeServer(FakeServer()) as server:
        model = factory_cls(_http_config(kind, server.url)).get_model(
            "synthetic-embed", dimensions=DIM
        )
        for _ in range(3):
            assert len(asyncio.run(model.aembed_query("Vendor"))) == DIM


def test_per_loop_client_is_reused_within_a_loop_and_rebuilt_across_loops() -> None:
    built: list[object] = []

    class Client:
        def __init__(self) -> None:
            built.append(self)

        async def chat(self) -> object:
            return self

    proxy = PerLoopAsyncClient(Client)

    async def twice() -> tuple[object, object]:
        return await proxy.chat(), await proxy.chat()

    first_a, first_b = asyncio.run(twice())
    second_a, _ = asyncio.run(twice())

    assert first_a is first_b
    assert second_a is not first_a
    assert len(built) == 2
    # The first loop is closed, so its client was dropped for the second's.
    assert len(proxy._clients) == 1


_FD_DIR = Path("/proc/self/fd")


def _open_fds() -> int:
    return len(os.listdir(_FD_DIR))


@pytest.mark.skipif(not _FD_DIR.is_dir(), reason="needs /proc/self/fd")
def test_successive_event_loops_leave_no_ollama_clients_or_sockets() -> None:
    # asyncio.run closes its loop but not the clients used on it. A client
    # kept per loop with a pooled keep-alive connection stayed reachable (the
    # connection references its loop) and kept its socket open: every run
    # left a client and two descriptors (both ends) behind.
    runs = 40
    with _HTTPFakeServer(FakeServer()) as server:
        config = _http_config("ollama", server.url)
        chat = OllamaLanguageModelFactory(config).get_model("synthetic-chat")
        embeddings = OllamaEmbeddingFactory(config).get_model(
            "synthetic-embed", dimensions=DIM
        )

        async def query() -> None:
            await chat.ainvoke([HumanMessage(content="Who ships?")])
            await embeddings.aembed_query("Vendor")

        asyncio.run(query())
        before = _open_fds()
        for _ in range(runs):
            asyncio.run(query())
        after = _open_fds()

        # Only the last (closed) loop's client is left until the next lookup.
        assert len(chat._async_client._clients) <= 1
        assert len(embeddings.inner._async_client._clients) <= 1
        assert after - before < runs


@pytest.mark.skipif(not _FD_DIR.is_dir(), reason="needs /proc/self/fd")
@pytest.mark.parametrize("max_attempts", [1, 3])
def test_successive_event_loops_send_one_openai_request_per_call(
    max_attempts: int,
) -> None:
    # langchain-openai shares one async HTTP client (and its keep-alive
    # pool) across event loops: the first call on every new loop hit a
    # connection of the closed previous loop, failed with "Event loop is
    # closed" and was retried, so each call sent two requests, and with
    # max_attempts 1 it failed with APIConnectionError.
    runs = 30
    fake = FakeServer()
    with _HTTPFakeServer(fake) as server:
        config = _http_config("openai_compatible", server.url)
        config.providers.llm.max_attempts = max_attempts
        config.providers.embedding.max_attempts = max_attempts
        chat = OpenAICompatibleLanguageModelFactory(config).get_model("synthetic-chat")
        embeddings = OpenAICompatibleEmbeddingFactory(config).get_model(
            "synthetic-embed", dimensions=DIM
        )

        async def query() -> None:
            reply = await chat.ainvoke([HumanMessage(content="Who ships?")])
            assert reply.content == "Vendor ships to Buyer."
            assert len(await embeddings.aembed_query("Vendor")) == DIM

        asyncio.run(query())
        before = _open_fds()
        for _ in range(runs):
            asyncio.run(query())
        after = _open_fds()

        paths = [path for path, _ in fake.requests]
        assert paths.count("/v1/chat/completions") == runs + 1
        assert paths.count("/v1/embeddings") == runs + 1
        assert after - before < runs
