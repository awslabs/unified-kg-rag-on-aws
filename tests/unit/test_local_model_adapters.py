# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local model adapters against fake HTTP servers (no network, no models).

The OpenAI-compatible and Ollama factories are driven through an
``httpx.MockTransport`` that answers like the real servers, so the tests see
the exact requests the adapters send (model, limits, num_ctx) and parse real
response shapes. The cross-encoder is a fake loader; tiktoken is not loaded.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from unified_kg_rag.adapters.aws.chain_factory import setup_chain
from unified_kg_rag.adapters.aws.token_counter import estimate_token_count
from unified_kg_rag.adapters.local.embeddings import (
    OllamaEmbeddingFactory,
    OpenAICompatibleEmbeddingFactory,
    truncate_to_estimate,
)
from unified_kg_rag.adapters.local.models import (
    PLACEHOLDER_API_KEY,
    OllamaLanguageModelFactory,
    OpenAICompatibleLanguageModelFactory,
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
from unified_kg_rag.domain.prompts import GraphExtractionPrompt
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
                "default_model_id": "synthetic-chat",
                **llm,
            },
            "embedding": {
                "kind": kind,
                "base_url": base_url,
                "model_id": "synthetic-embed",
                "dimension": DIM,
                "max_input_tokens": 16,
            },
        }
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


def test_embedding_dimension_mismatch_fails() -> None:
    factory = OllamaEmbeddingFactory(_config("ollama"))
    with pytest.raises(EmbeddingModelError, match="providers.embedding.dimension"):
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
        providers={"rerank": {"kind": "cross_encoder", "model_id": "synthetic-ce"}}
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
        providers={
            "llm": {"kind": "ollama", "default_model_id": "m", "tokenizer": "estimate"}
        }
    )
    counter = local_token_counter_factory(config)(
        "m", cache_maxsize=8, api_supported=True
    )
    assert isinstance(counter, LocalTokenCounter)
    assert counter.count_tokens("Vendor") == 1
