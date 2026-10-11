# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embedding factories for OpenAI-compatible servers and Ollama.

Both conform to ``EmbeddingFactoryPort``. The vector dimension is not
discoverable without a request, so it comes from
``providers.embedding.dimension`` (``get_model_info``), which fixes the
OpenSearch ``knn_vector`` mappings. Inputs are truncated to
``providers.embedding.max_input_tokens`` by the script-aware estimate (a
server rejects or silently cuts an over-long input), and each embed call is a
GenAI ``embeddings`` span when tracing is enabled, as on Bedrock.
"""

from __future__ import annotations

import importlib
from typing import Any, ClassVar

from langchain_core.embeddings import Embeddings

from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.local.models import api_key_value, import_optional
from unified_kg_rag.domain.models import Config, model_id_value
from unified_kg_rag.domain.models.provider_config import EmbeddingProviderConfig
from unified_kg_rag.ports.model_factory import EmbeddingModelInfo
from unified_kg_rag.shared import EmbeddingModelError, get_logger
from unified_kg_rag.shared.genai import embedding_span

logger = get_logger(__name__)


def truncate_to_estimate(text: str, max_tokens: int) -> str:
    """``text`` cut so its estimated token count is at most ``max_tokens``."""
    if estimate_token_count(text) <= max_tokens:
        return text
    low, high = 0, len(text)
    # Largest prefix that fits; the estimate is monotone in the prefix length.
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_token_count(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low]


class TracedEmbeddings(Embeddings):
    """Wraps a LangChain ``Embeddings`` with input truncation and a span."""

    def __init__(
        self,
        inner: Embeddings,
        *,
        model_id: str,
        provider_name: str,
        max_input_tokens: int,
    ) -> None:
        self.inner = inner
        self.model_id = model_id
        self.provider_name = provider_name
        self.max_input_tokens = max_input_tokens

    def _prepare(self, texts: list[str]) -> list[str]:
        return [truncate_to_estimate(text, self.max_input_tokens) for text in texts]

    def _span(self, texts: list[str]) -> Any:
        return embedding_span(
            self.provider_name,
            self.model_id,
            texts=len(texts),
            characters=sum(len(t) for t in texts),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = self._prepare(texts)
        with self._span(prepared):
            return self.inner.embed_documents(prepared)

    def embed_query(self, text: str) -> list[float]:
        prepared = self._prepare([text])
        with self._span(prepared):
            return self.inner.embed_query(prepared[0])

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = self._prepare(texts)
        with self._span(prepared):
            return await self.inner.aembed_documents(prepared)

    async def aembed_query(self, text: str) -> list[float]:
        prepared = self._prepare([text])
        with self._span(prepared):
            return await self.inner.aembed_query(prepared[0])


class _LocalEmbeddingFactory:
    provider_name: ClassVar[str]
    kind: ClassVar[str]

    def __init__(self, config: Config, *, transport: Any = None) -> None:
        """Args:
        config: Framework config; ``providers.embedding`` configures the server.
        transport: Optional ``httpx`` transport for the clients (tests).
        """
        self.config = config
        self.settings: EmbeddingProviderConfig = config.providers.embedding
        self.transport = transport

    def get_model_info(self, model_id: Any) -> EmbeddingModelInfo:
        tokens = self.settings.max_input_tokens
        return EmbeddingModelInfo(
            dimensions=self.settings.dimension,
            max_sequence_tokens=tokens,
            max_sequence_length=tokens * 4,
        )

    def get_model(self, model_id: Any, **kwargs: Any) -> TracedEmbeddings:
        name = model_id_value(model_id)
        dimensions = kwargs.get("dimensions")
        if dimensions is not None and dimensions != self.settings.dimension:
            raise EmbeddingModelError(
                f"Dimension {dimensions} requested for '{name}', but "
                f"providers.embedding.dimension is {self.settings.dimension}"
            )
        inner = self._build(name)
        logger.debug("Created %s embedding model '%s'", self.kind, name)
        return TracedEmbeddings(
            inner,
            model_id=name,
            provider_name=self.provider_name,
            max_input_tokens=self.settings.max_input_tokens,
        )

    def _build(self, model_id: str) -> Embeddings:
        raise NotImplementedError


class OpenAICompatibleEmbeddingFactory(_LocalEmbeddingFactory):
    """Embeddings from an OpenAI-compatible ``/v1/embeddings`` endpoint."""

    provider_name = "openai"
    kind = "openai_compatible"

    def _build(self, model_id: str) -> Embeddings:
        module = import_optional("langchain_openai", self.kind)
        extra: dict[str, Any] = {}
        if self.transport is not None:
            httpx = importlib.import_module("httpx")
            timeout = self.settings.timeout_seconds
            extra["http_client"] = httpx.Client(
                transport=self.transport, timeout=timeout
            )
            extra["http_async_client"] = httpx.AsyncClient(
                transport=self.transport, timeout=timeout
            )
        embeddings: Embeddings = module.OpenAIEmbeddings(
            model=model_id,
            base_url=self.settings.base_url,
            api_key=api_key_value(self.settings),
            timeout=self.settings.timeout_seconds,
            # The OpenAI client counts retries after the first attempt.
            max_retries=self.settings.max_attempts - 1,
            # Send text, not tiktoken ids: only OpenAI's own servers accept
            # pre-tokenized input, and the estimate already truncated it.
            check_embedding_ctx_length=False,
            **extra,
        )
        return embeddings


class OllamaEmbeddingFactory(_LocalEmbeddingFactory):
    """Embeddings from an Ollama server (``/api/embed``)."""

    provider_name = "ollama"
    kind = "ollama"

    def _build(self, model_id: str) -> Embeddings:
        module = import_optional("langchain_ollama", self.kind)
        client_kwargs: dict[str, Any] = {"timeout": self.settings.timeout_seconds}
        if self.transport is not None:
            client_kwargs["transport"] = self.transport
        embeddings: Embeddings = module.OllamaEmbeddings(
            model=model_id,
            base_url=self.settings.base_url,
            client_kwargs=client_kwargs,
        )
        return embeddings
