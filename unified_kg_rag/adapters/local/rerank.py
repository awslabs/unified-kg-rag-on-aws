# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local rerankers: a sentence-transformers cross-encoder and the no-op one.

Both conform to ``RerankFactoryPort``. ``HybridScorer`` treats a ``None``
model as "no reranking" and keeps the fused order, which is how the no-op
factory (``providers.rerank.kind: none``) and a cross-encoder that cannot load
(the ``[local]`` extra is not installed) degrade: logged once per process, and
visible as ``kg_rag.rerank_enabled=false`` on the fusion span.

The cross-encoder model is loaded once per factory on first use and shared by
every reranker copy (``HybridScorer`` copies the reranker per query to set
``top_n``). Scores are what ``CrossEncoder.predict`` returns (a sigmoid
probability for the usual single-label rerank models), recorded as
``relevance_score`` like the Bedrock reranker's.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict, Field

from unified_kg_rag.adapters.aws.bedrock_models import RerankModelInfo
from unified_kg_rag.adapters.local.embeddings import truncate_to_estimate
from unified_kg_rag.adapters.local.models import import_optional
from unified_kg_rag.domain.models import Config, model_id_value
from unified_kg_rag.shared import (
    ConfigurationError,
    RerankModelError,
    get_logger,
    observability,
)

logger = get_logger(__name__)

_warned: set[str] = set()
_warned_lock = threading.Lock()


def _warn_once(key: str, message: str, *args: Any) -> None:
    with _warned_lock:
        if key in _warned:
            return
        _warned.add(key)
    logger.warning(message, *args)


class CrossEncoderReranker(BaseDocumentCompressor):
    """Reranks documents by a cross-encoder's query/passage score."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str
    encoder: Any = Field(exclude=True)
    top_n: int | None = None
    batch_size: int = 16
    max_tokens: int = 512

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        # Query and passage share the model's window; cut each to its half.
        half = max(self.max_tokens // 2, 8)
        query_text = truncate_to_estimate(query, half)
        pairs = [
            (query_text, truncate_to_estimate(doc.page_content, half))
            for doc in documents
        ]
        with observability.span(
            "kg_rag.rerank",
            {
                "gen_ai.provider.name": "sentence_transformers",
                "gen_ai.request.model": self.model_id,
            },
            documents_in=len(documents),
            top_n=self.top_n,
        ) as span:
            try:
                scores = self.encoder.predict(
                    pairs, batch_size=self.batch_size, show_progress_bar=False
                )
            except Exception as e:
                span.set_attribute("kg_rag.rerank.status", "error")
                raise RerankModelError(f"Reranking failed: {e}") from e
            ranked = sorted(
                zip(documents, (float(s) for s in scores), strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )
            limit = self.top_n if self.top_n is not None else len(ranked)
            result = [
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "relevance_score": score},
                    id=doc.id,
                )
                for doc, score in ranked[:limit]
            ]
            span.set_attributes(
                {
                    "kg_rag.rerank.status": "ok" if result else "empty",
                    "kg_rag.documents_out": len(result),
                }
            )
            return result


class CrossEncoderRerankFactory:
    """``providers.rerank.kind: cross_encoder`` (sentence-transformers)."""

    kind = "cross_encoder"

    def __init__(self, config: Config, *, loader: Any = None) -> None:
        """Args:
        config: Framework config; ``providers.rerank`` configures the model.
        loader: Called as ``loader(model_id, device=..., max_length=...)`` to
            load the cross-encoder; defaults to
            ``sentence_transformers.CrossEncoder`` (tests pass a fake).
        """
        self.config = config
        self.settings = config.providers.rerank
        self._loader = loader
        self._encoders: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_model_info(self, model_id: Any) -> RerankModelInfo:
        return RerankModelInfo(
            max_documents=self.config.search.reranking.top_k,
            max_query_tokens=self.settings.max_length // 2,
            max_document_tokens=self.settings.max_length // 2,
        )

    def _encoder(self, model_id: str) -> Any:
        with self._lock:
            if model_id not in self._encoders:
                loader = self._loader
                if loader is None:
                    loader = import_optional(
                        "sentence_transformers", self.kind
                    ).CrossEncoder
                self._encoders[model_id] = loader(
                    model_id,
                    device=self.settings.device,
                    max_length=self.settings.max_length,
                )
            return self._encoders[model_id]

    def get_model(self, model_id: Any, **kwargs: Any) -> CrossEncoderReranker | None:
        name = model_id_value(model_id)
        try:
            encoder = self._encoder(name)
        except ConfigurationError as e:
            _warn_once(
                f"cross_encoder:{name}",
                "Reranking is off: %s Queries keep the fused order.",
                e,
            )
            return None
        logger.debug("Loaded cross-encoder reranker '%s'", name)
        return CrossEncoderReranker(
            model_id=name,
            encoder=encoder,
            top_n=kwargs.get("top_k"),
            batch_size=self.settings.batch_size,
            max_tokens=self.settings.max_length,
        )


class NoRerankFactory:
    """``providers.rerank.kind: none``: every query keeps the fused order."""

    kind = "none"

    def __init__(self, config: Config) -> None:
        self.config = config

    def get_model_info(self, model_id: Any) -> None:
        return None

    def get_model(self, model_id: Any, **kwargs: Any) -> None:
        if self.config.search.reranking.enabled:
            _warn_once(
                "none",
                "search.reranking.enabled is true but providers.rerank.kind is "
                "'none'; queries keep the fused order.",
            )
        return None
