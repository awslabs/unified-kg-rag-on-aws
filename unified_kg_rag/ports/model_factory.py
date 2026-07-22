# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-factory ports — the LLM / Embedding / Rerank provider boundary.

The domain/application layers obtain language, embedding, and rerank models
through a *factory* that resolves a typed model id to a ready-to-use,
LangChain-compatible object. These ``Protocol``s capture that contract so a
non-Bedrock provider can be introduced by writing a conforming factory adapter,
without changing call sites.

The concrete Bedrock factories in ``adapters.aws.bedrock`` already conform
structurally (``get_model`` / ``get_model_info``); these Protocols make the
boundary explicit and give callers a provider-agnostic type to annotate against.
Defined as ``runtime_checkable`` Protocols (structural typing), so no base-class
change to the existing factories is required.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

# Generic over the model-id enum and the returned model type so each concrete
# factory keeps its narrow id type (LanguageModelId etc.) WITHOUT breaking
# structural conformance. A non-generic union here would be contravariantly
# incompatible with the narrower concrete factories (they each accept only one
# enum). `Any` defaults preserve the prior permissive behaviour for untyped
# call sites while letting a typed adapter parameterize for static safety.
ModelIdT = TypeVar("ModelIdT", contravariant=True)
ModelT = TypeVar("ModelT", covariant=True)


@runtime_checkable
class ModelFactoryPort(Protocol[ModelIdT, ModelT]):
    """Resolves a typed model id to a provider model object.

    Implemented by ``BedrockLanguageModelFactory`` /
    ``BedrockEmbeddingModelFactory`` / ``BedrockRerankModelFactory``. ``get_model``
    returns a LangChain-compatible model (chat model, embeddings, or reranker);
    ``get_model_info`` returns the capability record for the id (or ``None``).

    Parameterized over the id enum and the model type, so a custom backend
    adapter can annotate ``ModelFactoryPort[LanguageModelId, ChatModel]`` and get
    static checking, while the concrete narrow factories still conform.
    """

    def get_model(self, model_id: ModelIdT, **kwargs: Any) -> ModelT: ...

    def get_model_info(self, model_id: ModelIdT) -> Any: ...


# Semantic aliases — the bare (unsubscripted) Protocol so they remain usable in
# runtime isinstance() checks (a subscripted generic raises TypeError there).
# An unsubscripted generic Protocol defaults its params to Any, so existing
# annotations keep working; a call site wanting static narrowing can write
# `ModelFactoryPort[LanguageModelId, ChatModel]` explicitly. The distinct names
# document intent and leave room to diverge.
LLMFactoryPort = ModelFactoryPort
EmbeddingFactoryPort = ModelFactoryPort
RerankFactoryPort = ModelFactoryPort
