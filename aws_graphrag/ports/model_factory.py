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

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelFactoryPort(Protocol):
    """Resolves a typed model id to a provider model object.

    Implemented by ``BedrockLanguageModelFactory`` /
    ``BedrockEmbeddingModelFactory`` / ``BedrockRerankModelFactory``. ``get_model``
    returns a LangChain-compatible model (chat model, embeddings, or reranker);
    ``get_model_info`` returns the capability record for the id (or ``None``).
    """

    def get_model(self, model_id: Any, **kwargs: Any) -> Any: ...

    def get_model_info(self, model_id: Any) -> Any: ...


# Semantic aliases. The contract is identical across model kinds today; the
# distinct names document intent at call sites and leave room to diverge later.
LLMFactoryPort = ModelFactoryPort
EmbeddingFactoryPort = ModelFactoryPort
RerankFactoryPort = ModelFactoryPort
