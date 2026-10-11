# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry of provider builders, keyed by role and ``providers.*.kind``.

Each role of the provider bundle (``adapters.providers.Providers``) — the
LLM, embedding and rerank factories, the token counter, the graph and vector
indexers and retrievers, the doc-status registry — is built by the builder
registered for the kind its config section names (``config.providers``). The
built-in kinds register in ``adapters.providers``; a custom backend registers
its own kind the same way, with no edit to the bundle or any call site::

    @register_provider(ProviderRole.LLM, "my_server")
    def _build(providers: Providers) -> LLMFactoryPort:
        return MyServerFactory(providers.config)

and selects it with ``providers.llm.kind: my_server``. A builder receives the
bundle itself, so it can reuse the shared session or another role (the vector
indexer takes ``providers.embedding_factory``).
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

from unified_kg_rag.shared import ConfigurationError

if TYPE_CHECKING:
    from unified_kg_rag.adapters.providers import Providers
    from unified_kg_rag.domain.models import Config


class ProviderRole(str, Enum):
    """A component of the provider bundle, built by one registered builder."""

    LLM = "llm"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    TOKEN_COUNTER = "token_counter"
    GRAPH_INDEXER = "graph_indexer"
    GRAPH_RETRIEVER = "graph_retriever"
    VECTOR_INDEXER = "vector_indexer"
    VECTOR_RETRIEVER = "vector_retriever"
    DOC_STATUS = "doc_status"

    @property
    def section(self) -> str:
        """The ``config.providers`` section whose ``kind`` selects this role."""
        return _SECTIONS[self]


_SECTIONS: dict[ProviderRole, str] = {
    ProviderRole.LLM: "llm",
    ProviderRole.EMBEDDING: "embedding",
    ProviderRole.RERANK: "rerank",
    # Token counting follows the language models it counts for.
    ProviderRole.TOKEN_COUNTER: "llm",
    ProviderRole.GRAPH_INDEXER: "graph",
    ProviderRole.GRAPH_RETRIEVER: "graph",
    ProviderRole.VECTOR_INDEXER: "vector",
    ProviderRole.VECTOR_RETRIEVER: "vector",
    ProviderRole.DOC_STATUS: "doc_status",
}

ProviderBuilder = Callable[["Providers"], Any]
BuilderT = TypeVar("BuilderT", bound=ProviderBuilder)

_REGISTRY: dict[tuple[ProviderRole, str], ProviderBuilder] = {}


def register_provider(role: ProviderRole, kind: str) -> Callable[[BuilderT], BuilderT]:
    """Register the decorated builder for ``role`` under ``kind``.

    A later registration of the same role and kind replaces the earlier one,
    so a deployment can override a built-in (e.g. a tuned Bedrock factory).
    """
    key = (role, kind.strip().lower())

    def decorator(builder: BuilderT) -> BuilderT:
        _REGISTRY[key] = builder
        return builder

    return decorator


def registered_kinds(role: ProviderRole) -> list[str]:
    """The kinds registered for ``role``, sorted."""
    return sorted(kind for registered, kind in _REGISTRY if registered is role)


def provider_kind(config: Config, role: ProviderRole) -> str:
    """The kind ``config`` selects for ``role``."""
    kind: str = getattr(config.providers, role.section).kind
    return kind


def get_provider_builder(role: ProviderRole, kind: str) -> ProviderBuilder:
    """The builder registered for ``role`` and ``kind``.

    Raises:
        ConfigurationError: Nothing is registered for the pair; the message
            lists the kinds that are.
    """
    builder = _REGISTRY.get((role, kind))
    if builder is None:
        raise ConfigurationError(
            f"No {role.value} provider is registered for "
            f"providers.{role.section}.kind '{kind}'. Registered kinds: "
            f"{', '.join(registered_kinds(role)) or 'none'}."
        )
    return builder
