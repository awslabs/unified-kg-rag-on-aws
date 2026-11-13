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

A store kind that connects to a server declares its endpoint setting
(``endpoint=("aws.neptune.endpoint", "UKR__AWS__NEPTUNE__ENDPOINT")``: the config key and
its environment variable), so the CLIs can reject a missing one at start-up
(``application.cli.preflight``); an embedded kind declares none. A kind
whose builder imports an optional extra names its module (``requires=``),
so the CLIs reject a missing extra at start-up as well.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from unified_kg_rag.domain.models import RetrieverRole
from unified_kg_rag.shared import ConfigurationError
from unified_kg_rag.shared.registry import Registry

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

# The provider role that serves each retriever role of a search strategy.
RETRIEVER_PROVIDER_ROLES: dict[RetrieverRole, ProviderRole] = {
    RetrieverRole.GRAPH: ProviderRole.GRAPH_RETRIEVER,
    RetrieverRole.DOCUMENT: ProviderRole.VECTOR_RETRIEVER,
}

ProviderBuilder = Callable[["Providers"], Any]
BuilderT = TypeVar("BuilderT", bound=ProviderBuilder)


class ProviderEntry(NamedTuple):
    """A registered kind: its builder and the ``(config key, environment
    variable)`` of the endpoint the built component connects to."""

    builder: ProviderBuilder
    endpoint: tuple[str, str] | None


# One registry per role, under the one registry rule (``shared.registry``).
_REGISTRIES: dict[ProviderRole, Registry[ProviderEntry]] = {
    role: Registry(f"{role.value} provider") for role in ProviderRole
}


def register_provider(
    role: ProviderRole,
    kind: str,
    *,
    endpoint: tuple[str, str] | None = None,
    requires: tuple[str, str] | None = None,
) -> Callable[[BuilderT], BuilderT]:
    """Register the decorated builder for ``role`` under ``kind``.

    ``endpoint`` is the ``(config key, environment variable)`` of the server
    the built component connects to, or None for a kind that needs none.
    ``requires`` is the ``(module, how to install it)`` of the optional
    dependency the built component imports, or None.

    The kind follows the one registry rule (``shared.registry``):
    case-insensitive, and a later registration of the same role and kind
    replaces the earlier one, endpoint and requirement included, so a
    deployment can override a built-in (e.g. a tuned Bedrock factory).

    Raises:
        ValueError: ``kind`` is not a registry name.
    """
    registry = _REGISTRIES[role]
    registry.check_name(kind)

    def decorator(builder: BuilderT) -> BuilderT:
        registry.register(kind, ProviderEntry(builder, endpoint), requires=requires)
        return builder

    return decorator


def unregister_provider(role: ProviderRole, kind: str) -> None:
    """Remove ``kind``'s registration for ``role`` (a test's own kind)."""
    _REGISTRIES[role].unregister(kind)


def provider_endpoint(role: ProviderRole, kind: str) -> tuple[str, str] | None:
    """The ``(config key, environment variable)`` ``kind`` declared for ``role``."""
    entry = _REGISTRIES[role].get(kind)
    return None if entry is None else entry.endpoint


def provider_requirement(role: ProviderRole, kind: str) -> tuple[str, str] | None:
    """The ``(module, how to install it)`` ``kind`` declared for ``role``."""
    return _REGISTRIES[role].requirement(kind)


def registered_kinds(role: ProviderRole) -> list[str]:
    """The kinds registered for ``role``, sorted."""
    return _REGISTRIES[role].names()


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
    entry = _REGISTRIES[role].get(kind)
    if entry is None:
        raise ConfigurationError(
            f"No {role.value} provider is registered for "
            f"providers.{role.section}.kind '{kind}'. Registered kinds: "
            f"{', '.join(registered_kinds(role)) or 'none'}."
        )
    return entry.builder
