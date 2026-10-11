# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Composition root: the provider bundle for one run, built from config.

``build_providers`` checks that every role ``config.providers`` selects has a
registered builder (failing before any model is paid for) and returns the
``Providers`` bundle that the CLIs pass to ``GraphRAGChain``,
``DataIngestionPipeline`` and ``EvaluationManager``; those hand it on to every
stage, strategy, scorer, token manager, memory manager, evaluator and store
they build. Code that constructs an orchestrator without a bundle gets the
same config-driven defaults (``Providers`` resolves each role through the
same registry), so the bundle is a single place to inject overrides rather
than a requirement.

``with_provider_overrides`` applies the CLI provider flags (``--backend``,
``--llm-provider`` ...) to a loaded config, re-running the ``backend`` preset
and the model-id defaults as if the file had set them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from unified_kg_rag.adapters.provider_registry import (
    ProviderRole,
    get_provider_builder,
    provider_kind,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import ConfigurationError, get_logger

if TYPE_CHECKING:
    import boto3

logger = get_logger(__name__)

# Roles whose kind the run needs; the token counter follows the LLM's.
_CHECKED_ROLES = tuple(ProviderRole)


def check_providers(config: Config) -> dict[ProviderRole, str]:
    """``{role: kind}`` for ``config``; every kind must be registered.

    Raises:
        ConfigurationError: A role names a kind nothing is registered for.
    """
    selected: dict[ProviderRole, str] = {}
    for role in _CHECKED_ROLES:
        kind = provider_kind(config, role)
        get_provider_builder(role, kind)
        selected[role] = kind
    return selected


def build_providers(
    config: Config,
    boto_session: boto3.Session | None = None,
    **overrides: Any,
) -> Providers:
    """The provider bundle for ``config``.

    Args:
        config: Framework config; ``config.providers`` selects every role.
        boto_session: Session for the AWS providers (built lazily otherwise;
            local providers never touch it).
        **overrides: Injected components, as accepted by ``Providers``
            (``llm_factory=``, ``embedding_factory=``, ``rerank_factory=``,
            ``token_counter_factory=``, ``graph_indexer_factory=``,
            ``vector_indexer_factory=``, ``retriever_builders=``,
            ``doc_status=``); each wins over the configured kind.
    """
    selected = check_providers(config)
    logger.info(
        "Providers (backend '%s'): %s",
        config.backend.value,
        ", ".join(
            f"{role.value}={kind}"
            for role, kind in selected.items()
            if role is not ProviderRole.TOKEN_COUNTER
        ),
    )
    return Providers(config, boto_session, **overrides)


def with_provider_overrides(
    config: Config,
    *,
    backend: str | None = None,
    llm_kind: str | None = None,
    llm_model_id: str | None = None,
    embedding_kind: str | None = None,
    embedding_model_id: str | None = None,
    embedding_dimension: int | None = None,
    rerank_kind: str | None = None,
    base_url: str | None = None,
) -> Config:
    """``config`` re-validated with the given provider settings.

    ``None`` leaves a setting as configured. ``base_url`` applies to the LLM
    and embedding servers alike. The preset and the model-id defaults are
    re-applied, so ``backend="local"`` behaves as if the file had said
    ``backend: local``. Returns ``config`` itself when nothing is given.
    """
    providers: dict[str, dict[str, Any]] = {}

    def put(section: str, key: str, value: Any) -> None:
        if value is not None:
            providers.setdefault(section, {})[key] = value

    put("llm", "kind", llm_kind)
    put("llm", "default_model_id", llm_model_id)
    put("llm", "base_url", base_url)
    put("embedding", "kind", embedding_kind)
    put("embedding", "model_id", embedding_model_id)
    put("embedding", "dimension", embedding_dimension)
    put("embedding", "base_url", base_url)
    put("rerank", "kind", rerank_kind)
    if backend is None and not providers:
        return config
    data = config.model_dump(exclude_unset=True)
    if backend is not None:
        data["backend"] = backend
    merged = data.get("providers") or {}
    for section, values in providers.items():
        merged[section] = {**(merged.get(section) or {}), **values}
    data["providers"] = merged
    try:
        return Config.model_validate(data)
    except ValueError as e:
        raise ConfigurationError(f"Invalid provider settings: {e}") from e
