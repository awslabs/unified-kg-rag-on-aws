# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Composition root: the provider bundle for one run, built from config.

``build_providers`` checks that every role ``config.providers`` selects has a
registered builder (failing before any model is paid for) and returns the
``Providers`` bundle. ``GraphRAGChain``, ``DataIngestionPipeline`` and
``EvaluationManager`` call it when they are not given a bundle
(``providers=``) and hand the bundle on to every stage, strategy, scorer,
token manager, evaluator and store they build, so an injected bundle is the
single place to override a role.

``with_provider_overrides`` applies the CLI provider flags (``--backend``,
``--llm-provider`` ...) to a loaded config, re-running the ``backend`` preset
and the model-id defaults as if the file had set them; switching the backend
drops the values the old one's preset filled in. It rebuilds the config from
its input (``Config.as_input``): the file and the environment overrides, which
``ConfigLoader`` merges into the input before validating, plus any change
made to the object since, so the flags are the last layer over all of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from unified_kg_rag.adapters.provider_registry import (
    ProviderRole,
    get_provider_builder,
    provider_kind,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.config import validation_summary
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import merge_over
from unified_kg_rag.ports.guardrail import ContentGuardPort
from unified_kg_rag.shared import ConfigurationError, get_logger

if TYPE_CHECKING:
    import boto3

logger = get_logger(__name__)


def check_providers(config: Config) -> dict[ProviderRole, str]:
    """``{role: kind}`` for ``config``; every kind must be registered.

    Raises:
        ConfigurationError: A role names a kind nothing is registered for.
    """
    selected: dict[ProviderRole, str] = {}
    for role in ProviderRole:
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


def build_query_guard(config: Config, providers: Providers) -> ContentGuardPort | None:
    """The guard that screens queries and answers, or ``None`` when off.

    Built when ``aws.bedrock.guardrail.identifier`` is set and either check is
    on; it calls Bedrock ``ApplyGuardrail`` with the bundle's session,
    whichever providers serve the models.
    """
    guardrail = config.aws.bedrock.guardrail
    if not guardrail.enabled or not (guardrail.check_input or guardrail.check_output):
        return None
    from unified_kg_rag.adapters.guardrail.bedrock import BedrockGuardrail

    return BedrockGuardrail(config, providers.boto_session)


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
    and embedding servers alike. The config is rebuilt from its input
    (``Config.as_input``: the file and environment it was loaded from, and
    every change made to it since), so the preset and the model defaults are
    applied again: ``backend="local"`` behaves as if the file had said
    ``backend: local``, and ``backend="aws"`` over a ``backend: local`` file
    drops what the local preset filled in, but never a value the file, the
    environment or the caller set. Returns ``config`` itself when nothing is
    given.
    """
    overrides: dict[str, dict[str, Any]] = {}

    def put(path: str, value: Any) -> None:
        if value is None:
            return
        *parents, key = path.split(".")
        node = overrides
        for part in parents:
            node = node.setdefault(part, {})
        node[key] = value

    put("providers.llm.kind", llm_kind)
    put("providers.llm.base_url", base_url)
    put("providers.embedding.kind", embedding_kind)
    put("providers.embedding.base_url", base_url)
    put("providers.rerank.kind", rerank_kind)
    put("models.default", llm_model_id)
    put("models.embedding.model_id", embedding_model_id)
    put("models.embedding.dimension", embedding_dimension)
    if backend is None and not overrides:
        return config
    data = config.as_input()
    if backend is not None:
        data["backend"] = backend
    data = merge_over(data, overrides)
    try:
        return Config.model_validate(data)
    except ValidationError as e:
        raise ConfigurationError(
            f"Invalid provider settings: {validation_summary(e)}"
        ) from e
