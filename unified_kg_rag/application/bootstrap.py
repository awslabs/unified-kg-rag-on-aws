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

``build_query_guard`` and ``build_checkpoint_sync`` build the other
backends a run wires (the Bedrock guardrail, the S3 sync of the stage
cache), so no orchestrator imports a vendor binding.

``resolve_extraction_schema`` loads the extraction schema a registered
source names (``processing.graph_extraction.schema_source``) once, into the
inline keys of a copy of the config, so every stage, prompt and cache key
reads the schema from the inline keys alone.

``with_provider_overrides`` applies the CLI provider flags (``--backend``,
``--llm-provider`` ...) to a loaded config, re-running the ``backend`` preset
and the model-id defaults as if the file had set them; switching the backend
drops the values the old one's preset filled in. It rebuilds the config from
its input (``Config.as_input``): the file and the environment overrides, which
``ConfigLoader`` merges into the input before validating, plus any change
made to the object since, so the flags are the last layer over all of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import ValidationError

from unified_kg_rag.adapters.provider_registry import (
    ProviderRole,
    get_provider_builder,
    provider_kind,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.schema_sources import get_schema_source
from unified_kg_rag.application.config import validation_summary
from unified_kg_rag.domain.models import Config, PipelineConfig
from unified_kg_rag.domain.models.config import INLINE_SCHEMA_SOURCE, merge_over
from unified_kg_rag.ports.guardrail import ContentGuardPort
from unified_kg_rag.shared import ConfigurationError, get_logger
from unified_kg_rag.shared.exceptions import SchemaSourceError

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


class CheckpointSync(Protocol):
    """Remote storage of a run's stage cache (``pipeline.s3_sync_enabled``).

    Each method raises ``CacheSyncError`` when a transfer fails.
    """

    def upload_file(
        self, pipeline_id: str, local_cache_dir: Path, local_path: Path
    ) -> None: ...

    def sync_pipeline_to_s3(self, pipeline_id: str, local_cache_dir: Path) -> None: ...

    def sync_pipeline_from_s3(
        self, pipeline_id: str, local_cache_dir: Path
    ) -> None: ...


def build_checkpoint_sync(
    config: Config, pipeline_config: PipelineConfig, providers: Providers
) -> CheckpointSync | None:
    """The S3 sync of the run's stage cache, or ``None`` when it is off."""
    if not pipeline_config.s3_sync_enabled:
        return None
    # Imported here: a run without S3 sync never loads the AWS SDK.
    from unified_kg_rag.adapters.aws.s3_cache import S3CacheManager

    return S3CacheManager(
        config=config,
        boto_session=providers.boto_session,
        bucket_name=pipeline_config.s3_bucket_name,
        prefix=pipeline_config.s3_prefix,
    )


def resolve_extraction_schema(config: Config) -> Config:
    """``config`` with its extraction schema in the inline keys.

    With ``schema_source.kind`` ``inline`` (the default) that is ``config``
    itself. Otherwise the registered source is loaded once and its keys
    (``relationship_types``, ``relationship_patterns`` and, when it has
    them, ``entity_types``) are written into a copy, whose source is then
    ``inline``: the loaded content keys the extraction caches, the location
    does not, and loading again changes nothing.

    Raises:
        SchemaSourceError: The kind is not registered, the source cannot be
            read, both it and the config set ``entity_types``, or the config
            refuses the loaded schema.
    """
    extraction = config.processing.graph_extraction
    source = extraction.schema_source
    if source.kind == INLINE_SCHEMA_SOURCE:
        return config
    document = get_schema_source(source.kind)(source)
    where = f"schema_source ({source.kind} '{source.path}')"
    keys: dict[str, Any] = document.model_dump(mode="json", exclude_none=True)
    if "entity_types" in keys and "entity_types" in extraction.model_fields_set:
        raise SchemaSourceError(
            f"Both the config and {where} set entity_types; keep one"
        )
    keys["schema_source"] = {"kind": INLINE_SCHEMA_SOURCE, "path": None}
    data = merge_over(config.as_input(), {"processing": {"graph_extraction": keys}})
    try:
        resolved = Config.model_validate(data)
    except ValidationError as e:
        raise SchemaSourceError(
            f"The schema of {where} is invalid: {validation_summary(e)}"
        ) from e
    schema = resolved.processing.graph_extraction.extraction_schema()
    logger.info(
        "Extraction schema from %s: %d entity types, %d relationship types, "
        "%d patterns (digest %s)",
        where,
        len(schema.entity_types),
        len(schema.relationship_types),
        len(schema.relationship_patterns),
        schema.digest[:12],
    )
    return resolved


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
