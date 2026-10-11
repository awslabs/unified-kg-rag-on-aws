# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider bundle shared by every component of one chain/pipeline/evaluation.

``Providers`` holds the AWS session, the model-provider ports (LLM,
embedding, rerank, token counter) and the store builders (graph and vector
indexers and retrievers, the doc-status registry). An orchestrator
(``GraphRAGChain``, ``DataIngestionPipeline``, ``EvaluationManager``) builds
it once from config — ``application.bootstrap.build_providers`` is the
composition root — or receives one, and passes it explicitly to the
components it constructs, so an injected provider reaches all of them and
default clients are created once instead of per component (or per query).

Anything not injected is built lazily, on first use, by the builder that
``adapters.provider_registry`` holds for the kind ``config.providers`` selects
for the role. The defaults are the AWS services (Bedrock, Neptune, OpenSearch,
DynamoDB), so a configuration that never reranks never creates a rerank
client, and one that selects local providers never creates an AWS client.
This is a value object with a registry lookup, not a DI container.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import boto3
from botocore.config import Config as BotoConfig

from unified_kg_rag.adapters.aws.bedrock import (
    BaseBedrockModelFactory,
    BedrockEmbeddingModelFactory,
    BedrockLanguageModelFactory,
    BedrockRerankModelFactory,
    get_assumed_role_boto_session,
)
from unified_kg_rag.adapters.aws.bedrock_models import (
    LanguageModelInfo,
    get_language_model_info,
)
from unified_kg_rag.adapters.aws.token_counter import BedrockTokenCounter
from unified_kg_rag.adapters.provider_registry import (
    ProviderRole,
    get_provider_builder,
    provider_kind,
    register_provider,
)
from unified_kg_rag.domain.models import Config, RetrieverRole
from unified_kg_rag.domain.models.provider_config import (
    BEDROCK,
    CROSS_ENCODER,
    DYNAMODB,
    GREMLIN,
    NEPTUNE,
    NO_RERANK,
    OLLAMA,
    OPENAI_COMPATIBLE,
    OPENSEARCH,
    SQLITE,
)
from unified_kg_rag.ports.model_factory import (
    EmbeddingFactoryPort,
    LLMFactoryPort,
    RerankFactoryPort,
    TokenCounterPort,
)

if TYPE_CHECKING:
    from unified_kg_rag.adapters.retrieval.base import BaseGraphRAGRetriever
    from unified_kg_rag.ports.doc_status import DocStatusPort
    from unified_kg_rag.ports.indexer import GraphIndexer, VectorIndexer

TokenCounterFactory = Callable[..., TokenCounterPort]

_RETRIEVER_ROLES: dict[RetrieverRole, ProviderRole] = {
    RetrieverRole.GRAPH: ProviderRole.GRAPH_RETRIEVER,
    RetrieverRole.DOCUMENT: ProviderRole.VECTOR_RETRIEVER,
}


class Providers:
    """Session, model-provider ports and store builders, each built once.

    Args:
        config: Framework config; ``config.providers`` selects the kind built
            for every role that is not injected.
        boto_session: Session shared by every AWS client. Defaults to
            ``boto3.Session(profile_name=config.aws.profile_name)``; building it
            makes no AWS call.
        llm_factory / embedding_factory / rerank_factory: Model factories
            (any ``ModelFactoryPort``).
        token_counter_factory: Called as
            ``factory(model_id, cache_maxsize=..., api_supported=...)`` and
            returns a ``TokenCounterPort``.
        graph_indexer_factory / vector_indexer_factory: Zero-argument callables
            returning a ``GraphIndexer`` / ``VectorIndexer`` (one per call).
        retriever_builders: Zero-argument retriever builders by
            ``RetrieverRole``, as accepted by ``GraphRAGChain``.
        doc_status: The doc-status registry (``DocStatusPort``), shared by
            every run that uses this bundle.
    """

    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        *,
        llm_factory: LLMFactoryPort | None = None,
        embedding_factory: EmbeddingFactoryPort | None = None,
        rerank_factory: RerankFactoryPort | None = None,
        token_counter_factory: TokenCounterFactory | None = None,
        graph_indexer_factory: Callable[[], GraphIndexer] | None = None,
        vector_indexer_factory: Callable[[], VectorIndexer] | None = None,
        retriever_builders: (
            Mapping[RetrieverRole, Callable[[], BaseGraphRAGRetriever]] | None
        ) = None,
        doc_status: DocStatusPort | None = None,
    ) -> None:
        self.config = config
        self._boto_session = boto_session
        self._llm_factory = llm_factory
        self._embedding_factory = embedding_factory
        self._rerank_factory = rerank_factory
        self._token_counter_factory = token_counter_factory
        self._graph_indexer_factory = graph_indexer_factory
        self._vector_indexer_factory = vector_indexer_factory
        self._retriever_builders = dict(retriever_builders or {})
        self._doc_status = doc_status
        self._bedrock_runtime_client: Any = None
        # Re-entrant: building a factory reads ``boto_session`` under the lock.
        self._lock = threading.RLock()

    @classmethod
    def resolve(
        cls,
        config: Config,
        providers: Providers | None = None,
        boto_session: boto3.Session | None = None,
    ) -> Providers:
        """The given bundle, or a default one over ``boto_session``."""
        return providers if providers is not None else cls(config, boto_session)

    def kind(self, role: ProviderRole) -> str:
        """The provider kind the config selects for ``role``."""
        return provider_kind(self.config, role)

    def _build(self, role: ProviderRole) -> Any:
        return get_provider_builder(role, self.kind(role))(self)

    @property
    def boto_session(self) -> boto3.Session:
        with self._lock:
            if self._boto_session is None:
                self._boto_session = boto3.Session(
                    profile_name=self.config.aws.profile_name
                )
            return self._boto_session

    @property
    def llm_factory(self) -> LLMFactoryPort:
        with self._lock:
            if self._llm_factory is None:
                self._llm_factory = self._build(ProviderRole.LLM)
            return self._llm_factory

    @property
    def embedding_factory(self) -> EmbeddingFactoryPort:
        with self._lock:
            if self._embedding_factory is None:
                self._embedding_factory = self._build(ProviderRole.EMBEDDING)
            return self._embedding_factory

    @property
    def rerank_factory(self) -> RerankFactoryPort:
        with self._lock:
            if self._rerank_factory is None:
                self._rerank_factory = self._build(ProviderRole.RERANK)
            return self._rerank_factory

    def token_counter(
        self, model_id: str, *, cache_maxsize: int = 1024, api_supported: bool = True
    ) -> TokenCounterPort:
        """A token counter for ``model_id`` (a new instance per call).

        Counters keep a per-instance LRU cache, so each consumer owns one; the
        default Bedrock counters share a single ``bedrock-runtime`` client and
        scale their local estimate by the model's ``token_estimate_multiplier``.
        """
        with self._lock:
            if self._token_counter_factory is None:
                self._token_counter_factory = self._build(ProviderRole.TOKEN_COUNTER)
            factory = self._token_counter_factory
        return factory(
            model_id, cache_maxsize=cache_maxsize, api_supported=api_supported
        )

    def llm_model_info(self, model_id: str) -> LanguageModelInfo:
        """Capability record of the language model ``model_id``.

        From the LLM factory when it returns one; the Bedrock catalog
        otherwise. With the Bedrock provider and no injected factory the
        catalog is read directly, so sizing a budget never builds a client.
        """
        if self._llm_factory is None and self.kind(ProviderRole.LLM) == BEDROCK:
            return get_language_model_info(
                model_id, self.config.aws.bedrock.model_overrides
            )
        info = self.llm_factory.get_model_info(model_id)
        if isinstance(info, LanguageModelInfo):
            return info
        return get_language_model_info(
            model_id, self.config.aws.bedrock.model_overrides
        )

    # --- stores --------------------------------------------------------------

    def build_graph_indexer(self) -> GraphIndexer:
        """A new graph-store writer (the caller closes it)."""
        if self._graph_indexer_factory is not None:
            return self._graph_indexer_factory()
        indexer: GraphIndexer = self._build(ProviderRole.GRAPH_INDEXER)
        return indexer

    def build_vector_indexer(self) -> VectorIndexer:
        """A new vector/lexical-store writer (the caller closes it)."""
        if self._vector_indexer_factory is not None:
            return self._vector_indexer_factory()
        indexer: VectorIndexer = self._build(ProviderRole.VECTOR_INDEXER)
        return indexer

    def build_retriever(self, role: RetrieverRole) -> BaseGraphRAGRetriever:
        """A new retriever bound to ``role`` (the caller closes it)."""
        builder = self._retriever_builders.get(role)
        if builder is not None:
            return builder()
        provider_role = _RETRIEVER_ROLES.get(RetrieverRole(role))
        if provider_role is None:
            raise ValueError(f"No retriever bound to role: '{role}'")
        retriever: BaseGraphRAGRetriever = self._build(provider_role)
        return retriever

    def build_doc_status(self) -> DocStatusPort:
        """The injected doc-status registry, else a new one of the configured kind.

        A new instance per call, like the indexers: a registry adapter keeps
        per-run state (the DynamoDB store remembers the overflow its last
        scan found), so two runs sharing a bundle must not share one. The
        pipeline builds it once and hands it to its stages. Building it does
        no I/O; whether incremental indexing uses it is
        ``config.doc_status_enabled``.
        """
        if self._doc_status is not None:
            return self._doc_status
        store: DocStatusPort = self._build(ProviderRole.DOC_STATUS)
        return store

    # --- Bedrock token counting ----------------------------------------------

    def _bedrock_token_counter(
        self, model_id: str, *, cache_maxsize: int = 1024, api_supported: bool = True
    ) -> TokenCounterPort:
        info = get_language_model_info(
            model_id, self.config.aws.bedrock.model_overrides
        )
        return BedrockTokenCounter(
            model_id=model_id,
            client=self._count_tokens_client() if api_supported else None,
            cache_maxsize=cache_maxsize,
            api_supported=api_supported,
            estimate_multiplier=info.token_estimate_multiplier,
        )

    def _count_tokens_client(self) -> Any:
        with self._lock:
            if self._bedrock_runtime_client is None:
                session = get_assumed_role_boto_session(
                    self.boto_session,
                    assumed_role_arn=self.config.aws.bedrock.assumed_role_arn,
                )
                self._bedrock_runtime_client = session.client(
                    "bedrock-runtime",
                    region_name=self.config.aws.bedrock_region,
                    # Sized like the model clients: every query's
                    # count_tokens_many sends several counts at once.
                    config=BotoConfig(
                        retries={"max_attempts": 3},
                        max_pool_connections=(
                            BaseBedrockModelFactory.max_pool_connections(self.config)
                        ),
                    ),
                )
            return self._bedrock_runtime_client


# --- built-in providers -------------------------------------------------------
# The Bedrock builders read the factory classes from this module's globals, so
# a test can patch them here. Every other builder imports its adapter on first
# use: selecting a provider never imports another provider's dependencies.


@register_provider(ProviderRole.LLM, BEDROCK)
def _bedrock_llm(providers: Providers) -> LLMFactoryPort:
    return BedrockLanguageModelFactory(
        config=providers.config,
        boto_session=providers.boto_session,
        region_name=providers.config.aws.bedrock_region,
    )


@register_provider(ProviderRole.EMBEDDING, BEDROCK)
def _bedrock_embedding(providers: Providers) -> EmbeddingFactoryPort:
    return BedrockEmbeddingModelFactory(
        config=providers.config,
        boto_session=providers.boto_session,
        region_name=providers.config.aws.bedrock_region,
    )


@register_provider(ProviderRole.RERANK, BEDROCK)
def _bedrock_rerank(providers: Providers) -> RerankFactoryPort:
    return BedrockRerankModelFactory(
        config=providers.config,
        boto_session=providers.boto_session,
        region_name=providers.config.aws.bedrock_region,
    )


@register_provider(ProviderRole.TOKEN_COUNTER, BEDROCK)
def _bedrock_token_counter(providers: Providers) -> TokenCounterFactory:
    return providers._bedrock_token_counter


@register_provider(ProviderRole.LLM, OPENAI_COMPATIBLE)
def _openai_compatible_llm(providers: Providers) -> LLMFactoryPort:
    from unified_kg_rag.adapters.local.models import (
        OpenAICompatibleLanguageModelFactory,
    )

    return OpenAICompatibleLanguageModelFactory(providers.config)


@register_provider(ProviderRole.LLM, OLLAMA)
def _ollama_llm(providers: Providers) -> LLMFactoryPort:
    from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory

    return OllamaLanguageModelFactory(providers.config)


@register_provider(ProviderRole.TOKEN_COUNTER, OPENAI_COMPATIBLE)
@register_provider(ProviderRole.TOKEN_COUNTER, OLLAMA)
def _local_token_counter(providers: Providers) -> TokenCounterFactory:
    from unified_kg_rag.adapters.local.token_counter import (
        local_token_counter_factory,
    )

    factory: TokenCounterFactory = local_token_counter_factory(providers.config)
    return factory


@register_provider(ProviderRole.EMBEDDING, OPENAI_COMPATIBLE)
def _openai_compatible_embedding(providers: Providers) -> EmbeddingFactoryPort:
    from unified_kg_rag.adapters.local.embeddings import (
        OpenAICompatibleEmbeddingFactory,
    )

    return OpenAICompatibleEmbeddingFactory(providers.config)


@register_provider(ProviderRole.EMBEDDING, OLLAMA)
def _ollama_embedding(providers: Providers) -> EmbeddingFactoryPort:
    from unified_kg_rag.adapters.local.embeddings import OllamaEmbeddingFactory

    return OllamaEmbeddingFactory(providers.config)


@register_provider(ProviderRole.RERANK, CROSS_ENCODER)
def _cross_encoder_rerank(providers: Providers) -> RerankFactoryPort:
    from unified_kg_rag.adapters.local.rerank import CrossEncoderRerankFactory

    return CrossEncoderRerankFactory(providers.config)


@register_provider(ProviderRole.RERANK, NO_RERANK)
def _no_rerank(providers: Providers) -> RerankFactoryPort:
    from unified_kg_rag.adapters.local.rerank import NoRerankFactory

    return NoRerankFactory(providers.config)


# Gremlin Server speaks the same TinkerPop protocol as Neptune; the config
# validator requires aws.neptune.use_iam: false for it (no SigV4).
@register_provider(ProviderRole.GRAPH_INDEXER, NEPTUNE)
@register_provider(ProviderRole.GRAPH_INDEXER, GREMLIN)
def _neptune_graph_indexer(providers: Providers) -> GraphIndexer:
    from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer

    return NeptuneIndexer(config=providers.config)


@register_provider(ProviderRole.GRAPH_RETRIEVER, NEPTUNE)
@register_provider(ProviderRole.GRAPH_RETRIEVER, GREMLIN)
def _neptune_graph_retriever(providers: Providers) -> BaseGraphRAGRetriever:
    from unified_kg_rag.adapters.aws.neptune import NeptuneClient
    from unified_kg_rag.adapters.retrievers.neptune_retriever import (
        NeptuneRetriever,
    )

    return NeptuneRetriever(
        config=providers.config,
        neptune_client=NeptuneClient(
            config=providers.config, boto_session=providers.boto_session
        ),
        boto_session=providers.boto_session,
    )


@register_provider(ProviderRole.VECTOR_INDEXER, OPENSEARCH)
def _opensearch_vector_indexer(providers: Providers) -> VectorIndexer:
    from unified_kg_rag.adapters.storage.opensearch_indexer import (
        OpenSearchIndexer,
    )

    return OpenSearchIndexer(
        config=providers.config,
        boto_session=providers.boto_session,
        embedding_factory=providers.embedding_factory,
    )


@register_provider(ProviderRole.VECTOR_RETRIEVER, OPENSEARCH)
def _opensearch_vector_retriever(providers: Providers) -> BaseGraphRAGRetriever:
    from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient
    from unified_kg_rag.adapters.retrievers.opensearch_retriever import (
        OpenSearchRetriever,
    )

    return OpenSearchRetriever(
        config=providers.config,
        opensearch_client=OpenSearchClient(
            config=providers.config, boto_session=providers.boto_session
        ),
        boto_session=providers.boto_session,
        embedding_factory=providers.embedding_factory,
    )


@register_provider(ProviderRole.DOC_STATUS, DYNAMODB)
def _dynamodb_doc_status(providers: Providers) -> DocStatusPort:
    from unified_kg_rag.adapters.aws.dynamodb import DynamoDBDocStatusStore

    return DynamoDBDocStatusStore(providers.config, boto_session=providers.boto_session)


@register_provider(ProviderRole.DOC_STATUS, SQLITE)
def _sqlite_doc_status(providers: Providers) -> DocStatusPort:
    from unified_kg_rag.adapters.local.sqlite_doc_status import SQLiteDocStatusStore

    return SQLiteDocStatusStore(providers.config)
