# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider bundle shared by every component of one chain/pipeline/evaluation.

``Providers`` holds the AWS session and the model-provider ports (LLM,
embedding, rerank, token counter). An orchestrator (``GraphRAGChain``,
``DataIngestionPipeline``, ``EvaluationManager``) builds it once from config —
or receives one — and passes it explicitly to the components it constructs, so
an injected provider reaches all of them and default clients are created once
instead of per component (or per query).

Anything not supplied is built lazily, on first use, with the Bedrock defaults,
so a configuration that never reranks never creates a rerank client. This is a
plain value object, not a DI container: there is no registry or scope, only the
handful of providers the framework actually consumes.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import boto3
from botocore.config import Config as BotoConfig

from unified_kg_rag.adapters.aws.bedrock import (
    BedrockEmbeddingModelFactory,
    BedrockLanguageModelFactory,
    BedrockRerankModelFactory,
    get_assumed_role_boto_session,
)
from unified_kg_rag.adapters.aws.token_counter import BedrockTokenCounter
from unified_kg_rag.domain.models import Config
from unified_kg_rag.ports.model_factory import (
    EmbeddingFactoryPort,
    LLMFactoryPort,
    RerankFactoryPort,
    TokenCounterPort,
)

TokenCounterFactory = Callable[..., TokenCounterPort]


class Providers:
    """Session + model-provider ports, each built at most once.

    Args:
        config: Framework config the Bedrock defaults are built from.
        boto_session: Session shared by every default client. Defaults to
            ``boto3.Session(profile_name=config.aws.profile_name)``.
        llm_factory / embedding_factory / rerank_factory: Model factories
            (any ``ModelFactoryPort``). Default to the Bedrock factories.
        token_counter_factory: Called as
            ``factory(model_id, cache_maxsize=..., api_supported=...)`` and
            returns a ``TokenCounterPort``. Defaults to ``BedrockTokenCounter``
            over one shared ``bedrock-runtime`` client.
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
    ) -> None:
        self.config = config
        self._boto_session = boto_session
        self._llm_factory = llm_factory
        self._embedding_factory = embedding_factory
        self._rerank_factory = rerank_factory
        self._token_counter_factory = token_counter_factory
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
                self._llm_factory = BedrockLanguageModelFactory(
                    config=self.config,
                    boto_session=self.boto_session,
                    region_name=self.config.aws.bedrock_region,
                )
            return self._llm_factory

    @property
    def embedding_factory(self) -> EmbeddingFactoryPort:
        with self._lock:
            if self._embedding_factory is None:
                self._embedding_factory = BedrockEmbeddingModelFactory(
                    config=self.config,
                    boto_session=self.boto_session,
                    region_name=self.config.aws.bedrock_region,
                )
            return self._embedding_factory

    @property
    def rerank_factory(self) -> RerankFactoryPort:
        with self._lock:
            if self._rerank_factory is None:
                self._rerank_factory = BedrockRerankModelFactory(
                    config=self.config,
                    boto_session=self.boto_session,
                    region_name=self.config.aws.bedrock_region,
                )
            return self._rerank_factory

    def token_counter(
        self, model_id: str, *, cache_maxsize: int = 1024, api_supported: bool = True
    ) -> TokenCounterPort:
        """A token counter for ``model_id`` (a new instance per call).

        Counters keep a per-instance LRU cache, so each consumer owns one; the
        default Bedrock counters share a single ``bedrock-runtime`` client.
        """
        if self._token_counter_factory is not None:
            return self._token_counter_factory(
                model_id, cache_maxsize=cache_maxsize, api_supported=api_supported
            )
        return BedrockTokenCounter(
            model_id=model_id,
            client=self._count_tokens_client() if api_supported else None,
            cache_maxsize=cache_maxsize,
            api_supported=api_supported,
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
                    config=BotoConfig(retries={"max_attempts": 3}),
                )
            return self._bedrock_runtime_client
