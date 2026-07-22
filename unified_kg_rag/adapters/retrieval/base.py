# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import boto3
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from unified_kg_rag.adapters.retrieval.hybrid_scorer import HybridScorer
from unified_kg_rag.adapters.retrieval.token_manager import TokenManager
from unified_kg_rag.domain.models import (
    Config,
    Constants,
    RetrievalResult,
    SearchQuery,
    SearchResult,
)
from unified_kg_rag.domain.retrieval.mixins import MetricsMixin
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

# Substrings in an error message that indicate a non-transient (fatal)
# misconfiguration — auth, credentials, endpoint/config, or an outright
# connection failure. A retriever should surface these to the caller rather
# than masking them as "0 results", which is indistinguishable from a genuine
# empty match and silently hides broken auth/config from the user.
_FATAL_ERROR_MARKERS: tuple[str, ...] = (
    "credential",
    "not configured",
    "endpoint",
    "auth",
    "forbidden",
    "unauthorized",
    "access denied",
    "accessdenied",
    "security token",
    "expiredtoken",
    "expired token",
    "invalidclienttoken",
    "failed to connect",
    "failed to establish connection",
)


def is_fatal_retrieval_error(exc: BaseException) -> bool:
    """Classify a retrieval error as fatal (re-raise) vs transient (degrade).

    Fatal = a clearly non-recoverable misconfiguration (auth/credentials/
    endpoint/config) or a connection failure: returning ``[]`` for these turns a
    real failure into a misleading "no results". Everything else (timeouts,
    throttling, a single malformed query) is treated as transient and the caller
    may degrade to an empty result list. ``ConnectionError`` is always fatal.
    The repo's adapters wrap backend failures in ``AWSServiceError`` carrying the
    original message, so the markers are matched against the message text.
    """
    if isinstance(exc, ConnectionError):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _FATAL_ERROR_MARKERS)


class BaseGraphRAGRetriever(BaseRetriever, MetricsMixin, ABC):
    def __init__(
        self, config: Config, boto_session: boto3.Session | None = None, **kwargs: Any
    ) -> None:
        BaseRetriever.__init__(self, **kwargs)
        MetricsMixin.__init__(self, **kwargs)
        self._config = config
        self._boto_session = boto_session or boto3.Session(
            profile_name=self._config.aws.profile_name
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        try:
            search_query = SearchQuery(query=query)
            results = asyncio.run(self.aretrieve(search_query))

            documents = []
            for result in results:
                doc = Document(
                    page_content=result.content,
                    metadata={
                        "score": result.score,
                        "source": result.source,
                        "retriever_type": result.retriever_type,
                        **(result.metadata or {}),
                    },
                )
                documents.append(doc)

            return documents
        except Exception as e:
            logger.error("Document retrieval failed: %s", str(e))
            raise

    def retrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        return asyncio.run(self.aretrieve(query))

    @abstractmethod
    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        pass

    def _get_name(
        self, base: str, suffix: str | None, add_timestamp: bool = False
    ) -> str:
        final_suffix = suffix or Constants.DEFAULT_SUFFIX.value

        if self._config.indexing.additional_suffix:
            final_suffix = f"{final_suffix}-{self._config.indexing.additional_suffix}"

        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            return f"{base}-{final_suffix}-{timestamp}"

        return f"{base}-{final_suffix}"


class BaseSearchStrategy(MetricsMixin, ABC):
    def __init__(
        self,
        config: Config,
        retrievers: dict[str, BaseGraphRAGRetriever],
        boto_session: boto3.Session | None = None,
        optimization_threshold_factor: int = 2,
        default_max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config
        # Retrievers are keyed by RetrieverRole value ("graph" / "document"),
        # not by concrete backend, so strategies stay backend-agnostic.
        self.retrievers = retrievers
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name
        )
        self.hybrid_scorer = HybridScorer(self.config, boto_session=self.boto_session)
        self.token_manager = TokenManager(self.config, boto_session=self.boto_session)
        self.optimization_threshold_factor = optimization_threshold_factor
        self.default_max_tokens = default_max_tokens

    @property
    def graph_retriever(self) -> BaseGraphRAGRetriever | None:
        """The retriever bound to the GRAPH role (graph traversal/expansion)."""
        from unified_kg_rag.domain.models import RetrieverRole

        return self.retrievers.get(RetrieverRole.GRAPH.value)

    @property
    def document_retriever(self) -> BaseGraphRAGRetriever | None:
        """The retriever bound to the DOCUMENT role (vector/lexical lookup)."""
        from unified_kg_rag.domain.models import RetrieverRole

        return self.retrievers.get(RetrieverRole.DOCUMENT.value)

    def search(self, query: SearchQuery) -> SearchResult:
        return asyncio.run(self.asearch(query))

    @abstractmethod
    async def asearch(self, query: SearchQuery) -> SearchResult:
        pass

    @staticmethod
    def _get_ids(results: list[RetrievalResult], key: str) -> list[str]:
        ids_set: set[str] = set()
        for result in results:
            ids = result.metadata.get(key)
            # OpenSearch hits carry the canonical id in `source` and only echo
            # metadata[key] when the indexed _source happens to include that
            # field; fall back to `source` so graph expansion still gets seeds
            # (matching local/drift candidate-entity resolution).
            if not ids and key == "id" and result.source:
                ids = result.source
            if ids:
                if isinstance(ids, list):
                    ids_set.update(str(id_val) for id_val in ids)
                else:
                    ids_set.add(str(ids))
        return list(ids_set)
