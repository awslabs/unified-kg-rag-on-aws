# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import boto3
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from aws_graphrag.core import get_logger
from aws_graphrag.domain.models import (
    Config,
    Constants,
    ContextBuilderResult,
    RetrievalResult,
    SearchQuery,
    SearchResult,
)
from aws_graphrag.domain.retrieval.mixins import MetricsMixin

from .hybrid_scorer import HybridScorer
from .token_manager import TokenManager

logger = get_logger(__name__)


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
            logger.error(f"Document retrieval failed: {str(e)}")
            raise

    def retrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        return asyncio.run(self.aretrieve(query))

    @abstractmethod
    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        pass

    def batch_retrieve(self, queries: list[SearchQuery]) -> list[list[RetrievalResult]]:
        return [self.retrieve(query) for query in queries]

    async def abatch_retrieve(
        self, queries: list[SearchQuery]
    ) -> list[list[RetrievalResult]]:
        tasks = [self.aretrieve(query) for query in queries]
        return await asyncio.gather(*tasks)

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


class BaseContextBuilder(MetricsMixin, ABC):
    def __init__(self, config: Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config

    @abstractmethod
    def build_context(
        self,
        query: SearchQuery,
        retrieval_results: list[RetrievalResult],
    ) -> ContextBuilderResult:
        pass

    @abstractmethod
    async def abuild_context(
        self,
        query: SearchQuery,
        retrieval_results: list[RetrievalResult],
    ) -> ContextBuilderResult:
        pass


class BaseSearchStrategy(MetricsMixin, ABC):
    def __init__(
        self,
        config: Config,
        retrievers: dict[str, BaseGraphRAGRetriever],
        context_builder: BaseContextBuilder | None = None,
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
        self.context_builder = context_builder
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
        from aws_graphrag.domain.models import RetrieverRole

        return self.retrievers.get(RetrieverRole.GRAPH.value)

    @property
    def document_retriever(self) -> BaseGraphRAGRetriever | None:
        """The retriever bound to the DOCUMENT role (vector/lexical lookup)."""
        from aws_graphrag.domain.models import RetrieverRole

        return self.retrievers.get(RetrieverRole.DOCUMENT.value)

    def search(self, query: SearchQuery) -> SearchResult:
        return asyncio.run(self.asearch(query))

    @abstractmethod
    async def asearch(self, query: SearchQuery) -> SearchResult:
        pass

    def batch_search(self, queries: list[SearchQuery]) -> list[SearchResult]:
        return [self.search(query) for query in queries]

    async def abatch_search(self, queries: list[SearchQuery]) -> list[SearchResult]:
        tasks = [self.asearch(query) for query in queries]
        return await asyncio.gather(*tasks)

    @staticmethod
    def _get_ids(results: list[RetrievalResult], key: str) -> list[str]:
        ids_set: set[str] = set()
        for result in results:
            if ids := result.metadata.get(key):
                if isinstance(ids, list):
                    ids_set.update(str(id_val) for id_val in ids)
                else:
                    ids_set.add(str(ids))
        return list(ids_set)
