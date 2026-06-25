# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retrieval error visibility (AWS-free).

Regression: the retrievers' top-level ``except Exception: return []`` turned a
transient/auth/config/connection error into "no results found", which the user
cannot distinguish from a genuine empty match. The retrievers now re-raise
clearly-fatal errors (auth/credentials/endpoint/connection) and degrade to an
empty list only on genuinely-transient failures.
"""

from __future__ import annotations

import pytest

from aws_graphrag.adapters.retrieval.base import is_fatal_retrieval_error
from aws_graphrag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from aws_graphrag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from aws_graphrag.domain.models import Config, SearchQuery
from aws_graphrag.shared import AWSServiceError

pytestmark = pytest.mark.unit


# --- classifier ----------------------------------------------------------


def test_connection_error_is_fatal() -> None:
    assert is_fatal_retrieval_error(ConnectionError("refused")) is True


@pytest.mark.parametrize(
    "message",
    [
        "Cannot get AWS credentials for OpenSearch IAM.",
        "OpenSearch endpoint is not configured.",
        "Failed to connect to OpenSearch.",
        "Failed to establish connection to Neptune: timeout",
        "403 Forbidden",
        "AccessDenied",
    ],
)
def test_misconfiguration_messages_are_fatal(message: str) -> None:
    assert is_fatal_retrieval_error(AWSServiceError(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "Read timed out",
        "429 Too Many Requests",
        "malformed query syntax",
    ],
)
def test_transient_messages_are_not_fatal(message: str) -> None:
    assert is_fatal_retrieval_error(AWSServiceError(message)) is False


# --- OpenSearchRetriever top-level handler -------------------------------


def _opensearch_retriever(config: Config) -> OpenSearchRetriever:
    inst = OpenSearchRetriever.__new__(OpenSearchRetriever)
    object.__setattr__(inst, "_config", config)
    object.__setattr__(inst, "_opensearch_config", config.indexing.opensearch)
    object.__setattr__(inst, "_max_size", config.indexing.opensearch.max_query_size)
    object.__setattr__(
        inst, "_terms_batch_size", config.indexing.opensearch.terms_batch_size
    )
    object.__setattr__(inst, "_field_mappings", inst._initialize_field_mappings())
    object.__setattr__(inst, "_record_timing", lambda *a, **k: None)
    object.__setattr__(inst, "_record_metric", lambda *a, **k: None)
    return inst


def _set_query_vector_raising(retriever: OpenSearchRetriever, exc: Exception) -> None:
    # Inject via object.__setattr__: BaseRetriever is a pydantic model and
    # mocker.patch.object's teardown (delattr) trips its __delattr__.
    async def _raise(*_args, **_kwargs):
        raise exc

    object.__setattr__(retriever, "_get_query_vector", _raise)


async def test_opensearch_retriever_reraises_fatal(config: Config) -> None:
    retriever = _opensearch_retriever(config)
    # Make the very first awaited step (query-vector embedding) blow up with a
    # fatal credentials error.
    _set_query_vector_raising(
        retriever,
        AWSServiceError("Cannot get AWS credentials for OpenSearch IAM."),
    )
    with pytest.raises(AWSServiceError, match="credentials"):
        await retriever.aretrieve(SearchQuery(query="hello"))


async def test_opensearch_retriever_degrades_on_transient(config: Config) -> None:
    retriever = _opensearch_retriever(config)
    _set_query_vector_raising(retriever, AWSServiceError("Read timed out"))
    results = await retriever.aretrieve(SearchQuery(query="hello"))
    assert results == []


def _set_asearch_raising(retriever: OpenSearchRetriever, exc: Exception) -> None:
    # Let the query-vector step succeed, then make the actual search execution
    # (``_opensearch_client.asearch``) raise — this is the path that previously
    # swallowed fatal errors inside ``_execute_search`` and returned [].
    async def _vec(*_args, **_kwargs):
        return [0.0] * 8

    async def _asearch(*_args, **_kwargs):
        raise exc

    object.__setattr__(retriever, "_get_query_vector", _vec)
    object.__setattr__(retriever, "_opensearch_client", _MockSearchClient(_asearch))


class _MockSearchClient:
    def __init__(self, asearch) -> None:
        self.asearch = asearch


async def test_opensearch_execute_search_reraises_fatal(config: Config) -> None:
    # Regression: a fatal error raised by the real asearch call must propagate
    # out of _execute_search, not be swallowed into an empty result list.
    retriever = _opensearch_retriever(config)
    _set_asearch_raising(
        retriever, AWSServiceError("The security token included is invalid.")
    )
    with pytest.raises(AWSServiceError, match="security token"):
        await retriever.aretrieve(SearchQuery(query="hello"))


async def test_opensearch_execute_search_degrades_on_transient(config: Config) -> None:
    retriever = _opensearch_retriever(config)
    _set_asearch_raising(retriever, AWSServiceError("Read timed out"))
    results = await retriever.aretrieve(SearchQuery(query="hello"))
    assert results == []


# --- NeptuneRetriever top-level handler ----------------------------------


def _neptune_retriever(config: Config, neptune_client) -> NeptuneRetriever:
    inst = NeptuneRetriever.__new__(NeptuneRetriever)
    object.__setattr__(inst, "_config", config)
    object.__setattr__(inst, "_neptune_config", config.indexing.neptune)
    object.__setattr__(inst, "_neptune_client", neptune_client)
    object.__setattr__(inst, "_max_hops", config.indexing.neptune.max_hops)
    object.__setattr__(
        inst, "_max_results_per_hop", config.indexing.neptune.max_results_per_hop
    )
    object.__setattr__(
        inst, "_min_entity_importance", config.indexing.neptune.min_entity_importance
    )
    # Stub out the metrics recorders (pydantic model -> inject, don't patch).
    object.__setattr__(inst, "_record_metric", lambda *a, **k: None)
    object.__setattr__(inst, "_record_timing", lambda *a, **k: None)
    return inst


class _ClientRaisingOnG:
    """A fake NeptuneClient whose ``.g`` property raises the given exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    @property
    def g(self):
        raise self._exc


async def test_neptune_retriever_reraises_fatal(config: Config) -> None:
    client = _ClientRaisingOnG(
        AWSServiceError("Failed to establish connection to Neptune: down")
    )
    retriever = _neptune_retriever(config, client)
    with pytest.raises(AWSServiceError, match="connection"):
        await retriever.aretrieve(SearchQuery(query="hello"))


async def test_neptune_retriever_degrades_on_transient(config: Config) -> None:
    client = _ClientRaisingOnG(AWSServiceError("Read timed out"))
    retriever = _neptune_retriever(config, client)
    results = await retriever.aretrieve(SearchQuery(query="hello"))
    assert results == []
