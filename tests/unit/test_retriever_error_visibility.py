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

import weakref

import pytest
from gremlin_python.process.graph_traversal import GraphTraversal
from gremlin_python.structure.graph import Graph
from opensearchpy.exceptions import NotFoundError

from unified_kg_rag.adapters.retrieval.base import is_fatal_retrieval_error
from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from unified_kg_rag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from unified_kg_rag.domain.models import Config, Constants, SearchQuery
from unified_kg_rag.shared import AWSServiceError, IndexNotFoundError

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


def _set_asearch_raising(
    retriever: OpenSearchRetriever, exc: Exception, existing: set[str] | None = None
) -> None:
    # Let the query-vector step succeed, then make the actual search execution
    # (``_opensearch_client.asearch``) raise — this is the path that previously
    # swallowed fatal errors inside ``_execute_search`` and returned [].
    async def _vec(*_args, **_kwargs):
        return [0.0] * 8

    async def _asearch(*_args, **_kwargs):
        raise exc

    object.__setattr__(retriever, "_get_query_vector", _vec)
    object.__setattr__(
        retriever, "_opensearch_client", _MockSearchClient(_asearch, existing)
    )


class _MockSearchClient:
    def __init__(self, asearch, existing: set[str] | None = None) -> None:
        self.asearch = asearch
        self.existing = existing  # None = every index exists
        self.exists_calls: list[str] = []

    async def aindex_exists(self, index: str) -> bool:
        self.exists_calls.append(index)
        return self.existing is None or index in self.existing


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
    object.__setattr__(inst, "_pool_size", config.aws.neptune.pool_size)
    object.__setattr__(inst, "_traversal_slots", weakref.WeakKeyDictionary())
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


# --- NeptuneRetriever._execute_traversal ------------------------------------
#
# Every Gremlin round trip goes through `_execute_traversal`, which used to
# catch everything and return [] — so a fatal error raised by the actual query
# (rather than by `.g`) became "no seed nodes found" and never reached the
# top-level re-raise.


class _RaisingTraversal:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def to_list(self) -> list:
        raise self._exc


async def test_neptune_execute_traversal_reraises_fatal(config: Config) -> None:
    retriever = _neptune_retriever(config, None)
    with pytest.raises(AWSServiceError, match="AccessDenied"):
        await retriever._execute_traversal(
            _RaisingTraversal(AWSServiceError("AccessDeniedException"))  # type: ignore[arg-type]
        )


async def test_neptune_execute_traversal_degrades_on_transient(
    config: Config,
) -> None:
    retriever = _neptune_retriever(config, None)
    results = await retriever._execute_traversal(
        _RaisingTraversal(AWSServiceError("Read timed out"))  # type: ignore[arg-type]
    )
    assert results == []


class _ClientWithLocalGraph:
    """A fake NeptuneClient exposing a real, connectionless traversal source."""

    @property
    def g(self):
        return Graph().traversal()


async def test_neptune_fatal_traversal_error_surfaces_from_aretrieve(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fatal_to_list(self) -> list:
        raise ConnectionError("Failed to connect to the Neptune endpoint")

    monkeypatch.setattr(GraphTraversal, "to_list", _fatal_to_list)
    retriever = _neptune_retriever(config, _ClientWithLocalGraph())
    with pytest.raises(ConnectionError, match="Neptune endpoint"):
        await retriever.aretrieve(SearchQuery(query="hello", entity_focus=["Vendor"]))


async def test_neptune_transient_traversal_error_degrades_from_aretrieve(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _transient_to_list(self) -> list:
        raise RuntimeError("Read timed out")

    monkeypatch.setattr(GraphTraversal, "to_list", _transient_to_list)
    retriever = _neptune_retriever(config, _ClientWithLocalGraph())
    assert await retriever.aretrieve(SearchQuery(query="hello")) == []


# --- OpenSearchRetriever: missing index (config mismatch) ------------------


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple] = []
        self.errors: list[tuple] = []

    def warning(self, *args, **_kwargs) -> None:
        self.warnings.append(args)

    def error(self, *args, **_kwargs) -> None:
        self.errors.append(args)

    def __getattr__(self, _name):
        return lambda *a, **k: None


@pytest.fixture
def recording_logger(monkeypatch) -> _RecordingLogger:
    from unified_kg_rag.adapters.retrievers import opensearch_retriever as mod

    rec = _RecordingLogger()
    monkeypatch.setattr(mod, "logger", rec)
    return rec


async def test_index_not_found_warns_instead_of_error(
    config: Config, recording_logger: _RecordingLogger
) -> None:
    # A 404 index_not_found is a config/index mismatch, not a transient
    # failure: a WARNING (not an ERROR) per query, empty results.
    retriever = _opensearch_retriever(config)
    _set_asearch_raising(
        retriever,
        NotFoundError(404, "index_not_found_exception", {"error": "no such index"}),
    )
    prefix = config.indexing.opensearch.entities_index_prefix
    for _ in range(3):
        results = await retriever.aretrieve(
            SearchQuery(query="hello", index_prefixes=[prefix])
        )
        assert results == []

    assert len(recording_logger.warnings) == 3
    assert recording_logger.errors == []


async def test_other_not_found_still_logs_error(
    config: Config, recording_logger: _RecordingLogger
) -> None:
    retriever = _opensearch_retriever(config)
    _set_asearch_raising(retriever, NotFoundError(404, "resource_not_found", {}))
    prefix = config.indexing.opensearch.entities_index_prefix
    results = await retriever.aretrieve(
        SearchQuery(query="hello", index_prefixes=[prefix])
    )

    assert results == []
    assert recording_logger.warnings == []
    assert len(recording_logger.errors) == 1


_INDEX_NOT_FOUND = NotFoundError(
    404, "index_not_found_exception", {"error": "no such index"}
)


@pytest.mark.parametrize("suffix", [None, "tenant-a"])
async def test_query_against_a_never_ingested_suffix_is_fatal(
    config: Config, suffix: str | None
) -> None:
    # Before ingestion, or with a typo'd --suffix, every index is missing: that
    # must not look like a successful search with 0 results.
    retriever = _opensearch_retriever(config)
    _set_asearch_raising(retriever, _INDEX_NOT_FOUND, existing=set())
    prefix = config.indexing.opensearch.entities_index_prefix
    query = SearchQuery(query="hello", index_prefixes=[prefix], suffix=suffix)

    with pytest.raises(IndexNotFoundError) as exc:
        await retriever.aretrieve(query)

    shown = suffix or Constants.DEFAULT_SUFFIX.value
    assert f"suffix '{shown}'" in str(exc.value)
    assert "index_value" in str(exc.value)
    assert is_fatal_retrieval_error(exc.value)


async def test_missing_optional_index_of_an_ingested_corpus_degrades(
    config: Config,
) -> None:
    # A corpus with no claims never creates its claims index; the text-units
    # index exists, so the missing index is skipped, not fatal.
    retriever = _opensearch_retriever(config)
    opensearch = config.indexing.opensearch
    text_units = retriever._get_name(opensearch.text_units_index_prefix, None)
    _set_asearch_raising(retriever, _INDEX_NOT_FOUND, existing={text_units})

    results = await retriever.aretrieve(
        SearchQuery(query="hello", index_prefixes=[opensearch.claims_index_prefix])
    )

    assert results == []
    assert retriever._opensearch_client.exists_calls == [text_units]
