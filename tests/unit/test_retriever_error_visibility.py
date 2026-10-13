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

import asyncio
import json
import socket
import weakref
from collections.abc import Iterator

import pytest
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import GraphTraversal
from gremlin_python.structure.graph import Graph
from opensearchpy import (
    AsyncHttpConnection,
    AsyncOpenSearch,
    Connection,
    OpenSearch,
    RequestsHttpConnection,
    Urllib3HttpConnection,
)
from opensearchpy.exceptions import NotFoundError, TransportError

from unified_kg_rag.adapters.aws.opensearch import (
    OpenSearchClient,
    _handle_async_opensearch_errors,
    _handle_opensearch_errors,
    is_index_not_found,
)
from unified_kg_rag.adapters.retrieval.base import is_fatal_retrieval_error
from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from unified_kg_rag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from unified_kg_rag.adapters.storage.embedding_settings import (
    configured_embedding_model,
)
from unified_kg_rag.domain.models import Config, Constants, SearchQuery, SearchType
from unified_kg_rag.domain.models.config import EmbeddingModelId
from unified_kg_rag.shared import (
    AWSServiceError,
    ConfigurationError,
    IndexNotFoundError,
    StoreConnectionError,
)

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


# --- unreachable stores, against real clients ------------------------------
#
# Regression: opensearch-py's ConnectionError is not a built-in ConnectionError
# and its message ("Cannot connect to host ...") matched no marker, so an
# OpenSearch that was down read as "no results" and every strategy abstained.


@pytest.fixture
def closed_port() -> int:
    """A localhost port nothing listens on (connections are refused)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


@pytest.fixture
def silent_port() -> Iterator[int]:
    """A localhost port that accepts connections and never answers."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(8)
        yield sock.getsockname()[1]


def _local_opensearch_config(config: Config, port: int) -> Config:
    opensearch = config.aws.opensearch
    opensearch.endpoint = "127.0.0.1"
    opensearch.port = port
    opensearch.use_ssl = False
    opensearch.use_iam = False
    opensearch.allow_anonymous = True
    return config


async def test_unreachable_opensearch_is_fatal(
    config: Config, closed_port: int
) -> None:
    client = OpenSearchClient(_local_opensearch_config(config, closed_port))
    try:
        with pytest.raises(StoreConnectionError, match="Cannot connect") as exc:
            await client.asearch(index="any", body={"query": {"match_all": {}}})
    finally:
        await client.aclose()

    assert is_fatal_retrieval_error(exc.value)


class _SyncSearch:
    def __init__(self, port: int, **kwargs) -> None:
        self.client = OpenSearch(hosts=[{"host": "127.0.0.1", "port": port}], **kwargs)

    @_handle_opensearch_errors
    def search(self) -> dict:
        return self.client.search(index="any", body={})


class _AsyncSearch:
    def __init__(self, port: int, **kwargs) -> None:
        self.client = AsyncOpenSearch(
            hosts=[{"host": "127.0.0.1", "port": port}], **kwargs
        )

    @_handle_async_opensearch_errors
    async def search(self) -> dict:
        return await self.client.search(index="any", body={})


@pytest.mark.parametrize("connection_cls", [RequestsHttpConnection, None])
def test_unreachable_opensearch_is_fatal_for_the_sync_client(
    closed_port: int, connection_cls: type[Connection] | None
) -> None:
    kwargs = {"connection_class": connection_cls} if connection_cls else {}
    search = _SyncSearch(closed_port, max_retries=0, **kwargs)

    with pytest.raises(StoreConnectionError) as exc:
        search.search()

    assert is_fatal_retrieval_error(exc.value)


async def test_opensearch_timeout_stays_transient(silent_port: int) -> None:
    search = _AsyncSearch(
        silent_port,
        connection_class=AsyncHttpConnection,
        timeout=0.2,
        max_retries=0,
    )
    try:
        with pytest.raises(AWSServiceError) as exc:
            await search.search()
    finally:
        await search.client.close()

    assert not isinstance(exc.value, StoreConnectionError)
    assert not is_fatal_retrieval_error(exc.value)


async def test_unreachable_opensearch_fails_the_retrieval(
    config: Config, closed_port: int
) -> None:
    retriever = _opensearch_retriever(config)
    client = OpenSearchClient(_local_opensearch_config(config, closed_port))
    object.__setattr__(retriever, "_opensearch_client", client)
    try:
        with pytest.raises(StoreConnectionError):
            await retriever.aretrieve(
                SearchQuery(query="hello", search_type=SearchType.LEXICAL)
            )
    finally:
        await client.aclose()


async def test_unreachable_neptune_fails_the_retrieval(
    config: Config, closed_port: int
) -> None:
    # A connection opened earlier whose server has gone: gremlinpython opens
    # its websocket on the first request and raises aiohttp's
    # ClientConnectorError, which `_execute_traversal` used to degrade to [].
    connection = DriverRemoteConnection(
        f"ws://127.0.0.1:{closed_port}/gremlin", "g", pool_size=1
    )

    class _Client:
        g = traversal().with_remote(connection)

    retriever = _neptune_retriever(config, _Client())
    try:
        with pytest.raises(StoreConnectionError, match="Neptune is not reachable"):
            await retriever.aretrieve(SearchQuery(query="", filters={"id": ["e1"]}))
    finally:
        await asyncio.to_thread(connection.close)


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
    object.__setattr__(
        inst,
        "_embedding_model_id",
        configured_embedding_model(config.indexing.opensearch),
    )
    object.__setattr__(inst, "_embedding_dimension", _QUERY_DIMENSION)
    object.__setattr__(inst, "_embedding_checked_aliases", set())
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


_QUERY_DIMENSION = 8


def _set_asearch_raising(
    retriever: OpenSearchRetriever, exc: Exception, existing: set[str] | None = None
) -> None:
    # Let the query-vector step succeed, then make the actual search execution
    # (``_opensearch_client.asearch``) raise — this is the path that previously
    # swallowed fatal errors inside ``_execute_search`` and returned [].
    async def _vec(*_args, **_kwargs):
        return [0.0] * _QUERY_DIMENSION

    async def _asearch(*_args, **_kwargs):
        raise exc

    object.__setattr__(retriever, "_get_query_vector", _vec)
    object.__setattr__(
        retriever, "_opensearch_client", _MockSearchClient(_asearch, existing)
    )


class _MockSearchClient:
    def __init__(
        self,
        asearch,
        existing: set[str] | None = None,
        mappings: dict[str, dict] | None = None,
    ) -> None:
        self.asearch = asearch
        self.existing = existing  # None = every index exists
        self.exists_calls: list[str] = []
        self.mappings = mappings or {}  # alias -> mappings of its one index
        self.mapping_calls: list[str] = []

    async def aget_mappings(self, index: str) -> dict[str, dict]:
        self.mapping_calls.append(index)
        if index not in self.mappings:
            raise _INDEX_NOT_FOUND
        return {f"{index}-20260101": self.mappings[index]}

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


def _server_error(
    connection_cls: type[Connection], status: int, body: dict
) -> TransportError:
    """The exception ``connection_cls`` raises for an HTTP error ``body``.

    Built through the connection's own ``_raise_error`` with the arguments its
    ``perform_request`` passes: the sync connections forward the response
    content type (so ``.error`` is the extracted type and ``.info`` the parsed
    body), ``AsyncHttpConnection`` forwards none (``.error`` is the raw JSON
    text, ``.info`` is ``None``).
    """
    raw = json.dumps(body)
    connection = connection_cls()
    try:
        if connection_cls is AsyncHttpConnection:
            connection._raise_error(status, raw)
        else:
            connection._raise_error(status, raw, "application/json; charset=UTF-8")
    except TransportError as exc:
        return exc
    raise AssertionError("_raise_error did not raise")


def _through_handler(exc: Exception) -> Exception:
    """``exc`` as the OpenSearch client's error handler re-raises it.

    The sync and async handlers wrap a transport error the same way.
    """

    @_handle_opensearch_errors
    def search(_client) -> None:
        raise exc

    try:
        search(None)
    except Exception as handled:  # noqa: BLE001 - returned for the test
        return handled
    raise AssertionError("the handler did not raise")


_NO_SUCH_INDEX_BODY = {
    "error": {
        "root_cause": [
            {
                "type": "index_not_found_exception",
                "reason": "no such index [graphrag-entities-default]",
            }
        ],
        "type": "index_not_found_exception",
        "reason": "no such index [graphrag-entities-default]",
    },
    "status": 404,
}
_CONNECTIONS = [AsyncHttpConnection, RequestsHttpConnection, Urllib3HttpConnection]


@pytest.mark.parametrize("connection_cls", _CONNECTIONS)
def test_index_not_found_is_recognised_for_every_connection_class(
    connection_cls: type[Connection],
) -> None:
    exc = _server_error(connection_cls, 404, _NO_SUCH_INDEX_BODY)
    assert isinstance(exc, NotFoundError)
    assert is_index_not_found(exc)


def test_async_connection_leaves_the_raw_body_in_error() -> None:
    # Pins the library shape the classifier depends on: if opensearch-py starts
    # extracting the type for the async client too, this documents the change.
    exc = _server_error(AsyncHttpConnection, 404, _NO_SUCH_INDEX_BODY)
    assert exc.info is None
    assert json.loads(exc.error)["error"]["type"] == "index_not_found_exception"


@pytest.mark.parametrize("connection_cls", _CONNECTIONS)
def test_root_cause_only_index_not_found_is_recognised(
    connection_cls: type[Connection],
) -> None:
    body = {
        "error": {
            "root_cause": [{"type": "index_not_found_exception"}],
            "type": "some_wrapper_exception",
        },
        "status": 404,
    }
    assert is_index_not_found(_server_error(connection_cls, 404, body))


@pytest.mark.parametrize("connection_cls", _CONNECTIONS)
def test_other_404_bodies_are_not_index_not_found(
    connection_cls: type[Connection],
) -> None:
    resource = {"error": {"type": "resource_not_found_exception"}, "status": 404}
    document = {"_index": "graphrag-entities-default", "_id": "x", "found": False}
    assert not is_index_not_found(_server_error(connection_cls, 404, resource))
    assert not is_index_not_found(_server_error(connection_cls, 404, document))


def test_non_404_is_not_index_not_found() -> None:
    exc = _server_error(AsyncHttpConnection, 400, _NO_SUCH_INDEX_BODY)
    assert not is_index_not_found(exc)
    assert not is_index_not_found(ValueError("index_not_found_exception"))


@pytest.mark.parametrize("connection_cls", _CONNECTIONS)
async def test_index_not_found_warns_instead_of_error(
    config: Config,
    recording_logger: _RecordingLogger,
    connection_cls: type[Connection],
) -> None:
    # A 404 index_not_found is a config/index mismatch, not a transient
    # failure: a WARNING (not an ERROR) per query, empty results.
    retriever = _opensearch_retriever(config)
    _set_asearch_raising(
        retriever, _server_error(connection_cls, 404, _NO_SUCH_INDEX_BODY)
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
    body = {"error": {"type": "resource_not_found_exception"}, "status": 404}
    _set_asearch_raising(retriever, _server_error(AsyncHttpConnection, 404, body))
    prefix = config.indexing.opensearch.entities_index_prefix
    results = await retriever.aretrieve(
        SearchQuery(query="hello", index_prefixes=[prefix])
    )

    assert results == []
    assert recording_logger.warnings == []
    assert len(recording_logger.errors) == 1


# The retriever searches through the async client, so this is the shape it
# actually sees.
_INDEX_NOT_FOUND = _server_error(AsyncHttpConnection, 404, _NO_SUCH_INDEX_BODY)


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


# --- query embedding settings vs the index's ------------------------------


def test_configuration_error_is_fatal() -> None:
    assert is_fatal_retrieval_error(ConfigurationError("bad config")) is True


def _knn_mappings(
    dimension: int, model: str | None = None, recorded_dimension: int | None = None
) -> dict:
    mappings: dict = {
        "properties": {
            "text": {"type": "text"},
            "text_embedding": {"type": "knn_vector", "dimension": dimension},
        }
    }
    meta = {}
    if model is not None:
        meta["embedding_model_id"] = model
    if recorded_dimension is not None:
        meta["embedding_dimension"] = recorded_dimension
    if meta:
        mappings["_meta"] = meta
    return mappings


def _retriever_over(
    config: Config, mappings: dict, search_error: Exception | None = None
) -> OpenSearchRetriever:
    """A retriever whose text-units alias has ``mappings``; searches hit once."""
    retriever = _opensearch_retriever(config)
    alias = retriever._get_name(
        config.indexing.opensearch.text_units_index_prefix, None
    )
    hit = {"_index": alias, "_id": "t1", "_score": 1.0, "_source": {"text": "hi"}}

    async def _asearch(*_args, **_kwargs):
        if search_error is not None:
            raise search_error
        return {"hits": {"hits": [hit]}}

    async def _vec(_text, search_type, _fields):
        if search_type is SearchType.LEXICAL:
            return None
        return [0.0] * _QUERY_DIMENSION

    object.__setattr__(retriever, "_get_query_vector", _vec)
    object.__setattr__(
        retriever,
        "_opensearch_client",
        _MockSearchClient(_asearch, mappings={alias: mappings}),
    )
    return retriever


def _text_units_query(
    config: Config, search_type: SearchType = SearchType.HYBRID
) -> SearchQuery:
    prefix = config.indexing.opensearch.text_units_index_prefix
    return SearchQuery(query="hello", index_prefixes=[prefix], search_type=search_type)


async def test_index_of_another_embedding_model_is_refused(config: Config) -> None:
    # Same dimension, another model: the search would "succeed" and rank by
    # unrelated vectors.
    model = configured_embedding_model(config.indexing.opensearch)
    retriever = _retriever_over(
        config, _knn_mappings(_QUERY_DIMENSION, "other-model", _QUERY_DIMENSION)
    )

    with pytest.raises(ConfigurationError) as exc:
        await retriever.aretrieve(_text_units_query(config))

    message = str(exc.value)
    assert "'other-model'" in message and f"'{model}'" in message
    assert is_fatal_retrieval_error(exc.value)


async def test_index_of_another_dimension_is_refused_without_meta(
    config: Config,
) -> None:
    # An index built before _meta was recorded is checked by its kNN dimension.
    retriever = _retriever_over(config, _knn_mappings(512))

    with pytest.raises(ConfigurationError, match=r"dimension 512.*dimension 8"):
        await retriever.aretrieve(_text_units_query(config))


async def test_matching_index_is_searched_and_checked_once(config: Config) -> None:
    model = configured_embedding_model(config.indexing.opensearch)
    retriever = _retriever_over(
        config, _knn_mappings(_QUERY_DIMENSION, model, _QUERY_DIMENSION)
    )

    for _ in range(2):
        results = await retriever.aretrieve(_text_units_query(config))
        assert [r.source for r in results] == ["t1"]

    assert len(retriever._opensearch_client.mapping_calls) == 1


async def test_enum_model_id_matches_the_recorded_string(config: Config) -> None:
    # The config holds an EmbeddingModelId member; the index records its value.
    config.indexing.opensearch.embedding_model_id = EmbeddingModelId.TITAN_EMBED_V2
    retriever = _retriever_over(
        config,
        _knn_mappings(
            _QUERY_DIMENSION, EmbeddingModelId.TITAN_EMBED_V2.value, _QUERY_DIMENSION
        ),
    )

    assert len(await retriever.aretrieve(_text_units_query(config))) == 1


async def test_lexical_search_does_not_check_embeddings(config: Config) -> None:
    retriever = _retriever_over(config, _knn_mappings(512, "other-model", 512))

    results = await retriever.aretrieve(_text_units_query(config, SearchType.LEXICAL))

    assert len(results) == 1
    assert retriever._opensearch_client.mapping_calls == []


async def test_unreadable_mappings_leave_the_check_to_the_search(
    config: Config,
) -> None:
    retriever = _retriever_over(config, _knn_mappings(_QUERY_DIMENSION))
    retriever._opensearch_client.mappings = {}  # every get_mapping 404s

    assert len(await retriever.aretrieve(_text_units_query(config))) == 1


_INVALID_DIMENSION_BODY = {
    "error": {
        "root_cause": [
            {
                "type": "query_shard_exception",
                "reason": "failed to create query: Query vector has invalid "
                "dimension: 8. Dimension should be: 512",
            }
        ],
        "type": "search_phase_execution_exception",
        "reason": "all shards failed",
    },
    "status": 400,
}


async def test_invalid_dimension_rejection_is_a_fatal_configuration_error(
    config: Config,
) -> None:
    raw = _server_error(AsyncHttpConnection, 400, _INVALID_DIMENSION_BODY)
    retriever = _retriever_over(config, {}, search_error=_through_handler(raw))
    retriever._opensearch_client.mappings = {}

    with pytest.raises(ConfigurationError) as exc:
        await retriever.aretrieve(_text_units_query(config))

    message = str(exc.value)
    assert "Query vector has invalid dimension: 8. Dimension should be: 512" in message
    assert "embedding" in message


class _RecordingEmbeddingFactory:
    """Model info like Titan Embed V2's: several supported dimensions."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def get_model_info(self, model_id):
        return type("Info", (), {"dimensions": [256, 512, 1024]})()

    def get_model(self, model_id, **kwargs):
        self.requests.append(kwargs)
        return object()


@pytest.mark.parametrize(("configured", "expected"), [(512, 512), (None, 1024)])
def test_queries_embed_with_the_indexed_dimension(
    config: Config, configured: int | None, expected: int
) -> None:
    # The indexer embeds with indexing.opensearch.embedding_dimension; queries
    # embedded at the model's default instead fail every kNN search.
    config.indexing.opensearch.embedding_dimension = configured
    factory = _RecordingEmbeddingFactory()

    retriever = OpenSearchRetriever(
        config, opensearch_client=object(), embedding_factory=factory
    )

    assert factory.requests == [{"dimensions": expected}]
    assert retriever._embedding_dimension == expected


def test_invalid_dimension_reason_is_cut_out_of_a_python_repr() -> None:
    # The sync client error wrapper renders the parsed body with str(dict),
    # so the reason sits in single quotes followed by the rest of the repr.
    from unified_kg_rag.adapters.retrievers.opensearch_retriever import (
        _invalid_dimension_error,
    )

    wrapped = AWSServiceError(
        f"Transport error in 'search': status=400, info={_INVALID_DIMENSION_BODY}"
    )

    assert _invalid_dimension_error(wrapped) == (
        "Query vector has invalid dimension: 8. Dimension should be: 512"
    )


# An Amazon OpenSearch Service access-policy denial (the text holds "auth").
_ACCESS_DENIED_BODY = {
    "Message": "User: arn:aws:iam::000000000000:role/search-only is not "
    "authorized to perform: es:ESHttpGet"
}


def _denied(operation: str) -> AWSServiceError:
    raw = _server_error(AsyncHttpConnection, 403, _ACCESS_DENIED_BODY)
    denied = AWSServiceError(
        f"Async transport error in '{operation}': status=403, "
        f"info={_ACCESS_DENIED_BODY}"
    )
    denied.__cause__ = raw
    return denied


async def test_a_denied_mapping_read_skips_the_check_with_one_warning(
    config: Config, recording_logger: _RecordingLogger
) -> None:
    # A search-only principal may not read mappings, and the 403's text holds
    # the fatal "auth" marker: every vector query used to fail on the check.
    denied = _denied("aget_mappings")
    assert is_fatal_retrieval_error(denied)
    retriever = _retriever_over(config, {})

    async def _aget_mappings(index: str) -> dict[str, dict]:
        retriever._opensearch_client.mapping_calls.append(index)
        raise denied

    retriever._opensearch_client.aget_mappings = _aget_mappings

    for _ in range(2):
        results = await retriever.aretrieve(_text_units_query(config))
        assert [r.source for r in results] == ["t1"]

    assert len(retriever._opensearch_client.mapping_calls) == 1
    assert len(recording_logger.warnings) == 1
    assert "Not allowed to read the mappings" in recording_logger.warnings[0][0]


async def test_a_denied_search_stays_fatal(config: Config) -> None:
    retriever = _retriever_over(config, {}, search_error=_denied("asearch"))
    retriever._opensearch_client.mappings = {}

    with pytest.raises(AWSServiceError, match="status=403"):
        await retriever.aretrieve(_text_units_query(config))


# --- rejected requests (HTTP 400) -------------------------------------------
#
# Regression: a 400 is the same on every retry, yet it degraded to [] like a
# timeout, so a missing search pipeline or a filter value of the wrong type
# answered from no context (or from the other, unfiltered indices), and the
# log kept only "status=400, info=None".


_MISSING_PIPELINE_BODY = {
    "error": {
        "root_cause": [
            {
                "type": "illegal_argument_exception",
                "reason": "Pipeline missing-pipeline is not defined",
            }
        ],
        "type": "illegal_argument_exception",
        "reason": "Pipeline missing-pipeline is not defined",
    },
    "status": 400,
}
_BAD_FILTER_VALUE_BODY = {
    "error": {
        "root_cause": [
            {
                "type": "query_shard_exception",
                "reason": 'failed to create query: For input string: "abc"',
            }
        ],
        "type": "search_phase_execution_exception",
        "reason": "all shards failed",
    },
    "status": 400,
}


@pytest.mark.parametrize("connection_cls", _CONNECTIONS)
def test_handler_keeps_the_server_reason(connection_cls: type[Connection]) -> None:
    handled = _through_handler(
        _server_error(connection_cls, 400, _MISSING_PIPELINE_BODY)
    )
    assert type(handled) is AWSServiceError
    assert "status=400" in str(handled)
    assert "Pipeline missing-pipeline is not defined" in str(handled)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (_MISSING_PIPELINE_BODY, "Pipeline missing-pipeline is not defined"),
        (_BAD_FILTER_VALUE_BODY, 'For input string: "abc"'),
    ],
)
async def test_rejected_search_is_a_fatal_configuration_error(
    config: Config, body: dict, reason: str
) -> None:
    error = _through_handler(_server_error(AsyncHttpConnection, 400, body))
    retriever = _retriever_over(config, {}, search_error=error)
    retriever._opensearch_client.mappings = {}

    with pytest.raises(ConfigurationError) as exc:
        await retriever.aretrieve(_text_units_query(config))

    assert reason in str(exc.value)
    assert is_fatal_retrieval_error(exc.value)


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_throttling_and_server_errors_stay_transient(
    config: Config, status: int
) -> None:
    body = {"error": {"type": "some_exception", "reason": "busy"}, "status": status}
    error = _through_handler(_server_error(AsyncHttpConnection, status, body))
    retriever = _retriever_over(config, {}, search_error=error)
    retriever._opensearch_client.mappings = {}

    assert await retriever.aretrieve(_text_units_query(config)) == []
