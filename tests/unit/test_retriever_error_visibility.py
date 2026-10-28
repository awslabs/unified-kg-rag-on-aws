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
import logging
import socket
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import openai
import pytest
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    ReadTimeoutError,
)
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.driver.protocol import GremlinServerError
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
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError
from opensearchpy.exceptions import ConnectionTimeout, NotFoundError, TransportError

from unified_kg_rag.adapters.aws.neptune import NeptuneClient
from unified_kg_rag.adapters.aws.opensearch import (
    OpenSearchClient,
    _handle_async_opensearch_errors,
    _handle_opensearch_errors,
    is_index_not_found,
)
from unified_kg_rag.adapters.retrieval.base import is_fatal_retrieval_error
from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from unified_kg_rag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from unified_kg_rag.adapters.storage import neptune_indexer
from unified_kg_rag.adapters.storage.embedding_settings import (
    configured_embedding_model,
)
from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer
from unified_kg_rag.domain.models import Config, Constants, SearchQuery, SearchType
from unified_kg_rag.domain.models.config import EmbeddingModelId
from unified_kg_rag.shared import (
    AWSServiceError,
    ConfigurationError,
    EmbeddingModelError,
    IndexNotFoundError,
    StoreConnectionError,
)
from unified_kg_rag.shared.exceptions import (
    StoreAccessDeniedError,
    StoreConfigurationError,
    StoreRequestRejectedError,
    StoreTimeoutError,
)

pytestmark = pytest.mark.unit

# Every strategy seeds Neptune graph expansion with entity ids.
_SEEDED = SearchQuery(query="hello", filters={"id": ["e1"]})


# --- classifier ----------------------------------------------------------


def test_connection_error_is_fatal() -> None:
    assert is_fatal_retrieval_error(ConnectionError("refused")) is True


def _client_error(code: str, message: str = "x") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, "InvokeModel")


@pytest.mark.parametrize(
    "exc",
    [
        StoreConnectionError("Neptune is not reachable"),
        StoreTimeoutError("OpenSearch did not answer"),
        StoreConfigurationError("OpenSearch endpoint is not configured."),
        StoreAccessDeniedError("status=403"),
        StoreRequestRejectedError("status=400"),
        _client_error(
            "ValidationException", "The provided model identifier is invalid."
        ),
        _client_error("ResourceNotFoundException"),
        _client_error("AccessDeniedException"),
        _client_error("UnrecognizedClientException"),
        NoCredentialsError(),
        EndpointConnectionError(endpoint_url="https://bedrock-runtime.example"),
        ReadTimeoutError(endpoint_url="https://bedrock-runtime.example"),
    ],
    ids=lambda exc: type(exc).__name__,
)
def test_fatal_errors_are_classified_by_type(exc: Exception) -> None:
    assert is_fatal_retrieval_error(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        # Messages are not matched: the old "auth" marker failed every query
        # whose error named a field such as "author".
        AWSServiceError("status=500, error=failed on field [author]"),
        AWSServiceError("AccessDenied"),
        RuntimeError("Read timed out"),
        _client_error("ThrottlingException"),
        _client_error("ModelErrorException"),
    ],
    ids=lambda exc: str(exc)[:30],
)
def test_other_errors_are_transient(exc: Exception) -> None:
    assert is_fatal_retrieval_error(exc) is False


# --- local model servers ----------------------------------------------------
#
# Regression: an OpenAI-compatible or Ollama server that refused every call (a
# bad key, an unknown model, a refused port) degraded each section to no
# results, so every strategy answered "could not find" and the logs blamed
# OpenSearch. The errors are the clients' real types, built without a network.

_MODEL_REQUEST = httpx.Request("POST", "http://model-server.test/v1/embeddings")


def _openai_status(name: str, status: int) -> Exception:
    openai = pytest.importorskip("openai")
    response = httpx.Response(status, request=_MODEL_REQUEST)
    cls = getattr(openai, name)
    error: Exception = cls("synthetic", response=response, body=None)
    return error


@pytest.mark.parametrize(
    ("name", "status", "fatal"),
    [
        ("AuthenticationError", 401, True),
        ("PermissionDeniedError", 403, True),
        ("NotFoundError", 404, True),
        ("BadRequestError", 400, True),
        ("RateLimitError", 429, False),
        ("InternalServerError", 500, False),
        ("APIStatusError", 408, False),
        ("APIStatusError", 503, False),
    ],
)
def test_openai_status_errors_are_fatal_unless_transient(
    name: str, status: int, fatal: bool
) -> None:
    assert is_fatal_retrieval_error(_openai_status(name, status)) is fatal


def test_an_unreachable_openai_server_is_fatal() -> None:
    openai = pytest.importorskip("openai")
    assert is_fatal_retrieval_error(openai.APIConnectionError(request=_MODEL_REQUEST))
    assert is_fatal_retrieval_error(openai.APITimeoutError(request=_MODEL_REQUEST))


@pytest.mark.parametrize(
    ("status", "fatal"),
    [(404, True), (401, True), (429, False), (500, False), (-1, False)],
)
def test_ollama_response_errors_are_fatal_unless_transient(
    status: int, fatal: bool
) -> None:
    ollama = pytest.importorskip("ollama")
    error = ollama.ResponseError("model 'synthetic' not found", status_code=status)
    assert is_fatal_retrieval_error(error) is fatal


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("refused", request=_MODEL_REQUEST),
        httpx.ReadTimeout("timed out", request=_MODEL_REQUEST),
    ],
    ids=lambda exc: type(exc).__name__,
)
def test_httpx_transport_errors_are_fatal(exc: Exception) -> None:
    assert is_fatal_retrieval_error(exc)


@pytest.mark.parametrize(
    "exc",
    [
        # Same class names as the model clients', other libraries.
        TransportError(500, "search_phase_execution_exception", "failed"),
        GremlinServerError({"code": 500, "message": "failed", "attributes": {}}),
    ],
    ids=lambda exc: type(exc).__module__,
)
def test_store_errors_of_similar_names_are_not_provider_errors(
    exc: Exception,
) -> None:
    assert is_fatal_retrieval_error(exc) is False


def test_a_context_overflow_is_not_fatal() -> None:
    """Only the one prompt is too long: a global map batch is left unrated."""
    from langchain_aws.chat_models.bedrock_converse import (
        BedrockContextOverflowError,
    )

    overflow = BedrockContextOverflowError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "Input is too long for requested model.",
            }
        },
        "Converse",
    )
    assert is_fatal_retrieval_error(overflow) is False
    # Any other ValidationException (an invalid model id) still fails.
    assert is_fatal_retrieval_error(_client_error("ValidationException")) is True


def test_an_openai_context_overflow_is_not_fatal() -> None:
    base = pytest.importorskip("langchain_openai.chat_models.base")
    overflow_cls = base.OpenAIContextOverflowError

    response = httpx.Response(400, request=_MODEL_REQUEST)
    overflow = overflow_cls("too long", response=response, body=None)
    assert is_fatal_retrieval_error(overflow) is False


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


async def test_opensearch_timeout_is_a_store_timeout(silent_port: int) -> None:
    search = _AsyncSearch(
        silent_port,
        connection_class=AsyncHttpConnection,
        timeout=0.2,
        max_retries=0,
    )
    try:
        with pytest.raises(StoreTimeoutError) as exc:
            await search.search()
    finally:
        await search.client.close()

    assert is_fatal_retrieval_error(exc.value)


class _ThrottlingOpenSearch:
    """A local HTTP server answering searches 429 for ``burst`` seconds."""

    def __init__(self, burst: float) -> None:
        self.burst = burst
        self.started: float | None = None
        self.requests: list[float] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                now = time.monotonic()
                owner.started = owner.started or now
                owner.requests.append(now)
                if now - owner.started < owner.burst:
                    status, body = 429, {
                        "error": {"type": "es_rejected_execution_exception"},
                        "status": 429,
                    }
                else:
                    status, body = 200, {"hits": {"total": {"value": 0}, "hits": []}}
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST

            def log_message(self, *_args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port: int = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


async def _search_throttled(config: Config, server: _ThrottlingOpenSearch) -> None:
    client = OpenSearchClient(_local_opensearch_config(config, server.port))
    try:
        await client.asearch(index="any", body={"query": {"match_all": {}}})
    finally:
        await client.aclose()
        server.close()


async def test_a_short_429_burst_is_waited_out(config: Config) -> None:
    # Regression: opensearch-py retried back to back, so a 429 burst of a few
    # milliseconds used up every attempt and failed the query.
    server = _ThrottlingOpenSearch(burst=0.2)
    await _search_throttled(config, server)
    assert len(server.requests) >= 2
    assert server.requests[1] - server.requests[0] >= 0.2


async def test_a_persistent_429_fails_after_store_max_attempts(
    config: Config,
) -> None:
    config.search.store_max_attempts = 3
    server = _ThrottlingOpenSearch(burst=60)
    with pytest.raises(StoreConnectionError, match="status=429"):
        await _search_throttled(config, server)
    assert len(server.requests) == 3


class _KnnRejectingOpenSearch:
    """A local HTTP server rejecting every kNN search with a 400.

    It answers an attribute probe (``_source: false``) with no hits after
    ``probe_delay`` seconds, so the searches started alongside it fail first,
    as OpenSearch 2.13 answers a kNN filter on an unmapped field.
    """

    def __init__(self, probe_delay: float) -> None:
        self.paths: list[str] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.loads(raw or b"{}")
                owner.paths.append(self.path)
                status, answer = 200, {"hits": {"total": {"value": 0}, "hits": []}}
                if body.get("_source") is False:
                    time.sleep(probe_delay)
                elif "knn" in json.dumps(body):
                    cause = {
                        "type": "illegal_argument_exception",
                        "reason": "Rewrite first",
                    }
                    status, answer = 400, {
                        "error": {**cause, "root_cause": [cause]},
                        "status": 400,
                    }
                data = json.dumps(answer).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port: int = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


async def _retrieve_rejecting_knn(
    config: Config, filters: dict, caplog: pytest.LogCaptureFixture
) -> list:
    server = _KnnRejectingOpenSearch(probe_delay=0.3)
    retriever = _opensearch_retriever(config)
    object.__setattr__(
        retriever, "_index_filter_fields", retriever._initialize_index_filter_fields()
    )
    client = OpenSearchClient(_local_opensearch_config(config, server.port))
    object.__setattr__(retriever, "_opensearch_client", client)

    async def _vector(*_args, **_kwargs) -> list[float]:
        return [0.1] * _QUERY_DIMENSION

    async def _checked(*_args, **_kwargs) -> None:
        return None

    object.__setattr__(retriever, "_get_query_vector", _vector)
    object.__setattr__(retriever, "_check_embedding_settings", _checked)
    query = SearchQuery(
        query="hello",
        search_type=SearchType.HYBRID,
        index_prefixes=[config.indexing.opensearch.community_reports_index_prefix],
        filters=filters,
    )
    caplog.set_level(logging.DEBUG)
    try:
        return await retriever.aretrieve(query)
    finally:
        await client.aclose()
        server.close()


async def test_a_search_discarded_after_the_probe_logs_no_error(
    config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    # Regression: the client logged every transport error at ERROR with its
    # traceback, so the kNN search started alongside an attribute probe drew
    # an ERROR for its 400 although the probe then discarded it and the
    # query was answered correctly (no document has the attribute).
    results = await _retrieve_rejecting_knn(config, {"attr_category": "a"}, caplog)

    assert results == []
    # (opensearch-py itself still logs each 4xx as a one-line WARNING.)
    # Only this package's records: another library's (asyncio's, from a
    # loop closing under load) is no sign of the regression.
    assert not [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name.startswith("unified_kg_rag")
    ]
    # The rejected kNN search did run and fail before the probe answered.
    assert any("Rewrite first" in r.getMessage() for r in caplog.records)


async def test_a_kept_search_error_is_logged_once(
    config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    # A declared field draws no probe, so the kNN 400 fails the query; it is
    # logged once (by the retriever), naming the index, not also by the client.
    with pytest.raises(ConfigurationError, match="Rewrite first"):
        await _retrieve_rejecting_knn(config, {"rank": {"gte": 1}}, caplog)

    # Only this package's records: another test's unclosed client session
    # (asyncio's ERROR, under load) is no sign of a second log.
    errors = [
        r
        for r in caplog.records
        if r.levelno >= logging.ERROR and r.name.startswith("unified_kg_rag")
    ]
    assert len(errors) == 1
    assert "Rewrite first" in errors[0].getMessage()


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

        def reset(self, g: object = None) -> None:
            pass

    retriever = _neptune_retriever(config, _Client())
    try:
        with pytest.raises(StoreConnectionError, match="Neptune is not reachable"):
            await retriever.aretrieve(_SEEDED)
    finally:
        await asyncio.to_thread(connection.close)


# --- stores that stop answering mid-session --------------------------------
#
# Regression: the query client shared the indexing client's 180 s timeout and
# six attempts, so a hung OpenSearch blocked a query for ~18 minutes and then
# answered from no context; a Neptune that accepted the socket but never
# answered blocked the query (and the process exit) forever; and a Neptune
# connection the server had closed (restart, failover) was reused forever.


def _local_neptune_config(config: Config, port: int, timeout: float) -> Config:
    config.aws.neptune.endpoint = "127.0.0.1"
    config.aws.neptune.port = port
    config.aws.neptune.use_ssl = False
    config.aws.neptune.use_iam = False
    config.search.store_timeout_seconds = timeout
    return config


@pytest.mark.timeout(30)
async def test_a_silent_opensearch_fails_the_query_within_its_timeout(
    config: Config, silent_port: int
) -> None:
    config = _local_opensearch_config(config, silent_port)
    config.search.store_timeout_seconds = 0.2
    config.search.store_max_attempts = 2
    retriever = _opensearch_retriever(config)
    client = OpenSearchClient(config)
    object.__setattr__(retriever, "_opensearch_client", client)
    try:
        with pytest.raises(StoreConnectionError, match="did not answer.*2 attempt"):
            await retriever.aretrieve(
                SearchQuery(query="hello", search_type=SearchType.LEXICAL)
            )
    finally:
        await client.aclose()


@pytest.mark.timeout(30)
async def test_a_silent_neptune_fails_the_query_within_its_timeout(
    config: Config, silent_port: int
) -> None:
    config = _local_neptune_config(config, silent_port, timeout=0.3)
    client = NeptuneClient(config, timeout_seconds=0.3)
    retriever = _neptune_retriever(config, client)
    try:
        with pytest.raises(StoreConnectionError, match="did not answer within 0.3"):
            await retriever.aretrieve(_SEEDED)
    finally:
        await asyncio.to_thread(client.close)


@pytest.mark.timeout(30)
def test_a_silent_neptune_fails_an_ingestion_write_within_its_timeout(
    config: Config, silent_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: the indexer's client had no timeout, so a Neptune that never
    # answered blocked ingestion for good (hours in the real-AWS E2E).
    monkeypatch.setattr(neptune_indexer, "INGESTION_REQUEST_TIMEOUT_SECONDS", 0.3)
    config = _local_neptune_config(config, silent_port, timeout=30)
    config.indexing.neptune.max_attempts = 1
    indexer = NeptuneIndexer(config)
    try:
        # No setting named: search.store_timeout_seconds bounds queries only.
        with pytest.raises(
            AWSServiceError, match=r"did not answer within 0\.3 s$"
        ) as e:
            indexer.read_entities(["e1"])
    finally:
        indexer.close()
    assert isinstance(e.value.__cause__, StoreTimeoutError)


@pytest.fixture
def forbidden_port() -> Iterator[int]:
    """A localhost port answering every HTTP request with 403 Forbidden."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(8)
        sock.settimeout(10)

        def _serve() -> None:
            try:
                while True:
                    conn, _ = sock.accept()
                    with conn:
                        conn.recv(65536)
                        conn.sendall(
                            b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n"
                            b"Connection: close\r\n\r\n"
                        )
            except OSError:
                return

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        yield sock.getsockname()[1]


@pytest.mark.timeout(30)
async def test_a_denied_neptune_connection_is_a_configuration_error(
    config: Config, forbidden_port: int
) -> None:
    config = _local_neptune_config(config, forbidden_port, timeout=5)
    client = NeptuneClient(config, timeout_seconds=5)
    retriever = _neptune_retriever(config, client)
    with pytest.raises(ConfigurationError, match="HTTP 403.*neptune-db:connect"):
        await retriever.aretrieve(_SEEDED)


class _FlakyConnection:
    """A connection whose traversals fail as a websocket the server closed."""

    def __init__(self, **_: object) -> None:
        self.closed = False

    def is_closed(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True


async def test_a_lost_neptune_connection_is_replaced_on_the_next_query(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unified_kg_rag.adapters.aws import neptune as neptune_module

    opened: list[_FlakyConnection] = []

    def _connect(self: NeptuneClient) -> _FlakyConnection:
        opened.append(_FlakyConnection())
        return opened[-1]

    monkeypatch.setattr(NeptuneClient, "_create_connection", _connect)
    monkeypatch.setattr(
        neptune_module, "traversal", lambda: _Remote(Graph().traversal())
    )
    outcomes: list[Exception | None] = [
        RuntimeError("Connection was already closed."),
        None,
    ]

    def _to_list(self) -> list:
        outcome = outcomes.pop(0)
        if outcome is not None:
            raise outcome
        return []

    monkeypatch.setattr(GraphTraversal, "to_list", _to_list)
    config.aws.neptune.endpoint = "neptune.example.invalid"
    client = NeptuneClient(config)
    retriever = _neptune_retriever(config, client)

    with pytest.raises(StoreConnectionError, match="Connection was already closed"):
        await retriever.aretrieve(_SEEDED)
    assert len(opened) == 1 and opened[0].closed

    assert await retriever.aretrieve(_SEEDED) == []
    assert len(opened) == 2 and not opened[1].closed


class _Remote:
    """``traversal()`` stand-in: ``withRemote`` returns a local source."""

    def __init__(self, source: object) -> None:
        self._source = source

    def withRemote(self, _connection: object) -> object:  # noqa: N802
        return self._source


def test_reset_spares_a_newer_connection(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unified_kg_rag.adapters.aws import neptune as neptune_module

    monkeypatch.setattr(
        NeptuneClient, "_create_connection", lambda self: _FlakyConnection()
    )
    monkeypatch.setattr(neptune_module, "traversal", lambda: _Remote(object()))
    config.aws.neptune.endpoint = "neptune.example.invalid"
    client = NeptuneClient(config)
    stale_g = client.g
    client.reset(stale_g)
    current_g, current_connection = client.g, client._connection
    # A second traversal that failed on the stale connection must not drop
    # the one the next query already opened.
    client.reset(stale_g)
    assert client.g is current_g and not current_connection.is_closed()


# --- OpenSearchRetriever top-level handler -------------------------------


def _opensearch_retriever(config: Config) -> OpenSearchRetriever:
    inst = OpenSearchRetriever.__new__(OpenSearchRetriever)
    object.__setattr__(inst, "_config", config)
    object.__setattr__(inst, "_opensearch_config", config.indexing.opensearch)
    object.__setattr__(inst, "_max_size", config.indexing.opensearch.max_query_size)
    object.__setattr__(inst, "_settings_denied_logged", False)
    object.__setattr__(inst, "_window_cut_warned", set())
    object.__setattr__(inst, "_configured_limits", IndexLimits())
    object.__setattr__(inst, "_index_limits", {})
    object.__setattr__(inst, "_field_mappings", inst._initialize_field_mappings())
    object.__setattr__(
        inst,
        "_embedding_model_id",
        configured_embedding_model(config),
    )
    object.__setattr__(inst, "_embedding_dimension", _QUERY_DIMENSION)
    object.__setattr__(inst, "_embedding_checked_aliases", set())
    object.__setattr__(inst, "_embedding_check_denied_logged", False)
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
        StoreConfigurationError("Cannot get AWS credentials for OpenSearch IAM."),
    )
    with pytest.raises(AWSServiceError, match="credentials"):
        await retriever.aretrieve(SearchQuery(query="hello"))


async def test_opensearch_retriever_degrades_on_transient(config: Config) -> None:
    retriever = _opensearch_retriever(config)
    _set_query_vector_raising(retriever, _client_error("ThrottlingException"))
    results = await retriever.aretrieve(SearchQuery(query="hello"))
    assert results == []


@pytest.mark.parametrize(
    "exc",
    [
        _client_error(
            "ValidationException", "The provided model identifier is invalid."
        ),
        _client_error(
            "ValidationException",
            "Invocation of model ID with on-demand throughput isn't supported.",
        ),
        _client_error("ResourceNotFoundException"),
        NoCredentialsError(),
        # A Bedrock endpoint that times out after the model wrapper's retries
        # fails the query, as a store that does not answer does.
        ReadTimeoutError(endpoint_url="https://bedrock-runtime.example"),
    ],
    ids=lambda exc: type(exc).__name__,
)
async def test_a_query_embedding_that_cannot_succeed_fails_the_retrieval(
    config: Config, exc: Exception
) -> None:
    # Regression: every query answered "no relevant information" while the
    # query embedding model was misconfigured.
    retriever = _opensearch_retriever(config)
    _set_query_vector_raising(retriever, exc)
    with pytest.raises(type(exc)):
        await retriever.aretrieve(SearchQuery(query="hello"))


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
        retriever,
        _through_handler(OpenSearchConnectionError("N/A", "refused", Exception())),
    )
    with pytest.raises(StoreConnectionError, match="refused"):
        await retriever.aretrieve(SearchQuery(query="hello"))


@pytest.mark.parametrize(
    "error",
    [
        # A crashed Ollama runner, a dead vLLM engine: a 5xx left after the
        # client's retries.
        lambda: openai.InternalServerError(
            "engine dead",
            response=httpx.Response(500, request=httpx.Request("POST", "http://x")),
            body=None,
        ),
        lambda: AWSServiceError("Read timed out"),
    ],
)
async def test_a_failed_query_embedding_fails_the_retrieval(
    config: Config, error
) -> None:
    # Regression: a 5xx from the embedding server read as transient, so every
    # vector section (and the lexical half of hybrid with it) came back empty
    # and the query answered "could not find relevant information".

    class _FailingEmbedder:
        async def aembed_query(self, text: str) -> list[float]:
            raise error()

    retriever = _opensearch_retriever(config)
    object.__setattr__(retriever, "_embedding_model", _FailingEmbedder())
    object.__setattr__(retriever, "_query_embedding_cache", OrderedDict())
    object.__setattr__(retriever, "_pending_embeddings", {})
    with pytest.raises(EmbeddingModelError, match="Query embedding failed") as info:
        await retriever.aretrieve(SearchQuery(query="hello"))
    assert is_fatal_retrieval_error(info.value)


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
    object.__setattr__(inst, "_executor", None)
    object.__setattr__(inst, "_closed", False)
    # Stub out the metrics recorders (pydantic model -> inject, don't patch).
    return inst


class _ClientRaisingOnG:
    """A fake NeptuneClient whose ``.g`` property raises the given exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    @property
    def g(self):
        raise self._exc


async def test_neptune_retriever_reraises_fatal(config: Config) -> None:
    client = _ClientRaisingOnG(StoreConnectionError("Neptune is not reachable"))
    retriever = _neptune_retriever(config, client)
    with pytest.raises(StoreConnectionError, match="not reachable"):
        await retriever.aretrieve(_SEEDED)


async def test_neptune_retriever_degrades_on_transient(config: Config) -> None:
    client = _ClientRaisingOnG(AWSServiceError("Read timed out"))
    retriever = _neptune_retriever(config, client)
    results = await retriever.aretrieve(_SEEDED)
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
    retriever = _neptune_retriever(config, _ClientWithLocalGraph())
    body = json.dumps({"code": "AccessDeniedException", "detailedMessage": "x"})
    denied = GremlinServerError({"code": 500, "message": body, "attributes": {}})
    with pytest.raises(StoreAccessDeniedError, match="neptune-db"):
        await retriever._execute_traversal(
            lambda g: _RaisingTraversal(denied)  # type: ignore[arg-type,return-value]
        )


async def test_neptune_execute_traversal_degrades_on_transient(
    config: Config,
) -> None:
    retriever = _neptune_retriever(config, _ClientWithLocalGraph())
    results = await retriever._execute_traversal(
        lambda g: _RaisingTraversal(  # type: ignore[arg-type,return-value]
            AWSServiceError("Read timed out")
        )
    )
    assert results == []


class _ClientWithLocalGraph:
    """A fake NeptuneClient exposing a real, connectionless traversal source."""

    def __init__(self) -> None:
        self.resets: list[object] = []

    @property
    def g(self):
        return Graph().traversal()

    def reset(self, g: object = None) -> None:
        self.resets.append(g)


async def test_neptune_fatal_traversal_error_surfaces_from_aretrieve(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fatal_to_list(self) -> list:
        raise ConnectionError("Failed to connect to the Neptune endpoint")

    monkeypatch.setattr(GraphTraversal, "to_list", _fatal_to_list)
    retriever = _neptune_retriever(config, _ClientWithLocalGraph())
    with pytest.raises(ConnectionError, match="Neptune endpoint"):
        await retriever.aretrieve(_SEEDED)


async def test_a_traversal_blocked_on_the_pool_fails_and_resets(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: after a lost connection gremlinpython's pool could stay
    # empty and to_list() block in pool.get() for good, hanging the query.
    release = threading.Event()

    def _blocked_to_list(self) -> list:
        release.wait()
        return []

    monkeypatch.setattr(GraphTraversal, "to_list", _blocked_to_list)
    config.search.store_timeout_seconds = 0.1
    client = _ClientWithLocalGraph()
    retriever = _neptune_retriever(config, client)
    try:
        with pytest.raises(StoreTimeoutError):
            await asyncio.wait_for(retriever.aretrieve(_SEEDED), 5)
    finally:
        release.set()
    assert client.resets


class _SwappableClient:
    """A fake NeptuneClient whose source a ``reset`` replaces."""

    def __init__(self) -> None:
        self.current = Graph().traversal()

    @property
    def g(self):
        return self.current

    def reset(self, g: object = None) -> None:
        self.current = Graph().traversal()

    def close(self) -> None:
        pass


async def test_a_queued_traversal_uses_the_source_of_the_moment(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: aretrieve took ``g`` before waiting for a pool slot, so a
    # query queued while another one reset the connection sent on the closed
    # connection, whose pool.get() then blocked its thread for good.
    config.aws.neptune.pool_size = 1
    release = threading.Event()
    graphs: list[object] = []

    def _to_list(self) -> list:
        graphs.append(self.graph)
        if len(graphs) == 1:
            release.wait(5)
        return []

    monkeypatch.setattr(GraphTraversal, "to_list", _to_list)
    client = _SwappableClient()
    retriever = _neptune_retriever(config, client)
    first = asyncio.ensure_future(retriever.aretrieve(_SEEDED))
    while not graphs:
        await asyncio.sleep(0.01)
    queued = asyncio.ensure_future(retriever.aretrieve(_SEEDED))
    await asyncio.sleep(0.05)
    stale = client.current.graph
    client.reset()
    release.set()
    await asyncio.gather(first, queued)
    assert graphs[0] is stale
    assert graphs[1] is client.current.graph


async def test_traversals_run_on_the_retrievers_own_threads(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: traversals ran on the loop's default executor, so ones that
    # hung starved the model calls and every other strategy that share it.
    names: list[str] = []

    def _to_list(self) -> list:
        names.append(threading.current_thread().name)
        return []

    monkeypatch.setattr(GraphTraversal, "to_list", _to_list)
    retriever = _neptune_retriever(config, _SwappableClient())
    await retriever.aretrieve(_SEEDED)
    assert names and names[0].startswith("neptune-retriever")
    retriever.close()
    assert retriever._executor is None


async def test_neptune_transient_traversal_error_degrades_from_aretrieve(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _transient_to_list(self) -> list:
        raise RuntimeError("Read timed out")

    monkeypatch.setattr(GraphTraversal, "to_list", _transient_to_list)
    retriever = _neptune_retriever(config, _ClientWithLocalGraph())
    assert await retriever.aretrieve(_SEEDED) == []


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
    model = configured_embedding_model(config)
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
    model = configured_embedding_model(config)
    retriever = _retriever_over(
        config, _knn_mappings(_QUERY_DIMENSION, model, _QUERY_DIMENSION)
    )

    for _ in range(2):
        results = await retriever.aretrieve(_text_units_query(config))
        assert [r.source for r in results] == ["t1"]

    assert len(retriever._opensearch_client.mapping_calls) == 1


async def test_enum_model_id_matches_the_recorded_string(config: Config) -> None:
    # The config holds an EmbeddingModelId member; the index records its value.
    config.models.embedding.model_id = EmbeddingModelId.TITAN_EMBED_V2
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


@pytest.mark.parametrize("connection_cls", _CONNECTIONS)
async def test_invalid_dimension_rejection_is_a_fatal_configuration_error(
    config: Config, connection_cls: type[Connection]
) -> None:
    raw = _server_error(connection_cls, 400, _INVALID_DIMENSION_BODY)
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
    config.models.embedding.dimension = configured
    factory = _RecordingEmbeddingFactory()

    retriever = OpenSearchRetriever(
        config, opensearch_client=object(), embedding_factory=factory
    )

    assert factory.requests == [{"dimensions": expected}]
    assert retriever._embedding_dimension == expected


# An Amazon OpenSearch Service access-policy denial (the text holds "auth").
_ACCESS_DENIED_BODY = {
    "Message": "User: arn:aws:iam::000000000000:role/search-only is not "
    "authorized to perform: es:ESHttpGet"
}


def _denied() -> Exception:
    return _through_handler(
        _server_error(AsyncHttpConnection, 403, _ACCESS_DENIED_BODY)
    )


async def test_a_denied_mapping_read_skips_the_check_with_one_warning(
    config: Config, recording_logger: _RecordingLogger
) -> None:
    # A search-only principal may not read mappings, and a 403 is fatal:
    # every vector query used to fail on the check.
    denied = _denied()
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
    retriever = _retriever_over(config, {}, search_error=_denied())
    retriever._opensearch_client.mappings = {}

    with pytest.raises(ConfigurationError, match="denied the search.*status=403"):
        await retriever.aretrieve(_text_units_query(config))


# Fine-grained access control's 403 names no fatal marker ("no permissions"),
# so a role without indices:data/read/search answered every query from no
# context while the query reported success.
_FGAC_DENIED_BODY = {
    "error": {
        "root_cause": [
            {
                "type": "security_exception",
                "reason": "no permissions for [indices:data/read/search] "
                "and User [name=query-role, backend_roles=[query-role]]",
            }
        ],
        "type": "security_exception",
        "reason": "no permissions for [indices:data/read/search] "
        "and User [name=query-role, backend_roles=[query-role]]",
    },
    "status": 403,
}


@pytest.mark.parametrize("status", [401, 403])
async def test_a_search_denied_by_fine_grained_access_control_is_fatal(
    config: Config, status: int
) -> None:
    raw = _server_error(AsyncHttpConnection, status, _FGAC_DENIED_BODY)
    with pytest.raises(StoreAccessDeniedError) as wrapped:
        await _handle_async_opensearch_errors(_raise_async(raw))(None)
    retriever = _retriever_over(config, {}, search_error=wrapped.value)
    retriever._opensearch_client.mappings = {}

    with pytest.raises(ConfigurationError, match="indices:data/read/search"):
        await retriever.aretrieve(_text_units_query(config))


def _raise_async(exc: Exception):
    async def _raise(_self: object) -> None:
        raise exc

    return _raise


# --- rejected requests (HTTP 400) -------------------------------------------
#
# Regression: a 400 is the same on every retry, yet it degraded to [] like a
# timeout, so a query OpenSearch cannot parse or a filter value of the wrong type
# answered from no context (or from the other, unfiltered indices), and the
# log kept only "status=400, info=None".


_UNPARSABLE_QUERY_BODY = {
    "error": {
        "root_cause": [
            {
                "type": "parsing_exception",
                "reason": "unknown query [match_phrase_infix]",
            }
        ],
        "type": "parsing_exception",
        "reason": "unknown query [match_phrase_infix]",
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
        _server_error(connection_cls, 400, _UNPARSABLE_QUERY_BODY)
    )
    assert isinstance(handled, StoreRequestRejectedError)
    assert "status=400" in str(handled)
    assert "unknown query [match_phrase_infix]" in str(handled)
    assert handled.reason is not None
    assert "unknown query [match_phrase_infix]" in handled.reason


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (_UNPARSABLE_QUERY_BODY, "unknown query [match_phrase_infix]"),
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


_CLAUSE_LIMIT_BODY = {
    "error": {
        "root_cause": [
            {
                "type": "too_many_nested_clauses",
                "reason": "Query contains too many nested clauses; "
                "maxClauseCount is set to 1024",
            }
        ],
        "type": "search_phase_execution_exception",
        "reason": "all shards failed",
    },
    "status": 500,
}


@pytest.mark.parametrize("connection_cls", _CONNECTIONS)
async def test_a_query_over_the_clause_limit_fails_the_query(
    config: Config, connection_cls: type[Connection]
) -> None:
    # Regression: OpenSearch answers it with a 500, which degraded the
    # section to no results although the same query fails every time.
    error = _through_handler(_server_error(connection_cls, 500, _CLAUSE_LIMIT_BODY))
    assert isinstance(error, StoreRequestRejectedError)
    retriever = _retriever_over(config, {}, search_error=error)
    retriever._opensearch_client.mappings = {}

    with pytest.raises(ConfigurationError, match="too many nested clauses"):
        await retriever.aretrieve(_text_units_query(config))


async def test_a_server_error_of_one_search_stays_transient(config: Config) -> None:
    body = {"error": {"type": "some_exception", "reason": "busy"}, "status": 500}
    error = _through_handler(_server_error(AsyncHttpConnection, 500, body))
    retriever = _retriever_over(config, {}, search_error=error)
    retriever._opensearch_client.mappings = {}

    assert await retriever.aretrieve(_text_units_query(config)) == []


@pytest.mark.parametrize("status", [429, 502, 503, 504])
async def test_throttling_or_unavailable_after_retries_fails_the_query(
    config: Config, status: int
) -> None:
    # Regression: the transport retries these statuses; a cluster still
    # answering them (overloaded, restarting) answered from no context.
    body = {"error": {"type": "some_exception", "reason": "busy"}, "status": status}
    error = _through_handler(_server_error(AsyncHttpConnection, status, body))
    retriever = _retriever_over(config, {}, search_error=error)
    retriever._opensearch_client.mappings = {}

    with pytest.raises(StoreConnectionError, match=f"status={status}"):
        await retriever.aretrieve(_text_units_query(config))


# --- the embedding-settings check's mapping read -----------------------------


def _mapping_read_failing_with(config: Config, exc: Exception) -> OpenSearchRetriever:
    retriever = _retriever_over(config, {})

    async def _aget_mappings(index: str) -> dict[str, dict]:
        retriever._opensearch_client.mapping_calls.append(index)
        raise exc

    retriever._opensearch_client.aget_mappings = _aget_mappings
    return retriever


async def test_a_mapping_read_that_timed_out_fails_the_query(config: Config) -> None:
    raw = ConnectionTimeout("TIMEOUT", "timed out", Exception())
    retriever = _mapping_read_failing_with(config, _through_handler(raw))

    with pytest.raises(StoreTimeoutError, match="did not answer a request on"):
        await retriever.aretrieve(_text_units_query(config))


async def test_an_unreachable_cluster_fails_the_mapping_read(config: Config) -> None:
    raw = OpenSearchConnectionError("N/A", "refused", Exception())
    retriever = _mapping_read_failing_with(config, _through_handler(raw))

    with pytest.raises(StoreConnectionError, match="refused"):
        await retriever.aretrieve(_text_units_query(config))


async def test_a_failed_mapping_read_leaves_the_check_to_the_search(
    config: Config,
) -> None:
    body = {"error": {"type": "some_exception", "reason": "busy"}, "status": 500}
    error = _through_handler(_server_error(AsyncHttpConnection, 500, body))
    retriever = _mapping_read_failing_with(config, error)

    assert len(await retriever.aretrieve(_text_units_query(config))) == 1


@pytest.mark.parametrize("search_type", [SearchType.HYBRID, SearchType.VECTOR])
async def test_a_fatal_search_error_cancels_the_searches_beside_it(
    config: Config, search_type: SearchType
) -> None:
    # The searches of the other indices, and the other half of an index's
    # lexical + kNN pair, kept going to the store after the query had failed.
    started: list[asyncio.Task] = []
    cancelled: list[str] = []

    async def _asearch(index: str, body: dict) -> dict:
        if not started:
            started.append(asyncio.current_task())  # type: ignore[arg-type]
            await asyncio.sleep(0.05)
            raise StoreConnectionError("refused")
        started.append(asyncio.current_task())  # type: ignore[arg-type]
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.append(index)
            raise
        return {}

    retriever = _opensearch_retriever(config)
    _set_asearch_raising(retriever, RuntimeError("unused"))
    object.__setattr__(retriever, "_opensearch_client", _MockSearchClient(_asearch))
    with pytest.raises(StoreConnectionError, match="refused"):
        await asyncio.wait_for(
            retriever.aretrieve(SearchQuery(query="hello", search_type=search_type)),
            timeout=30,
        )
    assert len(started) > 1
    assert len(cancelled) == len(started) - 1


@pytest.mark.parametrize("read", ["aget_mappings", "aget_index_settings"])
async def test_a_failed_index_read_cancels_the_reads_beside_it(
    config: Config, read: str
) -> None:
    # The mapping and settings reads before a search, one per index, ran
    # under a plain gather too: the others kept going to the store after one
    # had failed the query (a store that never answered).
    started: list[str] = []
    cancelled: list[str] = []

    async def _read(index: str) -> dict:
        started.append(index)
        if len(started) == 1:
            await asyncio.sleep(0.05)
            raise StoreTimeoutError("timed out")
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.append(index)
            raise
        return {}

    retriever = _opensearch_retriever(config)
    client = _MockSearchClient(None)
    setattr(client, read, _read)
    object.__setattr__(retriever, "_opensearch_client", client)
    o = config.indexing.opensearch
    prefixes = [o.entities_index_prefix, o.text_units_index_prefix]
    check = (
        retriever._check_embedding_settings
        if read == "aget_mappings"
        else retriever._limits_by_prefix
    )
    with pytest.raises(StoreTimeoutError, match="did not answer a request on"):
        await asyncio.wait_for(check(prefixes, None), timeout=30)
    assert len(started) == 2
    assert cancelled == started[1:]
