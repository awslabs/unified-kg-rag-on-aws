# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for NeptuneClient helper logic (AWS-free).

No real Neptune/Gremlin connection is opened. ``DriverRemoteConnection`` is
patched and the traversal source ``g`` is faked, so we exercise: SigV4 auth-header
assembly, ``_create_connection`` URL/header wiring and the missing-endpoint
guard.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from gremlin_python.driver.protocol import GremlinServerError

from unified_kg_rag.adapters.aws import neptune as neptune_mod
from unified_kg_rag.adapters.aws.neptune import NeptuneClient
from unified_kg_rag.adapters.retrieval.base import is_fatal_retrieval_error
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import AWSServiceError, StoreConnectionError
from unified_kg_rag.shared.exceptions import StoreAccessDeniedError, StoreTimeoutError

pytestmark = pytest.mark.unit


# --- fakes ---------------------------------------------------------------


class _FakeCreds:
    def get_frozen_credentials(self) -> Any:
        # botocore SigV4Auth uses .access_key/.secret_key/.token attributes.
        from botocore.credentials import ReadOnlyCredentials

        return ReadOnlyCredentials("AKIDEXAMPLE", "secret", None)


class _FakeSession:
    def __init__(self, creds: Any | None = _FakeCreds()) -> None:
        self._creds = creds

    def get_credentials(self) -> Any:
        return self._creds


def _client(config: Config, session: Any | None = None) -> NeptuneClient:
    return NeptuneClient(config, boto_session=session or _FakeSession())


def _config_with_neptune(
    endpoint: str = "db.example.com", use_iam: bool = True
) -> Config:
    config = Config()
    config.aws.neptune.endpoint = endpoint
    config.aws.neptune.use_iam = use_iam
    config.aws.region_name = "us-east-1"
    return config


# --- _get_auth_headers ---------------------------------------------------


def test_get_auth_headers_signs_request() -> None:
    client = _client(_config_with_neptune())
    headers = client._get_auth_headers("wss://db.example.com:8182/gremlin")
    # SigV4 stamps Authorization + X-Amz-Date headers.
    assert "Authorization" in headers
    assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
    assert "X-Amz-Date" in headers


def test_get_auth_headers_raises_without_credentials() -> None:
    client = _client(_config_with_neptune(), session=_FakeSession(creds=None))
    with pytest.raises(AWSServiceError, match="Unable to get AWS credentials"):
        client._get_auth_headers("wss://db.example.com:8182/gremlin")


# --- _create_connection --------------------------------------------------


def test_create_connection_missing_endpoint_raises() -> None:
    config = _config_with_neptune(endpoint="")
    client = _client(config)
    with pytest.raises(AWSServiceError, match="endpoint is not configured"):
        client._create_connection()


def test_create_connection_builds_url_and_iam_headers(mocker) -> None:
    config = _config_with_neptune(use_iam=True)
    config.aws.neptune.port = 8182
    client = _client(config)

    captured: dict[str, Any] = {}

    class _FakeConn:
        def __init__(
            self,
            url: str,
            traversal_source: str,
            pool_size: int | None = None,
            max_workers: int | None = None,
            transport_factory: Any = None,
        ) -> None:
            captured["url"] = url
            captured["traversal_source"] = traversal_source
            captured["transport"] = transport_factory()
            captured["pool_size"] = pool_size
            captured["max_workers"] = max_workers

    # Patch the connection + the traversal builder so no socket opens.
    mocker.patch.object(neptune_mod, "DriverRemoteConnection", _FakeConn)
    fake_g = mocker.MagicMock()
    fake_g.V.return_value.limit.return_value.toList.return_value = []
    mocker.patch.object(
        neptune_mod,
        "traversal",
        return_value=mocker.MagicMock(withRemote=lambda c: fake_g),
    )

    conn = client._create_connection()
    assert isinstance(conn, _FakeConn)
    assert captured["url"] == "wss://db.example.com:8182/gremlin"
    assert captured["traversal_source"] == "g"
    # IAM enabled -> every websocket handshake is signed when it opens.
    signed = captured["transport"]._sign(captured["url"])
    assert signed["Authorization"].startswith("AWS4-HMAC-SHA256")
    # pool_size/max_workers wired from config (default 4) for concurrent writes.
    assert captured["pool_size"] == 4
    assert captured["max_workers"] == 4


def test_create_connection_no_iam_empty_headers(mocker) -> None:
    config = _config_with_neptune(use_iam=False)
    client = _client(config)
    captured: dict[str, Any] = {}

    class _FakeConn:
        def __init__(
            self,
            url: str,
            traversal_source: str,
            pool_size: int | None = None,
            max_workers: int | None = None,
            transport_factory: Any = None,
        ) -> None:
            captured["transport"] = transport_factory()

    mocker.patch.object(neptune_mod, "DriverRemoteConnection", _FakeConn)
    fake_g = mocker.MagicMock()
    fake_g.V.return_value.limit.return_value.toList.return_value = []
    mocker.patch.object(
        neptune_mod,
        "traversal",
        return_value=mocker.MagicMock(withRemote=lambda c: fake_g),
    )
    client._create_connection()
    assert captured["transport"]._sign is None


def test_failed_connection_probe_closes_the_connection(mocker) -> None:
    # The probe query runs after DriverRemoteConnection opened its websocket
    # and thread pool; a failed probe must not leave them behind.
    config = _config_with_neptune(endpoint="localhost", use_iam=False)
    connection = mocker.patch.object(neptune_mod, "DriverRemoteConnection")
    g = mocker.MagicMock()
    g.V.return_value.limit.return_value.toList.side_effect = OSError("refused")
    mocker.patch.object(
        neptune_mod, "traversal", return_value=mocker.MagicMock(withRemote=lambda c: g)
    )
    with pytest.raises(AWSServiceError, match="refused"):
        _client(config)._create_connection()
    connection.return_value.close.assert_called_once()


def test_failed_connection_close_error_keeps_the_original_error(mocker) -> None:
    config = _config_with_neptune(endpoint="localhost", use_iam=False)
    connection = mocker.patch.object(neptune_mod, "DriverRemoteConnection")
    connection.return_value.close.side_effect = RuntimeError("close failed")
    g = mocker.MagicMock()
    g.V.return_value.limit.return_value.toList.side_effect = OSError("refused")
    mocker.patch.object(
        neptune_mod, "traversal", return_value=mocker.MagicMock(withRemote=lambda c: g)
    )
    with pytest.raises(AWSServiceError, match="refused"):
        _client(config)._create_connection()


def test_create_connection_without_ssl_uses_plain_websocket(mocker) -> None:
    # A local TinkerPop Gremlin Server listens on ws:// without TLS or IAM.
    config = _config_with_neptune(endpoint="localhost", use_iam=False)
    config.aws.neptune.use_ssl = False
    connection = mocker.patch.object(neptune_mod, "DriverRemoteConnection")
    mocker.patch.object(neptune_mod, "traversal")
    _client(config)._create_connection()
    kwargs = connection.call_args.kwargs
    assert kwargs["url"] == "ws://localhost:8182/gremlin"
    assert kwargs["transport_factory"]()._sign is None


@pytest.mark.parametrize(
    ("pool_size", "index_concurrency", "expected"),
    [(4, 1, 4), (4, 8, 8), (8, 8, 8)],
)
def test_create_connection_pool_covers_index_concurrency(
    mocker, pool_size: int, index_concurrency: int, expected: int
) -> None:
    # Concurrent write batches must not queue on a smaller connection pool.
    config = _config_with_neptune(use_iam=False)
    config.aws.neptune.pool_size = pool_size
    config.indexing.neptune.index_concurrency = index_concurrency
    connection = mocker.patch.object(neptune_mod, "DriverRemoteConnection")
    mocker.patch.object(neptune_mod, "traversal")
    _client(config)._create_connection()
    kwargs = connection.call_args.kwargs
    assert (kwargs["pool_size"], kwargs["max_workers"]) == (expected, expected)


def test_create_connection_wraps_failure(mocker) -> None:
    config = _config_with_neptune()
    client = _client(config)

    def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("network down")

    mocker.patch.object(neptune_mod, "DriverRemoteConnection", _boom)
    with pytest.raises(AWSServiceError, match="probe failed.*network down") as caught:
        client._create_connection()
    assert not isinstance(caught.value, StoreConnectionError)


def _probe_failing_with(mocker, exc: Exception) -> NeptuneClient:
    mocker.patch.object(neptune_mod, "DriverRemoteConnection")
    g = mocker.MagicMock()
    g.V.return_value.limit.return_value.toList.side_effect = exc
    mocker.patch.object(
        neptune_mod, "traversal", return_value=mocker.MagicMock(withRemote=lambda c: g)
    )
    return _client(_config_with_neptune(endpoint="localhost", use_iam=False))


def test_a_refused_connection_is_a_store_connection_error(mocker) -> None:
    client = _probe_failing_with(mocker, ConnectionRefusedError("refused"))
    with pytest.raises(StoreConnectionError, match="not reachable"):
        client._create_connection()


def test_a_throttled_probe_stays_transient(mocker) -> None:
    # Every probe failure was "Failed to establish connection" (a fatal
    # marker), so a probe that was throttled failed the query.
    exc = GremlinServerError(
        {"code": 500, "message": "ThrottlingException", "attributes": {}}
    )
    client = _probe_failing_with(mocker, exc)
    with pytest.raises(AWSServiceError) as caught:
        client._create_connection()
    assert not is_fatal_retrieval_error(caught.value)


def test_a_probe_that_timed_out_is_a_store_timeout(mocker) -> None:
    # A query fails on a Neptune that does not answer (search.store_timeout_seconds).
    client = _probe_failing_with(mocker, TimeoutError())
    with pytest.raises(StoreTimeoutError, match="did not answer"):
        client._create_connection()


# --- connection attempts under concurrency ---------------------------------
#
# Regression: each query waiting for the client lock made its own connection
# attempt after the previous one failed, so N queries of an unreachable
# Neptune took N x the connect timeout and held N executor threads meanwhile.


def test_callers_waiting_on_a_failed_attempt_share_its_failure(mocker) -> None:
    import threading
    import time

    attempts: list[float] = []

    def _unreachable(self: NeptuneClient) -> Any:
        attempts.append(time.monotonic())
        time.sleep(0.3)
        raise StoreConnectionError("Neptune is not reachable")

    mocker.patch.object(NeptuneClient, "_create_connection", _unreachable)
    client = _client(_config_with_neptune(endpoint="localhost", use_iam=False))
    errors: list[BaseException] = []
    start = threading.Barrier(8)

    def _query() -> None:
        start.wait()
        try:
            client.g
        except StoreConnectionError as e:
            errors.append(e)

    threads = [threading.Thread(target=_query) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert len(errors) == 8
    assert len(attempts) == 1
    # A caller arriving after the failure tries again.
    with pytest.raises(StoreConnectionError):
        client.g
    assert len(attempts) == 2


def _traceback_depth(exc: BaseException) -> int:
    depth, tb = 0, exc.__traceback__
    while tb is not None:
        depth, tb = depth + 1, tb.tb_next
    return depth


def test_each_waiter_gets_its_own_copy_of_the_failure() -> None:
    # Regression: every waiter re-raised the one shared error, which grew its
    # traceback by the waiter's frames each time and kept them alive.
    client = _client(_config_with_neptune(endpoint="localhost", use_iam=False))
    try:
        raise StoreAccessDeniedError("Neptune denied", reason="synthetic")
    except StoreAccessDeniedError as e:
        failure = e
    client._attempts, client._attempt_error = 1, failure
    depth = _traceback_depth(failure)

    raised = []
    for _ in range(5):
        with pytest.raises(StoreAccessDeniedError) as exc:
            client._connected(attempts_seen=0)
        raised.append(exc.value)

    assert _traceback_depth(failure) == depth
    assert all(e is not failure and e.__cause__ is failure for e in raised)
    assert {(str(e), e.reason) for e in raised} == {("Neptune denied", "synthetic")}


# --- _NeptuneTransport.connect -----------------------------------------------


class _Handshake:
    """Stands in for ``aiohttp.ClientSession.ws_connect``; records the headers."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.headers: list[dict[str, str]] = []

    async def __call__(self, url: str, **kwargs: Any) -> object:
        self.headers.append(dict(kwargs.get("headers") or {}))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(closed=True)  # a websocket close() skips


def _handshake_error(status: int) -> Exception:
    import aiohttp

    return aiohttp.WSServerHandshakeError(
        request_info=_request_info(), history=(), status=status
    )


def _request_info() -> Any:
    import aiohttp
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL

    url = URL("ws://localhost:8182/gremlin")
    return aiohttp.RequestInfo(url, "GET", CIMultiDictProxy(CIMultiDict()), url)


def test_each_socket_is_signed_on_its_handshake(mocker) -> None:
    import aiohttp

    handshake = _Handshake()
    mocker.patch.object(aiohttp.ClientSession, "ws_connect", handshake)
    signed: list[str] = []

    def _sign(url: str) -> dict[str, str]:
        signed.append(url)
        return {"Authorization": f"sig-{len(signed)}"}

    transport = neptune_mod._NeptuneTransport(5, _sign)
    try:
        transport.connect("wss://db.example.com:8182/gremlin", {"User-Agent": "x"})
    finally:
        transport.close()

    assert signed == ["wss://db.example.com:8182/gremlin"]
    assert handshake.headers == [{"User-Agent": "x", "Authorization": "sig-1"}]


@pytest.mark.parametrize("status", [401, 403])
def test_a_denied_handshake_is_an_access_denied_error(mocker, status: int) -> None:
    import aiohttp

    from unified_kg_rag.shared.exceptions import StoreAccessDeniedError

    mocker.patch.object(
        aiohttp.ClientSession, "ws_connect", _Handshake(_handshake_error(status))
    )
    transport = neptune_mod._NeptuneTransport(5, None)
    with pytest.raises(StoreAccessDeniedError, match=f"HTTP {status}") as caught:
        transport.connect("wss://db.example.com:8182/gremlin")
    # Not chained: the response repr carries the signed headers.
    assert caught.value.__cause__ is None
    assert transport._client_session is None or transport._client_session.closed


def test_another_handshake_failure_is_raised_as_is(mocker) -> None:
    import aiohttp

    mocker.patch.object(
        aiohttp.ClientSession, "ws_connect", _Handshake(_handshake_error(500))
    )
    transport = neptune_mod._NeptuneTransport(5, None)
    with pytest.raises(aiohttp.WSServerHandshakeError) as caught:
        transport.connect("wss://db.example.com:8182/gremlin")
    assert caught.value.status == 500


def test_reset_wakes_requests_waiting_for_a_pooled_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: gremlinpython's close() empties the pool, and a request
    # already waiting in pool.get() (no timeout) held its thread for good:
    # after a few outages the executor running the traversals had no thread
    # left and every query hung.
    import threading
    import time

    from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

    connection = DriverRemoteConnection("ws://127.0.0.1:9/gremlin", "g", pool_size=1)
    monkeypatch.setattr(NeptuneClient, "_create_connection", lambda self: connection)
    client = _client(_config_with_neptune(endpoint="localhost", use_iam=False))
    g = client.g
    connection._client._pool.get_nowait()  # its one connection is in use
    errors: list[Exception] = []

    def _query() -> None:
        try:
            g.V().limit(1).to_list()
        except Exception as e:  # noqa: BLE001 - asserted below
            errors.append(e)

    # More waiters than the pool has connections.
    threads = [threading.Thread(target=_query, daemon=True) for _ in range(3)]
    for thread in threads:
        thread.start()
    time.sleep(0.2)
    client.reset(g)
    for thread in threads:
        thread.join(5)

    assert not any(thread.is_alive() for thread in threads)
    assert len(errors) == 3
    assert all(neptune_mod.is_connection_lost(e) for e in errors)


def test_a_request_on_a_closed_client_is_a_lost_connection() -> None:
    assert neptune_mod.is_connection_lost(Exception("Client is closed"))
