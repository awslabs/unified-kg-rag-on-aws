# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discarding a rotated AsyncOpenSearch client is await-safe (AWS-free).

Regression: the discard path called aiohttp's ``connector.close()`` (a
coroutine-like awaitable in aiohttp 3.x) without awaiting it, and never
closed a client whose loop was still alive, which surfaced as "Unclosed client
session" / never-awaited warnings in E2E logs. The connectors are closed
synchronously on the thread of the loop that owns them (posted there as a
callback when that loop runs on another thread), or at once when the loop is
gone, the connector's awaitable settled.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import gc
import ssl
import tempfile
import threading
import time
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import aiohttp
import pytest

from unified_kg_rag.adapters.aws import opensearch
from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient
from unified_kg_rag.application.retrieval.rag_chain import _LoopRunner

pytestmark = pytest.mark.unit


def _client_over(session: aiohttp.ClientSession) -> MagicMock:
    connection = MagicMock(session=session)
    pool = MagicMock(connections=[connection])
    return MagicMock(transport=MagicMock(connection_pool=pool))


def test_discard_after_the_loop_closed_emits_no_warnings() -> None:
    loop = asyncio.new_event_loop()

    async def make_session() -> aiohttp.ClientSession:
        return aiohttp.ClientSession()

    session = loop.run_until_complete(make_session())
    loop.close()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        OpenSearchClient._discard_async_client(_client_over(session), loop)
        assert session.closed
        del session
        gc.collect()

    messages = [str(w.message) for w in caught]
    assert not [m for m in messages if "await" in m or "Unclosed" in m], messages


def _bound(
    async_client: MagicMock, loop: asyncio.AbstractEventLoop
) -> OpenSearchClient:
    client = OpenSearchClient.__new__(OpenSearchClient)
    client._client = None
    client._async_client = async_client
    client._bound_loop_id = id(loop)
    client._bound_loop = loop
    return client


def test_discard_while_the_loop_runs_closes_the_connections_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On the loop that owns them, as a callback (never a task its loop's
    # end can cancel before it runs).
    runner = _LoopRunner()
    seen: list[asyncio.AbstractEventLoop] = []
    close = opensearch._close_connections

    def recorded(async_client: MagicMock) -> None:
        seen.append(asyncio.get_running_loop())
        close(async_client)

    monkeypatch.setattr(opensearch, "_close_connections", recorded)

    async def make_session() -> aiohttp.ClientSession:
        return aiohttp.ClientSession()

    try:
        session = runner.run(make_session())
        loop = runner.run(_running())
        OpenSearchClient._discard_async_client(_client_over(session), loop)
        runner.run(asyncio.sleep(0.01))
        assert seen == [loop]
        assert session.closed
    finally:
        runner.stop()


async def _running() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


async def test_close_from_inside_the_bound_loop_closes_the_connections_at_once() -> (
    None
):
    session = aiohttp.ClientSession()
    client = _bound(_client_over(session), asyncio.get_running_loop())

    client.close()

    # Before any await: nothing is left for the loop to run.
    assert session.closed
    assert client._async_client is None and client._bound_loop is None


def test_a_sync_close_that_blocks_the_bound_loop_still_closes_the_session() -> None:
    # R4B-1: GraphRAGChain.close() called from a coroutine runs the
    # retrievers' sync closes on a worker thread while the loop's thread
    # waits for it; the awaited close the worker scheduled on that loop was
    # cancelled by asyncio.run before its first step, and the session and
    # its sockets stayed open (7 of 7 in probes/chain_close_leak.py).
    held: list[aiohttp.ClientSession] = []

    async def main() -> None:
        session = aiohttp.ClientSession()
        held.append(session)
        async_client = _client_over(session)

        async def close() -> None:
            await session.close()

        async_client.close = close
        client = _bound(async_client, asyncio.get_running_loop())
        worker = threading.Thread(target=client.close)
        worker.start()
        worker.join()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(main())
        assert held[0].closed
        held.clear()
        gc.collect()
    assert not [w for w in caught if "Unclosed" in str(w.message)]


async def test_async_client_binds_the_loop_object(monkeypatch) -> None:
    client = OpenSearchClient.__new__(OpenSearchClient)
    client._client = None
    client._async_client = None
    client._bound_loop_id = None
    client._bound_loop = None
    monkeypatch.setattr(client, "_create_async_client", MagicMock)

    client.async_client  # noqa: B018 - property access binds the loop

    assert client._bound_loop is asyncio.get_running_loop()


class _KeepAliveServer:
    """A keep-alive HTTP server on a loop thread of its own (plain or TLS):
    the connections it accepted and those whose client closed them."""

    def __init__(self, ssl_context: Any = None) -> None:
        self.accepted = 0
        self.closed = 0
        self._runner = _LoopRunner()
        self._server = self._runner.run(
            asyncio.start_server(self._handle, "127.0.0.1", 0, ssl=ssl_context)
        )
        self.port = self._server.sockets[0].getsockname()[1]

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.accepted += 1
        try:
            while True:
                await reader.readuntil(b"\r\n\r\n")
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                    b"Connection: keep-alive\r\n\r\nok"
                )
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            self.closed += 1
            writer.close()

    def wait_closed(self, count: int, timeout: float = 5.0) -> int:
        deadline = time.monotonic() + timeout
        while self.closed < count and time.monotonic() < deadline:
            time.sleep(0.01)
        return self.closed

    def stop(self) -> None:
        # Closed on its own loop: asyncio.Server is not thread-safe, and a
        # close() from this thread raced the loop thread detaching a
        # connection the test had just closed (``_wakeup`` iterated None).
        self._runner.run(self._close())
        self._runner.stop()

    async def _close(self) -> None:
        self._server.close()
        # Bounded: on 3.12+ wait_closed also waits for open connections,
        # which a failed assertion above may have left open.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._server.wait_closed(), timeout=5.0)


def _tls_contexts() -> tuple[ssl.SSLContext, ssl.SSLContext]:
    """A server context with a throwaway self-signed certificate, and a
    client context that does not verify it."""
    pytest.importorskip("cryptography")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .sign(key, hashes.SHA256())
    )
    with tempfile.TemporaryDirectory() as folder:
        cert_path, key_path = Path(folder, "c.pem"), Path(folder, "k.pem")
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        server = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        server.load_cert_chain(cert_path, key_path)
    client = ssl.create_default_context()
    client.check_hostname = False
    client.verify_mode = ssl.CERT_NONE
    return server, client


def _session_of_a_closed_loop(
    url: str, client_ssl: Any, connections: int
) -> tuple[aiohttp.ClientSession, asyncio.AbstractEventLoop]:
    """A session whose pool holds ``connections`` keep-alive connections,
    made in an ``asyncio.run`` that has returned (its loop is closed)."""
    made: list[Any] = []

    async def main() -> None:
        session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=client_ssl))

        async def get() -> None:
            async with session.get(url) as response:
                await response.read()

        await asyncio.gather(*(get() for _ in range(connections)))
        made.extend([session, asyncio.get_running_loop()])

    asyncio.run(main())
    return made[0], made[1]


@pytest.mark.parametrize("tls", [False, True])
@pytest.mark.parametrize("how", ["close", "aclose"])
def test_closing_a_client_whose_loop_is_closed_closes_its_sockets(
    tls: bool, how: str
) -> None:
    # R5B-2: aiohttp's connector close returns early on a closed loop, so a
    # client bound to the loop of an earlier asyncio.run (the chain's
    # dead-loop eviction, close() or aclose() after it) closed none of its
    # sockets: each stayed open until the cyclic GC collected its transport.
    server_ssl, client_ssl = _tls_contexts() if tls else (None, False)
    server = _KeepAliveServer(server_ssl)
    scheme = "https" if tls else "http"
    gc.disable()
    try:
        session, loop = _session_of_a_closed_loop(
            f"{scheme}://127.0.0.1:{server.port}/", client_ssl, connections=3
        )
        assert loop.is_closed()
        assert server.accepted == 3 and server.closed == 0
        client = _bound(_client_over(session), loop)
        transports = [
            opensearch._socket_transport(proto.transport)
            for held in session.connector._conns.values()  # type: ignore[union-attr]
            for proto, _ in held
        ]
        assert len(transports) == 3

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if how == "close":
                client.close()
            else:
                asyncio.run(client.aclose())
            # Without the GC: every connection is closed at once.
            assert server.wait_closed(3) == 3
            assert session.closed
            # Closed as the loop would have closed them: a later close of
            # one does nothing.
            assert all(t.is_closing() for t in transports)
            for transport in transports:
                transport.close()
            del session, client
            gc.collect()
        assert not [w for w in caught if issubclass(w.category, ResourceWarning)]
    finally:
        gc.enable()
        server.stop()
