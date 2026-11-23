# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closing an OpenSearch client whose event loop closed, when aiohttp's or
asyncio's private shapes differ from the ones the socket release reads
(AWS-free): the release is best-effort, the connector's close is not.

R6B-1: a raise while the release read a pool of another shape (``_conns``
a list, entries that are not pairs, a read-only ``_closing``, no
``_conns`` or no ``_loop``) skipped the connector's close for that
connection and every later one, so their sessions stayed open.
"""

from __future__ import annotations

import collections
import gc
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.unit.test_opensearch_async_discard import (
    _bound,
    _KeepAliveServer,
    _session_of_a_closed_loop,
    _tls_contexts,
)
from unified_kg_rag.adapters.aws import opensearch

pytestmark = pytest.mark.unit


def _conns_list(connector: Any) -> None:
    connector._conns = [pair for held in connector._conns.values() for pair in held]


def _conns_triples(connector: Any) -> None:
    connector._conns = collections.defaultdict(
        collections.deque,
        {
            key: collections.deque((proto, at, 0) for proto, at in held)
            for key, held in connector._conns.items()
        },
    )


def _read_only_closing(connector: Any) -> None:
    # The first connection's transport only: the release goes on to the
    # next one.
    proto, _ = next(iter(connector._conns.values()))[0]
    transport = opensearch._socket_transport(proto.transport)

    class ReadOnly(type(transport)):  # type: ignore[misc]
        _closing = property(lambda self: False)

    transport.__class__ = ReadOnly


def _no_conns(connector: Any) -> None:
    del connector._conns


def _no_loop(connector: Any) -> None:
    del connector._loop


_SHAPES: dict[str, Callable[[Any], None]] = {
    "conns-list": _conns_list,
    "conns-triples": _conns_triples,
    "read-only-closing": _read_only_closing,
    "no-conns": _no_conns,
    "no-loop": _no_loop,
}


@pytest.mark.parametrize("tls", [False, True])
@pytest.mark.parametrize("shape", [*_SHAPES, "release-raises"])
def test_every_session_closes_whatever_the_release_reads(
    tls: bool, shape: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_ssl, client_ssl = _tls_contexts() if tls else (None, False)
    server = _KeepAliveServer(server_ssl)
    scheme = "https" if tls else "http"
    url = f"{scheme}://127.0.0.1:{server.port}/"
    gc.disable()
    try:
        made = [_session_of_a_closed_loop(url, client_ssl, connections=2)]
        made.append(_session_of_a_closed_loop(url, client_ssl, connections=2))
        sessions = [session for session, _, _ in made]
        if shape == "release-raises":

            def raises(_connector: Any) -> None:
                raise RuntimeError("another shape")

            monkeypatch.setattr(opensearch, "_close_sockets_of_closed_loop", raises)
        else:
            for session in sessions:
                _SHAPES[shape](session.connector)
        # Two pooled connections, each a session of its own (the pool's
        # connections), both of closed loops.
        pool = MagicMock(
            connections=[MagicMock(session=session) for session in sessions]
        )
        async_client = MagicMock(transport=MagicMock(connection_pool=pool))
        client = _bound(async_client, made[0][1])

        client.close()  # never raises

        assert [session.closed for session in sessions] == [True, True]
        if shape == "read-only-closing":
            # The other connection of each session: its socket closed.
            assert server.wait_closed(2) >= 2
    finally:
        gc.enable()
        gc.collect()
        server.stop()


def test_a_pool_whose_entries_are_not_pairs_keeps_the_readable_ones() -> None:
    # One unreadable key does not hide the others' protocols.
    good, bad = MagicMock(), MagicMock()
    connector = MagicMock(_conns={"a": [(good, 1.0)], "b": [(bad, 1.0, 0)]})
    connector._acquired = {"in-use"}
    assert opensearch._pooled_protocols(connector) == [good, "in-use"]


def test_unreadable_pools_read_as_no_protocols() -> None:
    class Connector:
        _conns = ["not", "a", "mapping"]
        _acquired = 3  # not iterable

    assert opensearch._pooled_protocols(Connector()) == []
    assert opensearch._pooled_protocols(object()) == []
