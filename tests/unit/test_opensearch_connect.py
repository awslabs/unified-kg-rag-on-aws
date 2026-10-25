# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opening the OpenSearch indexing client against a cluster that never
answers (AWS-free: a local socket that accepts connections and stays silent).

The connection check sent its HEAD with the indexing client's 180 s timeout
and 5 retries, so the store preflight blocked for 18 minutes before it
reported the store unreachable.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Iterator

import pytest

from unified_kg_rag.adapters.aws import opensearch
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import StoreConnectionError

pytestmark = pytest.mark.unit


@pytest.fixture
def silent_port() -> Iterator[int]:
    """A port whose listener completes the TCP handshake (the kernel's
    backlog) and never reads or answers."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    try:
        yield server.getsockname()[1]
    finally:
        server.close()


@pytest.mark.timeout(30)
def test_a_silent_cluster_fails_the_connection_check_quickly(
    monkeypatch: pytest.MonkeyPatch, silent_port: int
) -> None:
    monkeypatch.setattr(opensearch, "_PING_TIMEOUT_SECONDS", 0.2, raising=False)
    config = Config()
    store = config.aws.opensearch
    store.endpoint, store.port = "127.0.0.1", silent_port
    store.use_ssl, store.use_iam, store.allow_anonymous = False, False, True
    started = time.monotonic()
    with pytest.raises(StoreConnectionError, match="Failed to connect"):
        opensearch.OpenSearchClient(config).client
    # Six attempts of 0.2 s, not of the 180 s an indexing request may take.
    assert time.monotonic() - started < 10
