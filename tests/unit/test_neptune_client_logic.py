# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for NeptuneClient helper logic (AWS-free).

No real Neptune/Gremlin connection is opened. ``DriverRemoteConnection`` is
patched and the traversal source ``g`` is faked, so we exercise: SigV4 auth-header
assembly, ``_create_connection`` URL/header wiring, the missing-endpoint guard,
the ``delete_vertices_in_batches`` no-progress guard, ``get_graph_stats``
parsing, and the ``_handle_neptune_errors`` wrapping.
"""

from __future__ import annotations

from typing import Any

import pytest

from aws_graphrag.adapters.aws import neptune as neptune_mod
from aws_graphrag.adapters.aws.neptune import NeptuneClient
from aws_graphrag.domain.models import Config
from aws_graphrag.shared import AWSServiceError

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
        def __init__(self, url: str, traversal_source: str, headers: dict) -> None:
            captured["url"] = url
            captured["traversal_source"] = traversal_source
            captured["headers"] = headers

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
    # IAM enabled -> signed headers present.
    assert "Authorization" in captured["headers"]


def test_create_connection_no_iam_empty_headers(mocker) -> None:
    config = _config_with_neptune(use_iam=False)
    client = _client(config)
    captured: dict[str, Any] = {}

    class _FakeConn:
        def __init__(self, url: str, traversal_source: str, headers: dict) -> None:
            captured["headers"] = headers

    mocker.patch.object(neptune_mod, "DriverRemoteConnection", _FakeConn)
    fake_g = mocker.MagicMock()
    fake_g.V.return_value.limit.return_value.toList.return_value = []
    mocker.patch.object(
        neptune_mod,
        "traversal",
        return_value=mocker.MagicMock(withRemote=lambda c: fake_g),
    )
    client._create_connection()
    assert captured["headers"] == {}


def test_create_connection_wraps_failure(mocker) -> None:
    config = _config_with_neptune()
    client = _client(config)

    def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("network down")

    mocker.patch.object(neptune_mod, "DriverRemoteConnection", _boom)
    with pytest.raises(AWSServiceError, match="Failed to establish connection"):
        client._create_connection()


# --- delete_vertices_in_batches ------------------------------------------


class _FakeTraversal:
    """A fake GraphTraversalSource where V().hasLabel().count().next() returns
    successive scripted values, and drop().iterate() is a no-op."""

    def __init__(self, counts: list[int]) -> None:
        self._counts = list(counts)

    # Chainable no-op steps.
    def V(self):  # noqa: N802
        return self

    def hasLabel(self, label):  # noqa: N802
        return self

    def limit(self, n):
        return self

    def drop(self):
        return self

    def count(self):
        return self

    def iterate(self):
        return None

    def next(self):
        return self._counts.pop(0)


def _client_with_fake_g(monkeypatch, counts: list[int]) -> NeptuneClient:
    client = _client(_config_with_neptune())
    fake = _FakeTraversal(counts)
    # Bypass the connection property by injecting _g directly.
    monkeypatch.setattr(type(client), "g", property(lambda self: fake))
    return client


def test_delete_batches_stops_when_count_zero(monkeypatch) -> None:
    client = _client_with_fake_g(monkeypatch, counts=[0])
    # No vertices -> returns immediately without error.
    client.delete_vertices_in_batches("MyLabel", delay=0.0)


def test_delete_batches_progresses_to_zero(monkeypatch) -> None:
    client = _client_with_fake_g(monkeypatch, counts=[10, 5, 0])
    client.delete_vertices_in_batches("MyLabel", batch_size=5, delay=0.0)


def test_delete_batches_no_progress_raises(monkeypatch) -> None:
    # Count stays stuck at 7 -> guard aborts (wrapped as AWSServiceError).
    client = _client_with_fake_g(monkeypatch, counts=[7, 7])
    with pytest.raises(AWSServiceError, match="made no progress"):
        client.delete_vertices_in_batches("MyLabel", delay=0.0)


def test_delete_batches_count_increases_raises(monkeypatch) -> None:
    client = _client_with_fake_g(monkeypatch, counts=[5, 9])
    with pytest.raises(AWSServiceError, match="made no progress"):
        client.delete_vertices_in_batches("MyLabel", delay=0.0)


# --- get_graph_stats -----------------------------------------------------


def test_get_graph_stats_parses_counts(monkeypatch) -> None:
    client = _client(_config_with_neptune())

    class _StatsTraversal:
        def V(self):  # noqa: N802
            return self

        def E(self):  # noqa: N802
            return self

        def count(self):
            return self

        def label(self):
            return self

        def dedup(self):
            return self

        def next(self):
            return 42

        def toList(self):
            return ["A", "B"]

    fake = _StatsTraversal()
    monkeypatch.setattr(type(client), "g", property(lambda self: fake))
    stats = client.get_graph_stats()
    assert stats["vertex_count"] == 42
    assert stats["edge_count"] == 42
    assert stats["vertex_labels"] == ["A", "B"]
    assert stats["edge_labels"] == ["A", "B"]


# --- _handle_neptune_errors decorator ------------------------------------


def test_handle_neptune_errors_wraps_exceptions(monkeypatch) -> None:
    client = _client(_config_with_neptune())

    class _BoomTraversal:
        def V(self):  # noqa: N802
            raise RuntimeError("gremlin exploded")

    monkeypatch.setattr(type(client), "g", property(lambda self: _BoomTraversal()))
    with pytest.raises(AWSServiceError, match="get_graph_stats.*failed"):
        client.get_graph_stats()
