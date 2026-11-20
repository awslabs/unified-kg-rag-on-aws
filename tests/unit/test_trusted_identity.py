# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The opt-in trusted identity header (``--trusted-identity-header``): off,
a client's header is ignored; on, the proxy's header is the caller, and a
request without exactly one valid copy is refused (401)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.console_demo import requires_console
from tests.fixtures.extras import requires_mcp
from unified_kg_rag.application.cli.trusted_identity import (
    MAX_IDENTITY_CHARS,
    TrustedIdentity,
    caller_user_id,
    header_identity,
    unauthorized,
    valid_header_name,
)

pytestmark = pytest.mark.unit

_HEADER = "x-amzn-oidc-identity"


@pytest.mark.parametrize(
    ("headers", "identity"),
    [
        ([(b"x-amzn-oidc-identity", b"user-42")], "user-42"),
        # Header names are case-insensitive; the value is stripped.
        ([(b"X-Amzn-Oidc-Identity", b"  user-42 ")], "user-42"),
        ([], None),
        ([(b"x-amzn-oidc-identity", b"   ")], None),
        # Two copies: a client's that the proxy appended to, not replaced.
        ([(b"x-amzn-oidc-identity", b"a"), (b"x-amzn-oidc-identity", b"b")], None),
        ([(b"x-amzn-oidc-identity", b"u" * (MAX_IDENTITY_CHARS + 1))], None),
        ([(b"x-amzn-oidc-identity", b"u" * MAX_IDENTITY_CHARS)], "u" * 255),
        ([(b"x-amzn-oidc-identity", b"user\x1b[2J")], None),
        ([(b"other", b"user-42")], None),
    ],
)
def test_the_header_gives_one_bounded_identity(
    headers: list[tuple[bytes, bytes]], identity: str | None
) -> None:
    assert header_identity(headers, _HEADER) == identity


def test_a_header_name_is_an_http_field_name() -> None:
    assert valid_header_name(_HEADER) == _HEADER
    for bad in ("", "x amzn", "x-id:", "ä"):
        with pytest.raises(ValueError, match="HTTP header name"):
            valid_header_name(bad)


async def _call(app: Any, headers: list[tuple[bytes, bytes]]) -> tuple[int, Any]:
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b""}

    async def send(message: Any) -> None:
        sent.append(message)

    await app(
        {"type": "http", "path": "/mcp", "headers": headers, "method": "POST"},
        receive,
        send,
    )
    return sent[0]["status"], sent


async def test_on_the_request_is_served_as_its_caller_and_refused_without_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    seen: list[str | None] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        # A task the request starts sees it too (the SDK's server task).
        seen.append(await asyncio.create_task(_identity()))
        await send({"type": "http.response.start", "status": 200, "headers": []})

    guarded = TrustedIdentity(app, header=_HEADER, refuse=unauthorized)
    status, _ = await _call(guarded, [(b"x-amzn-oidc-identity", b"user-42")])
    assert (status, seen) == (200, ["user-42"])
    assert caller_user_id() is None  # reset after the request
    status, sent = await _call(guarded, [])
    assert status == 401 and seen == ["user-42"]  # the app never ran
    assert b"trusted identity header" in sent[1]["body"]
    assert "x-amzn-oidc-identity" in caplog.text
    # The lifespan passes.
    lifespan: list[str] = []

    async def lifespan_app(scope: Any, receive: Any, send: Any) -> None:
        lifespan.append(scope["type"])

    await TrustedIdentity(lifespan_app, header=_HEADER, refuse=unauthorized)(
        {"type": "lifespan"}, None, None  # type: ignore[arg-type]
    )
    assert lifespan == ["lifespan"]


async def _identity() -> str | None:
    return caller_user_id()


@requires_mcp
def test_the_mcp_cli_takes_the_header_for_http_only() -> None:
    from unified_kg_rag.application.cli.run_mcp_server import parse_args

    args = parse_args(
        ["--transport", "streamable-http", "--trusted-identity-header", _HEADER]
    )
    assert args.trusted_identity_header == _HEADER
    assert parse_args(["--transport", "streamable-http"]).trusted_identity_header is (
        None
    )
    with pytest.raises(SystemExit):
        parse_args(["--trusted-identity-header", _HEADER])  # stdio
    with pytest.raises(SystemExit):
        parse_args(
            ["--transport", "streamable-http", "--trusted-identity-header", "a b"]
        )


@requires_console
@pytest.mark.parametrize("trusted", [False, True], ids=["off", "on"])
def test_the_console_s_questions_take_the_caller_only_when_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trusted: bool
) -> None:
    from fastapi.testclient import TestClient

    from unified_kg_rag.application.cli.console.server import ConsoleOptions, create_app

    options = ConsoleOptions(
        workspace=tmp_path,
        cache_directory=tmp_path / "cache",
        trusted_identity_header=_HEADER if trusted else None,
    )
    client = TestClient(create_app(options), base_url="http://127.0.0.1")

    async def ask(_request: Any) -> AsyncIterator[dict[str, Any]]:
        # What the question's RAGInput takes (console.ask._invoke).
        yield {"kind": "identity", "user_id": caller_user_id()}

    monkeypatch.setattr(client.app.state.console.ask, "ask", ask)
    headers = {"X-Console-Request": "1", _HEADER: "user-42"}
    response = client.post("/api/ask", json={"query": "Who?"}, headers=headers)
    assert response.status_code == 200
    assert '"user_id": ' + ('"user-42"' if trusted else "null") in response.text
    missing = client.post(
        "/api/ask", json={"query": "Who?"}, headers={"X-Console-Request": "1"}
    )
    assert missing.status_code == (401 if trusted else 200)
    if trusted:
        assert missing.json()["code"] == "unauthorized"
        # Every request, the page's too.
        assert client.get("/api/status").status_code == 401
