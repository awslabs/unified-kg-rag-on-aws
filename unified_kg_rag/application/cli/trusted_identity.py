# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Who asked: the caller identity an authenticating proxy sets in a header.

Off by default. ``run-mcp-server --transport streamable-http`` and
``run-console`` take ``--trusted-identity-header NAME`` (``build_mcp_app``
and ``ConsoleOptions`` the same setting): every HTTP request must then
carry exactly one non-blank ``NAME`` header of at most
``MAX_IDENTITY_CHARS`` characters, which the request's queries take as
``RAGInput.user_id`` (their spans' ``user.id``, their audit record's
``user_id``). A request without it (or with a blank one), with two (two
field lines, or one holding a comma: RFC 9110 lets a proxy fold two copies
into one comma-joined line), or with a value that is not UTF-8, too long or
holds a control character is refused (401, naming which of the three
:class:`IdentityRefusal` it is) and logged by the header's name and the
path, never its value: a request that reaches the server without the
header did not come through the proxy, or the proxy is misconfigured.

The header is trusted as it arrives: only a proxy that authenticates every
request and replaces any copy of the header the client sent (removes or
overwrites it, never appends to it) makes it an identity. Reached any other
way, a client names whom it likes.
"""

from __future__ import annotations

import enum
import json
import re
from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from contextvars import ContextVar
from typing import Any

from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.query_stop import blank_in_shared_calls

__all__ = [
    "MAX_IDENTITY_CHARS",
    "IdentityRefusal",
    "TrustedIdentity",
    "caller_user_id",
    "header_identity",
    "header_name_argument",
    "unauthorized",
    "valid_header_name",
]

logger = get_logger(__name__)

# The longest identity taken: OpenID Connect's limit on a subject (``sub``
# "MUST NOT exceed 255 ASCII characters", OpenID Connect Core 1.0 §2),
# the identity an OIDC-authenticating proxy forwards (an Application Load
# Balancer's ``x-amzn-oidc-identity``).
MAX_IDENTITY_CHARS = 255

# An HTTP field name (RFC 9110 §5.1: a token).
_FIELD_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_CALLER: ContextVar[str | None] = ContextVar("kg_rag_caller_identity", default=None)
# One request's caller: a call several queries share is no one caller's.
blank_in_shared_calls(_CALLER)


def caller_user_id() -> str | None:
    """The identity of the request being served (``None``: no trusted
    header is configured, or no request is being served)."""
    return _CALLER.get()


def valid_header_name(name: str) -> str:
    """``name`` if it is an HTTP field name, else ``ValueError``."""
    if not _FIELD_NAME.fullmatch(name or ""):
        raise ValueError(f"'{name}' is not an HTTP header name")
    return name


def header_name_argument(value: str) -> str:
    """An ``argparse`` type: :func:`valid_header_name`'s, as a usage error."""
    import argparse

    try:
        return valid_header_name(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


class IdentityRefusal(enum.Enum):
    """Why a request has no identity (:func:`header_identity`); the value is
    what the 401 says of the header (it never echoes the header's value)."""

    MISSING = "is missing or blank"
    REPEATED = (
        "was sent more than once (two lines, or one with a comma): the proxy "
        "must replace a client's copy, not append to it"
    )
    MALFORMED = (
        f"is malformed (not UTF-8, over {MAX_IDENTITY_CHARS} characters, or a "
        "control character)"
    )


def header_identity(
    headers: Iterable[tuple[bytes, bytes]], name: str
) -> str | IdentityRefusal:
    """The identity the ``name`` header of an ASGI request's ``headers``
    gives: its one value, stripped; else why there is none: missing or
    blank, repeated (two field lines, or a comma, which no subject an
    authenticating proxy forwards holds: a proxy that appended to a
    client's copy) or malformed (not UTF-8, longer than
    ``MAX_IDENTITY_CHARS``, or a control character)."""
    wanted = name.lower().encode("latin-1")
    values = [value for key, value in headers if key.lower() == wanted]
    if len(values) > 1:
        return IdentityRefusal.REPEATED
    if not values or not values[0].strip():
        return IdentityRefusal.MISSING
    try:
        identity = values[0].decode("utf-8").strip()
    except UnicodeDecodeError:
        return IdentityRefusal.MALFORMED
    if "," in identity:
        return IdentityRefusal.REPEATED
    if len(identity) > MAX_IDENTITY_CHARS or not identity.isprintable():
        return IdentityRefusal.MALFORMED
    return identity


class TrustedIdentity:
    """ASGI middleware: each HTTP request served under the identity its
    ``header`` gives (:func:`caller_user_id`), or refused without one by
    the ASGI app ``refuse(why)`` returns (answering 401; ``why``: what is
    wrong with the header, as the response may say it) (module docstring).
    Other scopes (the lifespan) pass as they are."""

    def __init__(
        self, app: ASGIApp, *, header: str, refuse: Callable[[str], ASGIApp]
    ) -> None:
        self.app = app
        self.header = valid_header_name(header)
        self.refuse = refuse

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        identity = header_identity(scope.get("headers") or (), self.header)
        if isinstance(identity, IdentityRefusal):
            why = f"the {self.header} header (the trusted identity header) {identity.value}"
            # The path has no query string (ASGI keeps it apart).
            logger.warning(
                "Refused a request to %s: %s (is the request coming through "
                "the authenticating proxy?)",
                scope.get("path", ""),
                why,
            )
            await self.refuse(why)(scope, receive, send)
            return
        token = _CALLER.set(identity)
        try:
            await self.app(scope, receive, send)
        finally:
            _CALLER.reset(token)


def unauthorized(why: str) -> ASGIApp:
    """A bare 401 with a JSON body naming ``why`` (the MCP server's
    refusal)."""
    body = json.dumps({"error": f"unauthorized: {why}"}).encode()

    async def refuse(scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return refuse
