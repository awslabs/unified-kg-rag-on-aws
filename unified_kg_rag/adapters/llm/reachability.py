# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A model server that cannot be reached: told apart, named and found early.

A model client that cannot connect raises its transport's error (httpx's
``ConnectError``, the OpenAI client's ``APIConnectionError``, the Ollama
client's ``ConnectionError``, botocore's ``EndpointConnectionError``). Read
as any other failure it was retried for minutes and reported as "produced 0
outputs" or a traceback. :func:`model_unreachable_error` turns it into a
:class:`ModelUnreachableError` naming the provider kind and ``base_url`` of
the configured model servers (a bedrock one by its region's endpoint), with
how to start one (``ollama serve``, the models to pull) or reach it: at
ingestion's embedding probe, before any extraction, and for a query.
"""

from __future__ import annotations

import socket
from urllib.parse import urlsplit

from unified_kg_rag.domain.models import Config, ModelRole
from unified_kg_rag.domain.models.provider_config import BEDROCK, OLLAMA
from unified_kg_rag.shared.exceptions import (
    AWSServiceError,
    ModelUnreachableError,
    causes,
)

# Errors of a connection that was never made, by class name so this module
# imports no client: httpx's ConnectError and ConnectTimeout, the OpenAI
# client's APIConnectionError (not its APITimeoutError subclass: a server
# that accepted the request and did not answer is reachable) and botocore's
# EndpointConnectionError.
_CONNECT_ERROR_NAMES = frozenset(
    {"ConnectError", "ConnectTimeout", "APIConnectionError", "EndpointConnectionError"}
)
_REACHED_NAMES = frozenset({"APITimeoutError"})
# Ollama's default endpoint, which its client uses without a base_url.
_OLLAMA_DEFAULT = "http://localhost:11434"


def is_unreachable_error(exc: BaseException) -> bool:
    """Whether ``exc`` (or an error it was raised from, as an
    ``EmbeddingModelError`` wraps the client's) is a connection to a model
    server that could not be made: refused, no route, a name that does not
    resolve, no connection in time. Never a store's (``AWSServiceError``:
    ``StoreConnectionError``, which the store clients raise for theirs)."""
    for error in causes(exc):
        if isinstance(error, ModelUnreachableError):
            return True
        if isinstance(error, AWSServiceError):
            return False
        names = {cls.__name__ for cls in type(error).__mro__}
        if names & _REACHED_NAMES:
            return False
        if (
            names & _CONNECT_ERROR_NAMES
            # The Ollama client's own "Failed to connect" is a bare one.
            or type(error) is ConnectionError
            or isinstance(error, ConnectionRefusedError | socket.gaierror)
        ):
            return True
    return False


def bedrock_runtime_url(region: str) -> str:
    """The Bedrock runtime endpoint of ``region``: what a bedrock model
    server is reached at (its ``base_url`` is ignored)."""
    return f"https://bedrock-runtime.{region}.amazonaws.com"


def _servers(config: Config) -> list[tuple[str, str, str | None]]:
    """``(settings, kind, url)`` of the language and embedding model
    servers, one entry for the two when they are the same: a bedrock one
    at its region's runtime endpoint (a ``base_url`` left by the local
    preset is not where it is reached)."""
    found: dict[tuple[str, str | None], list[str]] = {}
    for setting, provider in (
        ("providers.llm", config.providers.llm),
        ("providers.embedding", config.providers.embedding),
    ):
        if provider.kind == BEDROCK:
            url: str | None = bedrock_runtime_url(config.aws.bedrock_region)
        else:
            url = provider.base_url or (
                _OLLAMA_DEFAULT if provider.kind == OLLAMA else None
            )
        found.setdefault((provider.kind, url), []).append(setting)
    return [(" and ".join(names), kind, url) for (kind, url), names in found.items()]


def _failed_endpoint(exc: BaseException) -> tuple[str, int | None] | None:
    """The host and port the failed connection of ``exc`` was made to, when
    its transport says (httpx's and the OpenAI client's request, botocore's
    ``endpoint_url``); ``None`` when none does (the Ollama client's own)."""
    for error in causes(exc):
        url: object = None
        try:
            url = getattr(getattr(error, "request", None), "url", None)
        except RuntimeError:  # httpx: a request error raised without one
            url = None
        url = url or (getattr(error, "kwargs", None) or {}).get("endpoint_url")
        if url:
            return _endpoint(str(url))
    return None


def _url_of(endpoint: tuple[str, int | None]) -> str:
    host, port = endpoint
    return f"https://{host}" if port in (None, 443) else f"https://{host}:{port}"


def _endpoint(url: str) -> tuple[str, int | None]:
    parts = urlsplit(url)
    default = {"http": 80, "https": 443}.get(parts.scheme)
    return (parts.hostname or "").lower(), parts.port or default


def _hint(config: Config, servers: list[tuple[str, str, str | None]]) -> str:
    if any(kind == OLLAMA for _, kind, _ in servers):
        # The models of the servers named: every role's but the evaluation
        # judge's for the language model server, the embedder for its own.
        settings = " and ".join(
            setting for setting, kind, _ in servers if kind == OLLAMA
        )
        roles = (
            [r for r in ModelRole if r is not ModelRole.EVALUATION]
            if "providers.llm" in settings
            else []
        )
        embedding = (
            [str(config.embedding_model_id)]
            if "providers.embedding" in settings
            else []
        )
        models = ", ".join(
            dict.fromkeys([*(config.model_id(r) for r in roles), *embedding])
        )
        return (
            "Start it (ollama serve) and pull the models "
            f"(ollama pull, for each of: {models}), or set its base_url."
        )
    if all(kind == BEDROCK for _, kind, _ in servers):
        return (
            "Check aws.bedrock_region (a region Bedrock is offered in) and the "
            "network path to Bedrock (a VPC endpoint, a proxy)."
        )
    return "Start the server, or set its base_url."


def model_unreachable_error(
    exc: BaseException, config: Config
) -> ModelUnreachableError:
    """``exc``, a model server that could not be reached
    (:func:`is_unreachable_error`), as a :class:`ModelUnreachableError` that
    names the configured server it failed to reach (every one when its
    transport does not say which) and how to start it."""
    if isinstance(exc, ModelUnreachableError):
        return exc
    servers = _servers(config)
    if (failed := _failed_endpoint(exc)) is not None:
        servers = (
            [
                server
                for server in servers
                if server[2] is not None and _endpoint(server[2]) == failed
            ]
            or [
                # Another Bedrock endpoint (the control plane resolving a
                # profile, an AWS_ENDPOINT_URL override): the one that failed.
                (setting, kind, _url_of(failed))
                for setting, kind, _ in servers
                if kind == BEDROCK
            ]
            or servers
        )
    named = ", ".join(
        f"{setting} ({kind}{f' at {url}' if url else ''})"
        for setting, kind, url in servers
    )
    return ModelUnreachableError(
        f"A model server could not be reached ({type(exc).__name__}): {named}. "
        f"{_hint(config, servers)}"
    )
