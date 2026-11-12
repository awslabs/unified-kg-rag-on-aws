# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MCP binding of the read-only tools (the ``mcp`` extra).

:func:`build_mcp_server` registers :class:`~.mcp_tools.KnowledgeGraphTools`'
methods as tools of the official MCP SDK's ``MCPServer``, with input
schemas bounded by the ``mcp`` settings and read-only annotations
(docs/design.md#84-agent-access-the-read-only-mcp-server). The SDK is
imported when a server is built, so this module loads without the extra;
building one without it raises ``ConfigurationError`` naming the extra.

Annotations are evaluated here (no ``from __future__ import annotations``):
the tool signatures are built at run time from the settings, and the SDK
reads them as objects.
"""

import importlib.util
import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import BaseModel, Field

from unified_kg_rag import __version__
from unified_kg_rag.application.cli.mcp_tools import (
    UNTRUSTED_NOTE,
    AnswerRecords,
    AnswerToolResult,
    EntityToolResult,
    KnowledgeGraphTools,
    NeighborsToolResult,
    SchemaToolResult,
    SearchToolResult,
    describe_error,
)
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import SearchType
from unified_kg_rag.domain.models.config import McpServerConfig
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import ConfigurationError
from unified_kg_rag.shared.extras import extra_hint

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

logger = get_logger(__name__)

SERVER_NAME = "unified-kg-rag"
MCP_EXTRA_HINT = f"The MCP server needs {extra_hint('mcp')}."

_Scalar = str | int | float | bool
_Range = dict[Literal["gt", "gte", "lt", "lte"], str | int | float]
FilterValue = _Scalar | list[_Scalar] | _Range


def _sdk() -> tuple[Any, Any, Any]:
    """``MCPServer``, ``ToolAnnotations`` and ``ToolError`` of the SDK.

    Raises:
        ConfigurationError: The ``mcp`` extra is not installed.
    """
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError as e:
        raise ConfigurationError(MCP_EXTRA_HINT) from e
    return MCPServer, ToolAnnotations, ToolError


def _host_entry(host: str) -> str:
    """``host`` as a ``Host`` header names it, any port (``[::1]:*``)."""
    bare = host.strip("[]")
    return f"[{bare}]:*" if ":" in bare else f"{bare}:*"


def loopback_hosts(host: str) -> list[str]:
    """The ``Host`` values a server bound to the loopback ``host`` accepts:
    that address and the usual loopback names, any port."""
    return list(
        dict.fromkeys([_host_entry(host), "127.0.0.1:*", "localhost:*", "[::1]:*"])
    )


def _security(
    allowed_hosts: list[str] | None, allowed_origins: list[str] | None
) -> Any:
    """The SDK's ``TransportSecuritySettings``: the ``Host`` (and
    ``Origin``) checks on with ``allowed_hosts``, off without. Passed
    explicitly, since the SDK turns its own loopback-only check on for an
    unset host, which refuses every proxied request."""
    from mcp.server.transport_security import TransportSecuritySettings

    if allowed_hosts is None:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    origins = allowed_origins
    if origins is None:
        origins = [
            f"{scheme}://{entry}"
            for entry in allowed_hosts
            for scheme in ("http", "https")
        ]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(allowed_hosts),
        allowed_origins=list(origins),
    )


def _post_only(app: Any, path: str) -> Any:
    """``app`` answering 405 to anything but POST on ``path``: a stateless
    server has no session to delete and no stream to resume, and a GET
    would otherwise hold an event stream open for as long as the client
    keeps it."""

    async def guarded(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") == "http"
            and scope.get("path", "").rstrip("/") == path.rstrip("/")
            and scope.get("method") != "POST"
        ):
            await send(
                {
                    "type": "http.response.start",
                    "status": 405,
                    "headers": [(b"allow", b"POST"), (b"content-length", b"0")],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return
        await app(scope, receive, send)

    return guarded


def build_mcp_app(
    chain: GraphRAGChain | KnowledgeGraphTools,
    *,
    namespace: str | None = None,
    settings: McpServerConfig | None = None,
    allowed_hosts: list[str] | None = None,
    allowed_origins: list[str] | None = None,
    path: str = "/mcp",
) -> Any:
    """A stateless streamable-HTTP ASGI application of
    :func:`build_mcp_server`'s server, to mount in a host application
    behind an authenticating proxy.

    ``allowed_hosts`` (``Host`` values such as ``"mcp.example.com"`` or
    ``"127.0.0.1:*"``) turns the SDK's ``Host``/``Origin`` checks on, with
    ``allowed_origins`` (default: ``http://`` and ``https://`` of each
    host); without it they are off, as behind a proxy that forwards its own
    ``Host``. Only POST reaches the server (405 otherwise). Run the
    application's lifespan (any ASGI server does); the caller closes the
    chain.

    Raises:
        ConfigurationError: The ``mcp`` extra is not installed.
    """
    server = build_mcp_server(chain, namespace=namespace, settings=settings)
    app = server.streamable_http_app(
        streamable_http_path=path,
        stateless_http=True,
        json_response=True,
        transport_security=_security(allowed_hosts, allowed_origins),
    )
    return _post_only(app, path)


def mcp_extra_error() -> str | None:
    """The message naming the ``mcp`` extra when its SDK is not installed,
    else ``None`` (a CLI checks it before building a chain)."""
    try:
        missing = importlib.util.find_spec("mcp") is None
    except (ImportError, ValueError):
        missing = True
    return MCP_EXTRA_HINT if missing else None


def _parameter(name: str, annotation: Any, default: Any) -> inspect.Parameter:
    return inspect.Parameter(
        name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation
    )


def _tool_function(
    name: str,
    parameters: list[inspect.Parameter],
    returns: type[BaseModel],
    run: Callable[..., Awaitable[BaseModel]],
    tool_error: type[Exception],
) -> Callable[..., Awaitable[BaseModel]]:
    """An async function with the signature the SDK publishes as the
    tool's input schema, calling ``run``. A caller-correctable error keeps
    its message; any other is logged here with its traceback and reaches
    the caller as a generic message (``describe_error``)."""

    async def tool(**arguments: Any) -> BaseModel:
        try:
            return await run(**arguments)
        except Exception as e:
            message, expected = describe_error(e)
            if not expected:
                logger.error("MCP tool %s failed: %s", name, e, exc_info=True)
            raise tool_error(message) from None

    tool.__name__ = name
    tool.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters, return_annotation=returns
    )
    return tool


def _descriptions(tools: KnowledgeGraphTools) -> dict[str, str]:
    namespaces = ", ".join(tools.namespaces)
    return {
        "kg_search": (
            "Search the knowledge graph index for supporting content: ranked "
            "chunks of documents and graph items (entities, relationships, "
            "community reports), each with its chunk id, text, relevance "
            "score, source document and the entities it cites. Retrieval "
            "only, no answer. The query may be in any language."
        ),
        "kg_answer": (
            "Answer a question from the knowledge graph index (retrieval and "
            "a generated answer): the answer first, whether it abstained "
            "because the index does not cover the question, then the sources "
            "it cites (records: all for every source it saw, none for "
            "none), the steps it was given without (degraded: a partial "
            "answer) and the model usage. For a follow-up question, pass the "
            "conversation so far as history (the server keeps none). Costs "
            "model calls; prefer kg_search when you write the answer yourself."
        ),
        "kg_get_entity": (
            "Look up entities of the knowledge graph by id, IRI or name (give "
            "one). A name matches exactly first, else the closest names; "
            "'match' says which."
        ),
        "kg_neighbors": (
            "The entities within max_hops of one entity (by id, IRI or name) "
            "and the relationships among them, with the chunks and documents "
            "they come from."
        ),
        "kg_describe_schema": (
            "What the index namespace holds: its methodologies, the search "
            "strategies kg_search and kg_answer can run, its last build, the "
            "entity and relationship types of its extraction schema, the "
            f"filter keys a search may give and the server's limits. Namespaces: "
            f"{namespaces}."
        ),
    }


def build_mcp_server(
    chain: GraphRAGChain | KnowledgeGraphTools,
    *,
    namespace: str | None = None,
    settings: McpServerConfig | None = None,
) -> "MCPServer[Any]":
    """An MCP server serving ``chain``'s index read-only.

    Args:
        chain: The chain to query (its config's ``mcp`` section bounds the
            tools), or tools already built over one.
        namespace: The index suffix served (default: the config's
            ``processing.document_parsing.index_value``, else ``default``).
        settings: The ``mcp`` settings, if not the chain config's.

    Returns:
        The SDK's ``MCPServer``: ``run("stdio")`` serves one agent over
        stdio, ``streamable_http_app()`` an ASGI application to mount
        behind an authenticating proxy. The caller closes the chain.

    Raises:
        ConfigurationError: The ``mcp`` extra is not installed, or the
            settings are inconsistent (``mcp.strategies`` with ``auto``
            routing outside them).
    """
    server_class, annotations_class, tool_error = _sdk()
    tools = (
        chain
        if isinstance(chain, KnowledgeGraphTools)
        else KnowledgeGraphTools(chain, namespace=namespace, settings=settings)
    )
    limits = tools.settings
    server = server_class(
        SERVER_NAME,
        title="Unified KG RAG (read-only)",
        version=__version__,
        instructions=(
            "Read-only tools over a knowledge-graph RAG index (GraphRAG and "
            "LightRAG methodologies): search it, ask it, look up entities and "
            f"walk their neighbourhood. Namespace: {tools.namespace}. " + UNTRUSTED_NOTE
        ),
    )
    read_only = annotations_class(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )
    descriptions = _descriptions(tools)
    strategies = tuple(s.value for s in tools.strategies)
    namespace_parameter = (
        [
            _parameter(
                "namespace",
                Annotated[
                    Literal[tuple(tools.namespaces)] | None,
                    Field(
                        description=(
                            f"The index namespace to read (default "
                            f"{tools.namespace})"
                        )
                    ),
                ],
                None,
            )
        ]
        if len(tools.namespaces) > 1
        else []
    )
    query_parameters = [
        _parameter(
            "query",
            Annotated[
                str,
                Field(
                    min_length=1,
                    max_length=limits.max_query_chars,
                    description="The question or search text, in any language",
                ),
            ],
            inspect.Parameter.empty,
        ),
        _parameter(
            "strategy",
            Annotated[
                Literal[strategies] | None,
                Field(
                    description=(
                        "The search strategy (default "
                        f"{tools.default_strategy.value}; kg_describe_schema "
                        "lists those the namespace holds)"
                    )
                ),
            ],
            None,
        ),
        _parameter(
            "search_type",
            Annotated[
                Literal[tuple(t.value for t in SearchType)],
                Field(description="hybrid (BM25 and vector, fused), lexical or vector"),
            ],
            SearchType.HYBRID.value,
        ),
        _parameter(
            "top_k",
            Annotated[
                int,
                Field(ge=1, le=limits.max_top_k, description="Results to return"),
            ],
            min(10, limits.max_top_k),
        ),
        _parameter(
            "filters",
            Annotated[
                dict[str, FilterValue] | None,
                Field(
                    description=(
                        "Attribute filters (key: value, a list of values, or a "
                        "range of gt/gte/lt/lte); kg_describe_schema lists the "
                        "keys. They narrow the search only"
                    )
                ),
            ],
            None,
        ),
        *namespace_parameter,
    ]
    turn_chars = limits.max_response_chars

    class HistoryTurn(BaseModel):
        """One earlier message of the caller's conversation."""

        role: Literal["user", "assistant"] = Field(
            description="Who wrote it: the user, or the assistant"
        )
        content: str = Field(
            max_length=turn_chars,
            description="The message text (an earlier answer as returned)",
        )

    answer_parameters = [
        *query_parameters,
        _parameter(
            "history",
            Annotated[
                list[HistoryTurn] | None,
                Field(
                    max_length=tools.max_history_turns,
                    description=(
                        "The conversation so far, oldest turn first, to answer "
                        "a follow-up question from (the server keeps none); "
                        "screened by the input guardrail as the query is"
                    ),
                ),
            ],
            None,
        ),
        _parameter(
            "records",
            Annotated[
                AnswerRecords,
                Field(
                    description=(
                        "Which supporting records to return after the answer: "
                        "cited (the sections the answer cites; the first few "
                        "when it cites none), all (every source the answer "
                        "model saw and the model-written sections it may "
                        "cite) or none"
                    ),
                ),
            ],
            "cited",
        ),
    ]
    entity_parameters = [
        _parameter(
            key,
            Annotated[
                str | None,
                Field(max_length=limits.max_query_chars, description=text),
            ],
            None,
        )
        for key, text in (
            ("id", "The entity id (as other results give it)"),
            ("name", "The entity name, in any language"),
            ("iri", "The entity IRI (seed graphs, ontologies)"),
        )
    ]

    def bounded(name: str, high: int, default: int, text: str) -> inspect.Parameter:
        return _parameter(
            name,
            Annotated[int, Field(ge=1, le=high, description=text)],
            min(default, high),
        )

    async def search(**kw: Any) -> SearchToolResult:
        return await tools.search(**kw)

    async def answer(**kw: Any) -> AnswerToolResult:
        return await tools.answer(**kw)

    async def get_entity(**kw: Any) -> EntityToolResult:
        return await tools.get_entity(entity_id=kw.pop("id", None), **kw)

    async def neighbors(**kw: Any) -> NeighborsToolResult:
        return await tools.neighbors(entity_id=kw.pop("id", None), **kw)

    async def describe(**kw: Any) -> SchemaToolResult:
        return await tools.describe_schema(**kw)

    specs: dict[str, tuple[list[inspect.Parameter], type[BaseModel], Any]] = {
        "kg_search": (query_parameters, SearchToolResult, search),
        "kg_answer": (answer_parameters, AnswerToolResult, answer),
        "kg_get_entity": (
            [
                *entity_parameters,
                bounded("limit", limits.max_top_k, 5, "Entities to return"),
                *namespace_parameter,
            ],
            EntityToolResult,
            get_entity,
        ),
        "kg_neighbors": (
            [
                *entity_parameters,
                bounded("max_hops", limits.max_hops, 1, "Hops to walk"),
                bounded(
                    "limit", limits.max_neighbors, 20, "Neighbours (and relationships)"
                ),
                *namespace_parameter,
            ],
            NeighborsToolResult,
            neighbors,
        ),
        "kg_describe_schema": (namespace_parameter, SchemaToolResult, describe),
    }
    for name in limits.tools:
        parameters, returns, run = specs[name]
        server.add_tool(
            _tool_function(name, parameters, returns, run, tool_error),
            name=name,
            description=f"{descriptions[name]} {UNTRUSTED_NOTE}",
            annotations=read_only,
            structured_output=True,
        )
    return cast("MCPServer[Any]", server)
