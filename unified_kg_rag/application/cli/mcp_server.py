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

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

logger = get_logger(__name__)

SERVER_NAME = "unified-kg-rag"
MCP_EXTRA_HINT = (
    "The MCP server needs the mcp extra: install "
    "'unified-kg-rag-on-aws[mcp]' (uv sync --extra mcp)."
)

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
            "a generated answer) and return the sources the answer used, the "
            "model usage, and whether the answer abstained because the index "
            "does not cover the question. Costs model calls; prefer "
            "kg_search when you write the answer yourself."
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
        "kg_answer": (query_parameters, AnswerToolResult, answer),
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
