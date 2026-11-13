# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MCP server configuration."""

from typing import Any, Literal

from pydantic import (
    Field,
    field_validator,
)

from ..retrieval import (
    MAX_SERVED_TOP_K,
    SearchStrategy,
    validate_safe_name,
)
from ._common import ConfigSection

# The tools run-mcp-server can serve (mcp.tools).
McpToolName = Literal[
    "kg_search", "kg_answer", "kg_get_entity", "kg_neighbors", "kg_describe_schema"
]
MCP_TOOLS: tuple[McpToolName, ...] = (
    "kg_search",
    "kg_answer",
    "kg_get_entity",
    "kg_neighbors",
    "kg_describe_schema",
)


class McpServerConfig(ConfigSection):
    """The read-only MCP server of ``run-mcp-server`` (the ``mcp`` extra).

    The namespace it serves is fixed when it starts (``--suffix``, else
    ``processing.document_parsing.index_value``); these settings bound what a
    tool call may ask. Guide: docs/guides/mcp.md
    """

    tools: list[McpToolName] = Field(
        default_factory=lambda: list(MCP_TOOLS),
        min_length=1,
        description=(
            "The tools the server lists, of kg_search, kg_answer, "
            "kg_get_entity, kg_neighbors and kg_describe_schema (all by "
            "default). Leave kg_answer out to make no answer-model call: the "
            "agent then answers from kg_search's records with its own model"
        ),
    )
    allowed_namespaces: list[str] = Field(
        default_factory=list,
        description=(
            "Further index namespaces (suffixes) a tool call may name in its "
            "namespace argument. Empty (the default): the tools take no "
            "namespace argument and serve only the namespace the server "
            "started with"
        ),
    )
    strategies: list[SearchStrategy] | None = Field(
        default=None,
        min_length=1,
        description=(
            "The search strategies kg_search and kg_answer may run (null: "
            "every strategy; one the namespace does not hold is refused "
            "either way). The default strategy is auto when listed, else the "
            "first one; auto then routes (and falls back) only among them, "
            "and is refused before any model call when the namespace serves "
            "none of them"
        ),
    )
    max_top_k: int = Field(
        default=20,
        ge=1,
        le=MAX_SERVED_TOP_K,
        description="Largest top_k (and kg_get_entity limit) a call may ask",
    )
    max_hops: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Largest max_hops of a kg_neighbors call",
    )
    max_neighbors: int = Field(
        default=50,
        ge=1,
        le=500,
        description=(
            "Largest number of neighbours (and of relationships) one "
            "kg_neighbors call returns"
        ),
    )
    max_query_chars: int = Field(
        default=2000,
        ge=1,
        description="Longest query (and entity name) a call may give, in characters",
    )
    max_text_chars: int = Field(
        default=2000,
        ge=100,
        description=(
            "Characters of text each returned record keeps (chunk text, a "
            "description); a longer one is cut and marked text_truncated"
        ),
    )
    max_response_chars: int = Field(
        default=60000,
        ge=1000,
        description=(
            "Characters of serialized records one response holds; past it the "
            "lowest-ranked records are left out (truncated, omitted). The MCP "
            "message carries the result twice (structured content and its "
            "JSON text) and JSON escaping lengthens non-ASCII text, so the "
            "message on the wire is up to about 2-4 times this"
        ),
    )
    uncited_records: int = Field(
        default=5,
        ge=1,
        le=100,
        description=(
            "Records a kg_answer with records=cited returns when its answer "
            "cites none (its strategy's citations are off, global and drift by "
            "default, or it wrote no [n] marker): the first ones, best first "
            "(records_kept: top); records=all returns every one"
        ),
    )
    max_concurrent_calls: int = Field(
        default=4,
        ge=1,
        description=(
            "Tool calls the server runs at once; further calls wait, and the "
            "wait counts toward the call's one deadline "
            "(search.query_timeout_seconds)"
        ),
    )

    @field_validator("allowed_namespaces")
    @classmethod
    def _safe_namespaces(cls, values: list[str]) -> list[str]:
        return [validate_safe_name(v, "mcp.allowed_namespaces") for v in values]

    @field_validator("tools", "strategies")
    @classmethod
    def _distinct(cls, values: list[Any] | None) -> list[Any] | None:
        if values is not None and len(set(values)) != len(values):
            raise ValueError("lists a value more than once")
        return values
