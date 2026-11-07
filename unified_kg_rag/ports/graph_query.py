# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph-query port: structured queries over one suffix's graph.

The ``structured`` search strategy (docs/design.md §6.6) asks the graph
store bound to the GRAPH role two things: what a label holds, as far as a
plan may name it (``adescribe``), and one store operation over a validated
plan (``aexecute``). ``NeptuneRetriever`` (Neptune and Gremlin Server) and
``SQLiteGraphRetriever`` implement it; a custom graph retriever that does
not makes the strategy fail at construction.

An implementation owns the scope and the safety of what it runs: it
derives the vertex label from ``suffix`` itself (the request names none),
compiles the plan to a read-only form whose every vertex step stays in that
label, checks the compiled form before sending it, and bounds the query by
``GraphQueryRequest.timeout_seconds``. A store refusal or a time limit of
this query raises ``GraphQueryExecutionError`` (the strategy falls back); a
store that cannot be used raises what every retriever raises then
(``StoreConnectionError``, a configuration error), which fails the query.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from unified_kg_rag.domain.models.graph_query import (
    GraphProfile,
    GraphQueryRequest,
    GraphQueryResult,
)


@runtime_checkable
class GraphQueryPort(Protocol):
    """Structured queries over the graph of one suffix."""

    async def adescribe(
        self,
        suffix: str | None,
        *,
        sample_vertices: int,
        edges_per_vertex: int,
        max_examples: int,
    ) -> GraphProfile:
        """The profile of ``suffix``'s entity label: built from at most
        ``sample_vertices`` vertices and ``edges_per_vertex`` out-edges of
        each (``domain.retrieval.graph_query.build_profile``), with at most
        ``max_examples`` sampled values of a string property."""
        ...

    async def aexecute(self, request: GraphQueryRequest) -> GraphQueryResult:
        """The store operation ``request`` asks for, over the entity label
        of ``request.suffix``, with the ``hydrate`` properties of every
        vertex the result names.

        Raises:
            GraphQueryExecutionError: The store refused or timed out this
                query (non-fatal).
            GraphQueryCompileError: The compiled form failed the static
                checker (a bug; nothing was sent).
        """
        ...
