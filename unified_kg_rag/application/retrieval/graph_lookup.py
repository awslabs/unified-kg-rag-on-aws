# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only lookups of an index namespace's graph, for the MCP tools
(docs/design.md#84-agent-access-the-read-only-mcp-server).

An entity by id, IRI or name, the neighbourhood of one, the file names of
text units and what a namespace holds. They read the stores through the
chain's own retrievers (``GraphRAGChain._get_retriever``), with the read
paths the search strategies use: a text-less fetch by id on an index of
the document store and the graph walk of the graph store, so every backend
that serves queries serves these. Nothing here calls a model or writes.
The functions take the chain they read for, like ``methodology_check``.
"""

from __future__ import annotations

import asyncio
import math
import unicodedata
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from unified_kg_rag.adapters.retrieval.base import BaseGraphRAGRetriever
from unified_kg_rag.application.bootstrap import resolve_extraction_schema
from unified_kg_rag.application.retrieval import methodology_check
from unified_kg_rag.domain.models import (
    GraphEdge,
    IndexArtifact,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchType,
)
from unified_kg_rag.domain.models.methodology import methodologies_holding
from unified_kg_rag.domain.models.schema import ExtractionSchema
from unified_kg_rag.domain.retrieval import source_fields
from unified_kg_rag.domain.retrieval.linking import exact_links
from unified_kg_rag.domain.retrieval.store_names import resolve_namespace
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import MethodologyNotIndexedError
from unified_kg_rag.shared.utils import safe_float_parse

if TYPE_CHECKING:
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain

logger = get_logger(__name__)

# How an entity lookup matched: by its id, its IRI, its exact name, or (no
# exact match) a lexical search of the names.
EntityMatch = Literal["id", "iri", "name", "search", "none"]

# The artifacts a graph lookup reads: the entities index and the graph.
GRAPH_ARTIFACTS = frozenset({IndexArtifact.ENTITIES, IndexArtifact.GRAPH})


async def require_graph(chain: GraphRAGChain, suffix: str | None) -> None:
    """Raise ``MethodologyNotIndexedError`` if the namespace of ``suffix``
    holds no graph (a ``raptor``-only build): its entities index and graph
    store are what the graph lookups read."""
    artifacts = await methodology_check.namespace_artifacts(chain, suffix)
    if GRAPH_ARTIFACTS <= artifacts:
        return
    held = ", ".join(m.value for m in methodologies_holding(artifacts))
    raise MethodologyNotIndexedError(
        f"The index namespace '{resolve_namespace(chain.config, suffix)}' holds no "
        f"knowledge graph (it holds {held or 'no methodology'}); the graph "
        "tools need a namespace built with graphrag, lightrag or hipporag."
    )


def _document_retriever(chain: GraphRAGChain) -> BaseGraphRAGRetriever:
    return chain._get_retriever(RetrieverRole.DOCUMENT)


def _fetch(
    prefix: str, suffix: str | None, filters: dict[str, Any], top_k: int
) -> SearchQuery:
    """A text-less fetch on ``prefix``'s index: the hits ``filters`` pass,
    ordered by the store (``rank`` and ``weight`` first, where held)."""
    return SearchQuery(
        query="",
        search_type=SearchType.LEXICAL,
        top_k=top_k,
        index_prefixes=[prefix],
        suffix=suffix,
        filters=filters,
    )


async def find_entities(
    chain: GraphRAGChain,
    suffix: str | None,
    *,
    entity_id: str | None = None,
    iri: str | None = None,
    name: str | None = None,
    limit: int,
) -> tuple[EntityMatch, list[RetrievalResult]]:
    """The entities of the namespace with ``entity_id``, ``iri`` or
    ``name`` (give one), at most ``limit``, and how they matched.

    An id or IRI matches exactly. A name matches the stored name exactly
    (``name.keyword``) first; without an exact match, a lexical search of
    the names with the index's analyzer (the configured language analyzer:
    ``nori``, ``cjk``, ...) returns the best-scored ones (``search``). No
    embedding is computed: a name lookup costs no model call.

    The exact lookups read the value in NFC, as indexed text is (and as
    ``RAGInput`` composes a query): decomposed Hangul matches its composed
    name. Without an exact match, a name is looked up again in NFKC, and
    searched in NFKC, so a full-width ``Ｄｅｐｏｔ`` or a compatibility
    character finds the name it stands for; a searched hit that is the name
    as the strategies link one (``linking.exact_links``: an Arabic alef
    without its hamza, ``ё`` as ``е``, a leading ``شركة``) is a ``name``
    match.
    """
    suffix = resolve_namespace(chain.config, suffix)
    entity_id, iri, name = (
        unicodedata.normalize("NFC", v) if v else v for v in (entity_id, iri, name)
    )
    given = [v for v in (entity_id, iri, name) if v]
    if len(given) != 1:
        raise ValueError("give exactly one of entity_id, iri and name")
    await require_graph(chain, suffix)
    retriever = _document_retriever(chain)
    prefix = chain.config.indexing.opensearch.entities_index_prefix
    match: EntityMatch
    value: str | None
    if entity_id:
        key, value, match = "id", entity_id, "id"
    elif iri:
        key, value, match = "iri", iri, "iri"
    else:
        key, value, match = "name.keyword", name, "name"
    hits = await retriever.aretrieve(_fetch(prefix, suffix, {key: value}, limit))
    if hits:
        return match, _ranked(hits)[:limit]
    if name is None:
        return "none", []
    folded = unicodedata.normalize("NFKC", name)
    if folded != name:
        # A full-width or compatibility spelling of a stored name.
        hits = await retriever.aretrieve(
            _fetch(prefix, suffix, {"name.keyword": folded}, limit)
        )
        if hits:
            return "name", _ranked(hits)[:limit]
    hits = await retriever.aretrieve(
        SearchQuery(
            query=folded,
            search_type=SearchType.LEXICAL,
            top_k=limit,
            index_prefixes=[prefix],
            suffix=suffix,
        )
    )
    # A hit that is the name as a query's name links (linking.exact_links:
    # "أكمي" for "اكمي", "Федоров" for "Фёдоров").
    exact = exact_links(name, hits)
    if exact:
        return "name", _ranked(exact)[:limit]
    return ("search" if hits else "none"), hits[:limit]


def _ranked(hits: list[RetrievalResult]) -> list[RetrievalResult]:
    """Exact matches best first: by stored ``rank``, then by id."""
    return sorted(
        hits,
        key=lambda r: (
            -(safe_float_parse((r.metadata or {}).get("rank")) or 0.0),
            str(r.source),
        ),
    )


def hops_of(score: float) -> int:
    """The hops between a graph-walk entity and its seed, from its score.

    ``graph_relevance`` scores a vertex ``n`` hops out (path length
    ``L = n + 1``) in the upper half of the band ``(1 / (L + 1), 1 / L]``,
    so ``1 / score`` lies in ``[L, L + 0.5)``.
    """
    if score <= 0:
        return 0
    return max(0, math.floor(1.0 / score + 1e-9) - 1)


class Neighbourhood(BaseModel):
    """The entities a graph walk reached from one entity, and the
    relationships among them."""

    neighbours: list[RetrievalResult] = Field(
        description="The entities reached (not the start), nearest first"
    )
    hops: dict[str, int] = Field(
        description="Each neighbour's id -> its distance from the start"
    )
    relationships: list[RetrievalResult] = Field(
        default_factory=list,
        description="The stored relationships among the start and the "
        "neighbours, from the relationships index (with descriptions and "
        "text units), best (rank, weight) first",
    )
    edges: list[GraphEdge] = Field(
        default_factory=list,
        description="Without the relationships index: the graph's edges among "
        "them (ids and weights only)",
    )
    neighbours_capped: bool = Field(
        default=False, description="More neighbours were reached than returned"
    )
    relationships_capped: bool = Field(
        default=False,
        description="More relationships were stored among them than returned",
    )


async def neighbourhood(
    chain: GraphRAGChain,
    suffix: str | None,
    entity_id: str,
    *,
    max_hops: int,
    limit: int,
) -> Neighbourhood:
    """The entities within ``max_hops`` of ``entity_id`` (at most ``limit``,
    nearest and most important first, each hop at most
    ``search.graph_expansion.max_results_per_hop`` per vertex, as every graph
    expansion) and the relationships among them and the start (at most
    ``limit``).

    The relationships come from the relationships index when the namespace
    holds it (``relationship_vectors``: their descriptions, text units and
    IRIs), else from the graph store (``aneighbourhood_edges``).
    """
    suffix = resolve_namespace(chain.config, suffix)
    await require_graph(chain, suffix)
    graph = chain._get_retriever(RetrieverRole.GRAPH)
    walked = await graph.aretrieve(
        SearchQuery(
            query="",
            # One more than kept: whether the walk reached more.
            top_k=limit + 1,
            suffix=suffix,
            filters={"id": [entity_id]},
            max_hops=max_hops,
        )
    )
    others = [r for r in walked if str(r.source) != entity_id]
    kept = others[:limit]
    hops = {str(r.source): max(1, hops_of(r.score)) for r in kept}
    ids = [entity_id, *hops]
    out = Neighbourhood(
        neighbours=kept, hops=hops, neighbours_capped=len(others) > limit
    )
    if len(ids) < 2:
        return out
    artifacts = await methodology_check.namespace_artifacts(chain, suffix)
    if IndexArtifact.RELATIONSHIP_VECTORS in artifacts:
        prefix = chain.config.indexing.opensearch.relationships_index_prefix
        hits = await _document_retriever(chain).aretrieve(
            _fetch(prefix, suffix, {"source_id": ids, "target_id": ids}, limit + 1)
        )
        ordered = sorted(
            hits,
            key=lambda r: (
                -(safe_float_parse((r.metadata or {}).get("rank")) or 0.0),
                -(safe_float_parse((r.metadata or {}).get("weight")) or 0.0),
                str((r.metadata or {}).get("id") or r.source),
            ),
        )
        out.relationships = ordered[:limit]
        out.relationships_capped = len(ordered) > limit
        return out
    edges = await graph.aneighbourhood_edges(
        ids, SearchQuery(query="", suffix=suffix), per_vertex=limit
    )
    out.edges = (edges or [])[:limit]
    out.relationships_capped = len(edges or []) > limit
    return out


async def text_unit_file_names(
    chain: GraphRAGChain, suffix: str | None, unit_ids: list[str]
) -> dict[str, str]:
    """``{text_unit_id: file name}`` of ``unit_ids``' text units (the name,
    never the path), in fetches by id of at most ``id_batch_size`` ids."""
    suffix = resolve_namespace(chain.config, suffix)
    wanted = list(dict.fromkeys(str(i) for i in unit_ids if i))
    if not wanted:
        return {}
    retriever = _document_retriever(chain)
    prefix = chain.config.indexing.opensearch.text_units_index_prefix
    step = (await retriever.aindex_limits(prefix, suffix)).id_batch_size
    names: dict[str, str] = {}
    for start in range(0, len(wanted), step):
        batch = wanted[start : start + step]
        query = _fetch(prefix, suffix, {"id": batch}, len(batch))
        for hit in await retriever.aretrieve(query):
            metadata = hit.metadata or {}
            if name := source_fields.file_name(metadata):
                names[str(metadata.get("id") or hit.source)] = name
    return names


class NamespaceDescription(methodology_check.NamespaceSummary):
    """What one index namespace holds and how it can be searched."""

    namespace: str = Field(description="The namespace (index suffix)")
    build: dict[str, Any] | None = Field(
        default=None,
        description="The last ingestion run's record (pipeline_id, mode, "
        "built_at, documents, package_version); None when none was recorded",
    )
    schema_: ExtractionSchema | None = Field(
        default=None,
        alias="schema",
        description="The extraction schema the configuration resolves to; "
        "None when its source cannot be read",
    )
    schema_error: str | None = Field(
        default=None, description="Why the schema could not be read"
    )
    filter_keys: list[str] = Field(
        default_factory=list,
        description="The filter keys some index of the document store declares",
    )


_BUILD_KEYS = ("pipeline_id", "mode", "built_at", "documents", "package_version")


async def resolve_schema(
    chain: GraphRAGChain,
) -> tuple[ExtractionSchema | None, str | None]:
    """The extraction schema the chain's configuration resolves to, or
    ``None`` and the error's type when its source cannot be read. A schema
    source may read a file or an S3 object: it runs off the loop, and a
    caller keeps the result rather than resolving it again."""
    try:
        resolved = await asyncio.to_thread(
            resolve_extraction_schema, chain.config, providers=chain.providers
        )
        return resolved.processing.graph_extraction.extraction_schema(), None
    except Exception as e:  # noqa: BLE001 - reported, not fatal
        logger.warning("Could not resolve the extraction schema: %s", e)
        return None, type(e).__name__


async def describe_namespace(
    chain: GraphRAGChain,
    suffix: str | None,
    *,
    schema: tuple[ExtractionSchema | None, str | None] | None = None,
) -> NamespaceDescription:
    """What the namespace of ``suffix`` holds (its build record, else the
    configured methodologies), the strategies that serves, its last build
    and the extraction schema the configuration resolves to (``schema``,
    :func:`resolve_schema`'s result, when the caller holds it)."""
    suffix = resolve_namespace(chain.config, suffix)
    # Read fresh: the last build it reports is the current one.
    holdings = await methodology_check.read_holdings(chain, suffix, refresh=True)
    record = holdings.record
    resolved, error = schema if schema is not None else await resolve_schema(chain)
    fields = _document_retriever(chain).filter_fields()
    return NamespaceDescription(
        namespace=suffix,
        **methodology_check.summarize_namespace(holdings, chain.config),
        build=(
            {k: record[k] for k in _BUILD_KEYS if k in record}
            if isinstance(record, dict)
            else None
        ),
        schema=resolved,
        schema_error=error,
        filter_keys=fields.describe() if fields is not None else [],
    )
