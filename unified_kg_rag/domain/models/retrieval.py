# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, Field, field_validator
from typing_extensions import TypedDict

from unified_kg_rag.domain.retrieval.section_types import SectionType as SectionType
from unified_kg_rag.domain.retrieval.source_fields import decode_list_fields

_SAFE_NAME = re.compile(r"[a-z0-9_-]+")
# One path segment: starts with a letter or digit, so "." and ".." (and hidden
# names) cannot occur, and no separator can appear.
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def validate_safe_name(value: str, field: str) -> str:
    """Return ``value`` if it is safe as an index suffix, path or key segment.

    Lowercase letters, digits, hyphens and underscores only: no ``/``, ``.``,
    wildcard or comma, so the value can neither leave its directory or S3
    prefix nor widen an OpenSearch index target.
    """
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError(
            f"Invalid {field} '{value}': only lowercase letters, digits, "
            "hyphens, and underscores are allowed."
        )
    return value


def validate_path_segment(value: str, field: str) -> str:
    """Return ``value`` if it is safe as one directory name or S3 key segment.

    Letters, digits, ``.``, ``_`` and ``-``, starting with a letter or digit
    and containing no ``..``: it cannot contain a separator or climb out of
    its parent directory or prefix.
    """
    if not _SAFE_SEGMENT.fullmatch(value) or ".." in value:
        raise ValueError(
            f"Invalid {field} '{value}': use letters, digits, '.', '_' or '-' "
            "(starting with a letter or digit, no '..'), up to 128 characters."
        )
    return value


class RetrieverRole(str, Enum):
    """Abstract retriever roles a search strategy depends on.

    Strategies request retrievers by ROLE (what it does), not by concrete
    backend product. The composition root binds each role to an adapter — GRAPH
    to a graph store (Neptune today), DOCUMENT to a vector/lexical store
    (OpenSearch today) — so a backend can be swapped without touching strategy
    code.
    """

    GRAPH = "graph"
    DOCUMENT = "document"


class SearchStrategy(str, Enum):
    AUTO = "auto"
    DRIFT = "drift"
    GLOBAL = "global"
    LOCAL = "local"
    SIMPLE = "simple"
    # LightRAG dual-level keyword methodology (high/low keywords over the shared
    # 3-store hybrid infrastructure).
    MIX = "mix"
    HYBRID = "hybrid"
    NAIVE = "naive"
    # Wraps local or mix (search.iterative_search.base): bounded follow-up
    # rounds on the bridge entities a fast-model controller names.
    ITERATIVE = "iterative"
    # HippoRAG 2 Personalized PageRank over entities and passages (opt-in;
    # not routable by auto until measured).
    PPR = "ppr"
    # RAPTOR's summary tree (the raptor methodology): text units and every
    # level of summaries, fused (search.tree_search).
    TREE = "tree"
    # A graph query the model plans as JSON and code compiles, checks and
    # runs read-only (search.structured; opt-in, not routable by default).
    STRUCTURED = "structured"


def _always_reports_sources(config: Any) -> bool:
    return True


def _global_reports_sources(config: Any) -> bool:
    # Map-reduce with evidence_tokens 0 answers from the key points alone
    # (model output, not sources).
    settings = config.search.global_search
    return not (settings.enable_map_reduce and settings.evidence_tokens == 0)


@dataclass(frozen=True)
class StrategyTraits:
    """What the chain, the router, the config checks and the evaluator need
    to know of a search strategy, so none of them keys it by name
    (:data:`STRATEGY_TRAITS`; every strategy has a row, a unit test checks).
    The router's text for a strategy is ``STRATEGY_ROUTING_GUIDE``."""

    # Why ``search.auto_routable_strategies`` refuses it; None: it may list
    # it.
    not_routable_reason: str | None = None
    # Its place among the general-purpose strategies a query falls back to
    # (``GENERAL_FALLBACKS``: auto's when routing fails or nothing configured
    # is served, structured's when its fallback is not served); None: not
    # one.
    general_fallback_rank: int | None = None
    # Whether a query under a config (``Config``) reports sources.
    reports_sources: Callable[[Any], bool] = _always_reports_sources

    @property
    def auto_routable(self) -> bool:
        return self.not_routable_reason is None


STRATEGY_TRAITS: dict[SearchStrategy, StrategyTraits] = {
    SearchStrategy.AUTO: StrategyTraits(not_routable_reason="it is the router itself"),
    SearchStrategy.LOCAL: StrategyTraits(general_fallback_rank=0),
    SearchStrategy.MIX: StrategyTraits(general_fallback_rank=1),
    SearchStrategy.GLOBAL: StrategyTraits(reports_sources=_global_reports_sources),
    SearchStrategy.DRIFT: StrategyTraits(),
    SearchStrategy.SIMPLE: StrategyTraits(),
    SearchStrategy.HYBRID: StrategyTraits(),
    SearchStrategy.NAIVE: StrategyTraits(),
    SearchStrategy.ITERATIVE: StrategyTraits(),
    SearchStrategy.PPR: StrategyTraits(
        not_routable_reason="it is opt-in until a measurement shows where it "
        "wins (docs/design.md §6.4); select it explicitly"
    ),
    SearchStrategy.TREE: StrategyTraits(
        not_routable_reason="it is opt-in (it answered better than naive but "
        "well below the graph strategies, docs/concepts/"
        "defaults-and-evidence.md#raptor-summary-tree), so the router does "
        "not pick it; select it explicitly"
    ),
    # Routable once listed (not by default: docs/design.md §6.6).
    SearchStrategy.STRUCTURED: StrategyTraits(),
}

# The general-purpose strategies a query falls back to, in order.
GENERAL_FALLBACKS: tuple[SearchStrategy, ...] = tuple(
    sorted(
        (s for s, t in STRATEGY_TRAITS.items() if t.general_fallback_rank is not None),
        key=lambda s: STRATEGY_TRAITS[s].general_fallback_rank or 0,
    )
)


class SearchType(str, Enum):
    HYBRID = "hybrid"
    LEXICAL = "lexical"
    VECTOR = "vector"


class RetrievalResult(BaseModel):
    content: str = Field(description="Retrieved content text")
    score: float = Field(description="Relevance score (0-1)")
    source: str | None = Field(default=None, description="Content source identifier")
    retriever_type: SectionType = Field(
        description="The kind of item retrieved (an unknown kind reads as " "general)"
    )
    chunk_id: str | None = Field(default=None, description="Source chunk identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional retrieval result metadata (its id lists, "
        "source_fields.LIST_FIELDS, are lists of strings)",
    )

    @field_validator("retriever_type", mode="before")
    @classmethod
    def _known_kind(cls, value: Any) -> SectionType:
        return SectionType.of(value)

    @field_validator("metadata")
    @classmethod
    def _decoded_lists(cls, metadata: dict[str, Any]) -> dict[str, Any]:
        # The one place a store's list encodings are undone (a Gremlin
        # one-element list read as its value, an edge's JSON list string).
        return decode_list_fields(metadata)


class Source(TypedDict, total=False):
    """One source of an answer (``RAGOutput.sources``): the text the answer
    model saw (``content``), the stored item's id (``source``) and score,
    and its metadata with its provenance (``source_id``, ``document_ids``,
    ``chunk_id``, ``section_type``, ``score``, ``truncated``, and
    ``provenance`` for seed-graph evidence). A plain ``dict`` at run time,
    so it serializes and compares as one; a key may be missing (the chain
    always sets all four), and its metadata's id lists
    (``source_fields.LIST_FIELDS``) are validated into lists of strings."""

    content: str
    source: str | None
    score: float | None
    metadata: Annotated[dict[str, Any], AfterValidator(decode_list_fields)]


# The largest top_k a served surface takes from its caller: the console's
# Ask request, and the most mcp.max_top_k may allow (a library or CLI query
# has no bound).
MAX_SERVED_TOP_K = 100


class SearchQuery(BaseModel):
    query: str = Field(description="Search query text to be processed")
    search_type: SearchType = Field(
        default=SearchType.HYBRID, description="Search strategy type to use"
    )
    top_k: int = Field(
        default=10, ge=1, description="Maximum number of results to return"
    )
    retrieval_multiplier: int = Field(
        default=1,
        ge=1,
        description="Multiplier for retrieval operations to increase search depth",
    )
    index_prefixes: str | list[str] | None = Field(
        default=None,
        description=(
            "The indexes to search, by their configured prefix "
            "(indexing.opensearch.*_index_prefix: text units, entities, "
            "relationships, community reports, claims...), not their names: "
            "the retriever names each for suffix (store_name, "
            "'<prefix>-<suffix>'), on OpenSearch and the embedded store "
            "alike. None: every index the retriever reads."
        ),
    )
    suffix: str | None = Field(
        default=None, description="Suffix for multi-tenant or versioned indices"
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Attribute filters with attr_ prefix for result filtering",
    )
    exclude: dict[str, Any] | None = Field(
        default=None,
        description="Filters of the same form whose matching hits are left out "
        "(a must_not clause)",
    )
    any_of: list[dict[str, Any]] | None = Field(
        default=None,
        description="Filter groups of the same form, one of which a hit must "
        "pass beside filters (e.g. an entity's incident edges: "
        "[{source_id: ids}, {target_id: ids}]); a store that cannot apply "
        "one refuses the query (InvalidFilterError)",
    )
    page_through: bool = Field(
        default=False,
        description="A text-less search that returns every hit up to top_k x "
        "retrieval_multiplier, read in pages of the index's "
        "max_result_window (search_after) and not capped at "
        "indexing.opensearch.max_query_size",
    )
    source_fields: list[str] | None = Field(
        default=None,
        description="The stored fields each hit returns ([] returns the ids "
        "only); None returns every field but the embeddings",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Retrieval context budget in tokens, capped at "
        "search.token_manager.max_context_tokens",
    )
    entity_focus: list[str] = Field(
        default_factory=list, description="Entities to focus search on"
    )
    optional_keywords: list[str] = Field(
        default_factory=list,
        description="Optional keywords to boost relevance (not required for matching)",
    )
    hl_keywords: list[str] = Field(
        default_factory=list,
        description="High-level keywords (themes/intent) for relationship retrieval (LightRAG)",
    )
    ll_keywords: list[str] = Field(
        default_factory=list,
        description="Low-level keywords (specific entities) for entity retrieval (LightRAG)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional search query metadata"
    )
    seed_weights: dict[str, float] | None = Field(
        default=None,
        description="Graph expansion: each seed id's weight, its share of the "
        "fetch budget proportional to it (seed_shares); None splits it evenly",
    )
    max_hops: int | None = Field(
        default=None,
        ge=1,
        description="Graph expansion depth; None is search.graph_expansion.max_hops "
        "(on every graph store)",
    )

    @field_validator("suffix")
    @classmethod
    def _validate_suffix(cls, value: str | None) -> str | None:
        """Reject unsafe suffixes on the READ path.

        The write side validates the suffix (BaseIndexer.get_suffix),
        but the query-path suffix flows verbatim into the OpenSearch index target
        (`f"{prefix}-{suffix}"`). An unvalidated value enables index-name
        injection / cross-tenant reads (e.g. "*", "other-tenant", or comma-joined
        aliases). Enforce the same lowercase alnum/hyphen/underscore charset.
        """
        if value is None:
            return None
        return validate_safe_name(value, "suffix")


class GraphEdge(BaseModel):
    """One stored relationship as a graph edge between two entity ids
    (``BaseGraphRAGRetriever.aneighbourhood_edges``)."""

    id: str = Field(description="The relationship's id")
    source_id: str = Field(description="Source entity id")
    target_id: str = Field(description="Target entity id")
    weight: float = Field(default=1.0, description="The stored relationship weight")
    source_kinds: list[str] | None = Field(
        default=None,
        description="The kinds of source stating it (None: documents only)",
    )


class RerankStatus(str, Enum):
    """What reranking did for one query (``SearchResult.metadata["rerank"]``).

    - ``applied``: the rerank model reordered the candidates.
    - ``disabled``: reranking is off (``providers.rerank.kind: none``).
    - ``unavailable``: reranking is on but the rerank model could not be built.
    - ``skipped``: nothing to rerank (no candidates, none of a rerank-eligible
      type, no query text).
    - ``failed``: the rerank call raised or returned nothing usable; the fused
      order was kept.
    """

    APPLIED = "applied"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"
    FAILED = "failed"

    @property
    def severity(self) -> int:
        """Rank for the worst-case summary over a query's rerank stages."""
        return _RERANK_SEVERITY[self]


_RERANK_SEVERITY = {
    RerankStatus.FAILED: 4,
    RerankStatus.UNAVAILABLE: 3,
    RerankStatus.APPLIED: 2,
    RerankStatus.SKIPPED: 1,
    RerankStatus.DISABLED: 0,
}


class RerankStage(BaseModel):
    """One rerank step of a query: its outcome, what it covered and why."""

    status: RerankStatus = Field(description="Outcome of this rerank step")
    scope: str = Field(
        description="Result types offered to the reranker: 'all' or e.g. 'text'"
    )
    reason: str | None = Field(
        default=None, description="Why the step was not applied, if it was not"
    )


class SearchResult(BaseModel):
    query: SearchQuery = Field(description="Original search query that was processed")
    results: list[RetrievalResult] = Field(description="Retrieved search results")
    total_results: int = Field(description="Total number of results found in search")
    search_strategy: str = Field(description="Search strategy that was actually used")
    processing_time: float = Field(description="Processing time in seconds")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional search result metadata"
    )
