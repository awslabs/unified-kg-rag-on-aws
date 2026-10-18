# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

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


class FusionMethod(str, Enum):
    RRF = "rrf"
    WEIGHTED = "weighted"


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


class SearchType(str, Enum):
    HYBRID = "hybrid"
    LEXICAL = "lexical"
    VECTOR = "vector"


class RetrievalResult(BaseModel):
    content: str = Field(description="Retrieved content text")
    score: float = Field(description="Relevance score (0-1)")
    source: str | None = Field(default=None, description="Content source identifier")
    retriever_type: str = Field(
        description="Type of retriever that generated this result"
    )
    chunk_id: str | None = Field(default=None, description="Source chunk identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional retrieval result metadata"
    )


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
        description="Target OpenSearch index aliases (text_units, entities, community_reports)",
    )
    suffix: str | None = Field(
        default=None, description="Suffix for multi-tenant or versioned indices"
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Attribute filters with attr_ prefix for result filtering",
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


class RerankStatus(str, Enum):
    """What reranking did for one query (``SearchResult.metadata["rerank"]``).

    - ``applied``: the rerank model reordered the candidates.
    - ``disabled``: reranking is off (``search.reranking.enabled: false`` or
      ``providers.rerank.kind: none``).
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
