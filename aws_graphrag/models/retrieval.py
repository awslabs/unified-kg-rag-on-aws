from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FusionMethod(str, Enum):
    HYBRID = "hybrid"
    RRF = "rrf"
    WEIGHTED = "weighted"


class SearchStrategy(str, Enum):
    AUTO = "auto"
    DRIFT = "drift"
    GLOBAL = "global"
    LOCAL = "local"
    SIMPLE = "simple"


class SearchType(str, Enum):
    HYBRID = "hybrid"
    LEXICAL = "lexical"
    VECTOR = "vector"


class ContextBuilderResult(BaseModel):
    context_text: str = Field(description="Built context string ready for use")
    sections: list[dict[str, Any]] = Field(description="Context sections with metadata")
    total_tokens: int = Field(description="Total token count of built context")
    sections_included: int = Field(description="Number of sections included in context")
    optimization_applied: bool = Field(
        description="Whether context optimization was applied"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context builder metadata",
    )


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
    top_k: int = Field(default=10, description="Maximum number of results to return")
    retrieval_multiplier: int = Field(
        default=1,
        description="Multiplier for retrieval operations to increase search depth",
    )
    label_prefixes: str | list[str] | None = Field(
        default=None,
        description="Target node types for Neptune search (entity, community)",
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
    max_tokens: int | None = Field(
        default=None, description="Maximum number of context tokens"
    )
    entity_focus: list[str] = Field(
        default_factory=list, description="Entities to focus search on"
    )
    optional_keywords: list[str] = Field(
        default_factory=list,
        description="Optional keywords to boost relevance (not required for matching)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional search query metadata"
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
