# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Vector-store port — the boundary for the embedding/lexical search backend.

Production adapter: OpenSearch (``aws_graphrag.storage.opensearch_indexer`` for
writes, ``aws_graphrag.retrieval.retrievers.opensearch_retriever`` for search).
It holds chunk, entity, and community-report vectors today; M3 adds a
relationship vector index for LightRAG-style global retrieval. M2 adds id-based
upsert/delete for incremental runs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aws_graphrag.models import (
        CommunityReport,
        Entity,
        Relationship,
        TextUnit,
    )
    from aws_graphrag.storage.base import IndexingStats


@runtime_checkable
class VectorStorePort(Protocol):
    """Write-side contract for the vector/lexical store."""

    def index_text_units(self, text_units: list[TextUnit]) -> IndexingStats:
        """Embed and index text chunks."""
        ...

    def index_entities(self, entities: list[Entity]) -> IndexingStats:
        """Embed and index entity name/description vectors."""
        ...

    def index_community_reports(self, reports: list[CommunityReport]) -> IndexingStats:
        """Embed and index community-report vectors."""
        ...

    def index_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        """Embed and index relationship-description vectors (LightRAG global)."""
        ...

    def upsert(self, items: list[object]) -> IndexingStats:
        """Idempotently index ``items`` by id into the live index (delta)."""
        ...

    def delete_by_id(self, ids: list[str]) -> IndexingStats:
        """Delete documents by id (for content removed from the corpus)."""
        ...

    def clear(self, suffixes: list[str]) -> bool:
        """Remove all data for the given tenant/version ``suffixes``."""
        ...
