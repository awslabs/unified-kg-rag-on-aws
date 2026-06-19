# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Graph-store port — the boundary for the knowledge-graph backend.

Production adapter: Neptune (``aws_graphrag.storage.neptune_indexer`` for writes,
``aws_graphrag.retrieval.retrievers.neptune_retriever`` for traversal). The
incremental-indexing work (M2) adds idempotent ``upsert_*`` and ``delete_*_by_id``
to support delta updates without wiping a label; this port declares that
contract so the domain and tests target it uniformly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aws_graphrag.models import Community, Entity, Relationship
    from aws_graphrag.storage.base import IndexingStats


@runtime_checkable
class GraphStorePort(Protocol):
    """Write-side contract for the knowledge graph."""

    def index_entities(self, entities: list[Entity]) -> IndexingStats:
        """Create graph vertices for ``entities`` (full-run semantics)."""
        ...

    def index_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        """Create graph edges for ``relationships`` (full-run semantics)."""
        ...

    def index_communities(self, communities: list[Community]) -> IndexingStats:
        """Create community vertices and membership edges (full-run semantics)."""
        ...

    def upsert_entities(self, entities: list[Entity]) -> IndexingStats:
        """Idempotently merge ``entities`` into the live graph (delta semantics)."""
        ...

    def upsert_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        """Idempotently merge ``relationships`` into the live graph (delta)."""
        ...

    def delete_by_id(self, ids: list[str]) -> IndexingStats:
        """Delete vertices/edges by id (for documents removed from the corpus)."""
        ...

    def clear(self, suffixes: list[str]) -> bool:
        """Remove all data for the given tenant/version ``suffixes``."""
        ...
