# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The graph writer of a run that builds no graph-store artifact.

A run whose methodologies build neither the graph nor communities
(``methodologies.writes_graph_store``: ``raptor`` alone) writes no vertex or
edge, so it needs no graph store: ``Providers.build_graph_indexer`` returns
this writer instead of the configured one, and the run neither connects to
the store nor needs its endpoint. What a namespace's graph held before is
not maintained, as any artifact of a methodology no longer selected.
"""

from __future__ import annotations

from collections.abc import Collection

from unified_kg_rag.domain.models import Community, Entity, Relationship
from unified_kg_rag.ports.indexer import GraphIndexer, IndexingStats


class NullGraphIndexer(GraphIndexer):
    """Writes nothing and holds nothing; every read answers as the port's
    defaults do for a backend that cannot tell."""

    def clear(self, suffixes: list[str]) -> None:
        return None

    def index_entities(
        self, entities: list[Entity], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        return IndexingStats()

    def index_communities(
        self, communities: list[Community], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        return IndexingStats()

    def upsert_entities(self, entities: list[Entity]) -> IndexingStats:
        return IndexingStats()

    def upsert_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        return IndexingStats()

    def delete_by_id(self, ids: list[str], suffix: str | None = None) -> IndexingStats:
        return IndexingStats()

    def holds_any_item(self, suffix: str | None = None) -> bool | None:
        return False
