# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The graph writer of a run that builds no graph-store artifact.

A run whose methodologies build neither the graph nor communities
(``methodologies.writes_graph_store``: ``raptor`` alone) writes no vertex or
edge, so it needs no graph store: ``Providers.build_graph_indexer`` returns
this writer instead of the configured one, and the run neither connects to
the store nor needs its endpoint. What a namespace's graph held before is
not maintained, as any artifact of a methodology no longer selected, except
that a full build (a reset) replaces the graph of its namespaces with
nothing, as it empties their vector indices of what it does not build: with
a graph store configured (``store``) their vertices and edges are deleted.
"""

from __future__ import annotations

from collections.abc import Callable, Collection

from unified_kg_rag.domain.models import Community, Config, Entity, Relationship
from unified_kg_rag.ports.indexer import GraphIndexer, IndexingStats
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)


class NullGraphIndexer(GraphIndexer):
    """Writes nothing and holds nothing; every read answers as the port's
    defaults do for a backend that cannot tell.

    Args:
        config: The run's configuration.
        store: Builds the configured graph store's writer, when one is set
            up (its endpoint set): a full build clears its namespaces in it.
            ``None``: a full build leaves the store as it is.
    """

    def __init__(
        self, config: Config, store: Callable[[], GraphIndexer] | None = None
    ) -> None:
        super().__init__(config)
        self._store = store

    def clear(self, suffixes: list[str]) -> None:
        return None

    def index_entities(
        self, entities: list[Entity], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        # A full build writes the entities first: their suffixes' graph is
        # replaced with nothing here (the communities and edges too).
        if suffixes and self._store is not None:
            store = self._store()
            try:
                store.clear(sorted(suffixes))
            finally:
                store.close()
            logger.info(
                "Cleared the graph store of %s: the run builds no graph",
                ", ".join(sorted(suffixes)),
            )
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

    def holds_any_relationship(self, suffix: str | None = None) -> bool | None:
        return False
