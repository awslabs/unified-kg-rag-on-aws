# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The smallest vector backend docs/guides/extend.md's "A new vector backend"
describes: a ``VectorIndexer`` with only its abstract writes and
``clear()``, so every concern port answers with its "cannot" default.

``TOY_BACKEND`` registers it with the store conformance suites
(``UKR_TEST_BACKEND=tests.fixtures.toy_vector_backend:TOY_BACKEND``), as a
backend author registers theirs. ``BROKEN_TOY_BACKEND`` overrides some
optional methods wrongly, so the suites must fail it.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from tests.fixtures.fakes.stores import FakeGraphStore
from tests.fixtures.store_backends import ExtraBackend
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import Config
from unified_kg_rag.ports import IndexingStats, VectorIndexer


class ToyVectorStore(VectorIndexer):
    """Items by kind and id, in memory; only the recipe's methods."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.items: dict[str, dict[str, Any]] = {}

    def _put(self, kind: str, items: list[Any]) -> IndexingStats:
        bucket = self.items.setdefault(kind, {})
        for item in items:
            bucket[item.id] = item
        return IndexingStats(total_items=len(items), successful_items=len(items))

    def index_text_units(
        self, text_units: list[Any], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        return self._put("text_units", text_units)

    def index_entities(
        self, entities: list[Any], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        return self._put("entities", entities)

    def index_community_reports(
        self, reports: list[Any], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        return self._put("community_reports", reports)

    def index_relationships(
        self, relationships: list[Any], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        return self._put("relationships", relationships)

    def index_claims(
        self, claims: list[Any], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        return self._put("claims", claims)

    def upsert_text_units(self, text_units: list[Any]) -> IndexingStats:
        return self._put("text_units", text_units)

    def upsert_entities(self, entities: list[Any]) -> IndexingStats:
        return self._put("entities", entities)

    def upsert_community_reports(self, reports: list[Any]) -> IndexingStats:
        return self._put("community_reports", reports)

    def upsert_relationships(self, relationships: list[Any]) -> IndexingStats:
        return self._put("relationships", relationships)

    def upsert_claims(self, claims: list[Any]) -> IndexingStats:
        return self._put("claims", claims)

    def delete_by_id(
        self, ids: list[str], alias_prefix: str, suffix: str
    ) -> IndexingStats:
        removed = 0
        for bucket in self.items.values():
            for item_id in ids:
                removed += bucket.pop(item_id, None) is not None
        return IndexingStats(total_items=len(ids), successful_items=removed)

    def delete_document_artifacts(
        self,
        ids: list[str],
        suffix: str,
        extra_relationship_ids: list[str] | None = None,
    ) -> dict[str, IndexingStats]:
        doomed = set(ids) | set(extra_relationship_ids or ())
        stats = {}
        for kind, bucket in self.items.items():
            gone = [item_id for item_id in bucket if item_id in doomed]
            for item_id in gone:
                del bucket[item_id]
            stats[kind] = IndexingStats(
                total_items=len(gone), successful_items=len(gone)
            )
        return stats

    def clear(self, suffixes: list[str]) -> None:
        self.items.clear()


class BrokenToyVectorStore(ToyVectorStore):
    """The toy with a summary tree that reports a write it never keeps and
    a text-unit probe that always answers no; ``read_text_unit_fields``,
    which the port lets a tree store leave at its default (a caller scans
    instead), is left there. A store an author got wrong: the suites fail
    the tests of what it overrides and skip only that default's."""

    def embed_summary_texts(
        self, texts: list[str], suffix: str | None
    ) -> list[list[float] | None] | None:
        return [[0.0] for _ in texts]

    def write_summary_tree(
        self,
        nodes: list[Any],
        vectors: Mapping[str, Collection[float]],
        suffix: str | None,
        record: dict[str, Any],
    ) -> IndexingStats:
        return IndexingStats(total_items=len(nodes), successful_items=len(nodes))

    def read_summary_nodes(
        self, suffix: str | None, *, fields: list[str]
    ) -> list[dict[str, Any]] | None:
        return []

    def read_summary_tree_record(self, suffix: str | None) -> dict[str, Any] | None:
        return None

    def holds_any_text_unit(self, suffix: str | None = None) -> bool | None:
        return False


def _providers(
    config: Config, embedding_factory: Any, store: type[ToyVectorStore] = ToyVectorStore
) -> Providers:
    vectors = store(config)
    graph = FakeGraphStore(config)
    return Providers(
        config,
        embedding_factory=embedding_factory,
        vector_indexer_factory=lambda: vectors,
        graph_indexer_factory=lambda: graph,
    )


TOY_BACKEND = ExtraBackend(name="toy", providers=_providers)
BROKEN_TOY_BACKEND = ExtraBackend(
    name="brokentoy",
    providers=lambda config, embeddings: _providers(
        config, embeddings, BrokenToyVectorStore
    ),
)
