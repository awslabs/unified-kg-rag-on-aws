# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""In-memory fake graph/vector stores for fast, AWS-free integration tests.

These conform to the write-side indexer ports (``GraphIndexer`` /
``VectorIndexer``) closely enough to drive ``IndexingManager`` and the
incremental path end to end, while recording what was written so tests can
assert idempotent upsert and delete-by-id behaviour without touching Neptune or
OpenSearch.
"""

from __future__ import annotations

from typing import Any

from aws_graphrag.ports.indexer import IndexingStats


class _Recorder:
    """Shared id-keyed store with idempotent upsert + delete-by-id."""

    def __init__(self) -> None:
        # collection name -> {id: item}
        self.data: dict[str, dict[str, Any]] = {}

    def _put(self, collection: str, items: list[Any] | None) -> IndexingStats:
        stats = IndexingStats()
        bucket = self.data.setdefault(collection, {})
        for item in items or []:
            bucket[item.id] = item
            stats.add_success()
        return stats

    def ids(self, collection: str) -> set[str]:
        return set(self.data.get(collection, {}).keys())

    def delete(self, ids: list[str]) -> IndexingStats:
        stats = IndexingStats()
        id_set = set(ids)
        for bucket in self.data.values():
            for removed in id_set & set(bucket):
                del bucket[removed]
                stats.add_success()
        return stats


class FakeGraphStore(_Recorder):
    """In-memory stand-in for the Neptune graph indexer."""

    def clear(self, suffixes: list[str]) -> bool:
        self.data.clear()
        return True

    def initialize(self) -> bool:
        return True

    def get_stats(self) -> dict[str, Any]:
        return {k: len(v) for k, v in self.data.items()}

    def index_entities(self, entities: list[Any] | None = None) -> IndexingStats:
        return self._put("entities", entities)

    def index_relationships(self, rels: list[Any] | None = None) -> IndexingStats:
        return self._put("relationships", rels)

    def index_communities(self, comms: list[Any] | None = None) -> IndexingStats:
        return self._put("communities", comms)

    def upsert_entities(self, entities: list[Any] | None = None) -> IndexingStats:
        return self._put("entities", entities)

    def upsert_relationships(self, rels: list[Any] | None = None) -> IndexingStats:
        return self._put("relationships", rels)

    def delete_by_id(self, ids: list[str]) -> IndexingStats:
        return self.delete(ids)

    def get_entity_count(self, suffixes: list[str]) -> int:
        return len(self.data.get("entities", {}))


class FakeVectorStore(_Recorder):
    """In-memory stand-in for the OpenSearch vector indexer."""

    def __init__(self, opensearch_config: Any | None = None) -> None:
        super().__init__()
        # IndexingManager.delete_documents reads index prefixes off this.
        self.opensearch_config = opensearch_config

    def clear(self, suffixes: list[str]) -> bool:
        self.data.clear()
        return True

    def initialize(self) -> bool:
        return True

    def get_stats(self) -> dict[str, Any]:
        return {k: len(v) for k, v in self.data.items()}

    def index_text_units(self, items: list[Any] | None = None) -> IndexingStats:
        return self._put("text_units", items)

    def index_entities(self, items: list[Any] | None = None) -> IndexingStats:
        return self._put("entities", items)

    def index_relationships(self, items: list[Any] | None = None) -> IndexingStats:
        return self._put("relationships", items)

    def index_community_reports(self, items: list[Any] | None = None) -> IndexingStats:
        return self._put("community_reports", items)

    def index_claims(self, items: list[Any] | None = None) -> IndexingStats:
        return self._put("claims", items)

    def upsert_text_units(self, items: list[Any] | None = None) -> IndexingStats:
        return self._put("text_units", items)

    def upsert_entities(self, items: list[Any] | None = None) -> IndexingStats:
        return self._put("entities", items)

    def upsert_relationships(self, items: list[Any] | None = None) -> IndexingStats:
        return self._put("relationships", items)

    def upsert_claims(self, items: list[Any] | None = None) -> IndexingStats:
        return self._put("claims", items)

    def delete_by_id(
        self, ids: list[str], prefix: str | None = None, suffix: str | None = None
    ) -> IndexingStats:
        return self.delete(ids)
