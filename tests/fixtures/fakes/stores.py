# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-memory fake graph/vector stores for fast, AWS-free integration tests.

These conform to the write-side indexer ports (``GraphIndexer`` /
``VectorIndexer``) closely enough to drive ``IndexingManager`` and the
incremental path end to end, while recording what was written so tests can
assert idempotent upsert and delete-by-id behaviour without touching Neptune or
OpenSearch.
"""

from __future__ import annotations

from typing import Any

from unified_kg_rag.ports.indexer import IndexingStats


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

    def close(self) -> None:
        """No-op teardown (mirrors the BaseIndexer.close default for in-memory
        stores), so the manager exercises the real close contract rather than
        falling through its AttributeError guard."""
        return None

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

    def upsert_communities(self, comms: list[Any] | None = None) -> IndexingStats:
        return self._put("communities", comms)

    def delete_by_id(self, ids: list[str], suffix: str | None = None) -> IndexingStats:
        return self.delete(ids)

    def get_entity_count(self, suffixes: list[str]) -> int:
        return len(self.data.get("entities", {}))

    def read_entities(self, ids: list[str]) -> list[Any]:
        bucket = self.data.get("entities", {})
        return [bucket[i] for i in ids if i in bucket]

    def read_relationships(self, ids: list[str]) -> list[Any]:
        bucket = self.data.get("relationships", {})
        return [bucket[i] for i in ids if i in bucket]

    def read_entity_names(self, suffix: str | None = None) -> list[tuple[str, str]]:
        bucket = self.data.get("entities", {})
        return [(e.id, e.name) for e in bucket.values()]

    def find_incident_relationship_ids(
        self, entity_ids: list[str], suffix: str | None = None
    ) -> list[str]:
        # Model the real contract: ids of stored relationships whose source or
        # target endpoint is one of the given entities (these become orphaned
        # when the entity is deleted).
        if not entity_ids:
            return []
        targets = set(entity_ids)
        bucket = self.data.get("relationships", {})
        return sorted(
            rid
            for rid, rel in bucket.items()
            if getattr(rel, "source_id", None) in targets
            or getattr(rel, "target_id", None) in targets
        )


class FakeVectorStore(_Recorder):
    """In-memory stand-in for the OpenSearch vector indexer."""

    def __init__(self, opensearch_config: Any | None = None) -> None:
        super().__init__()
        # IndexingManager.delete_documents reads index prefixes off this.
        self.opensearch_config = opensearch_config
        # Records (prefix, suffix) of each delete_by_id call so tests can assert
        # per-index routing (delete is fanned out once per index prefix).
        self.delete_calls: list[tuple[str | None, str | None]] = []

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
        self.delete_calls.append((prefix, suffix))
        return self.delete(ids)

    def delete_document_artifacts(
        self,
        ids: list[str],
        suffix: str,
        extra_relationship_ids: list[str] | None = None,
    ) -> dict[str, IndexingStats]:
        """Mirror the real OpenSearch fan-out across artifact indices.

        Records a delete call per index prefix (so per-index routing can be
        asserted) and folds orphaned incident-edge ids into the relationship
        index only, matching OpenSearchIndexer.delete_document_artifacts.
        """
        results: dict[str, IndexingStats] = {}
        if not ids:
            return results
        oc = self.opensearch_config
        rel_prefix = getattr(oc, "relationships_index_prefix", "relationships")
        prefixes = [
            getattr(oc, "text_units_index_prefix", "text-units"),
            getattr(oc, "entities_index_prefix", "entities"),
            rel_prefix,
            getattr(oc, "claims_index_prefix", "claims"),
            getattr(oc, "community_reports_index_prefix", "community-reports"),
        ]
        orphan_ids = set(extra_relationship_ids or [])
        for prefix in prefixes:
            delete_ids = (
                sorted(set(ids) | orphan_ids)
                if prefix == rel_prefix and orphan_ids
                else ids
            )
            results[f"opensearch_delete_{prefix}_{suffix}"] = self.delete_by_id(
                delete_ids, prefix, suffix
            )
        return results
