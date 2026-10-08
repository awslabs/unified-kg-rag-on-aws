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

from unified_kg_rag.adapters.storage.neptune_codec import (
    entity_from_vertex,
    entity_properties,
    relationship_from_edge,
    relationship_label,
    relationship_properties,
)
from unified_kg_rag.domain.models import Config, Constants
from unified_kg_rag.ports.indexer import BaseIndexer, IndexingStats


class _Recorder:
    """Shared id-keyed store with idempotent upsert + delete-by-id."""

    def __init__(self) -> None:
        # collection name -> {id: item}
        self.data: dict[str, dict[str, Any]] = {}

    def _put(self, collection: str, items: list[Any]) -> IndexingStats:
        stats = IndexingStats()
        bucket = self.data.setdefault(collection, {})
        for item in items:
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


class FakeGraphStore:
    """In-memory stand-in for the Neptune graph indexer.

    Stores what Neptune would hold, not the models: each entity is the
    ``valueMap()`` of its vertex and each relationship is its edge (label,
    endpoints, properties), both encoded by the REAL ``neptune_codec`` the
    Neptune indexer writes with, and read back through the real decoders. A
    lossy encode/decode pair therefore fails the AWS-free tests too. Items are
    keyed by ``(suffix, id)``: ids are suffix-independent, and reads, deletes
    and counts are scoped per suffix like the label-scoped Neptune traversals.

    Neptune semantics kept: list properties of a vertex are set-cardinality
    (distinct values, a re-upsert replaces the list), a property absent from an
    upsert keeps its stored value, re-adding an edge replaces it, dropping a
    vertex drops its incident edges, and a full ``index_entities`` first clears
    the suffix's entity vertices.
    """

    def __init__(self, config: Config | None = None) -> None:
        self._max_length = (config or Config()).indexing.neptune.property_max_length
        # collection -> {(suffix, id): stored value map / edge row / model}
        self.data: dict[str, dict[tuple[str, str], Any]] = {}

    def close(self) -> None:
        return None

    def ids(self, collection: str) -> set[str]:
        return {item_id for _, item_id in self.data.get(collection, {})}

    def clear(self, suffixes: list[str]) -> bool:
        for bucket in self.data.values():
            for key in [key for key in bucket if key[0] in suffixes]:
                del bucket[key]
        return True

    def initialize(self) -> bool:
        return True

    # --- writes ------------------------------------------------------------

    def index_entities(self, entities: list[Any]) -> IndexingStats:
        for suffix in {BaseIndexer.get_suffix(e) for e in entities}:
            self._drop_vertices(suffix, list(self.ids("entities")))
        return self.upsert_entities(entities)

    def upsert_entities(self, entities: list[Any]) -> IndexingStats:
        stats = IndexingStats(total_items=len(entities))
        bucket = self.data.setdefault("entities", {})
        for entity in entities:
            value_map = bucket.setdefault(
                (BaseIndexer.get_suffix(entity), entity.id), {"id": [entity.id]}
            )
            for key, value in entity_properties(entity, self._max_length).items():
                values = value if isinstance(value, list) else [value]
                value_map[key] = list(dict.fromkeys(values))
            stats.add_success()
        return stats

    def index_relationships(self, relationships: list[Any]) -> IndexingStats:
        return self.upsert_relationships(relationships)

    def upsert_relationships(self, relationships: list[Any]) -> IndexingStats:
        stats = IndexingStats(total_items=len(relationships))
        bucket = self.data.setdefault("relationships", {})
        for rel in relationships:
            bucket[(BaseIndexer.get_suffix(rel), rel.id)] = {
                "props": {
                    "id": rel.id,
                    **relationship_properties(rel, self._max_length),
                },
                "label": relationship_label(rel),
                "source_id": rel.source_id,
                "target_id": rel.target_id,
            }
            stats.add_success()
        return stats

    def index_communities(self, communities: list[Any]) -> IndexingStats:
        return self.upsert_communities(communities)

    def upsert_communities(self, communities: list[Any]) -> IndexingStats:
        stats = IndexingStats(total_items=len(communities))
        bucket = self.data.setdefault("communities", {})
        for comm in communities:
            bucket[(BaseIndexer.get_suffix(comm), comm.id)] = comm
            stats.add_success()
        return stats

    def delete_by_id(self, ids: list[str], suffix: str | None = None) -> IndexingStats:
        stats = IndexingStats(total_items=len(ids))
        suffixes = (
            [suffix]
            if suffix is not None
            else sorted({s for bucket in self.data.values() for s, _ in bucket})
        )
        for scope in suffixes:
            edges = self.data.get("relationships", {})
            for item_id in ids:
                edges.pop((scope, item_id), None)
            self._drop_vertices(scope, ids)
        stats.add_success(len(ids))
        return stats

    def _drop_vertices(self, suffix: str, ids: list[str]) -> None:
        dropped = set(ids)
        for collection in ("entities", "communities"):
            bucket = self.data.get(collection, {})
            for item_id in dropped:
                bucket.pop((suffix, item_id), None)
        edges = self.data.get("relationships", {})
        for key in [
            key
            for key, edge in edges.items()
            if key[0] == suffix
            and (edge["source_id"] in dropped or edge["target_id"] in dropped)
        ]:
            del edges[key]

    # --- reads ---------------------------------------------------------------

    def read_entities(self, ids: list[str], suffix: str | None = None) -> list[Any]:
        bucket = self.data.get("entities", {})
        scope = suffix or Constants.DEFAULT_SUFFIX.value
        rows = [bucket[(scope, i)] for i in ids if (scope, i) in bucket]
        return [e for e in (entity_from_vertex(row) for row in rows) if e]

    def read_relationships(
        self, ids: list[str], suffix: str | None = None
    ) -> list[Any]:
        bucket = self.data.get("relationships", {})
        scope = suffix or Constants.DEFAULT_SUFFIX.value
        rows = [bucket[(scope, i)] for i in ids if (scope, i) in bucket]
        rels = (
            relationship_from_edge(
                row["props"], row["label"], row["source_id"], row["target_id"]
            )
            for row in rows
        )
        return [r for r in rels if r]

    def read_entity_names(self, suffix: str | None = None) -> list[tuple[str, str]]:
        scope = suffix or Constants.DEFAULT_SUFFIX.value
        return [
            (item_id, row["name"][0])
            for (s, item_id), row in self.data.get("entities", {}).items()
            if s == scope
        ]

    def find_incident_relationship_ids(
        self, entity_ids: list[str], suffix: str | None = None
    ) -> list[str]:
        # Model the real contract: ids of stored relationships whose source or
        # target endpoint is one of the given entities (these become orphaned
        # when the entity is deleted).
        targets = set(entity_ids)
        return sorted(
            {
                item_id
                for (s, item_id), edge in self.data.get("relationships", {}).items()
                if (suffix is None or s == suffix)
                and (edge["source_id"] in targets or edge["target_id"] in targets)
            }
        )


class FakeVectorStore(_Recorder):
    """In-memory stand-in for the OpenSearch vector indexer."""

    def __init__(self, opensearch_config: Any | None = None) -> None:
        super().__init__()
        # IndexingManager.delete_documents reads index prefixes off this.
        self.opensearch_config = opensearch_config
        # Records (alias_prefix, suffix) of each delete_by_id call so tests can
        # assert per-index routing (delete is fanned out once per index prefix).
        self.delete_calls: list[tuple[str, str]] = []

    def clear(self, suffixes: list[str]) -> bool:
        self.data.clear()
        return True

    def initialize(self) -> bool:
        return True

    def index_text_units(self, items: list[Any] | None = None) -> IndexingStats:
        return self._put("text_units", items)

    def index_text_units(self, text_units: list[Any]) -> IndexingStats:
        return self._put("text_units", text_units)

    def index_entities(self, entities: list[Any]) -> IndexingStats:
        return self._put("entities", entities)

    def index_relationships(self, relationships: list[Any]) -> IndexingStats:
        return self._put("relationships", relationships)

    def index_community_reports(self, reports: list[Any]) -> IndexingStats:
        return self._put("community_reports", reports)

    def index_claims(self, claims: list[Any]) -> IndexingStats:
        return self._put("claims", claims)

    def upsert_text_units(self, text_units: list[Any]) -> IndexingStats:
        return self._put("text_units", text_units)

    def upsert_entities(self, entities: list[Any]) -> IndexingStats:
        return self._put("entities", entities)

    def upsert_relationships(self, relationships: list[Any]) -> IndexingStats:
        return self._put("relationships", relationships)

    def upsert_claims(self, claims: list[Any]) -> IndexingStats:
        return self._put("claims", claims)

    def upsert_community_reports(self, reports: list[Any]) -> IndexingStats:
        return self._put("community_reports", reports)

    def delete_by_id(
        self, ids: list[str], alias_prefix: str, suffix: str
    ) -> IndexingStats:
        self.delete_calls.append((alias_prefix, suffix))
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
