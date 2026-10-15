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

from collections.abc import Collection
from typing import Any

from unified_kg_rag.adapters.storage.neptune_codec import (
    entity_from_vertex,
    entity_properties,
    relationship_from_edge,
    relationship_label,
    relationship_properties,
)
from unified_kg_rag.domain.models import Claim, Config, Constants
from unified_kg_rag.ports.indexer import BaseIndexer, IndexingStats

# The claim fields the OpenSearch claims index stores (``_prepare_claim_doc``).
_STORED_CLAIM_FIELDS = {
    "id",
    "subject_id",
    "subject_name",
    "object_id",
    "object_name",
    "type",
    "status",
    "description",
    "source_text",
    "text_unit_ids",
}


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
    upsert keeps its stored value, re-adding an edge replaces it, an edge whose
    endpoint vertex the suffix lacks is not written (a failed item, and the
    stored edge of its id is dropped), dropping a vertex drops its incident
    edges, and a full ``index_entities`` first clears the suffix's entity
    vertices.
    """

    def __init__(self, config: Config | None = None) -> None:
        self._max_length = (config or Config()).indexing.neptune.property_max_length
        # collection -> {(suffix, id): stored value map / edge row / model}
        self.data: dict[str, dict[tuple[str, str], Any]] = {}

    def close(self) -> None:
        return None

    def ids(self, collection: str) -> set[str]:
        return {item_id for _, item_id in self.data.get(collection, {})}

    def clear(self, suffixes: list[str]) -> None:
        for bucket in self.data.values():
            for key in [key for key in bucket if key[0] in suffixes]:
                del bucket[key]

    def initialize(self) -> None:
        return None

    # --- writes ------------------------------------------------------------

    def index_entities(
        self, entities: list[Any], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        for suffix in {BaseIndexer.get_suffix(e) for e in entities} | set(suffixes):
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
        vertices = self.data.get("entities", {})
        for rel in relationships:
            suffix = BaseIndexer.get_suffix(rel)
            # The stored edge of this id is dropped first (as
            # NeptuneIndexer._write_relationships does), so re-adding an edge
            # whose endpoint is gone leaves no edge behind.
            bucket.pop((suffix, rel.id), None)
            # addE runs between the suffix's entity vertices: an edge whose
            # endpoint vertex the suffix lacks is not written (see
            # NeptuneIndexer._build_add_edge_traversal).
            missing = [
                end
                for end in (rel.source_id, rel.target_id)
                if (suffix, end) not in vertices
            ]
            if missing:
                stats.add_error(
                    f"endpoint entity '{missing[0]}' not found", ids=[rel.id]
                )
                continue
            bucket[(suffix, rel.id)] = {
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

    def index_communities(
        self, communities: list[Any], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        bucket = self.data.setdefault("communities", {})
        replaced = {BaseIndexer.get_suffix(c) for c in communities} | set(suffixes)
        for key in [key for key in bucket if key[0] in replaced]:
            del bucket[key]
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

    def read_neighbor_ids(
        self, entity_ids: list[str], suffix: str | None = None
    ) -> dict[str, list[str]]:
        scope = suffix or Constants.DEFAULT_SUFFIX.value
        stored = {i for s, i in self.data.get("entities", {}) if s == scope}
        neighbors: dict[str, set[str]] = {i: set() for i in entity_ids if i in stored}
        for (s, _), edge in self.data.get("relationships", {}).items():
            if s != scope:
                continue
            for a, b in (
                (edge["source_id"], edge["target_id"]),
                (edge["target_id"], edge["source_id"]),
            ):
                if a in neighbors and b in stored:
                    neighbors[a].add(b)
        return {i: sorted(ids) for i, ids in neighbors.items()}

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

    def clear(self, suffixes: list[str]) -> None:
        # Only the given suffixes' documents, like the per-alias delete.
        for bucket in self.data.values():
            for item_id in [
                i
                for i, item in bucket.items()
                if BaseIndexer.get_suffix(item) in suffixes
            ]:
                del bucket[item_id]

    def check_index_settings(self, suffix: str | None = None) -> None:
        # Records no index settings, so any run's match.
        return None

    def set_doc_status_tracking(self, tracked: bool) -> None:
        return None

    def initialize(self) -> None:
        return None

    def _replace(
        self, collection: str, items: list[Any], suffixes: Collection[str]
    ) -> IndexingStats:
        """Full build: the items replace those of their and ``suffixes``'s
        suffixes (the alias swap onto a new index)."""
        bucket = self.data.setdefault(collection, {})
        replaced = {BaseIndexer.get_suffix(item) for item in items} | set(suffixes)
        for item_id in [
            i for i, item in bucket.items() if BaseIndexer.get_suffix(item) in replaced
        ]:
            del bucket[item_id]
        return self._put(collection, items)

    def index_text_units(
        self, text_units: list[Any], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        return self._replace("text_units", text_units, suffixes)

    def index_entities(
        self, entities: list[Any], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        return self._replace("entities", entities, suffixes)

    def index_relationships(
        self, relationships: list[Any], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        return self._replace("relationships", relationships, suffixes)

    def index_community_reports(
        self, reports: list[Any], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        return self._replace("community_reports", reports, suffixes)

    def index_claims(
        self, claims: list[Any], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        return self._replace("claims", claims, suffixes)

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

    def _update(
        self, collection: str, items: list[Any], fields: tuple[str, ...]
    ) -> IndexingStats:
        """Partial update: only ``fields`` of stored documents in the item's
        suffix (an item tagged with another suffix finds no document)."""
        stats = IndexingStats(total_items=len(items))
        bucket = self.data.setdefault(collection, {})
        for item in items:
            stored = bucket.get(item.id)
            if stored is not None and BaseIndexer.get_suffix(
                stored
            ) == BaseIndexer.get_suffix(item):
                bucket[item.id] = stored.model_copy(
                    update={name: getattr(item, name) for name in fields}
                )
            stats.add_success()
        return stats

    def update_entity_lineage(self, entities: list[Any]) -> IndexingStats:
        return self._update("entities", entities, ("text_unit_ids",))

    def read_text_unit_ids(
        self, ids: list[str], suffix: str | None = None
    ) -> list[str] | None:
        bucket = self.data.get("text_units", {})
        wanted = suffix or Constants.DEFAULT_SUFFIX.value
        return [
            i
            for i in ids
            if i in bucket and BaseIndexer.get_suffix(bucket[i]) == wanted
        ]

    def read_claims(self, ids: list[str], suffix: str | None = None) -> list[Any]:
        # Like the OpenSearch claims index: only the stored fields come back,
        # tagged with the suffix they were read from (the port's contract).
        bucket = self.data.get("claims", {})
        wanted = suffix or Constants.DEFAULT_SUFFIX.value
        return [
            Claim.model_validate(
                {
                    **bucket[i].model_dump(include=_STORED_CLAIM_FIELDS),
                    "attributes": ({Constants.INDEX.value: suffix} if suffix else {}),
                }
            )
            for i in ids
            if i in bucket and BaseIndexer.get_suffix(bucket[i]) == wanted
        ]

    def read_entity_descriptions(
        self, ids: list[str], suffix: str | None = None
    ) -> dict[str, str]:
        return self._descriptions("entities", ids, suffix)

    def read_relationship_descriptions(
        self, ids: list[str], suffix: str | None = None
    ) -> dict[str, str]:
        return self._descriptions("relationships", ids, suffix)

    def _descriptions(
        self, collection: str, ids: list[str], suffix: str | None
    ) -> dict[str, str]:
        bucket = self.data.get(collection, {})
        wanted = suffix or Constants.DEFAULT_SUFFIX.value
        return {
            i: bucket[i].description
            for i in ids
            if i in bucket
            and bucket[i].description
            and BaseIndexer.get_suffix(bucket[i]) == wanted
        }

    def update_claim_lineage(self, claims: list[Any]) -> IndexingStats:
        return self._update("claims", claims, ("text_unit_ids",))

    def update_relationship_lineage(self, relationships: list[Any]) -> IndexingStats:
        return self._update("relationships", relationships, ("text_unit_ids", "weight"))

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
