# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for IndexingManager delta routing (M2).

These verify that the incremental path dispatches to the indexers' idempotent
``upsert_*`` / ``delete_by_id`` methods (not the full-rebuild ``index_*``), with
the indexers themselves mocked so no AWS clients are constructed.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.models import Entity, Relationship
from unified_kg_rag.ports.indexer import BaseIndexer, IndexingStats

pytestmark = pytest.mark.unit


@pytest.fixture
def manager(mocker):
    # Patch the indexer classes before IndexingManager.__init__ instantiates them.
    os_indexer = mocker.MagicMock()
    neptune_indexer = mocker.MagicMock()
    os_indexer.upsert_entities.return_value = IndexingStats()
    os_indexer.upsert_text_units.return_value = IndexingStats()
    os_indexer.upsert_relationships.return_value = IndexingStats()
    os_indexer.index_community_reports.return_value = IndexingStats()
    os_indexer.delete_by_id.return_value = IndexingStats()
    os_indexer.opensearch_config.text_units_index_prefix = "text-units"
    os_indexer.opensearch_config.entities_index_prefix = "entities"
    os_indexer.opensearch_config.relationships_index_prefix = "relationships"
    os_indexer.opensearch_config.claims_index_prefix = "claims"
    os_indexer.opensearch_config.community_reports_index_prefix = "community-reports"
    neptune_indexer.upsert_entities.return_value = IndexingStats()
    neptune_indexer.upsert_relationships.return_value = IndexingStats()
    neptune_indexer.index_communities.return_value = IndexingStats()
    neptune_indexer.delete_by_id.return_value = IndexingStats()
    # No orphaned incident edges by default; orphan-cleanup test overrides this.
    neptune_indexer.find_incident_relationship_ids.return_value = []

    # Inject the mocked indexers via the port-based DI seam (no module-level
    # patching needed — the manager constructs concrete adapters only when none
    # are supplied).
    os_indexer.delete_document_artifacts.side_effect = (
        lambda ids, suffix, extra_relationship_ids=None: {
            f"opensearch_delete_{prefix}_{suffix}": IndexingStats()
            for prefix in (
                "text-units",
                "entities",
                "relationships",
                "claims",
                "community-reports",
            )
        }
    )
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager
    from unified_kg_rag.domain.models import Config

    mgr = IndexingManager(
        config=Config(), vector_indexer=os_indexer, graph_indexer=neptune_indexer
    )
    return mgr, os_indexer, neptune_indexer


def test_index_delta_routes_to_upserts(manager) -> None:
    mgr, os_indexer, neptune_indexer = manager
    entities = [Entity(id="e1", name="Alice")]
    relationships = [Relationship(id="r1", source_id="e1", target_id="e2")]

    mgr.index_delta(entities=entities, relationships=relationships)

    os_indexer.upsert_entities.assert_called_once_with(entities)
    neptune_indexer.upsert_entities.assert_called_once_with(entities)
    neptune_indexer.upsert_relationships.assert_called_once_with(relationships)
    # Relationship vectors (LightRAG global) are upserted on a delta run too.
    os_indexer.upsert_relationships.assert_called_once_with(relationships)
    # Full-rebuild entity path must NOT be used on a delta run.
    os_indexer.index_entities.assert_not_called()
    neptune_indexer.index_entities.assert_not_called()
    neptune_indexer.index_relationships.assert_not_called()


def test_delete_documents_routes_to_delete_by_id(manager) -> None:
    mgr, os_indexer, neptune_indexer = manager

    mgr.delete_documents({"default": ["e1", "r1", "t1"]})

    # Suffix is threaded through so the Neptune drop scopes to this suffix's
    # labels (cross-tenant safety: content-hash ids can collide across suffixes).
    neptune_indexer.delete_by_id.assert_called_once_with(
        ["e1", "r1", "t1"], suffix="default"
    )
    # The manager delegates the per-index vector fan-out to the vector port's
    # cohesive delete_document_artifacts (the prefix layout is the adapter's
    # concern, not the manager's), threading the suffix and no orphan edges.
    os_indexer.delete_document_artifacts.assert_called_once_with(
        ["e1", "r1", "t1"], "default", extra_relationship_ids=[]
    )


def test_delete_documents_skips_empty(manager) -> None:
    mgr, os_indexer, neptune_indexer = manager
    mgr.delete_documents({"default": []})
    neptune_indexer.delete_by_id.assert_not_called()
    os_indexer.delete_document_artifacts.assert_not_called()


def test_delete_documents_cascades_orphan_relationship_ids(manager) -> None:
    # Orphan-edge cleanup: a relationship pointing at a deleted entity is owned
    # (in lineage) by a surviving document, so it is NOT in the exclusive id-set.
    # delete_documents must query Neptune for incident edge ids and pass them as
    # extra_relationship_ids so the vector backend can fold them into the
    # relationship-index deletion (no dangling relationship document survives).
    mgr, os_indexer, neptune_indexer = manager
    neptune_indexer.find_incident_relationship_ids.return_value = ["r_orphan"]

    mgr.delete_documents({"default": ["e_target"]})

    # Incident edges are queried for the deleted entities, scoped to the suffix.
    neptune_indexer.find_incident_relationship_ids.assert_called_once_with(
        ["e_target"], suffix="default"
    )
    # The orphaned incident edge is handed to the vector port as an extra
    # relationship id to delete.
    os_indexer.delete_document_artifacts.assert_called_once_with(
        ["e_target"], "default", extra_relationship_ids=["r_orphan"]
    )


def test_cross_run_merge_threads_entity_id_remap_to_relationships(manager) -> None:
    # When the cross-run entity merge collapses two ids onto one survivor, the
    # delta relationship endpoints must be remapped so edges point at the
    # surviving entity id (the manager previously discarded the remap, leaving
    # relationships dangling on a now-nonexistent id).
    mgr, os_indexer, neptune_indexer = manager

    # Existing entity 'e_canon' and a delta entity 'e_dup' that normalizes to the
    # same name -> merge_entities collapses e_dup -> e_canon and returns a remap.
    survivor = Entity(id="e_canon", name="acme corp", text_unit_ids=["t1"])
    dup = Entity(id="e_dup", name="acme corp", text_unit_ids=["t2"])
    neptune_indexer.read_entities.return_value = [survivor]
    # No existing relationship rows read back — the remap path must still apply.
    neptune_indexer.read_relationships.return_value = []

    delta_rel = Relationship(
        id="r1", source_id="e_dup", target_id="e_other", type="WORKS_AT"
    )
    result = mgr.merge_with_existing_graph([dup], [delta_rel])
    merged_rels = result.relationships

    # The relationship endpoint that referenced the merged-away id is remapped.
    assert merged_rels is not None and len(merged_rels) == 1
    assert merged_rels[0].source_id == "e_canon"
    assert merged_rels[0].target_id == "e_other"


def test_cross_run_merge_of_a_same_id_entity_rewrites_no_unchanged_edge(
    manager,
) -> None:
    # The merge reads the delta entity's stored edges to find one of another
    # id between the same entities (none here: r-old joins another pair);
    # its other edges are rewritten only when the name or rank they carry
    # changes (here neither: the stored edge already counts the vendor's new
    # neighbour).
    mgr, _, neptune_indexer = manager
    neptune_indexer.read_entities.return_value = [
        Entity(id="e-vendor", name="Vendor", text_unit_ids=["t1"])
    ]
    other = Relationship(
        id="r-old",
        source_id="e-vendor",
        source_name="Vendor",
        target_id="e-depot",
        target_name="Depot",
        rank=3,
    )
    neptune_indexer.read_relationships.side_effect = lambda ids, suffix=None: (
        [other] if ids == ["r-old"] else []
    )
    neptune_indexer.find_incident_relationship_ids.return_value = ["r-old", "r-new"]
    neptune_indexer.read_neighbor_ids.return_value = {
        "e-vendor": ["e-depot"],
        "e-depot": ["e-vendor"],
    }
    delta_rel = Relationship(
        id="r-new", source_id="e-vendor", target_id="e-bank", text_unit_ids=["t2"]
    )

    result = mgr.merge_with_existing_graph(
        [Entity(id="e-vendor", name="Vendor", text_unit_ids=["t2"])], [delta_rel]
    )

    assert neptune_indexer.read_relationships.call_args_list[0].args == (
        ["r-new", "r-old"],
    )
    assert [r.id for r in result.relationships or []] == ["r-new"]
    assert result.id_remap_by_suffix == {}


def test_fuzzy_cross_run_merge_collapses_near_duplicate_entity(mocker) -> None:
    # With fuzzy_entity_resolution on, a delta entity whose name only FUZZILY
    # matches an existing one (different id) must collapse onto the existing
    # entity, not accumulate as a separate node.
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager
    from unified_kg_rag.domain.models import Config

    os_indexer = mocker.MagicMock()
    neptune_indexer = mocker.MagicMock()

    existing = Entity(id="e_old", name="Acme Corporation", text_unit_ids=["t1"])
    # The delta's id differs (derived from a different surface form), so exact-id
    # read-back alone would miss it — the fuzzy name projection must find it.
    neptune_indexer.read_entity_names.return_value = [("e_old", "Acme Corporation")]
    neptune_indexer.read_entities.return_value = [existing]
    neptune_indexer.read_relationships.return_value = []

    config = Config()
    config.processing.fuzzy_entity_resolution = True
    mgr = IndexingManager(
        config=config, vector_indexer=os_indexer, graph_indexer=neptune_indexer
    )
    # Keep the description summarizer a no-op (no Bedrock).
    mgr._description_summarizer = mocker.MagicMock()
    mgr._description_summarizer.summarize_entities.side_effect = lambda es: es
    mgr._description_summarizer.summarize_relationships.side_effect = lambda rs: rs

    delta = Entity(id="e_new", name="Acme Corporatio", text_unit_ids=["t2"])
    result = mgr.merge_with_existing_graph([delta], None)
    merged_entities, _ = result.entities, result.relationships

    assert merged_entities is not None and len(merged_entities) == 1
    assert merged_entities[0].id == "e_old"
    assert set(merged_entities[0].text_unit_ids) == {"t1", "t2"}


def test_cross_run_merge_is_scoped_per_suffix(manager) -> None:
    # Entity and relationship ids are suffix-independent: "Vendor" has the same
    # id in every tenant. Each tenant's delta must merge with that tenant's
    # stored state only and stay in its own suffix.
    mgr, _, neptune_indexer = manager
    stored = {
        suffix: Entity(
            id="e-vendor",
            name="Vendor",
            text_unit_ids=[f"{suffix}-old"],
            attributes={"index": suffix},
        )
        for suffix in ("tenant-a", "tenant-b")
    }
    neptune_indexer.read_entities.side_effect = lambda ids, suffix=None: (
        [stored[suffix]] if suffix in stored else []
    )
    neptune_indexer.read_relationships.return_value = []
    delta = [
        Entity(
            id="e-vendor",
            name="Vendor",
            text_unit_ids=[f"{suffix}-new"],
            attributes={"index": suffix},
        )
        for suffix in ("tenant-a", "tenant-b")
    ]

    result = mgr.merge_with_existing_graph(delta, None)
    merged, _ = result.entities, result.relationships

    assert merged is not None
    by_suffix = {BaseIndexer.get_suffix(e): e for e in merged}
    assert set(by_suffix) == {"tenant-a", "tenant-b"}
    for suffix, entity in by_suffix.items():
        assert entity.text_unit_ids == [f"{suffix}-old", f"{suffix}-new"]
    assert {
        call.kwargs["suffix"] for call in neptune_indexer.read_entities.call_args_list
    } == {"tenant-a", "tenant-b"}


def test_read_back_entity_without_attributes_keeps_the_delta_suffix(manager) -> None:
    # A merged entity is written back under get_suffix(entity): losing the
    # attributes would move it into the default suffix.
    mgr, _, neptune_indexer = manager
    neptune_indexer.read_entities.return_value = [
        Entity(id="e-vendor", name="Vendor", text_unit_ids=["t1"])
    ]
    delta = Entity(
        id="e-vendor",
        name="Vendor",
        text_unit_ids=["t2"],
        attributes={"index": "tenant-a"},
    )

    result = mgr.merge_with_existing_graph([delta], None)
    merged, _ = result.entities, result.relationships

    assert merged is not None
    assert [BaseIndexer.get_suffix(e) for e in merged] == ["tenant-a"]


def test_text_unit_removal_updates_vectors_in_place(manager) -> None:
    # Stripping a removed text unit changes lineage and weight only: the
    # vector store must not re-embed the unchanged text.
    mgr, os_indexer, neptune_indexer = manager
    neptune_indexer.read_entities.return_value = [
        Entity(id="e1", name="Vendor", text_unit_ids=["t1", "t2"])
    ]
    neptune_indexer.read_relationships.return_value = [
        Relationship(id="r1", source_id="e1", target_id="e2", text_unit_ids=["t2"])
    ]
    # The vector copies hold what the derived items embed.
    os_indexer.read_vector_copies.side_effect = lambda kind, ids, suffix=None: {
        "entities": {
            "e1": {"name": "Vendor", "type": "", "confidence": 1.0, "description": ""}
        },
        "relationships": {
            "r1": {
                "source_id": "e1",
                "target_id": "e2",
                "source_name": "",
                "target_name": "",
                "description": "",
            }
        },
    }[kind]
    os_indexer.update_entity_lineage.return_value = IndexingStats()
    os_indexer.update_relationship_lineage.return_value = IndexingStats()

    mgr.remove_text_units_from_shared(
        "default", entity_ids=["e1"], relationship_ids=["r1"], text_unit_ids=["t2"]
    )

    os_indexer.upsert_entities.assert_not_called()
    os_indexer.upsert_relationships.assert_not_called()
    (entities,) = os_indexer.update_entity_lineage.call_args.args
    assert entities[0].text_unit_ids == ["t1"]
    os_indexer.update_relationship_lineage.assert_called_once()
    neptune_indexer.upsert_entities.assert_called_once()
    neptune_indexer.upsert_relationships.assert_called_once()


def test_a_failed_shared_claim_read_fails_the_removal(manager) -> None:
    # The caller keeps the registry rows (and retries) on any failed item.
    mgr, os_indexer, _ = manager
    os_indexer.read_claims.side_effect = RuntimeError("search timed out")

    results = mgr.remove_text_units_from_shared(
        "default",
        entity_ids=[],
        relationship_ids=[],
        text_unit_ids=["t2"],
        claim_ids=["c1"],
    )

    stats = results["opensearch_read_claims_default"]
    assert stats.failed_items == 1
    assert stats.failed_ids == ["c1"]
    os_indexer.update_claim_lineage.assert_not_called()


def test_a_stripped_claim_is_written_back_to_the_index_it_was_read_from(
    manager,
) -> None:
    # A read_claims that tags nothing still writes back to the read suffix.
    from unified_kg_rag.domain.models import Claim

    mgr, os_indexer, _ = manager
    os_indexer.read_claims.return_value = [
        Claim(
            id="c1",
            subject_id="s",
            subject_name="S",
            object_name="O",
            type="T",
            text_unit_ids=["t1", "t2"],
        )
    ]
    os_indexer.update_claim_lineage.return_value = IndexingStats()

    mgr.remove_text_units_from_shared(
        "tenant-a",
        entity_ids=[],
        relationship_ids=[],
        text_unit_ids=["t2"],
        claim_ids=["c1"],
    )

    (written,) = os_indexer.update_claim_lineage.call_args.args
    assert [c.text_unit_ids for c in written] == [["t1"]]
    assert [BaseIndexer.get_suffix(c) for c in written] == ["tenant-a"]


def test_a_failed_stored_claim_read_raises_before_any_write(manager) -> None:
    from unified_kg_rag.domain.models import Claim

    mgr, os_indexer, _ = manager
    os_indexer.read_claims.side_effect = RuntimeError("search timed out")
    claim = Claim(id="c1", subject_id="s", subject_name="S", object_name="O", type="T")

    with pytest.raises(RuntimeError):
        mgr.merge_with_stored_claims([claim])


@pytest.mark.parametrize("kind", ["entity", "relationship"])
def test_cross_run_merge_keeps_the_description_neptune_capped(kind) -> None:
    # Neptune caps property values at property_max_length; the merged
    # description is written to OpenSearch too, so it must start from the
    # full stored one, not from the graph's cut copy.
    from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager
    from unified_kg_rag.domain.models import Config

    config = Config()
    config.indexing.neptune.property_max_length = 40
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    mgr = IndexingManager(
        config=config, vector_indexer=vector, graph_indexer=FakeGraphStore(config)
    )
    full = "Vendor supplies widgets to Buyer under a long-running agreement."
    entities = [Entity(id=i, name=n) for i, n in (("e1", "Vendor"), ("e2", "Buyer"))]
    if kind == "entity":
        stored = Entity(id="e1", name="Vendor", description=full, text_unit_ids=["t1"])
        delta = stored.model_copy(
            update={"description": "Ships.", "text_unit_ids": ["t2"]}
        )
        mgr.index_delta(entities=[stored, entities[1]])
        merged = mgr.merge_with_existing_graph([delta], None).entities
    else:
        stored = Relationship(
            id="r1",
            source_id="e1",
            target_id="e2",
            description=full,
            text_unit_ids=["t1"],
        )
        delta = stored.model_copy(
            update={"description": "Ships.", "text_unit_ids": ["t2"]}
        )
        mgr.index_delta(entities=entities, relationships=[stored])
        merged = mgr.merge_with_existing_graph(None, [delta]).relationships

    assert merged is not None
    (item,) = [m for m in merged if m.id == delta.id]
    assert item.description == f"{full}\nShips."


def test_a_delta_edge_takes_the_name_of_the_stored_entity_it_merged_into() -> None:
    # The delta's "Acme Inc" merges into the stored "ACME, Inc.": its
    # edge is re-pointed at the stored id and must name that entity too.
    from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager
    from unified_kg_rag.domain.models import Config

    config = Config()
    mgr = IndexingManager(
        config=config,
        vector_indexer=FakeVectorStore(opensearch_config=config.indexing.opensearch),
        graph_indexer=FakeGraphStore(config),
    )
    buyer = Entity(id="e-buyer", name="Buyer", text_unit_ids=["t1"])
    mgr.index_delta(
        entities=[
            Entity(id="e-old", name="ACME, Inc.", text_unit_ids=["t1"]),
            buyer,
        ]
    )
    delta_edge = Relationship(
        id="r-new",
        source_id="e-new",
        target_id="e-buyer",
        source_name="Acme Inc",
        target_name="Buyer",
        type="SUPPLIES",
        text_unit_ids=["t2"],
    )

    result = mgr.merge_with_existing_graph(
        [Entity(id="e-new", name="Acme Inc", text_unit_ids=["t2"]), buyer],
        [delta_edge],
    )

    assert result.relationships is not None
    (edge,) = result.relationships
    assert (edge.source_id, edge.source_name) == ("e-old", "ACME, Inc.")
    assert delta_edge.source_name == "Acme Inc"


def test_unit_records_reads_the_cited_ids_a_bounded_number_of_times() -> None:
    # The cited-id set was rebuilt for every delta unit: quadratic in a large
    # delta (seconds per derive round at 10k units citing 40k).
    from collections.abc import Iterator
    from types import SimpleNamespace

    from unified_kg_rag.application.storage.indexing_manager import IndexingManager
    from unified_kg_rag.domain.models import TextUnit

    class _CountedIds(list[str]):
        passes = 0

        def __iter__(self) -> Iterator[str]:
            type(self).passes += 1
            return super().__iter__()

    class _Store:
        def read_text_unit_attributes(self, ids, suffix=None):
            return {}

        def read_extraction_records(self, keys, suffix=None):
            return {}

    units = [TextUnit(id=f"t{i}", text="t") for i in range(50)]
    unit_ids = _CountedIds(f"t{i}" for i in range(100))
    me = SimpleNamespace(opensearch_indexer=_Store())
    IndexingManager._unit_records(me, "s", unit_ids, units)  # type: ignore[arg-type]
    assert _CountedIds.passes <= 2


def test_incident_edges_are_read_once_per_site_until_its_ids_shrink() -> None:
    # The stored edges do not change during a derivation: a site asking a
    # superset of its last ids reads only the ones it adds (all of them, in
    # each of up to 32 rounds, before). Asked other ids, it reads them all
    # again: the edges of ids it no longer asks must not come back.
    from unified_kg_rag.application.storage.indexing_manager import _IncidentEdges

    edges = {"a": {"r-a"}, "b": {"r-b"}, "c": {"r-c", "r-a"}}
    reads: list[list[str]] = []

    def read(ids: list[str]) -> list[str]:
        reads.append(ids)
        return sorted({r for i in ids for r in edges[i]})

    incident = _IncidentEdges(read)
    assert incident("given", ["a", "b"]) == {"r-a", "r-b"}
    assert incident("given", ["b", "a", "c"]) == {"r-a", "r-b", "r-c"}
    assert incident("moved", ["a"]) == {"r-a"}
    assert incident("given", ["c"]) == {"r-c", "r-a"}
    assert reads == [["a", "b"], ["c"], ["a"], ["c"]]


def test_claim_records_of_every_round_are_derived_with(manager) -> None:
    # The given claim (a stale id, citing u1) derives to the id a stored
    # claim citing u2 holds: the second round reads only u2's records, and
    # the claim is derived from u1's statement and u2's together. Had the
    # round's read replaced the first's, u1's statement was lost.
    from tests.unit.test_derivation import _entity, _pays, _record, _stored
    from unified_kg_rag.domain.ingestion.claim_resolver import resolved_claim_id

    mgr, _, _ = manager
    acme, buyer = _stored("Acme", ["u1"]), _stored("Buyer", ["u1"])
    records = {
        u: _record(u, _entity("Acme", u), _entity("Buyer", u)) for u in ("u1", "u2")
    }
    statements = {u: [_pays("Acme", "Buyer", [u], f"s-{u}")] for u in ("u1", "u2")}
    given = _pays("Acme", "Buyer", ["u1"], "stale").model_copy(
        update={"subject_id": acme.id, "object_id": buyer.id}
    )
    held = given.model_copy(
        update={"id": resolved_claim_id(given), "text_unit_ids": ["u2"]}
    )
    asked: list[list[str]] = []

    def unit_records(suffix, unit_ids, text_units):
        asked.append(list(unit_ids))
        return (
            {u: records[u] for u in unit_ids},
            {u: statements[u] for u in unit_ids},
        )

    mgr._unit_records = unit_records
    mgr._read_held = lambda suffix, ids: (
        [],
        [],
        [held] if held.id in ids else [],
    )

    derived = mgr.derive("default", [acme, buyer], [], [given])

    assert asked == [["u1"], ["u2"]]
    (claim,) = [c for c in derived.claims if c.id == held.id]
    assert sorted(claim.text_unit_ids or []) == ["u1", "u2"]
