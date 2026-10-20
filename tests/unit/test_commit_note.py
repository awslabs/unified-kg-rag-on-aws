# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The incremental commit's write-ahead note (``pending-commit``), AWS-free.

Before its first write a commit notes the stored entities its derivation
split, re-keyed or rewired; the next commit of the suffix derives them,
their stored edges and the claims about them again, and the note goes once
the old ids are deleted (see ``IndexingManager.note_pending_commit``).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.derivation import Derived
from unified_kg_rag.domain.models import (
    Claim,
    Config,
    DocumentLineage,
    Entity,
    Relationship,
)
from unified_kg_rag.ports.indexer import IndexingStats
from unified_kg_rag.shared import DataProcessingError, NamespaceLeaseError

pytestmark = pytest.mark.unit

NOTE = "pending-commit"


def _manager() -> tuple[IndexingManager, FakeGraphStore, FakeVectorStore]:
    config = Config()
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    return (
        IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph),
        graph,
        vector,
    )


def _entity(entity_id: str, name: str) -> Entity:
    return Entity(id=entity_id, name=name, type="ORGANIZATION", text_unit_ids=["t1"])


def _failed(count: int = 1) -> IndexingStats:
    stats = IndexingStats(total_items=count)
    stats.add_error("store down", count=count)
    return stats


def _note(vector: FakeVectorStore) -> dict[str, Any] | None:
    return vector.read_extraction_records([NOTE]).get(NOTE)


def test_a_note_that_cannot_be_written_raises(mocker) -> None:
    manager, _, vector = _manager()
    manager._commit_entities["default"] = {"e1"}
    mocker.patch.object(vector, "write_extraction_records", return_value=_failed())

    with pytest.raises(DataProcessingError):
        manager.note_pending_commit()


def test_a_commit_writes_nothing_when_its_note_fails(mocker) -> None:
    # The note is the only record of what the commit decides: without it a
    # killed commit could not be finished, so nothing is written either.
    manager, _, vector = _manager()
    incremental = IncrementalIndexer(FakeDocStatusStore(), manager)
    mocker.patch.object(
        manager,
        "merge_with_existing_graph",
        side_effect=lambda *a, **k: manager._commit_entities.update({"default": {"e1"}})
        or IndexingManager.merge_with_existing_graph(manager, *a, **k),
    )
    mocker.patch.object(vector, "write_extraction_records", return_value=_failed())
    index_delta = mocker.spy(manager, "index_delta")

    with pytest.raises(DataProcessingError):
        incremental.commit(
            lineages=[DocumentLineage(doc_id="doc-a")],
            fingerprints={"doc-a": "hash-a"},
            entities=[_entity("e-a", "Acme")],
        )

    assert index_delta.call_count == 0


def test_a_note_carries_the_ids_of_the_note_it_finishes() -> None:
    manager, _, vector = _manager()
    manager._noted_commit_entities["default"] = {"e-earlier"}
    manager._commit_entities["default"] = {"e-now"}

    manager.note_pending_commit()

    assert set((_note(vector) or {})["entity_ids"]) == {"e-earlier", "e-now"}


def test_a_read_note_is_deleted_with_this_commits(mocker) -> None:
    # A note read (an earlier commit's) is the suffix's to delete once this
    # commit retired its old ids.
    manager, _, vector = _manager()
    vector.write_extraction_records({NOTE: {"entity_ids": ["e1"]}})

    assert manager._read_commit_note("default") == {"e1"}
    manager.clear_pending_commit()

    assert _note(vector) is None


def test_a_discarded_commit_keeps_its_note() -> None:
    manager, _, vector = _manager()
    vector.write_extraction_records({NOTE: {"entity_ids": ["e1"]}})
    manager._read_commit_note("default")

    manager.discard_retired_ids()
    manager.clear_pending_commit()

    assert _note(vector) is not None


def test_retiring_the_old_ids_forgets_the_noted_entities() -> None:
    manager, _, _ = _manager()
    manager._commit_entities["default"] = {"e1"}
    manager._noted_commit_entities["default"] = {"e2"}

    manager.delete_retired_ids()

    assert manager._commit_entities == {}
    assert manager._noted_commit_entities == {}


def _with_run_state(manager: IndexingManager) -> IndexingManager:
    """``manager`` holding something in every per-run field."""
    by_suffix = {
        "_rekeyed": {"e1": "e2"},
        "_split_from": {"e1": "e2"},
        "_derived_entities": {"e1": _entity("e1", "Acme")},
        **dict.fromkeys(
            (
                "_commit_entities",
                "_noted_commit_entities",
                "_renamed",
                "_dropped",
                "_vector_only",
                "_removed_record_keys",
            ),
            {"e1"},
        ),
    }
    for field, value in by_suffix.items():
        getattr(manager, field)["default"] = value
    for field in ("_fresh", "_rank_only", "_commit_note_suffixes", "outdated_reports"):
        getattr(manager, field).add("default")
    return manager


_DECISIONS = (
    "_rekeyed",
    "_split_from",
    "_commit_entities",
    "_noted_commit_entities",
    "_renamed",
    "_dropped",
    "_vector_only",
    "_derived_entities",
    "_fresh",
    "_rank_only",
)


@pytest.mark.parametrize("end", ["delete_retired_ids", "discard_retired_ids"])
def test_a_commit_ends_without_its_derivation_decisions(end: str) -> None:
    # Whichever way it ends; the removals' findings stay for the commit's
    # drift measure and record collection.
    manager = _with_run_state(_manager()[0])

    getattr(manager, end)()

    assert {f: getattr(manager, f) for f in _DECISIONS} == {
        f: type(getattr(manager, f))() for f in _DECISIONS
    }
    assert manager.outdated_reports == {"default"}
    assert manager._removed_record_keys == {"default": {"e1"}}
    # A discarded commit's note is not deleted (a retry finishes from it).
    assert manager._commit_note_suffixes == (
        {"default"} if end == "delete_retired_ids" else set()
    )


def test_a_run_starts_without_anything_an_earlier_run_left() -> None:
    manager = _with_run_state(_manager()[0])

    manager.reset_run_state()

    for field in (*_DECISIONS, "_commit_note_suffixes", "outdated_reports"):
        assert not getattr(manager, field), field
    assert manager._removed_record_keys == {}


def test_a_suffix_the_run_only_removed_in_reads_its_commit_note() -> None:
    # No delta in "tenant" (not the run's own suffix either): the removal's
    # record keys name it, so the commit still finishes the note there.
    manager, _, vector = _manager()
    vector.write_extraction_records({NOTE: {"entity_ids": ["e1"]}}, suffix="tenant")
    manager._removed_record_keys["tenant"] = {"record-1"}

    manager.merge_with_existing_graph(None, None)

    assert manager._noted_commit_entities == {"tenant": {"e1"}}
    assert manager._commit_note_suffixes == {"tenant"}


def test_a_noted_vector_only_entity_and_edge_both_go() -> None:
    # A killed commit wrote "e-v" and the edge "r-v" from the noted "e1" to it
    # to the vector store only; "r-g" has its graph copy too.
    manager, graph, vector = _manager()
    e1, e2, ev = _entity("e1", "Acme"), _entity("e2", "Buyer"), _entity("e-v", "Depot")
    edges = [
        Relationship(
            id=f"r-{end}",
            source_id="e1",
            target_id=target.id,
            source_name="Acme",
            target_name=target.name,
            text_unit_ids=["t1"],
        )
        for end, target in (("g", e2), ("v", ev))
    ]
    graph.upsert_entities([e1, e2])
    graph.upsert_relationships(edges[:1])
    vector.upsert_entities([e1, e2, ev])
    vector.upsert_relationships(edges)
    vector.write_extraction_records({NOTE: {"entity_ids": ["e1", "e-v"]}})

    manager.merge_with_existing_graph(None, None)

    assert manager._vector_only == {"default": {"e-v"}}
    assert manager._dropped["default"] >= {"e-v", "r-v"}
    assert "r-g" not in manager._dropped["default"]


def test_clearing_the_note_stops_on_a_lost_lease(mocker) -> None:
    manager, _, vector = _manager()
    manager._commit_note_suffixes.add("default")
    mocker.patch.object(
        vector,
        "delete_extraction_records",
        side_effect=NamespaceLeaseError("lease lost"),
    )

    with pytest.raises(NamespaceLeaseError):
        manager.clear_pending_commit()


def test_a_failed_retire_keeps_the_note(mocker) -> None:
    # The old ids are still stored: the next commit derives the noted
    # entities again and retires them then.
    manager, _, vector = _manager()
    incremental = IncrementalIndexer(FakeDocStatusStore(), manager)
    vector.write_extraction_records({NOTE: {"entity_ids": ["e1"]}})
    mocker.patch.object(
        manager, "delete_retired_ids", return_value={"retire": _failed()}
    )

    incremental.commit(
        lineages=[DocumentLineage(doc_id="doc-a")],
        fingerprints={"doc-a": "hash-a"},
        entities=[_entity("e-a", "Acme")],
    )

    assert _note(vector) is not None


@pytest.mark.parametrize(
    ("derived", "expected"),
    [
        pytest.param(
            Derived(entities=[_entity("e-new", "Acme")], id_remap={"e-old": "e-new"}),
            {"e-old", "e-new"},
            id="re-key",
        ),
        pytest.param(
            Derived(
                entities=[_entity("e-a", "Acme"), _entity("e-part", "Acme Inc")],
                split_from={"e-part": "e-a"},
            ),
            {"e-part", "e-a"},
            id="split",
        ),
        pytest.param(
            Derived(entities=[_entity("e-a", "Acme")], rewired=["e-a"]),
            {"e-a"},
            id="rewired",
        ),
    ],
)
def test_the_note_names_each_kind_of_decision(
    mocker, derived: Derived, expected: set[str]
) -> None:
    manager, graph, _ = _manager()
    graph.upsert_entities([_entity("e-a", "Acme")])
    mocker.patch.object(manager, "derive", return_value=derived)
    mocker.patch.object(
        manager, "_finish", side_effect=lambda s, e, r, **k: (list(e), list(r))
    )

    manager._merge_suffix("default", [_entity("e-a", "Acme")], [])

    assert manager._commit_entities["default"] == expected


def _noted_merge(mocker, delta: list[Entity]):
    """Merge ``delta`` under a stored note naming e-noted, whose stored edge
    joins it to e-other; returns the spies of derive and _finish."""
    manager, graph, vector = _manager()
    graph.upsert_entities([_entity("e-noted", "Noted"), _entity("e-other", "Other")])
    graph.upsert_relationships(
        [
            Relationship(
                id="r-1",
                source_id="e-noted",
                target_id="e-other",
                source_name="Noted",
                target_name="Other",
                text_unit_ids=["t1"],
            )
        ]
    )
    vector.write_extraction_records({NOTE: {"entity_ids": ["e-noted"]}})
    derive = mocker.patch.object(
        manager,
        "derive",
        side_effect=lambda s, e, r, *a, **k: Derived(
            entities=list(e), relationships=list(r)
        ),
    )
    finish = mocker.patch.object(
        manager, "_finish", side_effect=lambda s, e, r, **k: (list(e), list(r))
    )
    manager._merge_suffix("default", delta, [])
    return manager, derive, finish


@pytest.mark.parametrize(
    "delta",
    [
        pytest.param([_entity("e-a", "Acme")], id="with-a-delta"),
        pytest.param([], id="without-one"),
    ],
)
def test_a_noted_entity_and_its_edges_are_derived_again(mocker, delta) -> None:
    manager, derive, finish = _noted_merge(mocker, delta)

    entities, relationships = derive.call_args.args[1], derive.call_args.args[2]
    assert "e-noted" in {e.id for e in entities}
    # The edge's other endpoint too: its rank follows the edge.
    assert "e-other" in {e.id for e in entities}
    assert "r-1" in {r.id for r in relationships}
    # Its stored links are all among the given edges: they do not count.
    assert "e-noted" in finish.call_args.kwargs["rewired"]
    assert manager._noted_commit_entities["default"] == {"e-noted"}


def test_the_claims_about_a_noted_entity_are_derived_again(mocker) -> None:
    manager, _, vector = _manager()
    vector.upsert_claims(
        [
            Claim(
                id="c-1",
                subject_id="e-noted",
                subject_name="Noted",
                object_name="100",
                type="PAYS",
                text_unit_ids=["t1"],
            )
        ]
    )
    manager._noted_commit_entities["default"] = {"e-noted"}
    derive_claims = mocker.patch.object(
        manager, "_derive_claims", side_effect=lambda s, claims, *a, **k: claims
    )

    claims = manager.merge_with_stored_claims(None)

    assert derive_claims.call_count == 1
    assert [c.id for c in claims or []] == ["c-1"]


@pytest.mark.parametrize(
    "delta",
    [
        pytest.param([_entity("e-a", "Acme")], id="with-a-delta"),
        pytest.param([], id="without-one"),
    ],
)
def test_a_noted_entity_without_edges_is_derived_again(mocker, delta) -> None:
    manager, graph, vector = _manager()
    graph.upsert_entities([_entity("e-noted", "Noted")])
    vector.write_extraction_records({NOTE: {"entity_ids": ["e-noted"]}})
    derive = mocker.patch.object(
        manager,
        "derive",
        side_effect=lambda s, e, r, *a, **k: Derived(
            entities=list(e), relationships=list(r)
        ),
    )
    mocker.patch.object(
        manager, "_finish", side_effect=lambda s, e, r, **k: (list(e), list(r))
    )

    manager._merge_suffix("default", delta, [])

    assert "e-noted" in {e.id for e in derive.call_args.args[1]}
