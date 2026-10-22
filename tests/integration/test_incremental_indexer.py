# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end incremental-indexing tests using in-memory fakes (M2).

Exercises the full add -> change -> delete cycle through the
:class:`IncrementalIndexer` and a real :class:`IndexingManager` over the
in-memory graph and vector stores and the fake doc-status store, with no AWS
clients involved. Each document's registry record lists its text units; a
removal finds what cites them in the stores, so deletion removes only what a
document stated *exclusively*.
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.delta_detector import (
    compute_doc_id,
    detect_delta,
    filter_documents_to_process,
)
from unified_kg_rag.domain.models import (
    Claim,
    Config,
    DocStatus,
    Document,
    DocumentLineage,
    Entity,
    TextUnit,
)

pytestmark = pytest.mark.integration


def _doc(path: str, text: str) -> Document:
    return Document(
        page_content=text,
        document_id="x",
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
    )


def _unit_id(path: str) -> str:
    return f"tu-{path.strip('/')}"


def _lineage(path: str) -> DocumentLineage:
    return DocumentLineage(doc_id=compute_doc_id(path), text_unit_ids=[_unit_id(path)])


def _unit(path: str) -> TextUnit:
    return TextUnit(id=_unit_id(path), text="...", document_ids=["x"])


def _entity(entity_id: str, name: str, path: str) -> Entity:
    return Entity(id=entity_id, name=name, text_unit_ids=[_unit_id(path)])


def _commit(inc, fps, path: str, entities=(), claims=()) -> None:
    """Commit ``path``'s one text unit with the artifacts stated in it."""
    inc.commit(
        lineages=[_lineage(path)],
        fingerprints=fps,
        text_units=[_unit(path)],
        entities=[_entity(i, n, path) for i, n in entities],
        claims=list(claims),
    )


class _Rig:
    def __init__(self) -> None:
        config = Config()
        self.graph = FakeGraphStore()
        self.vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
        self.manager = IndexingManager(
            config=config, vector_indexer=self.vector, graph_indexer=self.graph
        )
        self.store = FakeDocStatusStore()
        self.inc = IncrementalIndexer(self.store, self.manager)

    def entity_units(self) -> dict[str, list[str]]:
        entities = self.graph.read_entities(sorted(self.graph.ids("entities")))
        return {e.id: sorted(e.text_unit_ids or []) for e in entities}


@pytest.fixture
def rig() -> _Rig:
    return _Rig()


def test_first_run_marks_all_new_and_records_registry(rig) -> None:
    docs = [_doc("/a.txt", "A"), _doc("/b.txt", "B")]

    delta, fingerprints = detect_delta(docs, rig.inc.doc_status)
    assert set(delta.new) == set(fingerprints)
    assert not delta.deleted

    to_process = filter_documents_to_process(docs, delta)
    assert len(to_process) == 2

    rig.inc.commit(
        lineages=[_lineage("/a.txt"), _lineage("/b.txt")],
        fingerprints=fingerprints,
        text_units=[_unit("/a.txt"), _unit("/b.txt")],
        entities=[_entity("ea", "A", "/a.txt"), _entity("eb", "B", "/b.txt")],
    )
    assert rig.graph.ids("entities") == {"ea", "eb"}
    assert rig.vector.ids("text_units") == {"tu-a.txt", "tu-b.txt"}
    assert {r.doc_id: r.text_unit_ids for r in rig.store.list_all()} == {
        compute_doc_id("/a.txt"): ["tu-a.txt"],
        compute_doc_id("/b.txt"): ["tu-b.txt"],
    }


def test_second_run_unchanged_is_noop_delta(rig) -> None:
    docs = [_doc("/a.txt", "A")]
    _, fps = detect_delta(docs, rig.inc.doc_status)
    _commit(rig.inc, fps, "/a.txt", [("e1", "A")])

    delta2, _ = detect_delta(docs, rig.inc.doc_status)
    assert delta2.is_empty
    assert filter_documents_to_process(docs, delta2) == []


def test_changed_document_is_reprocessed(rig) -> None:
    docs = [_doc("/a.txt", "original")]
    _, fps = detect_delta(docs, rig.inc.doc_status)
    _commit(rig.inc, fps, "/a.txt", [("e1", "A")])

    edited = [_doc("/a.txt", "EDITED")]
    delta, _ = detect_delta(edited, rig.inc.doc_status)
    assert delta.changed == [compute_doc_id("/a.txt")]
    assert len(filter_documents_to_process(edited, delta)) == 1


def test_changed_document_prunes_stale_artifacts(rig) -> None:
    # First run: doc a produced entities e1, e2.
    _, fps = detect_delta([_doc("/a.txt", "v1")], rig.inc.doc_status)
    _commit(rig.inc, fps, "/a.txt", [("e1", "A"), ("e2", "B")])

    # Edit: every old artifact of the changed doc is pruned before its
    # re-extraction is committed.
    edited = [_doc("/a.txt", "v2")]
    delta, _ = detect_delta(edited, rig.inc.doc_status)
    assert rig.inc.remove_changed_and_deleted(delta)

    assert rig.graph.ids("entities") == set()
    assert rig.vector.ids("entities") == set()
    assert rig.vector.ids("text_units") == set()


def test_changed_document_does_not_prune_entity_shared_with_survivor(rig) -> None:
    # A changed doc and a surviving doc both state 'shared'; pruning the
    # changed doc's stale artifacts must NOT drop the shared entity (still
    # stated by the survivor), only strip the changed doc's text unit from it.
    a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
    _, fps = detect_delta([a, b], rig.inc.doc_status)
    _commit(rig.inc, fps, "/a.txt", [("shared", "S"), ("only_a", "A")])
    _commit(rig.inc, fps, "/b.txt", [("shared", "S")])
    assert rig.entity_units()["shared"] == ["tu-a.txt", "tu-b.txt"]

    edited = [_doc("/a.txt", "EDITED"), b]
    delta, _ = detect_delta(edited, rig.inc.doc_status)
    assert rig.inc.remove_changed_and_deleted(delta)

    assert rig.entity_units() == {"shared": ["tu-b.txt"]}


def test_deleted_document_removes_exclusive_artifacts(rig) -> None:
    a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
    _, fps = detect_delta([a, b], rig.inc.doc_status)
    _commit(rig.inc, fps, "/a.txt", [("ea", "A")])
    _commit(rig.inc, fps, "/b.txt", [("eb", "B")])

    delta, _ = detect_delta([a], rig.inc.doc_status)
    assert delta.deleted == [compute_doc_id("/b.txt")]

    assert rig.inc.remove_changed_and_deleted(delta)
    assert {r.doc_id for r in rig.store.list_all()} == {compute_doc_id("/a.txt")}
    assert rig.graph.ids("entities") == rig.vector.ids("entities") == {"ea"}
    assert rig.vector.ids("text_units") == {"tu-a.txt"}


def _claim(claim_id: str, subject: str, path: str) -> Claim:
    return Claim(
        id=claim_id,
        subject_id=subject,
        subject_name=subject.upper(),
        object_id="",
        object_name="Goods",
        type="SUPPLIES",
        status="TRUE",
        description=f"{subject} supplies goods.",
        text_unit_ids=[_unit_id(path)],
    )


def test_deleted_document_removes_claim_artifacts(rig) -> None:
    # Claims are first-class indexed artifacts: the removal finds them by the
    # text units they cite, or they would leak as orphans.
    a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
    _, fps = detect_delta([a, b], rig.inc.doc_status)
    _commit(rig.inc, fps, "/a.txt", [("ea", "A")], [_claim("ca", "ea", "/a.txt")])
    _commit(rig.inc, fps, "/b.txt", [("eb", "B")], [_claim("cb", "eb", "/b.txt")])
    assert rig.vector.ids("claims") == {"ca", "cb"}

    delta, _ = detect_delta([a], rig.inc.doc_status)  # delete b
    assert rig.inc.remove_changed_and_deleted(delta)

    assert rig.vector.ids("claims") == {"ca"}


def test_shared_artifacts_are_not_deleted(rig) -> None:
    a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
    _, fps = detect_delta([a, b], rig.inc.doc_status)
    # Both docs state the SAME shared entity id.
    _commit(rig.inc, fps, "/a.txt", [("shared", "S")])
    _commit(rig.inc, fps, "/b.txt", [("shared", "S")])

    delta, _ = detect_delta([a], rig.inc.doc_status)  # delete b
    assert rig.inc.remove_changed_and_deleted(delta)

    # Still stated by a: kept in both stores, citing a's text unit only.
    assert rig.entity_units() == {"shared": ["tu-a.txt"]}
    assert rig.vector.ids("entities") == {"shared"}
    # b's registry record is removed even though its only artifact survives.
    assert {r.doc_id for r in rig.store.list_all()} == {compute_doc_id("/a.txt")}


def _fail(inc: IncrementalIndexer, fps: dict[str, str], path: str) -> None:
    inc.commit([_lineage(path)], fps, failed_doc_ids={compute_doc_id(path)})


def test_a_document_failing_every_run_stops_being_retried(rig) -> None:
    # A document that fails deterministically is retried up to max_failures
    # consecutive FAILED records, then treated as unchanged (still FAILED).
    inc, store = rig.inc, rig.store
    docs = [_doc("/a.txt", "A")]
    doc_id = compute_doc_id("/a.txt")
    for attempt in (1, 2):
        delta, fps = detect_delta(docs, inc.doc_status, max_failures=2)
        assert doc_id in delta.to_process
        _fail(inc, fps, "/a.txt")
        record = store.get(doc_id)
        assert record is not None
        assert record.failure_count == attempt

    delta, _ = detect_delta(docs, inc.doc_status, max_failures=2)

    assert delta.unchanged == [doc_id]
    assert not delta.to_process
    record = store.get(doc_id)
    assert record is not None and record.status.value == "failed"


def test_failure_count_resets_on_success_and_on_new_content(rig) -> None:
    inc, store = rig.inc, rig.store
    doc_id = compute_doc_id("/a.txt")
    _, fps = detect_delta([_doc("/a.txt", "A")], inc.doc_status)
    _fail(inc, fps, "/a.txt")
    _fail(inc, fps, "/a.txt")

    # New content is a new document version: its count starts again.
    _, edited = detect_delta([_doc("/a.txt", "EDITED")], inc.doc_status)
    _fail(inc, edited, "/a.txt")
    record = store.get(doc_id)
    assert record is not None and record.failure_count == 1

    inc.commit([_lineage("/a.txt")], edited)
    record = store.get(doc_id)
    assert record is not None and record.failure_count == 0


def test_a_rebuild_writes_the_documents_it_drops_pending_and_leaves_them(
    rig,
) -> None:
    # A reset's rebuild records them PENDING before it writes, so an
    # interruption leaves no PROCESSED record over stores that no longer
    # hold the document. They are not in flight: the rebuild neither
    # removes nor records them (nor counts an attempt of them).
    _, fps = detect_delta([_doc("/a.txt", "A"), _doc("/b.txt", "B")], rig.store)
    _commit(rig.inc, fps, "/a.txt", [("ea", "A")])
    _commit(rig.inc, fps, "/b.txt", [("eb", "B")])
    dropped = rig.store.get(compute_doc_id("/b.txt"))
    delta, fps = detect_delta([_doc("/a.txt", "A")], rig.store)
    inc = IncrementalIndexer(rig.store, rig.manager)

    inc.write_ahead(delta, [_lineage("/a.txt")], dropped=[dropped])
    pending = rig.store.get(dropped.doc_id)
    assert pending.status is DocStatus.PENDING
    assert pending.text_unit_ids == dropped.text_unit_ids
    inc.record([_lineage("/a.txt")], fps, {})

    assert rig.store.get(dropped.doc_id) == pending
    assert rig.store.get(compute_doc_id("/a.txt")).status is DocStatus.PROCESSED
