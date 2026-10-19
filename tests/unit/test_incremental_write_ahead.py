# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write-ahead registry records make an interrupted incremental run repairable.

Before an incremental run removes or writes anything for its delta, every new
and changed document is recorded PENDING with a content hash no document has
and its stored + planned text units. A run interrupted anywhere after that leaves
those records, so the next run re-extracts a document that still exists (even
when its content went back to the indexed version) and removes everything the
interrupted run may have written for one that is gone. Every interruption
point below must converge to a fresh full build of the follow-up corpus.
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.incremental_runs import (
    POINTS,
    Corpus,
    IncrementalRun,
    Killed,
    document,
    expected_state,
)
from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
from unified_kg_rag.domain.ingestion.delta_detector import detect_delta, document_doc_id
from unified_kg_rag.domain.models import (
    PENDING_CONTENT_HASH,
    DocStatus,
    DocStatusRecord,
    DocumentDelta,
    DocumentLineage,
)
from unified_kg_rag.ports import IndexingStats

pytestmark = pytest.mark.unit

# R0 is indexed; R1 changes b and c, deletes d and g and adds e and f, around
# shared entities (Vendor, Carrier, Buyer appear in several documents).
R0: Corpus = {
    "/c/a.txt": [[("Vendor", "Depot", 1)]],
    "/c/b.txt": [[("Vendor", "Carrier", 2)], [("Depot", "Carrier", 1)]],
    "/c/c.txt": [[("Buyer", "Bank", 1)]],
    "/c/d.txt": [[("Vendor", "Buyer", 3)]],
    "/c/g.txt": [[("Depot", "Bank", 2)]],
}
R1: Corpus = {
    "/c/a.txt": R0["/c/a.txt"],
    "/c/b.txt": [[("Vendor", "Carrier", 2)], [("Carrier", "Bank", 2)]],
    "/c/c.txt": [[("Buyer", "Depot", 1)]],
    "/c/e.txt": [[("Vendor", "Bank", 1)]],
    "/c/f.txt": [[("Carrier", "Buyer", 2)]],
}
FOLLOW_UPS: dict[str, Corpus] = {
    "same": R1,
    # The changed documents went back to the content R0 indexed.
    "reverted": {**R1, "/c/b.txt": R0["/c/b.txt"], "/c/c.txt": R0["/c/c.txt"]},
    # The documents R1 added are gone again.
    "new_docs_removed": {
        p: c for p, c in R1.items() if p not in ("/c/e.txt", "/c/f.txt")
    },
}


def _assert_converged(harness: IncrementalRun, corpus: Corpus) -> None:
    assert harness.state() == expected_state(corpus)
    assert harness.registry_problems(corpus) == []
    # A further run over the same corpus has nothing to do.
    harness.run(corpus)
    assert harness.extracted == []
    assert harness.state() == expected_state(corpus)


# Random corpora, points, k and follow-ups are in
# tests/property/test_incremental_interruption_properties.py, and every point
# runs through each registry in test_doc_status_conformance.py. This keeps one
# named case per point that also proves the point fires on this corpus.
@pytest.mark.parametrize("point", POINTS)
def test_interrupted_run_converges_to_a_fresh_build(point: str) -> None:
    harness = IncrementalRun()
    harness.run(R0)
    assert harness.state() == expected_state(R0)

    assert harness.run(R1, interrupt=point), f"{point} did not fire"
    corpus = FOLLOW_UPS["reverted"]
    harness.run(corpus)

    _assert_converged(harness, corpus)


def test_crash_after_prune_then_revert_re_extracts_the_document() -> None:
    harness = IncrementalRun()
    harness.run(R0)
    edited = {**R0, "/c/b.txt": [[("Vendor", "Bank", 1)]]}

    assert harness.run(edited, interrupt="after_removal")
    # b's old artifacts are gone; its record is the write-ahead one.
    b = harness.store.get(document_doc_id(document("/c/b.txt", R0["/c/b.txt"])))
    assert b is not None
    assert b.status is DocStatus.PENDING
    assert b.content_hash == PENDING_CONTENT_HASH

    # Back to the indexed text: still re-extracted, not read as unchanged.
    harness.run(R0)
    assert harness.extracted == ["/c/b.txt"]
    _assert_converged(harness, R0)


def test_crash_after_writes_then_new_docs_removed_leaves_no_orphans() -> None:
    harness = IncrementalRun()
    harness.run(R0)

    assert harness.run(R1, interrupt="before_record")
    assert (
        harness.state()["vector_text_units"] >= expected_state(R1)["vector_text_units"]
    )

    corpus = FOLLOW_UPS["new_docs_removed"]
    harness.run(corpus)
    _assert_converged(harness, corpus)


def test_deleted_docs_restored_after_an_interrupted_removal_are_reindexed() -> None:
    # R1 deletes d and g; the run dies after the removal deleted the vector
    # copies and before the graph ones. Restored unchanged, d and g were read
    # as unchanged (their records still said PROCESSED) and their text
    # units, vector entities and claims stayed lost.
    harness = IncrementalRun()
    harness.run(R0)

    assert harness.run(R1, interrupt="mid_removal", k=2)
    corpus = {**R1, "/c/d.txt": R0["/c/d.txt"], "/c/g.txt": R0["/c/g.txt"]}
    harness.run(corpus)

    assert {"/c/d.txt", "/c/g.txt"} <= set(harness.extracted)
    _assert_converged(harness, corpus)


def test_a_deleted_doc_restored_after_a_failed_removal_is_reindexed() -> None:
    # Removing d deletes its vector copies, then the graph delete fails: the
    # removal reports it and keeps d's record. d restored unchanged must be
    # indexed again, not read as unchanged over its deleted copies.
    harness = IncrementalRun()
    before: Corpus = {
        "/c/a.txt": [[("Vendor", "Depot", 1)]],
        "/c/d.txt": [[("Buyer", "Bank", 1)]],
    }
    harness.run(before)

    def failing(ids: list[str], suffix: str | None = None) -> IndexingStats:
        stats = IndexingStats(total_items=len(ids))
        stats.add_error("timed out", count=len(ids))
        return stats

    inc = IncrementalIndexer(harness.store, harness.manager)
    delta, _ = detect_delta([document("/c/a.txt", before["/c/a.txt"])], inc.doc_status)
    original = harness.graph.delete_by_id
    harness.graph.delete_by_id = failing  # type: ignore[method-assign]
    try:
        assert inc.remove_changed_and_deleted(delta) is False
    finally:
        harness.graph.delete_by_id = original  # type: ignore[method-assign]
    record = harness.store.get(delta.deleted[0])
    assert record is not None and record.status is DocStatus.PENDING

    harness.run(before)
    assert harness.extracted == ["/c/d.txt"]
    _assert_converged(harness, before)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_crash_mid_commit_leaves_the_rest_pending(k: int) -> None:
    harness = IncrementalRun()
    harness.run(R0)

    assert harness.run(R1, interrupt="mid_commit", k=k)
    records = {r.file_path: r for r in harness.store.list_all()}
    processed = [p for p, r in records.items() if r.status is DocStatus.PROCESSED]
    pending = [p for p, r in records.items() if r.status is DocStatus.PENDING]
    # The commit wrote k-1 of the four new/changed records before dying.
    assert len(pending) == 4 - (k - 1)
    assert "/c/a.txt" in processed

    harness.run(R1)
    assert sorted(harness.extracted) == sorted(pending)
    _assert_converged(harness, R1)


def test_write_ahead_record_holds_old_and_planned_lineage() -> None:
    store = FakeDocStatusStore()
    store.put(
        DocStatusRecord(
            doc_id="ns#changed",
            content_hash="h1",
            status=DocStatus.PROCESSED,
            suffix="tenant",
            scope="ns|src",
            file_path="b.txt",
            text_unit_ids=["t-old", "t-shared"],
        )
    )
    inc = IncrementalIndexer(store, None, scope="ns|src")  # type: ignore[arg-type]
    inc.write_ahead(
        DocumentDelta(changed=["ns#changed"], new=["ns#new"]),
        [
            DocumentLineage(
                doc_id="ns#changed",
                suffix="tenant",
                file_path="b.txt",
                text_unit_ids=["t-shared", "t-new"],
            ),
            DocumentLineage(
                doc_id="ns#new",
                suffix="tenant",
                file_path="e.txt",
                text_unit_ids=["t-x"],
            ),
        ],
    )

    changed, new = store.get("ns#changed"), store.get("ns#new")
    assert changed is not None and new is not None
    assert changed.status is DocStatus.PENDING
    assert changed.content_hash == PENDING_CONTENT_HASH
    assert changed.text_unit_ids == ["t-new", "t-old", "t-shared"]
    assert (changed.suffix, changed.scope, changed.file_path) == (
        "tenant",
        "ns|src",
        "b.txt",
    )
    assert (new.status, new.text_unit_ids, new.scope) == (
        DocStatus.PENDING,
        ["t-x"],
        "ns|src",
    )
    # Neither can read as unchanged, whatever its content.
    delta = store.diff({"ns#changed": "h1", "ns#new": "anything"}, scope="ns|src")
    assert sorted(delta.changed) == ["ns#changed", "ns#new"]
    # Gone, both read as deleted in their scope, and only there.
    assert sorted(store.diff({}, scope="ns|src").deleted) == ["ns#changed", "ns#new"]
    assert store.diff({}, scope="ns|other").deleted == []


def test_scopeless_run_keeps_the_stored_scope() -> None:
    store = FakeDocStatusStore()
    store.put(DocStatusRecord(doc_id="d", content_hash="h", scope="ns|src"))
    inc = IncrementalIndexer(store, None)  # type: ignore[arg-type]
    inc.write_ahead(DocumentDelta(changed=["d"]), [])
    record = store.get("d")
    assert record is not None and record.scope == "ns|src"


def test_write_ahead_does_not_count_toward_max_document_failures() -> None:
    harness = IncrementalRun()
    harness.run(R0)
    a = document("/c/a.txt", R0["/c/a.txt"])
    doc_id = document_doc_id(a)
    failed = harness.store.get(doc_id)
    assert failed is not None
    harness.store.put(
        failed.model_copy(update={"status": DocStatus.FAILED, "failure_count": 2})
    )

    # The run fails a again: the count continues from the record stored
    # before the write-ahead one, not from the PENDING record.
    inc = IncrementalIndexer(harness.store, harness.manager)
    delta, fingerprints = detect_delta([a], inc.doc_status, max_failures=3)
    assert delta.changed == [doc_id]
    lineage = DocumentLineage(doc_id=doc_id, file_path=a.file_path)
    inc.write_ahead(delta, [lineage])
    pending = harness.store.get(doc_id)
    assert pending is not None
    assert (pending.status, pending.failure_count) == (DocStatus.PENDING, 0)
    inc.commit([lineage], fingerprints, failed_doc_ids={doc_id})
    record = harness.store.get(doc_id)
    assert record is not None
    assert (record.status, record.failure_count) == (DocStatus.FAILED, 3)

    # An interrupted run is no failure: the PENDING record is retried even
    # though the count had reached the limit.
    harness.store.put(pending)
    delta, _ = detect_delta([a], harness.store, max_failures=3)
    assert delta.changed == [doc_id]


def test_commit_reads_and_writes_the_registry_in_batches(mocker) -> None:
    harness = IncrementalRun()
    get = mocker.spy(harness.store, "get")
    get_many = mocker.spy(harness.store, "get_many")
    put_many = mocker.spy(harness.store, "put_many")

    harness.run(R0)

    get.assert_not_called()
    # One read each by open_attempts, write_ahead and close_attempts, nothing
    # re-read by the commit.
    assert [len(c.args[0]) for c in get_many.call_args_list if c.args[0]] == [
        len(R0)
    ] * 3
    # The attempt records, the write-ahead records, then the committed ones
    # (they have no open attempt left to close).
    assert [len(list(c.args[0])) for c in put_many.call_args_list] == [len(R0)] * 3


@pytest.mark.parametrize("outcome", ["killed", "failed"])
def test_retried_shared_strip_repairs_the_store_an_earlier_one_missed(
    outcome: str,
) -> None:
    """Deleting b strips its text unit from Vendor, which a also cites. When
    only the graph copy got stripped (the vector write died or failed), the
    retry reads the stripped graph copy back: it must still rewrite the
    vector copy, which cites the removed text unit."""
    harness = IncrementalRun()
    before: Corpus = {
        "/c/a.txt": [[("Vendor", "Depot", 1)]],
        "/c/b.txt": [[("Vendor", "Carrier", 1)]],
    }
    after = {"/c/a.txt": before["/c/a.txt"]}
    harness.run(before)

    def broken(name: str):
        def upsert(items):
            if outcome == "killed":
                raise Killed(name)
            raise RuntimeError("vector store unavailable")

        return upsert

    # The entity writes of both stores are the last step of a strip (a
    # retry finds the edges through the entities still citing the unit).
    originals = {
        name: getattr(harness.vector, name)
        for name in ("update_entity_lineage", "upsert_entities")
    }
    for name in originals:
        setattr(harness.vector, name, broken(name))
    inc = IncrementalIndexer(harness.store, harness.manager)
    delta, _ = detect_delta([document(p, c) for p, c in after.items()], inc.doc_status)
    try:
        assert inc.remove_changed_and_deleted(delta) is False
    except Killed:
        pass
    for name, original in originals.items():
        setattr(harness.vector, name, original)
    # The graph copy is stripped, the vector copy is not.
    assert harness.state()["graph_entities"] == expected_state(after)["graph_entities"]
    assert (
        harness.state()["vector_entities"] != expected_state(after)["vector_entities"]
    )

    harness.run(after)

    _assert_converged(harness, after)


class _RemovalRecorder:
    """Records the text units each removal asks the stores to remove."""

    def __init__(self) -> None:
        self.removed: dict[str, list[str]] = {}

    def remove_text_units(self, suffix, text_unit_ids):
        self.removed[suffix] = list(text_unit_ids)
        return {}


def test_removal_of_in_flight_documents_uses_their_stored_records() -> None:
    # The write-ahead PENDING records also list what this run writes after the
    # removal. A new document has no stored record: it removes nothing (its
    # pending "t-x" is not removed with the deleted document's units). A
    # changed document is removed by its stored record ("t-y"), not the
    # pending one (which also lists the "t-z" it is about to write).
    store = FakeDocStatusStore()
    deleted, changed, new = "doc-deleted", "doc-changed", "doc-new"
    for doc_id, unit in ((deleted, "t-w"), (changed, "t-y")):
        store.put(
            DocStatusRecord(
                doc_id=doc_id,
                content_hash="h",
                status=DocStatus.PROCESSED,
                text_unit_ids=[unit],
            )
        )
    manager = _RemovalRecorder()
    inc = IncrementalIndexer(store, indexing_manager=manager)  # type: ignore[arg-type]
    inc.write_ahead(
        DocumentDelta(new=[new], changed=[changed], deleted=[deleted]),
        [
            DocumentLineage(doc_id=new, text_unit_ids=["t-x"]),
            DocumentLineage(doc_id=changed, text_unit_ids=["t-y", "t-z"]),
        ],
    )

    assert inc.remove_obsolete_artifacts([changed, deleted])

    assert manager.removed == {"default": ["t-w", "t-y"]}


def test_a_deleted_documents_write_ahead_resets_its_failure_count() -> None:
    # A PENDING record has failure_count 0 (it is no failure of its own);
    # the write-ahead of a deleted document kept the stored count.
    harness = IncrementalRun()
    before: Corpus = {
        "/c/a.txt": [[("Vendor", "Depot", 1)]],
        "/c/d.txt": [[("Buyer", "Bank", 1)]],
    }
    harness.run(before)
    d_id = document_doc_id(document("/c/d.txt", before["/c/d.txt"]))
    stored = harness.store.get(d_id)
    assert stored is not None
    harness.store.put(
        stored.model_copy(update={"status": DocStatus.FAILED, "failure_count": 2})
    )

    def failing(ids: list[str], suffix: str | None = None) -> IndexingStats:
        stats = IndexingStats(total_items=len(ids))
        stats.add_error("timed out", count=len(ids))
        return stats

    inc = IncrementalIndexer(harness.store, harness.manager)
    delta, _ = detect_delta([document("/c/a.txt", before["/c/a.txt"])], inc.doc_status)
    assert delta.deleted == [d_id]
    original = harness.graph.delete_by_id
    harness.graph.delete_by_id = failing  # type: ignore[method-assign]
    try:
        assert inc.remove_changed_and_deleted(delta) is False
    finally:
        harness.graph.delete_by_id = original  # type: ignore[method-assign]
    record = harness.store.get(d_id)
    assert record is not None
    assert (record.status, record.failure_count) == (DocStatus.PENDING, 0)


def test_a_lost_lease_writes_no_deleted_document_ahead() -> None:
    # The deleted documents' write-ahead is a registry write: a run that
    # lost its lease must not record them PENDING over the new holder's.
    from unified_kg_rag.shared import NamespaceLeaseError
    from unified_kg_rag.shared.write_fence import (
        install_write_fence,
        remove_write_fence,
    )

    class _Lost:
        def check(self) -> None:
            raise NamespaceLeaseError("lost")

    harness = IncrementalRun()
    before: Corpus = {
        "/c/a.txt": [[("Vendor", "Depot", 1)]],
        "/c/d.txt": [[("Buyer", "Bank", 1)]],
    }
    harness.run(before)
    inc = IncrementalIndexer(harness.store, harness.manager)
    delta, _ = detect_delta([document("/c/a.txt", before["/c/a.txt"])], inc.doc_status)
    token = install_write_fence(_Lost())
    try:
        with pytest.raises(NamespaceLeaseError):
            inc.remove_changed_and_deleted(delta)
    finally:
        remove_write_fence(token)
    record = harness.store.get(delta.deleted[0])
    assert record is not None and record.status is DocStatus.PROCESSED
