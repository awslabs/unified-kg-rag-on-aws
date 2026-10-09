# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write-ahead registry records make an interrupted incremental run repairable.

Before an incremental run removes or writes anything for its delta, every new
and changed document is recorded PENDING with a content hash no document has
and its stored + planned lineage. A run interrupted anywhere after that leaves
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
from unified_kg_rag.domain.ingestion.delta_detector import document_doc_id
from unified_kg_rag.domain.models import (
    PENDING_CONTENT_HASH,
    DocStatus,
    DocStatusRecord,
    DocumentDelta,
    DocumentLineage,
)

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


@pytest.mark.parametrize("follow_up", sorted(FOLLOW_UPS))
@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("k", [1, 2])
def test_interrupted_run_converges_to_a_fresh_build(
    point: str, follow_up: str, k: int
) -> None:
    harness = IncrementalRun()
    harness.run(R0)
    assert harness.state() == expected_state(R0)

    assert harness.run(R1, interrupt=point, k=k), f"{point} did not fire"
    corpus = FOLLOW_UPS[follow_up]
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
            doc_id="changed",
            content_hash="h1",
            status=DocStatus.PROCESSED,
            suffix="tenant",
            scope="ns|src",
            file_path="b.txt",
            entity_ids=["e-old", "e-shared"],
            text_unit_ids=["t-old"],
        )
    )
    inc = IncrementalIndexer(store, None, scope="ns|src")  # type: ignore[arg-type]
    inc.write_ahead(
        DocumentDelta(changed=["changed"], new=["new"]),
        [
            DocumentLineage(
                doc_id="changed",
                suffix="tenant",
                file_path="b.txt",
                entity_ids=["e-shared", "e-new"],
                text_unit_ids=["t-new"],
            ),
            DocumentLineage(
                doc_id="new", suffix="tenant", file_path="e.txt", entity_ids=["e-x"]
            ),
        ],
    )

    changed, new = store.get("changed"), store.get("new")
    assert changed is not None and new is not None
    assert changed.status is DocStatus.PENDING
    assert changed.content_hash == PENDING_CONTENT_HASH
    assert changed.entity_ids == ["e-new", "e-old", "e-shared"]
    assert changed.text_unit_ids == ["t-new", "t-old"]
    assert (changed.suffix, changed.scope, changed.file_path) == (
        "tenant",
        "ns|src",
        "b.txt",
    )
    assert (new.status, new.entity_ids, new.scope) == (
        DocStatus.PENDING,
        ["e-x"],
        "ns|src",
    )
    # Neither can read as unchanged, whatever its content.
    delta = store.diff({"changed": "h1", "new": "anything"}, scope="ns|src")
    assert sorted(delta.changed) == ["changed", "new"]
    # Gone, both read as deleted in their scope, and only there.
    assert sorted(store.diff({}, scope="ns|src").deleted) == ["changed", "new"]
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
    delta, fingerprints = inc.plan([a], max_failures=3)
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
    delta, _ = IncrementalIndexer(harness.store, harness.manager).plan(
        [a], max_failures=3
    )
    assert delta.changed == [doc_id]


def test_commit_reads_and_writes_the_registry_in_batches(mocker) -> None:
    harness = IncrementalRun()
    get = mocker.spy(harness.store, "get")
    get_many = mocker.spy(harness.store, "get_many")
    put_many = mocker.spy(harness.store, "put_many")

    harness.run(R0)

    get.assert_not_called()
    # One read by write_ahead, nothing re-read by the commit.
    assert get_many.call_count == 1
    # The write-ahead records, then the committed ones.
    assert [len(c.args[0]) for c in put_many.call_args_list] == [len(R0)] * 2


def test_custom_store_without_batch_methods_still_works() -> None:
    class _Plain:
        """A DocStatusPort implemented before get_many/put_many existed."""

        def __init__(self) -> None:
            self._fake = FakeDocStatusStore()

        def get(self, doc_id):
            return self._fake.get(doc_id)

        def put(self, record):
            self._fake.put(record)

        def delete(self, doc_id):
            self._fake.delete(doc_id)

        def list_all(self):
            return self._fake.list_all()

        def diff(self, incoming, scope=None):
            return self._fake.diff(incoming, scope)

    harness = IncrementalRun()
    harness.store = _Plain()  # type: ignore[assignment]
    harness.run(R0)
    assert harness.run(R1, interrupt="after_removal")
    harness.run(FOLLOW_UPS["reverted"])
    assert harness.state() == expected_state(FOLLOW_UPS["reverted"])


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

    originals = {
        name: getattr(harness.vector, name)
        for name in ("upsert_entities", "upsert_relationships")
    }
    for name in originals:
        setattr(harness.vector, name, broken(name))
    inc = IncrementalIndexer(harness.store, harness.manager)
    delta, _ = inc.plan([document(p, c) for p, c in after.items()])
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
