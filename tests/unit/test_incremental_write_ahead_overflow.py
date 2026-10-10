# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A write-ahead record over the registry's item limit spills into overflow.

A changed document's PENDING record lists its stored and planned lineage. When
each fits one registry item but the union does not, the record keeps the
stored lineage and the other planned ids go to lineage overflow, so the run
neither fails nor gives up the crash-recovery guarantee. The fake store's
``max_record_ids`` stands in for DynamoDB's 400 KB item limit.
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
    largest_lineage,
)
from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
from unified_kg_rag.domain.ingestion.delta_detector import document_doc_id
from unified_kg_rag.domain.models import (
    DocStatus,
    DocStatusRecord,
    DocumentDelta,
    DocumentLineage,
)
from unified_kg_rag.shared import DataProcessingError

pytestmark = pytest.mark.unit

# b changes to other entities: its old and new lineage are each as large as
# any document's, their union is larger.
R0: Corpus = {
    "/c/a.txt": [[("Vendor", "Depot", 1)]],
    "/c/b.txt": [[("Vendor", "Carrier", 2)], [("Depot", "Carrier", 1)]],
    "/c/d.txt": [[("Vendor", "Buyer", 3)]],
}
R1: Corpus = {
    "/c/a.txt": R0["/c/a.txt"],
    "/c/b.txt": [[("Bank", "Broker", 2)], [("Bank", "Lender", 1)]],
    "/c/e.txt": [[("Vendor", "Bank", 1)]],
}
FOLLOW_UPS: dict[str, Corpus] = {
    "same": R1,
    "reverted": {**R1, "/c/b.txt": R0["/c/b.txt"]},
    "b_removed": {p: c for p, c in R1.items() if p != "/c/b.txt"},
}
LIMIT = largest_lineage(R0, R1, *FOLLOW_UPS.values())


def _doc_id(path: str, corpus: Corpus) -> str:
    return document_doc_id(document(path, corpus[path]))


def _harness() -> tuple[IncrementalRun, FakeDocStatusStore]:
    store = FakeDocStatusStore(max_record_ids=LIMIT)
    harness = IncrementalRun(store)
    harness.run(R0)
    assert harness.state() == expected_state(R0)
    return harness, store


def test_only_the_oversized_record_spills_and_every_doc_is_pending() -> None:
    harness, store = _harness()
    b = _doc_id("/c/b.txt", R1)
    stored_b = store.get(b)
    assert stored_b is not None

    assert harness.run(R1, interrupt="after_write_ahead")

    # Every document of the delta has its PENDING record.
    pending = {r.file_path for r in store.list_all() if r.status is DocStatus.PENDING}
    assert pending == {"/c/b.txt", "/c/e.txt"}
    record = store.get(b)
    assert record is not None and record.status is DocStatus.PENDING
    # b's record keeps its stored lineage; only b has overflow, which holds
    # the planned ids the record lacks.
    assert record.entity_ids == stored_b.entity_ids
    assert record.text_unit_ids == stored_b.text_unit_ids
    assert set(store.overflow) == {b}
    overflow = store.get_lineage_overflow([b])[b]
    assert overflow.entity_ids
    assert not set(overflow.entity_ids) & set(stored_b.entity_ids)
    # Overflow is not a record: diff, list_all and get_many never see it.
    assert {r.doc_id for r in store.list_all()} == {
        _doc_id(p, R1) for p in ("/c/a.txt", "/c/b.txt", "/c/e.txt")
    } | {_doc_id("/c/d.txt", R0)}


@pytest.mark.parametrize("follow_up", sorted(FOLLOW_UPS))
@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("k", [1, 2])
def test_interrupted_run_with_overflow_converges(
    point: str, follow_up: str, k: int
) -> None:
    harness, store = _harness()
    harness.run(R1, interrupt=point, k=k)
    corpus = FOLLOW_UPS[follow_up]

    harness.run(corpus)

    assert harness.state() == expected_state(corpus)
    # registry_problems also reports leftover overflow.
    assert harness.registry_problems(corpus) == []
    harness.run(corpus)
    assert harness.extracted == []


def test_uninterrupted_run_with_overflow_cleans_it_up() -> None:
    harness, store = _harness()
    harness.run(R1)
    assert store.overflow == {}
    assert harness.registry_problems(R1) == []


def test_interrupted_overflow_cleanup_is_dropped_by_the_next_run() -> None:
    """The commit deletes overflow after replacing the PENDING record; dying
    in between leaves overflow next to a committed record."""
    harness, store = _harness()
    b = _doc_id("/c/b.txt", R1)
    original = store.delete_lineage_overflow

    def kill_at_commit(doc_ids):
        # The removal of d deletes d's (absent) overflow first: let it pass.
        if b in doc_ids:
            raise Killed("overflow cleanup")
        original(doc_ids)

    store.delete_lineage_overflow = kill_at_commit  # type: ignore[method-assign]
    try:
        assert harness.run(R1), "the commit's overflow cleanup did not run"
    finally:
        store.delete_lineage_overflow = original  # type: ignore[method-assign]
    record = store.get(b)
    assert record is not None and record.status is DocStatus.PROCESSED
    assert b in store.overflow

    # The stale overflow is ignored, not taken for b's lineage, and dropped.
    edited = {**R1, "/c/b.txt": [[("Bank", "Broker", 3)]]}
    harness.run(edited)
    assert harness.state() == expected_state(edited)
    assert harness.registry_problems(edited) == []


def test_overflow_left_by_a_commit_is_dropped_while_the_document_is_unchanged() -> None:
    """b reads unchanged after its commit died before the overflow delete; the
    next run's diff reports the overflow and its write-ahead drops it."""
    harness, store = _harness()
    b = _doc_id("/c/b.txt", R1)
    original = store.delete_lineage_overflow

    def kill_at_commit(doc_ids):
        if b in doc_ids:
            raise Killed("overflow cleanup")
        original(doc_ids)

    store.delete_lineage_overflow = kill_at_commit  # type: ignore[method-assign]
    try:
        assert harness.run(R1)
    finally:
        store.delete_lineage_overflow = original  # type: ignore[method-assign]
    assert b in store.overflow
    assert store.diff({}).orphan_overflow == [b]

    harness.run(R1)

    assert harness.extracted == []
    assert store.overflow == {}
    assert harness.state() == expected_state(R1)
    assert harness.registry_problems(R1) == []


def _leftover(scope: str | None, status: DocStatus) -> DocStatusRecord:
    return DocStatusRecord(doc_id="owner", content_hash="h", status=status, scope=scope)


@pytest.mark.parametrize(
    ("record", "run_scope", "collected"),
    [
        # A committed record of the run's namespace (another source scope).
        (_leftover("ns-a|/other", DocStatus.PROCESSED), "ns-a|/c", True),
        (_leftover("ns-a|/c", DocStatus.FAILED), "ns-a|/c", True),
        # Another namespace's: its runs collect it.
        (_leftover("ns-b|/c", DocStatus.PROCESSED), "ns-a|/c", False),
        # Became PENDING since the diff: the overflow is that record's.
        (_leftover("ns-a|/c", DocStatus.PENDING), "ns-a|/c", False),
        # No record, and no document of this run: namespace unknown.
        (None, "ns-a|/c", False),
        # A run without a scope owns the whole registry.
        (_leftover("ns-b|/c", DocStatus.PROCESSED), None, True),
    ],
)
def test_write_ahead_collects_leftover_overflow_of_its_namespace(
    record: DocStatusRecord | None, run_scope: str | None, collected: bool
) -> None:
    store = FakeDocStatusStore()
    if record is not None:
        store.put(record)
    store.add_lineage_overflow([DocumentLineage(doc_id="owner", entity_ids=["e"])])
    inc = IncrementalIndexer(store, None, scope=run_scope)  # type: ignore[arg-type]

    inc.write_ahead(DocumentDelta(orphan_overflow=["owner"]), [])

    assert ("owner" not in store.overflow) is collected
    assert store.list_all() == ([record] if record is not None else [])


def test_write_ahead_collects_leftover_overflow_of_a_new_document() -> None:
    store = FakeDocStatusStore()
    store.add_lineage_overflow([DocumentLineage(doc_id="new", entity_ids=["e"])])
    inc = IncrementalIndexer(store, None, scope="ns-a|/c")  # type: ignore[arg-type]
    delta = store.diff({"new": "h"}, scope="ns-a|/c")
    assert delta.orphan_overflow == ["new"]

    inc.write_ahead(delta, [DocumentLineage(doc_id="new", entity_ids=["e2"])])

    assert store.overflow == {}
    pending = store.get("new")
    assert pending is not None and pending.entity_ids == ["e2"]


def test_recovery_reads_overflow_of_another_runs_pending_record() -> None:
    """A deleted document whose interrupted run left overflow is removed with
    everything that run may have written (it is not in this run's delta)."""
    harness, store = _harness()
    assert harness.run(R1, interrupt="before_record")
    assert store.overflow

    corpus = FOLLOW_UPS["b_removed"]
    harness.run(corpus)
    assert harness.state() == expected_state(corpus)
    assert harness.registry_problems(corpus) == []


def test_merge_kept_ids_join_the_overflow() -> None:
    store = FakeDocStatusStore(max_record_ids=3)
    store.put(
        DocStatusRecord(
            doc_id="d",
            content_hash="h",
            status=DocStatus.PROCESSED,
            entity_ids=["e1", "e2", "e3"],
        )
    )
    inc = IncrementalIndexer(store, None)  # type: ignore[arg-type]
    inc.write_ahead(
        DocumentDelta(changed=["d"]),
        [DocumentLineage(doc_id="d", entity_ids=["e4", "e5"])],
    )
    assert store.get_lineage_overflow(["d"])["d"].entity_ids == ["e4", "e5"]

    inc._extend_pending([DocumentLineage(doc_id="d", entity_ids=["e3", "e6"])])

    record = store.get("d")
    assert record is not None and record.entity_ids == ["e1", "e2", "e3"]
    assert store.get_lineage_overflow(["d"])["d"].entity_ids == ["e4", "e5", "e6"]


def test_merge_kept_ids_spill_when_the_record_would_outgrow_the_limit() -> None:
    store = FakeDocStatusStore(max_record_ids=3)
    inc = IncrementalIndexer(store, None)  # type: ignore[arg-type]
    inc.write_ahead(
        DocumentDelta(new=["d"]),
        [DocumentLineage(doc_id="d", entity_ids=["e1", "e2", "e3"])],
    )
    assert store.overflow == {}

    inc._extend_pending([DocumentLineage(doc_id="d", entity_ids=["e4"])])

    record = store.get("d")
    assert record is not None and record.entity_ids == ["e1", "e2", "e3"]
    assert store.get_lineage_overflow(["d"])["d"].entity_ids == ["e4"]


def test_document_too_large_to_commit_fails_before_any_write() -> None:
    store = FakeDocStatusStore(max_record_ids=2)
    store.put(DocStatusRecord(doc_id="small", content_hash="h", entity_ids=["x"]))
    inc = IncrementalIndexer(store, None)  # type: ignore[arg-type]

    with pytest.raises(DataProcessingError, match="big.txt .3 artifact ids"):
        inc.write_ahead(
            DocumentDelta(new=["big"], changed=["small"]),
            [
                DocumentLineage(
                    doc_id="big", file_path="big.txt", entity_ids=["a", "b", "c"]
                ),
                DocumentLineage(doc_id="small", entity_ids=["y"]),
            ],
        )

    # Nothing was written: no PENDING record, no overflow.
    assert [r.doc_id for r in store.list_all()] == ["small"]
    small = store.get("small")
    assert small is not None and small.status is DocStatus.PENDING  # as stored
    assert small.content_hash == "h"
    assert store.overflow == {}


def test_write_ahead_probes_overflow_only_of_pending_records() -> None:
    store = FakeDocStatusStore()
    for doc_id, status in (("done", DocStatus.PROCESSED), ("busy", DocStatus.PENDING)):
        store.put(DocStatusRecord(doc_id=doc_id, content_hash="h", status=status))
    probed: list[str] = []
    original = store.get_lineage_overflow

    def probe(doc_ids):
        doc_ids = list(doc_ids)
        probed.extend(doc_ids)
        return original(doc_ids)

    store.get_lineage_overflow = probe  # type: ignore[method-assign]
    inc = IncrementalIndexer(store, None)  # type: ignore[arg-type]

    inc.write_ahead(
        DocumentDelta(new=["new"], changed=["done", "busy"]),
        [DocumentLineage(doc_id=d, entity_ids=["e"]) for d in ("new", "done", "busy")],
    )

    # A new or committed record has no overflow to keep.
    assert probed == ["busy"]
