# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Documents whose indexing keeps dying are quarantined (AWS-free).

The write-ahead record counts the runs that started indexing a document's
content and never finished it (``attempt_count``); once the count reaches
``indexing.max_document_attempts`` the next run removes what the attempts
wrote, records the document QUARANTINED and skips it until its content
changes, so one document that kills the process no longer stops every run.
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.incremental_runs import (
    Corpus,
    IncrementalRun,
    document,
    expected_state,
)
from unified_kg_rag.domain.ingestion.delta_detector import (
    compute_content_hash,
    detect_delta,
    document_doc_id,
)
from unified_kg_rag.domain.models import (
    PENDING_CONTENT_HASH,
    DocStatus,
    DocStatusRecord,
)

pytestmark = pytest.mark.unit


def _pending(doc, attempts: int, attempted: str | None) -> DocStatusRecord:
    return DocStatusRecord(
        doc_id=document_doc_id(doc),
        content_hash=PENDING_CONTENT_HASH,
        status=DocStatus.PENDING,
        attempt_count=attempts,
        attempted_hash=attempted,
    )


@pytest.mark.parametrize(
    ("attempts", "same_content", "quarantined"),
    [(3, True, True), (2, True, False), (3, False, False)],
)
def test_detect_delta_quarantines_after_the_attempt_limit(
    attempts: int, same_content: bool, quarantined: bool
) -> None:
    doc = document("/c/a.txt", [[("Vendor", "Depot", 1)]])
    store = FakeDocStatusStore()
    attempted = compute_content_hash(doc) if same_content else "other-version"
    store.put(_pending(doc, attempts, attempted))

    delta, _ = detect_delta([doc], store, max_attempts=3)

    doc_id = document_doc_id(doc)
    assert delta.quarantined == ([doc_id] if quarantined else [])
    assert delta.changed == ([] if quarantined else [doc_id])


_A = {"/c/a.txt": [[("Vendor", "Depot", 1)]]}
_B = {"/c/b.txt": [[("Buyer", "Carrier", 2)]]}


def _crash_indexing(harness: IncrementalRun, corpus: Corpus, runs: int) -> None:
    for _ in range(runs):
        assert harness.run(corpus, interrupt="mid_index_delta", k=1)


def test_a_document_whose_indexing_keeps_dying_is_quarantined() -> None:
    harness = IncrementalRun()
    harness.run(_B)
    corpus = {**_A, **_B}
    _crash_indexing(harness, corpus, 3)
    doc_id = document_doc_id(document("/c/a.txt", _A["/c/a.txt"]))
    pending = harness.store.get(doc_id)
    assert (pending.status, pending.attempt_count) == (DocStatus.PENDING, 3)

    harness.run(corpus)

    # a.txt is skipped and what its attempts wrote is gone; b.txt is intact.
    assert harness.extracted == []
    record = harness.store.get(doc_id)
    assert record.status is DocStatus.QUARANTINED
    assert record.text_unit_ids == []
    assert harness.state() == expected_state(_B)
    # It stays skipped while its content stays.
    harness.run(corpus)
    assert harness.extracted == []
    assert harness.store.get(doc_id).status is DocStatus.QUARANTINED


def test_an_edited_quarantined_document_is_retried() -> None:
    harness = IncrementalRun()
    _crash_indexing(harness, _A, 3)
    harness.run(_A)
    edited = {"/c/a.txt": [[("Vendor", "Depot", 2)]]}

    harness.run(edited)

    assert harness.extracted == ["/c/a.txt"]
    assert harness.state() == expected_state(edited)
    assert harness.registry_problems(edited) == []


def test_a_finished_run_resets_the_attempt_count() -> None:
    harness = IncrementalRun()
    _crash_indexing(harness, _A, 2)
    harness.run(_A)
    doc_id = document_doc_id(document("/c/a.txt", _A["/c/a.txt"]))
    assert harness.store.get(doc_id).attempt_count == 0
    edited = {"/c/a.txt": [[("Vendor", "Depot", 2)]]}

    # Two more dying runs of other content start counting from one again.
    _crash_indexing(harness, edited, 2)
    harness.run(edited)

    assert harness.store.get(doc_id).status is DocStatus.PROCESSED
    assert harness.state() == expected_state(edited)
