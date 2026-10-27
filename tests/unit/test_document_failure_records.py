# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a document's FAILED record holds, and what counts toward its budget.

A stage that failed on a document's units for an error of their own and
stopped the run records the document FAILED itself
(``incremental.record_item_failures``): one more consecutive failure of the
content, its attempt closed, the run's settings and pipeline id. A resumed
delta is stale only once those failures, at that content and under those
settings, reach ``indexing.max_document_failures``
(``incremental.delta_still_applies``).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.incremental_runs import document
from unified_kg_rag.application.ingestion.incremental import (
    delta_still_applies,
    record_item_failures,
    run_failures,
)
from unified_kg_rag.domain.ingestion.delta_detector import document_doc_id
from unified_kg_rag.domain.models import (
    DocStatus,
    DocStatusRecord,
    Document,
    DocumentDelta,
)
from unified_kg_rag.domain.models.pipeline import PipelineContext
from unified_kg_rag.domain.models.text_unit import TextUnit

pytestmark = pytest.mark.unit


def _context(doc: Document, content_hash: str) -> PipelineContext:
    unit = TextUnit(id="u1", text="t", document_ids=[doc.document_id])
    return PipelineContext.model_construct(
        documents=[doc],
        text_units=[unit],
        translated_units=None,
        incremental_fingerprints={document_doc_id(doc): content_hash},
        incremental_extraction_fingerprint="fp-now",
        pipeline_id="P9",
    )


@pytest.mark.parametrize(
    ("status", "stored_hash", "stored_settings", "failures"),
    [
        (DocStatus.FAILED, "h-new", "fp-now", 3),  # one more failure of this content
        (DocStatus.FAILED, "h-new", None, 3),  # settings unknown: not changed
        (DocStatus.FAILED, "h-new", "fp-old", 1),  # failures at other settings
        (DocStatus.FAILED, "h-old", "fp-now", 1),  # failures of another version
        (DocStatus.PENDING, "h-new", "fp-old", 1),  # not failing before
    ],
)
def test_a_failed_record_counts_consecutive_failures_and_closes_the_attempt(
    status: DocStatus, stored_hash: str, stored_settings: str | None, failures: int
) -> None:
    doc = document("/c/a.txt", [[("Vendor", "Depot", 1)]])
    doc_id = document_doc_id(doc)
    store = FakeDocStatusStore()
    store.put(
        DocStatusRecord(
            doc_id=doc_id,
            content_hash=stored_hash,
            status=status,
            failure_count=2,
            attempt_count=2,
            crash_batch=3,
            attempted_hash="h-new",
            attempt_batch=1,
            extraction_fingerprint=stored_settings,
        )
    )
    context = _context(doc, "h-new")
    returned = record_item_failures(store, context, ["u1"], 3)
    record = store.get(doc_id)
    assert record is not None
    assert (record.status, record.content_hash, record.failure_count) == (
        DocStatus.FAILED,
        "h-new",
        failures,
    )
    assert (
        record.attempt_count,
        record.attempted_hash,
        record.attempt_batch,
        record.crash_batch,
    ) == (0, None, None, 0)
    assert record.extraction_fingerprint == "fp-now"
    assert record.written_by == "P9"
    # The records returned are the ones written (they lacked written_by).
    assert returned == [record]
    assert context.recorded_failed_doc_ids == [doc_id]


def test_the_failure_budget_counts_failures_of_this_content_and_settings_only() -> None:
    doc = document("/c/a.txt", [[("Vendor", "Depot", 1)]])
    doc_id = document_doc_id(doc)
    delta = DocumentDelta(new=[], changed=[doc_id], deleted=[], unchanged=[])

    def applies(**update: Any) -> bool:
        store = FakeDocStatusStore()
        failed = DocStatusRecord(
            doc_id=doc_id,
            content_hash="h-new",
            status=DocStatus.FAILED,
            failure_count=3,
            attempted_hash="h-new",
            extraction_fingerprint="fp",
            written_by="P1",
        )
        store.put(failed.model_copy(update=update))
        return delta_still_applies(
            store, delta, {doc_id: "h-new"}, 3, "fp", None, "P1", 3
        )

    assert not applies()  # at the budget: a new diff skips it
    assert applies(failure_count=2)
    assert applies(content_hash="h-old")  # failed at an older version
    assert applies(status=DocStatus.PENDING)
    assert applies(extraction_fingerprint="fp-other")  # settings changed since


_GRAPH, _CLAIMS = "graph_extraction", "claim_extraction"


@pytest.mark.parametrize(
    ("failed", "own", "expected"),
    [
        # Only claims failed, for an error of their own: indexed without them.
        ({_CLAIMS: ["a1"]}, {_CLAIMS: ["a1"]}, ((), (), ("a",))),
        # A claims outage on one unit: failed, uncounted, as in any stage.
        ({_CLAIMS: ["a1", "a2"]}, {_CLAIMS: ["a1"]}, (("a",), (), ())),
        # The graph failed too: counted for the graph's error alone.
        (
            {_GRAPH: ["a2"], _CLAIMS: ["a1"]},
            {_GRAPH: ["a2"], _CLAIMS: ["a1"]},
            (("a",), ("a",), ()),
        ),
        ({_GRAPH: ["a2"], _CLAIMS: ["a1"]}, {_CLAIMS: ["a1"]}, (("a",), (), ())),
        # Another document's units fail apart.
        (
            {_GRAPH: ["b1"], _CLAIMS: ["a1"]},
            {_GRAPH: ["b1"], _CLAIMS: ["a1"]},
            (("b",), ("b",), ("a",)),
        ),
    ],
)
def test_run_failures_leave_claims_optional(
    failed: dict[str, list[str]],
    own: dict[str, list[str]],
    expected: tuple[tuple[str, ...], ...],
) -> None:
    docs = {
        name: document(f"/c/{name}.txt", [[("Vendor", "Depot", 1)]])
        for name in ("a", "b")
    }
    units = [
        TextUnit(id=f"{name}{i}", text="t", document_ids=[doc.document_id])
        for name, doc in docs.items()
        for i in (1, 2)
    ]
    context = PipelineContext.model_construct(
        documents=list(docs.values()),
        text_units=units,
        translated_units=None,
        failed_text_unit_ids=failed,
        item_failed_text_unit_ids=own,
    )
    ids = {document_doc_id(doc): name for name, doc in docs.items()}
    result = run_failures(context, units)
    assert (
        tuple(
            tuple(sorted(ids[d] for d in group))
            for group in (result.failed, result.counted, result.claims_missing)
        )
        == expected
    )
