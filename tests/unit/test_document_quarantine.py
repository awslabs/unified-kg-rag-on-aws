# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Documents whose processing keeps killing the run are isolated (AWS-free).

Each run opens an attempt on the documents it processes before any stage
runs and closes it however it ends in the process; only a run that died
leaves it open, and the next run counts it (``domain.ingestion.attempts``).
Once the deaths reach ``indexing.max_document_attempts`` a document that died
alone is quarantined; otherwise the runs bisect the documents that died
together, so the culprit is found and the others are indexed. A full rebuild
and a run that stopped cleanly count nothing.
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.incremental_runs import (
    Corpus,
    IncrementalRun,
    document,
    expected_state,
    extract,
)
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    build_document_lineage,
    close_attempts,
    delta_still_applies,
    open_attempts,
)
from unified_kg_rag.domain.ingestion.attempts import settled_attempts
from unified_kg_rag.domain.ingestion.delta_detector import (
    compute_content_hash,
    detect_delta,
    document_doc_id,
    fingerprint_documents,
)
from unified_kg_rag.domain.models import (
    PENDING_CONTENT_HASH,
    DocStatus,
    DocStatusRecord,
    DocumentDelta,
)

pytestmark = pytest.mark.unit


def _record(
    doc,
    deaths: int,
    crash_batch: int,
    attempted: str | None,
    attempt_batch: int | None = None,
) -> DocStatusRecord:
    return DocStatusRecord(
        doc_id=document_doc_id(doc),
        content_hash=PENDING_CONTENT_HASH,
        status=DocStatus.PENDING,
        attempt_count=deaths,
        crash_batch=crash_batch,
        attempted_hash=attempted,
        attempt_batch=attempt_batch,
    )


@pytest.mark.parametrize(
    ("deaths", "crash_batch", "attempt_batch", "same_content", "quarantined"),
    [
        (3, 1, None, True, True),
        (2, 1, None, True, False),
        (3, 1, None, False, False),
        # Died with other documents: bisected, not quarantined.
        (3, 4, None, True, False),
        # The run that died left its attempt open: counted now.
        (2, 4, 1, True, True),
    ],
)
def test_detect_delta_quarantines_a_document_that_died_alone(
    deaths: int,
    crash_batch: int,
    attempt_batch: int | None,
    same_content: bool,
    quarantined: bool,
) -> None:
    doc = document("/c/a.txt", [[("Vendor", "Depot", 1)]])
    store = FakeDocStatusStore()
    attempted = compute_content_hash(doc) if same_content else "other-version"
    store.put(_record(doc, deaths, crash_batch, attempted, attempt_batch))

    delta, _ = detect_delta([doc], store, max_attempts=3)

    doc_id = document_doc_id(doc)
    assert delta.quarantined == ([doc_id] if quarantined else [])
    assert delta.changed == ([] if quarantined else [doc_id])


_A = {"/c/a.txt": [[("Vendor", "Depot", 1)]]}
_B = {"/c/b.txt": [[("Buyer", "Carrier", 2)]]}
_C = {"/c/c.txt": [[("Seller", "Port", 3)]]}
_D = {"/c/d.txt": [[("Lessor", "Lessee", 2)]]}


def _crash_indexing(harness: IncrementalRun, corpus: Corpus, runs: int) -> None:
    for _ in range(runs):
        assert harness.run(corpus, interrupt="mid_index_delta", k=1)


def _record_of(harness: IncrementalRun, path: str, corpus: Corpus) -> DocStatusRecord:
    record = harness.store.get(document_doc_id(document(path, corpus[path])))
    assert record is not None
    return record


def _run_until_settled(harness: IncrementalRun, corpus: Corpus, runs: int) -> int:
    """Run until a run ends in the process with nothing deferred; return how
    many runs it took."""
    for count in range(1, runs + 1):
        died = harness.run(corpus)
        records = harness.store.list_all()
        if not died and all(r.status is not DocStatus.PENDING for r in records):
            return count
    raise AssertionError(f"not settled after {runs} runs")


def test_a_document_whose_processing_keeps_killing_the_run_alone_is_quarantined() -> (
    None
):
    harness = IncrementalRun()
    harness.run(_B)
    corpus = {**_A, **_B}
    _crash_indexing(harness, corpus, 3)
    pending = _record_of(harness, "/c/a.txt", corpus)
    assert settled_attempts(pending, pending.attempted_hash or "") == (3, 1)

    harness.run(corpus)

    # a.txt is skipped and what its attempts wrote is gone; b.txt is intact.
    assert harness.extracted == []
    record = _record_of(harness, "/c/a.txt", corpus)
    assert record.status is DocStatus.QUARANTINED
    assert record.text_unit_ids == []
    assert harness.state() == expected_state(_B)
    # It stays skipped while its content stays.
    harness.run(corpus)
    assert harness.extracted == []
    assert _record_of(harness, "/c/a.txt", corpus).status is DocStatus.QUARANTINED


def test_the_culprit_is_isolated_and_the_documents_it_died_with_are_indexed() -> None:
    # Old behaviour: every document of the delta that kept dying (a.txt,
    # c.txt and d.txt) was quarantined with the culprit.
    harness = IncrementalRun()
    harness.run(_B)
    corpus = {**_A, **_B, **_C, **_D}
    harness.crash_on = lambda paths: "/c/c.txt" in paths
    for _ in range(3):
        assert harness.run(corpus)

    runs = _run_until_settled(harness, corpus, 8)

    assert runs <= 4  # log2(3 suspects) + quarantine + the rest
    assert _record_of(harness, "/c/c.txt", corpus).status is DocStatus.QUARANTINED
    for path in ("/c/a.txt", "/c/b.txt", "/c/d.txt"):
        assert _record_of(harness, path, corpus).status is DocStatus.PROCESSED
    assert harness.state() == expected_state({**_A, **_B, **_D})


def test_runs_that_die_on_large_batches_end_up_indexing_every_document() -> None:
    # No document is at fault: the task dies whenever a run processes more
    # than one document (out of memory, say). Bisecting processes them in
    # smaller batches; none is quarantined.
    harness = IncrementalRun()
    corpus = {**_A, **_B, **_C, **_D}
    harness.crash_on = lambda paths: len(paths) > 1
    for _ in range(3):
        assert harness.run(corpus)

    _run_until_settled(harness, corpus, 12)

    assert {r.status for r in harness.store.list_all()} == {DocStatus.PROCESSED}
    assert harness.state() == expected_state(corpus)


def test_a_bisecting_run_defers_the_other_documents_and_keeps_their_content() -> None:
    harness = IncrementalRun()
    harness.run({**_A, **_B})
    harness.crash_on = lambda paths: "/c/c.txt" in paths
    for _ in range(3):
        assert harness.run({**_A, **_B, **_C, **_D})
    # b.txt is edited after the deaths: not a suspect, so the bisecting run
    # defers it and its indexed version stays.
    edited = {"/c/b.txt": [[("Buyer", "Carrier", 3)]]}
    corpus = {**_A, **edited, **_C, **_D}

    harness.run(corpus)

    assert harness.extracted in (["/c/c.txt"], ["/c/d.txt"])
    b_record = _record_of(harness, "/c/b.txt", corpus)
    assert b_record.status is DocStatus.PROCESSED
    assert b_record.content_hash == compute_content_hash(
        document("/c/b.txt", _B["/c/b.txt"])
    )
    assert harness.graph.ids("entities") >= set(expected_state(_B)["graph_entities"])
    _run_until_settled(harness, corpus, 8)
    assert _record_of(harness, "/c/c.txt", corpus).status is DocStatus.QUARANTINED
    assert harness.state() == expected_state({**_A, **edited, **_D})


def test_a_document_that_dies_during_any_stage_counts() -> None:
    # The attempt opens right after the diff: a run that dies before the
    # indexing stage (in chunking or extraction) counts too. Old behaviour:
    # the attempt was counted only from the indexing stage's write-ahead, so
    # such a document blocked every run.
    harness = IncrementalRun()
    harness.run(_B)
    corpus = {**_A, **_B}
    harness.crash_on = lambda paths: "/c/a.txt" in paths
    for _ in range(3):
        assert harness.run(corpus)

    harness.run(corpus)

    assert _record_of(harness, "/c/a.txt", corpus).status is DocStatus.QUARANTINED
    assert harness.state() == expected_state(_B)


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
    assert _record_of(harness, "/c/a.txt", _A).attempt_count == 0
    edited = {"/c/a.txt": [[("Vendor", "Depot", 2)]]}

    # Two more dying runs of other content start counting from one again.
    _crash_indexing(harness, edited, 2)
    harness.run(edited)

    assert _record_of(harness, "/c/a.txt", edited).status is DocStatus.PROCESSED
    assert harness.state() == expected_state(edited)


def test_runs_that_stop_cleanly_count_nothing() -> None:
    # Old behaviour: a SIGTERM (a task timeout, StopTask) counted like a
    # crash, so three timed-out runs quarantined the whole delta.
    harness = IncrementalRun()
    corpus = {**_A, **_B}
    docs = [document(path, content) for path, content in corpus.items()]
    for _ in range(5):
        delta, fingerprints = detect_delta(docs, harness.store, max_attempts=3)
        open_attempts(harness.store, delta, fingerprints)
        # ... the run is stopped and unwinds ...
        close_attempts(harness.store, delta, fingerprints)

    harness.run(corpus)

    assert harness.extracted == sorted(corpus)
    assert {r.status for r in harness.store.list_all()} == {DocStatus.PROCESSED}


def test_full_rebuilds_that_die_count_no_attempt() -> None:
    # Old behaviour: each reset's write-ahead counted an attempt for every
    # document, so after three rebuilds that died the next normal run
    # quarantined the whole corpus and removed every artifact.
    corpus = {**_A, **_B}
    harness = IncrementalRun()
    harness.run(corpus)
    docs = [document(path, content) for path, content in corpus.items()]
    for _ in range(3):  # what the indexing stage does on indexing.reset
        fingerprints = fingerprint_documents(docs)
        built = extract(corpus)
        inc = IncrementalIndexer(harness.store, harness._manager())
        inc.write_ahead(
            DocumentDelta(new=list(fingerprints)),
            build_document_lineage(documents=docs, text_units=built.units),
        )
        # ... the task dies inside index_all_data ...

    harness.run(corpus)

    assert harness.extracted == sorted(corpus)
    assert {r.status for r in harness.store.list_all()} == {DocStatus.PROCESSED}
    assert harness.state() == expected_state(corpus)


def test_a_resumed_delta_is_recomputed_when_another_run_changed_its_documents() -> None:
    harness = IncrementalRun()
    corpus = {**_A, **_B}
    docs = [document(path, content) for path, content in corpus.items()]
    delta, fingerprints = detect_delta(docs, harness.store, max_attempts=3)
    open_attempts(harness.store, delta, fingerprints)
    assert delta_still_applies(harness.store, delta, fingerprints, 3)

    # Another run committed the same content: redoing it is idempotent.
    harness.run(corpus)
    assert delta_still_applies(harness.store, delta, fingerprints, 3)
    # Another run indexed another version of a.txt, or found b.txt deleted.
    harness.run({"/c/a.txt": [[("Vendor", "Depot", 2)]], **_B})
    assert not delta_still_applies(harness.store, delta, fingerprints, 3)
    harness.run(corpus)
    harness.run(_A)
    assert not delta_still_applies(harness.store, delta, fingerprints, 3)


def test_a_resumed_delta_is_replanned_once_its_runs_died_too_often() -> None:
    harness = IncrementalRun()
    docs = [document(path, content) for path, content in {**_A, **_B}.items()]
    delta, fingerprints = detect_delta(docs, harness.store, max_attempts=3)
    for runs in range(1, 4):
        open_attempts(harness.store, delta, fingerprints)  # ... and died
        assert delta_still_applies(harness.store, delta, fingerprints, 3) == (runs < 3)


def test_a_changed_document_that_dies_before_indexing_keeps_its_indexed_version() -> (
    None
):
    # Old behaviour: quarantine removed the indexed version too, although the
    # runs that died never reached the stores.
    harness = IncrementalRun()
    harness.run({**_A, **_B})
    edited = {"/c/a.txt": [[("Vendor", "Depot", 3)]]}
    corpus = {**edited, **_B}
    harness.crash_on = lambda paths: paths == ["/c/a.txt"]
    for _ in range(3):
        assert harness.run(corpus)

    harness.run(corpus)

    record = _record_of(harness, "/c/a.txt", corpus)
    assert record.status is DocStatus.QUARANTINED
    assert record.content_hash == compute_content_hash(
        document("/c/a.txt", edited["/c/a.txt"])
    )
    assert record.text_unit_ids
    assert harness.state() == expected_state({**_A, **_B})
    # Skipped while its content stays; an edit replaces the kept version.
    harness.run(corpus)
    assert harness.extracted == []
    harness.crash_on = None
    again = {"/c/a.txt": [[("Vendor", "Depot", 2)]], **_B}
    harness.run(again)
    assert harness.state() == expected_state(again)


def test_raising_the_limit_retries_a_quarantined_document() -> None:
    # Old behaviour: a QUARANTINED record read unchanged whatever the limit,
    # although the setting promised a retry once raised.
    harness = IncrementalRun()
    harness.run(_B)
    corpus = {**_A, **_B}
    _crash_indexing(harness, corpus, 3)
    harness.run(corpus)
    assert _record_of(harness, "/c/a.txt", corpus).status is DocStatus.QUARANTINED

    harness.config.indexing.max_document_attempts = 5
    harness.run(corpus)

    assert harness.extracted == ["/c/a.txt"]
    assert _record_of(harness, "/c/a.txt", corpus).status is DocStatus.PROCESSED
    assert harness.state() == expected_state(corpus)


def test_a_retried_quarantined_document_counts_on_from_its_deaths() -> None:
    harness = IncrementalRun()
    _crash_indexing(harness, _A, 3)
    harness.run(_A)
    harness.config.indexing.max_document_attempts = 4

    _crash_indexing(harness, _A, 1)
    harness.run(_A)

    record = _record_of(harness, "/c/a.txt", _A)
    assert (record.status, record.attempt_count) == (DocStatus.QUARANTINED, 4)
