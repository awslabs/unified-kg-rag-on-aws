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

import functools
from types import SimpleNamespace
from typing import Any

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
from unified_kg_rag.domain.ingestion.attempts import (
    closed_attempt,
    settled_attempts,
    unproductive_attempt,
)
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


def test_a_resumed_delta_is_recomputed_when_another_run_moved_a_document_outside_it() -> (
    None
):
    # A finished pipeline id run again restores its delta (b.txt new) too.
    harness = IncrementalRun()
    harness.run(_A)
    docs = [document(path, content) for path, content in {**_A, **_B}.items()]
    delta, fingerprints = detect_delta(docs, harness.store, max_attempts=3)
    open_attempts(harness.store, delta, fingerprints)
    scope = "default|corpus"
    assert delta.new == [document_doc_id(docs[1])]
    assert delta_still_applies(harness.store, delta, fingerprints, 3, scope=scope)

    # a.txt still failing on this content (max_document_failures left it
    # unchanged) is no change since the delta.
    a_id = document_doc_id(docs[0])
    failed = harness.store.get(a_id)
    assert failed is not None
    harness.store.put(failed.model_copy(update={"status": DocStatus.FAILED}))
    assert delta_still_applies(harness.store, delta, fingerprints, 3, scope=scope)
    # Another run indexed another version of a.txt, which the delta left
    # unchanged: indexing the delta again would not bring it back.
    harness.run({"/c/a.txt": [[("Vendor", "Depot", 2)]], **_B})
    assert not delta_still_applies(harness.store, delta, fingerprints, 3, scope=scope)
    assert delta_still_applies(harness.store, delta, fingerprints, 3)


def test_a_resumed_delta_is_recomputed_when_another_scope_indexed_a_document_it_deletes() -> (
    None
):
    # An interrupted reset of A moved B's record into A, a run of A deleted
    # it, a run of B indexed it again: A's restored delta still deletes it.
    store = FakeDocStatusStore()
    record = DocStatusRecord(
        doc_id="default#b",
        content_hash="hb",
        status=DocStatus.PROCESSED,
        scope="default|B",
        written_by="P1",
    )
    store.put(record)
    delta = DocumentDelta(deleted=[record.doc_id])
    check = functools.partial(
        delta_still_applies, store, delta, {}, 3, scope="default|A", pipeline_id="P0"
    )
    assert not check()
    # A scope the delta retired is the delta's to delete.
    assert check(retired=["B"])
    # Still in the delta's scope, or written by this pipeline id.
    store.put(record.model_copy(update={"scope": "default|A"}))
    assert check()
    store.put(record.model_copy(update={"written_by": "P0"}))
    assert check()


def _resumable_delta() -> tuple[IncrementalRun, list, DocumentDelta, dict[str, str]]:
    """a.txt indexed; the delta of a.txt + b.txt (b.txt new) computed and
    its attempts opened, as a pipeline id that is resumed later."""
    harness = IncrementalRun()
    harness.run(_A)
    docs = [document(path, content) for path, content in {**_A, **_B}.items()]
    delta, fingerprints = detect_delta(docs, harness.store, max_attempts=3)
    open_attempts(harness.store, delta, fingerprints)
    return harness, docs, delta, fingerprints


def test_a_resumed_delta_is_recomputed_when_another_run_deleted_a_document_outside_it() -> (
    None
):
    harness, docs, delta, fingerprints = _resumable_delta()
    scope = "default|corpus"
    assert delta_still_applies(harness.store, delta, fingerprints, 3, scope=scope)
    # a.txt's record is gone: the delta left it unchanged, a fresh diff has
    # it new.
    harness.store.delete(document_doc_id(docs[0]))
    assert not delta_still_applies(harness.store, delta, fingerprints, 3, scope=scope)


def test_a_resumed_delta_is_recomputed_when_another_run_indexed_a_document_outside_the_corpus() -> (
    None
):
    harness, docs, delta, fingerprints = _resumable_delta()
    scope = "default|corpus"
    a = harness.store.get(document_doc_id(docs[0]))
    assert a is not None
    c = document("/c/c.txt", [[("Seller", "Port", 3)]])
    harness.store.put(
        a.model_copy(
            update={"doc_id": document_doc_id(c), "content_hash": "hc", "scope": scope}
        )
    )
    assert not delta_still_applies(harness.store, delta, fingerprints, 3, scope=scope)


def test_a_resumed_delta_is_recomputed_when_another_run_re_extracted_a_document_outside_it() -> (
    None
):
    # a.txt committed at its content under other settings: unchanged for a
    # run with those settings, changed (and not a failing document) for
    # this one's.
    harness, docs, delta, fingerprints = _resumable_delta()
    scope = "default|corpus"
    a = harness.store.get(document_doc_id(docs[0]))
    assert a is not None
    harness.store.put(a.model_copy(update={"extraction_fingerprint": "other"}))
    assert delta_still_applies(
        harness.store, delta, fingerprints, 3, "other", scope=scope
    )
    assert not delta_still_applies(
        harness.store, delta, fingerprints, 3, "mine", scope=scope
    )


def test_a_resumed_delta_is_recomputed_when_another_pipeline_id_wrote_its_records() -> (
    None
):
    harness = IncrementalRun()
    docs = [document(path, content) for path, content in {**_A, **_B}.items()]
    delta, fingerprints = detect_delta(docs, harness.store, max_attempts=3)
    open_attempts(harness.store, delta, fingerprints, written_by="P1")
    assert {r.written_by for r in harness.store.list_all()} == {"P1"}
    assert delta_still_applies(harness.store, delta, fingerprints, 3, pipeline_id="P1")
    # P1 resumed after its own death opens its attempts again: still its delta.
    open_attempts(harness.store, delta, fingerprints, written_by="P1")
    assert delta_still_applies(harness.store, delta, fingerprints, 3, pipeline_id="P1")

    # Another pipeline id opened (or committed, or left PENDING) one of them:
    # P1's delta no longer describes what the registry holds.
    b_id = document_doc_id(docs[1])
    b = harness.store.get(b_id)
    assert b is not None
    harness.store.put(b.model_copy(update={"written_by": "C"}))
    assert not delta_still_applies(
        harness.store, delta, fingerprints, 3, pipeline_id="P1"
    )
    # Without a pipeline id (the direct API) the writer is not checked.
    assert delta_still_applies(harness.store, delta, fingerprints, 3)
    assert close_attempts(harness.store, delta, fingerprints, "P1") == 2
    assert {r.written_by for r in harness.store.list_all()} == {"P1"}


def test_a_processed_record_without_a_writer_is_left_to_the_other_checks() -> None:
    # Written before DocStatusRecord.written_by existed: a committed record
    # passes the writer check, and the content check still decides.
    harness = IncrementalRun()
    corpus = {**_A, **_B}
    docs = [document(path, content) for path, content in corpus.items()]
    delta, fingerprints = detect_delta(docs, harness.store, max_attempts=3)
    harness.run(corpus)
    assert {r.written_by for r in harness.store.list_all()} == {None}
    assert delta_still_applies(harness.store, delta, fingerprints, 3, pipeline_id="P1")
    harness.run({"/c/a.txt": [[("Vendor", "Depot", 2)]], **_B})
    assert {r.written_by for r in harness.store.list_all()} == {None}
    assert not delta_still_applies(
        harness.store, delta, fingerprints, 3, pipeline_id="P1"
    )


@pytest.mark.parametrize("state", ["open", "pending", "failed"])
def test_an_unfinished_record_without_a_writer_is_another_pipeline_ids(
    state: str,
) -> None:
    # Upgrade window: a run of a release before written_by left the record
    # with an open attempt, PENDING or FAILED; it may be another pipeline
    # id's, so a resumed delta is computed again.
    harness = IncrementalRun()
    docs = [document(path, content) for path, content in {**_A, **_B}.items()]
    delta, fingerprints = detect_delta(docs, harness.store, max_attempts=3)
    open_attempts(harness.store, delta, fingerprints)
    if state != "open":
        for record in harness.store.list_all():
            harness.store.put(
                closed_attempt(record).model_copy(update={"status": DocStatus(state)})
            )
    assert {r.written_by for r in harness.store.list_all()} == {None}
    assert delta_still_applies(harness.store, delta, fingerprints, 3)
    assert not delta_still_applies(
        harness.store, delta, fingerprints, 3, pipeline_id="P1"
    )


def test_a_delta_document_committed_under_other_settings_is_stale() -> None:
    # Records without a writer: only the settings check sees it.
    harness = IncrementalRun()
    docs = [document(path, content) for path, content in {**_A, **_B}.items()]
    delta, fingerprints = detect_delta(
        docs, harness.store, max_attempts=3, extraction_fingerprint="mine"
    )
    open_attempts(harness.store, delta, fingerprints)
    a_id = document_doc_id(docs[0])
    record = harness.store.get(a_id)
    assert record is not None
    committed = record.model_copy(
        update={
            "status": DocStatus.PROCESSED,
            "content_hash": fingerprints[a_id],
            "extraction_fingerprint": "other",
        }
    )
    harness.store.put(committed)
    assert not delta_still_applies(harness.store, delta, fingerprints, 3, "mine")
    harness.store.put(committed.model_copy(update={"extraction_fingerprint": "mine"}))
    assert delta_still_applies(harness.store, delta, fingerprints, 3, "mine")


def test_a_delta_re_extracting_for_new_settings_still_applies() -> None:
    harness = IncrementalRun()
    harness.run(_A)
    docs = [document(path, content) for path, content in _A.items()]
    a_id = document_doc_id(docs[0])
    record = harness.store.get(a_id)
    assert record is not None and record.status is DocStatus.PROCESSED
    harness.store.put(record.model_copy(update={"extraction_fingerprint": "old"}))
    delta, fingerprints = detect_delta(
        docs, harness.store, max_attempts=3, extraction_fingerprint="mine"
    )
    assert a_id in delta.settings_changed, delta
    open_attempts(harness.store, delta, fingerprints)
    assert delta_still_applies(harness.store, delta, fingerprints, 3, "mine")


def test_a_failed_document_retried_under_new_settings_still_applies() -> None:
    # Only a commit (PROCESSED) under other settings is another run's work:
    # a document that failed under them is retried by this delta.
    harness = IncrementalRun()
    harness.run(_A)
    docs = [document(path, content) for path, content in _A.items()]
    a_id = document_doc_id(docs[0])
    record = harness.store.get(a_id)
    assert record is not None
    harness.store.put(
        record.model_copy(
            update={"status": DocStatus.FAILED, "extraction_fingerprint": "old"}
        )
    )
    delta, fingerprints = detect_delta(
        docs, harness.store, max_attempts=3, extraction_fingerprint="mine"
    )
    assert delta.changed == [a_id] and not delta.settings_changed, delta
    open_attempts(harness.store, delta, fingerprints)
    assert delta_still_applies(harness.store, delta, fingerprints, 3, "mine")


def test_a_changed_document_whose_old_version_had_other_settings_still_applies() -> (
    None
):
    harness = IncrementalRun()
    harness.run(_A)
    a_id = document_doc_id(document(*next(iter(_A.items()))))
    record = harness.store.get(a_id)
    assert record is not None
    harness.store.put(record.model_copy(update={"extraction_fingerprint": "old"}))
    edited = document(next(iter(_A)), [[("Vendor", "Depot", 2)]])
    delta, fingerprints = detect_delta(
        [edited], harness.store, max_attempts=3, extraction_fingerprint="mine"
    )
    assert delta.changed == [a_id], delta
    open_attempts(harness.store, delta, fingerprints)
    assert delta_still_applies(harness.store, delta, fingerprints, 3, "mine")


def test_a_document_that_failed_to_load_is_no_move() -> None:
    harness = IncrementalRun()
    harness.run({**_A, **_B})
    for record in harness.store.list_all():
        harness.store.put(record.model_copy(update={"scope": "default|corpus"}))
    docs = [document(path, content) for path, content in {**_A, **_B}.items()]
    b_id = document_doc_id(docs[1])
    edited = document(next(iter(_A)), [[("Vendor", "Depot", 2)]])
    delta, fingerprints = detect_delta(
        [edited],
        harness.store,
        scope="default|corpus",
        failed_doc_ids=[b_id],
        max_attempts=3,
    )
    assert delta.failed == [b_id], delta
    open_attempts(harness.store, delta, fingerprints)
    assert delta_still_applies(
        harness.store, delta, fingerprints, 3, scope="default|corpus"
    )


def _bisected_delta(runs: int) -> tuple[IncrementalRun, DocumentDelta, dict[str, str]]:
    """The delta of a.txt-d.txt after ``runs`` runs died on c.txt, its
    attempts opened."""
    harness = IncrementalRun()
    harness.run(_B)
    corpus = {**_A, **_B, **_C, **_D}
    harness.crash_on = lambda paths: "/c/c.txt" in paths
    for _ in range(runs):
        assert harness.run(corpus)
    docs = [document(path, content) for path, content in corpus.items()]
    delta, fingerprints = detect_delta(docs, harness.store, max_attempts=3)
    open_attempts(harness.store, delta, fingerprints)
    return harness, delta, fingerprints


@pytest.mark.parametrize(("runs", "kind"), [(3, "deferred"), (5, "quarantined")])
def test_a_deferred_or_quarantined_document_is_no_move(runs: int, kind: str) -> None:
    harness, delta, fingerprints = _bisected_delta(runs)
    assert getattr(delta, kind), delta
    # max_attempts is raised past the deaths the runs left: with 3, the
    # deaths check alone makes the delta stale and the scope diff, this
    # test's subject, is never reached.
    assert delta_still_applies(
        harness.store, delta, fingerprints, 99, scope="default|corpus"
    )


def test_another_version_failing_outside_the_delta_is_a_move() -> None:
    # Another run tried another version of a.txt (left unchanged by the
    # delta) and it failed: not "still failing on this content".
    harness, docs, delta, fingerprints = _resumable_delta()
    record = harness.store.get(document_doc_id(docs[0]))
    assert record is not None
    harness.store.put(
        record.model_copy(update={"status": DocStatus.FAILED, "content_hash": "v2"})
    )
    assert not delta_still_applies(
        harness.store, delta, fingerprints, 3, scope="default|corpus"
    )


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


def test_a_run_without_text_units_counts_a_death_of_the_document_alone() -> None:
    record = DocStatusRecord(
        doc_id="d",
        content_hash=PENDING_CONTENT_HASH,
        status=DocStatus.PENDING,
        attempt_count=2,
        crash_batch=3,
        attempted_hash="h1",
        attempt_batch=1,
    )
    again = unproductive_attempt(record, "h1")
    assert (again.attempt_count, again.crash_batch, again.attempt_batch) == (3, 1, None)
    # Deaths on another content do not count toward this one.
    edited = unproductive_attempt(record, "h2")
    assert (edited.attempt_count, edited.attempted_hash) == (1, "h2")


def test_a_run_that_lost_its_lease_leaves_its_attempts_open() -> None:
    # The run holding the lease now owns the records: closing the attempts
    # would overwrite what it wrote. Left open, they count as one death.
    from unified_kg_rag.shared import NamespaceLeaseError
    from unified_kg_rag.shared.write_fence import (
        install_write_fence,
        remove_write_fence,
    )

    class _Lost:
        def check(self) -> None:
            raise NamespaceLeaseError("lost")

    harness = IncrementalRun()
    docs = [document(path, content) for path, content in _A.items()]
    delta, fingerprints = detect_delta(docs, harness.store, max_attempts=3)
    open_attempts(harness.store, delta, fingerprints)
    token = install_write_fence(_Lost())
    try:
        with pytest.raises(NamespaceLeaseError):
            close_attempts(harness.store, delta, fingerprints)
    finally:
        remove_write_fence(token)
    assert {r.attempt_batch is not None for r in harness.store.list_all()} == {True}


@pytest.mark.parametrize(
    ("holder", "closed"),
    [("none", True), ("this run", True), ("another run", False)],
)
def test_a_run_that_lost_its_lease_closes_its_attempts_unless_another_holds_it(
    holder: str, closed: bool
) -> None:
    # Old behaviour: the close failed on the lost lease whoever held it, so
    # a run that only lost its lease (a slow registry) counted as a death of
    # every document it processed. It retakes a lease no other run holds.
    import time
    from types import SimpleNamespace

    from unified_kg_rag.application.ingestion.namespace_lease import NamespaceLease
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
    from unified_kg_rag.domain.models import Config

    harness = IncrementalRun()
    docs = [document(path, content) for path, content in _A.items()]
    delta, fingerprints = detect_delta(docs, harness.store, max_attempts=3)
    open_attempts(harness.store, delta, fingerprints)
    pipe = object.__new__(DataIngestionPipeline)
    pipe.config = Config()
    pipe._doc_status = harness.store
    context = SimpleNamespace(
        incremental_delta=delta,
        incremental_fingerprints=fingerprints,
        pipeline_id="P1",
    )
    with NamespaceLease(harness.store, "ns", 60) as lease:
        lease._lose("a renewal failed")
        if holder == "none":
            harness.store.release_lease("ns", lease.owner)
        elif holder == "another run":
            harness.store.leases["ns"] = ("other", time.time() + 60)
        pipe._close_attempts(context, lease)  # type: ignore[arg-type]
    still_open = {r.attempt_batch is not None for r in harness.store.list_all()}
    assert still_open == {not closed}
    # The retaken lease is released at the end; another run's is kept.
    assert (harness.store.lease_holder("ns") or ("",))[0] == (
        "other" if holder == "another run" else ""
    )


def test_a_quarantined_document_restored_after_an_interrupted_removal() -> None:
    # c.txt's edit keeps killing the run: it is quarantined with its indexed
    # version kept. Deleting it is killed after the vector copies went. Back
    # with the edit, it was read as unchanged and kept a record listing text
    # units the stores no longer held; now it is quarantined again at once
    # (no further death) with nothing of it left.
    harness = IncrementalRun()
    indexed = {**_A, **_C}
    harness.run(indexed)
    edited = {**_A, "/c/c.txt": [[("Seller", "Bank", 1)]]}
    harness.crash_on = lambda paths: "/c/c.txt" in paths
    for _ in range(3):
        assert harness.run(edited)
    harness.run(edited)
    assert _record_of(harness, "/c/c.txt", edited).status is DocStatus.QUARANTINED
    assert harness.state() == expected_state(indexed)

    assert harness.run(_A, interrupt="mid_removal", k=2)
    harness.run(edited)

    record = _record_of(harness, "/c/c.txt", edited)
    assert (record.status, record.text_unit_ids) == (DocStatus.QUARANTINED, [])
    assert harness.extracted == []
    assert harness.state() == expected_state(_A)


def _resumed_pipeline(store: FakeDocStatusStore) -> Any:
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
    from unified_kg_rag.domain.models import Config

    pipe = object.__new__(DataIngestionPipeline)
    pipe.config = Config()
    pipe._doc_status = store
    pipe.stages = [  # type: ignore[misc]
        SimpleNamespace(name=name) for name in DataIngestionPipeline._ORDER
    ]
    return pipe


def test_a_resumed_run_opens_its_attempts_as_its_pipeline_id() -> None:
    # The writer is what tells this pipeline id's records from another's
    # (delta_still_applies): unstamped, a rerun reads its own as foreign.
    harness = IncrementalRun()
    docs = [document(path, content) for path, content in _A.items()]
    delta, fingerprints = detect_delta(docs, harness.store, max_attempts=3)
    context = SimpleNamespace(
        incremental_delta=delta,
        incremental_fingerprints=fingerprints,
        incremental_scope=None,
        pipeline_id="P1",
    )
    _resumed_pipeline(harness.store)._open_resumed_attempts(context, "graph_extraction")
    records = harness.store.list_all()
    assert records and {r.written_by for r in records} == {"P1"}
    assert {r.attempt_batch is not None for r in records} == {True}


def test_a_resume_checks_its_delta_under_the_runs_extraction_settings() -> None:
    # A document of the delta another run committed under other settings:
    # the resume computes the delta again rather than index over it.
    harness = IncrementalRun()
    docs = [document(path, content) for path, content in _A.items()]
    delta, fingerprints = detect_delta(
        docs, harness.store, max_attempts=3, extraction_fingerprint="mine"
    )
    open_attempts(harness.store, delta, fingerprints)
    a_id = document_doc_id(docs[0])
    record = harness.store.get(a_id)
    assert record is not None
    harness.store.put(
        record.model_copy(
            update={
                "status": DocStatus.PROCESSED,
                "content_hash": fingerprints[a_id],
                "extraction_fingerprint": "other",
            }
        )
    )
    context = SimpleNamespace(
        incremental_delta=delta,
        incremental_fingerprints=fingerprints,
        incremental_extraction_fingerprint="mine",
        incremental_scope=None,
        incremental_retired_scopes=None,
        pipeline_id=None,
    )
    pipe = _resumed_pipeline(harness.store)
    assert pipe._delta_is_stale(context, "graph_extraction")
    context.incremental_extraction_fingerprint = "other"
    assert not pipe._delta_is_stale(context, "graph_extraction")
