# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Processing attempts: finding the document that kills the run.

A document whose processing kills the process (out of memory, a parser or
model client that crashes, a hard task timeout) would stop every later run
the same way. Each incremental run therefore records, on the registry record
of every document it is about to process, that an attempt is open
(``DocStatusRecord.attempt_batch``: how many documents the run processes
together). It opens the attempt right after the diff, before any stage
processes the documents, and closes it whenever the run ends in the process
(it finished, a stage failed, or it was stopped, SIGTERM included). Only a
run that died leaves it open, and the next run settles it
(:func:`settled_attempts`): one more consecutive death of that content, by a
run that processed ``attempt_batch`` documents together (``crash_batch``).
A full rebuild (``indexing.reset``) opens no attempt.

Once a document's consecutive deaths reach ``indexing.max_document_attempts``
the next run plans around it (:func:`plan_attempts`):

- a document that died processed alone is quarantined: it alone killed the
  run;
- otherwise the culprit is one of the documents it died with. The run
  processes only half of the suspects that died in the smallest batch and
  defers every other new or changed document. If it dies, that half becomes
  the smallest batch; if it ends, the half is committed. Each run halves the
  suspects, so one culprit among ``n`` documents is quarantined after about
  ``log2(n) + 1`` runs, and the run that quarantines it processes the other
  suspects again, all together.

A document is quarantined only after it died alone, so a culprit never takes
innocent documents with it. A run that dies whatever it processes (a broken
deployment) quarantines the delta's documents one at a time.
"""

from __future__ import annotations

from collections.abc import Mapping

from unified_kg_rag.domain.models import DocStatus, DocStatusRecord, DocumentDelta
from unified_kg_rag.shared import get_logger

__all__ = [
    "closed_attempt",
    "opened_attempt",
    "plan_attempts",
    "release_quarantined",
    "settled_attempts",
    "unproductive_attempt",
]

logger = get_logger(__name__)


def settled_attempts(
    record: DocStatusRecord | None, content_hash: str
) -> tuple[int, int]:
    """``(deaths, crash batch)`` of ``record``'s attempts at ``content_hash``.

    Counts the attempt still open (its run died) as one more death in a batch
    of ``attempt_batch`` documents; ``(0, 0)`` when the attempts were at other
    content, or none was made.
    """
    if record is None or record.attempted_hash != content_hash:
        return 0, 0
    if record.attempt_batch is not None:
        return record.attempt_count + 1, record.attempt_batch
    return record.attempt_count, record.crash_batch


def opened_attempt(
    record: DocStatusRecord, content_hash: str, batch: int
) -> DocStatusRecord:
    """``record`` with an attempt at ``content_hash`` open, in a run that
    processes ``batch`` documents (an earlier open one settled first)."""
    deaths, crash_batch = settled_attempts(record, content_hash)
    return record.model_copy(
        update={
            "attempt_count": deaths,
            "crash_batch": crash_batch,
            "attempted_hash": content_hash,
            "attempt_batch": batch,
        }
    )


def closed_attempt(record: DocStatusRecord) -> DocStatusRecord:
    """``record`` with its open attempt closed: the run ended in the process,
    so the attempt is not a death."""
    return record.model_copy(update={"attempt_batch": None})


def unproductive_attempt(record: DocStatusRecord, content_hash: str) -> DocStatusRecord:
    """``record`` after a run that ended without a text unit to record for
    the document: counted as a death of the document alone, so a document
    that never produces one is quarantined instead of re-extracted every run."""
    deaths = record.attempt_count if record.attempted_hash == content_hash else 0
    return record.model_copy(
        update={
            "attempt_count": deaths + 1,
            "crash_batch": 1,
            "attempted_hash": content_hash,
            "attempt_batch": None,
        }
    )


def plan_attempts(
    delta: DocumentDelta,
    stored: Mapping[str, DocStatusRecord],
    fingerprints: Mapping[str, str],
    max_attempts: int,
) -> None:
    """Quarantine or defer documents of ``delta`` whose processing keeps dying.

    ``stored`` holds the changed documents' records. A changed document whose
    deaths at its current content (:func:`settled_attempts`) reached
    ``max_attempts`` is a suspect. Suspects that died alone move to
    ``delta.quarantined``; when there are none, half of the suspects that died
    in the smallest batch (the most recent deaths first, then by doc id) are
    processed and every other new or changed document moves to
    ``delta.deferred`` (see the module docstring).
    """
    suspects = {
        doc_id: attempts
        for doc_id in delta.changed
        if (attempts := settled_attempts(stored.get(doc_id), fingerprints[doc_id]))[0]
        >= max_attempts
    }
    if not suspects:
        return
    alone = sorted(doc_id for doc_id, (_, batch) in suspects.items() if batch == 1)
    if alone:
        delta.changed = [doc_id for doc_id in delta.changed if doc_id not in alone]
        delta.quarantined = alone
        logger.warning(
            "Quarantining %d documents whose processing killed the run %d or "
            "more consecutive runs, the last one processing it alone (or that "
            "produced no text units to record): they are skipped until their "
            "content changes or indexing.max_document_attempts is raised: %s",
            len(alone),
            max_attempts,
            ", ".join(_label(stored, doc_id) for doc_id in alone),
        )
        return
    smallest = min((batch, -deaths) for deaths, batch in suspects.values())
    group = sorted(
        doc_id
        for doc_id, (deaths, batch) in suspects.items()
        if (batch, -deaths) == smallest
    )
    batch = set(group[: (len(group) + 1) // 2])
    delta.deferred = [doc_id for doc_id in delta.to_process if doc_id not in batch]
    delta.new = [doc_id for doc_id in delta.new if doc_id in batch]
    delta.changed = [doc_id for doc_id in delta.changed if doc_id in batch]
    logger.warning(
        "Runs keep dying while processing %d documents (%d or more times in a "
        "row): isolating the one that kills them. This run processes %d of "
        "the %d that died together most recently and defers %d other new or "
        "changed documents to later runs: %s",
        len(suspects),
        max_attempts,
        len(batch),
        len(group),
        len(delta.deferred),
        ", ".join(_label(stored, doc_id) for doc_id in sorted(batch)),
    )


def release_quarantined(
    delta: DocumentDelta,
    stored: Mapping[str, DocStatusRecord],
    max_attempts: int,
) -> None:
    """Retry quarantined documents the current limit no longer quarantines.

    ``stored`` holds the records of ``delta.quarantined_unchanged``. A
    QUARANTINED record whose deaths (``attempt_count``) are below
    ``max_attempts`` (the limit was raised since) moves from ``unchanged`` to
    ``changed``; its deaths keep counting from there.
    """
    released = sorted(
        doc_id
        for doc_id in delta.quarantined_unchanged
        if (record := stored.get(doc_id)) is not None
        and record.status is DocStatus.QUARANTINED
        and record.attempt_count < max_attempts
    )
    if not released:
        return
    delta.unchanged = [doc_id for doc_id in delta.unchanged if doc_id not in released]
    delta.changed.extend(released)
    logger.warning(
        "Retrying %d quarantined documents: indexing.max_document_attempts (%d) "
        "is now above their deaths: %s",
        len(released),
        max_attempts,
        ", ".join(_label(stored, doc_id) for doc_id in released),
    )


def _label(stored: Mapping[str, DocStatusRecord], doc_id: str) -> str:
    record = stored.get(doc_id)
    return (record.file_path if record is not None else None) or doc_id
