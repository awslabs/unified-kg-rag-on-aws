# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The commit reads the stripped-summary note (``stripped-summaries``) of
every namespace its delta writes, and deletes it once its writes succeed; a
note it cannot read is kept for a later run. The note names the items an
earlier run's removal stripped for a commit that never ran: a commit that
did not read it would extend them (a summary of a summary) and leave the
note behind. AWS-free: the in-memory stores of ``IncrementalRun``."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from tests.fixtures.incremental_runs import IncrementalRun
from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
from unified_kg_rag.application.storage.stored_items import STRIPPED_SUMMARIES
from unified_kg_rag.domain.models import TextUnit

pytestmark = pytest.mark.unit

_TENANT = "tenant-b"


def _noted(harness: IncrementalRun) -> IncrementalIndexer:
    """A strip note in ``_TENANT`` (not the run's own namespace), and the
    indexer of a run whose commit writes there."""
    harness.vector.write_extraction_records(
        {STRIPPED_SUMMARIES: {"ids": ["e-stripped"]}}, suffix=_TENANT
    )
    return IncrementalIndexer(harness.store, harness.manager)


def _tenant_unit() -> TextUnit:
    # A text unit alone: the merge, which reads the notes of the namespaces
    # its entities are in, reads none here.
    return TextUnit(
        id="t-tenant",
        text="Vendor supplies widgets.",
        attributes={"index": _TENANT},
    )


def _note(harness: IncrementalRun) -> Any:
    return harness.vector.read_extraction_records(
        [STRIPPED_SUMMARIES], suffix=_TENANT
    ).get(STRIPPED_SUMMARIES)


def test_a_commit_reads_and_clears_the_strip_note_of_each_namespace_it_writes() -> None:
    # R4A-4: reading only the run's own namespace's note left this one,
    # and the next commit there extended the stripped items.
    harness = IncrementalRun()
    inc = _noted(harness)
    assert inc.suffix != _TENANT

    results = inc.commit([], {}, text_units=[_tenant_unit()])

    assert not any(stats.failed_items for stats in results.values())
    assert _note(harness) is None


def test_a_strip_note_the_commit_cannot_read_is_kept(caplog) -> None:
    # A failed read is no failed commit: the note stays for the next run,
    # whose merge reads it before it extends anything. A commit with no
    # delta (a resume after the removal) reads it only here.
    harness = IncrementalRun()
    inc = IncrementalIndexer(harness.store, harness.manager)
    own = inc.suffix
    harness.vector.write_extraction_records(
        {STRIPPED_SUMMARIES: {"ids": ["e-stripped"]}}, suffix=own
    )
    read = harness.vector.read_extraction_records

    def failing(keys: list[str], suffix: str | None = None) -> Any:
        if STRIPPED_SUMMARIES in keys:
            raise RuntimeError("extraction store unavailable")
        return read(keys, suffix=suffix)

    harness.vector.read_extraction_records = failing  # type: ignore[method-assign]
    caplog.set_level(logging.WARNING)

    inc.commit([], {})

    harness.vector.read_extraction_records = read  # type: ignore[method-assign]
    assert read([STRIPPED_SUMMARIES], suffix=own) == {
        STRIPPED_SUMMARIES: {"ids": ["e-stripped"]}
    }
    assert any(
        f"stripped-summary note of '{own}' failed" in r.getMessage()
        and "kept for the next run" in r.getMessage()
        for r in caplog.records
    )
