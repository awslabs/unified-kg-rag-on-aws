# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""How often an incremental run asks the model to summarize a description
(embedded stores).

A run's summaries are each made once and used: a summary graph resolution
made of the delta's lines is not made again by the indexing, and a summary a
removal makes of what an edited document's shared entity has left is not
summarized again by the commit (a summary of a summary where a full build
summarizes the lines), even when the run dies between the removal and the
commit.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_name_variant_convergence import _respond, _state
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.application.storage.indexing_manager import IndexingManager

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

no_aws = e2e.no_aws


def _recording(calls: list[tuple[str, int]]):
    """A model that answers as ``_respond`` and records each summarization
    call as (name, number of description lines)."""

    def respond(system: str, human: str) -> str:
        if "knowledge-graph editor" not in system:
            return _respond(system, human)
        match = re.search(r"<descriptions>\n(.*?)\n</descriptions>", human, re.S)
        name = re.search(r"descriptions of '([^']*)'", human)
        assert match and name
        calls.append((name.group(1), match.group(1).count("\n") + 1))
        return f"Summary of {name.group(1)}."

    return respond


def test_a_delta_summary_is_extended_not_made_again(tmp_path: Path) -> None:
    # "Acme" is stored unsummarized (two lines); four new documents add
    # four lines, which graph resolution summarizes. The indexing summarized
    # all six lines again, so the first summary was made in vain.
    calls: list[tuple[str, int]] = []
    hub = {f"h{i}.txt": f"Acme and Buyer trade goods {i}." for i in range(2)}
    grown = {
        **hub,
        **{f"n{i}.txt": f"Acme and Depot trade goods {i}." for i in range(4)},
    }

    state = _state(tmp_path, [hub, grown], respond=_recording(calls), summarize=True)

    assert [lines for name, lines in calls if name == "Acme"] == [4]
    acme = next(e for e in state["entities"] if e[1] == "Acme")
    assert acme[4].split("\n")[-1] == "Summary of Acme."


@pytest.mark.parametrize(
    ("spelling", "acme_calls"),
    [
        # The commit derives "Acme" again under its id: the removal leaves
        # its summary to the commit.
        ("Acme", [8, 8]),
        # The edited document spells it otherwise: the removal summarizes
        # the seven lines left, the commit the eight again, not that summary.
        ("Acme Inc", [8, 7, 8]),
    ],
)
def test_an_edited_document_summarizes_a_shared_entity_from_its_lines(
    tmp_path: Path, spelling: str, acme_calls: list[int]
) -> None:
    # Editing h0.txt strips its line from the summarized "Acme" and adds the
    # new one. The removal summarized the seven lines left, then the commit
    # extended that summary: "Acme" ended a summary of a summary, where a
    # full build summarizes the eight lines once.
    calls: list[tuple[str, int]] = []
    hub = {f"h{i}.txt": f"Acme and Buyer trade goods {i}." for i in range(8)}
    edited = {**hub, "h0.txt": f"{spelling} and Depot trade goods 9."}

    incremental = _state(
        tmp_path / "inc", [hub, edited], respond=_recording(calls), summarize=True
    )
    full = _state(tmp_path / "full", [edited], respond=_recording([]), summarize=True)

    assert [lines for name, lines in calls if name == "Acme"] == acme_calls
    # "Buyer" is not in the edited document: the removal summarizes it.
    assert [lines for name, lines in calls if name == "Buyer"] == [8, 7]
    assert incremental["entities"] == full["entities"]


def _digesting(system: str, human: str) -> str:
    """A model that answers as ``_respond`` and summarizes a description
    into a digest of its lines: a summary of a summary, or one keeping a
    removed line, differs from the summary of the lines. Not of the name: a
    delta's own spelling summarized before it folds into the stored one
    (graph resolution's summary, reused) has the lines of a full build."""
    if "knowledge-graph editor" not in system:
        return _respond(system, human)
    match = re.search(r"<descriptions>\n(.*?)\n</descriptions>", human, re.S)
    assert match
    lines = "\n".join(sorted(match.group(1).split("\n")))
    return f"Summary {hashlib.sha256(lines.encode()).hexdigest()[:8]}."


@pytest.mark.parametrize("spelling", ["Acme", "Acme Inc", "Acme GmbH"])
@pytest.mark.parametrize("edits", [4, 6])
def test_editing_several_documents_of_a_shared_entity_matches_a_full_build(
    tmp_path: Path, spelling: str, edits: int
) -> None:
    # Eight documents mention "Acme"; editing several strips their lines
    # from its summary. The removal left the summary to the commit (or
    # another spelling folded in), and the commit extended the stored
    # copy: six edits kept two removed lines before the summary, four
    # summarized the delta's summary again (a summary of a summary).
    hub = {f"h{i}.txt": f"Acme and Buyer trade goods {i}." for i in range(8)}
    edited = {
        **hub,
        **{
            f"h{i}.txt": f"{spelling} and Depot trade goods {i + 20}."
            for i in range(edits)
        },
    }

    incremental = _state(
        tmp_path / "inc", [hub, edited], respond=_digesting, summarize=True
    )
    full = _state(tmp_path / "full", [edited], respond=_digesting, summarize=True)

    assert incremental == full


NOTE = "stripped-summaries"


@pytest.fixture
def notes() -> Iterator[list[tuple[str, Any]]]:
    """The writes and deletes of the summary note, in order: ("write", its
    ids) or ("delete", None)."""
    events: list[tuple[str, Any]] = []
    write = OpenSearchIndexer.write_extraction_records
    delete = OpenSearchIndexer.delete_extraction_records

    def writing(self: Any, records: Any, *args: Any, **kwargs: Any) -> Any:
        if NOTE in records:
            events.append(("write", records[NOTE]["ids"]))
        return write(self, records, *args, **kwargs)

    def deleting(self: Any, keys: Any, *args: Any, **kwargs: Any) -> Any:
        if NOTE in keys:
            events.append(("delete", None))
        return delete(self, keys, *args, **kwargs)

    with (
        patch.object(OpenSearchIndexer, "write_extraction_records", writing),
        patch.object(OpenSearchIndexer, "delete_extraction_records", deleting),
    ):
        yield events


@pytest.mark.parametrize("dies", [True, False], ids=["dies", "finishes"])
@pytest.mark.parametrize("spelling", ["Acme", "Acme Inc", "Acme GmbH"])
@pytest.mark.parametrize("edits", [1, 4, 6])
def test_a_run_that_dies_between_the_strip_and_the_commit_matches_a_full_build(
    tmp_path: Path,
    notes: list[tuple[str, Any]],
    spelling: str,
    edits: int,
    dies: bool,
) -> None:
    # The run dies once its removal finished (the strip written, the retired
    # ids deleted) and before its commit: the resumed run's removal finds
    # nothing left citing the removed units, and its commit extended the
    # stripped copies (a summary of a summary, or kept removed lines). The
    # strip notes them before its writes; the commit deletes the note.
    hub = {f"h{i}.txt": f"Acme and Buyer trade goods {i}." for i in range(8)}
    edited = {
        **hub,
        **{
            f"h{i}.txt": f"{spelling} and Depot trade goods {i + 20}."
            for i in range(edits)
        },
    }

    incremental = _state(
        tmp_path / "inc",
        [hub, edited],
        respond=_digesting,
        summarize=True,
        die_in=1 if dies else None,
        die_at=("merge_with_existing_graph", 1, False, IndexingManager),
    )
    full = _state(tmp_path / "full", [edited], respond=_digesting, summarize=True)

    assert incremental == full
    assert notes and notes[0][0] == "write"
    assert notes[-1] == ("delete", None)
    if dies:
        # Not deleted by the run that died.
        assert [op for op, _ in notes].count("delete") == 1


def test_a_run_that_only_deletes_documents_deletes_its_summary_note(
    tmp_path: Path, notes: list[tuple[str, Any]]
) -> None:
    # Its removal strips "Acme" and notes it; its commit has no delta and
    # finishes all the same: a later run may extend the stored summary.
    hub = {f"h{i}.txt": f"Acme and Buyer trade goods {i}." for i in range(8)}
    kept = {k: v for k, v in hub.items() if k not in ("h0.txt", "h1.txt")}

    _state(tmp_path, [hub, kept], respond=_digesting, summarize=True)

    assert [op for op, _ in notes] == ["write", "delete"]


def test_a_summary_note_left_behind_is_deleted_by_the_next_commit(
    tmp_path: Path, notes: list[tuple[str, Any]]
) -> None:
    # The edit's commit cannot delete the note (best-effort); the next run
    # only adds a document, so no removal reads it: its commit reads it
    # (summarizing the noted items from their records, as a full build,
    # rather than extending them) and deletes it, so it does not outlive
    # the next commit of the suffix.
    hub = {f"h{i}.txt": f"Acme and Buyer trade goods {i}." for i in range(8)}
    edited = {**hub, "h0.txt": "Acme Inc and Depot trade goods 9."}
    grown = {**edited, "n0.txt": "Acme and Carrier trade goods 10."}
    delete = OpenSearchIndexer.delete_extraction_records
    failed = {"n": 0}

    def failing_once(self: Any, keys: Any, *args: Any, **kwargs: Any) -> Any:
        if NOTE in keys and not failed["n"]:
            failed["n"] += 1
            raise ConnectionError("store down")
        return delete(self, keys, *args, **kwargs)

    with patch.object(OpenSearchIndexer, "delete_extraction_records", failing_once):
        _state(tmp_path, [hub, edited, grown], respond=_digesting, summarize=True)

    assert failed["n"] == 1
    assert [op for op, _ in notes] == ["write", "delete"]
