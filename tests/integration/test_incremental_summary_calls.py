# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""How often an incremental run asks the model to summarize a description
(embedded stores).

A run's summaries are each made once and used: a summary graph resolution
made of the delta's lines is not made again by the indexing, and a summary a
removal makes of what an edited document's shared entity has left is not
summarized again by the commit (a summary of a summary where a full build
summarizes the lines).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_name_variant_convergence import _respond, _state

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
