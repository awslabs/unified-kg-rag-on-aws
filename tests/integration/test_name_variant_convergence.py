# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental runs over two spellings of one name converge with a full build
(embedded stores).

"Initech Corp" and "Initech" share a key; a full build merges them when their
types agree and keeps them apart when they do not (the word a sentence trades
in sets the type). An incremental run adding a document must end in the
state a full build of the final corpus gives: no claim of one spelling lost
when the other's claim takes its place, no duplicate edge between the
entities two spellings merged into, and the same state when a finished
pipeline id is run again.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration import test_ingestion_stages as stages
from tests.integration.test_embedded_e2e import (
    DIM,
    FakeCrossEncoder,
    _config,
    _errors,
    _ingest,
    _providers,
    _stored,
)
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.adapters.storage.sqlite_graph_indexer import SQLiteGraphIndexer
from unified_kg_rag.domain.models import PipelineStageStatus

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

no_aws = e2e.no_aws

NAMES = (
    "Initech Corp",
    "Initech",
    "Buyer",
    "Depot",
    "Carrier",
    "Acme Inc",
    "Acme Corp",
    "Acme",
)
TYPES = {"goods": "ORGANIZATION", "parts": "PERSON", "cash": "LOCATION"}


class _Died(BaseException):
    """A crash the pipeline does not catch (the task was killed)."""


def _dying(method: str, nth: int, after: bool) -> Callable[..., Any]:
    """``OpenSearchIndexer.<method>`` whose ``nth`` call kills the run (after
    the call completes, if ``after``)."""
    original = getattr(OpenSearchIndexer, method)
    calls = {"n": 0}

    def dying(self: OpenSearchIndexer, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] != nth:
            return original(self, *args, **kwargs)
        if after:
            original(self, *args, **kwargs)
        raise _Died

    return dying


def _names_in(text: str) -> list[str]:
    found, rest = [], text
    for name in sorted(NAMES, key=len, reverse=True):
        pattern = rf"\b{re.escape(name)}\b"
        if re.search(pattern, rest):
            found.append(name)
            rest = re.sub(pattern, " ", rest)
    return sorted(found, key=text.find)


def _respond(system: str, human: str) -> str:
    if "graph refinement specialist" in system:
        # Gleaning adds an edge between the first and last name of a unit
        # that names three or more.
        text = stages._source_text(human).strip()
        names = _names_in(text)
        issue = (
            "<issue><issue_type>MISSING_RELATIONSHIP</issue_type><details>"
            f"<source>{names[0]}</source><target>{names[-1]}</target>"
            f"<type>DEALS_WITH</type><description>{names[0]} deals with "
            f"{names[-1]}</description></details>"
            f"<text_evidence>{text}</text_evidence></issue>"
            if len(names) >= 3
            else ""
        )
        return (
            "<refinement_plan><identified_issues>"
            f"{issue}</identified_issues></refinement_plan>"
        )
    if "extraction" not in system:
        return stages._respond(system, human)
    text = stages._source_text(human).strip()
    names = _names_in(text)
    if "knowledge graph extraction expert" in system:
        type_ = next((v for k, v in TYPES.items() if k in text), "ORGANIZATION")
        entities = "".join(
            f"<entity><name>{n}</name><type>{type_}</type>"
            f"<description>{n}: {text}</description><confidence>{7 + len(text) % 3}</confidence>"
            f"<source_text>{n}</source_text></entity>"
            for n in names
        )
        relationships = "".join(
            f"<relationship><source>{a}</source><target>{b}</target>"
            f"<type>WORKS_WITH</type><description>{a}-{b}: {text}</description>"
            f"<strength>{1 + len(text) % 9}</strength><source_text>{text}</source_text>"
            "</relationship>"
            for a, b in pairwise(names)
        )
        return f"<entities>{entities}</entities><relationships>{relationships}</relationships>"
    if "claim extraction specialist" in system:
        if len(names) < 2:
            return "<claims></claims>"
        subject, obj = names[:2]
        status = "TRUE" if "goods" in text else "SUSPECTED"
        return (
            f"<claims><claim><subject>{subject}</subject><object>{obj}</object>"
            f"<claim_type>OBLIGATION</claim_type><claim_status>{status}"
            f"</claim_status><description>{subject} owes {obj}: {text}"
            f"</description><source_text>{text}</source_text></claim></claims>"
        )
    return stages._respond(system, human)


def _state(
    tmp: Path,
    corpora: list[dict[str, str]],
    repeat: int | None = None,
    respond: Callable[[str, str], str] = _respond,
    summarize: bool = False,
    die_in: int | None = None,
    die_at: tuple[str, int, bool] = ("upsert_claims", 1, False),
    gleaning: bool = False,
) -> dict[str, list[tuple]]:
    """The stored entities, edges and claims after indexing ``corpora`` one
    after the other (run ``repeat`` twice under its pipeline id; run
    ``die_in`` killed at ``die_at``, then resumed under its id).

    ``die_at`` is a vector-store method, the call that dies (1 = the first)
    and whether that call completes before the run dies."""
    config = _config(tmp / "stores", tmp / "viz")
    config.processing.claim_extraction.enabled = True
    config.processing.gleaning.enabled = gleaning
    config.processing.gleaning.max_rounds = 1
    summarization = config.processing.graph_extraction.description_summarization
    summarization.enabled = summarize
    summarization.force_summary_threshold_tokens = 30
    # Document ids (and so the text-unit order ties follow) independent of
    # the build's directory: both builds index the same documents.
    config.processing.document_parsing.source_scope = "fuzz-corpus"
    providers = _providers(config, FakeOllama(respond, "x", DIM), FakeCrossEncoder("r"))
    source = tmp / "corpus"
    for run, corpus in enumerate(corpora):
        if source.exists():
            shutil.rmtree(source)
        source.mkdir(parents=True)
        for name, text in corpus.items():
            (source / name).write_text(text)
        if run == die_in:
            with (
                patch.object(OpenSearchIndexer, die_at[0], _dying(*die_at)),
                pytest.raises(_Died),
            ):
                _ingest(config, providers, source, tmp / f"run-{run}")
        for _ in range(2 if run == repeat else 1):
            context = _ingest(config, providers, source, tmp / f"run-{run}")
            assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    graph = providers.build_graph_indexer()
    try:
        assert isinstance(graph, SQLiteGraphIndexer)
        ids = [i for i, _ in graph.read_entity_names()]
        entities = graph.read_entities(ids)
        edges = graph.read_relationships(graph.find_incident_relationship_ids(ids))
    finally:
        graph.close()
    claims = _stored(config, providers, config.indexing.opensearch.claims_index_prefix)
    return {
        "entities": sorted(
            (e.id, e.name, e.type, len(e.text_unit_ids or []), e.description)
            for e in entities
        ),
        "edges": sorted(
            (r.source_name, r.target_name, r.weight, r.description) for r in edges
        ),
        "claims": sorted(
            (
                hit["_source"]["subject_name"],
                hit["_source"]["object_name"],
                hit["_source"]["status"],
                hit["_source"]["description"],
            )
            for hit in claims
        ),
    }


@pytest.mark.parametrize(
    "runs",
    [
        pytest.param(
            [
                {
                    "a.txt": "Initech Corp trade cash 2. Depot and Buyer trade "
                    "cash 1. Depot and Initech Corp and Initech trade cash 1."
                },
                {
                    "a.txt": "Initech Corp trade cash 2. Depot and Buyer trade "
                    "cash 1. Depot and Initech Corp and Initech trade cash 1.",
                    "b.txt": "Depot and Initech trade parts 1.",
                },
            ],
            id="a-claim-the-other-spelling-merged-is-split-back",
        ),
        pytest.param(
            [
                {
                    "a.txt": "Initech Corp and Depot and Buyer trade parts 2. "
                    "Buyer and Initech and Initech Corp trade parts 2."
                },
                {
                    "a.txt": "Initech Corp and Depot and Buyer trade parts 2. "
                    "Buyer and Initech and Initech Corp trade parts 2.",
                    "b.txt": "Initech Corp and Initech and Depot trade goods 2. "
                    "Initech Corp and Depot trade parts 1.",
                },
            ],
            id="no-duplicate-edge-when-two-spellings-merge",
        ),
        pytest.param(
            [
                {
                    "d7.txt": "Initech trade goods 2.",
                    "d3.txt": "Initech trade cash 1. Initech and Initech Corp "
                    "and Acme trade parts 3.",
                },
                {
                    "d3.txt": "Initech trade cash 1. Initech and Initech Corp "
                    "and Acme trade parts 3."
                },
            ],
            id="a-deletion-merges-with-a-stored-spelling-of-no-shared-unit",
        ),
        pytest.param(
            [
                {"d3.txt": "Acme trade goods 2. Depot trade cash 1."},
                {
                    "d1.txt": "Acme trade parts 2.",
                    "d4.txt": "Initech and Depot trade goods 2. Initech Corp "
                    "and Acme Inc and Carrier trade cash 2.",
                    "d3.txt": "Acme trade goods 2. Depot trade cash 1.",
                    "d6.txt": "Initech and Carrier and Acme Inc trade cash 1.",
                    "d7.txt": "Acme Inc and Initech Corp trade goods 2.",
                },
            ],
            id="a-split-keeps-the-stored-spelling",
        ),
        pytest.param(
            [
                {"x.txt": "Acme Inc and Acme Corp trade goods 3."},
                {
                    "x.txt": "Acme Inc and Acme Corp trade goods 3.",
                    "y.txt": "Acme Inc and Acme trade cash 2.",
                },
            ],
            id="a-claim-of-a-third-spelling-is-not-kept-unresolved",
        ),
        pytest.param(
            [
                {
                    "d5.txt": "Buyer and Acme and Acme Inc trade cash 1. Depot and "
                    "Acme Inc and Acme Corp trade cash 2. Acme trade cash 3.",
                    "d0.txt": "Acme Corp and Acme Inc trade goods 3. Buyer trade "
                    "parts 3.",
                },
                {
                    "d0.txt": "Acme Corp and Acme Inc trade goods 3. Buyer trade "
                    "parts 3."
                },
            ],
            id="a-stored-claim-no-claim-of-its-units-matches-is-replaced",
        ),
        pytest.param(
            [
                {
                    "d5.txt": "Acme Inc and Acme trade parts 2. Acme Inc and Acme "
                    "Corp and Acme trade parts 1. Acme Corp trade parts 3."
                },
                {"d3.txt": "Acme Inc and Buyer and Acme Corp trade cash 3."},
                {
                    "d3.txt": "Acme Inc and Buyer and Acme Corp trade cash 3.",
                    "d4.txt": "Depot and Acme trade parts 1. Acme and Depot trade "
                    "cash 2.",
                },
            ],
            id="a-claim-names-the-spelling-its-unit-states",
        ),
    ],
)
def test_incremental_runs_over_two_spellings_match_a_full_build(
    tmp_path: Path, runs: list[dict[str, str]]
) -> None:
    full = _state(tmp_path / "full", [runs[-1]])
    incremental = _state(tmp_path / "inc", runs)

    assert incremental == full


def test_running_a_finished_pipeline_id_again_changes_nothing(
    tmp_path: Path,
) -> None:
    statement = "Initech Corp and Buyer and Acme trade cash 1."
    first = "Depot and Initech trade parts 1. Initech Corp trade cash 3."
    second = f"{first} Initech and Buyer trade cash 2."
    runs = [
        {"d5.txt": statement, "d6.txt": first},
        {"d6.txt": second, "r5.txt": statement},
        {
            "d6.txt": second,
            "r5.txt": statement,
            "d5.txt": "Buyer and Initech and " "Depot trade goods 2.",
        },
    ]
    full = _state(tmp_path / "full", [runs[-1]])
    incremental = _state(tmp_path / "inc", runs, repeat=1)

    assert ("Initech Corp", "Buyer") in {c[:2] for c in full["claims"]}
    assert incremental == full


@pytest.mark.parametrize("spelling", ["Acme", "Acme Inc"])
def test_a_hub_grown_under_another_spelling_extends_its_summary(
    tmp_path: Path, spelling: str
) -> None:
    # Regression: a delta entity that merged into a stored one of another
    # spelling was looked up under its own id, so the hub was summarized
    # again from every line its text units state instead of from its stored
    # summary and the one added line.
    calls: list[tuple[str, int]] = []

    def respond(system: str, human: str) -> str:
        if "knowledge-graph editor" not in system:
            return _respond(system, human)
        match = re.search(r"<descriptions>\n(.*?)\n</descriptions>", human, re.S)
        name = re.search(r"descriptions of '([^']*)'", human)
        assert match and name
        calls.append((name.group(1), match.group(1).count("\n") + 1))
        return f"Summary of {name.group(1)}."

    hub = {f"h{i}.txt": f"Acme and Buyer trade goods {i}." for i in range(8)}
    grown = {**hub, "new.txt": f"{spelling} and Depot trade goods 9."}
    _state(tmp_path, [hub, grown], respond=respond, summarize=True)

    # Eight lines in the first run; one call (or none) on the summary and
    # the added line in the second.
    assert [lines for name, lines in calls if name == "Acme"] in ([8], [8, 2])


def test_a_run_resumed_after_dying_mid_split_matches_a_full_build(
    tmp_path: Path,
) -> None:
    # The second run splits "Initech" (now a location) off "Initech Corp".
    # Killed after the graph took the split entities but before their edges
    # and claims were written, the resumed run found nothing left to split:
    # the edge and the claim "Initech Corp" had absorbed kept the other
    # spelling's statement (an edge of double weight, a claim about itself).
    first = {"r6.txt": "Initech Corp and Buyer and Initech trade goods 1."}
    final = {**first, "d6.txt": "Initech and Buyer trade cash 2."}

    full = _state(tmp_path / "full", [final])
    incremental = _state(tmp_path / "inc", [first, final], die_in=1)

    assert ("Initech Corp", "Initech") in {c[:2] for c in full["claims"]}
    assert incremental == full


D2 = (
    "Initech and Initech Corp trade parts 1. Buyer and Initech Corp and Depot "
    "trade goods 1."
)
D5 = (
    "Initech and Initech Corp trade goods 2. Initech and Depot and Initech Corp "
    "trade cash 3. Initech and Buyer trade goods 1."
)


@pytest.mark.parametrize(
    ("first", "kept"),
    [
        pytest.param({"d5.txt": D5, "d2.txt": D2}, {"d2.txt": D2}, id="edge"),
        pytest.param(
            {
                "d5.txt": D5,
                "d0.txt": "Initech Corp and Buyer and Initech trade cash 2.",
                "d2.txt": D2,
                "d6.txt": "Depot and Buyer and Initech Corp trade parts 2.",
            },
            {"d2.txt": D2, "d6.txt": "Depot and Buyer and Initech Corp trade parts 2."},
            id="edge-with-stripped-endpoints",
        ),
    ],
)
def test_a_removal_resumed_after_dying_between_the_stores_matches_a_full_build(
    tmp_path: Path, first: dict[str, str], kept: dict[str, str]
) -> None:
    # Removing d5.txt re-keys a stored Initech Corp-Buyer edge. The run is
    # killed after the old id's vector copy is deleted and before its graph
    # copy is. The resumed removal looked for edges only among those of the
    # entities citing the removed units, which the dead run had stripped, so
    # it never found the old edge and a duplicate stayed. Found by its text
    # units, the edge is derived with its endpoints, which no longer cite
    # them.
    full = _state(tmp_path / "full", [kept])
    incremental = _state(
        tmp_path / "inc",
        [first, kept],
        die_in=1,
        die_at=("delete_document_artifacts", 2, True),
    )

    assert incremental == full


def test_a_resumed_merge_keeps_a_gleaned_edge_on_the_spelling_it_names(
    tmp_path: Path,
) -> None:
    # d7.txt moves to r0.txt. Removing d7.txt merges "Initech Corp" into
    # "Initech" (their types now agree); indexing r0.txt splits them again.
    # Killed after the split entities were written and before their edges,
    # the resumed run left the edge gleaning adds for "Initech Corp" (d1.txt)
    # on "Initech": the stored "Initech" no longer held the other spelling's
    # key, so the edge had no derived entity to follow.
    d1 = "Initech Corp and Initech trade cash 1. Depot and Buyer trade parts 1."
    d7 = "Depot and Initech Corp and Buyer trade goods 1."
    runs = [
        {"d1.txt": "Initech trade parts 2."},
        {"d1.txt": d1, "d5.txt": "Buyer trade cash 2."},
        {"d1.txt": d1, "d5.txt": "Buyer trade cash 2.", "d7.txt": d7},
        {"d1.txt": d1, "d5.txt": "Buyer trade cash 2.", "r0.txt": d7},
    ]

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=3,
        die_at=("update_relationship_lineage", 2, False),
        gleaning=True,
    )

    assert ("Initech Corp", "Buyer") in {e[:2] for e in full["edges"]}
    assert incremental == full


def test_a_hub_that_changes_its_name_extends_its_summary(tmp_path: Path) -> None:
    # Nine new documents call the hub "Acme Inc", one more than say "Acme":
    # the delta entity merges into the stored "Acme", which the derivation
    # then re-keys to "Acme Inc". The delta's descriptions must be found
    # under that final id, or the hub is summarized again from every line.
    calls: list[tuple[str, int]] = []

    def respond(system: str, human: str) -> str:
        if "knowledge-graph editor" not in system:
            return _respond(system, human)
        match = re.search(r"<descriptions>\n(.*?)\n</descriptions>", human, re.S)
        name = re.search(r"descriptions of '([^']*)'", human)
        assert match and name
        calls.append((name.group(1), match.group(1).count("\n") + 1))
        return f"Summary of {name.group(1)}."

    hub = {f"h{i}.txt": f"Acme and Buyer trade goods {i}." for i in range(8)}
    grown = {
        **hub,
        **{f"n{i}.txt": f"Acme Inc and Depot trade goods {i}." for i in range(9)},
    }
    _state(tmp_path, [hub, grown], respond=respond, summarize=True)

    hub_calls = [lines for name, lines in calls if name in ("Acme", "Acme Inc")]
    # Eight lines in the first run; in the second the nine new lines (graph
    # resolution), then the stored summary and that one at most: never all
    # seventeen.
    assert hub_calls[:2] == [8, 9]
    assert all(lines <= 2 for lines in hub_calls[2:])


@pytest.mark.parametrize(
    "runs",
    [
        pytest.param(
            [
                {"a.txt": "Acme and Depot trade goods 1."},
                {
                    "a.txt": "Acme and Depot trade goods 1.",
                    "b.txt": "Acme Inc and Depot trade cash 1.",
                },
            ],
            id="spellings-kept-apart-in-other-units",
        ),
        pytest.param(
            [
                {f"h{i}.txt": f"Acme and Buyer trade goods {i}." for i in range(2)},
                {
                    **{
                        f"h{i}.txt": f"Acme and Buyer trade goods {i}."
                        for i in range(2)
                    },
                    **{
                        f"n{i}.txt": f"Acme Inc and Depot trade goods {i}."
                        for i in range(3)
                    },
                },
            ],
            id="a-spelling-re-keyed-to-the-other",
        ),
    ],
)
def test_only_a_half_written_split_marks_another_spelling_a_split_origin(
    tmp_path: Path, runs: list[dict[str, str]], mocker: Any
) -> None:
    # Another spelling of a derived entity is taken for a split origin (its
    # edges derived again, its claims followed) only when the two share a
    # text unit, as a split a killed run left half-written does, and not
    # when the derivation re-keyed it into the derived entity: those cost
    # reads and leave the result as it is.
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

    derive = mocker.spy(IndexingManager, "derive")

    _state(tmp_path, runs)

    results = [d for d in derive.spy_return_list if d.entities]
    assert results
    for derived in results:
        assert not set(derived.split_from.values()) & set(derived.id_remap)
        assert not derived.split_from
