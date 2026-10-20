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
from opensearchpy.exceptions import NotFoundError

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
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
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
    "ACME Inc",
    "Acme Corp",
    "Acme Ltd",
    "Acme LLC",
    "Acme",
    "ACME",
)
TYPES = {"goods": "ORGANIZATION", "parts": "PERSON", "cash": "LOCATION"}


class _Died(BaseException):
    """A crash the pipeline does not catch (the task was killed)."""


def _dying(
    method: str, nth: int, after: bool, owner: type = OpenSearchIndexer
) -> Callable[..., Any]:
    """``owner.<method>`` whose ``nth`` call kills the run (after the call
    completes, if ``after``)."""
    original = getattr(owner, method)
    calls = {"n": 0}

    def dying(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] != nth:
            return original(self, *args, **kwargs)
        if after:
            original(self, *args, **kwargs)
        raise _Died

    return dying


def _failing(owner: type, method: str) -> Callable[..., Any]:
    """``owner.<method>`` whose first call writes nothing and reports every
    item failed."""
    original = getattr(owner, method)
    calls = {"n": 0}

    def failing(self: Any, items: Any, *args: Any, **kwargs: Any) -> Any:
        from unified_kg_rag.ports.indexer import IndexingStats

        calls["n"] += 1
        if calls["n"] != 1:
            return original(self, items, *args, **kwargs)
        stats = IndexingStats(total_items=len(items))
        stats.add_error("store down", count=len(items))
        return stats

    return failing


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
    die_at: tuple[Any, ...] = ("upsert_claims", 1, False),
    gleaning: bool = False,
    suffix: str | None = None,
    resume: bool = True,
    fail_resumed_at: tuple[Any, ...] | None = None,
) -> dict[str, list[tuple]]:
    """The stored entities, edges and claims after indexing ``corpora`` one
    after the other (run ``repeat`` twice under its pipeline id; run
    ``die_in`` killed at ``die_at``, then resumed under its id, or, without
    ``resume``, followed by the next corpus).

    ``die_at`` is a method, the call that dies (1 = the first), whether
    that call completes before the run dies and, optionally, the class
    (default the vector store's). ``fail_resumed_at`` (a method and its
    class) makes the resumed run's first call of it report every item
    failed; that run fails and is resumed once more."""
    config = _config(tmp / "stores", tmp / "viz")
    config.processing.claim_extraction.enabled = True
    config.processing.gleaning.enabled = gleaning
    if suffix:
        config.indexing.additional_suffix = suffix
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
            owner = die_at[3] if len(die_at) > 3 else OpenSearchIndexer
            with (
                patch.object(owner, die_at[0], _dying(*die_at)),
                pytest.raises(_Died),
            ):
                _ingest(config, providers, source, tmp / f"run-{run}")
            if not resume:
                continue
            if fail_resumed_at:
                method, owner = fail_resumed_at
                with patch.object(owner, method, _failing(owner, method)):
                    context = _ingest(config, providers, source, tmp / f"run-{run}")
                assert context.status is not PipelineStageStatus.COMPLETED
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
    try:
        vector_edges = _stored(
            config, providers, config.indexing.opensearch.relationships_index_prefix
        )
    except NotFoundError:  # no edge was ever written
        vector_edges = []
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
                tuple(sorted(hit["_source"].get("text_unit_ids") or [])),
            )
            for hit in claims
        ),
        # The vector copies of the edges: their own names and lineage.
        "vector_edges": sorted(
            (
                hit["_source"].get("source_name"),
                hit["_source"].get("target_name"),
                hit["_source"].get("description"),
                tuple(sorted(hit["_source"].get("text_unit_ids") or [])),
            )
            for hit in vector_edges
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
        pytest.param(
            [
                {"d1.txt": "ACME Inc and Acme and Buyer trade goods 1."},
                {
                    "d1.txt": "ACME Inc and Acme and Buyer trade goods 1.",
                    "d2.txt": "Acme Corp trade cash 1.",
                    "r2.txt": "Acme Ltd and Acme and Acme Corp trade parts 3.",
                },
            ],
            id="a-claim-resolves-with-the-other-spellings-the-run-derived",
        ),
        pytest.param(
            [
                {
                    "r6.txt": "ACME Inc trade cash 2. Initech and Buyer and ACME Inc "
                    "trade cash 2.",
                    "d1.txt": "Buyer and Acme Inc trade cash 3.",
                },
                {
                    "r6.txt": "ACME Inc trade cash 2. Initech and Buyer and ACME Inc "
                    "trade cash 2."
                },
            ],
            id="a-claim-names-an-entity-renamed-under-its-id",
        ),
        pytest.param(
            [
                {
                    "r6.txt": "ACME Inc trade cash 2. Initech and Buyer and ACME Inc "
                    "trade cash 2."
                },
                {
                    "r6.txt": "ACME Inc trade cash 2. Initech and Buyer and ACME Inc "
                    "trade cash 2.",
                    "d1.txt": "Buyer and Acme Inc trade cash 3.",
                },
            ],
            id="a-claim-names-an-entity-a-delta-renamed-under-its-id",
        ),
        pytest.param(
            [
                {
                    "r6.txt": "ACME Inc trade cash 2. Initech and Buyer and ACME Inc "
                    "trade cash 2."
                },
                {
                    "r6.txt": "ACME Inc trade cash 2. Initech and Buyer and ACME Inc "
                    "trade cash 2.",
                    "a0.txt": "Acme Inc trade cash 3.",
                },
            ],
            id="a-claim-names-an-entity-a-delta-without-claims-renamed",
        ),
        pytest.param(
            [
                {"d0.txt": "Acme Ltd and Acme Corp and Buyer trade cash 2."},
                {
                    "d0.txt": "Acme Ltd and Acme Corp and Buyer trade cash 2.",
                    "d3.txt": "Acme Corp and Acme trade cash 1. Acme Ltd trade "
                    "parts 1.",
                },
            ],
            id="a-claim-cites-the-units-its-statements-are-in",
        ),
        pytest.param(
            [
                {"d0.txt": "Acme Ltd and Acme Corp and Buyer trade cash 2."},
                {
                    "d0.txt": "Acme Ltd and Acme Corp and Buyer trade cash 2.",
                    "d6.txt": "Acme Ltd and Depot trade goods 1. Acme trade goods 2.",
                    "d3.txt": "Acme Corp and Acme trade cash 1. Acme Ltd trade "
                    "parts 1.",
                },
                {
                    "d0.txt": "Acme LLC and Acme Corp and Buyer trade cash 2.",
                    "d6.txt": "Acme Ltd and Depot trade goods 1. Acme trade goods 2.",
                },
            ],
            id="a-removed-documents-claim-goes-with-it",
        ),
        pytest.param(
            [
                {
                    "d3.txt": "Depot and Acme Ltd and Acme Inc trade cash 2. Buyer "
                    "and Acme trade goods 3. Acme trade cash 1."
                },
                {
                    "d3.txt": "Depot and Acme Ltd and Acme Inc trade cash 2. Buyer "
                    "and Acme trade goods 3. Acme trade cash 1.",
                    "d7.txt": "Acme and Buyer and Acme Inc trade parts 1.",
                },
            ],
            id="a-claim-follows-its-spelling-split-off-another-entity",
        ),
        pytest.param(
            [
                {
                    "d3.txt": "Buyer and Acme Ltd and ACME Inc trade cash 3. Buyer "
                    "trade parts 1. Acme and Acme Inc trade cash 2.",
                    "d2.txt": "Acme Ltd trade parts 2. Initech Corp trade parts 2.",
                },
                {
                    "d3.txt": "Buyer and Acme Ltd and ACME Inc trade cash 3. Buyer "
                    "trade parts 1. Acme and Acme Inc trade cash 2."
                },
            ],
            id="a-claim-follows-its-spelling-a-removal-splits-off",
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
def test_a_spelling_re_keyed_into_another_is_not_a_split_origin(
    tmp_path: Path, runs: list[dict[str, str]], mocker: Any
) -> None:
    # Another spelling of a derived entity is taken for a split origin (its
    # edges derived again, its claims followed), but not when the derivation
    # re-keyed it into the derived entity: that costs reads and leaves the
    # result as it is.
    derive = mocker.spy(IndexingManager, "derive")

    _state(tmp_path, runs)

    results = [d for d in derive.spy_return_list if d.entities]
    assert results
    for derived in results:
        assert not set(derived.split_from.values()) & set(derived.id_remap)


def test_a_resumed_split_moves_an_edge_of_a_spelling_sharing_no_unit(
    tmp_path: Path,
) -> None:
    # r2.txt states "Initech" and "Initech Corp" in two units; while their
    # types agree they are one entity with one Initech-Depot edge citing
    # both. d4.txt makes "Initech" a location: the run splits them, and is
    # killed after the graph took the entities and before their edges. The
    # split entities share no text unit, so the resumed run did not take
    # "Initech Corp" for a split origin and the merged edge (and the claim)
    # stayed on "Initech". The suffix fixes the document ids, so that the
    # type tie of the second run goes the way that merges them.
    r2 = (
        "Acme and Buyer trade goods 2. Initech and Depot trade cash 1. Initech "
        "Corp and Depot trade goods 1."
    )
    runs = [
        {"d5.txt": "Acme and Buyer trade goods 2."},
        {"d5.txt": "Acme and Buyer trade goods 2. Initech and Depot trade cash 1."},
        {"r2.txt": r2},
        {
            "r2.txt": r2,
            "d4.txt": "Acme and Initech and Buyer trade cash 2. Acme Inc and Depot "
            "trade cash 1.",
        },
    ]

    full = _state(tmp_path / "full", [runs[-1]], suffix="c41dvt14")
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=3,
        die_at=("upsert_relationships", 1, False, SQLiteGraphIndexer),
        suffix="c41dvt14",
    )

    assert ("Initech Corp", "Depot") in {e[:2] for e in full["edges"]}
    assert incremental == full


def test_a_spelling_kept_apart_is_not_rewritten_by_a_delta_of_the_other(
    tmp_path: Path, mocker: Any
) -> None:
    # "Acme" (an organization) and "Acme Inc" (a location) share a name key
    # but a full build keeps them apart. A document about "Acme" must not
    # derive, rewrite and re-embed "Acme Inc"'s edges and claims: none of
    # them cites a unit "Acme" is stated in, so there is no half-written
    # split to finish. Each is written once, by the first run.
    first = {
        "a.txt": "Acme and Depot trade goods 1.",
        "b.txt": "Acme Inc and Buyer trade cash 1.",
    }
    second = {**first, "c.txt": "Acme and Carrier trade goods 2."}
    edges = mocker.spy(SQLiteGraphIndexer, "upsert_relationships")
    claims = mocker.spy(OpenSearchIndexer, "upsert_claims")

    incremental = _state(tmp_path / "inc", [first, second])

    written = [
        (r.source_name, r.target_name)
        for call in edges.call_args_list
        for r in call.args[1]
    ]
    stated = [c.subject_name for call in claims.call_args_list for c in call.args[1]]
    assert written.count(("Acme Inc", "Buyer")) == 1
    assert stated.count("Acme Inc") == 1
    assert incremental == _state(tmp_path / "full", [second])


def test_an_edge_the_derivation_drops_is_not_rewritten_as_a_stored_one(
    tmp_path: Path,
) -> None:
    # Gleaning links "Acme LLC" to "Acme Inc" in d1.txt. After the first run
    # that edge runs from "Acme LLC" to "Acme" (Acme Inc merged into it);
    # d2.txt makes "Acme LLC" and "Acme Inc" one entity, so the derivation
    # drops the edge as a self-loop. Re-pointed with its re-keyed endpoint,
    # it was written back among the entity's other stored edges and so not
    # deleted.
    d1 = (
        "Acme LLC and Acme Corp and Acme trade cash 2. Acme Inc and Acme LLC "
        "trade goods 3."
    )
    runs = [{"d1.txt": d1}, {"d1.txt": d1, "d2.txt": "Acme Inc trade goods 2."}]

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(tmp_path / "inc", runs, gleaning=True)

    assert not any("deals with" in (e[3] or "") for e in full["edges"])
    assert incremental == full


def test_a_stale_edge_on_a_third_spelling_is_moved(tmp_path: Path) -> None:
    # d6.txt splits the stored spellings of "Acme" apart. An edge stating
    # "Buyer" and "Acme Corp" was left on "ACME Inc": it cites a unit of
    # another spelling than the part split off "ACME Inc", so that split was
    # not taken for half-written and the edge stayed where a full build has
    # none.
    r2 = (
        "ACME Inc and Buyer trade goods 3. ACME Inc and Acme Corp trade parts 2. "
        "Acme and Acme Ltd and Depot trade cash 3."
    )
    first = {"d4.txt": "Acme LLC trade parts 3.", "r2.txt": r2}
    runs = [
        first,
        {**first, "d6.txt": "Buyer and Acme LLC and Acme Corp trade cash 3."},
    ]

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(tmp_path / "inc", runs, gleaning=True)

    assert incremental == full


@pytest.mark.parametrize(
    "die_at",
    [
        pytest.param(("upsert_entities", 1, True), id="after-the-vector-entities"),
        pytest.param(
            ("upsert_entities", 1, True, SQLiteGraphIndexer),
            id="after-the-graph-entities",
        ),
        pytest.param(
            ("delete_document_artifacts", 2, False), id="before-deleting-the-old-id"
        ),
    ],
)
@pytest.mark.parametrize(
    "fail_resumed_at",
    [
        pytest.param(None, id="resumed-once"),
        pytest.param(
            ("upsert_entities", SQLiteGraphIndexer), id="resumed-run-fails-again"
        ),
    ],
)
def test_a_resumed_removal_finishes_the_merge_it_started(
    tmp_path: Path,
    die_at: tuple[Any, ...],
    fail_resumed_at: tuple[Any, ...] | None,
) -> None:
    # Removing d1.txt leaves "Initech" stated only where "Initech Corp" is,
    # and the strip merges the two. Killed once "Initech" was written and
    # before the old "Initech Corp" was deleted, the resumed removal found
    # nothing citing d1.txt's unit (neither spelling does any more) and
    # "Initech Corp" stayed with its edge. A resumed run whose strip fails
    # again keeps the note for the next.
    runs = [
        {
            "d1.txt": "Initech trade goods 3.",
            "d5.txt": "Initech Corp and Initech trade cash 2.",
        },
        {"d5.txt": "Initech Corp and Initech trade cash 2."},
    ]

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=1,
        die_at=die_at,
        gleaning=True,
        fail_resumed_at=fail_resumed_at,
    )

    assert "Initech Corp" not in {e[1] for e in full["entities"]}
    assert incremental == full


@pytest.mark.parametrize(
    "die_at",
    [
        pytest.param(None, id="uninterrupted"),
        pytest.param(
            ("delete_extraction_records", 1, False), id="killed-before-deleting-it"
        ),
    ],
)
def test_a_finished_removal_deletes_its_strip_note(
    tmp_path: Path, mocker: Any, die_at: tuple[Any, ...] | None
) -> None:
    # The note a merging strip leaves for a retry goes once the old ids are
    # deleted: a finished removal leaves none in the extraction store, also
    # when the run that merged was killed before deleting it (the resumed
    # removal has nothing left to merge, and deletes it).
    runs = [
        {
            "d1.txt": "Initech trade goods 3.",
            "d5.txt": "Initech Corp and Initech trade cash 2.",
        },
        {"d5.txt": "Initech Corp and Initech trade cash 2."},
    ]
    written = mocker.spy(OpenSearchIndexer, "write_extraction_records")
    deleted = mocker.spy(OpenSearchIndexer, "delete_extraction_records")

    if die_at:
        _state(tmp_path, runs, die_in=1, die_at=die_at)
    else:
        _state(tmp_path, runs)

    notes = {
        key
        for call in written.call_args_list
        for key in call.args[1]
        if key.startswith("pending-strip")
    }
    assert notes
    assert notes <= {key for call in deleted.call_args_list for key in call.args[1]}


def test_a_document_restored_after_an_interrupted_removal_is_indexed_again(
    tmp_path: Path,
) -> None:
    # The second run deletes a.txt and is killed once the removal deleted its
    # vector copies. a.txt restored unchanged was read as unchanged (its
    # record still said PROCESSED): its text units, vector entities and
    # claims were lost, and the store probe failed every later run with
    # "the vector store does not hold" its text units.
    a = "Acme and Buyer trade goods 1."
    runs = [{"a.txt": a}, {"b.txt": "Depot trade cash 1."}]
    restored = {**runs[1], "a.txt": a}

    full = _state(tmp_path / "full", [restored])
    incremental = _state(
        tmp_path / "inc",
        [*runs, restored],
        die_in=1,
        die_at=("delete_document_artifacts", 1, True),
        resume=False,
    )

    assert incremental == full


@pytest.mark.parametrize(
    ("runs", "die_at", "gleaning"),
    [
        pytest.param(
            [
                {
                    "d6.txt": "Buyer and Acme Corp and Acme Ltd trade goods 1. Acme "
                    "Ltd trade cash 1.",
                    "d2.txt": "Buyer trade goods 2.",
                },
                {
                    "d2.txt": "Buyer trade goods 2. Acme and Depot trade parts 1.",
                    "r0.txt": "Buyer and Acme Corp and Acme Ltd trade goods 1. Acme "
                    "Ltd trade cash 1.",
                    "d1.txt": "Acme Inc and Acme and Acme Ltd trade cash 3. Depot "
                    "and Acme Corp and Acme LLC trade goods 3.",
                },
            ],
            ("upsert_entities", 1, False, SQLiteGraphIndexer),
            True,
            id="a-moved-document",
        ),
        pytest.param(
            [
                {"a.txt": "Buyer trade goods 2."},
                {
                    "a.txt": "Buyer trade goods 2. Acme and Depot trade goods 1.",
                    "b.txt": "Buyer and Acme Corp trade goods 1.",
                    "c.txt": "Acme and Acme Corp trade goods 3.",
                },
            ],
            ("upsert_entities", 1, False, SQLiteGraphIndexer),
            False,
            id="an-edited-document",
        ),
    ],
)
def test_a_resumed_run_derives_the_claims_its_removal_re_derived(
    tmp_path: Path,
    runs: list[dict[str, str]],
    die_at: tuple[Any, ...],
    gleaning: bool,
) -> None:
    # Killed after the delta's claims were written and before its entities,
    # the run is resumed. Its removal strips the killed run's claim of a
    # delta unit and derives it against a graph without the delta's
    # entities, under an id the delta does not write: the commit kept that
    # copy next to its own, a duplicate claim.
    full = _state(tmp_path / "full", [runs[-1]], gleaning=gleaning)
    incremental = _state(
        tmp_path / "inc", runs, die_in=1, die_at=die_at, gleaning=gleaning
    )

    assert incremental == full


def test_a_removal_resumed_with_a_note_derives_the_claims_it_re_keyed(
    tmp_path: Path,
) -> None:
    # The third run's removal merges spellings, notes them and re-keys a
    # claim onto a noted entity, and is killed once the claims were
    # stripped. The claim then cites no removed unit: the next run (an
    # edited corpus) removed the units again without finding it, and the
    # orphan stayed next to the claim a full build derives.
    d3 = "Acme LLC trade parts 3. Acme LLC and Acme Corp and Acme Ltd trade cash 2."
    d0 = "ACME and Buyer and Acme Ltd trade parts 3. ACME Inc and Buyer trade parts 1."
    runs = [
        {"d2.txt": "Acme Corp trade parts 3."},
        {"d2.txt": "ACME Inc trade parts 3.", "d3.txt": d3, "d0.txt": d0},
        {"d2.txt": "Acme Ltd trade parts 3.", "d3.txt": d3, "d0.txt": d0},
        {
            "d2.txt": "Acme Ltd trade parts 3.",
            "d3.txt": d3,
            "d0.txt": d0.replace("and Acme Ltd", "and Acme Corp"),
        },
    ]

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=2,
        die_at=("_remove_text_units_from_claims", 1, True, IndexingManager),
        gleaning=True,
        resume=False,
    )

    assert incremental == full


def test_a_strip_note_is_one_record_whatever_the_removed_units(
    tmp_path: Path, mocker: Any
) -> None:
    # Removing d1.txt and d2.txt (two text units) merges the two spellings.
    # The note is written once for the suffix, not once per removed unit
    # (20,000 units of a 3,000-edge note were 2.4 GB), and names the two
    # merged entities and the edge the merge dropped.
    runs = [
        {
            "d1.txt": "Initech trade goods 3.",
            "d2.txt": "Initech trade goods 4.",
            "d5.txt": "Initech Corp and Initech trade cash 2.",
        },
        {"d5.txt": "Initech Corp and Initech trade cash 2."},
    ]
    written = mocker.spy(OpenSearchIndexer, "write_extraction_records")

    full = _state(tmp_path / "full", [runs[-1]])
    written.reset_mock()
    incremental = _state(tmp_path / "inc", runs)

    notes = [
        call.args[1]
        for call in written.call_args_list
        if any(key.startswith("pending-strip") for key in call.args[1])
    ]
    assert [list(records) for records in notes] == [["pending-strip"]]
    (note,) = notes[0].values()
    assert set(note) == {"entity_ids", "relationship_ids"}
    assert len(note["entity_ids"]) == 2
    assert len(note["relationship_ids"]) == 1
    assert incremental == full


@pytest.mark.parametrize(
    "die_at",
    [
        pytest.param(
            ("upsert_entities", 1, True, SQLiteGraphIndexer),
            id="after-the-graph-entities",
        ),
        pytest.param(("upsert_entities", 1, False), id="before-the-vector-entities"),
        pytest.param(
            ("delete_document_artifacts", 2, False), id="before-deleting-the-old-ids"
        ),
    ],
)
def test_a_resumed_removal_that_merged_and_split_matches_a_full_build(
    tmp_path: Path, die_at: tuple[Any, ...]
) -> None:
    # Removing d2.txt merges "Acme Ltd" into "ACME Inc" and splits "Acme" off
    # it. The resumed removal strips the merged entities again; stripping the
    # split-off part and its origin again too (as #379 noted them) left
    # "Acme owes Acme Inc" with the wrong subject.
    runs = [
        {
            "d3.txt": "Buyer and Acme Ltd and ACME Inc trade cash 3. Buyer trade "
            "parts 1. Acme and Acme Inc trade cash 2.",
            "d2.txt": "Acme Ltd trade parts 2. Initech Corp trade parts 2.",
        },
        {
            "d3.txt": "Buyer and Acme Ltd and ACME Inc trade cash 3. Buyer trade "
            "parts 1. Acme and Acme Inc trade cash 2."
        },
    ]

    full = _state(tmp_path / "full", [runs[-1]])
    incremental = _state(tmp_path / "inc", runs, die_in=1, die_at=die_at)

    assert incremental == full


@pytest.mark.parametrize(
    "die_at",
    [
        pytest.param(("upsert_entities", 1, False), id="before-the-vector-entities"),
        pytest.param(
            ("upsert_entities", 2, True, SQLiteGraphIndexer),
            id="after-the-graph-entities",
        ),
        pytest.param(
            ("delete_document_artifacts", 2, False), id="before-deleting-the-old-ids"
        ),
    ],
)
def test_a_resumed_removal_finishes_the_edges_it_re_keyed(
    tmp_path: Path, die_at: tuple[Any, ...]
) -> None:
    # Removing r3.txt (d6.txt comes in) re-keys and drops edges of the
    # "Acme" spellings that need not cite r3.txt's units. Killed before
    # their old ids were deleted, the resumed removal finds those edges
    # through its note only.
    d1 = "Buyer and Depot and Acme Ltd trade parts 3. Acme trade goods 1."
    d7 = "Acme Inc and Acme LLC trade cash 3."
    runs = [
        {"d7.txt": d7},
        {
            "d7.txt": d7,
            "d1.txt": d1,
            "r3.txt": "Depot trade cash 1. Acme Inc and Buyer and Acme Ltd trade "
            "cash 2.",
        },
        {
            "d7.txt": d7,
            "d1.txt": d1,
            "d6.txt": "Depot and Acme Corp trade parts 2. Acme Inc and Acme Corp "
            "trade goods 1.",
        },
    ]

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(tmp_path / "inc", runs, die_in=2, die_at=die_at, gleaning=True)

    assert incremental == full


def test_a_strip_note_carries_over_the_items_of_an_earlier_note(
    tmp_path: Path, mocker: Any
) -> None:
    # A strip that merges while an earlier strip's note is still stored (the
    # earlier one was killed) notes the earlier items too: its note replaces
    # the stored one, and a retry must still finish both.
    runs = [
        {
            "d1.txt": "Initech trade goods 3.",
            "d2.txt": "Initech trade goods 4.",
            "d5.txt": "Initech Corp and Initech trade cash 2.",
        },
        {"d5.txt": "Initech Corp and Initech trade cash 2."},
    ]
    earlier = {"entity_ids": ["e-earlier"], "relationship_ids": ["r-earlier"]}
    read = OpenSearchIndexer.read_extraction_records

    def with_earlier_note(self, keys, suffix=None):  # type: ignore[no-untyped-def]
        found = read(self, keys, suffix=suffix)
        if "pending-strip" in keys and "pending-strip" not in found:
            found["pending-strip"] = dict(earlier)
        return found

    mocker.patch.object(OpenSearchIndexer, "read_extraction_records", with_earlier_note)
    written = mocker.spy(OpenSearchIndexer, "write_extraction_records")

    _state(tmp_path / "inc", runs)

    notes = [
        call.args[1]["pending-strip"]
        for call in written.call_args_list
        if "pending-strip" in call.args[1]
    ]
    assert notes, "the merging removal wrote no note"
    assert "e-earlier" in notes[0]["entity_ids"]
    assert "r-earlier" in notes[0]["relationship_ids"]


# Sequences the cycle-45 fuzz found (each diverged from a full build).
RUNS_GAINING = [
    {
        "d0.txt": "Buyer and Acme Ltd trade cash 2.",
        "d3.txt": "Acme LLC and Acme Inc trade parts 1. Acme and Acme Corp and ACME trade goods 3.",
        "d5.txt": "Acme Ltd and Acme Inc and Acme Corp trade goods 3. Acme LLC and ACME Inc trade parts 1.",
    },
    {
        "d0.txt": "Buyer and Acme Ltd trade cash 2.",
        "d5.txt": "Acme Ltd and Acme Inc and Acme Corp trade goods 3. Acme LLC and ACME Inc trade parts 1.",
        "d2.txt": "Acme and ACME trade cash 1. Acme Inc trade cash 2.",
        "d7.txt": "Depot and Acme LLC trade parts 3.",
    },
    {
        "d0.txt": "Buyer and Acme Ltd trade cash 2.",
        "d2.txt": "Acme and ACME trade cash 1. Acme Inc trade cash 2.",
        "r6.txt": "Acme Ltd and Acme Inc and Acme Corp trade goods 3. Acme LLC and ACME Inc trade parts 1.",
        "r0.txt": "Depot and Acme LLC trade parts 3.",
    },
]
RUNS_DUPKEY = [
    {
        "d0.txt": "Buyer and Acme Inc trade goods 1. Acme LLC and Acme Corp trade goods 2. Depot and Acme Inc trade cash 2.",
        "d6.txt": "Acme and Acme Corp and Acme Inc trade goods 1. Acme LLC and Depot and Acme trade cash 2. Acme Inc trade goods 2.",
        "d1.txt": "Acme LLC and Depot trade goods 1.",
    },
    {
        "d0.txt": "Buyer and Acme Inc trade goods 1. Acme LLC and Acme Corp trade goods 2. Depot and Acme Inc trade cash 2.",
        "d6.txt": "Acme and Acme Corp and Acme Inc trade goods 1. Acme LLC and Depot and Acme trade cash 2. Acme Inc trade goods 2.",
        "d1.txt": "Acme LLC and Depot trade goods 1.",
        "d4.txt": "Acme Corp trade cash 1. Depot and Acme and Acme Ltd trade cash 3.",
    },
]
RUNS_SPLITPART = [
    {"d0.txt": "Acme Ltd and Acme and Acme Corp trade parts 1."},
    {
        "d0.txt": "Acme Ltd and Acme and Acme Corp trade parts 1.",
        "d1.txt": "Acme Ltd and ACME Inc and Acme trade cash 3.",
    },
]
RUNS_RENAME = [
    {
        "r6.txt": "ACME Inc trade cash 2. Initech and Buyer and ACME Inc trade cash 2.",
        "d1.txt": "Buyer and Acme Inc trade cash 3.",
    },
    {"r6.txt": "ACME Inc trade cash 2. Initech and Buyer and ACME Inc trade cash 2."},
]
RUNS_VECTOR_ONLY = [
    {"d5.txt": "Acme Inc and Initech Corp and ACME Inc trade goods 2."},
    {
        "d5.txt": "Acme Inc and Initech Corp and ACME Inc trade goods 2.",
        "d1.txt": "Initech Corp trade parts 1. Acme Inc and Acme Corp and Initech Corp Corp trade parts 3.",
        "d7.txt": "Acme trade cash 3.",
    },
    {
        "d5.txt": "Acme Inc and Initech Corp and ACME Inc trade goods 2.",
        "d1.txt": "Initech Corp trade parts 1. Acme Inc and Acme Corp and Initech Corp Corp trade parts 3.",
        "d7.txt": "Acme trade cash 3.",
        "d0.txt": "Buyer and Initech Corp and Acme trade parts 2. Acme and Buyer and Acme trade parts 3.",
    },
    {
        "d5.txt": "Acme Inc and Initech Corp and ACME Inc trade goods 2.",
        "d1.txt": "Initech Corp trade parts 1. Acme Inc and Acme Corp and Initech Corp Corp trade parts 3.",
        "d7.txt": "Acme trade cash 3.",
        "d0.txt": "Buyer and Initech Corp and Acme trade parts 2. Acme and Buyer and Acme trade parts 3.",
        "d2.txt": "Buyer and Initech Corp trade parts 2. Acme Inc trade parts 1.",
    },
]


def test_a_strip_killed_between_phases_keeps_an_entity_that_gains_units(
    tmp_path: Path,
) -> None:
    # Removing d3.txt (d2.txt and d7.txt come in) strips a spelling that now
    # cites a unit of another one (its statement merged in). The strip wrote
    # that entity last, after the edges re-pointed at it; killed in between,
    # and run next on an edited corpus, the edge citing the kept unit hung
    # on a vertex citing only removed units and went with it.
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

    runs = RUNS_GAINING

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=1,
        die_at=("_run_indexing_phase", 2, True, IndexingManager),
        gleaning=True,
        resume=False,
    )

    assert incremental == full


def test_an_added_document_leaves_no_duplicate_of_a_re_pointed_edge(
    tmp_path: Path,
) -> None:
    # Adding d4.txt re-points an edge onto endpoints a stored edge of the
    # same type already joins under another id. A full build has one edge
    # there; the stored one, citing nothing the written one does not, stayed.
    runs = RUNS_DUPKEY

    full = _state(tmp_path / "full", [runs[-1]])
    incremental = _state(tmp_path / "inc", runs)

    assert incremental == full


def test_a_claim_about_a_split_off_part_is_derived_again(tmp_path: Path) -> None:
    # Adding d1.txt splits spellings apart. A stored claim already naming
    # the part's id (the spelling it states) was not derived again and kept
    # a stale self-claim.
    runs = RUNS_SPLITPART

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(tmp_path / "inc", runs, gleaning=True)

    assert incremental == full


def test_a_rename_killed_before_the_vector_edges_renames_them_on_retry(
    tmp_path: Path,
) -> None:
    # Removing d1.txt renames an entity under its id ("Acme Inc" to "ACME
    # Inc"). Killed after the graph edges took the new name and before
    # their vector copies did, the retry compared the derived names with the
    # graph copy only and left the vector copies under the old name.
    runs = RUNS_RENAME

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=1,
        die_at=("upsert_relationships", 1, False),
        gleaning=True,
    )

    assert incremental == full


def test_vector_copies_a_killed_run_left_without_graph_copies_go(
    tmp_path: Path,
) -> None:
    # Run 2 is killed at its first vector entity write; the next run, on
    # an edited corpus, kept the vector copy of an edge the graph store did
    # not hold (nothing strips a copy the graph cannot read back).
    runs = RUNS_VECTOR_ONLY

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=2,
        die_at=("upsert_entities", 1, False),
        gleaning=True,
        resume=False,
    )

    assert incremental == full


RUNS_COMMIT_SPLIT_CLAIM = [
    {"d7.txt": "Acme Inc and Acme LLC trade cash 3."},
    {
        "d7.txt": "Acme Inc and Acme LLC trade cash 3.",
        "d1.txt": "Buyer and Depot and Acme Ltd trade parts 3. Acme trade goods 1.",
        "r3.txt": "Depot trade cash 1. Acme Inc and Buyer and Acme Ltd trade cash 2.",
    },
    {
        "d7.txt": "Acme Inc and Acme LLC trade cash 3.",
        "d1.txt": "Buyer and Depot and Acme Ltd trade parts 3. Acme trade goods 1.",
        "d6.txt": "Depot and Acme Corp trade parts 2. Acme Inc and Acme Corp trade goods 1.",
    },
]
RUNS_COMMIT_SPLIT_EDGE = [
    {"d0.txt": "Acme Ltd trade parts 1. Buyer and Depot and Acme Inc trade goods 2."},
    {
        "d0.txt": "Acme Ltd trade parts 1. Buyer and Depot and Acme Inc trade goods 2. Acme Inc and Depot trade cash 2.",
        "d3.txt": "Acme Inc and Depot and Buyer trade parts 2.",
        "d1.txt": "Acme Inc and Acme Corp trade goods 2.",
    },
    {
        "d0.txt": "Acme Ltd trade parts 1. Buyer and Depot and Acme Inc trade goods 2. Acme Inc and Depot trade cash 2.",
        "d3.txt": "Acme Inc and Depot and Buyer trade parts 2.",
        "d5.txt": "Depot and Acme Corp trade cash 3.",
        "r1.txt": "Acme Inc and Acme Corp trade goods 2.",
    },
]


@pytest.mark.parametrize(
    ("runs", "die_at", "gleaning"),
    [
        pytest.param(
            RUNS_COMMIT_SPLIT_CLAIM,
            ("upsert_claims", 1, False),
            False,
            id="killed-before-the-claims",
        ),
        pytest.param(
            RUNS_COMMIT_SPLIT_EDGE,
            ("upsert_text_units", 1, False),
            True,
            id="killed-before-the-text-units",
        ),
    ],
)
def test_a_resumed_commit_finishes_the_split_it_started(
    tmp_path: Path, runs: list[dict[str, str]], die_at: tuple[Any, ...], gleaning: bool
) -> None:
    # The last run's commit splits spellings apart and re-points their edges
    # and claims. Killed partway, the resumed run derived from the half
    # written stored state, which no longer called for a split: the claim
    # stayed a self-claim, or an edge stayed on the spelling it was split
    # from. The commit notes what it splits, re-keys and rewires before its
    # first write.
    full = _state(tmp_path / "full", [runs[-1]], gleaning=gleaning)
    incremental = _state(
        tmp_path / "inc", runs, die_in=len(runs) - 1, die_at=die_at, gleaning=gleaning
    )

    assert incremental == full


def test_a_finished_commit_deletes_its_note(tmp_path: Path, mocker: Any) -> None:
    # The note a splitting commit writes before its first write goes once
    # the old ids are deleted.
    runs = RUNS_COMMIT_SPLIT_CLAIM
    written = mocker.spy(OpenSearchIndexer, "write_extraction_records")
    deleted = mocker.spy(OpenSearchIndexer, "delete_extraction_records")

    _state(tmp_path, runs)

    noted = [
        call for call in written.call_args_list if "pending-commit" in call.args[1]
    ]
    assert noted
    assert any("pending-commit" in call.args[1] for call in deleted.call_args_list)


def test_a_resumed_commit_that_fails_again_keeps_its_note(tmp_path: Path) -> None:
    # The resumed commit's claim write fails: the run fails, the note stays,
    # and the next run still finishes the split.
    runs = RUNS_COMMIT_SPLIT_CLAIM

    full = _state(tmp_path / "full", [runs[-1]])
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=len(runs) - 1,
        die_at=("upsert_claims", 1, False),
        fail_resumed_at=("upsert_claims", OpenSearchIndexer),
    )

    assert incremental == full
