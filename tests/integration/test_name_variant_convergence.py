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
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.adapters.storage.sqlite_graph_indexer import SQLiteGraphIndexer
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.models import Config, PipelineStageStatus

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
    ranks: bool = False,
) -> dict[str, list[tuple]]:
    """The stored entities, edges and claims after indexing ``corpora`` one
    after the other (run ``repeat`` twice under its pipeline id; run
    ``die_in`` killed at ``die_at``, then resumed under its id, or, without
    ``resume``, followed by the next corpus).

    ``die_at`` is a method, the call that dies (1 = the first), whether
    that call completes before the run dies and, optionally, the class
    (default the vector store's). ``fail_resumed_at`` (a method and its
    class) makes the resumed run's first call of it report every item
    failed; that run fails and is resumed once more. With ``ranks``, the
    graph's entity ranks and edge ids and ranks too."""
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
    return _read_state(config, providers, ranks)


def _read_state(
    config: Config, providers: Providers, ranks: bool = False
) -> dict[str, list[tuple]]:
    """The stored entities, edges and claims (see ``_state``)."""
    graph = providers.build_graph_indexer()
    try:
        assert isinstance(graph, SQLiteGraphIndexer)
        ids = [i for i, _ in graph.read_entity_names()]
        entities = graph.read_entities(ids)
        edges = graph.read_relationships(graph.find_incident_relationship_ids(ids))
    finally:
        graph.close()
    claims = _stored(config, providers, config.indexing.opensearch.claims_index_prefix)
    vector_entities = _stored(
        config, providers, config.indexing.opensearch.entities_index_prefix
    )
    try:
        vector_edges = _stored(
            config, providers, config.indexing.opensearch.relationships_index_prefix
        )
    except NotFoundError:  # no edge was ever written
        vector_edges = []
    state = {
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
        # The vector copies of the entities and edges: their own names and
        # lineage.
        "vector_entities": sorted(
            (
                hit["_id"],
                hit["_source"].get("name"),
                tuple(sorted(hit["_source"].get("text_unit_ids") or [])),
            )
            for hit in vector_entities
        ),
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
    if ranks:
        state["ranks"] = sorted((e.name, e.rank) for e in entities)
        state["edge_ranks"] = sorted(
            (r.id, r.source_name, r.target_name, r.rank) for r in edges
        )
    return state


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
    statement = "Initech Corp and Buyer and Acme trade parts 1."
    first = "Depot and Initech trade goods 1. Initech Corp trade parts 3."
    second = f"{first} Initech and Buyer trade parts 2."
    runs = [
        {"d5.txt": statement, "d6.txt": first},
        {"d6.txt": second, "r5.txt": statement},
        {
            "d6.txt": second,
            "r5.txt": statement,
            "d5.txt": "Buyer and Initech and " "Depot trade cash 2.",
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
    d1 = "Initech Corp and Initech trade parts 1. Depot and Buyer trade cash 1."
    d7 = "Depot and Initech Corp and Buyer trade goods 1."
    runs = [
        {"d1.txt": "Initech trade cash 2."},
        {"d1.txt": d1, "d5.txt": "Buyer trade parts 2."},
        {"d1.txt": d1, "d5.txt": "Buyer trade parts 2.", "d7.txt": d7},
        {"d1.txt": d1, "d5.txt": "Buyer trade parts 2.", "r0.txt": d7},
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


def test_a_gleaned_edge_follows_a_part_split_off_two_stored_spellings(
    tmp_path: Path,
) -> None:
    # No fault. Adding a7.txt splits "Acme LLC" off both stored "Acme" and
    # "Acme Inc"; only the first was rewired, so the edge gleaning adds for
    # "Acme Ltd" (r3.txt), stored on "Acme Inc", stayed there instead of
    # following the part, and the ranks counted it on the wrong end.
    base = {
        "c3.txt": "Acme LLC and Acme LLC trade cash 1.",
        "r3.txt": "Depot and Acme trade parts 1. Acme and Acme Inc and Acme Ltd "
        "trade parts 3.",
        "b0.txt": "Buyer and Acme and Acme Inc trade goods 3.",
        "b7.txt": "Acme and Buyer and Depot trade goods 1. Buyer and Acme and "
        "Acme Inc trade parts 1.",
    }
    runs = [
        base,
        {
            **base,
            "a7.txt": "Acme LLC and Acme trade cash 2. Acme Ltd and Acme LLC and "
            "Depot trade goods 2.",
        },
    ]

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True, ranks=True)
    incremental = _state(tmp_path / "inc", runs, gleaning=True, ranks=True)

    assert ("Depot", "Acme LLC") in {e[:2] for e in full["edges"]}
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
            "d5.txt": "Initech Corp and Initech trade parts 2.",
        },
        {"d5.txt": "Initech Corp and Initech trade parts 2."},
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
            "d5.txt": "Initech Corp and Initech trade parts 2.",
        },
        {"d5.txt": "Initech Corp and Initech trade parts 2."},
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
            "d5.txt": "Initech Corp and Initech trade parts 2.",
        },
        {"d5.txt": "Initech Corp and Initech trade parts 2."},
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
    assert set(note) == {"entity_ids", "relationship_ids", "claim_ids"}
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
    # it. Killed partway, the resumed removal strips the merged entities
    # again from its note, and derives the claims about the split's
    # entities again (also noted).
    runs = [
        {
            "d3.txt": "Buyer and Acme Ltd and ACME Inc trade parts 3. Buyer trade "
            "goods 1. Acme and Acme Inc trade parts 2.",
            "d2.txt": "Acme Ltd trade goods 2. Initech Corp trade goods 2.",
        },
        {
            "d3.txt": "Buyer and Acme Ltd and ACME Inc trade parts 3. Buyer trade "
            "goods 1. Acme and Acme Inc trade parts 2."
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
        "d2.txt": "Acme Inc trade goods 1.",
    },
    {"r6.txt": "ACME Inc trade cash 2. Initech and Buyer and ACME Inc trade cash 2."},
]
# A commit renames "Acme" to "ACME" under its id (two more statements of
# that spelling); the next corpus drops the documents that renamed it.
RUNS_RENAME_BACK = [
    {"d1.txt": "Depot and Acme trade goods 1."},
    {
        "d1.txt": "Depot and Acme trade goods 1.",
        "d2.txt": "ACME trade goods 2.",
        "d3.txt": "ACME trade goods 3.",
    },
    {"d1.txt": "Depot and Acme trade goods 1."},
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
    # Removing d1.txt and d2.txt renames an entity under its id ("Acme
    # Inc" to "ACME Inc"). Killed after the graph edges took the new name
    # and before their vector copies did, the retry compared the derived
    # names with the graph copy only and left the vector copies under the
    # old name.
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
    "die_at",
    [
        # After the vector edges, before the graph edges.
        ("upsert_relationships", 1, False, SQLiteGraphIndexer),
        # After the vector copies, before the graph entities.
        ("upsert_entities", 1, False, SQLiteGraphIndexer),
    ],
    ids=["vector-edges-ahead", "vector-claims-ahead"],
)
def test_a_rename_killed_between_the_stores_then_undone_renames_the_vector_copies(
    tmp_path: Path, die_at: tuple[Any, ...]
) -> None:
    # Run 1's commit renames "Acme" to "ACME" and is killed between the
    # stores; run 2 drops the documents that renamed it. The rename was not
    # in the commit note and the stored edges were compared with their
    # graph copies only: the vector edge (or the claim) kept "ACME".
    runs = RUNS_RENAME_BACK

    full = _state(tmp_path / "full", [runs[-1]])
    incremental = _state(tmp_path / "inc", runs, die_in=1, die_at=die_at, resume=False)

    assert incremental == full


def test_a_strip_killed_at_its_graph_edges_leaves_no_vector_edge_ahead(
    tmp_path: Path,
) -> None:
    # Run 1 edits d5.txt; its strip renames the entity the edge of d1.txt
    # ends at and is killed at its graph edge write; run 2 edits the corpus
    # again. The strip wrote the vector edges first, and run 2, deriving the
    # graph copy's name back, found the graph copy unchanged: the vector
    # edge kept the strip's name.
    d0, d1 = (
        "Acme LLC and ACME Inc trade goods 3.",
        "Depot trade goods 3. Acme trade cash 3.",
    )
    runs = [
        {
            "d0.txt": d0,
            "d1.txt": d1,
            "d5.txt": "Depot and ACME Ltd and ACME trade goods 3.",
        },
        {
            "d0.txt": d0,
            "d1.txt": d1,
            "d5.txt": "Depot and Acme Ltd and ACME trade goods 3.",
        },
        {"d0.txt": "Acme Ltd trade parts 2.", "d1.txt": d1},
    ]

    full = _state(tmp_path / "full", [runs[-1]])
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=1,
        die_at=("upsert_relationships", 1, False, SQLiteGraphIndexer),
        resume=False,
    )

    assert incremental == full


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


NOTE_COMMIT = "pending-commit"


def test_a_killed_commit_is_finished_by_a_run_that_only_deletes(
    tmp_path: Path, mocker: Any
) -> None:
    # The run after the killed commit only removes a document: there is no
    # delta for the suffix, and the note is read, finished and deleted all
    # the same.
    runs = RUNS_COMMIT_SPLIT_CLAIM
    last = {k: v for k, v in runs[-1].items() if k != "d6.txt"}
    deleted = mocker.spy(OpenSearchIndexer, "delete_extraction_records")

    full = _state(tmp_path / "full", [last])
    deleted.reset_mock()
    incremental = _state(
        tmp_path / "inc",
        [*runs, last],
        die_in=len(runs) - 1,
        die_at=("upsert_claims", 1, False),
        resume=False,
    )

    assert incremental == full
    assert any(NOTE_COMMIT in call.args[1] for call in deleted.call_args_list)


RUNS_VECTOR_ORPHAN_EDGE = [
    {"a.txt": "Acme Corp and Acme trade goods 2."},
    {"a.txt": "Acme Corp and Acme trade goods 2.", "b.txt": "Acme trade cash 2."},
    {"a.txt": "Acme Corp and Acme trade goods 2."},
]
RUNS_VECTOR_ORPHAN_ENTITY = [
    {"d0.txt": "Depot and Acme Ltd and Acme trade goods 1."},
    {
        "d0.txt": "Depot and Acme Ltd and Acme trade goods 1.",
        "d5.txt": "Acme Inc Inc and Acme Inc trade cash 3.",
    },
    {
        "d0.txt": "Depot and Acme Ltd and Acme trade goods 1.",
        "d5.txt": "Acme Corp Inc and Acme Corp trade cash 3.",
        "d6.txt": "Acme Inc and Acme LLC trade cash 2. Acme Corp and ACME trade cash 1.",
    },
    {
        "d0.txt": "Depot and Acme Ltd and Acme trade goods 1.",
        "r7.txt": "Acme Inc and Acme LLC trade cash 2. Acme Corp and ACME trade cash 1.",
    },
]


@pytest.mark.parametrize(
    ("runs", "die_in", "die_at"),
    [
        pytest.param(
            RUNS_VECTOR_ORPHAN_EDGE,
            1,
            ("upsert_relationships", 1, False, SQLiteGraphIndexer),
            id="an-edge",
        ),
        pytest.param(
            RUNS_VECTOR_ORPHAN_ENTITY,
            2,
            ("upsert_entities", 1, False, SQLiteGraphIndexer),
            id="an-entity",
        ),
    ],
)
def test_a_killed_commits_vector_only_copies_go_with_its_note(
    tmp_path: Path, runs: list[dict[str, str]], die_in: int, die_at: tuple[Any, ...]
) -> None:
    # Killed between its vector and its graph writes, the commit left a
    # vector copy the graph never held; the next run, on another corpus, no
    # longer derives the item. The commit note names its entities: the next
    # commit retires the noted copies the graph does not hold (with the
    # vector edges at them), as nothing else finds them.
    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=die_in,
        die_at=die_at,
        gleaning=True,
        resume=False,
    )

    assert incremental == full


RUNS_SIBLING_CLAIM = [
    {
        "d2.txt": "Acme Inc and Acme Ltd and Acme trade cash 3. Acme Ltd and Acme Corp and Acme trade goods 3.",
        "d5.txt": "Acme Ltd and Acme LLC and Buyer trade cash 1. Acme Inc and Acme Ltd and Depot trade goods 3.",
    },
    {
        "d2.txt": "Acme Inc and Acme Ltd and Acme trade cash 3. Acme Ltd and Acme Corp and Acme trade goods 3."
    },
]


def test_a_removal_derives_the_claims_on_a_changed_entitys_spellings_again(
    tmp_path: Path,
) -> None:
    # The removal re-keys or renames an entity: a stored claim on another
    # stored spelling of its name may resolve to it now, so the claims about
    # every spelling of its name key are derived again (no crash needed).
    runs = RUNS_SIBLING_CLAIM

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(tmp_path / "inc", runs, gleaning=True)

    assert incremental == full


def test_a_commit_killed_twice_carries_the_first_notes_ids(
    tmp_path: Path, mocker: Any
) -> None:
    # Killed after its note, then failing again on the retry, the commit's
    # successors still name what the first one decided: the retry derives
    # from a stored state that no longer calls for the split, so only the
    # carried note keeps those entities.
    runs = RUNS_COMMIT_SPLIT_CLAIM
    written = mocker.spy(OpenSearchIndexer, "write_extraction_records")

    full = _state(tmp_path / "full", [runs[-1]])
    written.reset_mock()
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=len(runs) - 1,
        die_at=("upsert_claims", 1, False),
        fail_resumed_at=("upsert_claims", OpenSearchIndexer),
    )

    notes = [
        set(call.args[1][NOTE_COMMIT]["entity_ids"])
        for call in written.call_args_list
        if NOTE_COMMIT in call.args[1]
    ]
    assert len(notes) >= 2
    assert all(notes[0] <= later for later in notes[1:])
    assert incremental == full


RUNS_SIBLING_CLAIM_COMMIT = [
    {
        "d1.txt": "Depot and Acme Corp trade goods 3. Acme Corp trade cash 3.",
        "d4.txt": "Acme Corp and Acme Inc and Depot trade cash 3.",
    },
    {
        "d1.txt": "Depot and Acme Corp trade goods 3. Acme Corp trade cash 3.",
        "d4.txt": "Acme Corp and Acme Inc and Depot trade cash 3.",
        "d0.txt": "Acme Ltd and Acme Inc and Acme LLC trade goods 1.",
    },
    {
        "d1.txt": "Depot and Acme Ltd trade goods 3. Acme Ltd trade cash 3.",
        "d4.txt": "Acme Corp and Acme Inc and Depot trade cash 3.",
        "d0.txt": "Acme Ltd and Acme Inc and Acme LLC trade goods 1.",
    },
]


def test_a_commit_derives_the_claims_on_a_re_keyed_entitys_spellings_again(
    tmp_path: Path,
) -> None:
    # The last commit re-keys "Acme LLC" to "Acme Ltd" (d1.txt now names it).
    # d0.txt's claim "Acme Ltd owes Acme Inc" had resolved to "Acme Inc", the
    # smallest of the tied spellings, and merged with its self-claim; now
    # "Acme Ltd" is an entity of its own, so the claims on the other stored
    # spellings of the re-keyed entity's name are derived again.
    runs = RUNS_SIBLING_CLAIM_COMMIT

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(tmp_path / "inc", runs, gleaning=True)

    assert incremental == full


RUNS_SIBLING_CLAIM_SPLIT = [
    {
        "d5.txt": "Depot and Acme Corp and Acme Ltd trade goods 2.",
        "d6.txt": "Acme Corp trade parts 3.",
    },
    {
        "d5.txt": "Depot and Acme Corp and Acme Ltd trade cash 2.",
        "d6.txt": "Acme Corp trade parts 3.",
        "d7.txt": "Acme Ltd and Acme trade parts 3.",
    },
    {
        "d5.txt": "Depot and Acme Corp and Acme Ltd trade cash 2. Acme Inc and Acme LLC trade cash 2.",
        "d6.txt": "Acme Corp trade parts 3.",
        "d7.txt": "Acme Ltd and Acme trade parts 3.",
    },
    {
        "d5.txt": "Depot and Acme Corp and Acme Ltd trade cash 2. Acme Inc and Acme LLC trade cash 2.",
        "d6.txt": "Acme LLC and Acme Ltd trade parts 2.",
        "d7.txt": "Acme Ltd and Acme trade parts 3.",
    },
]


def test_a_commit_derives_the_claims_on_a_split_parts_spellings_again(
    tmp_path: Path,
) -> None:
    # The last commit splits "Acme LLC" off and re-keys the entity it split
    # from into "Acme Ltd": the stored claim "Depot owes Acme Corp" on "Acme
    # Ltd" resolves to "Acme LLC" in a full build, so the claims on every
    # spelling of the part's name, the re-keyed one included, are derived
    # again.
    runs = RUNS_SIBLING_CLAIM_SPLIT

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(tmp_path / "inc", runs, gleaning=True)

    assert incremental == full


def test_an_endpoint_joining_later_is_derived_with_its_other_spellings(
    tmp_path: Path,
) -> None:
    # The new document flips "Initech"'s type tie, so its edge to "Acme" is
    # rewired and "Acme" joins the derivation as an endpoint on a later
    # attempt. Its other spelling "Acme Inc" (another type in the corpus,
    # the same in the first document's first unit only) is derived with it,
    # as on the first attempt: held apart, the subset merged that unit's
    # "Acme Inc" statement into "Acme".
    first = {
        "d1.txt": "Initech Corp and Acme trade goods 3. "
        "Initech and Acme Inc and Acme Inc trade parts 3.",
        "r7.txt": "Acme Inc trade parts 1.",
    }
    runs = [first, {**first, "d4.txt": "Initech trade cash 3."}]

    full = _state(tmp_path / "full", [runs[-1]])
    incremental = _state(tmp_path / "inc", runs)

    assert incremental == full


def test_a_strip_killed_after_its_claim_writes_leaves_no_claim_behind(
    tmp_path: Path,
) -> None:
    # The removal splits "Acme Corp" off "ACME": the strip derives the
    # claims about them under new ids and writes them before the entities.
    # Killed right after, then the corpus edited so the next strip splits
    # nothing: the claims derived under the split must be in the strip's
    # note, or nothing finds them again.
    runs = [
        {
            "d4.txt": "Buyer trade goods 3. Buyer and Acme LLC trade parts 1.",
            "d2.txt": "Depot trade parts 3.",
        },
        {
            "d4.txt": "Buyer trade goods 3. Buyer and Acme LLC trade parts 1.",
            "d2.txt": "Buyer and Acme Corp trade goods 2.",
            "d5.txt": "Depot and Acme trade goods 2. Depot and ACME Inc trade cash 3.",
            "d7.txt": "Acme Corp and ACME trade parts 3.",
        },
        {
            "d2.txt": "Buyer and Acme Corp trade goods 2.",
            "d7.txt": "Acme Corp and ACME trade parts 3.",
            "d4.txt": "Depot and Buyer and ACME Inc trade cash 2.",
        },
        {
            "d7.txt": "Acme Corp and ACME trade parts 3.",
            "d4.txt": "Acme and Acme LLC trade parts 1. Buyer and ACME trade cash 1.",
        },
    ]

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=2,
        die_at=("upsert_claims", 1, True),
        resume=False,
        gleaning=True,
    )

    assert incremental == full


def test_a_leftover_strip_note_keeps_an_entity_the_strip_writes(
    tmp_path: Path,
) -> None:
    # Run 2's strip re-keys "ACME" onto "Acme LLC" (both noted) and dies
    # before it deletes its note (a failed best-effort delete leaves it
    # too). Run 3's strip reads the note: "ACME" is no longer in the graph,
    # so it was retired as a vector-only copy, yet the strip splits "ACME"
    # off again under that id; deleting it dropped d7.txt's unchanged edges.
    d7 = (
        "Depot and Acme Ltd and Acme Inc trade cash 2. "
        "Buyer and ACME Inc and Acme LLC trade goods 1."
    )
    d4 = "Acme Corp and Acme Corp trade cash 2. Acme LLC and Acme Corp trade cash 3."
    runs = [
        {
            "d7.txt": d7,
            "d4.txt": "Acme Corp and ACME Inc trade cash 2. "
            "Acme LLC and Acme Corp trade cash 3.",
        },
        {
            "d7.txt": d7,
            "d4.txt": d4,
            "r2.txt": "Acme and ACME and Acme Corp trade cash 1.",
        },
        {
            "d7.txt": d7,
            "d4.txt": d4,
            "r2.txt": "Acme and Acme and Acme Corp trade cash 1.",
        },
        {
            "d7.txt": d7,
            "d4.txt": "Acme and Acme Corp and Acme LLC trade parts 1. ACME trade cash 2.",
            "r7.txt": "Acme and Acme and Acme Corp trade cash 1.",
        },
    ]

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=2,
        die_at=("_clear_pending_strip", 1, False, IndexingManager),
        resume=False,
        gleaning=True,
    )

    assert incremental == full


def test_a_resumed_strip_rewrites_the_vector_copies_of_its_re_pointed_edges(
    tmp_path: Path,
) -> None:
    # Removing d7.txt re-points edges onto other spellings. Killed between
    # the strip's graph edge phase and its vector edge phase, the retry
    # found the graph copies current and left the vector copies naming (and
    # pointing at) the old endpoint, or missing.
    d0 = (
        "Acme and Depot trade parts 2. Acme Corp and Acme Inc and Acme Ltd "
        "trade cash 3. Acme Inc and Acme trade cash 1."
    )
    d4 = "Depot and Buyer trade parts 3. Acme Corp and Acme Inc trade parts 3."
    runs = [
        {
            "d7.txt": "Acme Ltd and Depot and Acme LLC trade cash 1. "
            "Acme trade cash 2.",
            "d0.txt": d0,
        },
        {
            "d7.txt": "Acme Ltd and Depot and Acme LLC trade parts 1. "
            "Acme trade parts 2.",
            "d0.txt": d0,
            "d4.txt": d4,
        },
        {"d0.txt": d0, "d4.txt": d4},
    ]

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=2,
        die_at=("_run_indexing_phase", 3, False, IndexingManager),
        gleaning=True,
    )

    assert incremental == full


def test_a_resumed_split_strip_ranks_and_keys_edges_as_a_full_build(
    tmp_path: Path,
) -> None:
    # Removing d4.txt splits "Acme Corp" off "Acme". Killed after the split
    # was written and before the re-keyed edge's old copy was deleted, the
    # retry re-derived that edge under its old id and ranked its endpoints
    # with the old copy's link still counted.
    kept = {
        "d7.txt": "Acme trade goods 2. Acme and Acme Inc and Buyer trade goods 2.",
        "d1.txt": "Acme Corp and Depot trade parts 1.",
    }
    runs = [
        {"d4.txt": "Acme Corp trade goods 3. Buyer trade cash 1.", **kept},
        kept,
    ]

    full = _state(tmp_path / "full", [runs[-1]], ranks=True)
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=1,
        die_at=("delete_by_id", 2, False, SQLiteGraphIndexer),
        ranks=True,
    )

    assert incremental == full


@pytest.mark.parametrize("rounds", [None, 4], ids=["converged", "out-of-rounds"])
def test_a_removal_whose_re_key_cascades_ranks_as_a_full_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rounds: int | None
) -> None:
    # No fault. Removing d7.txt re-derives entities whose edges reach other
    # spellings, whose edges reach others in turn: five rounds of reads. The
    # derivation stopped after four and ranked "Acme" and "Buyer" 0, the
    # edges it had not read uncounted. Out of rounds, an entity whose stored
    # edges were not all read keeps its stored links in its rank.
    import unified_kg_rag.application.storage.indexing_manager as manager

    if rounds is not None:
        monkeypatch.setattr(manager, "_DERIVE_ROUNDS", rounds)
    kept = {
        "r4.txt": "Buyer and Acme trade cash 2. Acme and Acme and Acme trade goods 1.",
        "d1.txt": "Acme and Acme Inc trade parts 3.",
        "r7.txt": "Initech Corp and Acme Ltd and Initech trade parts 3.",
    }
    runs = [{**kept, "d7.txt": "Initech and Buyer trade cash 2."}, kept]

    full = _state(tmp_path / "full", [runs[-1]], ranks=True)
    incremental = _state(tmp_path / "inc", runs, ranks=True)

    assert ("Acme", 2) in full["ranks"]
    assert incremental == full


def test_a_cascading_derivation_reads_each_record_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cascade above takes several rounds. Each re-read the extraction
    # records of every unit read in the rounds before: quadratic beside a
    # hub of many units.
    kept = {
        "r4.txt": "Buyer and Acme trade cash 2. Acme and Acme and Acme trade goods 1.",
        "d1.txt": "Acme and Acme Inc trade parts 3.",
        "r7.txt": "Initech Corp and Acme Ltd and Initech trade parts 3.",
    }
    reads: list[list[str]] = []
    derive = IndexingManager.derive
    read_records = OpenSearchIndexer.read_extraction_records

    def counted_derive(self: Any, *args: Any, **kwargs: Any) -> Any:
        reads.append([])
        return derive(self, *args, **kwargs)

    def counted_records(self: Any, keys: list[str], *args: Any, **kwargs: Any) -> Any:
        if reads:
            reads[-1].extend(keys)
        return read_records(self, keys, *args, **kwargs)

    monkeypatch.setattr(IndexingManager, "derive", counted_derive)
    monkeypatch.setattr(OpenSearchIndexer, "read_extraction_records", counted_records)

    _state(tmp_path, [{**kept, "d7.txt": "Initech and Buyer trade cash 2."}, kept])

    assert reads
    for read in reads:
        assert len(read) == len(set(read))


def test_an_edit_leaves_no_interim_claim(tmp_path: Path) -> None:
    # No fault. The edit makes the removal derive a claim of a kept unit
    # under an interim id (its subject resolved without the edited
    # document's spellings); the commit then derives the same statement
    # into the full build's claim, and the interim copy stayed: a duplicate,
    # here a self-claim.
    first = "Acme Inc and Acme and ACME trade cash 2. ACME Inc and ACME and Acme Ltd trade goods 3."
    d4 = "Acme Ltd and Acme Corp trade parts 2."
    runs = [
        {"d5.txt": first, "d4.txt": d4},
        {
            "d5.txt": f"{first} Acme Corp and Acme Ltd and ACME trade parts 1.",
            "d4.txt": d4,
        },
    ]

    full = _state(tmp_path / "full", [runs[-1]])
    incremental = _state(tmp_path / "inc", runs)

    assert incremental == full


@pytest.mark.parametrize(
    "corpora",
    [
        pytest.param(
            [
                {"d0.txt": "Acme and Buyer trade goods 3."},
                {"d4.txt": "Depot and Buyer trade cash 1."},
                {"d0.txt": "Buyer and Acme trade cash 2."},
            ],
            id="the-next-run-passes-its-store-probe",
        ),
        pytest.param(
            [
                {
                    "d0.txt": "Acme and Buyer trade goods 3.",
                    "d1.txt": "Depot and Carrier trade parts 1.",
                },
                {"d1.txt": "Depot and Carrier trade parts 1."},
                {
                    "d0.txt": "Acme and Buyer trade goods 3.",
                    "d1.txt": "Depot and Carrier trade parts 1.",
                },
            ],
            id="a-dropped-document-back-unchanged-is-indexed",
        ),
    ],
)
def test_a_reset_killed_after_its_index_swap_leaves_no_stale_record(
    tmp_path: Path, corpora: list[dict[str, str]]
) -> None:
    # The reset of the second corpus drops a document and is killed after
    # it swapped an index in. The dropped document's record stayed
    # PROCESSED over stores that no longer hold it: every later incremental
    # run failed its store probe, or diffed the document unchanged when it
    # came back and never indexed it again.
    config = _config(tmp_path / "stores", tmp_path / "viz")
    providers = _providers(
        config, FakeOllama(_respond, "x", DIM), FakeCrossEncoder("r")
    )
    source = tmp_path / "corpus"

    def write(corpus: dict[str, str]) -> None:
        shutil.rmtree(source, ignore_errors=True)
        source.mkdir(parents=True)
        for name, text in corpus.items():
            (source / name).write_text(text)

    write(corpora[0])
    assert _ingest(config, providers, source, tmp_path / "run-0").status is (
        PipelineStageStatus.COMPLETED
    )
    write(corpora[1])
    config.indexing.reset = True
    with (
        patch.object(
            IndexingManager,
            "_run_indexing_phase",
            _dying("_run_indexing_phase", 2, True, IndexingManager),
        ),
        pytest.raises(_Died),
    ):
        _ingest(config, providers, source, tmp_path / "run-1")
    config.indexing.reset = False
    write(corpora[2])
    context = _ingest(config, providers, source, tmp_path / "run-2")

    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    texts = _stored(
        config, providers, config.indexing.opensearch.text_units_index_prefix
    )
    assert {hit["_source"]["text"].strip() for hit in texts} == set(corpora[2].values())


KEPT_REMOVAL = {
    "d7.txt": "Acme Corp and Depot and Acme Ltd trade goods 3. Acme Ltd trade cash 2.",
    "r2.txt": "Acme Ltd and Acme LLC trade goods 2.",
}
RUNS_COMMIT_REMOVAL = [
    {
        "d7.txt": "Acme LLC and Depot and Acme Ltd trade goods 3. Acme Inc trade cash 2.",
        "r2.txt": "Acme Ltd and Acme LLC trade goods 2.",
    },
    KEPT_REMOVAL,
    {
        **KEPT_REMOVAL,
        "d4.txt": "Acme LLC trade cash 1. Depot and Acme LLC trade goods 2.",
    },
    KEPT_REMOVAL,
]


@pytest.mark.parametrize("first", [0, 1], ids=["with-r0", "without-r0"])
def test_a_removal_after_a_killed_commit_leaves_no_interim_claim(
    tmp_path: Path, first: int
) -> None:
    # The commit adding d4 re-keys the claims of the kept units onto the
    # spelling d4 adds (Acme LLC) and dies at its first graph entity write:
    # the vector store holds the re-keyed claims, the graph not their
    # subject. The next run removes d4. Its strip derived the d7 claim
    # against that graph: the subject, absent, resolved to the stated
    # spelling (Acme Corp) under a new id no later derivation read, beside
    # the full build's claim the finished commit note derived.
    runs = RUNS_COMMIT_REMOVAL[first:]
    full = _state(tmp_path / "full", [runs[-1]])
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=len(runs) - 2,
        die_at=("upsert_entities", 1, False, SQLiteGraphIndexer),
        resume=False,
    )
    assert incremental == full


def test_a_removal_after_a_commit_killed_past_its_retire_keeps_the_strips_claim(
    tmp_path: Path,
) -> None:
    # The commit's writes and retire all landed; it died before deleting its
    # note. The removal's strip re-keys the noted ACME onto Acme Ltd and
    # derives the claim about it with that re-key: the commit finishing the
    # note must not derive it again from a graph without the old id.

    d0 = {
        "d0.txt": "Depot and Acme LLC and Acme Ltd trade cash 3. "
        "ACME and ACME Inc and Acme Ltd trade goods 1."
    }
    runs = [d0, {**d0, "d7.txt": "Acme and ACME and Acme Inc trade goods 2."}, d0]
    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=1,
        die_at=("delete_retired_ids", 1, True, IndexingManager),
        gleaning=True,
        resume=False,
    )
    assert incremental == full


RESUMED_V1 = {"d0.txt": "Depot and Buyer trade parts 1. Acme and Depot trade cash 2."}
RESUMED_V2 = {
    "d0.txt": "Depot and Buyer trade parts 1. Acme LLC and Depot trade cash 2.",
    "r4.txt": "Acme Inc and Depot trade parts 1.",
}


@pytest.mark.parametrize("between", ["reset", "corpus"])
def test_a_resume_restores_documents_with_the_delta_they_were_filtered_by(
    tmp_path: Path, between: str
) -> None:
    # The incremental attempt caches its loading output (d0 changed, r4 new)
    # and dies in extraction. The same pipeline id is run again, as a reset
    # (every document new) or over a corpus with d0 back at its stored
    # version (only r4 new); that attempt caches its loading output under
    # another key, saves its delta with the metadata and dies before
    # chunking. Resumed once more over the first attempt's inputs, the run
    # restored the first attempt's documents with the second attempt's
    # delta and did not end as a full build (under the reset's, d0 was
    # committed as new and its old version's text unit and statements
    # stayed).
    full = _state(tmp_path / "full", [RESUMED_V2])
    inc = tmp_path / "inc"
    config = _config(inc / "stores", inc / "viz")
    config.processing.claim_extraction.enabled = True
    config.processing.gleaning.max_rounds = 1
    config.processing.document_parsing.source_scope = "fuzz-corpus"
    providers = _providers(
        config, FakeOllama(_respond, "x", DIM), FakeCrossEncoder("r")
    )
    source = inc / "corpus"

    def write(corpus: dict[str, str]) -> None:
        if source.exists():
            shutil.rmtree(source)
        source.mkdir(parents=True)
        for name, text in corpus.items():
            (source / name).write_text(text)

    def dying(stage: type) -> Any:
        def die(self: Any, context: Any) -> Any:
            raise _Died

        return patch.object(stage, "_execute_core", die)

    write(RESUMED_V1)
    _ingest(config, providers, source, inc / "run-0")
    write(RESUMED_V2)
    run = inc / "run-1"
    with dying(ps.GraphExtractionStage), pytest.raises(_Died):
        _ingest(config, providers, source, run)
    if between == "reset":
        config.indexing.reset = True
    else:
        write({**RESUMED_V2, **RESUMED_V1})
    with dying(ps.TextChunkingStage), pytest.raises(_Died):
        _ingest(config, providers, source, run)
    config.indexing.reset = False
    write(RESUMED_V2)
    _ingest(config, providers, source, run)

    # _state over no corpora reads the stores under ``inc``.
    assert _state(inc, []) == full


STRIP_SPLIT_RUNS = [
    {
        "d3.txt": "Buyer and Acme trade parts 1. Acme Corp and Acme Inc trade "
        "parts 1. Acme trade parts 3."
    },
    {
        "d3.txt": "Buyer and Acme trade parts 1. Acme Corp and Acme Inc trade "
        "parts 1. Acme trade parts 3.",
        "r0.txt": "Acme Corp and Acme and Buyer trade goods 2. "
        "Acme Ltd and Acme LLC and Buyer trade parts 2.",
    },
    {
        "r0.txt": "Acme Corp and Acme and Buyer trade goods 2. "
        "Acme Ltd and Acme LLC and Buyer trade parts 2.",
        "d2.txt": "Depot and Acme trade parts 1. Acme Ltd and Depot trade parts 3.",
    },
]


@pytest.mark.parametrize("call", [1, 2, 3, 4])
@pytest.mark.parametrize("resume", [True, False], ids=["resume", "fresh"])
def test_a_strip_killed_before_its_vector_edges_writes_the_missing_ones(
    tmp_path: Path, call: int, resume: bool
) -> None:
    # Removing d3.txt splits "Acme LLC" off "Acme": the strip writes the
    # edge "Acme LLC"-"Buyer" under a new id to the graph and is killed
    # (call 3) before its vector edges. The retry found the graph copy
    # current and wrote a vector copy only where one was stored and
    # stale: the edge had none for good.
    runs = STRIP_SPLIT_RUNS if resume else [*STRIP_SPLIT_RUNS, STRIP_SPLIT_RUNS[-1]]

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=2,
        die_at=("_run_indexing_phase", call, False, IndexingManager),
        gleaning=True,
        resume=resume,
    )

    assert incremental["vector_edges"] == full["vector_edges"]
    assert incremental == full


SPLIT_RETRY_KEPT = {
    "d4.txt": "Acme Corp and Acme trade cash 2.",
    "d3.txt": "Depot and Acme Ltd and Buyer trade goods 1. "
    "Acme and Acme Inc and Acme LLC trade goods 2.",
}
SPLIT_RETRY_RUNS = [
    {**SPLIT_RETRY_KEPT, "d0.txt": "Acme Inc and Acme and Acme Ltd trade cash 2."},
    SPLIT_RETRY_KEPT,
]


@pytest.mark.parametrize("resume", [True, False], ids=["resume", "fresh"])
@pytest.mark.parametrize("method", ["delete_retired_ids", "_clear_pending_strip"])
def test_a_retried_split_strip_resolves_the_claim_object_to_the_part(
    tmp_path: Path, method: str, resume: bool
) -> None:
    # d0.txt bridged "Acme Ltd" into "Acme"; removing it splits "Acme Ltd"
    # off. Killed once every strip write landed (at its retire, or at the
    # note delete), the retry derived the noted claims about the split's
    # entities without the part: the claim object resolved to the stated
    # spelling under a new id no later derivation read.
    runs = SPLIT_RETRY_RUNS if resume else [*SPLIT_RETRY_RUNS, SPLIT_RETRY_KEPT]

    full = _state(tmp_path / "full", [runs[-1]], gleaning=True)
    incremental = _state(
        tmp_path / "inc",
        runs,
        die_in=1,
        die_at=(method, 1, False, IndexingManager),
        gleaning=True,
        resume=resume,
    )

    assert incremental == full


def _run_steps(tmp: Path, steps: list[tuple[dict[str, str], bool, str]]) -> Providers:
    """Index each ``(corpus, gleaning, pipeline_id)`` step in turn (claims
    on) and return the providers of the stores."""
    config = _config(tmp / "stores", tmp / "viz")
    config.processing.claim_extraction.enabled = True
    config.processing.gleaning.max_rounds = 1
    config.processing.document_parsing.source_scope = "fuzz-corpus"
    providers = _providers(
        config, FakeOllama(_respond, "x", DIM), FakeCrossEncoder("r")
    )
    source = tmp / "corpus"
    for corpus, gleaning, pipeline_id in steps:
        if source.exists():
            shutil.rmtree(source)
        source.mkdir(parents=True)
        for name, text in corpus.items():
            (source / name).write_text(text)
        config.processing.gleaning.enabled = gleaning
        context = _ingest(config, providers, source, tmp / pipeline_id)
        assert context.status is PipelineStageStatus.COMPLETED
    return providers


def _entities(providers: Providers) -> list[tuple]:
    graph = providers.build_graph_indexer()
    try:
        assert isinstance(graph, SQLiteGraphIndexer)
        ids = [i for i, _ in graph.read_entity_names()]
        return sorted((e.name, e.description) for e in graph.read_entities(ids))
    finally:
        graph.close()


PID_V0 = {
    "d7.txt": "Acme and Acme Ltd trade cash 3. "
    "Acme and Acme Ltd and Acme Corp trade cash 1."
}
PID_V1 = {
    **PID_V0,
    "d6.txt": "Acme Inc and Acme Corp trade goods 1. Acme Ltd trade parts 3.",
}


def test_a_finished_pipeline_run_again_does_not_index_over_other_settings(
    tmp_path: Path,
) -> None:
    # P1 indexes d6.txt with gleaning; C then re-extracts the corpus without
    # it. P1 run again restored its own delta (d6.txt new) and indexed it
    # over C's artifacts of d6.txt; the registry then recorded P1's
    # settings, so no later run repaired the store.
    full = _entities(_run_steps(tmp_path / "full", [(PID_V1, True, "F")]))
    incremental = _entities(
        _run_steps(
            tmp_path / "inc",
            [(PID_V0, False, "A"), (PID_V1, True, "P1"), (PID_V1, False, "C")]
            + [(PID_V1, True, "P1")],
        )
    )

    assert incremental == full


def test_a_finished_pipeline_run_again_sees_another_runs_changes(
    tmp_path: Path,
) -> None:
    # P1 finds nothing to do (A indexed v1); C then indexes v2 (d1.txt
    # renamed to r4.txt). P1 run again over v1 restored its empty delta:
    # the stores kept r4.txt and never got d1.txt back.
    v1 = {"d1.txt": "Acme Corp and Buyer trade goods 1."}
    v2 = {"r4.txt": v1["d1.txt"]}
    steps = [(v1, "A"), (v1, "P1"), (v2, "C"), (v1, "P1"), (v1, "P1")]
    providers = _run_steps(
        tmp_path, [(corpus, False, pipeline_id) for corpus, pipeline_id in steps]
    )

    registry = providers.build_doc_status().list_all()
    assert {r.file_path: r.status.value for r in registry} == {"d1.txt": "processed"}


def _scoped_state(
    tmp: Path,
    corpora: dict[str, dict[str, str]],
    steps: list[tuple[str, str, bool, tuple[Any, ...] | None, str | None]],
) -> dict[str, list[tuple]]:
    """The state after each ``(scope, pipeline_id, gleaning, die_at,
    retired_scope)`` step indexes ``corpora[scope]`` under its own source
    scope (``die_at`` as in ``_state``: that run is killed; ``retired_scope``
    is retired by the run). The cache directory is the pipeline id's, as
    for runs of one id over several scopes."""
    config = _config(tmp / "stores", tmp / "viz")
    config.processing.claim_extraction.enabled = True
    config.processing.gleaning.max_rounds = 1
    providers = _providers(
        config, FakeOllama(_respond, "x", DIM), FakeCrossEncoder("r")
    )
    for scope, pipeline_id, gleaning, die_at, retired in steps:
        source = tmp / f"corpus-{scope}"
        if source.exists():
            shutil.rmtree(source)
        source.mkdir(parents=True)
        for name, text in corpora[scope].items():
            (source / name).write_text(text)
        config.processing.document_parsing.source_scope = scope
        config.processing.gleaning.enabled = gleaning
        config.indexing.retire_source_scopes = [retired] if retired else []
        if die_at:
            with (
                patch.object(die_at[3], die_at[0], _dying(*die_at)),
                pytest.raises(_Died),
            ):
                _ingest(config, providers, source, tmp / pipeline_id)
            continue
        context = _ingest(config, providers, source, tmp / pipeline_id)
        assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    return _read_state(config, providers)


@pytest.mark.parametrize(
    ("corpora", "inc_steps", "gleaning"),
    [
        pytest.param(
            {
                "A": {
                    "d0.txt": "ACME Inc and Acme Corp trade parts 3. Buyer and "
                    "ACME trade parts 2."
                },
                "B": {
                    "d6.txt": "Acme trade goods 2.",
                    "r2.txt": "Acme Corp and Acme Ltd trade parts 3.",
                },
                "C": {
                    "d1.txt": "Acme Inc and Buyer and Acme Ltd trade cash 2. "
                    "Buyer and Depot trade parts 3.",
                    "d6.txt": "Acme and Acme Corp trade parts 3.",
                },
            },
            [
                ("C", "P1", False, None, None),
                ("B", "P1", False, None, None),
                (
                    "A",
                    "P1",
                    False,
                    ("upsert_relationships", 1, True, SQLiteGraphIndexer),
                    "B",
                ),
                ("A", "P0", False, None, None),
                ("B", "P0", False, None, None),
                ("C", "P1", False, None, None),
            ],
            False,
            id="killed-after-the-graph-edges",
        ),
        pytest.param(
            {
                "A": {
                    "d7.txt": "Acme LLC and Buyer trade goods 2. Buyer trade goods 1.",
                    "d4.txt": "Depot and Acme Corp and Buyer trade cash 2. Acme "
                    "trade goods 2.",
                },
                "B": {"d0.txt": "Acme trade goods 3. Acme Corp trade parts 3."},
                "C": {
                    "d6.txt": "Acme Corp and Acme Ltd and Depot trade parts 1.",
                    "d7.txt": "Depot and Acme Inc and Acme trade parts 2.",
                },
            },
            [
                ("B", "P0", False, None, None),
                ("C", "P1", False, None, None),
                (
                    "A",
                    "P1",
                    False,
                    ("upsert_relationships", 1, False, SQLiteGraphIndexer),
                    "B",
                ),
                ("C", "PC", True, None, None),
                ("A", "P0", True, None, None),
                ("B", "PC", True, None, None),
                ("C", "PC", True, None, None),
            ],
            True,
            id="killed-before-the-graph-edges",
        ),
    ],
)
def test_a_claim_about_an_underived_entitys_spelling_survives_a_killed_retire(
    tmp_path: Path,
    corpora: dict[str, dict[str, str]],
    inc_steps: list[tuple[str, str, bool, tuple[Any, ...] | None, str | None]],
    gleaning: bool,
) -> None:
    # A run of scope A retiring scope B is killed at its graph edges, after
    # deleting B's extraction records: a stored entity citing B's units is
    # left underived. The next run derived a claim of one of its spellings
    # kept apart by a full build as nothing (its endpoint not derived) and
    # took the unit off the given claim: no claim cited the statement.
    full = _scoped_state(
        tmp_path / "full",
        corpora,
        [(scope, f"F{scope}", gleaning, None, None) for scope in "ABC"],
    )
    incremental = _scoped_state(tmp_path / "inc", corpora, inc_steps)

    assert incremental == full
