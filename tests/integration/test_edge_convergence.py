# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edges of a full build and of incremental runs converge (embedded stores).

* A relationship a gleaning RELATIONSHIP_CORRECTION reverses, stated the
  corrected way round by another document, is one edge in a full build. An
  incremental run indexing that document later must merge into it, not store
  a second, parallel edge.
* A document whose only relationship has strength 0 indexes (Leiden panicked
  on a graph whose every edge weighs 0, ending the run and every resume).
"""

from __future__ import annotations

import re
import shutil
from itertools import pairwise
from pathlib import Path

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
from unified_kg_rag.adapters.local.sqlite_graph import SQLiteGraphIndexer
from unified_kg_rag.domain.models import PipelineStageStatus

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

no_aws = e2e.no_aws

NAMES = ("Acme", "Buyer")


def _names_in(text: str) -> list[str]:
    return sorted((n for n in NAMES if re.search(rf"\b{n}\b", text)), key=text.find)


def _responder(strength: int):
    def respond(system: str, human: str) -> str:
        if "knowledge graph extraction expert" not in system and (
            "graph refinement specialist" not in system
        ):
            return stages._respond(system, human)
        text = stages._source_text(human)
        names = _names_in(text)
        if "knowledge graph extraction expert" in system:
            entities = "".join(
                f"<entity><name>{n}</name><type>ORGANIZATION</type>"
                f"<description>{n} desc</description><confidence>9</confidence>"
                f"<source_text>{n}</source_text></entity>"
                for n in names
            )
            relationships = "".join(
                f"<relationship><source>{a}</source><target>{b}</target>"
                f"<type>TRADES_WITH</type><description>{a} trades with {b}"
                f"</description><strength>{strength}</strength>"
                f"<source_text>{text.strip()}</source_text></relationship>"
                for a, b in pairwise(names)
            )
            return (
                f"<entities>{entities}</entities>"
                f"<relationships>{relationships}</relationships>"
            )
        # Refinement: units about parts state their edge the wrong way round.
        issue = ""
        if "parts" in text and len(names) == 2:
            a, b = names
            issue = (
                "<issue><issue_type>RELATIONSHIP_CORRECTION</issue_type>"
                f"<details><source>{a}</source><target>{b}</target>"
                f"<type>TRADES_WITH</type><corrected_source>{b}"
                f"</corrected_source><corrected_target>{a}</corrected_target>"
                "</details><justification>x</justification>"
                f"<text_evidence>{text.strip()}</text_evidence></issue>"
            )
        return (
            "<refinement_plan><identified_issues>"
            f"{issue}</identified_issues></refinement_plan>"
        )

    return respond


def _edges_after(
    tmp: Path, corpora: list[dict[str, str]], *, strength: int = 8
) -> tuple[list[tuple], int]:
    """(source, target, type, weight, cited units, description lines) of every
    graph edge, and the number of relationship documents in the vector store."""
    config = _config(tmp / "stores", tmp / "viz")
    config.processing.gleaning.enabled = True
    config.processing.gleaning.max_rounds = 1
    providers = _providers(
        config, FakeOllama(_responder(strength), "x", DIM), FakeCrossEncoder("r")
    )
    source = tmp / "corpus"
    for run, corpus in enumerate(corpora):
        if source.exists():
            shutil.rmtree(source)
        source.mkdir(parents=True)
        for name, text in corpus.items():
            (source / name).write_text(text)
        context = _ingest(config, providers, source, tmp / f"run-{run}")
        assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    graph = providers.build_graph_indexer()
    try:
        assert isinstance(graph, SQLiteGraphIndexer)
        ids = [i for i, _ in graph.read_entity_names()]
        edges = graph.read_relationships(graph.find_incident_relationship_ids(ids))
    finally:
        graph.close()
    stored = _stored(
        config, providers, config.indexing.opensearch.relationships_index_prefix
    )
    return (
        sorted(
            (
                r.source_name,
                r.target_name,
                r.type,
                r.weight,
                len(r.text_unit_ids or []),
                frozenset((r.description or "").split("\n")),
            )
            for r in edges
        ),
        len(stored),
    )


CORPUS = {
    "a.txt": "Acme and Buyer trade goods.",
    "b.txt": "Buyer and Acme trade parts.",
}


def test_a_reversed_edge_merges_with_its_corrected_form_across_runs(
    tmp_path: Path,
) -> None:
    full = _edges_after(tmp_path / "full", [CORPUS])
    incremental = _edges_after(tmp_path / "inc", [{"b.txt": CORPUS["b.txt"]}, CORPUS])

    edges, documents = full
    assert [edge[:5] for edge in edges] == [("Acme", "Buyer", "TRADES_WITH", 16.0, 2)]
    assert documents == 1
    assert incremental == full


def test_a_document_whose_only_edge_has_strength_zero_indexes(
    tmp_path: Path,
) -> None:
    corpus = {"a.txt": CORPUS["a.txt"]}
    full = _edges_after(tmp_path / "full", [corpus], strength=0)
    # The incremental run clusters only the new document's subgraph.
    incremental = _edges_after(
        tmp_path / "inc",
        [{"c.txt": "Acme ships."}, {**corpus, "c.txt": "Acme ships."}],
        strength=0,
    )

    edges, _ = full
    assert [edge[:5] for edge in edges] == [("Acme", "Buyer", "TRADES_WITH", 0.0, 1)]
    assert incremental == full
