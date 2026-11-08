# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An entity's id follows its current name across incremental runs (embedded
stores, the real pipeline).

"Acme" is indexed first; two later documents call it "Acme Inc", the spelling
a full build keeps (more text units). The incremental run must store it under
that name's id, as the full build does, and leave nothing under the old id.
"""

from __future__ import annotations

import re
import shutil
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
from unified_kg_rag.domain.models import PipelineStageStatus

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

no_aws = e2e.no_aws

NAMES = ("Acme Inc", "Acme", "Buyer")
FIRST = {"a.txt": "Acme supplies Buyer."}
FLIPPED = {
    **FIRST,
    "d.txt": "Acme Inc supplies Buyer.",
    "e.txt": "Acme Inc pays Buyer.",
}


def _respond(system: str, human: str) -> str:
    if "knowledge graph extraction expert" not in system:
        return stages._respond(system, human)
    text = stages._source_text(human)
    names, rest = [], text
    for name in NAMES:
        if re.search(rf"\b{re.escape(name)}\b", rest):
            names.append(name)
            rest = re.sub(rf"\b{re.escape(name)}\b", " ", rest)
    entities = "".join(
        f"<entity><name>{n}</name><type>ORGANIZATION</type><description>{n}"
        f"</description><confidence>9</confidence><source_text>{n}</source_text>"
        "</entity>"
        for n in names
    )
    source, target = sorted(names, key=text.find)
    relationship = (
        f"<relationship><source>{source}</source><target>{target}</target>"
        f"<type>WORKS_WITH</type><description>{text.strip()}</description>"
        f"<strength>8</strength><source_text>{text.strip()}</source_text>"
        "</relationship>"
    )
    return (
        f"<entities>{entities}</entities><relationships>{relationship}</relationships>"
    )


def _entities_after(tmp: Path, corpora: list[dict[str, str]]) -> dict[str, str]:
    """``{name: id}`` of the stored entities after indexing ``corpora`` in turn."""
    config = _config(tmp / "stores", tmp / "viz")
    providers = _providers(
        config, FakeOllama(_respond, "x", DIM), FakeCrossEncoder("r")
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
    names = dict(graph.read_entity_names())
    graph.close()
    vector_ids = {
        hit["_id"]
        for hit in _stored(
            config, providers, config.indexing.opensearch.entities_index_prefix
        )
    }
    assert vector_ids == set(names)
    return {name: entity_id for entity_id, name in names.items()}


def test_an_incremental_run_stores_the_full_builds_entity_ids(tmp_path: Path) -> None:
    full = _entities_after(tmp_path / "full", [FLIPPED])
    incremental = _entities_after(tmp_path / "inc", [FIRST, FLIPPED])

    assert set(full) == {"Acme Inc", "Buyer"}
    assert incremental == full


# Every report's one finding cites every record of its input.
_REPORT = (
    "<community_name>Group</community_name><summary>S.</summary><rating>5"
    "</rating><rating_explanation>r</rating_explanation><findings><finding>"
    "<summary>F</summary><explanation>E [Data: Entities (1, 2, 3, 4, 5); "
    "Relationships (1, 2, 3)]</explanation></finding></findings>"
)
_CHAIN = ("Acme Inc", "Acme", "Buyer", "Depot", "Carrier")
_CHAIN_FIRST = {"a.txt": "Acme supplies Buyer. Buyer uses Depot. Depot hires Carrier."}
_CHAIN_FLIPPED = {
    **_CHAIN_FIRST,
    "d.txt": "Acme Inc supplies Buyer.",
    "e.txt": "Acme Inc pays Buyer.",
}


def _chain_respond(system: str, human: str) -> str:
    """Entities in text order, each linked to the next; every report cites
    all its records."""
    if "community analysis" in system:
        return _REPORT
    if "knowledge graph extraction expert" not in system:
        return stages._respond(system, human)
    text = stages._source_text(human)
    names, rest = [], text
    for name in _CHAIN:
        if re.search(rf"\b{re.escape(name)}\b", rest):
            names.append(name)
            rest = re.sub(rf"\b{re.escape(name)}\b", " ", rest)
    names.sort(key=text.find)
    entities = "".join(
        f"<entity><name>{n}</name><type>ORGANIZATION</type><description>{n}"
        f"</description><confidence>9</confidence><source_text>{n}</source_text>"
        "</entity>"
        for n in names
    )
    relationships = "".join(
        f"<relationship><source>{a}</source><target>{b}</target><type>WORKS_WITH"
        f"</type><description>{a} with {b}</description><strength>8</strength>"
        f"<source_text>{text.strip()}</source_text></relationship>"
        for a, b in zip(names, names[1:], strict=False)
    )
    return (
        f"<entities>{entities}</entities><relationships>{relationships}</relationships>"
    )


def test_stored_reports_cite_the_ids_a_later_run_re_keys_to(tmp_path: Path) -> None:
    # The first run's reports cite "Acme" and its edge under their first ids;
    # the second re-keys them to "Acme Inc"'s and deletes the old ids. The
    # stored reports it did not regenerate cited the deleted ids.
    config = _config(tmp_path / "stores", tmp_path / "viz")
    providers = _providers(
        config, FakeOllama(_chain_respond, "x", DIM), FakeCrossEncoder("r")
    )
    source = tmp_path / "corpus"
    for run, corpus in enumerate([_CHAIN_FIRST, _CHAIN_FLIPPED]):
        if source.exists():
            shutil.rmtree(source)
        source.mkdir(parents=True)
        for name, text in corpus.items():
            (source / name).write_text(text)
        context = _ingest(config, providers, source, tmp_path / f"run-{run}")
        assert context.status is PipelineStageStatus.COMPLETED, _errors(context)

    os_config = config.indexing.opensearch
    entities = {
        hit["_id"]
        for hit in _stored(config, providers, os_config.entities_index_prefix)
    }
    edges = {
        hit["_id"]
        for hit in _stored(config, providers, os_config.relationships_index_prefix)
    }
    reports = [
        hit["_source"]
        for hit in _stored(config, providers, os_config.community_reports_index_prefix)
    ]
    cited = [
        (key, cited_id)
        for report in reports
        for finding in report["findings"]
        for key in ("entity_ids", "relationship_ids")
        for cited_id in finding[key]
    ]
    assert len(reports) > 1 and cited
    assert [
        pair
        for pair in cited
        if pair[1] not in (entities if pair[0] == "entity_ids" else edges)
    ] == []
    # The searchable list follows the findings.
    for report in reports:
        assert set(report["findings_record_ids"]) == {
            cited_id
            for finding in report["findings"]
            for key in ("entity_ids", "relationship_ids")
            for cited_id in finding[key]
        }
