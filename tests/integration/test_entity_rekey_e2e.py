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
