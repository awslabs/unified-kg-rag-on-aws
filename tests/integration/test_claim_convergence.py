# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Claims of a full build and of incremental runs converge (embedded stores).

One claim stated with two spellings of its subject ("Acme Inc", "Acme") in
two documents is one claim in a full build. Incremental runs that index the
documents one after the other must store that same single claim (and remove
the right text unit when a document is deleted), not one claim per spelling.
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
from unified_kg_rag.domain.models import PipelineStageStatus

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

no_aws = e2e.no_aws

NAMES = ("Acme Inc", "Acme", "Buyer", "Depot", "Globex")


def _names_in(text: str) -> list[str]:
    found, rest = [], text
    for name in sorted(NAMES, key=len, reverse=True):
        pattern = rf"\b{re.escape(name)}\b"
        if re.search(pattern, rest):
            found.append(name)
            rest = re.sub(pattern, " ", rest)
    return sorted(found, key=text.find)


def _respond(system: str, human: str) -> str:
    if "knowledge graph extraction expert" in system:
        text = stages._source_text(human)
        names = _names_in(text)
        entities = "".join(
            f"<entity><name>{n}</name><type>ORGANIZATION</type>"
            f"<description>{n} desc</description><confidence>9</confidence>"
            f"<source_text>{n}</source_text></entity>"
            for n in names
        )
        relationships = "".join(
            f"<relationship><source>{a}</source><target>{b}</target>"
            f"<type>WORKS_WITH</type><description>{a} works with {b}"
            f"</description><strength>8</strength>"
            f"<source_text>{text.strip()}</source_text></relationship>"
            for a, b in pairwise(names)
        )
        return f"<entities>{entities}</entities><relationships>{relationships}</relationships>"
    if "claim extraction specialist" in system:
        names = _names_in(stages._source_text(human))
        if len(names) < 2:
            return "<claims></claims>"
        subject, obj = names[:2]
        return (
            f"<claims><claim><subject>{subject}</subject><object>{obj}</object>"
            "<claim_type>OBLIGATION</claim_type><claim_status>TRUE</claim_status>"
            f"<description>{subject} must deliver to {obj}</description>"
            f"<source_text>{subject} supplies {obj}.</source_text></claim></claims>"
        )
    return stages._respond(system, human)


def _claims_after(tmp: Path, corpora: list[dict[str, str]]) -> list[tuple]:
    """(subject name, object name, cited text units) of every stored claim.

    Compared by the names of the entities the claim points at: the entity
    ids themselves may differ between the two builds (an incremental run
    keeps the stored entity's id).
    """
    config = _config(tmp / "stores", tmp / "viz")
    config.processing.claim_extraction.enabled = True
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
    names = {
        hit["_id"]: hit["_source"]["name"]
        for hit in _stored(
            config, providers, config.indexing.opensearch.entities_index_prefix
        )
    }
    hits = _stored(config, providers, config.indexing.opensearch.claims_index_prefix)
    return sorted(
        (
            names[hit["_source"]["subject_id"]].removesuffix(" Inc"),
            names[hit["_source"]["object_id"]],
            len(hit["_source"].get("text_unit_ids") or []),
        )
        for hit in hits
    )


FIRST = {"a.txt": "Acme Inc supplies Buyer.", "c.txt": "Globex pays Depot."}
FINAL = {"a.txt": "Acme Inc supplies Buyer.", "d.txt": "Acme supplies Buyer."}


def test_incremental_claims_match_a_full_build(tmp_path: Path) -> None:
    full = _claims_after(tmp_path / "full", [FINAL])
    incremental = _claims_after(tmp_path / "inc", [FIRST, FINAL])

    assert full == [("Acme", "Buyer", 2)]
    assert incremental == full


def test_deleting_a_document_drops_its_text_unit_from_the_claim(
    tmp_path: Path,
) -> None:
    after_delete = {"a.txt": FINAL["a.txt"]}
    full = _claims_after(tmp_path / "full", [after_delete])
    incremental = _claims_after(tmp_path / "inc", [FIRST, FINAL, after_delete])

    assert full == [("Acme", "Buyer", 1)]
    assert incremental == full
