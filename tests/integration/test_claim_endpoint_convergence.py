# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A claim's subject and object after a removal, and of a claim naming an
entity its own text unit's extraction did not list (embedded stores).

A removal that deletes an entity (only the removed text units stated its
spelling) must leave no stored claim pointing at it: the claim follows the
spelling its units state now, as a full build of the remaining corpus
resolves it. A claim resolves only against the entities its own text units
stated, in a full build as in an incremental run, so whichever documents
came first the stored claims are the same.
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

NAMES = ("Acme Inc", "Acme", "Acme Corp", "Acme LLC", "Buyer", "Depot")
TYPES = {"goods": "ORGANIZATION", "parts": "PERSON", "cash": "LOCATION"}


def _names_in(text: str) -> list[str]:
    found, rest = [], text
    for name in sorted(NAMES, key=len, reverse=True):
        pattern = rf"\b{re.escape(name)}\b"
        if re.search(pattern, rest):
            found.append(name)
            rest = re.sub(pattern, " ", rest)
    return sorted(found, key=text.find)


def _entities(names: list[str], kind: str, text: str) -> str:
    entities = "".join(
        f"<entity><name>{n}</name><type>{kind}</type><description>{n}: {text}"
        f"</description><confidence>9</confidence><source_text>{n}</source_text>"
        "</entity>"
        for n in names
    )
    relationships = "".join(
        f"<relationship><source>{a}</source><target>{b}</target>"
        f"<type>WORKS_WITH</type><description>{a}-{b}: {text}</description>"
        f"<strength>8</strength><source_text>{text}</source_text></relationship>"
        for a, b in pairwise(names)
    )
    return (
        f"<entities>{entities}</entities><relationships>{relationships}</relationships>"
    )


def _claim(subject: str, obj: str, text: str) -> str:
    return (
        f"<claims><claim><subject>{subject}</subject><object>{obj}</object>"
        "<claim_type>OBLIGATION</claim_type><claim_status>SUSPECTED</claim_status>"
        f"<description>{subject} owes {obj}: {text}</description>"
        f"<source_text>{text}</source_text></claim></claims>"
    )


_EXTRACTION = ("knowledge graph extraction expert", "claim extraction specialist")


def _spellings(system: str, human: str) -> str:
    # Every name a unit states is listed; a claim names its first two.
    if not any(kind in system for kind in _EXTRACTION):
        return stages._respond(system, human)
    text = stages._source_text(human).strip()
    names = _names_in(text)
    if "knowledge graph extraction expert" in system:
        kind = next((v for k, v in TYPES.items() if k in text), "ORGANIZATION")
        return _entities(names, kind, text)
    if "claim extraction specialist" in system:
        return _claim(*names[:2], text) if len(names) > 1 else "<claims></claims>"
    return stages._respond(system, human)


def _unlisted(system: str, human: str) -> str:
    # Entity extraction lists Globex only where it is the topic; a claim
    # about it is stated elsewhere too.
    if not any(kind in system for kind in _EXTRACTION):
        return stages._respond(system, human)
    text = stages._source_text(human).strip()
    if "knowledge graph extraction expert" in system:
        names = ["Globex"] if "Globex trades" in text else ["Buyer", "Depot"]
        return _entities(names, "ORGANIZATION", text)
    if "claim extraction specialist" in system:
        if "owes" not in text:
            return "<claims></claims>"
        pair = ("Globex", "Buyer") if "Globex owes" in text else ("Buyer", "Globex")
        return _claim(*pair, text)
    return stages._respond(system, human)


def _claims_after(
    tmp: Path, respond, corpora: list[dict[str, str]]
) -> tuple[list[tuple], set[str]]:
    """(subject name, object name, literal object) of every stored claim, and
    the entity ids claims point at that no stored entity has."""
    config = _config(tmp / "stores", tmp / "viz")
    config.processing.claim_extraction.enabled = True
    providers = _providers(config, FakeOllama(respond, "x", DIM), FakeCrossEncoder("r"))
    source = tmp / "corpus"
    for run, corpus in enumerate(corpora):
        if source.exists():
            shutil.rmtree(source)
        source.mkdir(parents=True)
        for name, text in corpus.items():
            (source / name).write_text(text)
        context = _ingest(config, providers, source, tmp / f"run-{run}")
        assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    entity_ids = {
        hit["_id"]
        for hit in _stored(
            config, providers, config.indexing.opensearch.entities_index_prefix
        )
    }
    try:
        hits = _stored(
            config, providers, config.indexing.opensearch.claims_index_prefix
        )
    except Exception:  # noqa: BLE001 - no claim was ever written
        hits = []
    claims = sorted(
        (
            hit["_source"]["subject_name"],
            hit["_source"]["object_name"],
            hit["_source"].get("object_id") is None,
        )
        for hit in hits
    )
    endpoints = {
        hit["_source"].get(field)
        for hit in hits
        for field in ("subject_id", "object_id")
    }
    return claims, endpoints - entity_ids - {None}


KEPT = {
    "d2.txt": "Buyer and Acme Corp and Acme Inc trade parts 1.",
    "d7.txt": "Depot and Acme Inc trade parts 3.",
    "d3.txt": "Acme Inc trade goods 3.",
}
REMOVED = {
    "r2.txt": "Buyer and Acme Corp and Acme trade parts 3. Acme Inc and Acme Corp "
    "trade cash 2. Acme and Acme LLC and Acme Corp trade goods 2."
}


def test_a_claim_whose_object_a_removal_deletes_follows_the_kept_spelling(
    tmp_path: Path,
) -> None:
    # "Acme Corp" is an entity of its own while the removed document states
    # it; once removed, "Acme Corp" (still stated in d2.txt) is a spelling of
    # "Acme Inc", which a full build points the d2.txt claim at.
    full, _ = _claims_after(tmp_path / "full", _spellings, [KEPT])
    incremental, dangling = _claims_after(
        tmp_path / "inc", _spellings, [{**REMOVED, **KEPT}, KEPT]
    )
    assert ("Buyer", "Acme Inc", False) in full
    assert not dangling, dangling
    assert incremental == full


@pytest.mark.parametrize(
    "statement", ["Buyer owes Globex and Depot.", "Globex owes Buyer and Depot."]
)
def test_a_claim_about_an_entity_a_removal_deletes(
    tmp_path: Path, statement: str
) -> None:
    # The claim's unit names Globex, its entity extraction did not list it:
    # a subject stated elsewhere only drops the claim, an object stays a
    # literal value, before and after the document stating Globex goes.
    kept = {"k.txt": statement}
    removed = {"r.txt": "Globex trades goods."}
    full, _ = _claims_after(tmp_path / "full", _unlisted, [kept])
    incremental, dangling = _claims_after(
        tmp_path / "inc", _unlisted, [{**kept, **removed}, kept]
    )
    both, _ = _claims_after(tmp_path / "both", _unlisted, [{**kept, **removed}])
    expected = [("Buyer", "Globex", True)] if statement.startswith("Buyer") else []
    assert full == both == expected
    assert not dangling, dangling
    assert incremental == full


@pytest.mark.parametrize(
    "statement", ["Buyer owes Globex and Depot.", "Globex owes Buyer and Depot."]
)
def test_a_claim_added_about_an_entity_another_document_states(
    tmp_path: Path, statement: str
) -> None:
    # An earlier run stored Globex; the claim's own unit does not list it.
    kept = {"r.txt": "Globex trades goods."}
    added = {"k.txt": statement}
    full, _ = _claims_after(tmp_path / "full", _unlisted, [{**kept, **added}])
    incremental, dangling = _claims_after(
        tmp_path / "inc", _unlisted, [kept, {**kept, **added}]
    )
    assert not dangling, dangling
    assert incremental == full
