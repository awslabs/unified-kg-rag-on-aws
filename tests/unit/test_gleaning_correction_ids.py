# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A gleaning correction edits its own text unit's record and nothing else.

Each unit's statements live in its own record, so a correction renames or
re-keys only what the correcting unit said: another unit's statement of the
old name or the uncorrected edge stays a distinct item with its own id, as it
is when an incremental run indexes that unit in another run. Ids stay
name-derived, so the merged graph never holds two items with one id.
"""

from __future__ import annotations

from collections import Counter

import pytest

import unified_kg_rag.adapters.ingestion.gleaner as gleaner_module
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.ingestion.gleaner import GraphGleaner
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.contributions import merge_contributions
from unified_kg_rag.domain.ingestion.graph_resolver import GraphResolver
from unified_kg_rag.domain.ingestion.relationship_weights import (
    apply_text_unit_weights,
)
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)

pytestmark = pytest.mark.unit

TYPO = BaseProcessor._generate_entity_id("Acme Crop")
FIXED = BaseProcessor._generate_entity_id("Acme Corp")
DEPOT = BaseProcessor._generate_entity_id("Depot")
EDGE = BaseProcessor._generate_relationship_id("Acme Crop", "Depot", "SUPPLIES")
U1 = TextUnit(
    id="u1", text="Acme Corp supplies the Depot.", attributes={"filters": {"d": "a"}}
)


@pytest.fixture
def gleaner(config: Config, mocker) -> GraphGleaner:
    mocker.patch.object(bedrock_module, "BedrockLanguageModelFactory")
    mocker.patch.object(gleaner_module, "create_robust_xml_output_parser")
    mocker.patch.object(gleaner_module, "setup_chain")
    return GraphGleaner(config)


def _record(unit: str, strength: float, document: str) -> TextUnitExtraction:
    """``unit`` lists the misspelled name and Depot and states one edge."""
    placement = {"filters": {"d": document}}
    edge = Relationship(
        id=EDGE,
        source_id=TYPO,
        source_name="Acme Crop",
        target_id=DEPOT,
        target_name="Depot",
        type="SUPPLIES",
        attributes=dict(placement),
    )
    apply_text_unit_weights(edge, {unit: strength})
    return TextUnitExtraction(
        text_unit_id=unit,
        entities=[
            Entity(
                id=entity_id,
                name=name,
                type="ORG",
                text_unit_ids=[unit],
                attributes=dict(placement),
            )
            for entity_id, name in ((TYPO, "Acme Crop"), (DEPOT, "Depot"))
        ],
        relationships=[edge],
    )


def _correct(gleaner: GraphGleaner, record: TextUnitExtraction, *issues) -> None:
    """Apply one answer of ``issues`` (they name things as the prompt did)."""
    plan = {
        "identified_issues": {
            "issue": [
                {"issue_type": issue_type, "details": details}
                for issue_type, details in issues
            ]
        }
    }
    gleaner._apply_answer(plan, U1, record)


RENAME = ("ENTITY_CORRECTION", {"name": "Acme Crop", "corrected_name": "Acme Corp"})
RETYPE = (
    "RELATIONSHIP_CORRECTION",
    {
        "source": "Acme Crop",
        "target": "Depot",
        "type": "SUPPLIES",
        "corrected_type": "SHIPS_TO",
    },
)
REVERSE = (
    "RELATIONSHIP_CORRECTION",
    {
        "source": "Acme Crop",
        "target": "Depot",
        "corrected_source": "Depot",
        "corrected_target": "Acme Crop",
    },
)


def _assert_unique_ids(entities: list[Entity], relationships: list[Relationship]):
    assert [i for i, n in Counter(e.id for e in entities).items() if n > 1] == []
    assert [i for i, n in Counter(r.id for r in relationships).items() if n > 1] == []


def test_corrections_reach_only_the_correcting_units_statements(gleaner) -> None:
    u1, u2 = _record("u1", 3.0, "a"), _record("u2", 5.0, "b")
    _correct(gleaner, u1, RENAME, RETYPE)

    entities, relationships = merge_contributions([u1, u2])

    _assert_unique_ids(entities, relationships)
    by_name = {e.name: e for e in entities}
    assert by_name["Acme Corp"].id == FIXED
    assert by_name["Acme Corp"].text_unit_ids == ["u1"]
    assert by_name["Acme Crop"].text_unit_ids == ["u2"]
    assert sorted(by_name["Depot"].text_unit_ids) == ["u1", "u2"]
    edges = {(r.source_name, r.type): r for r in relationships}
    corrected = edges[("Acme Corp", "SHIPS_TO")]
    assert corrected.id == BaseProcessor._generate_relationship_id(
        "Acme Corp", "Depot", "SHIPS_TO"
    )
    assert (corrected.text_unit_ids, corrected.weight) == (["u1"], 3.0)
    assert corrected.attributes["filters"] == {"d": "a"}
    stated = edges[("Acme Crop", "SUPPLIES")]
    assert (stated.id, stated.text_unit_ids, stated.weight) == (EDGE, ["u2"], 5.0)
    assert stated.attributes["filters"] == {"d": "b"}


def test_a_reversal_leaves_the_other_units_statement_as_stated(gleaner) -> None:
    u1, u2 = _record("u1", 3.0, "a"), _record("u2", 5.0, "b")
    _correct(gleaner, u1, REVERSE)

    _, relationships = merge_contributions([u1, u2])

    by_id = {r.id: r for r in relationships}
    reversed_id = BaseProcessor._generate_relationship_id(
        "Depot", "Acme Crop", "SUPPLIES"
    )
    assert set(by_id) == {EDGE, reversed_id}
    assert (by_id[reversed_id].source_name, by_id[reversed_id].weight) == (
        "Depot",
        3.0,
    )
    assert (by_id[EDGE].text_unit_ids, by_id[EDGE].weight) == (["u2"], 5.0)
    assert u2 == _record("u2", 5.0, "b")


@pytest.mark.parametrize("fuzzy", [False, True])
def test_resolution_of_corrected_records_keeps_ids_unique(gleaner, fuzzy) -> None:
    u1, u2 = _record("u1", 3.0, "a"), _record("u2", 5.0, "b")
    _correct(gleaner, u1, RENAME, RETYPE)
    config = Config()
    config.processing.fuzzy_entity_resolution = fuzzy

    result, _ = GraphResolver(config, use_process_pool=False).resolve_graph(
        *merge_contributions([u1, u2])
    )

    _assert_unique_ids(result["entities"], result["relationships"])
