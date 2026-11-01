# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declared relationship types in the extraction prompt, and in-schema labels
written as declared before ids are built."""

from __future__ import annotations

from typing import Any

import pytest

import unified_kg_rag.adapters.ingestion.gleaner as gleaner_module
import unified_kg_rag.adapters.ingestion.graph_extractor as extractor_module
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.ingestion.gleaner import GraphGleaner
from unified_kg_rag.adapters.ingestion.graph_extractor import GraphExtractor
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.cache_keys import unit_record_fingerprint
from unified_kg_rag.domain.ingestion.schema_conformance import SchemaConformance
from unified_kg_rag.domain.models import (
    Config,
    ExtractionSchema,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.domain.prompts import GraphExtractionPrompt
from unified_kg_rag.domain.prompts.graph_extraction import (
    DEFAULT_RELATIONSHIP_TYPES,
    format_relationship_types,
)

pytestmark = pytest.mark.unit

TEXT = "Alice works for Acme Corp."
UNIT = TextUnit(id="t1", text=TEXT)
_SCHEMA: dict[str, Any] = {
    "entity_types": ["PERSON: People", "ORGANIZATION: Companies"],
    "relationship_types": [
        "WORKS_FOR: Employment",
        {"label": "PARTNERS_WITH", "iri": "urn:x:partner"},
    ],
    "relationship_patterns": ["PERSON -WORKS_FOR-> ORGANIZATION"],
}


def _config(**extraction: Any) -> Config:
    return Config.model_validate({"processing": {"graph_extraction": extraction}})


@pytest.fixture
def patched(mocker) -> None:
    mocker.patch.object(bedrock_module, "BedrockLanguageModelFactory")
    for module in (extractor_module, gleaner_module):
        mocker.patch.object(module, "create_robust_xml_output_parser")
        mocker.patch.object(module, "setup_chain")


def _record(config: Config, entity_type: str, relation: str) -> TextUnitExtraction:
    extractor = GraphExtractor(config)
    return extractor._parse_extraction_result(
        {
            "entities": [
                {"name": "Alice", "type": entity_type},
                {"name": "Acme Corp", "type": "organization"},
            ],
            "relationships": [
                {"source": "Alice", "target": "Acme Corp", "type": relation}
            ],
        },
        UNIT,
    )


# --- prompt --------------------------------------------------------------------


def test_declared_relationship_types_and_patterns_render() -> None:
    block = GraphExtractionPrompt.config_variables(_config(**_SCHEMA))[
        "relationship_types"
    ]
    assert block == (
        "## Relationship Types (STRICT - use only these types):\n"
        "- **WORKS_FOR**: Employment\n"
        "- **PARTNERS_WITH**\n"
        "Allowed patterns:\n"
        "- (PERSON)-[WORKS_FOR]->(ORGANIZATION)"
    )


def test_types_without_patterns_render_no_pattern_block() -> None:
    schema = ExtractionSchema.from_items(relationship_types=["LOCATED_IN: Where"])
    assert format_relationship_types(schema) == (
        "## Relationship Types (STRICT - use only these types):\n"
        "- **LOCATED_IN**: Where"
    )


@pytest.mark.parametrize(
    "extraction",
    [{}, {"schema_mode": "strict"}, {"entity_types": ["GENE"]}],
)
def test_no_declared_relationship_type_keeps_the_examples(
    extraction: dict[str, Any],
) -> None:
    variables = GraphExtractionPrompt.config_variables(_config(**extraction))
    assert variables["relationship_types"] == DEFAULT_RELATIONSHIP_TYPES


def test_the_mode_does_not_change_the_prompt() -> None:
    rendered = {
        mode: GraphExtractionPrompt.config_variables(
            _config(**_SCHEMA, schema_mode=mode)
        )
        for mode in ("off", "align", "strict")
    }
    assert rendered["off"] == rendered["align"] == rendered["strict"]


def test_the_record_fingerprint_hashes_the_declared_block() -> None:
    # Two schemas whose keys tell them apart only by the rendered block
    # still differ (the config keys them as well).
    one = _config(relationship_types=["WORKS_FOR: Employment"])
    other = _config(relationship_types=["WORKS_FOR: Jobs"])
    assert unit_record_fingerprint(one, gleaned=False) != unit_record_fingerprint(
        other, gleaned=False
    )


# --- when labels are written as declared --------------------------------------


@pytest.mark.parametrize(
    ("extraction", "applies"),
    [
        ({}, False),
        ({"entity_types": ["GENE", "DISEASE"]}, False),
        ({"schema_mode": "off", **_SCHEMA}, False),
        ({"relationship_types": ["WORKS_FOR"]}, True),
        (_SCHEMA, True),
        ({"schema_mode": "strict"}, True),
        ({"schema_mode": "strict", "entity_types": ["GENE"]}, True),
    ],
)
def test_when_the_schema_applies(extraction: dict[str, Any], applies: bool) -> None:
    schema = _config(**extraction).processing.graph_extraction.extraction_schema()
    assert SchemaConformance(schema).applies is applies


def test_labels_are_matched_by_schema_key_only() -> None:
    conformance = SchemaConformance(
        ExtractionSchema.from_items(
            ["PERSON", "LEGAL_ENTITY"], ["WORKS_FOR"], mode="align"
        )
    )
    assert conformance.relationship_type("works for") == "WORKS_FOR"
    assert conformance.relationship_type(" Works__For ") == "WORKS_FOR"
    assert conformance.entity_type("legal entity") == "LEGAL_ENTITY"
    # Nothing looser: kept as written.
    assert conformance.relationship_type("works-for") == "works-for"
    assert conformance.relationship_type("WORKSFOR") == "WORKSFOR"
    assert conformance.entity_type("PERSONS") == "PERSONS"
    assert conformance.entity_type("") == ""


@pytest.mark.parametrize("mode", ["align", "strict"])
def test_spellings_of_a_declared_type_share_one_id(patched, mode: str) -> None:
    config = _config(**_SCHEMA, schema_mode=mode)
    records = [
        _record(config, "person", spelling)
        for spelling in ("works for", "WORKS_FOR", "Works_For", "works  for")
    ]
    relationships = [r.relationships[0] for r in records]
    assert {rel.type for rel in relationships} == {"WORKS_FOR"}
    assert len({rel.id for rel in relationships}) == 1
    assert relationships[0].id == BaseProcessor._generate_relationship_id(
        "Alice", "Acme Corp", "WORKS_FOR"
    )


def test_off_and_entity_types_alone_keep_the_label_as_written(patched) -> None:
    for config in (
        _config(**_SCHEMA, schema_mode="off"),
        _config(entity_types=["PERSON", "ORGANIZATION"]),
        Config(),
    ):
        record = _record(config, "Legal  Entity", "works for")
        assert record.relationships[0].type == "works for"
        assert record.entities[0].type == "legal  entity"


def test_an_undeclared_type_is_kept_as_written_under_align(patched) -> None:
    record = _record(_config(**_SCHEMA), "wizard", "located in")
    assert record.relationships[0].type == "located in"
    assert {e.name: e.type for e in record.entities} == {
        "Alice": "wizard",
        "Acme Corp": "organization",
    }


def test_entity_types_are_written_as_declared_before_the_vote(patched) -> None:
    # "legal entity" and "LEGAL_ENTITY" are one declared type: the unit's two
    # statements vote for it together instead of splitting.
    config = _config(
        entity_types=["PERSON", "LEGAL_ENTITY"],
        relationship_types=["WORKS_FOR"],
    )
    extractor = GraphExtractor(config)
    record = extractor._parse_extraction_result(
        {
            "entities": [
                {"name": "Acme", "type": "legal entity"},
                {"name": "Acme", "type": "LEGAL_ENTITY"},
                {"name": "Acme", "type": "person"},
            ]
        },
        UNIT,
    )
    assert [e.type for e in record.entities] == ["legal_entity"]


def test_gleaning_corrections_write_declared_labels(patched) -> None:
    config = _config(**_SCHEMA)
    gleaner = GraphGleaner(config, show_progress=False)
    record = _record(config, "person", "partners with")
    assert record.relationships[0].type == "PARTNERS_WITH"
    gleaner._apply_answer(
        {
            "identified_issues": {
                "issue": [
                    {
                        "issue_type": "RELATIONSHIP_CORRECTION",
                        "details": {
                            "source": "Alice",
                            "target": "Acme Corp",
                            "corrected_type": "works for",
                        },
                    },
                    {
                        "issue_type": "ENTITY_CORRECTION",
                        "details": {"name": "Acme Corp", "corrected_type": "Person"},
                    },
                ]
            }
        },
        UNIT,
        record,
    )
    (relationship,) = record.relationships
    assert relationship.type == "WORKS_FOR"
    assert relationship.id == BaseProcessor._generate_relationship_id(
        "Alice", "Acme Corp", "WORKS_FOR"
    )
    assert {e.name: e.type for e in record.entities}["Acme Corp"] == "PERSON"


def test_a_gleaned_relationship_takes_the_declared_label(patched) -> None:
    config = _config(**_SCHEMA)
    gleaner = GraphGleaner(config, show_progress=False)
    record = TextUnitExtraction(text_unit_id=UNIT.id)
    gleaner._apply_answer(
        {
            "identified_issues": {
                "issue": [
                    {
                        "issue_type": "MISSING_RELATIONSHIP",
                        "details": {
                            "source": "Alice",
                            "target": "Acme Corp",
                            "type": "Works For",
                        },
                    }
                ]
            }
        },
        UNIT,
        record,
    )
    (relationship,) = record.relationships
    assert isinstance(relationship, Relationship)
    assert relationship.type == "WORKS_FOR"
