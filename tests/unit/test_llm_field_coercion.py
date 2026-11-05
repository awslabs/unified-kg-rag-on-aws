# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-parsed fields that arrive as lists (repeated XML tags) are coerced."""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.ingestion.claim_extractor import ClaimExtractor
from unified_kg_rag.adapters.ingestion.community_detector import CommunityDetector
from unified_kg_rag.adapters.llm.langchain import RobustXMLOutputParser
from unified_kg_rag.domain.ingestion.base_processor import (
    BaseProcessor,
    coerce_llm_text,
)
from unified_kg_rag.domain.models import Config, TextUnit

pytestmark = pytest.mark.unit


def _text_unit() -> TextUnit:
    return TextUnit(id="tu1", short_id="tu1", text="Vendor supplies Buyer.")


@pytest.mark.parametrize(
    ("value", "join", "expected"),
    [
        (None, False, ""),
        ({}, False, ""),  # an empty element (<type></type>)
        ([{}, "Vendor"], False, "Vendor"),
        ("  Vendor ", False, "Vendor"),
        (["", " Vendor ", "Supplier"], False, "Vendor"),
        (["", "  "], False, ""),
        (["First.", "Second.", "First."], True, "First.\nSecond."),
        (3, False, "3"),
        # Markup in a field: its own text, else its children's.
        ({"#text": " Beta ", "@lang": "en"}, False, "Beta"),
        ({"p": "one"}, False, "one"),
        ({"p": ["one", "two"]}, True, "one\ntwo"),
        ({"b": "one", "i": "two"}, False, "one"),
        ({"@lang": "en"}, False, ""),
    ],
)
def test_coerce_llm_text(value, join: bool, expected: str) -> None:
    assert coerce_llm_text(value, join=join) == expected


def test_relationship_with_repeated_tags_is_parsed() -> None:
    proc = BaseProcessor(config=Config())
    rel = proc.parse_relationship_data(
        {
            "source": ["Vendor", "Vendor"],
            "target": ["Buyer"],
            "type": ["SUPPLIES", "SELLS_TO"],
            "description": ["Vendor supplies parts.", "Paid monthly."],
        },
        _text_unit(),
        entity_name_to_id={},
    )
    assert rel is not None
    assert (rel.source_name, rel.target_name, rel.type) == (
        "Vendor",
        "Buyer",
        "SUPPLIES",
    )
    assert rel.description == "Vendor supplies parts.\nPaid monthly."


def test_entity_with_repeated_tags_is_parsed() -> None:
    proc = BaseProcessor(config=Config())
    entity = proc.parse_entity_data(
        {
            "name": ["Vendor", "Vendor Inc"],
            "type": ["ORGANIZATION", "COMPANY"],
            "description": ["A supplier.", "Ships parts."],
        },
        _text_unit(),
    )
    assert entity is not None
    assert (entity.name, entity.type) == ("Vendor", "ORGANIZATION")
    assert entity.description == "A supplier.\nShips parts."


@pytest.mark.parametrize(
    ("strength", "expected"),
    [
        ("7", 7.0),
        (None, 1.0),
        ("high", 1.0),
        ("nan", 1.0),
        (float("nan"), 1.0),
        ("inf", 1.0),
        ("-inf", 1.0),
        (-3, 0.0),
        (0, 0.0),
        ("1e300", 1e9),
        (1e155, 1e9),
        ("1e-320", 0.0),
        (1e-12, 0.0),
        (1e-9, 1e-9),
        (0.5, 0.5),
    ],
)
def test_parse_weight_is_finite_and_in_range(strength, expected: float) -> None:
    # Leiden panics (an uncatchable BaseException) on a NaN, denormal or huge
    # weight and rejects infinite ones, so a bad strength from the model must
    # not reach the graph.
    assert BaseProcessor._parse_weight({"strength": strength}) == expected


def test_empty_elements_parse_to_empty_text() -> None:
    # The XML parser turns an empty element into {}, which str() made the
    # literal "{}": an entity described "{}", a claim about an object "{}",
    # a report titled "{}".
    entity = RobustXMLOutputParser(tags=["entities"]).parse(
        "<entities><entity><name>Acme</name><type></type>"
        "<description></description></entity></entities>"
    )["entities"]["entity"]
    parsed = BaseProcessor(config=Config()).parse_entity_data(entity, _text_unit())
    assert parsed is not None
    assert (parsed.type, parsed.description) == ("", "")

    claim = RobustXMLOutputParser(tags=["claims"]).parse(
        "<claims><claim><subject>Acme</subject><object></object>"
        "<description/></claim></claims>"
    )["claims"]["claim"]
    assert coerce_llm_text(claim["object"]) == ""
    assert coerce_llm_text(claim["description"], join=True) == ""

    report = RobustXMLOutputParser(tags=["community_name", "summary"]).parse(
        "<community_name></community_name><summary>S</summary>"
    )
    assert coerce_llm_text(report["community_name"]) == ""


def test_markup_in_a_field_is_read_as_its_text() -> None:
    # A field holding markup parses to a dict: the entity and claim parsers
    # took no dict (the description was "{'p': 'one'}") and the claim and
    # report readers each had their own copy of the rule.
    parser = RobustXMLOutputParser(tags=["entities"])
    entity = parser.parse(
        '<entities><entity><name lang="en">Beta</name><type>ORG</type>'
        "<description><p>one</p></description></entity></entities>"
    )["entities"]["entity"]
    parsed = BaseProcessor(config=Config()).parse_entity_data(entity, _text_unit())
    assert parsed is not None
    assert (parsed.name, parsed.description) == ("Beta", "one")

    claim = RobustXMLOutputParser(tags=["claims"]).parse(
        '<claims><claim><subject lang="en">Beta</subject><object>Acme</object>'
        "<claim_type>OWNERSHIP</claim_type>"
        "<description><p>Beta owns Acme.</p></description></claim></claims>"
    )["claims"]["claim"]
    extractor = ClaimExtractor.__new__(ClaimExtractor)
    made = ClaimExtractor._parse_claim_data(extractor, claim, _text_unit())
    assert made is not None
    assert (made.subject_name, made.description) == ("Beta", "Beta owns Acme.")


def test_a_finding_with_a_repeated_summary_keeps_both() -> None:
    report = RobustXMLOutputParser(tags=["findings"]).parse(
        "<findings><finding><summary>First.</summary><summary>Second.</summary>"
        "<explanation><p>Why.</p></explanation></finding></findings>"
    )
    (finding,) = CommunityDetector._extract_findings(report)
    # It was the list's repr: "['First.', 'Second.']".
    assert (finding.summary, finding.explanation) == ("First.\nSecond.", "Why.")
