# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GraphGleaner pure logic (AWS-free).

Gleaning edits one text unit's record (``TextUnitExtraction``): additions,
corrections and the record's own dedupe; plus gleaner grounding and the
module-level format_entities_with_limit_task. Static methods are called
directly; instance methods run on a real GraphGleaner whose Bedrock/boto
wiring is patched out.
"""

from __future__ import annotations

import pytest

import unified_kg_rag.adapters.ingestion.gleaner as gleaner_module
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.ingestion.gleaner import (
    GraphGleaner,
    _dedupe_relationships,
    format_entities_with_limit_task,
)
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)

pytestmark = pytest.mark.unit

UNIT = TextUnit(id="t1", text="x")


@pytest.fixture
def gleaner(config: Config, mocker) -> GraphGleaner:
    mocker.patch.object(bedrock_module, "BedrockLanguageModelFactory")
    mocker.patch.object(gleaner_module, "create_robust_xml_output_parser")
    mocker.patch.object(gleaner_module, "setup_chain")
    return GraphGleaner(config)


def _rel(id_, src, tgt, **kw) -> Relationship:
    return Relationship(id=id_, source_id=src, target_id=tgt, type="X", **kw)


def _entity(name: str, **kw) -> Entity:
    entity_id = BaseProcessor._generate_entity_id(name)
    return Entity(id=entity_id, name=name, text_unit_ids=["t1"], **kw)


def _edge(source: str, target: str, type_: str = "SHIPS_TO", **kw) -> Relationship:
    return Relationship(
        id=BaseProcessor._generate_relationship_id(source, target, type_),
        source_id=BaseProcessor._generate_entity_id(source),
        target_id=BaseProcessor._generate_entity_id(target),
        source_name=source,
        target_name=target,
        type=type_,
        text_unit_ids=["t1"],
        **kw,
    )


def _issue(issue_type: str, **details) -> dict:
    return {"issue_type": issue_type, "details": details}


def _answer(gleaner: GraphGleaner, record: TextUnitExtraction, *issues) -> bool:
    """Apply one refinement answer made of ``issues`` to ``record``."""
    return gleaner._apply_answer(
        {"identified_issues": {"issue": list(issues)}}, UNIT, record
    )


# --------------------------------------------------------------------------- #
# A unit's relationships: one per id, no self-loops
# --------------------------------------------------------------------------- #
class TestDedupeRelationships:
    def test_self_loop_dropped(self) -> None:
        assert _dedupe_relationships([_rel("r1", "e1", "e1")]) == []

    def test_restated_edge_keeps_its_strength_and_gains_the_text(self) -> None:
        # The unit already states the edge: an answer restating it is not a
        # second statement, so the strength stays and the description joins.
        stored = _rel("r1", "e1", "e2", weight=0.8, description="a")
        again = _rel("r1", "e1", "e2", weight=1.0, description="b")
        (out,) = _dedupe_relationships([stored, again])
        assert out.weight == 0.8
        assert out.description == "a\nb"

    def test_document_filters_are_unioned(self) -> None:
        stored = _rel("r1", "e1", "e2", attributes={"filters": {"k": "x"}})
        again = _rel("r1", "e1", "e2", attributes={"filters": {"k": "y"}})
        (out,) = _dedupe_relationships([stored, again])
        assert out.attributes == {"filters": {"k": ["x", "y"]}}

    def test_distinct_ids_kept(self) -> None:
        assert (
            len(_dedupe_relationships([_rel("r1", "a", "b"), _rel("r2", "a", "b")]))
            == 2
        )


class TestAddToRecord:
    def test_entity_the_unit_lists_keeps_its_type_and_confidence(self) -> None:
        record = TextUnitExtraction(
            text_unit_id="t1",
            entities=[_entity("Alice", type="PERSON", confidence=0.6, description="d")],
        )
        GraphGleaner._add_to_record(
            record, [_entity("ALICE", type="ORG", confidence=1.0, description="e")], []
        )
        (alice,) = record.entities
        assert (alice.name, alice.type, alice.confidence) == ("Alice", "PERSON", 0.6)
        assert alice.description == "d\ne"

    def test_new_entity_and_edge_are_added(self) -> None:
        record = TextUnitExtraction(text_unit_id="t1", entities=[_entity("Alice")])
        GraphGleaner._add_to_record(
            record, [_entity("Acme")], [_edge("Alice", "Acme", "WORKS_AT")]
        )
        assert [e.name for e in record.entities] == ["Alice", "Acme"]
        assert [r.type for r in record.relationships] == ["WORKS_AT"]


class TestCorrectionDescriptions:
    """The prompt lists names only, so a correction's description is merged."""

    def test_entity_correction_description_is_appended(self, gleaner) -> None:
        record = TextUnitExtraction(
            text_unit_id="t1", entities=[_entity("Acme", description="Supplier.")]
        )
        issue = _issue("ENTITY_CORRECTION", name="Acme", description="In Springfield.")
        gleaner._process_issue(issue, UNIT, record, [], [], [])
        assert record.entities[0].description == "Supplier.\nIn Springfield."

    def test_relationship_correction_description_is_appended(self, gleaner) -> None:
        record = TextUnitExtraction(
            text_unit_id="t1",
            relationships=[_edge("Acme", "Depot", description="Ships weekly.")],
        )
        issue = _issue(
            "RELATIONSHIP_CORRECTION",
            source="Acme",
            target="Depot",
            description="By rail.",
        )
        gleaner._process_issue(issue, UNIT, record, [], [], [])
        assert record.relationships[0].description == "Ships weekly.\nBy rail."


class TestEntityCorrection:
    def test_rename_takes_the_new_names_id_and_moves_the_units_edges(
        self, gleaner
    ) -> None:
        record = TextUnitExtraction(
            text_unit_id="t1",
            entities=[_entity("Acme", type="PERSON"), _entity("Depot")],
            relationships=[_edge("Acme", "Depot")],
        )
        rename = _issue("ENTITY_CORRECTION", name="Acme", corrected_name="Acme Corp")
        retype = _issue(
            "ENTITY_CORRECTION", name="Acme Corp", corrected_type="ORGANIZATION"
        )
        _answer(gleaner, record, rename, retype)
        acme = record.entities[0]
        assert (acme.id, acme.name, acme.type) == (
            BaseProcessor._generate_entity_id("Acme Corp"),
            "Acme Corp",
            "ORGANIZATION",
        )
        (edge,) = record.relationships
        assert edge.id == _edge("Acme Corp", "Depot").id
        assert (edge.source_id, edge.source_name) == (acme.id, "Acme Corp")

    def test_rename_onto_another_entity_of_the_unit_merges_them(self, gleaner) -> None:
        record = TextUnitExtraction(
            text_unit_id="t1",
            entities=[_entity("Acme"), _entity("Acme Corp"), _entity("Depot")],
            relationships=[_edge("Acme", "Acme Corp", "OWNS"), _edge("Acme", "Depot")],
        )
        issue = _issue("ENTITY_CORRECTION", name="Acme", corrected_name="Acme Corp")
        _answer(gleaner, record, issue)
        assert sorted(e.name for e in record.entities) == ["Acme Corp", "Depot"]
        # The edge between the two became a self-loop and is gone.
        assert [(r.source_name, r.target_name) for r in record.relationships] == [
            ("Acme Corp", "Depot")
        ]

    def test_an_endpoint_named_only_in_an_edge_can_be_renamed(self, gleaner) -> None:
        # Depot is no entity of the record (a stub the prompt lists): the
        # rename moves the edge, and its type is not the unit's to correct.
        record = TextUnitExtraction(
            text_unit_id="t1",
            entities=[_entity("Acme")],
            relationships=[_edge("Acme", "Depot")],
        )
        issue = _issue(
            "ENTITY_CORRECTION",
            name="Depot",
            corrected_name="Depot Ltd",
            corrected_type="ORG",
        )
        _answer(gleaner, record, issue)
        assert [e.name for e in record.entities] == ["Acme"]
        assert record.relationships[0].target_name == "Depot Ltd"

    def test_an_entity_the_unit_does_not_name_is_not_corrected(self, gleaner) -> None:
        record = TextUnitExtraction(text_unit_id="t1", entities=[_entity("Acme")])
        issue = _issue("ENTITY_CORRECTION", name="Depot", corrected_name="Depot Ltd")
        assert not gleaner._apply_entity_correction(issue["details"], record, UNIT)
        _answer(gleaner, record, issue)
        assert [e.name for e in record.entities] == ["Acme"]

    def test_issues_of_one_answer_name_entities_as_the_prompt_did(
        self, gleaner
    ) -> None:
        # The rename applies after the answer's other issues, which name the
        # entity as listed (or by its new name).
        record = TextUnitExtraction(
            text_unit_id="t1",
            entities=[_entity("Acme"), _entity("Depot")],
            relationships=[_edge("Acme", "Depot", "SUPPLIES")],
        )
        _answer(
            gleaner,
            record,
            _issue("ENTITY_CORRECTION", name="Acme", corrected_name="Acme Corp"),
            _issue(
                "RELATIONSHIP_CORRECTION",
                source="Acme",
                target="Depot",
                corrected_type="SHIPS_TO",
            ),
            _issue("ENTITY_CORRECTION", name="Acme Corp", corrected_type="ORG"),
        )
        assert [(e.name, e.type) for e in record.entities] == [
            ("Acme Corp", "ORG"),
            ("Depot", None),
        ]
        assert [(r.source_name, r.type) for r in record.relationships] == [
            ("Acme Corp", "SHIPS_TO")
        ]


class TestRelationshipCorrection:
    def test_reversal_takes_the_reversed_edges_id(self, gleaner) -> None:
        record = TextUnitExtraction(
            text_unit_id="t1", relationships=[_edge("Acme", "Depot", weight=3.0)]
        )
        details = {
            "source": "Acme",
            "target": "Depot",
            "corrected_source": "Depot",
            "corrected_target": "Acme",
        }
        assert gleaner._apply_relationship_correction(details, record, UNIT)
        (edge,) = record.relationships
        assert edge.id == _edge("Depot", "Acme").id
        assert (edge.source_name, edge.target_name, edge.weight) == (
            "Depot",
            "Acme",
            3.0,
        )

    def test_retype_onto_a_stated_edge_merges_into_it(self, gleaner) -> None:
        record = TextUnitExtraction(
            text_unit_id="t1",
            relationships=[
                _edge("Acme", "Depot", "SUPPLIES", weight=8.0),
                _edge("Acme", "Depot", "TRADES_WITH", weight=5.0),
            ],
        )
        details = {
            "source": "Acme",
            "target": "Depot",
            "type": "TRADES_WITH",
            "corrected_type": "SUPPLIES",
        }
        assert gleaner._apply_relationship_correction(details, record, UNIT)
        (edge,) = record.relationships
        assert (edge.type, edge.weight) == ("SUPPLIES", 8.0)

    def test_ambiguous_correction_is_dropped(self, gleaner) -> None:
        record = TextUnitExtraction(
            text_unit_id="t1",
            relationships=[
                _edge("Acme", "Depot", "SUPPLIES"),
                _edge("Acme", "Depot", "TRADES_WITH"),
            ],
        )
        details = {"source": "Acme", "target": "Depot", "corrected_type": "OWNS"}
        assert not gleaner._apply_relationship_correction(details, record, UNIT)


# --------------------------------------------------------------------------- #
# format_entities_with_limit_task
# --------------------------------------------------------------------------- #
class TestFormatEntitiesWithLimit:
    def test_under_limit_lists_all_names(self) -> None:
        ents = [Entity(id="e1", name="Alice"), Entity(id="e2", name="Bob")]
        out = format_entities_with_limit_task(ents, max_entities=5)
        assert out == "Alice\nBob"

    def test_over_limit_truncates_with_suffix(self) -> None:
        ents = [Entity(id=f"e{i}", name=f"n{i}") for i in range(5)]
        out = format_entities_with_limit_task(ents, max_entities=2)
        assert "... and 3 more entities" in out
        assert len(out.splitlines()) == 3  # 2 names + suffix line

    def test_prioritizes_described_and_high_support_entities(self) -> None:
        # An entity with a description and more text_unit_ids should be picked
        # over a bare one when truncating.
        rich = Entity(id="e1", name="Rich", description="d", text_unit_ids=["t1", "t2"])
        bare = Entity(id="e2", name="Bare")
        out = format_entities_with_limit_task([bare, rich], max_entities=1)
        assert out.splitlines()[0] == "Rich"


# --------------------------------------------------------------------------- #
# Gleaner provenance grounding (hallucination guard on gleaner-added items)
# --------------------------------------------------------------------------- #


class TestGleanerGrounding:
    UNIT = TextUnit(id="u1", text="Alice works at Acme Corp in Seattle.")

    def _missing_entity_issue(self, evidence: str) -> dict:
        return {
            "issue_type": "MISSING_ENTITY",
            "details": {"name": "Acme Corp", "type": "ORG", "description": "A company"},
            "text_evidence": evidence,
        }

    def test_disabled_keeps_addition(self, gleaner) -> None:
        gleaner.extraction_config.entity_grounding.enabled = False
        new_e, new_r = [], []
        # ungrounded evidence, but gate off -> kept
        gleaner._process_issue(
            self._missing_entity_issue("A fabricated clause never in the text."),
            self.UNIT,
            TextUnitExtraction(text_unit_id="u1"),
            new_e,
            new_r,
            [],
        )
        assert len(new_e) == 1

    def test_grounded_addition_kept(self, gleaner) -> None:
        gleaner.extraction_config.entity_grounding.enabled = True
        new_e, new_r = [], []
        gleaner._process_issue(
            self._missing_entity_issue("Alice works at Acme Corp in Seattle."),
            self.UNIT,
            TextUnitExtraction(text_unit_id="u1"),
            new_e,
            new_r,
            [],
        )
        assert len(new_e) == 1
        assert new_e[0].name == "Acme Corp"

    def test_ungrounded_addition_dropped(self, gleaner) -> None:
        gleaner.extraction_config.entity_grounding.enabled = True
        new_e, new_r = [], []
        gleaner._process_issue(
            self._missing_entity_issue(
                "The Warranty Period from the Provisional Acceptance Date."
            ),
            self.UNIT,
            TextUnitExtraction(text_unit_id="u1"),
            new_e,
            new_r,
            [],
        )
        assert new_e == []  # hallucinated gleaner entity rejected

    def test_source_text_not_persisted_on_kept_entity(self, gleaner) -> None:
        gleaner.extraction_config.entity_grounding.enabled = True
        new_e, new_r = [], []
        gleaner._process_issue(
            self._missing_entity_issue("Alice works at Acme Corp in Seattle."),
            self.UNIT,
            TextUnitExtraction(text_unit_id="u1"),
            new_e,
            new_r,
            [],
        )
        assert "_source_text" not in (new_e[0].attributes or {})
