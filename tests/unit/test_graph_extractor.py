# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GraphExtractor pure parse/merge/materialize logic (AWS-free).

The parse_entity_data / parse_relationship_data / _parse_confidence helpers live
on the (domain) BaseProcessor and need no boto, so they are exercised directly.
The GraphExtractor merge / materialize / confidence-filter methods are exercised
on a real instance whose Bedrock/boto wiring is patched out so __init__ never
touches AWS.
"""

from __future__ import annotations

import logging

import pytest

import unified_kg_rag.adapters.ingestion.graph_extractor as ge_module
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.ingestion.graph_extractor import (
    ExtractionStats,
    GraphExtractor,
)
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.contributions import (
    merge_contributions,
    merge_entity_contributions,
    merge_relationship_contributions,
)
from unified_kg_rag.domain.ingestion.relationship_endpoints import (
    cite_relationship_endpoints,
)
from unified_kg_rag.domain.models import Config, Entity, Relationship, TextUnit
from unified_kg_rag.domain.models.config import LanguageCode
from unified_kg_rag.domain.prompts.graph_extraction import (
    GraphExtractionPrompt,
    GraphRefinementPrompt,
)
from unified_kg_rag.shared import DataProcessingError
from unified_kg_rag.shared.utils.langchain import BATCH_ITEM_FAILED

pytestmark = pytest.mark.unit


def _parsed(extractor, result, unit):
    """The unit's record of ``result`` as (entities, relationships)."""
    record = extractor._parse_extraction_result(result, unit)
    return record.entities, record.relationships


def _processed(extractor, units, results):
    """The graph the units' records state (merged, endpoints cited)."""
    return merge_contributions(extractor._process_extraction_results(units, results))


def _extracted(extractor, units):
    records, stats = extractor.extract_units(units)
    entities, relationships = merge_contributions(records)
    return entities, relationships, stats


@pytest.fixture
def text_unit() -> TextUnit:
    return TextUnit(id="t1", text="Alice works at Acme Corp.")


@pytest.fixture
def processor(config: Config) -> BaseProcessor:
    return BaseProcessor(config)


@pytest.fixture
def extractor(config: Config, mocker) -> GraphExtractor:
    """A real GraphExtractor with all AWS/Bedrock wiring stubbed out."""
    mocker.patch.object(bedrock_module, "BedrockLanguageModelFactory")
    mocker.patch.object(ge_module, "create_robust_xml_output_parser")
    mocker.patch.object(ge_module, "setup_chain")
    return GraphExtractor(config)


# --------------------------------------------------------------------------- #
# parse_entity_data
# --------------------------------------------------------------------------- #
class TestParseEntityData:
    def test_valid_entity_parsed(self, processor, text_unit) -> None:
        ent = processor.parse_entity_data(
            {"name": "Acme Corp", "type": "ORG", "description": "A company"},
            text_unit,
        )
        assert ent is not None
        assert ent.name == "Acme Corp"  # display form preserved
        assert ent.type == "ORG"
        assert ent.text_unit_ids == ["t1"]
        assert ent.id  # stable id derived

    def test_missing_name_returns_none(self, processor, text_unit) -> None:
        assert processor.parse_entity_data({"type": "ORG"}, text_unit) is None

    def test_whitespace_only_name_returns_none(self, processor, text_unit) -> None:
        assert processor.parse_entity_data({"name": "   "}, text_unit) is None

    def test_punctuation_only_name_survives_via_fallback(
        self, processor, text_unit
    ) -> None:
        # normalize_name falls back to casefolded original rather than "".
        ent = processor.parse_entity_data({"name": "!!!"}, text_unit)
        assert ent is not None
        assert ent.name == "!!!"

    def test_same_name_yields_stable_id(self, processor, text_unit) -> None:
        a = processor.parse_entity_data({"name": "Acme"}, text_unit)
        b = processor.parse_entity_data({"name": "acme"}, text_unit)
        assert a.id == b.id  # id is a hash of the normalized name


# --------------------------------------------------------------------------- #
# _parse_confidence scale heuristic
# --------------------------------------------------------------------------- #
class TestParseConfidence:
    def test_missing_uses_default(self) -> None:
        assert BaseProcessor._parse_confidence({}) == 1.0

    def test_explicit_default_override(self) -> None:
        assert BaseProcessor._parse_confidence({}, default=0.3) == 0.3

    def test_fractional_passthrough(self) -> None:
        assert BaseProcessor._parse_confidence({"confidence": 0.7}) == 0.7

    def test_ten_scale_rescaled(self) -> None:
        # > 1.0 is treated as a 0-10 scale and divided by 10.
        assert BaseProcessor._parse_confidence({"confidence": 8}) == 0.8

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (1, 0.1),  # lowest point of the prompt's 1-10 scale, not top
            ("1", 0.1),
            (" 1 ", 0.1),
            ("10", 1.0),
            ("8.5", 0.85),
            (1.0, 1.0),  # written as a fraction -> already normalized
            ("1.0", 1.0),
            ("0.4", 0.4),
            (0, 0.0),
        ],
    )
    def test_scale_rule(self, raw, expected) -> None:
        assert BaseProcessor._parse_confidence({"confidence": raw}) == pytest.approx(
            expected
        )

    def test_above_scale_clamped_to_one(self) -> None:
        assert BaseProcessor._parse_confidence({"confidence": 15}) == 1.0

    def test_invalid_value_uses_default(self) -> None:
        assert (
            BaseProcessor._parse_confidence({"confidence": "abc"}, default=0.5) == 0.5
        )

    def test_parse_entity_data_reads_confidence(self, processor, text_unit) -> None:
        ent = processor.parse_entity_data({"name": "Acme", "confidence": 9}, text_unit)
        assert ent.confidence == 0.9


# --------------------------------------------------------------------------- #
# parse_relationship_data
# --------------------------------------------------------------------------- #
class TestParseRelationshipData:
    def test_valid_relationship(self, processor, text_unit) -> None:
        rel = processor.parse_relationship_data(
            {"source": "Alice", "target": "Acme", "type": "WORKS_AT"},
            text_unit,
            entity_name_to_id={},
        )
        assert rel is not None
        assert rel.source_name == "Alice"
        assert rel.target_name == "Acme"
        assert rel.type == "WORKS_AT"
        # No local id map -> endpoint ids derived from normalized names.
        assert rel.source_id == BaseProcessor._generate_entity_id("alice")

    def test_uses_local_entity_id_when_available(self, processor, text_unit) -> None:
        rel = processor.parse_relationship_data(
            {"source": "Alice", "target": "Acme", "type": "X"},
            text_unit,
            entity_name_to_id={"alice": "ID-A", "acme": "ID-B"},
        )
        assert rel.source_id == "ID-A"
        assert rel.target_id == "ID-B"

    def test_missing_type_returns_none(self, processor, text_unit) -> None:
        assert (
            processor.parse_relationship_data(
                {"source": "Alice", "target": "Acme"}, text_unit, {}
            )
            is None
        )

    def test_missing_target_returns_none(self, processor, text_unit) -> None:
        assert (
            processor.parse_relationship_data(
                {"source": "Alice", "type": "X"}, text_unit, {}
            )
            is None
        )

    def test_weight_from_strength_field(self, processor, text_unit) -> None:
        rel = processor.parse_relationship_data(
            {"source": "A", "target": "B", "type": "X", "strength": "2.5"},
            text_unit,
            {},
        )
        assert rel.weight == 2.5

    def test_invalid_weight_falls_back_to_default(self, processor, text_unit) -> None:
        rel = processor.parse_relationship_data(
            {"source": "A", "target": "B", "type": "X", "weight": "heavy"},
            text_unit,
            {},
        )
        assert rel.weight == 1.0


# --------------------------------------------------------------------------- #
# _merge_entities
# --------------------------------------------------------------------------- #
class TestMergeEntities:
    def test_duplicate_ids_merged_text_units_unioned(self, extractor) -> None:
        e1 = Entity(
            id="e1", name="Alice", type="PERSON", text_unit_ids=["t1"], confidence=0.5
        )
        e2 = Entity(
            id="e1", name="Alice", type="PERSON", text_unit_ids=["t2"], confidence=0.9
        )
        merged = merge_entity_contributions([e1, e2])
        assert len(merged) == 1
        survivor = merged[0]
        assert set(survivor.text_unit_ids) == {"t1", "t2"}

    def test_confidence_is_max_not_mean(self, extractor) -> None:
        e1 = Entity(
            id="e1", name="Alice", type="PERSON", text_unit_ids=["t1"], confidence=0.3
        )
        e2 = Entity(
            id="e1", name="Alice", type="PERSON", text_unit_ids=["t2"], confidence=0.9
        )
        merged = merge_entity_contributions([e1, e2])
        assert merged[0].confidence == 0.9  # max, not 0.6 mean

    @pytest.mark.parametrize("reverse", [False, True])
    def test_document_filters_are_unioned(self, extractor, reverse) -> None:
        # The entity cites a research and a finance chunk: it must match a
        # filter on either, whichever instance the merge keeps.
        e1 = Entity(
            id="e1",
            name="Acme",
            text_unit_ids=["t1"],
            attributes={"filters": {"category": "research", "year": 2024}},
        )
        e2 = Entity(
            id="e1",
            name="Acme",
            text_unit_ids=["t2"],
            attributes={"filters": {"category": "finance", "year": 2024}},
        )
        entities = [e2, e1] if reverse else [e1, e2]
        (merged,) = merge_entity_contributions(entities)
        assert merged.attributes["filters"] == {
            "category": ["finance", "research"],
            "year": 2024,
        }

    def test_relationship_document_filters_are_unioned(self, extractor) -> None:
        r1 = Relationship(
            id="r1",
            source_id="a",
            target_id="b",
            text_unit_ids=["t1"],
            attributes={"filters": {"category": "research"}},
        )
        r2 = r1.model_copy(
            update={
                "text_unit_ids": ["t2"],
                "attributes": {"filters": {"category": "finance"}},
            }
        )
        (merged,) = merge_relationship_contributions([r1, r2])
        assert merged.attributes["filters"] == {"category": ["finance", "research"]}

    def test_distinct_ids_not_merged(self, extractor) -> None:
        e1 = Entity(id="e1", name="Alice", type="PERSON", text_unit_ids=["t1"])
        e2 = Entity(id="e2", name="Bob", type="PERSON", text_unit_ids=["t1"])
        merged = merge_entity_contributions([e1, e2])
        assert len(merged) == 2

    def test_empty_input(self, extractor) -> None:
        assert merge_entity_contributions([]) == []


# --------------------------------------------------------------------------- #
# _merge_relationships
# --------------------------------------------------------------------------- #
class TestMergeRelationships:
    def test_weights_summed(self, extractor) -> None:
        r1 = Relationship(
            id="r1", source_id="e1", target_id="e2", weight=1.0, text_unit_ids=["t1"]
        )
        r2 = Relationship(
            id="r1", source_id="e1", target_id="e2", weight=3.0, text_unit_ids=["t2"]
        )
        merged = merge_relationship_contributions([r1, r2])
        assert len(merged) == 1
        assert merged[0].weight == 4.0
        assert set(merged[0].text_unit_ids) == {"t1", "t2"}
        # Strength per text unit, kept for the incremental merge.
        assert merged[0].attributes == {"text_unit_weights": [1.0, 3.0]}

    def test_distinct_relationship_ids_not_merged(self, extractor) -> None:
        r1 = Relationship(id="r1", source_id="e1", target_id="e2")
        r2 = Relationship(id="r2", source_id="e2", target_id="e3")
        assert len(merge_relationship_contributions([r1, r2])) == 2


# --------------------------------------------------------------------------- #
# _materialize_relationship_endpoints
# --------------------------------------------------------------------------- #
class TestMaterializeRelationshipEndpoints:
    def test_missing_endpoint_materialized_as_stub(self, extractor) -> None:
        existing = [Entity(id="e1", name="Alice")]
        rel = Relationship(
            id="r1",
            source_id="e1",
            target_id="e2",
            source_name="Alice",
            target_name="Bob",
            text_unit_ids=["t9"],
        )
        out = cite_relationship_endpoints(existing, [rel])[0]
        ids = {e.id for e in out}
        assert ids == {"e1", "e2"}
        stub = next(e for e in out if e.id == "e2")
        assert stub.name == "Bob"
        assert stub.text_unit_ids == ["t9"]

    def test_existing_endpoint_not_duplicated(self, extractor) -> None:
        existing = [Entity(id="e1", name="Alice"), Entity(id="e2", name="Bob")]
        rel = Relationship(
            id="r1",
            source_id="e1",
            target_id="e2",
            source_name="Alice",
            target_name="Bob",
        )
        out = cite_relationship_endpoints(existing, [rel])[0]
        assert len(out) == 2

    def test_endpoint_without_id_or_name_skipped(self, extractor) -> None:
        existing = [Entity(id="e1", name="Alice")]
        rel = Relationship(
            id="r1", source_id="e1", target_id="", source_name="Alice", target_name=""
        )
        out = cite_relationship_endpoints(existing, [rel])[0]
        assert len(out) == 1

    def test_endpoints_cite_their_relationships_text_units(self, extractor) -> None:
        # An extracted endpoint also cites the chunks naming it only inside a
        # relationship; a stub cites every relationship it is an endpoint of.
        existing = [Entity(id="e1", name="Alice", text_unit_ids=["t1"])]
        rels = [
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                source_name="Alice",
                target_name="Bob",
                text_unit_ids=["t2"],
            ),
            Relationship(
                id="r2",
                source_id="e2",
                target_id="e1",
                source_name="Bob",
                target_name="Alice",
                text_unit_ids=["t1", "t3"],
            ),
        ]
        out = {e.id: e for e in cite_relationship_endpoints(existing, rels)[0]}
        assert out["e1"].text_unit_ids == ["t1", "t2", "t3"]
        assert out["e2"].text_unit_ids == ["t2", "t1", "t3"]
        # The input entity is not mutated.
        assert existing[0].text_unit_ids == ["t1"]

    def test_same_missing_endpoint_materialized_once(self, extractor) -> None:
        existing: list[Entity] = []
        rels = [
            Relationship(
                id="r1",
                source_id="e2",
                target_id="e3",
                source_name="Bob",
                target_name="Carol",
            ),
            Relationship(
                id="r2",
                source_id="e2",
                target_id="e4",
                source_name="Bob",
                target_name="Dan",
            ),
        ]
        out = cite_relationship_endpoints(existing, rels)[0]
        # e2 (Bob) appears in both rels but is materialized once.
        assert len([e for e in out if e.id == "e2"]) == 1
        assert {e.id for e in out} == {"e2", "e3", "e4"}


# --------------------------------------------------------------------------- #
# _apply_confidence_threshold
# --------------------------------------------------------------------------- #
class TestConfidenceFiltering:
    def test_threshold_zero_disables_filtering(self, extractor, text_unit) -> None:
        extractor.extraction_config.entity_confidence_threshold = 0.0
        ents = [Entity(id="e1", name="A", confidence=0.1)]
        out, dropped = extractor._apply_confidence_threshold(ents, text_unit)
        assert dropped == set()
        assert out == ents

    def test_low_confidence_filtered(self, extractor, text_unit) -> None:
        extractor.extraction_config.entity_confidence_threshold = 0.5
        ents = [
            Entity(id="e1", name="A", confidence=0.9),
            Entity(id="e2", name="B", confidence=0.1),
        ]
        out, dropped = extractor._apply_confidence_threshold(ents, text_unit)
        assert dropped == {"b"}
        assert {e.id for e in out} == {"e1"}
        assert extractor.stats.entities_filtered_by_confidence == 1

    def test_none_confidence_treated_as_one(self, extractor, text_unit) -> None:
        extractor.extraction_config.entity_confidence_threshold = 0.5
        ents = [Entity(id="e1", name="A", confidence=None)]
        out, dropped = extractor._apply_confidence_threshold(ents, text_unit)
        assert dropped == set()
        assert out == ents

    def test_kept_copy_in_the_chunk_lifts_the_key(self, extractor, text_unit) -> None:
        extractor.extraction_config.entity_confidence_threshold = 0.5
        ents = [
            Entity(id="e1", name="A", confidence=0.1),
            Entity(id="e1", name="A", confidence=0.9),
        ]
        out, dropped = extractor._apply_confidence_threshold(ents, text_unit)
        assert dropped == set()
        assert [e.confidence for e in out] == [0.9]


# --------------------------------------------------------------------------- #
# ExtractionStats derived properties
# --------------------------------------------------------------------------- #
class TestExtractionStats:
    def test_success_rate_zero_processed(self) -> None:
        assert ExtractionStats().success_rate == 0.0

    def test_average_processing_time_zero_processed(self) -> None:
        assert ExtractionStats().average_processing_time == 0.0

    def test_rates_computed(self) -> None:
        s = ExtractionStats(
            num_successful_extractions=3,
            num_failed_extractions=1,
            total_processing_time=8.0,
        )
        assert s.processed_unit_count == 4
        assert s.success_rate == 75.0
        assert s.average_processing_time == 2.0


# --------------------------------------------------------------------------- #
# _format_entity_types
# --------------------------------------------------------------------------- #
class TestFormatEntityTypes:
    def test_empty_list_uses_free_choice_message(self) -> None:
        out = GraphExtractor._format_entity_types([])
        assert "any entity types" in out.lower()

    def test_label_only(self) -> None:
        assert GraphExtractor._format_entity_types(["PERSON"]) == "- **PERSON**"

    def test_label_with_description(self) -> None:
        out = GraphExtractor._format_entity_types(["ORG: a company"])
        assert out == "- **ORG**: a company"

    def test_blank_label_skipped(self) -> None:
        out = GraphExtractor._format_entity_types([": no label", "PERSON"])
        assert out == "- **PERSON**"


# --------------------------------------------------------------------------- #
# _parse_extraction_result
# --------------------------------------------------------------------------- #
class TestParseExtractionResult:
    def test_non_dict_is_malformed(self, extractor, text_unit) -> None:
        with pytest.raises(DataProcessingError):
            _parsed(extractor, "nope", text_unit)

    @pytest.mark.parametrize(
        "result",
        [
            {"relationships": {}},
            {"entities": "Alice"},
            {"entities": {}, "relationships": "x"},
        ],
    )
    def test_missing_or_mistyped_section_is_malformed(
        self, extractor, text_unit, result
    ) -> None:
        with pytest.raises(DataProcessingError):
            _parsed(extractor, result, text_unit)

    @pytest.mark.parametrize(
        "result",
        [{"entities": {}}, {"entities": {}, "relationships": {}}],
    )
    def test_empty_entities_section_is_valid(
        self, extractor, text_unit, result
    ) -> None:
        ents, rels = _parsed(extractor, result, text_unit)
        assert ents == [] and rels == []

    def test_missing_relationships_keeps_entities(self, extractor, text_unit) -> None:
        # The XML parser drops an empty <relationships/> section.
        result = {"entities": {"entity": {"name": "Bob", "type": "PERSON"}}}
        ents, rels = _parsed(extractor, result, text_unit)
        assert [e.name for e in ents] == ["Bob"] and rels == []

    def test_entities_and_relationships_parsed(self, extractor, text_unit) -> None:
        result = {
            "entities": {
                "entity": [
                    {"name": "Alice", "type": "PERSON"},
                    {"name": "Acme", "type": "ORG"},
                ]
            },
            "relationships": {
                "relationship": {
                    "source": "Alice",
                    "target": "Acme",
                    "type": "WORKS_AT",
                }
            },
        }
        ents, rels = _parsed(extractor, result, text_unit)
        assert {e.name for e in ents} == {"Alice", "Acme"}
        assert len(rels) == 1
        # Relationship endpoint id resolved through the local name->id map.
        alice_id = next(e.id for e in ents if e.name == "Alice")
        assert rels[0].source_id == alice_id

    def test_invalid_entity_skipped(self, extractor, text_unit) -> None:
        result = {
            "entities": {"entity": [{"name": ""}, {"name": "Bob"}]},
            "relationships": {},
        }
        ents, rels = _parsed(extractor, result, text_unit)
        assert [e.name for e in ents] == ["Bob"]


# --------------------------------------------------------------------------- #
# _prepare_extraction_inputs
# --------------------------------------------------------------------------- #
class TestPrepareExtractionInputs:
    def test_inputs_built_for_each_unit(self, extractor) -> None:
        extractor.extraction_config.entity_types = ["PERSON", "ORG"]
        units = [TextUnit(id="t1", text="hello"), TextUnit(id="t2", text="world")]
        inputs = extractor._prepare_extraction_inputs(units)
        assert len(inputs) == 2
        assert inputs[0]["input_text"] == "hello"
        assert "PERSON" in inputs[0]["entity_types"]
        assert inputs[0]["max_entities_per_chunk"] == str(
            extractor.extraction_config.max_entities_per_chunk
        )

    def test_inputs_carry_target_language(self, extractor) -> None:
        # Descriptions are written in the graph's (target) language.
        extractor.config.processing.translation.target_language = LanguageCode.KO
        inputs = extractor._prepare_extraction_inputs([TextUnit(id="t1", text="x")])
        assert inputs[0]["target_language"] == "ko"


# --------------------------------------------------------------------------- #
# _process_extraction_results
# --------------------------------------------------------------------------- #
class TestProcessExtractionResults:
    def test_none_result_counts_failure(self, extractor) -> None:
        units = [TextUnit(id="t1", text="x")]
        ents, rels = _processed(extractor, units, [None])
        assert ents == [] and rels == []
        assert extractor.stats.num_failed_extractions == 1

    def test_successful_result_accumulates(self, extractor) -> None:
        units = [TextUnit(id="t1", text="x")]
        results = [
            {
                "entities": {
                    "entity": [{"name": "Alice", "type": "PERSON", "confidence": 0.9}]
                },
                "relationships": {},
            }
        ]
        extractor.extraction_config.entity_confidence_threshold = 0.0
        ents, _ = _processed(extractor, units, results)
        assert [e.name for e in ents] == ["Alice"]
        assert extractor.stats.num_successful_extractions == 1
        assert extractor.stats.average_entity_confidence == 0.9

    def test_confidence_filter_drops_orphan_relationship(self, extractor) -> None:
        extractor.extraction_config.entity_confidence_threshold = 0.5
        units = [TextUnit(id="t1", text="x")]
        results = [
            {
                "entities": {
                    "entity": [
                        {"name": "Alice", "type": "PERSON", "confidence": 0.1},
                        {"name": "Acme", "type": "ORG", "confidence": 0.9},
                    ]
                },
                "relationships": {
                    "relationship": {
                        "source": "Alice",
                        "target": "Acme",
                        "type": "WORKS_AT",
                    }
                },
            }
        ]
        ents, rels = _processed(extractor, units, results)
        # Alice (0.1) filtered out -> its relationship is now an orphan.
        assert "alice" not in {e.name for e in ents}
        assert extractor.stats.entities_filtered_by_confidence == 1
        assert extractor.stats.relationships_filtered_by_confidence == 1
        assert rels == []

    def test_confidence_filter_drops_the_stubs_it_orphans(self, extractor) -> None:
        # "Depot" is named only in the filtered entity's relationship: its stub
        # would be left with no description and no edge. "Harbor", named in a
        # surviving relationship, stays.
        extractor.extraction_config.entity_confidence_threshold = 0.5
        units = [TextUnit(id="t1", text="x")]
        results = [
            {
                "entities": {
                    "entity": [
                        {"name": "Alice", "type": "PERSON", "confidence": 0.1},
                        {"name": "Acme", "type": "ORG", "confidence": 0.9},
                    ]
                },
                "relationships": {
                    "relationship": [
                        {"source": "Alice", "target": "Depot", "type": "VISITS"},
                        {"source": "Acme", "target": "Harbor", "type": "SHIPS_TO"},
                    ]
                },
            }
        ]
        ents, rels = _processed(extractor, units, results)
        assert sorted(e.name for e in ents) == ["Acme", "Harbor"]
        assert [(r.source_name, r.target_name) for r in rels] == [("Acme", "Harbor")]

    @staticmethod
    def _output(acme_confidence: float) -> dict:
        return {
            "entities": {
                "entity": [
                    {"name": "Acme", "type": "ORG", "confidence": acme_confidence},
                    {"name": "Bob", "type": "PERSON", "confidence": 0.9},
                ]
            },
            "relationships": {
                "relationship": {"source": "Bob", "target": "Acme", "type": "WORKS_AT"}
            },
        }

    def _lineage(self, extractor, units, results) -> tuple[dict, list]:
        extractor.stats = ExtractionStats()
        ents, rels = _processed(extractor, units, results)
        return (
            {e.name: e.text_unit_ids for e in ents},
            [(r.source_name, r.target_name, r.text_unit_ids) for r in rels],
        )

    def test_confidence_threshold_does_not_depend_on_the_batch(self, extractor) -> None:
        # The threshold judges each chunk's own instance. Applied to the
        # merged entity (max confidence over the batch), "Acme" from the
        # low-confidence chunk B survived next to chunk A but not alone, so a
        # full build and an incremental run of B disagreed.
        extractor.extraction_config.entity_confidence_threshold = 0.5
        unit_a, unit_b = TextUnit(id="A", text="x"), TextUnit(id="B", text="x")
        together = self._lineage(
            extractor, [unit_a, unit_b], [self._output(0.9), self._output(0.2)]
        )
        alone_a = self._lineage(extractor, [unit_a], [self._output(0.9)])
        alone_b = self._lineage(extractor, [unit_b], [self._output(0.2)])
        assert together == (
            {"Acme": ["A"], "Bob": ["A", "B"]},
            [("Bob", "Acme", ["A"])],
        )
        assert alone_b == ({"Bob": ["B"]}, [])
        assert alone_a == ({"Acme": ["A"], "Bob": ["A"]}, [("Bob", "Acme", ["A"])])
        (record_b,) = extractor._process_extraction_results(
            [unit_b], [self._output(0.2)]
        )
        assert record_b.rejected == {"acme": "low_confidence"}

    def test_low_confidence_elsewhere_does_not_drop_a_stub(self, extractor) -> None:
        # "X" is listed at low confidence in A only; B names it only in a
        # relationship, so B's stub cites B whether or not A shares the batch.
        extractor.extraction_config.entity_confidence_threshold = 0.5
        unit_a, unit_b = TextUnit(id="A", text="x"), TextUnit(id="B", text="x")
        out_a = {
            "entities": {"entity": [{"name": "X", "type": "ORG", "confidence": 0.2}]}
        }
        out_b = {
            "entities": {
                "entity": [{"name": "Bob", "type": "PERSON", "confidence": 0.9}]
            },
            "relationships": {
                "relationship": {"source": "Bob", "target": "X", "type": "WORKS_AT"}
            },
        }
        together = self._lineage(extractor, [unit_a, unit_b], [out_a, out_b])
        alone_b = self._lineage(extractor, [unit_b], [out_b])
        assert (
            together
            == alone_b
            == (
                {"Bob": ["B"], "X": ["B"]},
                [("Bob", "X", ["B"])],
            )
        )


# --------------------------------------------------------------------------- #
# extract_from_text_units (batch chain mocked)
# --------------------------------------------------------------------------- #
class TestExtractFromTextUnits:
    def test_empty_units_returns_empty(self, extractor) -> None:
        ents, rels, stats = _extracted(extractor, [])
        assert ents == [] and rels == []
        assert stats.num_total_units == 0

    def test_end_to_end_with_mocked_chain(self, extractor, mocker) -> None:
        extractor.extraction_config.entity_confidence_threshold = 0.0
        units = [TextUnit(id="t1", text="Alice works at Acme.")]
        extractor.graph_extractor.invoke = mocker.Mock(
            side_effect=[
                {
                    "entities": {
                        "entity": [
                            {"name": "Alice", "type": "PERSON"},
                            {"name": "Acme", "type": "ORG"},
                        ]
                    },
                    "relationships": {
                        "relationship": {
                            "source": "Alice",
                            "target": "Acme",
                            "type": "WORKS_AT",
                        }
                    },
                }
            ]
        )
        ents, rels, stats = _extracted(extractor, units)
        assert {e.name for e in ents} == {"Alice", "Acme"}
        assert len(rels) == 1
        assert stats.total_entities_extracted == 2
        assert stats.total_relationships_extracted == 1
        assert stats.num_total_units == 1

    def test_failed_and_malformed_items_count_as_failures(
        self, extractor, mocker
    ) -> None:
        extractor.batch_processor = mocker.Mock()
        extractor.batch_processor.execute_with_fallback.return_value = [
            BATCH_ITEM_FAILED,
            {"unexpected": "shape"},
            {"entities": {}},
        ]
        units = [TextUnit(id=f"t{i}", text="x") for i in range(3)]
        _, _, stats = _extracted(extractor, units)
        assert stats.num_failed_extractions == 2
        assert stats.num_successful_extractions == 1
        assert stats.failed_text_unit_ids == ["t0", "t1"]

    def test_batch_error_with_ignore_errors_returns_empty(
        self, extractor, mocker
    ) -> None:
        extractor.ignore_errors = True
        fake_bp = mocker.Mock()
        fake_bp.execute_with_fallback.side_effect = RuntimeError("boom")
        extractor.batch_processor = fake_bp
        ents, rels, stats = _extracted(
            extractor, [TextUnit(id="t1", text="x"), TextUnit(id="t2", text="y")]
        )
        assert ents == [] and rels == []
        # Every unit failed; the stats must say so, not report zero failures.
        assert stats.num_total_units == 2
        assert stats.num_failed_extractions == 2
        assert stats.failed_text_unit_ids == ["t1", "t2"]

    def test_batch_error_without_ignore_raises(self, extractor, mocker) -> None:
        extractor.ignore_errors = False
        fake_bp = mocker.Mock()
        fake_bp.execute_with_fallback.side_effect = RuntimeError("boom")
        extractor.batch_processor = fake_bp
        with pytest.raises(RuntimeError, match="boom"):
            _extracted(extractor, [TextUnit(id="t1", text="x")])


# --------------------------------------------------------------------------- #
# _log_completion_summary
# --------------------------------------------------------------------------- #
class TestLogCompletionSummary:
    def test_logs_with_filtering_and_failures(self, caplog) -> None:
        stats = ExtractionStats(
            num_total_units=3,
            num_successful_extractions=2,
            num_failed_extractions=1,
            total_entities_extracted=5,
            total_relationships_extracted=4,
            entities_filtered_by_confidence=2,
            relationships_filtered_by_confidence=1,
            average_entity_confidence=0.7,
            confidence_threshold_applied=0.5,
            total_processing_time=1.23,
        )
        with caplog.at_level(logging.INFO):
            GraphExtractor._log_completion_summary(stats)
        assert "Entities filtered: 2, Relationships filtered: 1" in caplog.text
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "Failed to extract from 1 text units" in caplog.text


# --------------------------------------------------------------------------- #
# Entity grounding (hallucination guard) via _parse_extraction_result
# --------------------------------------------------------------------------- #
class TestEntityGrounding:
    """The chunk text comes from the `text_unit` fixture: 'Alice works at Acme Corp.'"""

    @staticmethod
    def _result() -> dict:
        # Two entities: one grounded (span in the chunk), one hallucinated
        # (span absent — the model invented it from domain priors).
        return {
            "entities": [
                {
                    "name": "Acme Corp",
                    "type": "ORG",
                    "description": "A company",
                    "source_text": "Alice works at Acme Corp.",
                },
                {
                    "name": "24 months",
                    "type": "TEMPORAL",
                    "description": "The warranty period",
                    "source_text": "The Warranty Period from the Provisional Acceptance Date.",
                },
            ],
            "relationships": [],
        }

    def test_disabled_keeps_all(self, extractor, text_unit) -> None:
        extractor.extraction_config.entity_grounding.enabled = False
        entities, _ = _parsed(extractor, self._result(), text_unit)
        assert {e.name for e in entities} == {"Acme Corp", "24 months"}
        assert extractor.stats.entities_ungrounded == 0

    def test_drop_removes_ungrounded(self, extractor, text_unit) -> None:
        extractor.extraction_config.entity_grounding.enabled = True
        extractor.extraction_config.entity_grounding.action = "drop"
        entities, _ = _parsed(extractor, self._result(), text_unit)
        names = {e.name for e in entities}
        assert "Acme Corp" in names
        assert "24 months" not in names  # hallucination dropped
        assert extractor.stats.entities_ungrounded == 1

    def test_penalize_keeps_but_lowers_confidence(self, extractor, text_unit) -> None:
        g = extractor.extraction_config.entity_grounding
        g.enabled = True
        g.action = "penalize"
        g.penalty_factor = 0.5
        entities, _ = _parsed(extractor, self._result(), text_unit)
        by_name = {e.name: e for e in entities}
        assert "24 months" in by_name  # kept
        assert by_name["24 months"].confidence == pytest.approx(0.5)  # penalized
        assert by_name["Acme Corp"].confidence == pytest.approx(1.0)  # untouched
        assert extractor.stats.entities_ungrounded == 1

    def test_source_text_not_persisted_as_attribute(self, extractor, text_unit) -> None:
        extractor.extraction_config.entity_grounding.enabled = True
        entities, _ = _parsed(extractor, self._result(), text_unit)
        for e in entities:
            assert "_source_text" not in (e.attributes or {})


# --------------------------------------------------------------------------- #
# Relationship grounding (hallucination guard)
# --------------------------------------------------------------------------- #
class TestRelationshipGrounding:
    """Chunk text from `text_unit` fixture: 'Alice works at Acme Corp.'"""

    @staticmethod
    def _result() -> dict:
        return {
            "entities": [
                {
                    "name": "Alice",
                    "type": "PERSON",
                    "source_text": "Alice works at Acme Corp.",
                },
                {
                    "name": "Acme Corp",
                    "type": "ORG",
                    "source_text": "Alice works at Acme Corp.",
                },
            ],
            "relationships": [
                {
                    "source": "Alice",
                    "target": "Acme Corp",
                    "type": "WORKS_AT",
                    "strength": 8,
                    "source_text": "Alice works at Acme Corp.",
                },
                {
                    "source": "Alice",
                    "target": "Acme Corp",
                    "type": "FOUNDED",
                    "strength": 9,
                    "source_text": "Alice founded Acme Corp in 1998 with venture funding.",
                },
            ],
        }

    def test_disabled_keeps_all(self, extractor, text_unit) -> None:
        extractor.extraction_config.entity_grounding.enabled = False
        _, rels = _parsed(extractor, self._result(), text_unit)
        assert {r.type for r in rels} == {"WORKS_AT", "FOUNDED"}
        assert extractor.stats.relationships_ungrounded == 0

    def test_drop_removes_ungrounded(self, extractor, text_unit) -> None:
        extractor.extraction_config.entity_grounding.enabled = True
        extractor.extraction_config.entity_grounding.action = "drop"
        _, rels = _parsed(extractor, self._result(), text_unit)
        types = {r.type for r in rels}
        assert "WORKS_AT" in types
        assert "FOUNDED" not in types  # hallucinated relationship dropped
        assert extractor.stats.relationships_ungrounded == 1

    def test_penalize_scales_weight(self, extractor, text_unit) -> None:
        g = extractor.extraction_config.entity_grounding
        g.enabled = True
        g.action = "penalize"
        g.penalty_factor = 0.5
        _, rels = _parsed(extractor, self._result(), text_unit)
        by_type = {r.type: r for r in rels}
        assert "FOUNDED" in by_type  # kept
        assert by_type["FOUNDED"].weight == pytest.approx(4.5)  # 9 * 0.5
        assert by_type["WORKS_AT"].weight == pytest.approx(8.0)  # untouched
        assert extractor.stats.relationships_ungrounded == 1

    def test_source_text_not_persisted(self, extractor, text_unit) -> None:
        extractor.extraction_config.entity_grounding.enabled = True
        _, rels = _parsed(extractor, self._result(), text_unit)
        for r in rels:
            assert "_source_text" not in (r.attributes or {})

    def test_penalize_drops_ungrounded_edge_naming_an_unlisted_entity(
        self, extractor, text_unit
    ) -> None:
        g = extractor.extraction_config.entity_grounding
        g.enabled = True
        g.action = "penalize"
        # t0 lists Acme Corp with a grounded span; t1 lists Alice and an
        # ungrounded edge to Acme Corp, which t1 does not list.
        acme_unit = TextUnit(id="t0", text="Acme Corp is a Vendor of parts.")
        acme = {
            "entities": [
                {
                    "name": "Acme Corp",
                    "type": "ORG",
                    "source_text": "Acme Corp is a Vendor of parts.",
                }
            ]
        }
        result = self._result()
        result["entities"] = result["entities"][:1]
        result["relationships"].append(
            {
                "source": "Alice",
                "target": "Alice's Garage",
                "type": "OWNS",
                "strength": 5,
                "source_text": "Alice owns a garage with three bays downtown.",
            }
        )
        entities, rels = _processed(extractor, [acme_unit, text_unit], [acme, result])
        by_name = {e.name: e for e in entities}
        # The grounded WORKS_AT edge still makes Acme Corp cite t1. The
        # ungrounded FOUNDED edge to it and OWNS (whose target no chunk lists)
        # are dropped rather than penalized.
        assert {r.type for r in rels} == {"WORKS_AT"}
        assert set(by_name["Acme Corp"].text_unit_ids) == {"t0", "t1"}
        assert "Alice's Garage" not in by_name
        assert extractor.stats.relationships_ungrounded == 2
        assert extractor.stats.relationships_dropped_ungrounded_unlisted_endpoint == 2

    def test_penalize_never_cites_a_chunk_through_an_ungrounded_edge(
        self, extractor, text_unit
    ) -> None:
        g = extractor.extraction_config.entity_grounding
        g.enabled = True
        g.action = "penalize"
        acme_unit = TextUnit(id="t0", text="Acme Corp is a Vendor of parts.")
        acme = {
            "entities": [
                {
                    "name": "Acme Corp",
                    "type": "ORG",
                    "source_text": "Acme Corp is a Vendor of parts.",
                }
            ]
        }
        result = self._result()
        result["entities"] = result["entities"][:1]
        result["relationships"] = result["relationships"][1:]  # FOUNDED only
        entities, rels = _processed(extractor, [acme_unit, text_unit], [acme, result])
        by_name = {e.name: e for e in entities}
        assert rels == []
        assert by_name["Acme Corp"].text_unit_ids == ["t0"]


# --------------------------------------------------------------------------- #
# A relationship must not resurrect an entity dropped by the grounding guard
# --------------------------------------------------------------------------- #
class TestDroppedEntityNotResurrected:
    """Chunk text from `text_unit` fixture: 'Alice works at Acme Corp.'"""

    @staticmethod
    def _result() -> dict:
        return {
            "entities": [
                {
                    "name": "Acme Corp",
                    "type": "ORG",
                    "source_text": "Alice works at Acme Corp.",
                },
                {
                    "name": "Service Term",
                    "type": "CONCEPT",
                    "source_text": "The Vendor grants a long Service Term to Buyer.",
                },
            ],
            "relationships": [
                {
                    # Grounded span, but its target is the dropped entity.
                    "source": "Acme Corp",
                    "target": "service term",
                    "type": "OFFERS",
                    "source_text": "Alice works at Acme Corp.",
                },
                {
                    "source": "Alice",
                    "target": "Acme Corp",
                    "type": "WORKS_AT",
                    "source_text": "Alice works at Acme Corp.",
                },
            ],
        }

    def test_drop_mode_removes_relationship_and_no_stub(
        self, extractor, text_unit
    ) -> None:
        g = extractor.extraction_config.entity_grounding
        g.enabled = True
        g.action = "drop"
        entities, rels = _processed(extractor, [text_unit], [self._result()])
        keys = {e.name.casefold() for e in entities}
        assert "service term" not in keys  # not recreated as a stub
        assert {r.type for r in rels} == {"WORKS_AT"}
        assert "alice" in keys  # a legitimate endpoint stub is still created
        assert extractor.stats.relationships_dropped_ungrounded_endpoint == 1

    def test_penalize_mode_keeps_relationship(self, extractor, text_unit) -> None:
        g = extractor.extraction_config.entity_grounding
        g.enabled = True
        g.action = "penalize"
        _, rels = _processed(extractor, [text_unit], [self._result()])
        assert {r.type for r in rels} == {"OFFERS", "WORKS_AT"}
        assert extractor.stats.relationships_dropped_ungrounded_endpoint == 0

    def test_grounded_duplicate_keeps_relationship(self, extractor, text_unit) -> None:
        g = extractor.extraction_config.entity_grounding
        g.enabled = True
        g.action = "drop"
        result = self._result()
        # The same entity also listed with a grounded span is kept, so its
        # relationships must be kept too.
        result["entities"].append(
            {
                "name": "Service Term",
                "type": "CONCEPT",
                "source_text": "Alice works at Acme Corp.",
            }
        )
        _, rels = _processed(extractor, [text_unit], [result])
        assert {r.type for r in rels} == {"OFFERS", "WORKS_AT"}
        assert extractor.stats.relationships_dropped_ungrounded_endpoint == 0


@pytest.mark.parametrize(
    "prompt", [GraphExtractionPrompt, GraphRefinementPrompt], ids=lambda p: p.__name__
)
def test_description_prompts_take_the_output_language(prompt) -> None:
    assert "target_language" in prompt.input_variables
    human = prompt.resolve().human_prompt_template
    assert "Write every <description> in the language with code" in human
    assert "'{target_language}'" in human
