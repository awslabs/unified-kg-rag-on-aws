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

import pytest

import unified_kg_rag.adapters.ingestion.graph_extractor as ge_module
from unified_kg_rag.adapters.ingestion.graph_extractor import (
    ExtractionStats,
    GraphExtractor,
)
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.models import Config, Entity, Relationship, TextUnit

pytestmark = pytest.mark.unit


@pytest.fixture
def text_unit() -> TextUnit:
    return TextUnit(id="t1", text="Alice works at Acme Corp.")


@pytest.fixture
def processor(config: Config) -> BaseProcessor:
    return BaseProcessor(config)


@pytest.fixture
def extractor(config: Config, mocker) -> GraphExtractor:
    """A real GraphExtractor with all AWS/Bedrock wiring stubbed out."""
    mocker.patch.object(ge_module, "boto3")
    mocker.patch.object(ge_module, "BedrockLanguageModelFactory")
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
        assert ent.name == "acme corp"  # normalized (casefolded)
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
        assert rel.source_name == "alice"
        assert rel.target_name == "acme"
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
    def test_duplicate_ids_merged_frequency_from_text_units(self, extractor) -> None:
        e1 = Entity(
            id="e1", name="Alice", type="PERSON", text_unit_ids=["t1"], confidence=0.5
        )
        e2 = Entity(
            id="e1", name="Alice", type="PERSON", text_unit_ids=["t2"], confidence=0.9
        )
        merged = extractor._merge_entities([e1, e2])
        assert len(merged) == 1
        survivor = merged[0]
        # frequency = count of unique text_unit_ids
        assert survivor.frequency == 2
        assert set(survivor.text_unit_ids) == {"t1", "t2"}

    def test_confidence_is_max_not_mean(self, extractor) -> None:
        e1 = Entity(
            id="e1", name="Alice", type="PERSON", text_unit_ids=["t1"], confidence=0.3
        )
        e2 = Entity(
            id="e1", name="Alice", type="PERSON", text_unit_ids=["t2"], confidence=0.9
        )
        merged = extractor._merge_entities([e1, e2])
        assert merged[0].confidence == 0.9  # max, not 0.6 mean

    def test_distinct_ids_not_merged(self, extractor) -> None:
        e1 = Entity(id="e1", name="Alice", type="PERSON", text_unit_ids=["t1"])
        e2 = Entity(id="e2", name="Bob", type="PERSON", text_unit_ids=["t1"])
        merged = extractor._merge_entities([e1, e2])
        assert len(merged) == 2

    def test_frequency_set_from_text_units_single(self, extractor) -> None:
        e1 = Entity(
            id="e1", name="Alice", type="PERSON", text_unit_ids=["t1", "t2", "t3"]
        )
        merged = extractor._merge_entities([e1])
        assert merged[0].frequency == 3

    def test_empty_input(self, extractor) -> None:
        assert extractor._merge_entities([]) == []


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
        merged = extractor._merge_relationships([r1, r2])
        assert len(merged) == 1
        assert merged[0].weight == 4.0
        assert set(merged[0].text_unit_ids) == {"t1", "t2"}

    def test_distinct_relationship_ids_not_merged(self, extractor) -> None:
        r1 = Relationship(id="r1", source_id="e1", target_id="e2")
        r2 = Relationship(id="r2", source_id="e2", target_id="e3")
        assert len(extractor._merge_relationships([r1, r2])) == 2


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
        out = extractor._materialize_relationship_endpoints(existing, [rel])
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
        out = extractor._materialize_relationship_endpoints(existing, [rel])
        assert len(out) == 2

    def test_endpoint_without_id_or_name_skipped(self, extractor) -> None:
        existing = [Entity(id="e1", name="Alice")]
        rel = Relationship(
            id="r1", source_id="e1", target_id="", source_name="Alice", target_name=""
        )
        out = extractor._materialize_relationship_endpoints(existing, [rel])
        assert len(out) == 1

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
        out = extractor._materialize_relationship_endpoints(existing, rels)
        # e2 (Bob) appears in both rels but is materialized once.
        assert len([e for e in out if e.id == "e2"]) == 1
        assert {e.id for e in out} == {"e2", "e3", "e4"}


# --------------------------------------------------------------------------- #
# _filter_entities_by_confidence / _filter_orphan_relationships
# --------------------------------------------------------------------------- #
class TestConfidenceFiltering:
    def test_threshold_zero_disables_filtering(self, extractor) -> None:
        extractor.extraction_config.entity_confidence_threshold = 0.0
        ents = [Entity(id="e1", name="A", confidence=0.1)]
        out, removed = extractor._filter_entities_by_confidence(ents)
        assert removed == 0
        assert out == ents

    def test_low_confidence_filtered(self, extractor) -> None:
        extractor.extraction_config.entity_confidence_threshold = 0.5
        ents = [
            Entity(id="e1", name="A", confidence=0.9),
            Entity(id="e2", name="B", confidence=0.1),
        ]
        out, removed = extractor._filter_entities_by_confidence(ents)
        assert removed == 1
        assert {e.id for e in out} == {"e1"}

    def test_none_confidence_treated_as_one(self, extractor) -> None:
        extractor.extraction_config.entity_confidence_threshold = 0.5
        ents = [Entity(id="e1", name="A", confidence=None)]
        out, removed = extractor._filter_entities_by_confidence(ents)
        assert removed == 0
        assert out == ents

    def test_orphan_relationship_filtered(self, extractor) -> None:
        valid = [Entity(id="e1", name="A"), Entity(id="e2", name="B")]
        rels = [
            Relationship(id="r1", source_id="e1", target_id="e2"),
            Relationship(id="r2", source_id="e1", target_id="GONE"),
        ]
        out, removed = extractor._filter_orphan_relationships(rels, valid)
        assert removed == 1
        assert {r.id for r in out} == {"r1"}


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
    def test_non_dict_returns_empty(self, extractor, text_unit) -> None:
        ents, rels = extractor._parse_extraction_result("nope", text_unit)
        assert ents == [] and rels == []

    def test_missing_keys_returns_empty(self, extractor, text_unit) -> None:
        ents, rels = extractor._parse_extraction_result({"entities": {}}, text_unit)
        assert ents == [] and rels == []

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
        ents, rels = extractor._parse_extraction_result(result, text_unit)
        assert {e.name for e in ents} == {"alice", "acme"}
        assert len(rels) == 1
        # Relationship endpoint id resolved through the local name->id map.
        alice_id = next(e.id for e in ents if e.name == "alice")
        assert rels[0].source_id == alice_id

    def test_invalid_entity_skipped(self, extractor, text_unit) -> None:
        result = {
            "entities": {"entity": [{"name": ""}, {"name": "Bob"}]},
            "relationships": {},
        }
        ents, rels = extractor._parse_extraction_result(result, text_unit)
        assert [e.name for e in ents] == ["bob"]


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


# --------------------------------------------------------------------------- #
# _process_extraction_results
# --------------------------------------------------------------------------- #
class TestProcessExtractionResults:
    def test_none_result_counts_failure(self, extractor) -> None:
        units = [TextUnit(id="t1", text="x")]
        ents, rels = extractor._process_extraction_results(units, [None])
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
        ents, _ = extractor._process_extraction_results(units, results)
        assert [e.name for e in ents] == ["alice"]
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
        ents, rels = extractor._process_extraction_results(units, results)
        # Alice (0.1) filtered out -> its relationship is now an orphan.
        assert "alice" not in {e.name for e in ents}
        assert extractor.stats.entities_filtered_by_confidence == 1
        assert extractor.stats.relationships_filtered_by_confidence == 1
        assert rels == []


# --------------------------------------------------------------------------- #
# extract_from_text_units (batch chain mocked)
# --------------------------------------------------------------------------- #
class TestExtractFromTextUnits:
    def test_empty_units_returns_empty(self, extractor) -> None:
        ents, rels, stats = extractor.extract_from_text_units([])
        assert ents == [] and rels == []
        assert stats.num_total_units == 0

    def test_end_to_end_with_mocked_chain(self, extractor, mocker) -> None:
        extractor.extraction_config.entity_confidence_threshold = 0.0
        units = [TextUnit(id="t1", text="Alice works at Acme.")]
        extractor.graph_extractor.batch = mocker.Mock(
            return_value=[
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
        ents, rels, stats = extractor.extract_from_text_units(units)
        assert {e.name for e in ents} == {"alice", "acme"}
        assert len(rels) == 1
        assert stats.total_entities_extracted == 2
        assert stats.total_relationships_extracted == 1
        assert stats.num_total_units == 1

    def test_batch_error_with_ignore_errors_returns_empty(
        self, extractor, mocker
    ) -> None:
        extractor.ignore_errors = True
        fake_bp = mocker.Mock()
        fake_bp.execute_with_fallback.side_effect = RuntimeError("boom")
        extractor.batch_processor = fake_bp
        ents, rels, stats = extractor.extract_from_text_units(
            [TextUnit(id="t1", text="x")]
        )
        assert ents == [] and rels == []

    def test_batch_error_without_ignore_raises(self, extractor, mocker) -> None:
        extractor.ignore_errors = False
        fake_bp = mocker.Mock()
        fake_bp.execute_with_fallback.side_effect = RuntimeError("boom")
        extractor.batch_processor = fake_bp
        with pytest.raises(RuntimeError, match="boom"):
            extractor.extract_from_text_units([TextUnit(id="t1", text="x")])


# --------------------------------------------------------------------------- #
# _log_completion_summary (smoke — exercises the logging branches)
# --------------------------------------------------------------------------- #
class TestLogCompletionSummary:
    def test_logs_with_filtering_and_failures(self) -> None:
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
        # Should not raise; covers both the confidence and failure log branches.
        GraphExtractor._log_completion_summary(stats)


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
        entities, _ = extractor._parse_extraction_result(self._result(), text_unit)
        assert {e.name for e in entities} == {"acme corp", "24 months"}
        assert extractor.stats.entities_ungrounded == 0

    def test_drop_removes_ungrounded(self, extractor, text_unit) -> None:
        extractor.extraction_config.entity_grounding.enabled = True
        extractor.extraction_config.entity_grounding.action = "drop"
        entities, _ = extractor._parse_extraction_result(self._result(), text_unit)
        names = {e.name for e in entities}
        assert "acme corp" in names
        assert "24 months" not in names  # hallucination dropped
        assert extractor.stats.entities_ungrounded == 1

    def test_penalize_keeps_but_lowers_confidence(self, extractor, text_unit) -> None:
        g = extractor.extraction_config.entity_grounding
        g.enabled = True
        g.action = "penalize"
        g.penalty_factor = 0.5
        entities, _ = extractor._parse_extraction_result(self._result(), text_unit)
        by_name = {e.name: e for e in entities}
        assert "24 months" in by_name  # kept
        assert by_name["24 months"].confidence == pytest.approx(0.5)  # penalized
        assert by_name["acme corp"].confidence == pytest.approx(1.0)  # untouched
        assert extractor.stats.entities_ungrounded == 1

    def test_source_text_not_persisted_as_attribute(self, extractor, text_unit) -> None:
        extractor.extraction_config.entity_grounding.enabled = True
        entities, _ = extractor._parse_extraction_result(self._result(), text_unit)
        for e in entities:
            assert "_source_text" not in (e.attributes or {})
