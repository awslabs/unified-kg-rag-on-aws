# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gleaned relationships do not bring back entities grounding rejected.

Every kept gleaned relationship makes its endpoints cite its unit and adds a
stub for an endpoint the batch lacks. The grounding check on a relationship
covers only its quote, which need not name its endpoints, so without a guard a
relationship naming an entity rejected as ungrounded (by extraction, or in the
same gleaning answer) recreates it as a stub. Runs the real
``_apply_answer`` on the unit's record, then the merge of the records; only
the model is replaced.
"""

from __future__ import annotations

import pytest

import unified_kg_rag.adapters.ingestion.gleaner as gleaner_module
import unified_kg_rag.adapters.ingestion.graph_extractor as extractor_module
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.ingestion.gleaner import GraphGleaner
from unified_kg_rag.adapters.ingestion.graph_extractor import GraphExtractor
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.contributions import merge_contributions
from unified_kg_rag.domain.ingestion.entity_grounding import is_name_grounded
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)

pytestmark = pytest.mark.unit

TEXT = "Vendor ships the goods to Depot under the supply agreement."
UNIT = TextUnit(id="t1", text=TEXT)
GROUNDED_QUOTE = "Vendor ships the goods to Depot under the supply agreement."
FABRICATED_QUOTE = "Phantom Holdings guarantees every payment obligation of Vendor."
VENDOR = BaseProcessor._generate_entity_id("Vendor")
PHANTOM = BaseProcessor._generate_entity_id("Phantom Holdings")


@pytest.fixture
def gleaner(config: Config, mocker) -> GraphGleaner:
    mocker.patch.object(bedrock_module, "BedrockLanguageModelFactory")
    mocker.patch.object(gleaner_module, "create_robust_xml_output_parser")
    mocker.patch.object(gleaner_module, "setup_chain")
    return GraphGleaner(config, show_progress=False)


def _ground(gleaner: GraphGleaner, action: str = "drop") -> None:
    grounding = gleaner.extraction_config.entity_grounding
    grounding.enabled = True
    grounding.action = action


def _vendor() -> Entity:
    return Entity(id=VENDOR, name="Vendor", type="ORG", text_unit_ids=[UNIT.id])


def _entity_issue(name: str, evidence: str) -> dict:
    return {
        "issue_type": "MISSING_ENTITY",
        "details": {"name": name, "type": "ORG", "description": "An organization"},
        "text_evidence": evidence,
    }


def _relationship_issue(source: str, target: str, evidence: str) -> dict:
    return {
        "issue_type": "MISSING_RELATIONSHIP",
        "details": {
            "source": source,
            "target": target,
            "type": "GUARANTEED_BY",
            "description": "A relationship",
        },
        "text_evidence": evidence,
    }


def _glean(gleaner, issues, entities=None, unit=UNIT, record=None):
    """Apply one refinement answer to ``unit``'s record and merge the record.

    The record lists the given entities that cite ``unit``; pass ``record``
    to apply a later round's answer to the same record.
    """
    if record is None:
        entities = [_vendor()] if entities is None else entities
        record = TextUnitExtraction(
            text_unit_id=unit.id,
            entities=[e for e in entities if unit.id in (e.text_unit_ids or [])],
        )
    gleaner._apply_answer({"identified_issues": {"issue": issues}}, unit, record)
    merged_entities, merged_relationships = merge_contributions([record])
    return {e.id: e for e in merged_entities}, merged_relationships


def _glean_records(gleaner, units, records):
    """Glean ``records`` and return the graph they state, and the stats."""
    records, stats = gleaner.glean(units, records)
    entities, relationships = merge_contributions(records)
    return entities, relationships, stats


def test_relationship_to_a_rejected_gleaned_entity_is_dropped(gleaner) -> None:
    _ground(gleaner)
    entities, relationships = _glean(
        gleaner,
        [
            _entity_issue("Phantom Holdings", FABRICATED_QUOTE),
            _relationship_issue("Vendor", "Phantom Holdings", GROUNDED_QUOTE),
        ],
    )
    assert PHANTOM not in entities
    assert relationships == []
    assert gleaner._dropped_rejected_endpoint == 1


def test_rejection_holds_whatever_the_issue_order(gleaner) -> None:
    _ground(gleaner)
    entities, relationships = _glean(
        gleaner,
        [
            _relationship_issue("Vendor", "Phantom Holdings", GROUNDED_QUOTE),
            _entity_issue("Phantom Holdings", FABRICATED_QUOTE),
        ],
    )
    assert PHANTOM not in entities
    assert relationships == []


def test_rejection_carries_into_later_rounds(gleaner) -> None:
    # The name is in this chunk, so only the earlier rejection stops the stub.
    _ground(gleaner)
    unit = TextUnit(id="t1", text=f"{TEXT} Phantom Holdings is named once.")
    record = TextUnitExtraction(text_unit_id="t1", entities=[_vendor()])
    _glean(
        gleaner,
        [_entity_issue("Phantom Holdings", FABRICATED_QUOTE)],
        unit=unit,
        record=record,
    )
    entities, relationships = _glean(
        gleaner,
        [_relationship_issue("Vendor", "Phantom Holdings", GROUNDED_QUOTE)],
        unit=unit,
        record=record,
    )
    assert PHANTOM not in entities
    assert relationships == []


def test_a_grounded_copy_lifts_the_rejection(gleaner) -> None:
    _ground(gleaner)
    unit = TextUnit(id="t1", text=f"{TEXT} Phantom Holdings guarantees payment.")
    entities, relationships = _glean(
        gleaner,
        [
            _entity_issue("Phantom Holdings", FABRICATED_QUOTE),
            _entity_issue("Phantom Holdings", "Phantom Holdings guarantees payment."),
            _relationship_issue("Vendor", "Phantom Holdings", GROUNDED_QUOTE),
        ],
        unit=unit,
    )
    assert entities[PHANTOM].text_unit_ids == ["t1"]
    assert len(relationships) == 1


@pytest.mark.parametrize("action", ["drop", "penalize"])
def test_stub_needs_its_name_in_the_chunk(gleaner, action) -> None:
    _ground(gleaner, action)
    entities, relationships = _glean(
        gleaner,
        [
            _relationship_issue("Vendor", "Ghost Ltd", GROUNDED_QUOTE),
            _relationship_issue("Vendor", "Depot", GROUNDED_QUOTE),
        ],
    )
    assert BaseProcessor._generate_entity_id("Ghost Ltd") not in entities
    depot = entities[BaseProcessor._generate_entity_id("Depot")]
    assert depot.text_unit_ids == ["t1"]
    assert [(r.source_name, r.target_name) for r in relationships] == [
        ("Vendor", "Depot")
    ]
    assert gleaner._dropped_ungrounded_endpoint_name == 1


def test_unnamed_endpoint_listed_by_another_unit_is_dropped_too(gleaner) -> None:
    # Ghost Ltd is extracted from another unit, whose record this unit's
    # answer never sees: the edge is dropped as when that unit is absent.
    _ground(gleaner)
    ghost = Entity(
        id=BaseProcessor._generate_entity_id("Ghost Ltd"),
        name="Ghost Ltd",
        text_unit_ids=["t2"],
    )
    _, relationships = _glean(
        gleaner,
        [_relationship_issue("Vendor", "Ghost Ltd", GROUNDED_QUOTE)],
        entities=[_vendor(), ghost],
    )
    assert relationships == []


def test_grounding_off_keeps_the_stub(gleaner) -> None:
    entities, relationships = _glean(
        gleaner,
        [
            _entity_issue("Phantom Holdings", FABRICATED_QUOTE),
            _relationship_issue("Vendor", "Ghost Ltd", GROUNDED_QUOTE),
        ],
    )
    assert PHANTOM in entities
    assert BaseProcessor._generate_entity_id("Ghost Ltd") in entities
    assert len(relationships) == 1


# --------------------------------------------------------------------------- #
# Entities extraction dropped for the unit
# --------------------------------------------------------------------------- #
@pytest.fixture
def extractor(config: Config, mocker) -> GraphExtractor:
    mocker.patch.object(bedrock_module, "BedrockLanguageModelFactory")
    mocker.patch.object(extractor_module, "create_robust_xml_output_parser")
    mocker.patch.object(extractor_module, "setup_chain")
    return GraphExtractor(config)


def _extract(extractor: GraphExtractor, mocker, unit: TextUnit):
    answer = {
        "entities": [
            {"name": "Vendor", "type": "ORG", "source_text": GROUNDED_QUOTE},
            {
                "name": "Phantom Holdings",
                "type": "ORG",
                "source_text": FABRICATED_QUOTE,
            },
        ],
        "relationships": [],
    }
    extractor.batch_processor = mocker.Mock()
    extractor.batch_processor.execute_with_fallback.return_value = [answer]
    return extractor.extract_units([unit])


@pytest.mark.parametrize("named_in_chunk", [False, True])
def test_entity_extraction_dropped_is_not_regleaned(
    extractor, gleaner, mocker, named_in_chunk
) -> None:
    unit = UNIT
    if named_in_chunk:
        unit = TextUnit(id="t1", text=f"{TEXT} Phantom Holdings is named once.")
    extractor.extraction_config.entity_grounding.enabled = True
    records, _ = _extract(extractor, mocker, unit)
    assert [e.name for e in records[0].entities] == ["Vendor"]
    assert records[0].rejected == {"phantom holdings": "ungrounded"}

    gleaner.gleaning_config.max_rounds = 1
    gleaner.graph_refiner = mocker.Mock()
    gleaner.graph_refiner.invoke.return_value = {
        "refinement_plan": {
            "identified_issues": {
                "issue": [
                    _relationship_issue("Vendor", "Phantom Holdings", GROUNDED_QUOTE)
                ]
            }
        }
    }
    entities, relationships, stats = _glean_records(gleaner, [unit], records)
    assert PHANTOM not in {e.id for e in entities}
    assert relationships == []
    assert stats.relationships_dropped_rejected_endpoint == 1


def test_rejected_entities_reach_gleaning_through_the_stages(mocker) -> None:
    from unified_kg_rag.adapters.ingestion.gleaner import GleaningStats
    from unified_kg_rag.adapters.ingestion.graph_extractor import ExtractionStats
    from unified_kg_rag.application.ingestion import pipeline_stages as ps
    from unified_kg_rag.domain.models import PipelineConfig, PipelineContext
    from unified_kg_rag.domain.models.pipeline import PipelineStageStatus

    record = TextUnitExtraction(text_unit_id="t1", rejected={"phantom": "ungrounded"})
    extractor = mocker.MagicMock()
    extractor.extract_units.return_value = ([record], ExtractionStats())
    mocker.patch.object(ps, "GraphExtractor", return_value=extractor)
    gleaner = mocker.MagicMock()
    gleaner.glean.return_value = ([record], GleaningStats())
    mocker.patch.object(ps, "GraphGleaner", return_value=gleaner)
    context = PipelineContext(
        pipeline_id="p",
        config=PipelineConfig(),
        status=PipelineStageStatus.RUNNING,
        start_time="2026-01-01T00:00:00",
        source_directory=".",
        text_units=[UNIT],
    )

    ps.GraphExtractionStage(config=Config())._execute_core(context)
    ps.GleaningStage(config=Config())._execute_core(context)

    (passed,) = gleaner.glean.call_args.args[1]
    assert passed.rejected == {"phantom": "ungrounded"}


@pytest.mark.parametrize(
    ("name", "text", "grounded"),
    [
        ("Vendor", "The Vendor ships.", True),
        ("Vendor", "Vendors ship.", True),
        ("ACME Corp.", "acme corp signs", True),
        ("Vendor", "Advendor ships.", False),
        ("Phantom Holdings", "Vendor ships.", False),
        ("벤더", "벤더는 물품을 납품한다.", True),
        ("", "Vendor ships.", False),
        ("Vendor", "", True),
    ],
)
def test_is_name_grounded(name: str, text: str, grounded: bool) -> None:
    assert is_name_grounded(name, text) is grounded


def test_relationship_list_is_not_mutated(gleaner) -> None:
    _ground(gleaner)
    relationships = [
        Relationship(
            id="r",
            source_id=VENDOR,
            target_id=PHANTOM,
            source_name="Vendor",
            target_name="Phantom Holdings",
            type="X",
            text_unit_ids=["t1"],
        )
    ]
    kept = gleaner._drop_unsupported_relationships(
        relationships,
        UNIT,
        [_vendor()],
        TextUnitExtraction(
            text_unit_id="t1", rejected={"phantom holdings": "ungrounded"}
        ),
    )
    assert kept == [] and len(relationships) == 1


# --------------------------------------------------------------------------- #
# Entities the confidence threshold removed
# --------------------------------------------------------------------------- #
def _extract_low_confidence(extractor: GraphExtractor, mocker):
    extractor.extraction_config.entity_confidence_threshold = 0.5
    answer = {
        "entities": [
            {"name": "Vendor", "type": "ORG", "confidence": "9"},
            {"name": "Phantom Holdings", "type": "ORG", "confidence": "2"},
        ],
        "relationships": [],
    }
    extractor.batch_processor = mocker.Mock()
    extractor.batch_processor.execute_with_fallback.return_value = [answer]
    return extractor.extract_units([UNIT])


def _glean_answer(gleaner, mocker, issues) -> None:
    gleaner.gleaning_config.max_rounds = 1
    gleaner.graph_refiner = mocker.Mock()
    gleaner.graph_refiner.invoke.return_value = {
        "refinement_plan": {"identified_issues": {"issue": issues}}
    }


@pytest.mark.parametrize("grounding", [False, True])
def test_low_confidence_entity_is_not_resurrected_by_a_stub(
    extractor, gleaner, mocker, grounding
) -> None:
    gleaner.extraction_config.entity_grounding.enabled = grounding
    records, stats = _extract_low_confidence(extractor, mocker)
    assert [e.name for e in records[0].entities] == ["Vendor"]
    assert stats.entities_filtered_by_confidence == 1
    assert records[0].rejected == {"phantom holdings": "low_confidence"}

    _glean_answer(
        gleaner,
        mocker,
        [_relationship_issue("Vendor", "Phantom Holdings", GROUNDED_QUOTE)],
    )
    entities, relationships, gleaning_stats = _glean_records(gleaner, [UNIT], records)
    assert PHANTOM not in {e.id for e in entities}
    assert relationships == []
    assert gleaning_stats.relationships_dropped_rejected_endpoint == 1


def test_low_confidence_entity_is_not_regleaned_as_an_entity(
    extractor, gleaner, mocker
) -> None:
    # The gleaner's answer carries no score, so it cannot overturn the
    # extraction's; without this the entity would come back at 1.0.
    records, _ = _extract_low_confidence(extractor, mocker)
    _glean_answer(
        gleaner,
        mocker,
        [
            _entity_issue("Phantom Holdings", GROUNDED_QUOTE),
            _relationship_issue("Vendor", "Phantom Holdings", GROUNDED_QUOTE),
        ],
    )
    entities, relationships, stats = _glean_records(gleaner, [UNIT], records)
    assert PHANTOM not in {e.id for e in entities}
    assert relationships == []
    assert stats.entities_dropped_low_confidence == 1


def test_the_low_confidence_rejection_is_per_unit(extractor, gleaner, mocker) -> None:
    # Another unit naming the entity is not bound by t1's low score.
    records, _ = _extract_low_confidence(extractor, mocker)
    other = TextUnit(id="t2", text="Vendor and Phantom Holdings sign the lease.")
    _glean_answer(
        gleaner,
        mocker,
        [_relationship_issue("Vendor", "Phantom Holdings", other.text)],
    )
    gleaned, _ = gleaner.glean([other], [TextUnitExtraction(text_unit_id="t2")])
    entities, relationships = merge_contributions(records + gleaned)
    by_id = {e.id: e for e in entities}
    assert by_id[PHANTOM].text_unit_ids == ["t2"]
    assert by_id[PHANTOM].confidence == 1.0
    assert len(relationships) == 1


def test_gleaned_entity_below_the_threshold_is_dropped(gleaner) -> None:
    gleaner.extraction_config.entity_confidence_threshold = 0.5
    issue = _entity_issue("Phantom Holdings", GROUNDED_QUOTE)
    issue["details"]["confidence"] = "2"
    entities, relationships = _glean(
        gleaner,
        [issue, _relationship_issue("Vendor", "Phantom Holdings", GROUNDED_QUOTE)],
    )
    assert PHANTOM not in entities
    assert relationships == []
    assert gleaner._dropped_low_confidence_entities == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_penalize_keeps_ungrounded_additions_down_weighted(gleaner, reverse) -> None:
    # As at extraction: the entity stays at confidence x penalty_factor, and
    # the relationship between two entities the chunk lists stays at weight x
    # penalty_factor, whatever the order of the issues.
    _ground(gleaner, "penalize")
    issues = [
        _entity_issue("Phantom Holdings", FABRICATED_QUOTE),
        _relationship_issue("Vendor", "Phantom Holdings", FABRICATED_QUOTE),
    ]
    entities, relationships = _glean(gleaner, issues[::-1] if reverse else issues)
    assert entities[PHANTOM].confidence == pytest.approx(0.5)
    (rel,) = relationships
    assert (rel.source_name, rel.target_name) == ("Vendor", "Phantom Holdings")
    assert rel.weight == pytest.approx(0.5)


def test_penalize_drops_an_ungrounded_relationship_to_an_unlisted_entity(
    gleaner,
) -> None:
    # Depot is named in the chunk but not listed by it: keeping the ungrounded
    # edge would make Depot cite the chunk, as extraction's penalize rule
    # forbids.
    _ground(gleaner, "penalize")
    entities, relationships = _glean(
        gleaner, [_relationship_issue("Vendor", "Depot", FABRICATED_QUOTE)]
    )
    assert relationships == []
    assert BaseProcessor._generate_entity_id("Depot") not in entities


def test_penalized_entity_below_the_threshold_is_dropped(gleaner) -> None:
    _ground(gleaner, "penalize")
    gleaner.extraction_config.entity_confidence_threshold = 0.6
    entities, _ = _glean(gleaner, [_entity_issue("Phantom Holdings", FABRICATED_QUOTE)])
    assert PHANTOM not in entities
