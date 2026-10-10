# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gleaned relationships do not bring back entities grounding rejected.

Every kept gleaned relationship makes its endpoints cite its unit and adds a
stub for an endpoint the batch lacks. The grounding check on a relationship
covers only its quote, which need not name its endpoints, so without a guard a
relationship naming an entity rejected as ungrounded (by extraction, or in the
same gleaning answer) recreates it as a stub. Runs the real
``_parse_refinement_output`` -> ``_merge_round`` path; only the model is
replaced.
"""

from __future__ import annotations

import pytest

import unified_kg_rag.adapters.ingestion.gleaner as gleaner_module
import unified_kg_rag.adapters.ingestion.graph_extractor as extractor_module
from unified_kg_rag.adapters import providers as providers_module
from unified_kg_rag.adapters.ingestion.gleaner import GraphGleaner
from unified_kg_rag.adapters.ingestion.graph_extractor import GraphExtractor
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.entity_grounding import is_name_grounded
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    RejectedEntity,
    Relationship,
    TextUnit,
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
    mocker.patch.object(gleaner_module, "boto3")
    mocker.patch.object(providers_module, "BedrockLanguageModelFactory")
    mocker.patch.object(gleaner_module, "create_robust_xml_output_parser")
    mocker.patch.object(gleaner_module, "setup_chain")
    return GraphGleaner(config, use_process_pool=False, show_progress=False)


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


def _glean(gleaner, issues, entities=None, unit=UNIT):
    """Parse one refinement answer for ``unit`` and merge it as a round does."""
    entities = [_vendor()] if entities is None else entities
    plan = {"identified_issues": {"issue": issues}}
    new_entities, new_relationships = gleaner._parse_refinement_output(
        plan, unit, entities, []
    )
    merged_entities, merged_relationships = GraphGleaner._merge_round(
        entities + new_entities, new_relationships
    )
    return {e.id: e for e in merged_entities}, merged_relationships


@pytest.mark.parametrize("action", ["drop", "penalize"])
def test_relationship_to_a_rejected_gleaned_entity_is_dropped(gleaner, action) -> None:
    _ground(gleaner, action)
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


@pytest.mark.parametrize("action", ["drop", "penalize"])
def test_rejection_holds_whatever_the_issue_order(gleaner, action) -> None:
    _ground(gleaner, action)
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
    _glean(gleaner, [_entity_issue("Phantom Holdings", FABRICATED_QUOTE)], unit=unit)
    entities, relationships = _glean(
        gleaner,
        [_relationship_issue("Vendor", "Phantom Holdings", GROUNDED_QUOTE)],
        unit=unit,
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
    # Ghost Ltd is extracted from another unit; keeping the edge only then
    # would make the result depend on which units share the batch.
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
    mocker.patch.object(extractor_module, "boto3")
    mocker.patch.object(providers_module, "BedrockLanguageModelFactory")
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
    return extractor.extract_from_text_units([unit])


@pytest.mark.parametrize("named_in_chunk", [False, True])
def test_entity_extraction_dropped_is_not_regleaned(
    extractor, gleaner, mocker, named_in_chunk
) -> None:
    unit = UNIT
    if named_in_chunk:
        unit = TextUnit(id="t1", text=f"{TEXT} Phantom Holdings is named once.")
    extractor.extraction_config.entity_grounding.enabled = True
    entities, relationships, _ = _extract(extractor, mocker, unit)
    assert [e.name for e in entities] == ["Vendor"]
    assert extractor.rejected_entities == [
        RejectedEntity(
            text_unit_id="t1", entity_key="phantom holdings", reason="ungrounded"
        )
    ]

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
    entities, relationships, stats = gleaner.glean_graph(
        [unit],
        entities,
        relationships,
        rejected_entities=extractor.rejected_entities,
    )
    assert PHANTOM not in {e.id for e in entities}
    assert relationships == []
    assert stats.relationships_dropped_rejected_endpoint == 1


def test_rejected_entities_reach_gleaning_through_the_stages(mocker) -> None:
    from unified_kg_rag.adapters.ingestion.gleaner import GleaningStats
    from unified_kg_rag.adapters.ingestion.graph_extractor import ExtractionStats
    from unified_kg_rag.application.ingestion import pipeline_stages as ps
    from unified_kg_rag.domain.models import PipelineConfig, PipelineContext
    from unified_kg_rag.domain.models.pipeline import PipelineStageStatus

    rejected = [
        RejectedEntity(text_unit_id="t1", entity_key="phantom", reason="ungrounded")
    ]
    extractor = mocker.MagicMock()
    extractor.extract_from_text_units.return_value = ([], [], ExtractionStats())
    extractor.rejected_entities = rejected
    mocker.patch.object(ps, "GraphExtractor", return_value=extractor)
    gleaner = mocker.MagicMock()
    gleaner.glean_graph.return_value = ([], [], GleaningStats())
    mocker.patch.object(ps, "GraphGleaner", return_value=gleaner)
    context = PipelineContext(
        pipeline_id="p",
        config=PipelineConfig(),
        status=PipelineStageStatus.RUNNING,
        start_time="2026-01-01T00:00:00",
        source_directory=".",
        text_units=[UNIT],
    )

    session = mocker.MagicMock()
    ps.GraphExtractionStage(config=Config(), boto_session=session)._execute_core(
        context
    )
    ps.GleaningStage(config=Config(), boto_session=session)._execute_core(context)

    assert context.rejected_entities == rejected
    assert gleaner.glean_graph.call_args.kwargs["rejected_entities"] == rejected


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
        relationships, UNIT, [_vendor()], {"phantom holdings"}
    )
    assert kept == [] and len(relationships) == 1
