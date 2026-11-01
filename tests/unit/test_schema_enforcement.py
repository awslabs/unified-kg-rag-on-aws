# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The extraction schema applied to what extraction and gleaning parse.

``align`` keeps every statement and counts the off-schema ones; ``strict``
drops off-schema entities (rejected for the unit as ``off_schema``) with the
unit's relationships naming them, off-schema relationship types, and
relationships no pattern allows between two endpoints of declared types.
Drops are not failures. Runs the real parse and gleaning code; only the
model is replaced.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

import unified_kg_rag.adapters.ingestion.gleaner as gleaner_module
import unified_kg_rag.adapters.ingestion.graph_extractor as extractor_module
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.ingestion.gleaner import (
    _SCHEMA_STATS,
    GleaningStats,
    GraphGleaner,
)
from unified_kg_rag.adapters.ingestion.graph_extractor import (
    ExtractionStats,
    GraphExtractor,
)
from unified_kg_rag.domain.ingestion.contributions import (
    record_from_payload,
    record_payload,
)
from unified_kg_rag.domain.ingestion.schema_conformance import (
    SCHEMA_COUNTERS,
    SchemaConformance,
)
from unified_kg_rag.domain.models import (
    Config,
    ExtractionSchema,
    TextUnit,
    TextUnitExtraction,
)

pytestmark = pytest.mark.unit

TEXT = "Alice works for Acme Corp. Acme Corp makes the Widget in Springfield."
UNIT = TextUnit(id="t1", text=TEXT)
SCHEMA: dict[str, Any] = {
    "entity_types": ["PERSON", "ORGANIZATION", "PRODUCT"],
    "relationship_types": ["WORKS_FOR", "MAKES", "PARTNERS_WITH"],
    "relationship_patterns": [
        "PERSON -WORKS_FOR-> ORGANIZATION",
        "ORGANIZATION -MAKES-> PRODUCT",
    ],
}


def _config(mode: str = "strict", **extraction: Any) -> Config:
    return Config.model_validate(
        {
            "processing": {
                "graph_extraction": {**SCHEMA, **extraction, "schema_mode": mode}
            }
        }
    )


def _entity(name: str, type_: str) -> dict[str, Any]:
    return {"name": name, "type": type_, "description": f"{name} here"}


def _rel(source: str, target: str, type_: str) -> dict[str, Any]:
    return {"source": source, "target": target, "type": type_, "strength": "5"}


def _output(entities: list[dict], relationships: list[dict]) -> dict[str, Any]:
    return {"entities": entities, "relationships": relationships}


@pytest.fixture
def patched(mocker) -> None:
    mocker.patch.object(bedrock_module, "BedrockLanguageModelFactory")
    for module in (extractor_module, gleaner_module):
        mocker.patch.object(module, "create_robust_xml_output_parser")
        mocker.patch.object(module, "setup_chain")


def _extract(
    config: Config, output: dict[str, Any]
) -> tuple[TextUnitExtraction, ExtractionStats]:
    extractor = GraphExtractor(config)
    (record,) = extractor._process_extraction_results([UNIT], [output])
    return record, extractor.stats


def _names(record: TextUnitExtraction) -> dict[str, str]:
    return {e.name: e.type or "" for e in record.entities}


def _edges(record: TextUnitExtraction) -> set[tuple[str, str, str]]:
    return {
        (r.source_name or "", r.type or "", r.target_name or "")
        for r in record.relationships
    }


MIXED = _output(
    [
        _entity("Alice", "person"),
        _entity("Acme Corp", "Organization"),
        _entity("Widget", "product"),
        _entity("Springfield", "city"),
        _entity("Monday", ""),
    ],
    [
        _rel("Alice", "Acme Corp", "works for"),
        _rel("Acme Corp", "Widget", "makes"),
        _rel("Widget", "Springfield", "MAKES"),
        _rel("Monday", "Alice", "WORKS_FOR"),
        _rel("Alice", "Widget", "likes"),
        _rel("Acme Corp", "Alice", "WORKS_FOR"),
    ],
)


# --- extraction ----------------------------------------------------------------


def test_strict_drops_off_schema_entities_and_their_relationships(patched) -> None:
    record, stats = _extract(_config(), MIXED)
    assert _names(record) == {
        "Alice": "person",
        "Acme Corp": "organization",
        "Widget": "product",
    }
    # Untyped and undeclared types are rejected for the unit.
    assert record.rejected == {"springfield": "off_schema", "monday": "off_schema"}
    assert _edges(record) == {
        ("Alice", "WORKS_FOR", "Acme Corp"),
        ("Acme Corp", "MAKES", "Widget"),
    }
    assert stats.entities_off_schema == stats.entities_dropped_off_schema == 2
    assert stats.relationships_dropped_off_schema_endpoint == 2
    assert stats.relationships_off_schema == stats.relationships_dropped_off_schema
    assert stats.relationships_dropped_off_schema == 1  # likes
    # Reversed: dropped, not flipped.
    assert stats.relationships_off_pattern == 1
    assert stats.relationships_dropped_off_pattern == 1


def test_align_keeps_everything_and_counts(patched) -> None:
    record, stats = _extract(_config("align"), MIXED)
    assert set(_names(record)) == {
        "Alice",
        "Acme Corp",
        "Widget",
        "Springfield",
        "Monday",
    }
    assert len(record.relationships) == 6
    assert record.rejected == {}
    assert stats.entities_off_schema == 2
    assert stats.relationships_off_schema == 1
    assert stats.relationships_off_pattern == 1
    for name in SCHEMA_COUNTERS:
        if "dropped" in name:
            assert getattr(stats, name) == 0, name


def test_off_counts_nothing(patched) -> None:
    record, stats = _extract(_config("off"), MIXED)
    assert len(record.entities) == 5 and len(record.relationships) == 6
    assert all(getattr(stats, name) == 0 for name in SCHEMA_COUNTERS)


def test_strict_is_a_subset_of_align(patched) -> None:
    strict, _ = _extract(_config(), MIXED)
    align, _ = _extract(_config("align"), MIXED)
    assert {e.id for e in strict.entities} <= {e.id for e in align.entities}
    assert {r.id for r in strict.relationships} <= {r.id for r in align.relationships}


def test_an_in_schema_statement_lifts_the_rejection(patched) -> None:
    output = _output(
        [_entity("Acme Corp", "vendor"), _entity("Acme Corp", "organization")],
        [_rel("Alice", "Acme Corp", "WORKS_FOR")],
    )
    record, stats = _extract(_config(), output)
    assert _names(record) == {"Acme Corp": "organization"}
    assert record.rejected == {}
    assert len(record.relationships) == 1
    assert stats.entities_dropped_off_schema == 1


def test_the_in_schema_type_does_not_win_the_vote_under_align(patched) -> None:
    output = _output(
        [
            _entity("Acme Corp", "vendor"),
            _entity("Acme Corp", "vendor"),
            _entity("Acme Corp", "organization"),
        ],
        [],
    )
    record, _ = _extract(_config("align"), output)
    assert _names(record) == {"Acme Corp": "vendor"}
    record, _ = _extract(_config(), output)
    assert _names(record) == {"Acme Corp": "organization"}


@pytest.mark.parametrize(
    "relationship",
    [
        # Ghost is not listed by the unit: its type is unknown.
        _rel("Ghost", "Widget", "WORKS_FOR"),
        # PARTNERS_WITH has no pattern: not restricted.
        _rel("Widget", "Alice", "PARTNERS_WITH"),
    ],
)
def test_a_pattern_needs_both_endpoint_types_known(patched, relationship) -> None:
    output = _output(
        [_entity("Alice", "person"), _entity("Widget", "product")], [relationship]
    )
    record, stats = _extract(_config(), output)
    assert len(record.relationships) == 1
    assert stats.relationships_off_pattern == 0


def test_an_endpoint_of_an_undeclared_type_is_unknown_under_align(patched) -> None:
    output = _output(
        [_entity("Alice", "wizard"), _entity("Widget", "product")],
        [_rel("Alice", "Widget", "WORKS_FOR")],
    )
    _, stats = _extract(_config("align"), output)
    assert stats.relationships_off_pattern == 0
    assert stats.entities_off_schema == 1


def test_strict_drops_are_not_failures(patched, mocker) -> None:
    extractor = GraphExtractor(_config())
    extractor.batch_processor = mocker.Mock()
    extractor.batch_processor.execute_with_fallback.return_value = [
        _output([_entity("Springfield", "city")], [])
    ]
    records, stats = extractor.extract_units([UNIT])
    (record,) = records
    assert record.entities == [] and record.relationships == []
    assert record.rejected == {"springfield": "off_schema"}
    assert stats.num_successful_extractions == 1
    assert stats.num_failed_extractions == 0
    assert stats.failed_text_unit_ids == []
    assert stats.item_failed_text_unit_ids == []


def test_the_off_schema_rejection_round_trips_through_the_store(patched) -> None:
    record, _ = _extract(_config(), MIXED)
    restored = record_from_payload(record_payload(record), UNIT)
    assert restored.rejected == record.rejected
    assert "off_schema" in restored.rejected.values()


def test_one_info_summary_and_labels_at_debug_only(patched, mocker, caplog) -> None:
    extractor = GraphExtractor(_config())
    extractor.batch_processor = mocker.Mock()
    extractor.batch_processor.execute_with_fallback.return_value = [MIXED, MIXED]
    units = [UNIT, TextUnit(id="t2", text=TEXT)]
    with caplog.at_level(logging.DEBUG, logger=extractor_module.__name__):
        with caplog.at_level(
            logging.DEBUG, logger="unified_kg_rag.domain.ingestion.schema_conformance"
        ):
            extractor.extract_units(units)
    above_debug = [r for r in caplog.records if r.levelno > logging.DEBUG]
    summaries = [r for r in above_debug if "Extraction schema" in r.getMessage()]
    assert len(summaries) == 1
    assert "dropped 4" in summaries[0].getMessage()
    # Model text (names, labels) only at debug.
    for record in above_debug:
        message = record.getMessage()
        for text in ("Springfield", "city", "likes", "Monday"):
            assert text not in message, message
    debug = " ".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG
    )
    assert "city (2)" in debug and "likes (2)" in debug
    assert "Dropping entity 'Springfield'" in debug


@pytest.mark.parametrize(
    "extraction",
    [{}, {"entity_types": ["GENE", "DISEASE"]}],
)
def test_entity_types_alone_check_nothing(patched, caplog, extraction) -> None:
    config = Config.model_validate({"processing": {"graph_extraction": extraction}})
    extractor = GraphExtractor(config)
    with caplog.at_level(logging.INFO, logger=extractor_module.__name__):
        (record,) = extractor._process_extraction_results([UNIT], [MIXED])
        extractor._log_schema_summary()
    assert len(record.entities) == 5 and len(record.relationships) == 6
    assert all(getattr(extractor.stats, name) == 0 for name in SCHEMA_COUNTERS)
    assert "Extraction schema" not in caplog.text


# --- the domain check ----------------------------------------------------------


def test_conform_is_idempotent_and_order_independent(patched) -> None:
    record, _ = _extract(_config("off"), MIXED)
    conformance = SchemaConformance(
        _config().processing.graph_extraction.extraction_schema()
    )
    once = conformance.conform(record.entities, record.relationships)
    twice = conformance.conform(once.entities, once.relationships)
    assert [e.id for e in twice.entities] == [e.id for e in once.entities]
    assert [r.id for r in twice.relationships] == [r.id for r in once.relationships]
    assert twice.off_schema_keys == set()
    backwards = conformance.conform(record.entities[::-1], record.relationships[::-1])
    assert {e.id for e in backwards.entities} == {e.id for e in once.entities}
    assert {r.id for r in backwards.relationships} == {r.id for r in once.relationships}
    assert backwards.off_schema_keys == once.off_schema_keys


def test_strict_without_declared_types_checks_nothing() -> None:
    conformance = SchemaConformance(ExtractionSchema(mode="strict"))
    assert conformance.applies
    assert not conformance.entity_off_schema("")
    assert not conformance.relationship_off_schema("ANYTHING")
    assert not conformance.off_pattern("ANYTHING", "a", "b")


# --- gleaning ------------------------------------------------------------------


def _gleaner(config: Config) -> GraphGleaner:
    return GraphGleaner(config, show_progress=False)


def _seed(config: Config) -> TextUnitExtraction:
    record, _ = _extract(
        config,
        _output(
            [
                _entity("Alice", "person"),
                _entity("Acme Corp", "organization"),
                _entity("Widget", "product"),
                _entity("Springfield", "city"),
            ],
            [_rel("Alice", "Acme Corp", "WORKS_FOR")],
        ),
    )
    return record


def _answer(*issues: dict[str, Any]) -> dict[str, Any]:
    return {"identified_issues": {"issue": list(issues)}}


def _missing_entity(name: str, type_: str) -> dict[str, Any]:
    return {"issue_type": "MISSING_ENTITY", "details": _entity(name, type_)}


def _missing_rel(source: str, target: str, type_: str) -> dict[str, Any]:
    return {
        "issue_type": "MISSING_RELATIONSHIP",
        "details": _rel(source, target, type_),
    }


def _apply(gleaner: GraphGleaner, record: TextUnitExtraction, *issues) -> bool:
    return gleaner._apply_answer(_answer(*issues), UNIT, record)


def _counts(gleaner: GraphGleaner) -> dict[str, int]:
    return {name: getattr(gleaner._schema_counts, name) for name in _SCHEMA_STATS}


def test_strict_gleaning_rejects_an_off_schema_entity(patched) -> None:
    config = _config()
    gleaner, record = _gleaner(config), _seed(config)
    gained = _apply(
        gleaner,
        record,
        _missing_entity("Gizmo Ltd", "spaceship"),
        _missing_rel("Gizmo Ltd", "Acme Corp", "PARTNERS_WITH"),
    )
    assert not gained
    assert "Gizmo Ltd" not in _names(record)
    assert record.rejected["gizmo ltd"] == "off_schema"
    assert len(record.relationships) == 1
    assert _counts(gleaner)["entities_dropped_off_schema"] == 1
    assert _counts(gleaner)["relationships_dropped_off_schema_endpoint"] == 1


def test_a_relationship_to_an_extraction_rejected_entity_is_dropped(patched) -> None:
    config = _config()
    gleaner, record = _gleaner(config), _seed(config)
    assert record.rejected == {"springfield": "off_schema"}
    _apply(gleaner, record, _missing_rel("Acme Corp", "Springfield", "PARTNERS_WITH"))
    assert len(record.relationships) == 1
    assert _counts(gleaner)["relationships_dropped_off_schema_endpoint"] == 1
    # Listed with a declared type, it is the unit's entity again.
    gained = _apply(
        gleaner,
        record,
        _missing_entity("Springfield", "organization"),
        _missing_rel("Acme Corp", "Springfield", "PARTNERS_WITH"),
    )
    assert gained
    assert _names(record)["Springfield"] == "ORGANIZATION"
    assert ("Acme Corp", "PARTNERS_WITH", "Springfield") in _edges(record)


def test_strict_gleaning_drops_off_schema_and_off_pattern_relationships(
    patched,
) -> None:
    config = _config()
    gleaner, record = _gleaner(config), _seed(config)
    gained = _apply(
        gleaner,
        record,
        _missing_rel("Alice", "Widget", "likes"),
        _missing_rel("Widget", "Acme Corp", "makes"),
    )
    assert not gained
    assert _edges(record) == {("Alice", "WORKS_FOR", "Acme Corp")}
    assert _counts(gleaner)["relationships_dropped_off_schema"] == 1
    assert _counts(gleaner)["relationships_dropped_off_pattern"] == 1


def test_strict_ignores_corrections_to_off_schema_types(patched) -> None:
    config = _config()
    gleaner, record = _gleaner(config), _seed(config)
    _apply(
        gleaner,
        record,
        {
            "issue_type": "ENTITY_CORRECTION",
            "details": {
                "name": "Alice",
                "corrected_name": "Alice Smith",
                "corrected_type": "wizard",
            },
        },
        {
            "issue_type": "RELATIONSHIP_CORRECTION",
            "details": {
                "source": "Alice",
                "target": "Acme Corp",
                "corrected_type": "admires",
            },
        },
    )
    assert _names(record)["Alice"] == "person"
    assert _edges(record) == {("Alice", "WORKS_FOR", "Acme Corp")}
    assert _counts(gleaner)["corrections_ignored_off_schema"] == 2


def test_strict_ignores_a_reversal_no_pattern_allows(patched) -> None:
    config = _config()
    gleaner, record = _gleaner(config), _seed(config)
    _apply(
        gleaner,
        record,
        {
            "issue_type": "RELATIONSHIP_CORRECTION",
            "details": {
                "source": "Alice",
                "target": "Acme Corp",
                "corrected_source": "Acme Corp",
                "corrected_target": "Alice",
            },
        },
    )
    assert _edges(record) == {("Alice", "WORKS_FOR", "Acme Corp")}
    assert _counts(gleaner)["corrections_ignored_off_schema"] == 1


def test_a_type_correction_breaking_a_pattern_drops_the_relationship(
    patched,
) -> None:
    config = _config()
    gleaner, record = _gleaner(config), _seed(config)
    _apply(
        gleaner,
        record,
        {
            "issue_type": "ENTITY_CORRECTION",
            "details": {"name": "Acme Corp", "corrected_type": "person"},
        },
    )
    assert _names(record)["Acme Corp"] == "PERSON"
    assert record.relationships == []
    assert _counts(gleaner)["relationships_dropped_off_pattern"] == 1


def test_align_gleaning_keeps_everything(patched) -> None:
    config = _config("align")
    gleaner, record = _gleaner(config), _seed(config)
    _apply(
        gleaner,
        record,
        _missing_entity("Gizmo Ltd", "spaceship"),
        _missing_rel("Gizmo Ltd", "Acme Corp", "likes"),
        {
            "issue_type": "ENTITY_CORRECTION",
            "details": {"name": "Alice", "corrected_type": "wizard"},
        },
    )
    assert _names(record)["Gizmo Ltd"] == "spaceship"
    assert _names(record)["Alice"] == "wizard"
    assert ("Gizmo Ltd", "likes", "Acme Corp") in _edges(record)
    assert set(_counts(gleaner).values()) == {0}


def test_glean_reports_the_strict_counters(patched, mocker, caplog) -> None:
    config = _config()
    gleaner = _gleaner(config)
    gleaner.graph_refiner = mocker.Mock()
    gleaner.graph_refiner.invoke.return_value = {
        "refinement_plan": _answer(_missing_entity("Gizmo Ltd", "spaceship"))
    }
    with caplog.at_level(logging.INFO, logger=gleaner_module.__name__):
        records, stats = gleaner.glean([UNIT], [_seed(config)])
    assert isinstance(stats, GleaningStats)
    assert stats.entities_dropped_off_schema == 1
    assert stats.failed_text_unit_ids == [] and stats.num_failed_units == 0
    assert records[0].gleaned
    summary = [r for r in caplog.records if "Gleaning schema" in r.getMessage()]
    assert len(summary) == 1
    assert "Gizmo" not in caplog.text and "spaceship" not in caplog.text
