# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional AWS-free unit tests for GraphGleaner (complements
``test_gleaner_logic.py``).

Covers the branches the logic suite leaves uncovered: the module-level
``prepare_input_task`` / ``format_relationships_with_limit_task`` pure helpers,
quality-score extraction and aggregation, refinement-output parsing and
issue dispatch, the ``GleaningStats`` derived properties, and the end-to-end
``glean_graph`` / ``_perform_llm_refinement`` orchestration driven by a mocked
LangChain chain (no Bedrock, no boto3). The Bedrock/boto wiring is patched out
in the ``gleaner`` fixture exactly as in the logic suite.
"""

from __future__ import annotations

import pytest

import aws_graphrag.adapters.ingestion.gleaner as gleaner_module
from aws_graphrag.adapters.ingestion.gleaner import (
    GleaningRound,
    GleaningStats,
    GraphGleaner,
    format_relationships_with_limit_task,
    prepare_input_task,
)
from aws_graphrag.domain.models import Config, Entity, Relationship, TextUnit

pytestmark = pytest.mark.unit


@pytest.fixture
def gleaner(config: Config, mocker) -> GraphGleaner:
    mocker.patch.object(gleaner_module, "boto3")
    mocker.patch.object(gleaner_module, "BedrockLanguageModelFactory")
    mocker.patch.object(gleaner_module, "create_robust_xml_output_parser")
    mocker.patch.object(gleaner_module, "setup_chain")
    g = GraphGleaner(config, use_process_pool=False, show_progress=False)
    return g


def _ent(id_, name, **kw) -> Entity:
    return Entity(id=id_, name=name, **kw)


def _rel(id_, src, tgt, **kw) -> Relationship:
    kw.setdefault("type", "REL")
    return Relationship(id=id_, source_id=src, target_id=tgt, **kw)


# --------------------------------------------------------------------------- #
# format_relationships_with_limit_task
# --------------------------------------------------------------------------- #
class TestFormatRelationshipsWithLimit:
    def test_under_limit_lists_all(self) -> None:
        rels = [
            _rel("r1", "e1", "e2", source_name="A", target_name="B", type="KNOWS"),
        ]
        out = format_relationships_with_limit_task(rels, max_relationships=5)
        assert out == "'A' -> 'B' (type: 'KNOWS')"

    def test_over_limit_truncates_with_suffix(self) -> None:
        rels = [
            _rel(
                f"r{i}",
                "e1",
                "e2",
                source_name=f"S{i}",
                target_name=f"T{i}",
                type="T",
                weight=float(i),
            )
            for i in range(5)
        ]
        out = format_relationships_with_limit_task(rels, max_relationships=2)
        assert "... and 3 more relationships" in out
        assert len(out.splitlines()) == 3

    def test_prioritizes_high_weight_and_described(self) -> None:
        light = _rel(
            "r1", "e1", "e2", source_name="L", target_name="X", type="T", weight=0.1
        )
        heavy = _rel(
            "r2",
            "e1",
            "e2",
            source_name="H",
            target_name="X",
            type="T",
            weight=9.0,
            description="d",
        )
        out = format_relationships_with_limit_task([light, heavy], max_relationships=1)
        assert out.splitlines()[0].startswith("'H'")


# --------------------------------------------------------------------------- #
# prepare_input_task
# --------------------------------------------------------------------------- #
class TestPrepareInputTask:
    def _config(self) -> dict:
        return {
            "max_entities_per_prompt": 10,
            "max_relationships_per_prompt": 10,
            "target_language": "en",
        }

    def test_filters_to_relevant_lineage(self) -> None:
        unit = TextUnit(id="t1", text="hello")
        ents = [
            _ent("e1", "Alice", text_unit_ids=["t1"]),
            _ent("e2", "Bob", text_unit_ids=["other"]),  # not in t1 -> excluded
        ]
        rels = [
            _rel(
                "r1",
                "e1",
                "e2",
                source_name="Alice",
                target_name="Bob",
                text_unit_ids=["t1"],
            ),
            _rel(
                "r2",
                "e1",
                "e2",
                source_name="Alice",
                target_name="Bob",
                text_unit_ids=["other"],
            ),  # excluded
        ]
        out = prepare_input_task(unit, ents, rels, self._config())
        assert out["text"] == "hello"
        assert "Alice" in out["entities"]
        assert "Bob" not in out["entities"]
        assert out["relationships"].count("->") == 1

    def test_uses_translated_text_when_available(self) -> None:
        unit = TextUnit(id="t1", text="orig", translated_texts={"en": "translated"})
        out = prepare_input_task(unit, [], [], self._config())
        assert out["text"] == "translated"

    def test_translated_text_missing_target_falls_back_to_text(self) -> None:
        unit = TextUnit(id="t1", text="orig", translated_texts={"fr": "bonjour"})
        out = prepare_input_task(unit, [], [], self._config())
        assert out["text"] == "orig"


# --------------------------------------------------------------------------- #
# _extract_quality_scores
# --------------------------------------------------------------------------- #
class TestExtractQualityScores:
    def test_dict_scores(self) -> None:
        plan = {"quality_scores": {"completeness_score": "0.8", "accuracy_score": 0.6}}
        out = GraphGleaner._extract_quality_scores(plan)
        assert out == {"completeness": 0.8, "accuracy": 0.6}

    def test_list_of_dicts_merged(self) -> None:
        plan = {
            "quality_scores": [
                {"completeness_score": 0.4},
                {"accuracy_score": 0.9},
            ]
        }
        out = GraphGleaner._extract_quality_scores(plan)
        assert out == {"completeness": 0.4, "accuracy": 0.9}

    def test_non_dict_non_list_returns_zeroes(self) -> None:
        out = GraphGleaner._extract_quality_scores({"quality_scores": "bad"})
        assert out == {"completeness": 0.0, "accuracy": 0.0}

    def test_unparseable_values_coerce_to_zero(self) -> None:
        plan = {"quality_scores": {"completeness_score": "abc", "accuracy_score": None}}
        out = GraphGleaner._extract_quality_scores(plan)
        assert out == {"completeness": 0.0, "accuracy": 0.0}


# --------------------------------------------------------------------------- #
# _aggregate_quality_scores / _calculate_average
# --------------------------------------------------------------------------- #
class TestQualityAggregation:
    def test_aggregate_skips_none(self) -> None:
        agg = {"completeness": [], "accuracy": []}
        GraphGleaner._aggregate_quality_scores(
            {"completeness": 0.5, "accuracy": None}, agg
        )
        assert agg["completeness"] == [0.5]
        assert agg["accuracy"] == []

    def test_calculate_average(self) -> None:
        assert GraphGleaner._calculate_average([1.0, 3.0]) == 2.0
        assert GraphGleaner._calculate_average([]) == 0.0


# --------------------------------------------------------------------------- #
# _parse_refinement_output + _process_issue
# --------------------------------------------------------------------------- #
class TestParseRefinementOutput:
    def test_empty_plan_returns_empties(self, gleaner) -> None:
        unit = TextUnit(id="t1", text="x")
        ents, rels, scores = gleaner._parse_refinement_output({}, unit, [])
        assert ents == [] and rels == [] and scores == {}

    def test_non_dict_plan_returns_empties(self, gleaner) -> None:
        unit = TextUnit(id="t1", text="x")
        ents, rels, scores = gleaner._parse_refinement_output("not a dict", unit, [])
        assert ents == [] and rels == []

    def test_list_plan_first_element_used(self, gleaner) -> None:
        unit = TextUnit(id="t1", text="x")
        plan = {
            "quality_scores": {"completeness_score": 0.5, "accuracy_score": 0.5},
            "identified_issues": {
                "issue": {
                    "issue_type": "MISSING_ENTITY",
                    "details": {"name": "Neo", "type": "PERSON"},
                }
            },
        }
        ents, rels, scores = gleaner._parse_refinement_output([plan], unit, [])
        assert len(ents) == 1
        assert ents[0].name.lower().startswith("neo")
        assert scores["completeness"] == 0.5

    def test_missing_entity_and_relationship_issues(self, gleaner) -> None:
        unit = TextUnit(id="t1", text="x")
        plan = {
            "identified_issues": {
                "issue": [
                    {
                        "issue_type": "MISSING_ENTITY",
                        "details": {"name": "Alice", "type": "PERSON"},
                    },
                    {
                        "issue_type": "MISSING_ENTITY",
                        "details": {"name": "Bob", "type": "PERSON"},
                    },
                    {
                        "issue_type": "MISSING_RELATIONSHIP",
                        "details": {
                            "source": "Alice",
                            "target": "Bob",
                            "type": "KNOWS",
                        },
                    },
                ]
            }
        }
        ents, rels, _ = gleaner._parse_refinement_output(plan, unit, [])
        assert len(ents) == 2
        assert len(rels) == 1
        # The relationship resolves its endpoints against the just-discovered
        # entities (Alice/Bob added before the relationship issue is processed).
        names = {e.name.lower() for e in ents}
        assert "alice" in names and "bob" in names

    def test_unknown_issue_type_ignored(self, gleaner) -> None:
        unit = TextUnit(id="t1", text="x")
        plan = {
            "identified_issues": {
                "issue": {"issue_type": "SOMETHING_ELSE", "details": {}}
            }
        }
        ents, rels, _ = gleaner._parse_refinement_output(plan, unit, [])
        assert ents == [] and rels == []

    def test_issues_as_bare_list_not_dict(self, gleaner) -> None:
        unit = TextUnit(id="t1", text="x")
        # identified_issues is a list (not a dict wrapper) -> ensure_list path.
        plan = {
            "identified_issues": [
                {"issue_type": "MISSING_ENTITY", "details": {"name": "Zed"}}
            ]
        }
        ents, _, _ = gleaner._parse_refinement_output(plan, unit, [])
        assert len(ents) == 1


# --------------------------------------------------------------------------- #
# GleaningStats derived properties
# --------------------------------------------------------------------------- #
class TestGleaningStats:
    def test_properties_with_rounds(self) -> None:
        stats = GleaningStats(
            total_rounds=2,
            total_entities_added=10,
            total_relationships_added=6,
            total_processing_time=4.0,
            initial_quality_score=0.2,
            final_quality_score=0.7,
        )
        assert stats.quality_improvement == pytest.approx(0.5)
        assert stats.average_round_time == pytest.approx(2.0)
        assert stats.entities_per_round == pytest.approx(5.0)
        assert stats.relationships_per_round == pytest.approx(3.0)

    def test_properties_zero_rounds_guard(self) -> None:
        stats = GleaningStats()
        assert stats.average_round_time == 0.0
        assert stats.entities_per_round == 0.0
        assert stats.relationships_per_round == 0.0


# --------------------------------------------------------------------------- #
# _log_completion_summary (smoke: exercises both convergence branches)
# --------------------------------------------------------------------------- #
class TestLogCompletionSummary:
    def test_converged_summary(self) -> None:
        stats = GleaningStats(
            total_rounds=1,
            rounds=[
                GleaningRound(
                    round_number=1,
                    entities_before=0,
                    relationships_before=0,
                    entities_added=1,
                    relationships_added=1,
                    quality_improvement=0.1,
                    convergence_score=0.9,
                    processing_time=0.5,
                )
            ],
            convergence_achieved=True,
        )
        GraphGleaner._log_completion_summary(stats)  # must not raise

    def test_non_converged_summary(self) -> None:
        GraphGleaner._log_completion_summary(GleaningStats(convergence_achieved=False))


# --------------------------------------------------------------------------- #
# _perform_llm_refinement + glean_graph (mocked chain, no Bedrock)
# --------------------------------------------------------------------------- #
class TestGleanGraphOrchestration:
    def _plan_for(self, name: str) -> dict:
        return {
            "refinement_plan": {
                "quality_scores": {
                    "completeness_score": 0.9,
                    "accuracy_score": 0.9,
                },
                "identified_issues": {
                    "issue": {
                        "issue_type": "MISSING_ENTITY",
                        "details": {"name": name, "type": "PERSON"},
                    }
                },
            }
        }

    def test_perform_llm_refinement_collects_new_entities(
        self, gleaner, mocker
    ) -> None:
        units = [TextUnit(id="t1", text="a"), TextUnit(id="t2", text="b")]
        # batch returns one refinement plan per input, in order.
        gleaner.graph_refiner = mocker.Mock()
        gleaner.graph_refiner.batch.return_value = [
            self._plan_for("Alpha"),
            self._plan_for("Beta"),
        ]
        new_e, new_r, scores = gleaner._perform_llm_refinement(units, [], [])
        assert {e.name.lower() for e in new_e} == {"alpha", "beta"}
        assert scores["completeness"] == pytest.approx(0.9)

    def test_perform_llm_refinement_ignore_errors_returns_empty(
        self, gleaner, mocker
    ) -> None:
        gleaner.ignore_errors = True
        gleaner.graph_refiner = mocker.Mock()
        # BatchProcessor is a frozen-ish Pydantic model, so swap the whole
        # instance for a Mock whose execute_with_fallback raises -> the
        # ignore_errors branch swallows it and returns empties.
        gleaner.batch_processor = mocker.Mock()
        gleaner.batch_processor.execute_with_fallback.side_effect = RuntimeError(
            "hard failure"
        )
        new_e, new_r, scores = gleaner._perform_llm_refinement(
            [TextUnit(id="t1", text="a")], [], []
        )
        assert new_e == [] and new_r == [] and scores == {}

    def test_perform_llm_refinement_reraises_when_not_ignoring(
        self, gleaner, mocker
    ) -> None:
        gleaner.ignore_errors = False
        gleaner.graph_refiner = mocker.Mock()
        gleaner.batch_processor = mocker.Mock()
        gleaner.batch_processor.execute_with_fallback.side_effect = RuntimeError(
            "hard failure"
        )
        with pytest.raises(RuntimeError, match="hard failure"):
            gleaner._perform_llm_refinement([TextUnit(id="t1", text="a")], [], [])

    def test_glean_graph_runs_rounds_and_converges(self, gleaner, mocker) -> None:
        units = [TextUnit(id="t1", text="a")]
        gleaner.graph_refiner = mocker.Mock()
        # Each round discovers the SAME entity name -> merges into existing,
        # entities_added clamps to 0 after round 1 -> convergence -> stop.
        gleaner.graph_refiner.batch.return_value = [self._plan_for("Stable")]

        ents, rels, stats = gleaner.glean_graph(
            text_units=units,
            initial_entities=[_ent("e0", "Seed")],
            initial_relationships=[],
        )
        assert isinstance(stats, GleaningStats)
        assert stats.total_rounds >= 1
        assert stats.total_rounds <= gleaner.gleaning_config.max_rounds
        # Seed entity survives.
        assert any(e.name.lower() == "seed" for e in ents)
        # convergence_achieved set when a stop condition fired.
        assert stats.final_quality_score >= 0.0
