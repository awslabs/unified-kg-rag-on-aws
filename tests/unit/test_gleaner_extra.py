# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional AWS-free unit tests for GraphGleaner (complements
``test_gleaner_logic.py``).

Covers the branches the logic suite leaves uncovered: the module-level
``prepare_input_task`` / ``format_relationships_with_limit_task`` pure helpers,
refinement-output parsing and issue dispatch, the completion summary, the
stop rule, and the end-to-end
``glean_graph`` / ``_perform_llm_refinement`` orchestration driven by a mocked
LangChain chain (no Bedrock, no boto3). The Bedrock/boto wiring is patched out
in the ``gleaner`` fixture exactly as in the logic suite.
"""

from __future__ import annotations

import logging

import pytest

import unified_kg_rag.adapters.ingestion.gleaner as gleaner_module
from unified_kg_rag.adapters import providers as providers_module
from unified_kg_rag.adapters.ingestion.gleaner import (
    GleaningRound,
    GleaningStats,
    GraphGleaner,
    format_relationships_with_limit_task,
    prepare_input_task,
)
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.models import Config, Entity, Relationship, TextUnit
from unified_kg_rag.shared.utils.langchain import BATCH_ITEM_FAILED

pytestmark = pytest.mark.unit


@pytest.fixture
def gleaner(config: Config, mocker) -> GraphGleaner:
    mocker.patch.object(gleaner_module, "boto3")
    mocker.patch.object(providers_module, "BedrockLanguageModelFactory")
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
# _parse_refinement_output + _process_issue
# --------------------------------------------------------------------------- #
class TestParseRefinementOutput:
    def test_empty_plan_returns_empties(self, gleaner) -> None:
        unit = TextUnit(id="t1", text="x")
        ents, rels = gleaner._parse_refinement_output({}, unit, [])
        assert ents == [] and rels == []

    def test_non_dict_plan_returns_empties(self, gleaner) -> None:
        unit = TextUnit(id="t1", text="x")
        ents, rels = gleaner._parse_refinement_output("not a dict", unit, [])
        assert ents == [] and rels == []

    def test_list_plan_first_element_used(self, gleaner) -> None:
        unit = TextUnit(id="t1", text="x")
        plan = {
            "identified_issues": {
                "issue": {
                    "issue_type": "MISSING_ENTITY",
                    "details": {"name": "Neo", "type": "PERSON"},
                }
            },
        }
        ents, rels = gleaner._parse_refinement_output([plan], unit, [])
        assert len(ents) == 1
        assert ents[0].name.lower().startswith("neo")

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
        ents, rels = gleaner._parse_refinement_output(plan, unit, [])
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
        ents, rels = gleaner._parse_refinement_output(plan, unit, [])
        assert ents == [] and rels == []

    def test_issues_as_bare_list_not_dict(self, gleaner) -> None:
        unit = TextUnit(id="t1", text="x")
        # identified_issues is a list (not a dict wrapper) -> ensure_list path.
        plan = {
            "identified_issues": [
                {"issue_type": "MISSING_ENTITY", "details": {"name": "Zed"}}
            ]
        }
        ents, _ = gleaner._parse_refinement_output(plan, unit, [])
        assert len(ents) == 1


# --------------------------------------------------------------------------- #
# ENTITY_CORRECTION / RELATIONSHIP_CORRECTION issues
# --------------------------------------------------------------------------- #
class TestCorrectionIssues:
    """Corrections the refinement prompt asks for must reach the graph.

    ``GraphRefinementPrompt`` instructs the model to emit ENTITY_CORRECTION and
    RELATIONSHIP_CORRECTION issues; the dispatch branched only on the two
    MISSING_* types, so every correction was discarded for every chunk of every
    run. Corrections mutate the entity/relationship the round already carries,
    so these tests assert on the input objects.
    """

    UNIT = TextUnit(id="t1", text="acme corp is a company based in seattle.")

    @staticmethod
    def _plan(*issues: dict) -> dict:
        return {"identified_issues": {"issue": list(issues)}}

    @staticmethod
    def _entity_correction(**details: str) -> dict:
        return {"issue_type": "ENTITY_CORRECTION", "details": details}

    @staticmethod
    def _relationship_correction(**details: str) -> dict:
        return {"issue_type": "RELATIONSHIP_CORRECTION", "details": details}

    def test_entity_type_correction_applied(self, gleaner) -> None:
        entity = _ent("e1", "acme corp")
        entity.type = "PERSON"
        ents, rels = gleaner._parse_refinement_output(
            self._plan(
                self._entity_correction(name="acme corp", corrected_type="ORGANIZATION")
            ),
            self.UNIT,
            [entity],
        )
        assert entity.type == "ORGANIZATION"
        # A correction is not a new artifact.
        assert ents == [] and rels == []

    def test_entity_rename_keeps_display_form_and_the_id(self, gleaner) -> None:
        entity = _ent("e1", "acme")
        entity.name_embedding = [0.1, 0.2]
        gleaner._parse_refinement_output(
            self._plan(
                self._entity_correction(
                    name="acme", corrected_name="  Acme   Corporation "
                )
            ),
            self.UNIT,
            [entity],
        )
        # Extracted names keep their display form (whitespace cleaned only), so
        # a corrected one is cleaned the same way.
        assert entity.name == "Acme Corporation"
        # Relationships reference the id; re-deriving it would orphan them.
        assert entity.id == "e1"
        # The stored embedding described the old name.
        assert entity.name_embedding is None

    def test_entity_correction_for_unknown_name_is_dropped(self, gleaner) -> None:
        entity = _ent("e1", "acme corp")
        entity.type = "PERSON"
        gleaner._parse_refinement_output(
            self._plan(
                self._entity_correction(
                    name="never extracted", corrected_type="ORGANIZATION"
                )
            ),
            self.UNIT,
            [entity],
        )
        assert entity.type == "PERSON"

    def test_relationship_type_correction_applied(self, gleaner) -> None:
        rel = Relationship(
            id="r1",
            source_id="e1",
            target_id="e2",
            source_name="acme corp",
            target_name="seattle",
            type="EMPLOYS",
        )
        gleaner._parse_refinement_output(
            self._plan(
                self._relationship_correction(
                    source="acme corp",
                    target="seattle",
                    type="EMPLOYS",
                    corrected_type="LOCATED_IN",
                )
            ),
            self.UNIT,
            [],
            [rel],
        )
        assert rel.type == "LOCATED_IN"

    def test_reversed_direction_swaps_ids_and_names_together(self, gleaner) -> None:
        rel = Relationship(
            id="r1",
            source_id="e-seattle",
            target_id="e-acme",
            source_name="seattle",
            target_name="acme corp",
            type="LOCATED_IN",
        )
        gleaner._parse_refinement_output(
            self._plan(
                self._relationship_correction(
                    source="seattle",
                    target="acme corp",
                    corrected_source="acme corp",
                    corrected_target="seattle",
                )
            ),
            self.UNIT,
            [],
            [rel],
        )
        assert (rel.source_id, rel.source_name) == ("e-acme", "acme corp")
        assert (rel.target_id, rel.target_name) == ("e-seattle", "seattle")

    def test_relationship_correction_for_unknown_edge_is_dropped(self, gleaner) -> None:
        rel = Relationship(
            id="r1",
            source_id="e1",
            target_id="e2",
            source_name="acme corp",
            target_name="seattle",
            type="EMPLOYS",
        )
        gleaner._parse_refinement_output(
            self._plan(
                self._relationship_correction(
                    source="nobody", target="nowhere", corrected_type="LOCATED_IN"
                )
            ),
            self.UNIT,
            [],
            [rel],
        )
        assert rel.type == "EMPLOYS"

    def test_ambiguous_pair_without_a_current_type_is_dropped(self, gleaner) -> None:
        # Two edges join the same pair; with no stated current type there is no
        # way to tell which one the correction means, so neither is touched.
        shared = {
            "source_id": "e1",
            "target_id": "e2",
            "source_name": "acme corp",
            "target_name": "seattle",
        }
        first = Relationship(id="r1", type="EMPLOYS", **shared)
        second = Relationship(id="r2", type="FOUNDED_IN", **shared)
        gleaner._parse_refinement_output(
            self._plan(
                self._relationship_correction(
                    source="acme corp", target="seattle", corrected_type="LOCATED_IN"
                )
            ),
            self.UNIT,
            [],
            [first, second],
        )
        assert (first.type, second.type) == ("EMPLOYS", "FOUNDED_IN")

    def test_current_type_disambiguates_a_multi_edge_pair(self, gleaner) -> None:
        shared = {
            "source_id": "e1",
            "target_id": "e2",
            "source_name": "acme corp",
            "target_name": "seattle",
        }
        first = Relationship(id="r1", type="EMPLOYS", **shared)
        second = Relationship(id="r2", type="FOUNDED_IN", **shared)
        gleaner._parse_refinement_output(
            self._plan(
                self._relationship_correction(
                    source="acme corp",
                    target="seattle",
                    type="FOUNDED_IN",
                    corrected_type="LOCATED_IN",
                )
            ),
            self.UNIT,
            [],
            [first, second],
        )
        assert (first.type, second.type) == ("EMPLOYS", "LOCATED_IN")

    def test_ungrounded_correction_dropped_when_grounding_enabled(
        self, gleaner
    ) -> None:
        # The hallucination guard covers corrections too, not just additions.
        gleaner.extraction_config.entity_grounding.enabled = True
        entity = _ent("e1", "acme corp")
        entity.type = "PERSON"
        issue = self._entity_correction(name="acme corp", corrected_type="ORGANIZATION")
        issue["text_evidence"] = "A clause that appears nowhere in the chunk."
        gleaner._parse_refinement_output(self._plan(issue), self.UNIT, [entity])
        assert entity.type == "PERSON"

    def test_correction_survives_a_full_gleaning_round(self, gleaner, mocker) -> None:
        # End-to-end through glean_graph: the corrected entity is the one the
        # stage returns, so the correction reaches the graph the pipeline indexes.
        seed = _ent("e1", "acme corp")
        seed.type = "PERSON"
        gleaner.graph_refiner = mocker.Mock()
        gleaner.graph_refiner.batch.return_value = [
            {
                "refinement_plan": {
                    "identified_issues": {
                        "issue": self._entity_correction(
                            name="acme corp", corrected_type="ORGANIZATION"
                        )
                    },
                }
            }
        ]

        entities, _, _ = gleaner.glean_graph(
            text_units=[self.UNIT],
            initial_entities=[seed],
            initial_relationships=[],
        )
        corrected = next(e for e in entities if e.name == "acme corp")
        # The round's duplicate merge lowercases the winning type.
        assert (corrected.type or "").upper() == "ORGANIZATION"

    # -- endpoint names after a rename ------------------------------------- #
    # An edge carries a denormalised copy of each endpoint's name, and the
    # indexer writes that copy rather than resolving the id. Renaming an entity
    # in place therefore has to be followed by rewriting the copies, or the
    # graph the stage hands on names one thing on the node and another on the
    # edge. These run the whole round, with only the model's answer mocked.

    def _glean_one_round(
        self,
        gleaner,
        mocker,
        entities: list[Entity],
        relationships: list[Relationship],
        *issues: dict,
    ) -> tuple[list[Entity], list[Relationship]]:
        gleaner.graph_refiner = mocker.Mock()
        gleaner.graph_refiner.batch.return_value = [
            {
                "refinement_plan": {
                    "identified_issues": {"issue": list(issues)},
                }
            }
        ]
        entities, relationships, _ = gleaner.glean_graph(
            text_units=[self.UNIT],
            initial_entities=entities,
            initial_relationships=relationships,
        )
        return entities, relationships

    @staticmethod
    def _assert_indexed_endpoint_names_match_entities(
        entities: list[Entity], relationships: list[Relationship]
    ) -> None:
        # Assert on the document the indexer builds, since that is where the
        # denormalised name is read, not on the model field alone.
        name_by_id = {e.id: e.name for e in entities}
        for rel in relationships:
            doc = OpenSearchIndexer._prepare_relationship_doc(rel, ([],))
            assert doc["source_name"] == name_by_id[rel.source_id]
            assert doc["target_name"] == name_by_id[rel.target_id]

    def test_rename_resyncs_the_source_endpoint_name_in_a_full_round(
        self, gleaner, mocker
    ) -> None:
        edge = Relationship(
            id="r1",
            source_id="e1",
            target_id="e2",
            source_name="acme",
            target_name="seattle",
            type="LOCATED_IN",
        )
        entities, relationships = self._glean_one_round(
            gleaner,
            mocker,
            [_ent("e1", "acme"), _ent("e2", "seattle")],
            [edge],
            self._entity_correction(name="acme", corrected_name="Acme Corporation"),
        )
        renamed = next(e for e in entities if e.id == "e1")
        assert renamed.name != "acme"
        (out,) = relationships
        # The id was kept, so the edge is still attached; the name must follow.
        assert (out.source_id, out.target_id) == ("e1", "e2")
        assert out.source_name == renamed.name
        assert out.target_name == "seattle"
        self._assert_indexed_endpoint_names_match_entities(entities, relationships)

    def test_rename_resyncs_the_target_endpoint_name_in_a_full_round(
        self, gleaner, mocker
    ) -> None:
        edge = Relationship(
            id="r1",
            source_id="e1",
            target_id="e2",
            source_name="jane roe",
            target_name="acme",
            type="WORKS_FOR",
        )
        entities, relationships = self._glean_one_round(
            gleaner,
            mocker,
            [_ent("e1", "jane roe"), _ent("e2", "acme")],
            [edge],
            self._entity_correction(name="acme", corrected_name="Acme Corporation"),
        )
        renamed = next(e for e in entities if e.id == "e2")
        (out,) = relationships
        assert (out.source_id, out.target_id) == ("e1", "e2")
        assert out.source_name == "jane roe"
        assert out.target_name == renamed.name
        self._assert_indexed_endpoint_names_match_entities(entities, relationships)

    def test_rename_that_merges_into_an_existing_entity_resyncs_the_remapped_edge(
        self, gleaner, mocker
    ) -> None:
        # A rename can make two entities share a name, at which point the round's
        # duplicate merge re-points every edge at the surviving id. The edge must
        # then carry the survivor's name, not the one it was extracted under.
        canonical = _ent("e1", "acme corporation")
        edge = Relationship(
            id="r1",
            source_id="e2",
            target_id="e3",
            source_name="acme",
            target_name="seattle",
            type="LOCATED_IN",
        )
        entities, relationships = self._glean_one_round(
            gleaner,
            mocker,
            [canonical, _ent("e2", "acme"), _ent("e3", "seattle")],
            [edge],
            self._entity_correction(name="acme", corrected_name="Acme Corporation"),
        )
        assert {e.id for e in entities} == {"e1", "e3"}
        (out,) = relationships
        assert (out.source_id, out.target_id) == ("e1", "e3")
        assert out.source_name == canonical.name
        self._assert_indexed_endpoint_names_match_entities(entities, relationships)

    def test_reversed_edge_takes_the_renamed_endpoint_on_its_new_side(
        self, gleaner, mocker
    ) -> None:
        # Reversal swaps ids and names together; a rename in the same round must
        # then land on the side the swapped id now occupies, not the old one.
        edge = Relationship(
            id="r1",
            source_id="e-seattle",
            target_id="e-acme",
            source_name="seattle",
            target_name="acme corp",
            type="LOCATED_IN",
        )
        entities, relationships = self._glean_one_round(
            gleaner,
            mocker,
            [_ent("e-acme", "acme corp"), _ent("e-seattle", "seattle")],
            [edge],
            self._relationship_correction(
                source="seattle",
                target="acme corp",
                corrected_source="acme corp",
                corrected_target="seattle",
            ),
            self._entity_correction(
                name="acme corp", corrected_name="Acme Corporation"
            ),
        )
        renamed = next(e for e in entities if e.id == "e-acme")
        assert renamed.name != "acme corp"
        (out,) = relationships
        assert (out.source_id, out.source_name) == ("e-acme", renamed.name)
        assert (out.target_id, out.target_name) == ("e-seattle", "seattle")
        self._assert_indexed_endpoint_names_match_entities(entities, relationships)


# --------------------------------------------------------------------------- #
# _log_completion_summary
# --------------------------------------------------------------------------- #
class TestLogCompletionSummary:
    @staticmethod
    def _round(units_gleaned: int) -> GleaningRound:
        return GleaningRound(
            round_number=1,
            units_gleaned=units_gleaned,
            units_gained=0,
            entities_before=0,
            relationships_before=0,
            entities_added=1,
            relationships_added=1,
            processing_time=0.5,
        )

    def test_summary_reports_refinement_calls(self, caplog) -> None:
        stats = GleaningStats(total_rounds=2, rounds=[self._round(4), self._round(1)])
        with caplog.at_level(logging.INFO):
            GraphGleaner._log_completion_summary(stats, units_still_gaining=0)
        assert stats.total_refinement_calls == 5
        assert "2 rounds, 5 refinement calls" in caplog.text
        assert "still gaining" not in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_summary_notes_units_cut_off_by_max_rounds(self, caplog) -> None:
        with caplog.at_level(logging.INFO):
            GraphGleaner._log_completion_summary(
                GleaningStats(num_failed_units=1), units_still_gaining=3
            )
        assert "3 text units still gaining" in caplog.text
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1


# --------------------------------------------------------------------------- #
# _perform_llm_refinement + glean_graph (mocked chain, no Bedrock)
# --------------------------------------------------------------------------- #
def _missing_entity_plan(name: str | None) -> dict:
    """A refinement answer adding entity ``name``, or an empty answer for None."""
    issues = (
        {
            "issue": {
                "issue_type": "MISSING_ENTITY",
                "details": {"name": name, "type": "PERSON"},
            }
        }
        if name
        else {}
    )
    return {"refinement_plan": {"identified_issues": issues}}


class TestGleanGraphOrchestration:
    def test_perform_llm_refinement_collects_new_entities(
        self, gleaner, mocker
    ) -> None:
        units = [TextUnit(id="t1", text="a"), TextUnit(id="t2", text="b")]
        # batch returns one refinement plan per input, in order.
        gleaner.graph_refiner = mocker.Mock()
        gleaner.graph_refiner.batch.return_value = [
            _missing_entity_plan("Alpha"),
            _missing_entity_plan("Beta"),
        ]
        new_e, new_r = gleaner._perform_llm_refinement(units, [], [])
        assert {e.name.lower() for e in new_e} == {"alpha", "beta"}
        assert new_r == []

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
        new_e, new_r = gleaner._perform_llm_refinement(
            [TextUnit(id="t1", text="a")], [], []
        )
        assert new_e == [] and new_r == []
        assert gleaner._failed_unit_ids == ["t1"]

    def test_glean_graph_counts_failed_unit_refinements(self, gleaner, mocker) -> None:
        units = [TextUnit(id="t1", text="a"), TextUnit(id="t2", text="b")]
        gleaner.gleaning_config.max_rounds = 1
        gleaner.graph_refiner = mocker.Mock()
        gleaner.batch_processor = mocker.Mock()
        # execute_with_fallback marks an item that failed every retry.
        gleaner.batch_processor.execute_with_fallback.return_value = [
            _missing_entity_plan("Alpha"),
            BATCH_ITEM_FAILED,
        ]

        _, _, stats = gleaner.glean_graph(units, [], [])

        assert stats.num_failed_units == 1
        assert stats.failed_text_unit_ids == ["t2"]

    def test_failed_unit_is_not_regleaned_and_failures_sum_over_rounds(
        self, gleaner, mocker
    ) -> None:
        gleaner.gleaning_config.max_rounds = 3
        units = [TextUnit(id="t1", text="a"), TextUnit(id="t2", text="b")]
        answers = iter(
            [
                [_missing_entity_plan("Alpha"), BATCH_ITEM_FAILED],
                [BATCH_ITEM_FAILED],
            ]
        )
        gleaner.graph_refiner = mocker.Mock()
        gleaner.batch_processor = mocker.Mock()
        gleaner.batch_processor.execute_with_fallback.side_effect = (
            lambda **kwargs: next(answers)
        )

        _, _, stats = gleaner.glean_graph(units, [], [])

        # Round 1: t2 fails, t1 gains. Round 2: only t1, which fails -> stop.
        assert [r.units_gleaned for r in stats.rounds] == [2, 1]
        assert stats.num_failed_units == 2
        assert stats.failed_text_unit_ids == ["t1", "t2"]

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


# --------------------------------------------------------------------------- #
# Stop rule: at most max_rounds per unit, re-glean only units that gained
# --------------------------------------------------------------------------- #
class TestStopRule:
    @staticmethod
    def _run(gleaner, mocker, answer, units, entities=(), relationships=()):
        """Run glean_graph with ``answer(round, unit_index)`` as the model."""
        sent: list[int] = []

        def _batch(inputs, *args, **kwargs):
            sent.append(len(inputs))
            return [answer(len(sent), i) for i in range(len(inputs))]

        gleaner.graph_refiner = mocker.Mock()
        gleaner.graph_refiner.batch.side_effect = _batch
        _, _, stats = gleaner.glean_graph(units, list(entities), list(relationships))
        return sent, stats

    def test_stops_at_max_rounds_while_units_keep_gaining(
        self, gleaner, mocker
    ) -> None:
        gleaner.gleaning_config.max_rounds = 3
        sent, stats = self._run(
            gleaner,
            mocker,
            lambda round_num, _: _missing_entity_plan(f"New{round_num}"),
            [TextUnit(id="t1", text="a")],
        )
        assert sent == [1, 1, 1]
        assert stats.total_rounds == 3
        assert stats.total_entities_added == 3
        assert stats.total_refinement_calls == 3

    def test_second_round_sends_only_units_with_new_items(
        self, gleaner, mocker
    ) -> None:
        gleaner.gleaning_config.max_rounds = 3
        units = [TextUnit(id="t1", text="a"), TextUnit(id="t2", text="b")]

        def _answer(round_num: int, index: int) -> dict:
            # Round 1: only t1 adds an entity; nobody adds anything afterwards.
            return _missing_entity_plan(
                "Alpha" if round_num == 1 and index == 0 else None
            )

        sent, stats = self._run(gleaner, mocker, _answer, units)

        assert sent == [2, 1]
        assert [r.units_gained for r in stats.rounds] == [1, 0]
        assert stats.total_rounds == 2

    def test_empty_answer_stops_the_unit(self, gleaner, mocker) -> None:
        # An empty <identified_issues> is the model saying nothing is missing.
        gleaner.gleaning_config.max_rounds = 3
        sent, stats = self._run(
            gleaner,
            mocker,
            lambda *_: _missing_entity_plan(None),
            [TextUnit(id="t1", text="a")],
        )
        assert sent == [1]
        assert stats.total_rounds == 1
        assert stats.total_entities_added == 0

    def test_reproposing_a_known_entity_is_not_a_gain(self, gleaner, mocker) -> None:
        # The answer names an entity the graph already has (it merges away), so
        # the unit gained nothing and is not re-sent.
        gleaner.gleaning_config.max_rounds = 3
        seed = _ent("e0", "Stable", text_unit_ids=["t1"])
        sent, stats = self._run(
            gleaner,
            mocker,
            lambda *_: _missing_entity_plan("stable"),
            [TextUnit(id="t1", text="a")],
            entities=[seed],
        )
        assert sent == [1]
        assert stats.total_entities_added == 0

    def test_new_relationship_alone_is_a_gain(self, gleaner, mocker) -> None:
        gleaner.gleaning_config.max_rounds = 2
        alice = _ent("e-alice", "Alice", text_unit_ids=["t1"])
        acme = _ent("e-acme", "Acme", text_unit_ids=["t1"])

        def _answer(round_num: int, _: int) -> dict:
            if round_num > 1:
                return _missing_entity_plan(None)
            return {
                "refinement_plan": {
                    "identified_issues": {
                        "issue": {
                            "issue_type": "MISSING_RELATIONSHIP",
                            "details": {
                                "source": "Alice",
                                "target": "Acme",
                                "type": "WORKS_AT",
                            },
                        }
                    }
                }
            }

        sent, stats = self._run(
            gleaner,
            mocker,
            _answer,
            [TextUnit(id="t1", text="a")],
            entities=[alice, acme],
        )
        assert sent == [1, 1]
        assert stats.total_relationships_added == 1


def test_default_gleaning_keeps_three_rounds() -> None:
    # Later rounds recover entities the first round misses, which multi-hop
    # strategies such as DRIFT depend on; they stay cheap because they only
    # re-send units that gained items.
    from unified_kg_rag.domain.models import Config

    assert Config().processing.gleaning.max_rounds == 3
