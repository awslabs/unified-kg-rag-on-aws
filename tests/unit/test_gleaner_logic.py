# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GraphGleaner pure logic (AWS-free).

Covers the post-merge relationship reconciliation (orphan/self-loop drop
counting, weight summing), the duplicate-entity merge, gleaner grounding, and
the module-level format_entities_with_limit_task. The
static methods are called directly; instance methods are exercised on a real
GraphGleaner whose Bedrock/boto wiring is patched out.
"""

from __future__ import annotations

import pytest

import unified_kg_rag.adapters.ingestion.gleaner as gleaner_module
from unified_kg_rag.adapters import providers as providers_module
from unified_kg_rag.adapters.ingestion.gleaner import (
    GraphGleaner,
    format_entities_with_limit_task,
)
from unified_kg_rag.domain.models import Config, Entity, Relationship

pytestmark = pytest.mark.unit


@pytest.fixture
def gleaner(config: Config, mocker) -> GraphGleaner:
    mocker.patch.object(gleaner_module, "boto3")
    mocker.patch.object(providers_module, "BedrockLanguageModelFactory")
    mocker.patch.object(gleaner_module, "create_robust_xml_output_parser")
    mocker.patch.object(gleaner_module, "setup_chain")
    return GraphGleaner(config)


def _rel(id_, src, tgt, **kw) -> Relationship:
    return Relationship(id=id_, source_id=src, target_id=tgt, type="X", **kw)


# --------------------------------------------------------------------------- #
# _update_relationships_after_merge
# --------------------------------------------------------------------------- #
class TestUpdateRelationshipsAfterMerge:
    def test_orphan_endpoint_dropped(self) -> None:
        rels = [_rel("r1", "e1", "GONE")]
        out = GraphGleaner._update_relationships_after_merge(
            rels, unique_entity_ids={"e1"}, id_remap={}
        )
        assert out == []  # target not in unique set -> dropped

    def test_self_loop_dropped(self) -> None:
        rels = [_rel("r1", "e1", "e1")]
        out = GraphGleaner._update_relationships_after_merge(
            rels, unique_entity_ids={"e1"}, id_remap={}
        )
        assert out == []

    def test_self_loop_after_remap_dropped(self) -> None:
        # e2 remaps to e1, collapsing the edge into a self-loop -> dropped.
        rels = [_rel("r1", "e1", "e2")]
        out = GraphGleaner._update_relationships_after_merge(
            rels, unique_entity_ids={"e1"}, id_remap={"e2": "e1"}
        )
        assert out == []

    def test_surviving_edge_kept_with_remapped_endpoint(self) -> None:
        rels = [_rel("r1", "e9", "e2")]
        out = GraphGleaner._update_relationships_after_merge(
            rels, unique_entity_ids={"e1", "e2"}, id_remap={"e9": "e1"}
        )
        assert len(out) == 1
        assert out[0].source_id == "e1"
        assert out[0].target_id == "e2"

    def test_duplicate_edges_merged_and_weight_summed(self) -> None:
        rels = [
            _rel("r1", "e1", "e2", weight=1.0, description="a"),
            _rel("r2", "e1", "e2", weight=2.0, description="b"),
        ]
        out = GraphGleaner._update_relationships_after_merge(
            rels, unique_entity_ids={"e1", "e2"}, id_remap={}
        )
        assert len(out) == 1
        assert out[0].weight == 3.0
        assert "a" in out[0].description and "b" in out[0].description

    def test_distinct_types_not_merged(self) -> None:
        rels = [
            Relationship(id="r1", source_id="e1", target_id="e2", type="A"),
            Relationship(id="r2", source_id="e1", target_id="e2", type="B"),
        ]
        out = GraphGleaner._update_relationships_after_merge(
            rels, unique_entity_ids={"e1", "e2"}, id_remap={}
        )
        assert len(out) == 2


# --------------------------------------------------------------------------- #
# _merge_duplicate_entities
# --------------------------------------------------------------------------- #
class TestMergeDuplicateEntities:
    def test_name_keyed_dedup_with_id_remap(self) -> None:
        ents = [
            Entity(id="e1", name="Alice", type="PERSON", text_unit_ids=["t1"]),
            Entity(id="e9", name="alice", type="PERSON", text_unit_ids=["t2"]),
        ]
        unique, remap = GraphGleaner._merge_duplicate_entities(ents)
        assert len(unique) == 1
        assert remap == {"e9": "e1"}  # e9 merged into master e1
        assert set(unique[0].text_unit_ids) == {"t1", "t2"}

    def test_most_frequent_type_wins(self) -> None:
        ents = [
            Entity(id="e1", name="A", type="PERSON"),
            Entity(id="e2", name="A", type="PERSON"),
            Entity(id="e3", name="A", type="ORG"),
        ]
        unique, _ = GraphGleaner._merge_duplicate_entities(ents)
        assert len(unique) == 1
        assert unique[0].type == "person"  # lowercased, most frequent


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
from unified_kg_rag.domain.models import TextUnit  # noqa: E402


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
            [],
            new_e,
            new_r,
        )
        assert len(new_e) == 1

    def test_grounded_addition_kept(self, gleaner) -> None:
        gleaner.extraction_config.entity_grounding.enabled = True
        new_e, new_r = [], []
        gleaner._process_issue(
            self._missing_entity_issue("Alice works at Acme Corp in Seattle."),
            self.UNIT,
            [],
            new_e,
            new_r,
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
            [],
            new_e,
            new_r,
        )
        assert new_e == []  # hallucinated gleaner entity rejected

    def test_source_text_not_persisted_on_kept_entity(self, gleaner) -> None:
        gleaner.extraction_config.entity_grounding.enabled = True
        new_e, new_r = [], []
        gleaner._process_issue(
            self._missing_entity_issue("Alice works at Acme Corp in Seattle."),
            self.UNIT,
            [],
            new_e,
            new_r,
        )
        assert "_source_text" not in (new_e[0].attributes or {})
