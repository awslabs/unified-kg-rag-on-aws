# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A gleaning correction keeps ids unique after a re-proposal of the old item.

An ENTITY_CORRECTION renames an entity and keeps its id. A later unit that
proposes the old name again carries the same name-derived id. Merging by name
alone kept both, so one id named two entities (two Neptune vertices).

A RELATIONSHIP_CORRECTION re-keys the edge to the id of its corrected
(source, target, type), so the uncorrected edge another unit states is a
distinct edge with its own id, as it is when an incremental run indexes that
unit later.
"""

from __future__ import annotations

from collections import Counter

import pytest

import unified_kg_rag.adapters.ingestion.gleaner as gleaner_module
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.ingestion.gleaner import GleaningIndex, GraphGleaner
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.graph_resolver import GraphResolver
from unified_kg_rag.domain.models import Config, Entity, Relationship, TextUnit

pytestmark = pytest.mark.unit

TYPO = BaseProcessor._generate_entity_id("Acme Crop")
DEPOT = BaseProcessor._generate_entity_id("Depot")
EDGE = BaseProcessor._generate_relationship_id("Acme Crop", "Depot", "SUPPLIES")
CORRECTED_EDGE = BaseProcessor._generate_relationship_id(
    "Acme Crop", "Depot", "SHIPS_TO"
)
U1 = TextUnit(id="u1", text="Acme Corp supplies the Depot.")


@pytest.fixture
def gleaner(config: Config, mocker) -> GraphGleaner:
    mocker.patch.object(bedrock_module, "BedrockLanguageModelFactory")
    mocker.patch.object(gleaner_module, "create_robust_xml_output_parser")
    mocker.patch.object(gleaner_module, "setup_chain")
    return GraphGleaner(config)


def _entity(id_: str, name: str, unit: str) -> Entity:
    return Entity(id=id_, name=name, type="ORG", text_unit_ids=[unit])


def _edge(type_: str, unit: str) -> Relationship:
    return Relationship(
        id=EDGE,
        source_id=TYPO,
        source_name="Acme Crop",
        target_id=DEPOT,
        target_name="Depot",
        type=type_,
        text_unit_ids=[unit],
    )


def _corrected_round(gleaner: GraphGleaner):
    """u1 corrects the name and the edge type; u2 re-proposes both old forms."""
    entities = [_entity(TYPO, "Acme Crop", "u1"), _entity(DEPOT, "Depot", "u1")]
    relationships = [_edge("SUPPLIES", "u1")]
    index = GleaningIndex(entities, relationships)
    for issue in (
        {
            "issue_type": "ENTITY_CORRECTION",
            "details": {"name": "Acme Crop", "corrected_name": "Acme Corp"},
        },
        {
            "issue_type": "RELATIONSHIP_CORRECTION",
            "details": {
                "source": "Acme Crop",
                "target": "Depot",
                "type": "SUPPLIES",
                "corrected_type": "SHIPS_TO",
            },
        },
    ):
        gleaner._process_issue(issue, U1, index, [], [])
    assert entities[0].name == "Acme Corp" and entities[0].id == TYPO
    assert relationships[0].type == "SHIPS_TO"
    assert relationships[0].id == CORRECTED_EDGE

    proposed_entities = [_entity(TYPO, "Acme Crop", "u2")]
    proposed_relationships = [_edge("SUPPLIES", "u2")]
    return (
        entities + proposed_entities,
        relationships + proposed_relationships,
    )


def _assert_unique_ids(entities: list[Entity], relationships: list[Relationship]):
    entity_ids = Counter(e.id for e in entities)
    relationship_ids = Counter(r.id for r in relationships)
    assert [i for i, n in entity_ids.items() if n > 1] == []
    assert [i for i, n in relationship_ids.items() if n > 1] == []


def test_merge_round_keeps_ids_unique_after_corrections(
    gleaner,
) -> None:
    entities, relationships = _corrected_round(gleaner)

    merged_entities, merged_relationships = GraphGleaner._merge_round(
        entities, relationships
    )

    _assert_unique_ids(merged_entities, merged_relationships)
    corrected = next(e for e in merged_entities if e.id == TYPO)
    assert corrected.name == "Acme Corp"
    assert corrected.text_unit_ids == ["u1", "u2"]
    edges = {r.id: r for r in merged_relationships}
    assert set(edges) == {CORRECTED_EDGE, EDGE}
    assert edges[CORRECTED_EDGE].type == "SHIPS_TO"
    assert edges[CORRECTED_EDGE].source_name == "Acme Corp"
    assert edges[CORRECTED_EDGE].text_unit_ids == ["u1"]
    assert edges[EDGE].type == "SUPPLIES"
    assert edges[EDGE].text_unit_ids == ["u2"]


@pytest.mark.parametrize("fuzzy", [False, True])
def test_resolver_returns_unique_ids_for_unmerged_corrections(gleaner, fuzzy) -> None:
    # Resolution must hold the invariant on its own: a cached gleaning output
    # (or any other producer) can carry two items with one id. A correction
    # cached before corrections re-keyed edges kept the original edge id.
    entities, relationships = _corrected_round(gleaner)
    relationships[0].id = EDGE
    config = Config()
    config.processing.fuzzy_entity_resolution = fuzzy

    result, _ = GraphResolver(config, use_process_pool=False).resolve_graph(
        entities, relationships
    )

    _assert_unique_ids(result["entities"], result["relationships"])
    assert {e.id for e in result["entities"]} == {TYPO, DEPOT}
    (edge,) = result["relationships"]
    assert sorted(edge.text_unit_ids or []) == ["u1", "u2"]


def test_resolver_joins_same_id_entities_of_conflicting_types() -> None:
    # One id is one vertex, so a type conflict cannot keep them apart.
    entities = [
        Entity(id="e1", name="Acme Corp", type="ORG", text_unit_ids=["u1"]),
        Entity(id="e1", name="Acme Crop", type="PERSON", text_unit_ids=["u2"]),
        Entity(id="e2", name="Depot", type="ORG", text_unit_ids=["u1"]),
    ]

    result, _ = GraphResolver(Config(), use_process_pool=False).resolve_graph(
        entities, []
    )

    _assert_unique_ids(result["entities"], result["relationships"])
    assert len(result["entities"]) == 2


def test_a_correction_splits_off_its_own_units_share(gleaner) -> None:
    # Extraction merged u1's and u2's statements of the edge before gleaning.
    # u1 reversing it must leave u2's statement as u2 made it: an incremental
    # run that extracts u2 in another run (or deletes u1's document) has only
    # u1's statement to correct.
    edge = _edge("SUPPLIES", "u1")
    edge.text_unit_ids = ["u1", "u2"]
    edge.weight = 8.0
    edge.attributes = {
        "text_unit_weights": [3.0, 5.0],
        "filters": {"doc": ["a", "b"]},
        "kept": True,
    }
    entities = [_entity(TYPO, "Acme Crop", "u1"), _entity(DEPOT, "Depot", "u1")]
    relationships = [edge]
    index = GleaningIndex(entities, relationships)
    unit = TextUnit(id="u1", text=U1.text, attributes={"filters": {"doc": "a"}})
    reverse = {
        "issue_type": "RELATIONSHIP_CORRECTION",
        "details": {
            "source": "Acme Crop",
            "target": "Depot",
            "corrected_source": "Depot",
            "corrected_target": "Acme Crop",
        },
    }
    added: list[Relationship] = []

    gleaner._process_issue(reverse, unit, index, [], added)

    # A correction is not a new relationship (a gain would re-glean the unit).
    assert added == []
    original, copy = relationships
    assert (original.id, original.source_name, original.type) == (
        EDGE,
        "Acme Crop",
        "SUPPLIES",
    )
    assert (original.text_unit_ids, original.weight) == (["u2"], 5.0)
    assert original.attributes["filters"] == {"doc": ["a", "b"]}
    assert copy.id == BaseProcessor._generate_relationship_id(
        "Depot", "Acme Crop", "SUPPLIES"
    )
    assert (copy.source_name, copy.target_name) == ("Depot", "Acme Crop")
    assert (copy.text_unit_ids, copy.weight) == (["u1"], 3.0)
    assert copy.attributes == {
        "text_unit_weights": [3.0],
        "filters": {"doc": "a"},
        "kept": True,
    }
    # The unit's prompt and later issues see its own (corrected) edge.
    assert index.relationships_for("u1") == [copy]
    assert index.relationships_for("u2") == [original]


def test_a_correction_of_an_edge_only_its_unit_states_is_in_place(gleaner) -> None:
    edge = _edge("SUPPLIES", "u1")
    relationships = [edge]
    index = GleaningIndex(
        [_entity(TYPO, "Acme Crop", "u1"), _entity(DEPOT, "Depot", "u1")],
        relationships,
    )
    retype = {
        "issue_type": "RELATIONSHIP_CORRECTION",
        "details": {
            "source": "Acme Crop",
            "target": "Depot",
            "corrected_type": "SHIPS_TO",
        },
    }

    gleaner._process_issue(retype, U1, index, [], [])

    assert relationships == [edge]
    assert (edge.id, edge.type) == (CORRECTED_EDGE, "SHIPS_TO")
