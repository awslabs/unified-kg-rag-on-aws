# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed neutrality: a seed graph enriches the entities documents state and
never changes what they decide.

Over random document records and seed records (seed nodes reuse document
names, with other types, confidences, descriptions, IRIs and properties),
under exact resolution (names that resolve only to their own key):

- every entity some document states keeps the name, id, type (unless no
  document gave one) and confidence of the document-only build, and its
  description starts with the document-only one (an endpoint stub keeps
  only the description);
- an extracted attribute is never overwritten by a seed property;
- removing the seed units from the built graph and deriving it from the
  records left gives the document-only graph;
- ranks count document-backed edges only: every entity and edge of the
  document-only graph keeps its rank there, in a full build and a
  derivation, and an entity only a seed states ranks 0.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.contributions import merge_contributions
from unified_kg_rag.domain.ingestion.derivation import derive
from unified_kg_rag.domain.ingestion.graph_resolver import GraphResolver
from unified_kg_rag.domain.ingestion.merge.merger import remove_text_units
from unified_kg_rag.domain.ingestion.relationship_weights import (
    apply_text_unit_weights,
)
from unified_kg_rag.domain.ingestion.seed_graph import (
    SeedEdgeStatement,
    SeedNodeStatement,
    is_seed_unit,
    parse_seed_unit,
    render_unit,
    seed_only,
    seed_unit_id,
)
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.shared.utils import generate_stable_id

pytestmark = pytest.mark.property

# Names whose keys differ and that no compact key or legal form joins:
# exact resolution. "ACME" is "Acme"'s key in another case.
_DOCUMENT_NAMES = ["Acme", "ACME", "Globex", "Initech", "Hooli"]
_SEED_ONLY_NAMES = ["Umbrella", "Soylent"]
_TYPES = ["VENDOR", "BUYER", "vendor", ""]
_EX = "https://example.org/procure#"

_unit_ids = st.integers(min_value=0, max_value=5)


def _id(name: str) -> str:
    return BaseProcessor._generate_entity_id(name)


def _edge(source: str, target: str, unit: str, weight: float) -> Relationship:
    edge = Relationship(
        id=BaseProcessor._generate_relationship_id(source, target, "SUPPLIES"),
        source_id=_id(source),
        target_id=_id(target),
        source_name=source,
        target_name=target,
        type="SUPPLIES",
        description=f"{source} supplies {target} ({unit[:4]})",
        weight=weight,
        text_unit_ids=[unit],
    )
    apply_text_unit_weights(edge, {unit: weight})
    return edge


@st.composite
def _document_records(draw: st.DrawFn) -> list[TextUnitExtraction]:
    records = []
    for n in sorted(draw(st.sets(_unit_ids, min_size=1, max_size=4))):
        unit = generate_stable_id(f"text_unit:doc:{n}")
        names = draw(st.lists(st.sampled_from(_DOCUMENT_NAMES), max_size=3))
        entities = [
            Entity(
                id=_id(name),
                name=name,
                type=draw(st.sampled_from(_TYPES)),
                description=draw(st.sampled_from(["", f"{name} line {n}"])),
                confidence=draw(st.sampled_from([0.2, 0.5, 0.8])),
                attributes=draw(
                    st.sampled_from([None, {"country": "FR"}, {"size": "large"}])
                ),
                text_unit_ids=[unit],
            )
            for name in names
        ]
        pairs = draw(
            st.lists(
                st.tuples(
                    st.sampled_from(_DOCUMENT_NAMES), st.sampled_from(_DOCUMENT_NAMES)
                ),
                max_size=2,
            )
        )
        edges = [
            _edge(source, target, unit, 2.0)
            for source, target in pairs
            if _id(source) != _id(target)
        ]
        records.append(
            TextUnitExtraction(
                text_unit_id=unit, entities=entities, relationships=edges
            )
        )
    return records


@st.composite
def _seed_records(draw: st.DrawFn) -> list[TextUnitExtraction]:
    records = []
    names = _DOCUMENT_NAMES + _SEED_ONLY_NAMES
    for n in sorted(draw(st.sets(_unit_ids, max_size=3))):
        nodes = [
            SeedNodeStatement(
                id=f"n{n}-{i}",
                name=name,
                type=draw(st.sampled_from(["BUYER", "PRODUCT", None])),
                iri=draw(st.sampled_from([None, _EX + name.lower(), _EX + "other"])),
                description=draw(st.sampled_from([None, f"seed fact {n}"])),
                confidence=draw(st.sampled_from([0.1, 1.0])),
                properties=draw(st.sampled_from([{}, {"country": "DE"}, {"size": 3}])),
            )
            for i, name in enumerate(
                draw(st.lists(st.sampled_from(names), max_size=3, unique=True))
            )
        ]
        edges = [
            SeedEdgeStatement(
                source=source,
                target=target,
                type="SUPPLIES",
                description=f"seed edge {n}",
                weight=5.0,
            )
            for source, target in draw(
                st.lists(st.tuples(st.sampled_from(names), st.sampled_from(names)))
            )
            if _id(source) != _id(target)
        ][:2]
        if not nodes and not edges:
            continue
        unit = TextUnit(
            id=seed_unit_id("seed-doc", n), text=render_unit([*nodes, *edges])
        )
        records.append(parse_seed_unit(unit))
    return records


def _build(
    records: list[TextUnitExtraction],
) -> tuple[dict[str, Entity], dict[str, Relationship]]:
    entities, relationships = merge_contributions(records)
    resolver = GraphResolver(Config(), max_workers=1, use_process_pool=False)
    graph, _ = resolver.resolve_graph(entities, relationships)
    return (
        {e.id: e for e in graph["entities"]},
        {r.id: r for r in graph["relationships"]},
    )


@given(documents=_document_records(), seeds=_seed_records())
def test_seeds_never_change_what_documents_decide(
    documents: list[TextUnitExtraction], seeds: list[TextUnitExtraction]
) -> None:
    document_only, _ = _build(documents)
    seeded, _ = _build([*seeds, *documents])
    # An endpoint no document lists as an entity is a stub: a seed node
    # fills its type and confidence, as an entity statement of another
    # document does.
    stated = {e.id for r in documents for e in r.entities}

    for entity_id, expected in document_only.items():
        entity = seeded[entity_id]
        if entity_id not in stated:
            assert (entity.description or "").startswith(expected.description or "")
            continue
        assert entity.name == expected.name
        if expected.type:
            # A seed's type fills only a gap: no document gave one.
            assert entity.type == expected.type
        assert entity.confidence == expected.confidence
        assert (entity.description or "").startswith(expected.description or "")
        for key, value in (expected.attributes or {}).items():
            if key != "entity_keys":
                assert entity.attributes[key] == value
    # What only a seed states is a seed-only entity.
    for entity_id in seeded.keys() - document_only.keys():
        assert seed_only(seeded[entity_id].text_unit_ids)


@given(documents=_document_records(), seeds=_seed_records())
def test_removing_the_seeds_gives_the_document_only_graph(
    documents: list[TextUnitExtraction], seeds: list[TextUnitExtraction]
) -> None:
    expected_entities, expected_edges = _build(documents)
    entities, edges = _build([*seeds, *documents])
    seed_units = {r.text_unit_id for r in seeds}

    stripped_entities, stripped_edges = remove_text_units(
        list(entities.values()), list(edges.values()), seed_units
    )
    stripped = {e.id: e for e in stripped_entities} | {e.id: e for e in stripped_edges}
    # Items left without a text unit are deleted with the seed.
    kept_entities = [
        stripped.get(e.id, e)
        for e in entities.values()
        if stripped.get(e.id, e).text_unit_ids
    ]
    kept_edges = [
        stripped.get(r.id, r)
        for r in edges.values()
        if stripped.get(r.id, r).text_unit_ids
    ]
    records = {r.text_unit_id: r for r in documents}
    derived = derive(Config(), kept_entities, kept_edges, [], records)

    def entity_view(e: Entity) -> tuple:
        return (
            e.name,
            e.type,
            e.description,
            e.confidence,
            e.iri,
            set(e.text_unit_ids or []),
        )

    def edge_view(r: Relationship) -> tuple:
        return (r.source_id, r.target_id, r.type, r.description, r.weight)

    assert {e.id: entity_view(e) for e in derived.entities} == {
        i: entity_view(e) for i, e in expected_entities.items()
    }
    assert {r.id: edge_view(r) for r in derived.relationships} == {
        i: edge_view(r) for i, r in expected_edges.items()
    }
    assert not any(
        is_seed_unit(u)
        for item in [*derived.entities, *derived.relationships]
        for u in item.text_unit_ids or []
    )


@given(documents=_document_records(), seeds=_seed_records())
def test_seeds_never_change_document_ranks(
    documents: list[TextUnitExtraction], seeds: list[TextUnitExtraction]
) -> None:
    expected_entities, expected_edges = _build(documents)
    entities, edges = _build([*seeds, *documents])
    records = {r.text_unit_id: r for r in [*seeds, *documents]}
    derived = derive(
        Config(), list(entities.values()), list(edges.values()), [], records
    )
    derived_entities = {e.id: e for e in derived.entities}
    derived_edges = {r.id: r for r in derived.relationships}

    for ranked_entities, ranked_edges in (
        (entities, edges),
        (derived_entities, derived_edges),
    ):
        assert {i: ranked_entities[i].rank for i in expected_entities} == {
            i: e.rank for i, e in expected_entities.items()
        }
        assert {i: ranked_edges[i].rank for i in expected_edges} == {
            i: r.rank for i, r in expected_edges.items()
        }
        for entity_id in ranked_entities.keys() - expected_entities.keys():
            assert ranked_entities[entity_id].rank == 0
