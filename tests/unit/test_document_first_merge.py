# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Documents first: where a merge picks one value of an entity (type, name,
confidence), the statements of document units decide and a seed graph's
only fill a gap; descriptions are unioned with document lines first; the
IRI is the smallest one stated. Every merge path follows the same rule:
the statements' merge, graph resolution, the incremental merge and
derivation. Synthetic data only."""

from __future__ import annotations

import logging

import pytest

from unified_kg_rag.domain.ingestion.base_processor import (
    BaseProcessor,
    merge_attributes,
)
from unified_kg_rag.domain.ingestion.contributions import (
    merge_contributions,
    merge_entity_contributions,
    record_from_payload,
    record_payload,
)
from unified_kg_rag.domain.ingestion.derivation import derive
from unified_kg_rag.domain.ingestion.graph_resolver import EntityResolver
from unified_kg_rag.domain.ingestion.merge.merger import merge_entities
from unified_kg_rag.domain.ingestion.seed_graph import (
    annotate_sources,
    document_first,
    merged_iri,
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

pytestmark = pytest.mark.unit

EX = "https://example.org/procure#"


def _doc(n: int) -> str:
    return generate_stable_id(f"text_unit:doc:{n}")


def _seed(n: int) -> str:
    return seed_unit_id("seed-doc", n)


def _id(name: str) -> str:
    return BaseProcessor._generate_entity_id(name)


def _stated(name: str, unit: str, **kw) -> Entity:
    return Entity(id=_id(name), name=name, text_unit_ids=[unit], **kw)


def _record(unit: str, *entities: Entity) -> TextUnitExtraction:
    return TextUnitExtraction(text_unit_id=unit, entities=list(entities))


def _resolver() -> EntityResolver:
    return EntityResolver(Config(), max_workers=1, use_process_pool=False)


# -- The helpers ---------------------------------------------------------------


def test_seed_only_needs_seed_units_and_no_document_unit() -> None:
    assert seed_only([_seed(0), _seed(1)])
    assert not seed_only([_seed(0), _doc(0)])
    assert not seed_only([_doc(0)])
    assert not seed_only([])
    assert not seed_only(None)


def test_document_first_keeps_every_item_without_seed_units() -> None:
    items = [_stated("Acme", _doc(0)), _stated("Acme", _doc(1))]
    seeded = [_stated("Acme", _seed(0)), *items]

    assert document_first(items) == items
    assert document_first(seeded) == items
    assert document_first(seeded[:1]) == seeded[:1]


@pytest.mark.parametrize(
    ("units", "kinds"),
    [
        ([], None),
        (["t1", "t2"], None),
        (["t1", _seed(0)], ["document", "seed_graph"]),
        ([_seed(1), _seed(0)], ["seed_graph"]),
    ],
)
def test_annotate_sources(units: list[str], kinds: list[str] | None) -> None:
    entity = Entity(id="e", name="Acme", text_unit_ids=units)
    edge = Relationship(id="r", source_id="a", target_id="b", text_unit_ids=units)

    assert annotate_sources(entity) == kinds
    assert annotate_sources(edge) == kinds


def test_merged_iri_is_the_smallest_whatever_the_order() -> None:
    assert merged_iri([None, EX + "b", EX + "a", None]) == (EX + "a", True)
    assert merged_iri([EX + "a", EX + "b"]) == merged_iri([EX + "b", EX + "a"])
    assert merged_iri([EX + "a", EX + "a"]) == (EX + "a", False)
    assert merged_iri([None, None]) == (None, False)


def test_the_seed_fields_are_left_out_of_a_dump_while_unset() -> None:
    entity = Entity(id="e", name="Acme")
    edge = Relationship(id="r", source_id="a", target_id="b")

    assert {"iri", "source_kinds"}.isdisjoint(entity.model_dump())
    assert "source_kinds" not in edge.model_dump(mode="json")
    # Stored data without the fields reads back as before.
    assert Entity.model_validate({"id": "e", "name": "Acme"}).iri is None
    set_ = entity.model_copy(update={"iri": EX + "a", "source_kinds": ["seed_graph"]})
    assert set_.model_dump()["iri"] == EX + "a"


def test_a_record_keeps_the_iri_and_drops_source_kinds() -> None:
    unit = TextUnit(id=_seed(0), text="x")
    entity = _stated("Acme", unit.id, iri=EX + "acme", source_kinds=["seed_graph"])
    record = _record(unit.id, entity)

    payload = record_payload(record)

    assert payload["entities"][0]["iri"] == EX + "acme"
    assert "source_kinds" not in payload["entities"][0]
    (reloaded,) = record_from_payload(payload, unit).entities
    assert (reloaded.iri, reloaded.source_kinds) == (EX + "acme", None)


# -- The statements' merge -----------------------------------------------------


@pytest.mark.parametrize("seed_first", [True, False])
def test_documents_outvote_any_number_of_seed_statements(seed_first: bool) -> None:
    statements = [
        _stated("Acme", _doc(0), type="VENDOR"),
        *(_stated("Acme", _seed(n), type="BUYER") for n in range(3)),
    ]
    if seed_first:
        statements.reverse()

    (merged,) = merge_entity_contributions(statements)

    assert merged.type == "vendor"


def test_seed_votes_decide_only_when_no_document_gives_a_type() -> None:
    statements = [
        _stated("Acme", _doc(0), type=""),
        _stated("Acme", _seed(0), type="VENDOR"),
        _stated("Acme", _seed(1), type="BUYER"),
    ]

    (merged,) = merge_entity_contributions(statements)

    # The seeds tie: the smallest, as for documents.
    assert merged.type == "buyer"


def test_the_name_is_the_documents_spelling() -> None:
    statements = [
        _stated("Acme", _doc(0)),
        _stated("ACME", _seed(0)),
        _stated("ACME", _seed(1)),
    ]

    (merged,) = merge_entity_contributions(statements)

    assert merged.name == "Acme"


def test_the_confidence_is_the_documents_max_else_the_seeds() -> None:
    mixed = [
        _stated("Acme", _seed(0), confidence=1.0),
        _stated("Acme", _doc(0), confidence=0.4),
        _stated("Acme", _doc(1), confidence=0.6),
    ]
    seeds = [
        _stated("Acme", _seed(0), confidence=0.3),
        _stated("Acme", _seed(1), confidence=0.7),
    ]

    assert merge_entity_contributions(mixed)[0].confidence == 0.6
    assert merge_entity_contributions(seeds)[0].confidence == 0.7


def test_the_iri_is_the_smallest_and_conflicts_are_counted(caplog) -> None:
    statements = [
        _stated("Acme", _doc(0)),
        _stated("Acme", _seed(0), iri=EX + "b"),
        _stated("Acme", _seed(1), iri=EX + "a"),
    ]

    with caplog.at_level(logging.INFO):
        (merged,) = merge_entity_contributions(statements)

    assert merged.iri == EX + "a"
    assert "1 entities stated with several (entities_with_conflicting_iris)" in (
        caplog.text
    )


def test_document_lines_come_first_in_the_description() -> None:
    # The records merge in unit-id order and "sg-" ids sort after hex ids.
    seed = _seed(0)
    documents = sorted(_doc(n) for n in range(5))
    assert all(unit < seed for unit in documents)
    records = [
        _record(seed, _stated("Acme", seed, description="Seed line")),
        *(
            _record(u, _stated("Acme", u, description=f"Doc line {n}"))
            for n, u in enumerate(documents)
        ),
    ]

    entities, _ = merge_contributions(reversed(records))

    (acme,) = entities
    assert acme.description.split("\n") == [
        *(f"Doc line {n}" for n in range(5)),
        "Seed line",
    ]


def test_seed_properties_never_overwrite_extracted_attributes() -> None:
    doc, seed = _doc(0), _seed(0)
    records = [
        _record(doc, _stated("Acme", doc, attributes={"country": "FR"})),
        _record(
            seed,
            _stated("Acme", seed, attributes={"seed_properties": {"country": "DE"}}),
        ),
        _record(
            _seed(1),
            _stated(
                "Acme",
                _seed(1),
                attributes={"seed_properties": {"country": "AT", "size": 3}},
            ),
        ),
    ]

    entities, _ = merge_contributions(records)

    (acme,) = entities
    assert acme.attributes["country"] == "FR"
    # Kept beside the document's, merged per key in unit order.
    later = max(seed, _seed(1))
    assert acme.attributes["seed_properties"] == {
        "country": "AT" if later == _seed(1) else "DE",
        "size": 3,
    }
    resolved = merge_attributes(
        [
            {"country": "FR", "seed_properties": {"country": "DE"}},
            {"seed_properties": {"size": 3}},
        ]
    )
    assert resolved == {
        "country": "FR",
        "seed_properties": {"country": "DE", "size": 3},
    }


# -- Graph resolution ----------------------------------------------------------


def test_the_canonical_name_counts_document_units_first() -> None:
    entities = [
        Entity(id=_id("Acme"), name="Acme", text_unit_ids=[_doc(0)]),
        Entity(
            id=_id("Acme Inc"),
            name="Acme Inc",
            text_unit_ids=[_seed(0), _seed(1), _seed(2)],
        ),
    ]

    assert EntityResolver._best_supported_name(entities) == "Acme"
    # Seed units break a tie of document units.
    entities[1].text_unit_ids = [_doc(1), _seed(0)]
    assert EntityResolver._best_supported_name(entities) == "Acme Inc"
    # Without seed units, the most units win, as before.
    entities[1].text_unit_ids = [_doc(1), _doc(2)]
    assert EntityResolver._best_supported_name(entities) == "Acme Inc"


def test_resolution_is_document_first_and_keeps_the_smallest_iri(caplog) -> None:
    entities = [
        Entity(
            id=_id("Acme"),
            name="Acme",
            type="vendor",
            confidence=0.5,
            iri=EX + "z",
            text_unit_ids=[_doc(0), _seed(3)],
        ),
        Entity(
            id=_id("Acme Inc"),
            name="Acme Inc",
            type="vendor",
            confidence=0.9,
            iri=EX + "a",
            text_unit_ids=[_seed(0), _seed(1), _seed(2)],
        ),
    ]

    with caplog.at_level(logging.INFO):
        (merged,), _, _ = _resolver().resolve(entities)

    assert (merged.name, merged.id) == ("Acme", _id("Acme"))
    assert merged.confidence == 0.5
    assert merged.iri == EX + "a"
    assert "(entities_with_conflicting_iris)" in caplog.text


# -- The incremental merge and derivation --------------------------------------


@pytest.mark.parametrize("stored_is_seed", [True, False])
def test_the_incremental_merge_is_document_first(stored_is_seed: bool) -> None:
    seed = Entity(
        id=_id("Acme"),
        name="Acme",
        type="buyer",
        confidence=1.0,
        iri=EX + "acme",
        text_unit_ids=[_seed(0)],
    )
    document = Entity(
        id=_id("Acme"),
        name="Acme",
        type="vendor",
        confidence=0.4,
        text_unit_ids=[_doc(0)],
    )
    old, delta = (seed, document) if stored_is_seed else (document, seed)

    (merged,), _ = merge_entities([old], [delta])

    assert (merged.type, merged.confidence, merged.iri) == ("vendor", 0.4, EX + "acme")
    assert set(merged.text_unit_ids) == {_seed(0), _doc(0)}


@pytest.mark.parametrize("stored_iri", [EX + "a", EX + "b"])
def test_the_incremental_merge_keeps_the_smallest_iri(stored_iri: str) -> None:
    # Whichever side holds it: a full build keeps the smallest too.
    other = EX + "b" if stored_iri == EX + "a" else EX + "a"
    stored = _stated("Acme", _seed(0), iri=stored_iri)
    delta = _stated("Acme", _seed(1), iri=other)

    (merged,), _ = merge_entities([stored], [delta])

    assert merged.iri == EX + "a"


def test_derivation_takes_the_iri_and_drops_a_removed_seeds() -> None:
    doc, seed = _doc(0), _seed(0)
    stored = Entity(
        id=_id("Acme"),
        name="Acme",
        type="vendor",
        iri=EX + "stale",
        text_unit_ids=[doc, seed],
    )
    records = {
        doc: _record(doc, _stated("Acme", doc, type="VENDOR", confidence=0.4)),
        seed: _record(
            seed, _stated("Acme", seed, type="BUYER", confidence=1.0, iri=EX + "acme")
        ),
    }

    (derived,) = derive(Config(), [stored], [], [], records).entities
    assert (derived.type, derived.confidence, derived.iri) == (
        "vendor",
        0.4,
        EX + "acme",
    )

    # The seed is gone: its IRI goes with it.
    stored = derived.model_copy(update={"text_unit_ids": [doc]})
    (derived,) = derive(Config(), [stored], [], [], {doc: records[doc]}).entities
    assert derived.iri is None
