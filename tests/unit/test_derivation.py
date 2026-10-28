# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stored items' content derived from their text units' current records.

What a stored entity, relationship or claim says must follow the text units
it cites now: a removed unit's description lines, type votes and claim text
go with it, a name follows the units' support, and an item a record is
missing for keeps its stored content.
"""

from __future__ import annotations

import time

import pytest

from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.derivation import _ordered_units, derive
from unified_kg_rag.domain.ingestion.graph_resolver import ENTITY_KEYS_ATTRIBUTE
from unified_kg_rag.domain.ingestion.relationship_weights import (
    apply_text_unit_weights,
)
from unified_kg_rag.domain.models import (
    Claim,
    Config,
    Entity,
    Relationship,
    TextUnitExtraction,
)
from unified_kg_rag.shared.utils import entity_key

pytestmark = pytest.mark.unit


def _id(name: str) -> str:
    return BaseProcessor._generate_entity_id(name)


def _entity(name: str, unit: str, type_: str = "ORG") -> Entity:
    return Entity(
        id=_id(name),
        name=name,
        type=type_,
        description=f"{name} in {unit}",
        text_unit_ids=[unit],
    )


def _edge(source: str, target: str, unit: str) -> Relationship:
    edge = Relationship(
        id=BaseProcessor._generate_relationship_id(source, target, "SUPPLIES"),
        source_id=_id(source),
        target_id=_id(target),
        source_name=source,
        target_name=target,
        type="SUPPLIES",
        description=f"{source} supplies {target} in {unit}",
    )
    apply_text_unit_weights(edge, {unit: 1.0})
    return edge


def _record(unit: str, *entities: Entity, edges=()) -> TextUnitExtraction:
    return TextUnitExtraction(
        text_unit_id=unit, entities=list(entities), relationships=list(edges)
    )


def _stored(name: str, units: list[str], **kw) -> Entity:
    return Entity(
        id=kw.pop("id", _id(name)),
        name=name,
        type=kw.pop("type", "org"),
        description=kw.pop("description", "stale text"),
        text_unit_ids=units,
        **kw,
    )


def test_a_removed_units_description_and_type_vote_are_gone() -> None:
    records = {
        "u2": _record("u2", _entity("Acme", "u2", "PERSON")),
        "u3": _record("u3", _entity("Acme", "u3", "PERSON")),
    }
    stored = _stored("Acme", ["u2", "u3"], description="Acme in u1\nAcme in u2")

    (derived,) = derive(Config(), [stored], [], [], records).entities

    assert derived.description.split("\n") == ["Acme in u2", "Acme in u3"]
    assert derived.type == "person"
    assert (derived.id, derived.text_unit_ids) == (stored.id, ["u2", "u3"])


def test_the_name_and_the_id_follow_the_support() -> None:
    # "Acme" was indexed first; two units now say "Acme Inc".
    records = {
        "u2": _record("u2", _entity("Acme Inc", "u2")),
        "u3": _record("u3", _entity("Acme Inc", "u3")),
        "u4": _record("u4", _entity("Acme", "u4")),
    }
    stored = _stored(
        "Acme",
        ["u2", "u3", "u4"],
        attributes={ENTITY_KEYS_ATTRIBUTE: ["acme", "acme inc"]},
    )

    result = derive(Config(), [stored], [], [], records)
    (derived,) = result.entities

    # A full build names it "Acme Inc" and keys it by that name.
    assert (derived.id, derived.name) == (_id("Acme Inc"), "Acme Inc")
    assert result.id_remap == {_id("Acme"): _id("Acme Inc")}
    assert set(derived.attributes[ENTITY_KEYS_ATTRIBUTE]) == {
        entity_key("Acme"),
        entity_key("Acme Inc"),
    }


def test_edges_and_claims_take_the_live_units_text() -> None:
    records = {
        "u2": _record(
            "u2",
            _entity("Acme", "u2"),
            _entity("Buyer", "u2"),
            edges=[_edge("Acme", "Buyer", "u2")],
        )
    }
    claim = Claim(
        id="c",
        subject_id="",
        subject_name="Acme",
        object_name="100",
        type="PAYMENT",
        description="Acme pays 100 (u2)",
        source_text="pays 100",
        text_unit_ids=["u2"],
    )
    stored_edge = _edge("Acme", "Buyer", "u2").model_copy(
        update={"description": "Acme supplies Buyer in u1\nAcme supplies Buyer in u2"}
    )
    stored_claim = Claim(
        id="",
        subject_id=_id("Acme"),
        subject_name="Acme",
        object_name="100",
        type="payment",
        description="Acme pays 120 (u1)\nAcme pays 100 (u2)",
        source_text="pays 120\npays 100",
        text_unit_ids=["u2"],
    )
    from unified_kg_rag.domain.ingestion.claim_resolver import resolved_claim_id

    stored_claim.id = resolved_claim_id(stored_claim)
    entities = [_stored("Acme", ["u2"]), _stored("Buyer", ["u2"])]

    derived = derive(
        Config(),
        entities,
        [stored_edge],
        [stored_claim],
        records,
        {"u2": [claim]},
    )

    (edge,) = derived.relationships
    assert edge.description == "Acme supplies Buyer in u2"
    (derived_claim,) = derived.claims
    assert (derived_claim.description, derived_claim.source_text) == (
        "Acme pays 100 (u2)",
        "pays 100",
    )


def test_an_item_whose_record_is_missing_keeps_its_content(caplog) -> None:
    stored = _stored("Acme", ["u1", "u2"])
    records = {"u2": _record("u2", _entity("Acme", "u2"))}

    derived = derive(Config(), [stored], [], [], records)

    assert derived.entities[0].description == "stale text"
    assert derived.underived == [stored.id]
    assert "without an extraction record" in caplog.text


def _merged_acme() -> tuple[Entity, dict[str, TextUnitExtraction]]:
    """ "Acme" and "Acme Inc" stored as one entity (indexed when both were
    persons); a later unit states "Acme" as a location, so a full build keeps
    "Acme Inc" apart."""
    records = {
        "u1": _record(
            "u1",
            _entity("Acme", "u1", "PERSON"),
            _entity("Acme Inc", "u1", "PERSON"),
            _entity("Initech", "u1", "PERSON"),
            edges=[_edge("Acme", "Acme Inc", "u1"), _edge("Acme Inc", "Initech", "u1")],
        ),
        "u0": _record("u0", _entity("Acme", "u0", "LOCATION")),
    }
    stored = _stored(
        "Acme",
        ["u1", "u0"],
        type="person",
        attributes={ENTITY_KEYS_ATTRIBUTE: ["acme", "acme inc"]},
    )
    return stored, records


def test_a_stored_entity_a_full_build_keeps_apart_is_split() -> None:
    # Regression: the stored entity took the content of its best match only,
    # so "Acme Inc" and the edges and claims it states were lost until a
    # rebuild.
    stored, records = _merged_acme()
    initech = _stored("Initech", ["u1"], type="person")
    # The stored edge "Acme Inc -> Initech", merged onto "Acme".
    edge = _edge("Acme Inc", "Initech", "u1").model_copy(
        update={"source_id": stored.id, "source_name": "Acme"}
    )

    derived = derive(Config(), [stored, initech], [edge], [], records)

    by_name = {e.name: e for e in derived.entities}
    assert set(by_name) == {"Acme", "Acme Inc", "Initech"}
    assert (by_name["Acme"].id, by_name["Acme"].text_unit_ids) == (
        stored.id,
        ["u1", "u0"],
    )
    assert (by_name["Acme Inc"].id, by_name["Acme Inc"].text_unit_ids) == (
        _id("Acme Inc"),
        ["u1"],
    )
    assert derived.split_from == {_id("Acme Inc"): stored.id}
    # The stored edge follows the part that states it; the edge the merge
    # had folded into a self-loop is back.
    assert sorted((r.source_name, r.target_name) for r in derived.relationships) == [
        ("Acme", "Acme Inc"),
        ("Acme Inc", "Initech"),
    ]
    (moved,) = [r for r in derived.relationships if r.target_name == "Initech"]
    assert (moved.id, moved.source_id) == (edge.id, _id("Acme Inc"))


def test_a_part_split_off_under_an_underived_entitys_id_keeps_its_units() -> None:
    # Regression: "Acme Inc" splits off the merged "Acme" under the id of a
    # stored "Acme Inc" one of whose units has no record (a removal killed
    # between the stores deleted it). That entity keeps its stored content,
    # and the part was dropped with its unit: u1 no longer cited by any
    # "Acme Inc", so a later derivation of it lost what u1 states.
    stored, records = _merged_acme()
    holder = _stored("Acme Inc", ["u9"], type="person")

    derived = derive(Config(), [stored, holder], [], [], records)

    by_id = {e.id: e for e in derived.entities}
    assert by_id[holder.id].text_unit_ids == ["u9", "u1"]
    assert by_id[holder.id].description == "stale text"
    assert derived.underived == [holder.id]
    assert by_id[stored.id].text_unit_ids == ["u1", "u0"]


def test_an_underived_holder_keeps_the_keys_of_the_units_it_takes() -> None:
    # Regression: the holder took the part's units (u1 "Acme Inc", u2 "Acme
    # LLC") but not its keys. Once the removal completed and "Acme LLC" no
    # longer resolved into "Acme Inc", the holder's derivation found no
    # candidate under "acme llc": no entity cited u2 and what it states was
    # lost (a full build has "Acme LLC" citing u2).
    stored, records = _merged_acme()
    records["u2"] = _record("u2", _entity("Acme LLC", "u2", "PERSON"))
    stored.text_unit_ids = [*(stored.text_unit_ids or []), "u2"]
    stored.attributes = {ENTITY_KEYS_ATTRIBUTE: ["acme", "acme inc", "acme llc"]}
    holder = _stored("Acme Inc", ["u9"], type="person")

    derived = derive(Config(), [stored, holder], [], [], records)

    (kept,) = [e for e in derived.entities if e.id == holder.id]
    assert kept.text_unit_ids == ["u9", "u1", "u2"]
    assert kept.attributes[ENTITY_KEYS_ATTRIBUTE] == ["acme inc", "acme llc"]

    # The removal completes (u9 is gone) and u1 now states an ORG "Acme Inc".
    kept.text_unit_ids = ["u1", "u2"]
    records = {
        "u1": _record("u1", _entity("Acme Inc", "u1")),
        "u2": records["u2"],
    }
    derived = derive(Config(), [kept], [], [], records)

    assert sorted((e.name, tuple(e.text_unit_ids or [])) for e in derived.entities) == [
        ("Acme Inc", ("u1",)),
        ("Acme LLC", ("u2",)),
    ]


def test_a_split_keeps_the_stored_name_under_its_id() -> None:
    # Regression: the stored entity matched the part sharing most of its
    # units ("Acme Inc"), so the part under its own id ("Acme") was "held by
    # another item" and dropped.
    records = {
        "u1": _record("u1", _entity("Acme", "u1", "LOCATION")),
        "u2": _record("u2", _entity("Acme Inc", "u2", "PERSON")),
        "u3": _record("u3", _entity("Acme Inc", "u3", "PERSON")),
    }
    stored = _stored(
        "Acme",
        ["u1", "u2", "u3"],
        attributes={ENTITY_KEYS_ATTRIBUTE: ["acme", "acme inc"]},
    )

    derived = derive(Config(), [stored], [], [], records)

    by_name = {e.name: e for e in derived.entities}
    assert set(by_name) == {"Acme", "Acme Inc"}
    assert (by_name["Acme"].id, by_name["Acme"].text_unit_ids) == (stored.id, ["u1"])
    assert derived.split_from == {_id("Acme Inc"): stored.id}
    assert derived.id_remap == {}


def test_a_claim_about_a_split_entity_follows_the_part() -> None:
    stored, records = _merged_acme()
    statement = Claim(
        id="c",
        subject_id="",
        subject_name="Acme",
        object_name="Acme Inc",
        type="OWNS",
        description="Acme owns Acme Inc",
        text_unit_ids=["u1"],
    )
    from unified_kg_rag.domain.ingestion.claim_resolver import resolved_claim_id

    # Stored about the merged entity, on both ends.
    stored_claim = statement.model_copy(
        update={"subject_id": stored.id, "object_id": stored.id, "type": "owns"}
    )
    stored_claim.id = resolved_claim_id(stored_claim)

    derived = derive(
        Config(), [stored], [], [stored_claim], records, {"u1": [statement]}
    )

    (claim,) = derived.claims
    assert (claim.subject_id, claim.object_id) == (stored.id, _id("Acme Inc"))
    # L1: the object's name is the derived entity's too.
    assert (claim.subject_name, claim.object_name) == ("Acme", "Acme Inc")
    assert derived.id_remap == {stored_claim.id: claim.id}


def test_the_claim_object_name_follows_the_derived_entity() -> None:
    # Regression: only the subject's name was derived; the object kept the
    # spelling the unit used ("Acme"), not the entity's name ("Acme Inc").
    records = {
        "u1": _record("u1", _entity("Acme Inc", "u1"), _entity("Buyer", "u1")),
        "u2": _record("u2", _entity("Acme Inc", "u2")),
        "u3": _record("u3", _entity("Acme", "u3"), _entity("Buyer", "u3")),
    }
    statement = Claim(
        id="c",
        subject_id="",
        subject_name="Buyer",
        object_name="Acme",
        type="PAYS",
        description="Buyer pays Acme",
        text_unit_ids=["u3"],
    )
    acme = _stored(
        "Acme Inc",
        ["u1", "u2", "u3"],
        attributes={ENTITY_KEYS_ATTRIBUTE: ["acme", "acme inc"]},
    )
    buyer = _stored("Buyer", ["u1", "u3"])
    from unified_kg_rag.domain.ingestion.claim_resolver import resolved_claim_id

    stored_claim = statement.model_copy(
        update={"subject_id": buyer.id, "object_id": acme.id, "type": "pays"}
    )
    stored_claim.id = resolved_claim_id(stored_claim)

    derived = derive(
        Config(), [acme, buyer], [], [stored_claim], records, {"u3": [statement]}
    )

    (claim,) = derived.claims
    assert (claim.object_id, claim.object_name) == (acme.id, "Acme Inc")


def test_the_claim_object_name_is_the_stored_entitys_when_it_is_underived() -> None:
    # The object entity cannot be derived (a unit it cites has no record), so
    # it keeps its stored name: the claim names it so too, not as the claim's
    # unit spells it.
    records = {"u1": _record("u1", _entity("Acme Inc", "u1"), _entity("Buyer", "u1"))}
    statement = Claim(
        id="c",
        subject_id="",
        subject_name="Buyer",
        object_name="Acme Inc",
        type="PAYS",
        description="Buyer pays Acme Inc",
        text_unit_ids=["u1"],
    )
    acme = _stored("ACME INC", ["u1", "u9"])
    buyer = _stored("Buyer", ["u1"])
    from unified_kg_rag.domain.ingestion.claim_resolver import resolved_claim_id

    stored_claim = statement.model_copy(
        update={"subject_id": buyer.id, "object_id": acme.id, "type": "pays"}
    )
    stored_claim.id = resolved_claim_id(stored_claim)

    derived = derive(
        Config(), [acme, buyer], [], [stored_claim], records, {"u1": [statement]}
    )

    assert derived.underived == [acme.id]
    (claim,) = derived.claims
    assert (claim.object_id, claim.object_name) == (acme.id, "ACME INC")


def test_an_edge_neither_stored_nor_given_is_derived() -> None:
    # Two derived entities a full build links: the edge is derived when the
    # caller's stored incident edges do not hold it.
    records = {
        "u1": _record(
            "u1",
            _entity("Acme", "u1"),
            _entity("Buyer", "u1"),
            edges=[_edge("Acme", "Buyer", "u1")],
        )
    }
    entities = [_stored("Acme", ["u1"]), _stored("Buyer", ["u1"])]

    assert derive(Config(), entities, [], [], records).relationships == []
    derived = derive(Config(), entities, [], [], records, stored_relationship_ids=[])
    (edge,) = derived.relationships
    assert (edge.source_name, edge.target_name, edge.text_unit_ids) == (
        "Acme",
        "Buyer",
        ["u1"],
    )
    stored = derive(
        Config(), entities, [], [], records, stored_relationship_ids=[edge.id]
    )
    assert stored.relationships == []


def test_an_edge_between_entities_that_now_merge_is_dropped() -> None:
    # Regression: the stored edge "Acme -> Acme Inc" was kept, re-pointed into
    # a self-loop a full build never draws, once both spellings merged.
    records = {
        "u1": _record(
            "u1",
            _entity("Acme", "u1"),
            _entity("Acme Inc", "u1"),
            edges=[_edge("Acme", "Acme Inc", "u1")],
        )
    }
    acme, acme_inc = _stored("Acme", ["u1"]), _stored("Acme Inc", ["u1"])
    edge = _edge("Acme", "Acme Inc", "u1")

    derived = derive(Config(), [acme, acme_inc], [edge], [], records)

    assert len(derived.entities) == 1
    assert (derived.relationships, derived.dropped) == ([], [edge.id])


def test_unit_order_is_linear_in_the_units() -> None:
    # Regression: the set of the stored units was rebuilt for every unit, so
    # a hub citing 40k units took ~100 s to order.
    units = [f"u{i}" for i in range(20_000)]
    stored = units[::-1][:-1]

    started = time.perf_counter()
    ordered = _ordered_units(stored, units)

    assert time.perf_counter() - started < 1.0
    assert ordered == [*stored, units[0]]


def test_only_claims_the_given_claims_units_state_are_derived() -> None:
    # The entities' text units hold claims too, but only the given claims'
    # units are in scope: a claim of another unit may be stored with more
    # text units than the ones read here, and deriving it from these alone
    # would overwrite it.
    from unified_kg_rag.domain.ingestion.claim_resolver import resolved_claim_id

    records = {
        "u1": _record("u1", _entity("Acme", "u1"), _entity("Buyer", "u1")),
        "u2": _record("u2", _entity("Acme", "u2"), _entity("Depot", "u2")),
    }

    def statement(obj: str, unit: str) -> Claim:
        return Claim(
            id=f"c-{unit}",
            subject_id="",
            subject_name="Acme",
            object_name=obj,
            type="PAYS",
            description=f"Acme pays {obj}",
            text_unit_ids=[unit],
        )

    acme = _stored("Acme", ["u1", "u2"])
    buyer = _stored("Buyer", ["u1"])
    depot = _stored("Depot", ["u2"])
    given = statement("Buyer", "u1").model_copy(
        update={"subject_id": acme.id, "object_id": buyer.id, "type": "pays"}
    )
    given.id = resolved_claim_id(given)

    derived = derive(
        Config(),
        [acme, buyer, depot],
        [],
        [given],
        records,
        {"u1": [statement("Buyer", "u1")], "u2": [statement("Depot", "u2")]},
    )

    assert [c.object_name for c in derived.claims] == ["Buyer"]


def _pays(subject: str, obj: str, units: list[str], claim_id: str = "") -> Claim:
    return Claim(
        id=claim_id or f"{subject}-{obj}",
        subject_id="",
        subject_name=subject,
        object_name=obj,
        type="PAYS",
        description=f"{subject} pays {obj}",
        text_unit_ids=units,
    )


def test_a_claim_no_claim_of_its_units_states_is_dropped() -> None:
    # Its unit's records hold no claim at all now: a full build has no such
    # claim, so the stored one goes (the caller deletes it).
    records = {"u1": _record("u1", _entity("Acme", "u1"), _entity("Buyer", "u1"))}
    acme, buyer = _stored("Acme", ["u1"]), _stored("Buyer", ["u1"])
    stale = _pays("Acme", "Buyer", ["u1"], "stale").model_copy(
        update={"subject_id": acme.id, "object_id": buyer.id}
    )

    derived = derive(Config(), [acme, buyer], [], [stale], records, {"u1": []})

    assert derived.claims == []
    assert derived.dropped == ["stale"]


def test_an_unmatched_claim_takes_the_claim_sharing_most_of_its_units() -> None:
    # The given claim matches no resolved claim by id; of the two its units
    # state, the one citing both of them is it, whatever their ids.
    from unified_kg_rag.domain.ingestion.claim_resolver import resolved_claim_id

    def best(other: str) -> tuple[str, str]:
        records = {
            "u1": _record("u1", _entity("Acme", "u1"), _entity("Buyer", "u1")),
            "u2": _record(
                "u2",
                _entity("Acme", "u2"),
                _entity("Buyer", "u2"),
                _entity(other, "u2"),
            ),
        }
        entities = [
            _stored("Acme", ["u1", "u2"]),
            _stored("Buyer", ["u1", "u2"]),
            _stored(other, ["u2"]),
        ]
        claim_records = {
            "u1": [_pays("Acme", "Buyer", ["u1"])],
            "u2": [_pays("Acme", "Buyer", ["u2"]), _pays("Acme", other, ["u2"])],
        }
        given = _pays("Acme", "Nobody", ["u1", "u2"], "unmatched")
        derived = derive(Config(), entities, [], [given], records, claim_records)
        ids = {c.object_name: c.id for c in derived.claims}
        assert derived.id_remap["unmatched"] == ids["Buyer"]
        return ids["Buyer"], resolved_claim_id(
            _pays("Acme", other, ["u2"]).model_copy(
                update={"subject_id": entities[0].id, "object_id": entities[2].id}
            )
        )

    # One of the object names gives the one-unit claim the larger id, so a
    # choice by id alone would pick it.
    results = [best(name) for name in ("Depot", "Carrier", "Globex", "Initech")]
    assert any(other > buyer for buyer, other in results)


@pytest.mark.parametrize("given_dates", [(None, None), ("2019-05-05", None)])
def test_a_derived_claim_takes_the_dates_of_its_new_id(given_dates: tuple) -> None:
    # The given claim (stored without dates, or with others) matches the
    # dated claim its unit states: it took that claim's id, whose dates are
    # part of it, but kept its own dates, so the id and the dates disagreed
    # and a full build stored the date the incremental run did not.
    from unified_kg_rag.domain.ingestion.claim_resolver import resolved_claim_id

    records = {"u1": _record("u1", _entity("Acme", "u1"), _entity("Buyer", "u1"))}
    acme, buyer = _stored("Acme", ["u1"]), _stored("Buyer", ["u1"])
    dated = _pays("Acme", "Buyer", ["u1"]).model_copy(
        update={"start_date": "2020-01-01"}
    )
    given = _pays("Acme", "Nobody", ["u1"], "given").model_copy(
        update={"start_date": given_dates[0], "end_date": given_dates[1]}
    )

    derived = derive(Config(), [acme, buyer], [], [given], records, {"u1": [dated]})

    (claim,) = derived.claims
    assert claim.start_date == "2020-01-01"
    assert claim.id == resolved_claim_id(claim)


def test_another_spelling_is_an_endpoint_only_where_the_edge_states_it() -> None:
    # "Acme Inc" (another type) is stated in u2 only. The stored Acme-Depot
    # edge of u1, which u1's records no longer state, is dropped; it must not
    # take the Acme Inc-Depot edge of u2, which shares none of its units.
    records = {
        "u1": _record("u1", _entity("Acme", "u1"), _entity("Depot", "u1")),
        "u2": _record(
            "u2",
            _entity("Acme Inc", "u2", "PERSON"),
            _entity("Depot", "u2"),
            edges=[_edge("Acme Inc", "Depot", "u2")],
        ),
    }
    acme = _stored("Acme", ["u1"])
    acme_inc = _stored("Acme Inc", ["u2"], type="person")
    depot = _stored("Depot", ["u1", "u2"])
    stale = _edge("Acme", "Depot", "u1")

    derived = derive(Config(), [acme, acme_inc, depot], [stale], [], records)

    assert derived.dropped == [stale.id]
    assert stale.id not in {r.id for r in derived.relationships}


def test_a_tie_between_candidates_does_not_depend_on_their_order() -> None:
    # The candidates of a stored entity are collected over its keys, a set
    # whose order follows PYTHONHASHSEED: a tie went to whichever came first.
    from unified_kg_rag.domain.ingestion.derivation import _correspondence

    stored = _stored("Acme", ["u1", "u2"])
    a = _entity("Acme Corp", "u1").model_copy(update={"text_unit_ids": ["u1"]})
    b = _entity("Acme Ltd", "u2").model_copy(update={"text_unit_ids": ["u2"]})

    matches = {
        _correspondence([stored], {stored.id: order}, {stored.id: True})[2][stored.id][
            0
        ]
        for order in ([a, b], [b, a])
    }

    assert matches == {min(a.id, b.id)}


def test_a_claim_keeps_an_underived_endpoint_the_resolved_one_is_not() -> None:
    # Neither stored entity can be derived (u2, u3 have no record). u1 alone
    # resolves "Acme" apart from "Acme LLC", so the claim's subject resolved
    # to an "Acme" entity nothing stores or derives: a dangling endpoint. It
    # keeps the stored entity it names.
    from unified_kg_rag.domain.ingestion.claim_resolver import resolved_claim_id

    records = {
        "u1": _record(
            "u1",
            _entity("Acme", "u1"),
            _entity("Depot", "u1"),
            _entity("Acme LLC", "u1"),
            edges=[_edge("Acme", "Depot", "u1"), _edge("Acme LLC", "Depot", "u1")],
        )
    }
    acme = _stored(
        "Acme LLC",
        ["u1", "u2"],
        attributes={ENTITY_KEYS_ATTRIBUTE: ["acme", "acme llc"]},
    )
    depot = _stored("Depot", ["u1", "u3"])
    stored = _pays("Acme LLC", "Depot", ["u1"]).model_copy(
        update={"subject_id": acme.id, "object_id": depot.id}
    )
    stored.id = resolved_claim_id(stored)

    derived = derive(
        Config(),
        [acme, depot],
        [],
        [stored],
        records,
        {"u1": [_pays("Acme", "Depot", ["u1"])]},
    )

    assert sorted(derived.underived) == sorted([acme.id, depot.id])
    (claim,) = derived.claims
    assert (claim.subject_id, claim.subject_name) == (acme.id, "Acme LLC")
    assert (claim.object_id, claim.object_name) == (depot.id, "Depot")


def test_an_underived_stored_object_endpoint_is_kept() -> None:
    # The subject case above, with the endpoint that cannot be derived (a
    # unit it cites, u2, has no record) on the object side: the claim keeps
    # the stored entity, not the id its own unit resolves to alone.
    from unified_kg_rag.domain.ingestion.claim_resolver import resolved_claim_id

    records = {
        "u1": _record(
            "u1",
            _entity("Acme", "u1"),
            _entity("Depot", "u1"),
            _entity("Acme LLC", "u1"),
            edges=[_edge("Depot", "Acme", "u1"), _edge("Depot", "Acme LLC", "u1")],
        )
    }
    acme = _stored(
        "Acme LLC",
        ["u1", "u2"],
        attributes={ENTITY_KEYS_ATTRIBUTE: ["acme", "acme llc"]},
    )
    depot = _stored("Depot", ["u1", "u3"])
    stored = _pays("Depot", "Acme LLC", ["u1"]).model_copy(
        update={"subject_id": depot.id, "object_id": acme.id}
    )
    stored.id = resolved_claim_id(stored)
    derived = derive(
        Config(),
        [acme, depot],
        [],
        [stored],
        records,
        {"u1": [_pays("Depot", "Acme", ["u1"])]},
    )
    (claim,) = derived.claims
    assert (claim.object_id, claim.object_name) == (acme.id, "Acme LLC")
    assert (claim.subject_id, claim.subject_name) == (depot.id, "Depot")


def test_a_derived_endpoint_wins_over_an_underived_stored_one() -> None:
    # The stored subject (Acme LLC) cannot be derived (u2 has no record),
    # but the claim's subject resolves to a derived entity (Acme, every unit
    # it cites recorded): the claim follows the derived one.
    from unified_kg_rag.domain.ingestion.claim_resolver import resolved_claim_id

    records = {
        "u1": _record(
            "u1",
            _entity("Acme", "u1"),
            _entity("Depot", "u1"),
            _entity("Acme LLC", "u1"),
            edges=[_edge("Acme", "Depot", "u1"), _edge("Acme LLC", "Depot", "u1")],
        )
    }
    acme_llc = _stored(
        "Acme LLC", ["u1", "u2"], attributes={ENTITY_KEYS_ATTRIBUTE: ["acme llc"]}
    )
    acme = _stored("Acme", ["u1"])
    depot = _stored("Depot", ["u1"])
    stored = _pays("Acme LLC", "Depot", ["u1"]).model_copy(
        update={"subject_id": acme_llc.id, "object_id": depot.id}
    )
    stored.id = resolved_claim_id(stored)
    derived = derive(
        Config(),
        [acme_llc, acme, depot],
        [],
        [stored],
        records,
        {"u1": [_pays("Acme", "Depot", ["u1"])]},
    )
    assert acme_llc.id in derived.underived
    assert acme.id not in derived.underived
    (claim,) = derived.claims
    assert claim.subject_id == acme.id, (claim.subject_id, claim.subject_name)


@pytest.mark.parametrize("side", ["object", "subject"])
def test_a_claim_about_an_underived_entitys_spelling_keeps_its_unit(
    side: str,
) -> None:
    # Regression: "Acme" (stored merged with "Acme Inc") cites u9, whose
    # record a killed removal deleted, so it is underived. The stored claim
    # "Depot pays Acme" cites u1, which states "Depot pays Acme Inc" (a claim
    # a full build keeps apart, about a spelling only the underived entity
    # holds), and u2. That claim could not be emitted and u1 left the given
    # claim too: no claim cited u1, and its statement was lost. Emitted, it
    # would point at an entity no derived or stored item is. Alike with
    # Acme the subject.
    from unified_kg_rag.domain.ingestion.claim_resolver import resolved_claim_id

    records = {
        "u0": _record("u0", _entity("Acme", "u0", "LOCATION")),
        "u1": _record(
            "u1",
            _entity("Depot", "u1", "PERSON"),
            _entity("Acme Inc", "u1", "PERSON"),
            _entity("Acme", "u1", "PERSON"),
        ),
        "u2": _record(
            "u2", _entity("Depot", "u2", "PERSON"), _entity("Acme", "u2", "LOCATION")
        ),
    }
    acme = _stored("Acme", ["u0", "u1", "u2", "u9"])
    depot = _stored("Depot", ["u1", "u2"])
    pair, split = (("Depot", "Acme"), ("Depot", "Acme Inc"))
    ids = (depot.id, acme.id)
    if side == "subject":
        pair, split, ids = pair[::-1], split[::-1], ids[::-1]
    stored = _pays(*pair, ["u1", "u2"]).model_copy(
        update={"subject_id": ids[0], "object_id": ids[1]}
    )
    stored.id = resolved_claim_id(stored)

    derived = derive(
        Config(),
        [acme, depot],
        [],
        [stored],
        records,
        {
            "u1": [_pays(*split, ["u1"], "s1")],
            "u2": [_pays(*pair, ["u2"], "s2")],
        },
    )

    assert acme.id in derived.underived
    known = {e.id for e in derived.entities} | {acme.id, depot.id}
    for claim in derived.claims:
        assert claim.subject_id in known, claim.subject_name
        assert claim.object_id is None or claim.object_id in known, claim.object_name
    cited = [u for c in derived.claims for u in c.text_unit_ids or []]
    assert sorted(cited) == ["u1", "u2"]
