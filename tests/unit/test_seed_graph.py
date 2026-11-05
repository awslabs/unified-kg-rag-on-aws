# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A seed graph converted into virtual shard documents, and the records
parsed back from their text units (``domain.ingestion.seed_graph``)."""

from __future__ import annotations

import json
import random
import time
from typing import Any, NoReturn

import pytest
from pydantic import ValidationError

from tests.fixtures.seed_graphs import seed_statements
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.contributions import (
    record_from_payload,
    record_payload,
)
from unified_kg_rag.domain.ingestion.seed_graph import (
    MAX_UNITS_PER_SHARD,
    SEED_FORMAT_VERSION,
    SEED_RECORD_FINGERPRINT,
    SeedConversion,
    SeedConversionSettings,
    SeedEdgeStatement,
    SeedNodeStatement,
    _Converter,
    convert_seed,
    is_seed_document,
    is_seed_unit,
    parse_seed_unit,
    parse_statement,
    parse_unit_statements,
    render_statement,
    render_unit,
    seed_text_units,
    shard_of,
)
from unified_kg_rag.domain.models import (
    Document,
    EntityTypeSpec,
    ExtractionSchema,
    RelationshipTypeSpec,
)
from unified_kg_rag.domain.models.seed_graph import (
    MAX_IRI_CHARS,
    MAX_NAME_CHARS,
    MAX_TYPE_CHARS,
    SeedEdge,
    SeedGraph,
    SeedNode,
)
from unified_kg_rag.shared.exceptions import (
    ConfigurationError,
    DataProcessingError,
    SeedGraphError,
)
from unified_kg_rag.shared.utils import generate_stable_id

pytestmark = pytest.mark.unit

EX = "https://example.org/procure#"


def _schema(mode: str = "align", **overrides: Any) -> ExtractionSchema:
    items: dict[str, Any] = {
        "entity_types": [
            {"label": "VENDOR", "iri": EX + "Vendor"},
            {"label": "BUYER", "iri": EX + "Buyer"},
            "PRODUCT",
        ],
        "relationship_types": [
            {"label": "SUPPLIES", "iri": EX + "supplies"},
            "BUYS",
        ],
        "relationship_patterns": [
            "VENDOR -SUPPLIES-> BUYER",
            "BUYER -BUYS-> PRODUCT",
        ],
    }
    items.update(overrides)
    return ExtractionSchema.from_items(
        [
            EntityTypeSpec(**t) if isinstance(t, dict) else t
            for t in items["entity_types"]
        ],
        [
            RelationshipTypeSpec(**t) if isinstance(t, dict) else t
            for t in items["relationship_types"]
        ],
        items["relationship_patterns"],
        mode=mode,  # type: ignore[arg-type]
    )


def _settings(**overrides: Any) -> SeedConversionSettings:
    values: dict[str, Any] = {"name": "procure", "shards": 1}
    values.update(overrides)
    return SeedConversionSettings(**values)


def _graph(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]] | None = None
) -> SeedGraph:
    return SeedGraph(
        nodes=tuple(SeedNode(**n) for n in nodes),
        edges=tuple(SeedEdge(**e) for e in edges or []),
    )


def _statements(conversion: SeedConversion) -> list[dict[str, Any]]:
    return [json.loads(line) for line in seed_statements(conversion)]


def _procure() -> SeedGraph:
    return _graph(
        [
            {
                "id": "v1",
                "name": "Acme Supplies",
                "type": "vendor",
                "iri": EX + "acme",
                "description": "Industrial valve vendor.",
                "properties": {"country": "DE", "rating": None},
            },
            {"id": "b1", "name": "Globex Buyer", "type_iri": EX + "Buyer"},
            {"id": "p1", "name": "Valve", "type": "PRODUCT"},
        ],
        [
            {
                "source": "v1",
                "target": "b1",
                "type": "supplies",
                "description": "Acme supplies Globex with valves.",
                "weight": 2,
            },
            {
                "source": "b1",
                "target": "p1",
                "predicate_iri": EX + "buys",
                "type": "buys",
            },
        ],
    )


# -- Golden conversion ---------------------------------------------------------


def test_golden_conversion() -> None:
    conversion = convert_seed(
        _procure(),
        _settings(
            filters={"department": "buying"}, description_properties=("country",)
        ),
        _schema(),
    )

    (document,) = conversion.documents()
    assert document.file_path == "seed_graphs/procure/0000-of-0001.jsonl"
    assert document.file_name == "0000-of-0001.jsonl"
    assert document.metadata == {
        "source_kind": "seed_graph",
        "seed_graph": "procure",
        "filters": {"department": "buying", "seed_graph": "procure"},
        "relative_path": "seed_graphs/procure/0000-of-0001.jsonl",
    }
    assert is_seed_document(document)
    assert document.page_content == "\n".join(
        [
            '{"filters":{"department":"buying","seed_graph":"procure"},"format":1,'
            '"seed_graph":"procure"}',
            "",
            '{"confidence":1.0,"description":"Industrial valve vendor.\\ncountry: DE",'
            '"id":"v1","iri":"https://example.org/procure#acme","kind":"node",'
            '"name":"Acme Supplies","properties":{"country":"DE"},"type":"VENDOR"}',
            '{"description":"Acme supplies Globex with valves.","kind":"edge",'
            '"source":"Acme Supplies","target":"Globex Buyer","type":"SUPPLIES",'
            '"weight":2.0}',
            '{"confidence":1.0,"id":"b1","kind":"node","name":"Globex Buyer",'
            '"type":"BUYER"}',
            '{"kind":"edge","predicate_iri":"https://example.org/procure#buys",'
            '"source":"Globex Buyer","target":"Valve","type":"BUYS","weight":1.0}',
            '{"confidence":1.0,"id":"p1","kind":"node","name":"Valve",'
            '"type":"PRODUCT"}',
        ]
    )
    assert conversion.stats.nodes == 3
    assert conversion.stats.edges == 2
    assert conversion.stats.units == 1
    assert conversion.stats.documents == 1
    # buys has no IRI in the schema: the label decides, the IRI is kept.
    assert conversion.stats.edges_unmapped_predicate_iri == 1
    assert len(conversion.digest) == 64


def test_conversion_is_inert_on_its_documents() -> None:
    conversion = convert_seed(_procure(), _settings(), _schema())

    first = conversion.documents()
    first[0].metadata["relative_path"] = "elsewhere"

    assert (
        conversion.documents()[0].metadata["relative_path"].startswith("seed_graphs/")
    )


def test_a_type_iri_names_the_type_declared_last() -> None:
    schema = _schema(
        entity_types=[
            {"label": "PARTY", "iri": EX + "Party"},
            {"label": "VENDOR", "iri": EX + "Party"},
        ],
        relationship_types=[],
        relationship_patterns=[],
    )
    graph = _graph(
        [
            {"id": "v1", "name": "Acme", "type": "OTHER", "type_iri": EX + "Party"},
            {"id": "v2", "name": "Initech", "type": "VENDOR", "type_iri": EX + "Gone"},
        ]
    )

    conversion = convert_seed(graph, _settings(), schema)

    assert {s["id"]: s["type"] for s in _statements(conversion)} == {
        "v1": "VENDOR",
        "v2": "VENDOR",
    }
    assert conversion.stats.nodes_unmapped_type_iri == 1


def test_an_edge_without_a_type_fails() -> None:
    graph = _graph(
        [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
        [{"source": "a", "target": "b", "predicate_iri": EX + "unknown"}],
    )

    with pytest.raises(SeedGraphError, match="has no type"):
        convert_seed(graph, _settings(), _schema())


# -- Conformance ---------------------------------------------------------------


def _off_schema_seed() -> SeedGraph:
    return _graph(
        [
            {"id": "v1", "name": "Acme", "type": "VENDOR"},
            {"id": "b1", "name": "Globex", "type": "BUYER"},
            {"id": "x1", "name": "Mystery", "type": "ALIEN"},
            {"id": "p1", "name": "Valve", "type": "PRODUCT"},
        ],
        [
            {"source": "v1", "target": "b1", "type": "SUPPLIES"},
            # off pattern: a BUYER does not supply
            {"source": "b1", "target": "v1", "type": "SUPPLIES"},
            # off-schema type
            {"source": "v1", "target": "p1", "type": "MAKES"},
            # an endpoint off schema
            {"source": "x1", "target": "p1", "type": "BUYS"},
        ],
    )


def test_off_keeps_everything_and_counts_nothing() -> None:
    conversion = convert_seed(_off_schema_seed(), _settings(), _schema("off"))

    assert (conversion.stats.nodes, conversion.stats.edges) == (4, 4)
    assert conversion.stats.entities_off_schema == 0
    assert conversion.stats.relationships_off_pattern == 0


def test_align_keeps_everything_and_counts() -> None:
    conversion = convert_seed(_off_schema_seed(), _settings(), _schema("align"))

    stats = conversion.stats
    assert (stats.nodes, stats.edges) == (4, 4)
    assert stats.entities_off_schema == 1
    assert stats.relationships_off_schema == 1
    assert stats.relationships_off_pattern == 1
    assert stats.entities_dropped_off_schema == 0


def test_strict_drops_off_schema_nodes_their_edges_and_off_pattern_edges() -> None:
    conversion = convert_seed(_off_schema_seed(), _settings(), _schema("strict"))

    stats = conversion.stats
    statements = _statements(conversion)
    assert sorted(s["name"] for s in statements if s["kind"] == "node") == [
        "Acme",
        "Globex",
        "Valve",
    ]
    assert [
        (s["source"], s["type"], s["target"]) for s in statements if s["kind"] == "edge"
    ] == [("Acme", "SUPPLIES", "Globex")]
    assert stats.entities_dropped_off_schema == 1
    assert stats.relationships_dropped_off_schema == 1
    assert stats.relationships_dropped_off_schema_endpoint == 1
    assert stats.relationships_dropped_off_pattern == 1


def test_strict_checks_patterns_with_both_endpoint_types_across_shards() -> None:
    # The two endpoints land in different shards; the check is over the
    # whole seed, so the pattern still sees both types.
    nodes = [{"id": f"b{n}", "name": f"Buyer {n}", "type": "BUYER"} for n in range(8)]
    nodes.append({"id": "v1", "name": "Acme", "type": "VENDOR"})
    edges = [{"source": f"b{n}", "target": "v1", "type": "SUPPLIES"} for n in range(8)]
    settings = _settings(shards=8)
    assert len({shard_of(n["id"], 8) for n in nodes}) > 1

    conversion = convert_seed(_graph(nodes, edges), settings, _schema("strict"))

    assert conversion.stats.edges == 0
    assert conversion.stats.relationships_dropped_off_pattern == 8


def test_strict_drops_the_edges_of_a_dropped_node_another_node_names() -> None:
    graph = _graph(
        [
            {"id": "v1", "name": "Acme", "type": "VENDOR"},
            {"id": "v2", "name": "ACME", "type": "ALIEN"},
            {"id": "b1", "name": "Globex", "type": "BUYER"},
        ],
        [{"source": "v2", "target": "b1", "type": "SUPPLIES"}],
    )

    conversion = convert_seed(graph, _settings(), _schema("strict"))

    assert conversion.stats.edges == 0
    assert conversion.stats.relationships_dropped_off_schema_endpoint == 1


# -- Name collisions -----------------------------------------------------------


def _homonyms(**second: Any) -> SeedGraph:
    return _graph(
        [
            {
                "id": "m1",
                "name": "Mercury",
                "type": "VENDOR",
                "iri": EX + "mercury-inc",
            },
            {"id": "m2", "name": "MERCURY", "type": "BUYER", **second},
            {"id": "b1", "name": "Globex", "type": "BUYER"},
        ],
        [{"source": "m1", "target": "b1", "type": "SUPPLIES"}],
    )


def test_a_name_collision_fails_by_default() -> None:
    with pytest.raises(SeedGraphError, match="m1, m2.*on_name_collision"):
        convert_seed(_homonyms(), _settings(), _schema())


def test_nodes_sharing_an_iri_do_not_collide() -> None:
    conversion = convert_seed(_homonyms(iri=EX + "mercury-inc"), _settings(), _schema())

    assert conversion.stats.name_collisions_merged == 0
    assert conversion.stats.nodes == 3


def test_merge_keeps_both_statements_of_one_entity() -> None:
    conversion = convert_seed(
        _homonyms(), _settings(on_name_collision="merge"), _schema()
    )
    records = [
        parse_seed_unit(unit)
        for document in conversion.documents()
        for unit in seed_text_units(document)
    ]

    ids = [e.id for r in records for e in r.entities if e.name.lower() == "mercury"]
    assert len(ids) == 2 and len(set(ids)) == 1
    assert conversion.stats.name_collisions_merged == 1


def test_merge_drops_an_edge_between_merged_nodes_as_a_self_loop() -> None:
    graph = _graph(
        [{"id": "a", "name": "Acme"}, {"id": "b", "name": "ACME"}],
        [{"source": "a", "target": "b", "type": "BUYS"}],
    )

    conversion = convert_seed(graph, _settings(on_name_collision="merge"), _schema())

    assert conversion.stats.edges == 0
    assert conversion.stats.self_loops_dropped == 1


def test_qualify_renames_the_later_nodes() -> None:
    conversion = convert_seed(
        _homonyms(), _settings(on_name_collision="qualify"), _schema()
    )

    names = {s["id"]: s["name"] for s in _statements(conversion) if s["kind"] == "node"}
    assert names == {"m1": "Mercury", "m2": "MERCURY (m2)", "b1": "Globex"}
    assert conversion.stats.names_qualified == 1


def test_qualify_uses_the_qualifier_property_then_the_iri() -> None:
    by_property = convert_seed(
        _homonyms(properties={"city": "Springfield"}),
        _settings(on_name_collision="qualify", qualifier_property="city"),
        _schema(),
    )
    by_iri = convert_seed(
        _homonyms(iri=EX + "mercury-planet"),
        _settings(on_name_collision="qualify", qualifier_property="city"),
        _schema(),
    )

    def m2(conversion: SeedConversion) -> str:
        return next(s["name"] for s in _statements(conversion) if s.get("id") == "m2")

    assert m2(by_property) == "MERCURY (Springfield)"
    assert m2(by_iri) == "MERCURY (mercury-planet)"


def test_strict_checks_qualified_homonyms_against_their_own_types() -> None:
    # Two Mercury nodes, a VENDOR and a BUYER, each in a pattern of its type:
    # qualify makes them two entities, so neither edge may be checked against
    # the other's type.
    graph = _graph(
        [
            {"id": "m1", "name": "Mercury", "type": "VENDOR"},
            {"id": "m2", "name": "Mercury", "type": "BUYER"},
            {"id": "b", "name": "Globex", "type": "BUYER"},
            {"id": "p", "name": "Valve", "type": "PRODUCT"},
        ],
        [
            {"source": "m1", "target": "b", "type": "SUPPLIES"},
            {"source": "m2", "target": "p", "type": "BUYS"},
        ],
    )

    conversion = convert_seed(
        graph, _settings(on_name_collision="qualify"), _schema("strict")
    )

    edges = {
        (s["source"], s["type"], s["target"])
        for s in _statements(conversion)
        if s["kind"] == "edge"
    }
    assert edges == {
        ("Mercury", "SUPPLIES", "Globex"),
        ("Mercury (m2)", "BUYS", "Valve"),
    }
    assert conversion.stats.relationships_off_pattern == 0
    # merge makes them one entity, typed once, as a full build would: one of
    # the two edges is off its pattern.
    merged = convert_seed(
        graph, _settings(on_name_collision="merge"), _schema("strict")
    )
    assert merged.stats.relationships_dropped_off_pattern == 1


def test_strict_checks_homonyms_sharing_an_iri_as_one_entity_under_qualify() -> None:
    # One IRI, one identity: nothing to qualify, so qualify checks the
    # patterns of the one entity as merge does (one edge off its pattern).
    graph = _graph(
        [
            {"id": "m1", "name": "Mercury", "type": "VENDOR", "iri": EX + "m"},
            {"id": "m2", "name": "MERCURY", "type": "BUYER", "iri": EX + "m"},
            {"id": "b", "name": "Globex", "type": "BUYER"},
            {"id": "a", "name": "Acme", "type": "VENDOR"},
        ],
        [
            {"source": "m1", "target": "b", "type": "SUPPLIES"},
            {"source": "a", "target": "m2", "type": "SUPPLIES"},
        ],
    )

    qualified, merged = (
        convert_seed(graph, _settings(on_name_collision=policy), _schema("strict"))
        for policy in ("qualify", "merge")
    )

    assert qualified.stats.names_qualified == 0
    assert qualified.stats.relationships_dropped_off_pattern == 1
    assert _statements(qualified) == _statements(merged)


def test_a_qualified_name_over_the_name_cap_fails() -> None:
    # The name cap holds for a qualified name too (a store keeps it as an
    # indexed keyword).
    graph = _homonyms(properties={"city": "S" * MAX_NAME_CHARS})

    with pytest.raises(SeedGraphError, match="nodes 'm2' are longer than 1000"):
        convert_seed(
            graph,
            _settings(on_name_collision="qualify", qualifier_property="city"),
            _schema(),
        )


def test_a_qualified_name_of_exactly_the_name_cap_converts() -> None:
    city = "S" * (MAX_NAME_CHARS - len("MERCURY ()"))
    conversion = convert_seed(
        _homonyms(properties={"city": city}),
        _settings(on_name_collision="qualify", qualifier_property="city"),
        _schema(),
    )

    names = {s["name"] for s in _statements(conversion) if s["kind"] == "node"}
    assert f"MERCURY ({city})" in names
    assert max(len(name) for name in names) == MAX_NAME_CHARS


def test_qualified_names_that_still_collide_fail() -> None:
    graph = _graph(
        [
            {"id": "m1", "name": "Mercury"},
            {"id": "m2", "name": "Mercury"},
            {"id": "m3", "name": "Mercury (m2)"},
        ]
    )

    with pytest.raises(SeedGraphError, match="still collide"):
        convert_seed(graph, _settings(on_name_collision="qualify"), _schema())


# -- Shards and units ----------------------------------------------------------


def _catalog(count: int) -> list[dict[str, Any]]:
    return [
        {"id": f"n{i}", "name": f"Part {i}", "type": "PRODUCT"} for i in range(count)
    ]


def test_adding_one_node_changes_one_shard() -> None:
    settings = _settings(shards=8)
    before = convert_seed(_graph(_catalog(40)), settings, _schema())
    after = convert_seed(_graph(_catalog(41)), settings, _schema())

    old = {s.relative_path: s.page_content for s in before.documents_content}
    new = {s.relative_path: s.page_content for s in after.documents_content}
    changed = {path for path in new if old.get(path) != new[path]}
    assert changed == {f"seed_graphs/procure/{shard_of('n40', 8):04d}-of-0008.jsonl"}
    assert before.digest != after.digest


def test_an_edge_lives_in_its_source_nodes_shard() -> None:
    nodes = _catalog(20)
    edges = [{"source": "n0", "target": f"n{i}", "type": "BUYS"} for i in range(1, 20)]

    conversion = convert_seed(_graph(nodes, edges), _settings(shards=4), _schema())

    home = f"{shard_of('n0', 4):04d}-of-0004"
    for shard in conversion.documents_content:
        has_edges = '"kind":"edge"' in shard.page_content
        assert has_edges == (home in shard.relative_path)


def test_cards_pack_into_units_of_at_most_max_items() -> None:
    conversion = convert_seed(
        _graph(_catalog(7)), _settings(max_items_per_unit=3), _schema()
    )

    (document,) = conversion.documents()
    units = seed_text_units(document)
    assert [len(u.text.split("\n")) for u in units] == [3, 3, 1]


def test_a_hub_is_split_into_full_units_by_edge_order() -> None:
    nodes = _catalog(8)
    edges = [{"source": "n0", "target": f"n{i}", "type": "BUYS"} for i in range(1, 8)]

    conversion = convert_seed(
        _graph(nodes, edges), _settings(max_items_per_unit=3), _schema()
    )

    (document,) = conversion.documents()
    units = [parse_unit_statements(u.text) for u in seed_text_units(document)]
    assert all(len(unit) <= 3 for unit in units)
    hub = [s for unit in units for s in unit][:8]
    assert isinstance(hub[0], SeedNodeStatement) and hub[0].id == "n0"
    targets = [s.target for s in hub[1:] if isinstance(s, SeedEdgeStatement)]
    assert targets == [f"Part {i}" for i in range(1, 8)]
    assert conversion.stats.units == len(units)


def test_the_header_covers_the_name_and_filters() -> None:
    base = convert_seed(_procure(), _settings(), _schema())
    renamed = convert_seed(_procure(), _settings(name="procure-2"), _schema())
    filtered = convert_seed(_procure(), _settings(filters={"team": "a"}), _schema())

    header, *_ = base.documents_content[0].page_content.split("\n")
    assert json.loads(header) == {
        "seed_graph": "procure",
        "format": SEED_FORMAT_VERSION,
        "filters": {"seed_graph": "procure"},
    }
    texts = {c.documents_content[0].page_content for c in (base, renamed, filtered)}
    assert len(texts) == 3
    assert len({c.digest for c in (base, renamed, filtered)}) == 3


# -- Units and records ---------------------------------------------------------


def test_render_parse_round_trip() -> None:
    statements = [
        SeedNodeStatement(
            id="v1",
            name="Acme ✓ 공급",
            type="VENDOR",
            iri=EX + "acme",
            description="line one\nline two",
            confidence=0.5,
            properties={"n": 3, "f": 1.5, "b": True, "s": "x"},
        ),
        SeedEdgeStatement(source="Acme", target="Globex", type="SUPPLIES", weight=2.5),
    ]

    text = render_unit(statements)

    assert len(text.split("\n")) == 2
    assert parse_unit_statements(text) == statements
    assert [parse_statement(render_statement(s)) for s in statements] == statements


def test_parse_rejects_text_that_is_not_seed_statements() -> None:
    with pytest.raises(DataProcessingError):
        parse_statement('{"kind":"node","name":"x"}')
    with pytest.raises(DataProcessingError):
        parse_statement("not json")


def test_unit_ids_are_prefixed_and_sort_after_document_unit_ids() -> None:
    conversion = convert_seed(
        _graph(_catalog(5)), _settings(max_items_per_unit=2), _schema()
    )
    (document,) = conversion.documents()

    units = seed_text_units(document)

    assert [u.id for u in units] == [
        "sg-" + generate_stable_id(f"text_unit:{document.document_id}:{n}")
        for n in range(3)
    ]
    assert all(is_seed_unit(u.id) for u in units)
    assert all(u.short_id == u.id[3:11] for u in units)
    document_unit = generate_stable_id("text_unit:any:0")
    assert not is_seed_unit(document_unit)
    assert max("f" * 8, document_unit) < min(u.id for u in units)
    assert [u.attributes["chunk_id"] for u in units] == [0, 1, 2]  # type: ignore[index]


def test_seed_text_units_carry_placement_and_token_counts() -> None:
    conversion = convert_seed(_procure(), _settings(filters={"team": "a"}), _schema())
    (document,) = conversion.documents()
    document.metadata["index"] = "tenant"

    (unit,) = seed_text_units(document, count_tokens=len)

    assert unit.n_tokens == len(unit.text)
    assert unit.document_ids == [document.document_id]
    assert unit.attributes is not None
    assert unit.attributes["source_kind"] == "seed_graph"
    assert unit.attributes["index"] == "tenant"
    assert unit.attributes["filters"] == {"team": "a", "seed_graph": "procure"}


def test_seed_text_units_refuse_a_document_that_is_no_shard() -> None:
    document = Document(
        page_content="plain text", document_id="d", file_name="a.txt", file_path="a.txt"
    )

    with pytest.raises(DataProcessingError, match="not a seed graph shard"):
        seed_text_units(document)


def test_the_record_uses_extraction_ids_and_survives_the_store() -> None:
    conversion = convert_seed(
        _procure(), _settings(description_properties=("country",)), _schema()
    )
    (document,) = conversion.documents()
    document.metadata["index"] = "tenant"
    (unit,) = seed_text_units(document)

    record = parse_seed_unit(unit)

    acme = next(e for e in record.entities if e.name == "Acme Supplies")
    assert acme.id == BaseProcessor._generate_entity_id("acme supplies")
    assert acme.type == "VENDOR"
    assert acme.iri == EX + "acme"
    assert acme.attributes == {
        "seed_properties": {"country": "DE"},
        "index": "tenant",
        "filters": {"seed_graph": "procure"},
    }
    supplies = next(r for r in record.relationships if r.type == "SUPPLIES")
    assert supplies.id == BaseProcessor._generate_relationship_id(
        "Acme Supplies", "Globex Buyer", "SUPPLIES"
    )
    assert supplies.source_id == acme.id
    assert supplies.weight == 2.0
    assert record.text_unit_id == unit.id
    assert (record.rejected, record.gleaned, record.stored) == ({}, False, False)
    assert parse_seed_unit(unit) == record
    reloaded = record_from_payload(record_payload(record), unit)
    assert reloaded.model_copy(update={"stored": False}) == record


def test_the_record_fingerprint_depends_on_the_format_only() -> None:
    # Pinned: a change re-keys every stored seed record, so it must come
    # with a SEED_FORMAT_VERSION bump.
    assert SEED_FORMAT_VERSION == 1
    assert SEED_RECORD_FINGERPRINT == "671c46b50362"


def test_descriptions_are_cut_at_a_word_boundary() -> None:
    graph = _graph(
        [
            {
                "id": "a",
                "name": "A",
                "description": "word " * 30,
                "properties": {"k": "v"},
            }
        ]
    )

    conversion = convert_seed(
        graph,
        _settings(max_description_chars=22, description_properties=("k",)),
        _schema(),
    )

    (statement,) = _statements(conversion)
    assert statement["description"] == "word word word word…"
    assert conversion.stats.descriptions_truncated == 1


# -- Refusals, merges and drops ------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_nodes": 2}, "3 nodes exceed max_nodes"),
        ({"max_edges": 1}, "2 edges exceed max_edges"),
        ({"max_property_keys": 0}, "1 distinct property keys exceed"),
    ],
)
def test_caps_fail_instead_of_truncating(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(SeedGraphError, match=message):
        convert_seed(_procure(), _settings(**overrides), _schema())


def test_a_shard_over_the_unit_limit_fails_asking_for_more_shards() -> None:
    graph = _graph(_catalog(MAX_UNITS_PER_SHARD + 1))

    with pytest.raises(SeedGraphError, match="raise shards"):
        convert_seed(graph, _settings(max_items_per_unit=1, max_nodes=10**6), _schema())
    # At the limit it converts, every node kept.
    at_limit = convert_seed(
        _graph(_catalog(MAX_UNITS_PER_SHARD)),
        _settings(max_items_per_unit=1),
        _schema(),
    )
    assert at_limit.stats.units == MAX_UNITS_PER_SHARD


def _random_seed(nodes: int, edges: int) -> tuple[SeedGraph, SeedConversionSettings]:
    generator = random.Random(7)
    graph = SeedGraph(
        nodes=tuple(
            SeedNode(id=f"n{i}", name=f"Part {i}", type="PRODUCT") for i in range(nodes)
        ),
        edges=tuple(
            SeedEdge(
                source=f"n{generator.randrange(nodes)}",
                target=f"n{generator.randrange(nodes)}",
                type="BUYS",
            )
            for _ in range(edges)
        ),
    )
    return graph, _settings(shards=16, max_nodes=nodes, max_edges=edges)


def _conversion_seconds(nodes: int, edges: int, runs: int, cap: float) -> float:
    """The least CPU time of ``runs`` conversions of a random seed of this
    size; stops after a run over ``cap`` seconds (it already fails)."""
    graph, settings = _random_seed(nodes, edges)
    best = float("inf")
    for _ in range(runs):
        # CPU time: a busy runner delays the process, it does not add to it.
        start = time.process_time()
        conversion = convert_seed(graph, settings, _schema("off"))
        best = min(best, time.process_time() - start)
        assert conversion.stats.nodes == nodes
        if best > cap:
            break
    return best


# Four times the seed takes about four times as long (about 0.9 s, then
# 3.5 s); the quadratic edge check (a set difference against every node id
# per edge) takes more than ten times as long. Shared CI runners measured up
# to 8.7 for the linear code, so this timing check is `perf`: it runs in the
# weekly exploration job, and the lookup test above pins the known cause on
# every PR.
_SCALING_BOUND = 8.0


class _LookupOnlyNodes(dict[str, SeedNode]):
    """A node table that fails on a walk over its keys: an edge must look
    its endpoints up, not scan every node (that made a large seed quadratic)."""

    def _walked(self, *args: object) -> NoReturn:
        raise AssertionError("edge resolution walked the node table")

    keys = values = items = __iter__ = _walked


def test_edges_look_their_endpoints_up_without_walking_the_nodes() -> None:
    graph, settings = _random_seed(200, 1_000)
    converter = _Converter(graph, settings, _schema("off"))
    plain = converter.unique_nodes()

    edges = converter.resolved_edges(_LookupOnlyNodes(plain))

    assert edges == converter.resolved_edges(plain)
    assert len(edges) > 900


@pytest.mark.perf
@pytest.mark.slow
@pytest.mark.timeout(300)
def test_a_large_seed_converts_in_linear_time() -> None:
    small = _conversion_seconds(2_500, 12_500, runs=3, cap=float("inf"))
    large = _conversion_seconds(10_000, 50_000, runs=3, cap=_SCALING_BOUND * small)
    assert large / small < _SCALING_BOUND, f"{small:.2f} s -> {large:.2f} s"


def test_a_hub_over_the_unit_limit_fails_naming_it() -> None:
    # A node's card (it and its outgoing edges) stays in one shard: more
    # shards cannot split a hub, a larger max_items_per_unit can.
    spokes = 3 * MAX_UNITS_PER_SHARD
    graph = _graph(
        [{"id": "hub", "name": "Hub"}, *_catalog(spokes)],
        [{"source": "hub", "target": f"n{i}", "type": "BUYS"} for i in range(spokes)],
    )
    settings = _settings(max_items_per_unit=3, shards=1024, max_nodes=10**6)

    with pytest.raises(SeedGraphError) as error:
        convert_seed(graph, settings, _schema())

    message = str(error.value)
    assert "node 'Hub' (id 'hub')" in message
    assert f"is {spokes + 1} statements" in message
    # 3 * MAX_UNITS_PER_SHARD + 1 statements, 3 a unit: one more unit.
    assert f"{MAX_UNITS_PER_SHARD + 1} units at max_items_per_unit 3" in message
    assert "raise max_items_per_unit" in message
    assert "raise shards" not in message
    converted = convert_seed(
        graph, _settings(max_items_per_unit=10, shards=1024), _schema()
    )
    assert converted.stats.edges == spokes


def test_a_dangling_edge_fails_or_is_skipped() -> None:
    graph = _graph(
        [{"id": "a", "name": "A"}],
        [{"source": "a", "target": "ghost", "type": "BUYS"}],
    )

    with pytest.raises(SeedGraphError, match="unknown nodes 'ghost'"):
        convert_seed(graph, _settings(), _schema())
    skipped = convert_seed(graph, _settings(dangling="skip"), _schema())
    assert skipped.stats.dangling_edges_skipped == 1
    assert skipped.stats.edges == 0


def test_a_duplicate_node_id_fails() -> None:
    graph = _graph([{"id": "a", "name": "A"}, {"id": "a", "name": "B"}])

    with pytest.raises(SeedGraphError, match="duplicate node ids 'a'"):
        convert_seed(graph, _settings(), _schema())


def test_duplicate_edges_merge() -> None:
    graph = _graph(
        [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
        [
            {
                "source": "a",
                "target": "b",
                "type": "buys",
                "weight": 2,
                "description": "one",
            },
            {
                "source": "a",
                "target": "b",
                "type": "BUYS",
                "weight": 1.5,
                "description": "two",
            },
            {"source": "a", "target": "b", "type": "BUYS", "description": "one"},
        ],
    )

    conversion = convert_seed(graph, _settings(), _schema())

    edges = [s for s in _statements(conversion) if s["kind"] == "edge"]
    assert len(edges) == 1
    assert edges[0]["weight"] == 4.5
    assert sorted(edges[0]["description"].split("\n")) == ["one", "two"]
    assert conversion.stats.duplicate_edges_merged == 2


def test_self_loops_are_dropped_and_counted() -> None:
    graph = _graph(
        [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
        [
            {"source": "a", "target": "a", "type": "BUYS"},
            {"source": "a", "target": "a", "type": "BUYS", "weight": 2},
            {"source": "a", "target": "b", "type": "BUYS"},
        ],
    )

    conversion = convert_seed(graph, _settings(), _schema())

    # Each dropped as read: never merged, typed or checked against the schema.
    assert conversion.stats.self_loops_dropped == 2
    assert conversion.stats.duplicate_edges_merged == 0
    assert conversion.stats.edges == 1


# -- Models and settings -------------------------------------------------------


def test_seed_node_rules() -> None:
    node = SeedNode(id="v1", name="  Acme   Supplies ", properties={"a": None, "b": 1})

    assert node.name == "Acme Supplies"
    assert node.properties == {"b": 1}
    assert SeedNode(id="v2").name == "v2"
    assert SeedNode(id="v3", name="  ").name == "v3"
    with pytest.raises(ValidationError):
        SeedNode(id="v1", properties={"a": [1]})
    with pytest.raises(ValidationError):
        SeedNode(id="v1", properties={"a": float("nan")})
    with pytest.raises(ValidationError):
        SeedNode(id="v1", colour="red")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        SeedNode(id="")
    with pytest.raises(ValidationError):
        node.name = "x"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        SeedEdge(source="a", target="b", weight=float("inf"))
    with pytest.raises(ValidationError):
        SeedEdge(source="a", target="b", weight=-1)


@pytest.mark.parametrize(
    ("model", "field", "cap"),
    [
        (SeedNode, "name", MAX_NAME_CHARS),
        (SeedNode, "type", MAX_TYPE_CHARS),
        (SeedNode, "type_iri", MAX_IRI_CHARS),
        (SeedNode, "iri", MAX_IRI_CHARS),
        (SeedEdge, "type", MAX_TYPE_CHARS),
        (SeedEdge, "predicate_iri", MAX_IRI_CHARS),
    ],
)
def test_names_types_and_iris_are_capped(model: Any, field: str, cap: int) -> None:
    ends = {"id": "v1"} if model is SeedNode else {"source": "a", "target": "b"}
    assert len(getattr(model(**ends, **{field: "x" * cap}), field)) == cap
    with pytest.raises(ValidationError, match=f"at most {cap} characters"):
        model(**ends, **{field: "x" * (cap + 1)})


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "Procure"},
        {"name": "a/b"},
        {"filters": {"seed_graph": "x"}},
        {"filters": {"k": ["list"]}},
        {"shards": 0},
        {"max_items_per_unit": 0},
        {"confidence": 1.5},
    ],
)
def test_settings_refuse_invalid_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(SeedGraphError) as caught:
        _settings(**overrides)
    assert isinstance(caught.value, ConfigurationError)


def test_settings_are_frozen_and_order_free() -> None:
    settings = _settings(filters={"b": 1, "a": 2})

    assert list(settings.filters) == ["a", "b"]
    assert settings.document_filters == {"a": 2, "b": 1, "seed_graph": "procure"}
    with pytest.raises(TypeError):
        settings.filters["c"] = 3  # type: ignore[index]


def test_a_nodes_own_confidence_beats_the_seed_default_into_its_record() -> None:
    conversion = convert_seed(
        _graph([{"id": "a", "name": "Alpha", "confidence": 0.4}, {"id": "b"}]),
        _settings(confidence=0.9),
        _schema("off"),
    )

    by_id = {s["id"]: s["confidence"] for s in _statements(conversion)}
    assert by_id == {"a": 0.4, "b": 0.9}
    (document,) = conversion.documents()
    parsed = {
        e.name: e.confidence
        for unit in seed_text_units(document)
        for e in parse_seed_unit(unit).entities
    }
    assert parsed == {"Alpha": 0.4, "b": 0.9}


def test_a_seed_of_exactly_max_nodes_converts() -> None:
    conversion = convert_seed(
        _graph(_catalog(3)), _settings(max_nodes=3), _schema("off")
    )

    assert conversion.stats.nodes == 3
    with pytest.raises(SeedGraphError, match="max_nodes"):
        convert_seed(_graph(_catalog(4)), _settings(max_nodes=3), _schema("off"))


def test_a_description_of_exactly_the_limit_is_not_cut() -> None:
    text = "word " * 4 + "word"  # 24 characters

    def described(limit: int) -> str:
        conversion = convert_seed(
            _graph([{"id": "a", "description": text}]),
            _settings(max_description_chars=limit),
            _schema("off"),
        )
        assert conversion.stats.descriptions_truncated == int(limit < len(text))
        return str(_statements(conversion)[0]["description"])

    assert described(len(text)) == text
    assert described(len(text) - 1) == "word word word word…"


def test_edge_weights_are_clamped_as_extracted_ones_are() -> None:
    conversion = convert_seed(
        _graph(
            [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            [
                {"source": "a", "target": "b", "type": "BUYS", "weight": 1e12},
                {"source": "a", "target": "c", "type": "BUYS", "weight": 1e-12},
            ],
        ),
        _settings(),
        _schema("off"),
    )

    weights = {
        s["target"]: s["weight"] for s in _statements(conversion) if s["kind"] == "edge"
    }
    assert weights == {"b": 1e9, "c": 0.0}


def test_duplicate_edges_merge_by_type_key_in_a_canonical_order() -> None:
    # "ships to" and "SHIPS_TO" share a type key: one edge, the smallest
    # label and IRI, weights added; later edges of the canonical order (by
    # their whole statement, so "a" before "b") win a property.
    edges = [
        {
            "source": "a",
            "target": "b",
            "type": "ships to",
            "predicate_iri": EX + "shipsB",
            "description": "b",
            "weight": 2,
            "properties": {"k": "from b", "only_b": 1},
        },
        {
            "source": "a",
            "target": "b",
            "type": "SHIPS_TO",
            "predicate_iri": EX + "shipsA",
            "description": "a",
            "weight": 1,
            "properties": {"k": "from a", "only_a": 1},
        },
    ]
    for ordered in (edges, edges[::-1]):
        conversion = convert_seed(
            _graph([{"id": "a"}, {"id": "b"}], ordered), _settings(), _schema("off")
        )

        (edge,) = [s for s in _statements(conversion) if s["kind"] == "edge"]
        assert conversion.stats.duplicate_edges_merged == 1
        assert edge["type"] == "SHIPS_TO"
        assert edge["predicate_iri"] == EX + "shipsA"
        assert edge["weight"] == 3.0
        assert edge["description"] == "a\nb"
        assert edge["properties"] == {"k": "from b", "only_a": 1, "only_b": 1}
