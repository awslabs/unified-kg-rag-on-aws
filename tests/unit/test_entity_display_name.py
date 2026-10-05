# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entity names keep their display form; ids come from a symbol-aware key.

Regression: ``Entity.name`` was set to ``normalize_name(name)``, which drops
every symbol. "C++" and "C#" both became "c" (same id, silently merged),
"$1,000 penalty" became "1 000 penalty" and "Section 4.2" became
"section 4 2", so retrieval context and community reports showed mangled
names. All names below are synthetic.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.ingestion.gleaner import GraphGleaner
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.graph_resolver import EntityResolver
from unified_kg_rag.domain.ingestion.merge import merge_entities
from unified_kg_rag.domain.models import Config, Entity, TextUnit
from unified_kg_rag.shared.utils import clean_display_name, entity_key

pytestmark = pytest.mark.unit


@pytest.fixture
def processor() -> BaseProcessor:
    return BaseProcessor(Config(), show_progress=False)


@pytest.fixture
def unit() -> TextUnit:
    return TextUnit(id="t1", text="synthetic chunk")


def _entity(processor: BaseProcessor, unit: TextUnit, name: str) -> Entity:
    entity = processor.parse_entity_data({"name": name, "type": "CONCEPT"}, unit)
    assert entity is not None
    return entity


# --------------------------------------------------------------------------- #
# Key / display helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("C++", "c++"),
        ("C#", "c#"),
        (".NET", ".net"),
        ("Section 4.2", "section 4.2"),
        ("$1,000 penalty", "$1000 penalty"),
        ("  ACME   Corp. ", "acme corp"),
        ("Vendor's Office", "vendors office"),
        ("follow-up_task", "follow up task"),
        ("가나 상사", "가나 상사"),
        ("...", "..."),  # never collapses to an empty key
    ],
)
def test_entity_key(name: str, key: str) -> None:
    assert entity_key(name) == key


def test_clean_display_name_only_trims_and_collapses_whitespace() -> None:
    assert clean_display_name("  $1,000\n penalty  ") == "$1,000 penalty"
    assert clean_display_name("C++") == "C++"
    assert clean_display_name(None) == ""


# --------------------------------------------------------------------------- #
# Extraction: display name + id
# --------------------------------------------------------------------------- #


def test_symbol_names_get_distinct_ids(processor, unit) -> None:
    cpp, csharp = _entity(processor, unit, "C++"), _entity(processor, unit, "C#")
    assert cpp.id != csharp.id
    assert (cpp.name, csharp.name) == ("C++", "C#")


@pytest.mark.parametrize("name", ["$1,000 penalty", "Section 4.2", "ISO 9001:2015"])
def test_display_name_preserves_figures(processor, unit, name: str) -> None:
    assert _entity(processor, unit, name).name == name


def test_case_and_whitespace_variants_share_one_id(processor, unit) -> None:
    variants = ["Acme Corp", "ACME  Corp", " acme corp. ", "Acme Corp."]
    ids = {_entity(processor, unit, v).id for v in variants}
    assert len(ids) == 1


def test_relationship_links_to_right_entities(processor, unit) -> None:
    cpp = _entity(processor, unit, "C++")
    csharp = _entity(processor, unit, "C#")
    vendor = _entity(processor, unit, "Vendor Alpha")
    index = processor.build_entity_key_index([cpp, csharp, vendor])

    uses_cpp = processor.parse_relationship_data(
        {"source": "VENDOR ALPHA", "target": "c++", "type": "USES"}, unit, index
    )
    uses_csharp = processor.parse_relationship_data(
        {"source": "Vendor Alpha", "target": "C#", "type": "USES"}, unit, index
    )
    assert uses_cpp is not None and uses_csharp is not None
    assert (uses_cpp.source_id, uses_cpp.target_id) == (vendor.id, cpp.id)
    assert (uses_csharp.source_id, uses_csharp.target_id) == (vendor.id, csharp.id)
    assert uses_cpp.id != uses_csharp.id
    # Endpoint names carry the display form too.
    assert uses_csharp.target_name == "C#"


def test_relationship_resolves_renamed_entity_through_key_index(
    processor, unit
) -> None:
    # A gleaner correction renames in place and keeps the id, so the id is no
    # longer derivable from the name; the key index must still find it.
    entity = Entity(id="kept-id", name="Acme Corporation")
    index = processor.build_entity_key_index([entity])
    rel = processor.parse_relationship_data(
        {"source": "ACME Corporation", "target": "Buyer", "type": "SELLS_TO"},
        unit,
        index,
    )
    assert rel is not None and rel.source_id == "kept-id"


def test_relationship_without_local_entity_derives_matching_id(processor, unit) -> None:
    entity = _entity(processor, unit, "Section 4.2")
    rel = processor.parse_relationship_data(
        {"source": "Buyer", "target": "section 4.2", "type": "BOUND_BY"}, unit, {}
    )
    assert rel is not None and rel.target_id == entity.id


# --------------------------------------------------------------------------- #
# Downstream consumers stay consistent with the key
# --------------------------------------------------------------------------- #


def test_resolver_does_not_merge_symbol_names(processor, unit) -> None:
    resolver = EntityResolver(Config(), max_workers=1, use_process_pool=False)
    resolver.show_progress = False
    entities = [_entity(processor, unit, n) for n in ("C++", "C#", "C")]
    resolved, _, _ = resolver.resolve(entities)
    assert sorted(e.name for e in resolved) == ["C", "C#", "C++"]


def test_incremental_merge_uses_identity_key(processor, unit) -> None:
    old = [_entity(processor, unit, "C++"), _entity(processor, unit, "Acme Corp")]
    delta = [Entity(id="d1", name="C#"), Entity(id="d2", name="ACME Corp.")]
    merged, remap = merge_entities(old, delta)
    assert remap == {"d2": old[1].id}
    assert sorted(e.name for e in merged) == ["Acme Corp", "C#", "C++"]


def test_gleaner_duplicate_merge_uses_identity_key() -> None:
    entities = [
        Entity(id="a", name="C++"),
        Entity(id="b", name="C#"),
        Entity(id="c", name="Acme Corp"),
        Entity(id="d", name="ACME Corp."),
    ]
    unique, remap = GraphGleaner._merge_duplicate_entities(entities)
    assert sorted(e.id for e in unique) == ["a", "b", "c"]
    assert remap == {"d": "c"}
