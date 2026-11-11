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
from unified_kg_rag.domain.models import Config, Entity, TextUnit, TextUnitExtraction
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
        # Quote marks of every script carry no identity.
        ("ООО «Ромашка»", "ооо ромашка"),
        ("„Vendor“ GmbH", "vendor gmbh"),
        ("「供給者」", "供給者"),
        ("《民法典》", "民法典"),  # a Chinese title mark
        ("〈가나다〉", "가나다"),
        ('בע"מ', "בעמ"),
        ("בע״מ", "בעמ"),  # the gershayim written for '"'
        # Every dash joins words as "-" does.
        ("E‐Mail–Vendor", "e mail vendor"),
        ("A−B", "a b"),
        ("가나-다라", "가나 다라"),
        ("エー゠ビー", "エー ビー"),
        # A middle dot between dense letters separates the words of a name.
        ("ジョン・スミス", "ジョンスミス"),
        ("ｼﾞｮﾝ･ｽﾐｽ", "ジョンスミス"),  # half-width, as NFKC reads it
        ("한·미 협정", "한미 협정"),
        ("中·美", "中美"),
        ("l·l", "l·l"),
        # Arabic short vowels and Hebrew points are written at will.
        ("مُحَمَّد", "محمد"),
        ("שָׁלוֹם", "שלום"),
        # A wave dash is a range, not a hyphen.
        ("第1〜3条", "第1〜3条"),
        # A minus sign opening the name is a sign, not a hyphen.
        ("−40 °C", "-40 °c"),
        ("-5%", "-5%"),
        ("«–40»", "-40"),
        ("Acme−40", "acme 40"),
        # What a writer types for an apostrophe is one; the ʻokina a letter.
        ("McDonald´s", "mcdonalds"),
        ("McDonaldʼs", "mcdonalds"),
        ("O′Brien", "obrien"),
        ("Hawaiʻi", "hawaiʻi"),
        # Invisible format characters carry nothing; the joiners shape letters.
        ("Wi\u00adFi", "wifi"),
        ("Ven\u200bdor", "vendor"),
        ("\ufeffVendor\u2060", "vendor"),
        ("\u200f\u202bשלום\u202c", "שלום"),
        ("क्\u200dष", "क्\u200dष"),
        ("می\u200cخواهم", "می\u200cخواهم"),
        # Korean writes "ㆍ" for a middle dot between syllables.
        ("한ㆍ미 FTA", "한미 fta"),
        ("ㆍ미", "ᆞ미"),
        # The ideographic full stop ends a name as "." does.
        ("「君の名は。」", "君の名は"),
        # Quotes go before the dots are joined: a key is its own key.
        ("漢・》漢", "漢漢"),
        # A dot beside another stays: the katakana one is in the Kana block.
        ("가・·漢", "가・·漢"),
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
    record = TextUnitExtraction(
        text_unit_id="t1",
        entities=[
            Entity(id="a", name="C++"),
            Entity(id="b", name="C#"),
            Entity(id="c", name="Acme Corp"),
        ],
    )
    GraphGleaner._add_to_record(record, [Entity(id="d", name="ACME Corp.")], [])
    assert sorted(e.id for e in record.entities) == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# Display name of spellings that tie
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("spellings", "shown"),
    [
        (["「토지」", "토지"], "토지"),
        (["Ｗｉ－Ｆｉ", "Wi-Fi"], "Wi-Fi"),
        (["“Vendor”", "Vendor"], "Vendor"),
    ],
)
def test_a_tie_shows_the_spelling_its_key_writes_as_it_is(
    spellings: list[str], shown: str
) -> None:
    # One id, one text unit each: the smallest name alone put a quote mark
    # or a full-width letter first. Either order gives one name.
    from unified_kg_rag.domain.ingestion.contributions import (
        merge_entity_contributions,
    )

    key = entity_key(shown)
    assert {entity_key(s) for s in spellings} == {key}
    for order in (spellings, spellings[::-1]):
        entities = [
            Entity(id=key, name=s, type="CONCEPT", text_unit_ids=[f"t{i}"])
            for i, s in enumerate(order)
        ]
        assert EntityResolver._best_supported_name(entities) == shown
        (merged,) = merge_entity_contributions(entities)
        assert merged.name == shown
