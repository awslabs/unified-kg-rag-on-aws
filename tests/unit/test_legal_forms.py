# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one legal-form table of entity resolution and name grounding.

Regressions: resolution stripped only Inc, Ltd, LLC, Co and Corp (and nine
Korean, Japanese and Chinese forms), so "Acme GmbH", "Acme S.A.", "Acme Pty
Ltd" or "甲乙股份有限公司" stayed apart from "Acme" / "甲乙", and even "Acme
Corporation" from "Acme Corp", while name grounding read Corporation as Corp
from a table of its own. All names are synthetic.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.base_resolver import compact_name_key
from unified_kg_rag.domain.ingestion.entity_grounding import is_name_grounded
from unified_kg_rag.domain.ingestion.graph_resolver import EntityResolver
from unified_kg_rag.domain.languages.legal_forms import (
    latin_legal_form,
    strip_legal_forms,
)
from unified_kg_rag.domain.models import Config, Entity

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "name",
    [
        "Acme Inc",
        "Acme, Inc.",
        "Acme Incorporated",
        "Acme Corp.",
        "Acme Corporation",
        "Acme Co., Ltd.",
        "Acme Company",
        "Acme Limited",
        "Acme, LLC",
        "Acme L.L.C.",
        "Acme LLP",
        "Acme L.P.",
        "Acme PLC",
        "Acme Pty Ltd",
        "Acme Pte. Ltd.",
        "Acme GmbH",
        "Acme AG",
        "Acme S.A.",
        "Acme SA",
        "Acme S.A.S.",
        "Acme SARL",
        "Acme S.r.l.",
        "Acme S.p.A.",
        "Acme Ltda.",
        "Acme B.V.",
        "Acme N.V.",
        "Acme K.K.",
        "ＡＣＭＥ　ＧｍｂＨ",
    ],
)
def test_latin_designators_are_stripped(name: str) -> None:
    assert compact_name_key(name) == compact_name_key("Acme") == "acme"


@pytest.mark.parametrize(
    ("name", "bare"),
    [
        ("(주)가나다", "가나다"),
        ("㈜가나다", "가나다"),
        ("( 주 ) 가나다", "가나다"),
        ("가나다(주)", "가나다"),
        ("주식회사 가나다", "가나다"),
        ("가나다 유한책임회사", "가나다"),
        ("가나다 합자회사", "가나다"),
        ("(사)가나다", "가나다"),
        ("재단법인 가나다", "가나다"),
        ("株式会社甲乙", "甲乙"),
        ("甲乙株式會社", "甲乙"),
        ("㈱甲乙", "甲乙"),
        ("甲乙合同会社", "甲乙"),
        ("甲乙股份有限公司", "甲乙"),
        ("甲乙有限责任公司", "甲乙"),
        ("甲乙有限責任公司", "甲乙"),
        ("甲乙有限公司", "甲乙"),
    ],
)
def test_dense_script_designators_are_stripped(name: str, bare: str) -> None:
    assert compact_name_key(name) == compact_name_key(bare)


@pytest.mark.parametrize(
    "name",
    [
        "Taco",  # a designator is a whole word
        "Co Holdings",  # Latin designators trail
        "Acme Spa",  # "spa" is a word: S.p.A. only with its dots
        "Smith & Co",  # a firm's name, not "Smith"
        "Smith and Company",
        "The Company",  # a defined party, not "The"
        "Inc.",  # never stripped to nothing
        "주식회사",
        "股份有限公司",
        "Acme Inc (Korea)",  # not at the end
    ],
)
def test_names_that_keep_their_designator(name: str) -> None:
    folded = " ".join(name.casefold().split())
    assert strip_legal_forms(name) in (folded, folded.rstrip("."))


def test_designators_alone_keep_the_first() -> None:
    # Stripping stops before nothing is left.
    assert strip_legal_forms("Pty Ltd") == "pty"
    assert strip_legal_forms("(주)주식회사") == "주식회사"


@pytest.mark.parametrize(
    ("token", "canonical"),
    [
        ("corporation", "corp"),
        ("corp.", "corp"),
        ("incorporated", "inc"),
        ("company", "co"),
        ("limited", "ltd"),
        ("l.l.c.", "llc"),
        ("s.p.a.", "spa"),
        ("spa", None),
        ("acme", None),
    ],
)
def test_latin_legal_form(token: str, canonical: str | None) -> None:
    assert latin_legal_form(token) == canonical


def test_grounding_reads_one_form_in_either_spelling() -> None:
    assert is_name_grounded("Acme Corporation", "Acme Corp ships widgets.")
    assert is_name_grounded("Acme Ltd", "Acme Limited ships widgets.")
    assert not is_name_grounded("Acme GmbH", "Acme AG ships widgets.")


def _groups(names: dict[str, str]) -> list[list[str]]:
    resolver = EntityResolver(
        Config(), max_workers=1, use_process_pool=False, show_progress=False
    )
    entities = [
        Entity(
            id=BaseProcessor._generate_entity_id(name),
            name=name,
            type=entity_type,
            text_unit_ids=[f"t{i}"],
        )
        for i, (name, entity_type) in enumerate(names.items())
    ]
    return sorted(
        sorted(e.name for e in group)
        for group in resolver._group_similar_entities(entities)
    )


def test_resolution_merges_designator_variants_of_one_type() -> None:
    assert _groups(
        {"Acme GmbH": "organization", "Acme": "organization", "Acme AG": ""}
    ) == [["Acme", "Acme AG", "Acme GmbH"]]


def test_a_type_mismatch_still_blocks_the_merge() -> None:
    assert _groups({"Jefferson Co": "organization", "Jefferson": "person"}) == [
        ["Jefferson"],
        ["Jefferson Co"],
    ]
