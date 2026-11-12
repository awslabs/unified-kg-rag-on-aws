# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The section types and their traits: every kind has a row, and every
reader (the evaluator's file ranking, the console's passage links, the MCP
source view, the type quota) reads the trait instead of a list of kinds."""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.models import (
    Citation,
    RetrievalResult,
    SectionType,
)
from unified_kg_rag.domain.models.config import TypeQuotaConfig
from unified_kg_rag.domain.retrieval.section_types import SECTION_TRAITS
from unified_kg_rag.evaluation.retrieval_evaluator import ranked_files

pytestmark = pytest.mark.unit


def test_every_section_type_has_traits() -> None:
    assert set(SECTION_TRAITS) == set(SectionType)


def test_wire_values_are_unchanged() -> None:
    # Stored metadata and the APIs carry these strings.
    assert {t.value for t in SectionType} == {
        "claim",
        "community",
        "entity",
        "general",
        "relationship",
        "summary",
        "text",
        "query_result",
    }
    assert str(SectionType.TEXT) == "text" and f"{SectionType.ENTITY}" == "entity"


def test_unknown_kind_reads_as_general() -> None:
    result = RetrievalResult(content="c", score=1.0, retriever_type="Plugin_Kind")
    assert result.retriever_type is SectionType.GENERAL
    assert RetrievalResult(
        content="c", score=1.0, retriever_type="ENTITY"
    ).retriever_type is (SectionType.ENTITY)
    assert Citation(marker=1, section_type="nope").section_type is SectionType.GENERAL
    assert SectionType.of(None) is SectionType.GENERAL


def test_only_a_passage_names_its_own_file() -> None:
    assert [t for t in SectionType if not t.lineage_attributed] == [
        SectionType.GENERAL,
        SectionType.TEXT,
    ]


@pytest.mark.parametrize("kind", [t for t in SectionType if t.lineage_attributed])
def test_lineage_sources_rank_after_the_passages(kind: SectionType) -> None:
    # Regression: query_result was missing from the evaluator's list, so a
    # structured run's lineage files ranked ahead of the passages'.
    files = ranked_files([["lineage.txt"], ["passage.txt"]], [kind.value, "text"])
    assert [sorted(f)[0] for f in files][0].startswith("passage")


def test_query_result_lineage_ranks_like_an_entity() -> None:
    sources = [["lineage.txt"], ["passage.txt"]]
    assert ranked_files(sources, ["query_result", "text"]) == ranked_files(
        sources, ["entity", "text"]
    )


def test_stated_by_passages_includes_claims_not_reports() -> None:
    stated = {t for t in SectionType if t.stated_by_passages}
    assert stated == {
        SectionType.ENTITY,
        SectionType.RELATIONSHIP,
        SectionType.CLAIM,
        SectionType.QUERY_RESULT,
    }


def test_type_quota_slots_are_section_types() -> None:
    slots = TypeQuotaConfig().slots(10)
    assert set(slots) == {"text", "entity", "relationship", "community", "claim"}
    assert all(SectionType(k) for k in slots)
