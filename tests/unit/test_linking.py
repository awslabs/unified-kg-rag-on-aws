# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one name-to-entity linking rule (domain/retrieval/linking.py) local
search's focus seeds and the structured strategy's mentions share."""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.search_strategies.local_search import (
    LocalSearchStrategy,
)
from unified_kg_rag.domain.models import RetrievalResult
from unified_kg_rag.domain.retrieval.linking import choose_links, covers, name_terms

pytestmark = pytest.mark.unit


def _hit(name: str, score: float, source: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        content=name,
        score=score,
        source=source if source is not None else name.lower().replace(" ", "-"),
        retriever_type="entity",
        metadata={"name": name},
    )


@pytest.mark.parametrize(
    ("name", "mention", "covered"),
    [
        # One shared word is not the entity.
        ("Acme Supplies", "Zorblax Supplies", False),
        ("가나다 상사", "라마바 상사", False),
        ("가나다상사", "라마바 상사", False),
        ("XYZ电子", "ABC电子", False),
        ("东芝", "东芝电子", False),
        # Legal forms are no term: they never cover nor need covering.
        ("Acme Supplies Inc", "Zorblax Inc", False),
        ("Acme Corporation", "Acme Inc", True),
        ("株式会社サクラ", "サクラ", True),
        ("(주)가나다", "가나다 주식회사", True),
        # A part of the name, in any script, with or without spaces.
        ("Acme Supplies", "Acme", True),
        ("가나다 상사", "가나다", True),
        ("东芝电子", "东芝", True),
        ("三菱商事株式会社", "三菱", True),
        ("ACME supplies", "acme SUPPLIES", True),
        ("Acme", "", False),
    ],
)
def test_coverage_needs_every_term_of_the_mention(
    name: str, mention: str, covered: bool
) -> None:
    assert covers(name, mention) is covered


def test_dense_scripts_compare_bigrams_and_latin_words() -> None:
    assert name_terms("가나다 상사") == {"가나", "나다", "다상", "상사"}
    assert name_terms("ABC电子 Co., Ltd.") == {"abc", "电子"}
    assert name_terms("東") == {"東"}


def test_a_spelling_variant_covers_at_the_similarity() -> None:
    assert not covers("Bexley", "Bexly")
    assert covers("Bexley", "Bexly", 0.85)
    # Bigrams of different characters are never alike enough.
    assert not covers("가나다 상사", "가나라 상사", 0.85)
    assert not covers("Acme Supply", "Acme Supplies", 0.85)


def test_choose_links_prefers_every_exact_name() -> None:
    hits = [_hit("Acme Holdings", 1.0), _hit("ACME Corp.", 0.9, "e1")]
    hits.append(_hit("Acme Corp", 0.2, "e2"))
    assert [h.source for h in choose_links("acme corp", hits, limit=1)] == ["e1", "e2"]
    # limit 0: exact names only.
    assert choose_links("Acme", [_hit("Acme Holdings", 1.0)], limit=0) == []


def test_choose_links_keeps_covering_hits_above_the_floor() -> None:
    hits = [
        _hit("Acme Buyer", 0.0),
        _hit("Globex Supplies", 0.9),
        _hit("Acme Supplies", 0.5),
        _hit("Acme Parts", 0.4),
        _hit("Acme Retail", 0.3),
        _hit("Acme", 0.2, ""),
    ]
    chosen = choose_links("acme", hits, limit=2)
    # A zero score, a non-covering name and a hit without a source never link.
    assert [h.source for h in chosen] == ["acme-supplies", "acme-parts"]
    floored = choose_links("acme", hits, limit=5, floor=0.35)
    assert [h.source for h in floored] == ["acme-supplies", "acme-parts"]


def test_local_focus_seeds_link_by_the_same_rule() -> None:
    """The unify probe: an absent focus name sharing one word with an
    entity seeded it when it was the only focus (0.3 is its own best)."""
    focus = ["Zorblax Supplies"]
    per_focus = [[_hit("Acme Supplies", 0.3)]]
    assert LocalSearchStrategy.focus_seeds(focus, per_focus, 0.6, 0.85) == []
    focus, per_focus = ["Acme"], [[_hit("Acme Supplies", 0.3)]]
    seeds = LocalSearchStrategy.focus_seeds(focus, per_focus, 0.6, 0.85)
    assert [s.source for s in seeds] == ["acme-supplies"]
