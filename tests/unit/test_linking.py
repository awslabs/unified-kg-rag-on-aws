# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one name-to-entity linking rule (domain/retrieval/linking.py) local
search's focus seeds and the structured strategy's mentions share."""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.search_strategies.local_search import (
    LocalSearchStrategy,
)
from unified_kg_rag.domain.ingestion.names import compact_name_key
from unified_kg_rag.domain.models import RetrievalResult
from unified_kg_rag.domain.retrieval.linking import (
    choose_links,
    covers,
    hit_name,
    link_key,
    name_terms,
)

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


def test_a_word_keeps_its_marks_and_a_spaceless_word_reads_as_bigrams() -> None:
    # Regression (rv23): "\\w" leaves out Devanagari and Thai vowel signs,
    # so "टाटा" was the fragments "ट", "ट", and a Thai word, written without
    # spaces, was neither one word nor its bigrams.
    assert name_terms("टाटा मोटर्स") == {"टाटा", "मोटर्स"}
    assert name_terms("กรุงเทพ") == {"กร", "รุ", "ุง", "งเ", "เท", "ทพ"}


@pytest.mark.parametrize(
    ("name", "mention", "covered"),
    [
        # An absent name sharing a fragment or a word is not the entity.
        ("टी-सीरीज़", "टाटा", False),
        ("टीसीएस", "टाटा", False),
        ("पंजाब नेशनल बैंक", "भारतीय स्टेट बैंक", False),
        ("ธนาคารกสิกรไทย", "ธนาคารกรุงเทพ", False),
        # A part of the name.
        ("टाटा मोटर्स", "टाटा", True),
        ("रिलायंस इंडस्ट्रीज", "रिलायंस", True),
        ("ธนาคารกรุงเทพ", "กรุงเทพ", True),
    ],
)
@pytest.mark.parametrize("similarity", [1.0, 0.85])
def test_hindi_and_thai_names_link_by_their_words(
    name: str, mention: str, covered: bool, similarity: float
) -> None:
    assert covers(name, mention, similarity) is covered
    links = choose_links(
        mention, [_hit(name, 1.0)], policy="cover", limit=3, similarity=similarity
    )
    assert bool(links) is covered


@pytest.mark.parametrize(
    ("focus", "name"),
    [("भारतीय स्टेट बैंक", "पंजाब नेशनल बैंक"), ("ธนาคารกรุงเทพ", "ธนาคารกสิกรไทย")],
)
def test_the_rank_policy_refuses_a_hindi_or_thai_lookalike(
    focus: str, name: str
) -> None:
    assert choose_links(focus, [_hit(name, 1.0)], policy="rank", similarity=0.85) == []


def test_a_spelling_variant_covers_at_the_similarity() -> None:
    assert not covers("Bexley", "Bexly")
    assert covers("Bexley", "Bexly", 0.85)
    # Bigrams of different characters are never alike enough.
    assert not covers("가나다 상사", "가나라 상사", 0.85)
    assert not covers("Acme Supply", "Acme Supplies", 0.85)


def test_choose_links_prefers_every_exact_name() -> None:
    hits = [_hit("Acme Holdings", 1.0), _hit("ACME Corp.", 0.9, "e1")]
    hits.append(_hit("Acme Corp", 0.2, "e2"))
    assert [
        h.source for h in choose_links("acme corp", hits, policy="cover", limit=1)
    ] == ["e1", "e2"]
    # limit 0: exact names only.
    assert (
        choose_links("Acme", [_hit("Acme Holdings", 1.0)], policy="cover", limit=0)
        == []
    )


def test_choose_links_keeps_covering_hits_above_the_floor() -> None:
    hits = [
        _hit("Acme Buyer", 0.0),
        _hit("Globex Supplies", 0.9),
        _hit("Acme Supplies", 0.5),
        _hit("Acme Parts", 0.4),
        _hit("Acme Retail", 0.3),
        _hit("Acme", 0.2, ""),
    ]
    chosen = choose_links("acme", hits, policy="cover", limit=2)
    # A zero score, a non-covering name and a hit without a source never link.
    assert [h.source for h in chosen] == ["acme-supplies", "acme-parts"]
    floored = choose_links("acme", hits, policy="cover", limit=5, floor=0.35)
    assert [h.source for h in floored] == ["acme-supplies", "acme-parts"]


def _old_focus_seeds(
    focus: list[str], per_focus: list[list[RetrievalResult]], ratio: float
) -> list[RetrievalResult]:
    """Local's seed rule before choose_links (the oracle its seeds keep)."""
    best = max((h.score or 0.0 for hits in per_focus for h in hits), default=0.0)
    seeds: dict[str, RetrievalResult] = {}
    for name, hits in zip(focus, per_focus, strict=True):
        key = compact_name_key(name)
        chosen = [h for h in hits if key and compact_name_key(hit_name(h)) == key]
        if not chosen and hits and (hits[0].score or 0.0) >= ratio * best:
            chosen = hits[:1]
        for h in chosen:
            seeds.setdefault(str(h.source), h)
    return list(seeds.values())


# Semantic, alias and spelling hits local seeds (no shared word, or a
# part of the name): the rank policy keeps every one, as before.
ALIASES = [
    ("NYC", "New York City"),
    ("U.S.", "United States"),
    ("Bexly", "Bexley"),
    ("The Beatles", "Beatles"),
    ("Acme", "Acme Supplies"),
    ("가나다", "가나다 상사"),
    ("サクラ", "株式会社サクラ"),
    ("car maker", "Automobile manufacturer"),
    ("东芝", "东芝电子"),
]


@pytest.mark.parametrize(("focus", "name"), ALIASES)
@pytest.mark.parametrize("score", [0.3, 0.9])
def test_local_seeds_semantic_and_alias_hits_as_before(
    focus: str, name: str, score: float
) -> None:
    per_focus = [[_hit(name, score), _hit("Other", score / 2)], [_hit("Buyer", 0.5)]]
    focus_names = [focus, "Buyer"]
    new = LocalSearchStrategy.focus_seeds(focus_names, per_focus, 0.6)
    assert [h.source for h in new] == [
        h.source for h in _old_focus_seeds(focus_names, per_focus, 0.6)
    ]
    assert new[0].metadata["name"] == name


@pytest.mark.parametrize(
    ("focus", "name"),
    [("Zorblax Supplies", "Acme Supplies"), ("라마바 상사", "가나다 상사")],
)
def test_local_never_seeds_a_one_word_lookalike(focus: str, name: str) -> None:
    """The unify probe: an absent focus name sharing one word with an
    entity seeded it when it was the only focus (0.3 is its own best)."""
    per_focus = [[_hit(name, 0.3)]]
    assert [h.source for h in _old_focus_seeds([focus], per_focus, 0.6)]
    assert LocalSearchStrategy.focus_seeds([focus], per_focus, 0.6) == []


def test_the_rank_policy_breaks_a_top_tie_by_coverage() -> None:
    hits = [_hit("Acme Holdings", 0.8), _hit("Acme Retail Group", 0.8)]
    chosen = choose_links("acme retail", hits, policy="rank", floor=0.5)
    assert [h.source for h in chosen] == ["acme-retail-group"]
    # Below the floor nothing links; without a tie the top hit stands.
    assert choose_links("acme retail", hits, policy="rank", floor=0.9) == []
    hits = [_hit("Initech", 0.9), _hit("Acme Retail Group", 0.8)]
    assert [h.source for h in choose_links("acme retail", hits, policy="rank")] == [
        "initech"
    ]


@pytest.mark.parametrize(
    ("mention", "stored"),
    [
        # Regression (rv8 retrieval M4): the katakana middle dot is a word
        # separator a query writes as nothing, a space or the dot.
        ("オリオンラボ", "オリオン・ラボ"),
        ("オリオン ラボ", "オリオン・ラボ"),
        ("オリオン･ラボ", "オリオン・ラボ"),
        ("홍길동", "홍·길동"),
        # Arabic: alef with or without hamza, alef maksura, short vowels,
        # tatweel and the leading "company".
        ("اكمي للروبوتات", "أكمي للروبوتات"),
        ("مستشفى النور", "مستشفي النور"),
        ("أَكْمِي للروبوتات", "أكمي للروبوتات"),
        ("أكمـــي للروبوتات", "أكمي للروبوتات"),
        ("شركة أكمي", "أكمي"),
        ("أكمي", "شركة أكمي"),
        # Regression (rv9 query L6): Russian writes ё as е at will.
        ("Федоров", "Фёдоров"),
        ("ФЁДОРОВ", "Федоров"),
    ],
)
@pytest.mark.parametrize("policy", ["cover", "rank"])
def test_a_written_variant_of_the_name_links_exactly(
    mention: str, stored: str, policy: str
) -> None:
    hits = [_hit("Unrelated Corp", 0.9, "e0"), _hit(stored, 0.1, "e1")]
    linked = choose_links(mention, hits, policy=policy)  # type: ignore[arg-type]
    assert [h.source for h in linked] == ["e1"]


def test_the_fold_is_query_time_only() -> None:
    # Stored keys and ids keep the letters as written.
    assert compact_name_key("أكمي") != compact_name_key("اكمي")
    assert link_key("أكمي") == link_key("اكمي") == "اكمي"
    # A designator alone stays (never down to nothing); a middle dot
    # between Latin letters is no separator.
    assert link_key("شركة") == "شركة"
    assert link_key("a·b") == "a·b"


@pytest.mark.parametrize("policy", ["cover", "rank"])
def test_a_name_written_as_asked_links_alone(policy: str) -> None:
    # Regression (rv9 query L5): the link_key fold made "شركة النور" (the
    # company) and "النور" one name, so both linked.
    hits = [_hit("النور", 0.9, "e0"), _hit("شركة النور", 0.5, "e1")]
    linked = choose_links("شركة النور", hits, policy=policy)  # type: ignore[arg-type]
    assert [h.source for h in linked] == ["e1"]
    # Without a hit written as asked, the fold still links.
    linked = choose_links("شركة النور", hits[:1], policy=policy)  # type: ignore[arg-type]
    assert [h.source for h in linked] == ["e0"]


@pytest.mark.parametrize("policy", ["cover", "rank"])
@pytest.mark.parametrize(
    ("mention", "names"),
    [
        ("Acme GmbH", ("Acme", "Acme GmbH")),
        ("ООО «Ромашка»", ("Ромашка", "ООО Ромашка")),
        ("(주)가나다", ("가나다", "㈜가나다")),
    ],
)
def test_the_most_literal_key_that_finds_a_hit_links(
    mention: str, names: tuple[str, str], policy: str
) -> None:
    # Two entities stored apart whose compact keys are one (another type
    # kept them apart): the one written as asked links alone.
    hits = [_hit(names[0], 0.9, "e0"), _hit(names[1], 0.5, "e1")]
    linked = choose_links(mention, hits, policy=policy)  # type: ignore[arg-type]
    assert [h.source for h in linked] == ["e1"]


def test_coverage_leaves_latin_accents_out() -> None:
    # Regression (rv9 query L6): "Cafe" never covered "Café".
    assert covers("Café Central Wien", "Cafe Central")
    assert covers("Cafe Central Wien", "Café Central")
    linked = choose_links(
        "Cafe Central", [_hit("Café Central Wien", 0.5, "e1")], policy="cover"
    )
    assert [h.source for h in linked] == ["e1"]
    # An exact link keeps them: "Cafe" is not written as "Café".
    assert link_key("Café") != link_key("Cafe")


def test_a_word_is_no_legal_form_without_its_accents() -> None:
    # "Sá" read without its accent was the form "SA" and left out of the
    # terms, so the person "Marcelo" covered "Marcelo Sá".
    assert name_terms("Marcelo Sá") == {"marcelo", "sa"}
    assert not covers("Marcelo", "Marcelo Sá")
    assert name_terms("Acme S.A.") == {"acme"}


@pytest.mark.parametrize(
    ("mention", "name"),
    [
        ("Nguyen Van An", "Nguyễn Văn An"),
        ("Cong ty Thanh Cong", "Công ty Thành Công"),
        ("Cafe de Flore", "Café de Flore"),
        ("Muller Logistik", "Müller Logistik"),
        # Regression (rv11 query M3): letters no decomposition takes a mark
        # from (a stroke, a ligature, a dotless i) were never folded.
        ("Dang Van Lam", "Đặng Văn Lâm"),
        ("Do Hung Dung", "Đỗ Hùng Dũng"),
        ("Lodz Logistics", "Łódź Logistics"),
        ("Yildiz Holding", "Yıldız Holding"),
        ("Soren Kierkegaard Institut", "Søren Kierkegaard Institut"),
        ("Aero Faergeri", "Ærø Færgeri"),
        ("Thorsdottir Fisheries", "Þórsdóttir Fisheries"),
    ],
)
def test_the_rank_policy_leaves_latin_accents_out(mention: str, name: str) -> None:
    # Regression (rv10 query M5): coverage left the accents out, the
    # lookalike test did not, so local search rejected its right top hit.
    hits = [_hit(name, 0.9, "e1"), _hit("Other", 0.4, "e2")]
    assert [h.source for h in choose_links(mention, hits, policy="rank")] == ["e1"]
    # A one-word lookalike is still none ("Zorblax Flore").
    lookalike = f"Zorblax {mention.split()[-1]}"
    assert choose_links(lookalike, hits, policy="rank") == []
    assert [h.source for h in choose_links(mention, hits, policy="cover")] == ["e1"]
