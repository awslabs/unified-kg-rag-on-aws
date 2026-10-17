# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ranking parity of the embedded SQLite engine with a real OpenSearch.

The same synthetic corpus, embedded by ``HashingEmbeddingFactory``, goes into
the SQLite engine and the OpenSearch container of ``docker/compose.local.yaml``
through the unchanged ``OpenSearchIndexer``; the same queries then go through
the unchanged ``OpenSearchRetriever``. Lexical search (our BM25 against
Lucene's) must rank the same top 10 with scores within 5 %; hybrid search
(the retriever's min-max fusion of a lexical and a kNN search, exact kNN
against approximate) must rank the same top 10 with scores within 0.01.

The corpus is ASCII under the ``standard`` analyzer, which both engines
tokenize identically, and its words are at least three edits apart, so the
``fuzziness: AUTO`` that OpenSearch applies (and the embedded engine ignores)
expands no query term. A second, English corpus runs the lexical comparison
under the ``english`` analyzer: inflected forms, Lucene's stop words and words
other stop lists drop ("amount", "bill", "system"), with stems three edits
apart. Hybrid queries include common words that match far more documents
than the search returns: OpenSearch 2.13's own ``hybrid`` query normalized
such a lexical sub-query over a document outside its hits (the last-indexed
match), which the retriever's client-side fusion does not depend on.
Skipped unless
``LOCAL_STORES=1``.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.fixtures.store_backends import StoreHarness, open_harness
from unified_kg_rag.adapters.local.search_engine.analyzers import get_analyzer
from unified_kg_rag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchType,
    TextUnit,
)
from unified_kg_rag.domain.retrieval.hybrid_normalization import combine_min_max

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("LOCAL_STORES") != "1",
        reason="set LOCAL_STORES=1 with docker/compose.local.yaml running",
    ),
]

_TOP = 10
# Hits a hybrid search fetches (and each of its searches normalizes over).
_HYBRID_SIZE = 50


def _distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def _vocabulary(rng: random.Random, size: int) -> list[str]:
    words: list[str] = []
    while len(words) < size:
        word = "".join(rng.choice("bcdfghklmnprstvz") for _ in range(rng.randint(6, 8)))
        if all(_distance(word, other) >= 3 for other in words):
            words.append(word)
    return words


def _corpus() -> tuple[list[TextUnit], list[str], list[str]]:
    """Documents, queries of common words, and queries of mid-frequency words
    (each lexically matching at most ``_HYBRID_SIZE`` documents)."""
    rng = random.Random(91)
    words = _vocabulary(rng, 60)
    # Zipf-like word frequencies, so document frequencies (and idf) vary.
    weights = [1.0 / (rank + 1) for rank in range(len(words))]
    texts = [rng.choices(words, weights, k=rng.randint(4, 14)) for _ in range(150)]
    units = [
        TextUnit(id=f"d{i:03d}", text=" ".join(text)) for i, text in enumerate(texts)
    ]
    common = [" ".join(rng.sample(words[:30], k=rng.randint(1, 3))) for _ in range(12)]
    matching = {
        word: {i for i, text in enumerate(texts) if word in text} for word in words
    }
    mid = [word for word in words if 5 <= len(matching[word]) <= 25]
    pairs = [
        f"{a} {b}"
        for a, b in zip(mid, mid[1:], strict=False)
        if len(matching[a] | matching[b]) <= _HYBRID_SIZE
    ]
    return units, common, mid[:8] + pairs[:4]


# Inflected forms of one stem each, the stems at least three edits apart.
_ENGLISH_WORDS = [
    ["amount", "amounts"],
    ["bill", "bills", "billed"],
    ["interest", "interests"],
    ["system", "systems"],
    ["computer", "computers"],
    ["detail", "details", "detailed"],
    ["supply", "supplies", "supplied"],
    ["vendor", "vendors"],
    ["warehouse", "warehouses"],
    ["invoice", "invoices", "invoiced"],
    ["shipping", "shipped", "ships"],
    ["payment", "payments"],
    ["fire", "fires"],
    ["contract", "contracts"],
    ["delivery", "deliveries"],
    ["quarter", "quarters"],
]
# Lucene's stop words: dropped from documents and queries alike.
_STOP = ["the", "a", "of", "and", "is", "to", "with", "their"]


def _english_corpus() -> tuple[list[TextUnit], list[str]]:
    rng = random.Random(7)
    weights = [1.0 / (rank + 1) for rank in range(len(_ENGLISH_WORDS))]
    texts = []
    for _ in range(150):
        words = [
            rng.choice(forms)
            for forms in rng.choices(_ENGLISH_WORDS, weights, k=rng.randint(4, 12))
        ]
        texts.append(words + rng.sample(_STOP, k=3))
    units = [
        TextUnit(id=f"e{i:03d}", text=" ".join(rng.sample(text, k=len(text))))
        for i, text in enumerate(texts)
    ]
    queries = ["interest bill amount", "fire system", "computer name detail"] + [
        " ".join(
            [rng.choice(forms) for forms in rng.sample(_ENGLISH_WORDS, k=2)]
            + rng.sample(_STOP, k=1)
        )
        for _ in range(9)
    ]
    return units, queries


def _analyzer(name: str) -> Callable[[Config], None]:
    def configure(config: Config) -> None:
        config.indexing.opensearch.language_analyzers = {}
        config.indexing.opensearch.default_analyzer = name

    return configure


@contextmanager
def _indexed_pair(
    tmp_path: Path, units: list[TextUnit], analyzer: str
) -> Iterator[tuple[StoreHarness, StoreHarness]]:
    sqlite = open_harness("sqlite", tmp_path)
    container = open_harness("container", tmp_path)
    pair = []
    try:
        for opened in (sqlite, container):
            harness = next(opened).derive(_analyzer(analyzer))
            harness.initialize()
            stats = harness.vector_indexer.index_text_units(units)
            assert stats.failed_items == 0, stats.errors
            pair.append(harness)
        yield pair[0], pair[1]
    finally:
        for opened in (sqlite, container):
            opened.close()


@pytest.fixture
def stores(tmp_path: Path) -> Iterator[tuple[StoreHarness, StoreHarness]]:
    units, _, _ = _corpus()
    with _indexed_pair(tmp_path, units, "standard") as pair:
        yield pair


def _search(
    h: StoreHarness, search_type: SearchType, queries: list[str], size: int = _TOP
) -> list[list[RetrievalResult]]:
    prefix = h.config.indexing.opensearch.text_units_index_prefix
    return h.retrieve(
        RetrieverRole.DOCUMENT,
        *(
            SearchQuery(
                query=text,
                search_type=search_type,
                index_prefixes=prefix,
                top_k=size,
                retrieval_multiplier=1,
            )
            for text in queries
        ),
    )


def _assert_same_lexical_ranking(
    sqlite: StoreHarness, container: StoreHarness, queries: list[str]
) -> None:
    for ours, theirs in zip(
        _search(sqlite, SearchType.LEXICAL, queries),
        _search(container, SearchType.LEXICAL, queries),
        strict=True,
    ):
        assert theirs, "a query without hits compares nothing"
        assert [r.source for r in ours] == [r.source for r in theirs]
        for mine, reference in zip(ours, theirs, strict=True):
            assert mine.score == pytest.approx(reference.score, rel=0.05)


def test_lexical_ranking_matches_opensearch(
    stores: tuple[StoreHarness, StoreHarness],
) -> None:
    _, common, _ = _corpus()
    _assert_same_lexical_ranking(*stores, common)


def test_english_analyzer_ranking_matches_opensearch(tmp_path: Path) -> None:
    units, queries = _english_corpus()
    with _indexed_pair(tmp_path, units, "english") as (sqlite, container):
        _assert_same_lexical_ranking(sqlite, container, queries)


def test_hybrid_ranking_matches_opensearch(
    stores: tuple[StoreHarness, StoreHarness],
) -> None:
    sqlite, container = stores
    _, common, mid = _corpus()
    assert len(mid) >= 8
    queries = common + mid
    for ours, theirs in zip(
        _search(sqlite, SearchType.HYBRID, queries, _HYBRID_SIZE),
        _search(container, SearchType.HYBRID, queries, _HYBRID_SIZE),
        strict=True,
    ):
        assert len(theirs) >= _TOP
        assert [r.source for r in ours[:_TOP]] == [r.source for r in theirs[:_TOP]]
        for mine, reference in zip(ours[:_TOP], theirs[:_TOP], strict=True):
            assert mine.score == pytest.approx(reference.score, abs=0.01)


def test_hybrid_normalizes_each_search_over_its_own_hits(tmp_path: Path) -> None:
    # Forty documents match "widget"; the search returns five. The last one
    # indexed is the best lexical match, not the worst: OpenSearch 2.13's
    # hybrid query min-max normalized the lexical side against it whenever
    # it was outside the top hits, skewing every fused score. Each side is
    # normalized over the hits it fetched (the normalization window, at
    # least), then cut to the five asked for, on both stores.
    rng = random.Random(5)
    units = [
        TextUnit(
            id=f"w{i:02d}",
            text=" ".join(
                ["widget"] * (12 if i == 39 else rng.randint(1, 8))
                + ["filler"] * rng.randint(1, 20)
            ),
        )
        for i in range(40)
    ]
    size = 5
    with _indexed_pair(tmp_path, units, "standard") as pair:
        found = {
            h.kind: _search(h, SearchType.HYBRID, ["widget"], size)[0] for h in pair
        }
        window = max(size, OpenSearchRetriever.HYBRID_NORMALIZATION_WINDOW)
        lexical, vector = (
            _search(pair[1], search_type, ["widget"], window)[0]
            for search_type in (SearchType.LEXICAL, SearchType.VECTOR)
        )
    weights = pair[1].config.search.hybrid
    expected = combine_min_max(
        [
            [(r.source, r.score or 0.0) for r in lexical],
            [(r.source, r.score or 0.0) for r in vector],
        ],
        [weights.lexical_weight, weights.vector_weight],
    )[:size]
    for hits in found.values():
        assert [r.source for r in hits] == [source for source, _ in expected]
        for hit, (_, score) in zip(hits, expected, strict=True):
            assert hit.score == pytest.approx(score, abs=0.01)


def test_vector_search_recall_against_exact_knn(tmp_path: Path) -> None:
    # The lucene engine ignores index.knn.algo_param.ef_search and explores k
    # candidates: with k = the 10 hits requested, OpenSearch found about 91 %
    # of the exact top 10 here; with k = ef_search (100), all but a few.
    rng = random.Random(3)
    words = _vocabulary(rng, 400)
    units = [
        TextUnit(id=f"u{i:05d}", text=" ".join(rng.choices(words, k=12)))
        for i in range(10_000)
    ]
    queries = [" ".join(rng.choices(words, k=4)) for _ in range(30)]
    with _indexed_pair(tmp_path, units, "standard") as (sqlite, container):
        exact = _search(sqlite, SearchType.VECTOR, queries)
        approximate = _search(container, SearchType.VECTOR, queries)
    found = sum(
        len({r.source for r in ours} & {r.source for r in theirs})
        for ours, theirs in zip(exact, approximate, strict=True)
    )
    assert found / (_TOP * len(queries)) >= 0.98


def test_hybrid_keeps_the_last_lexical_hit(tmp_path: Path) -> None:
    # Five documents match "widget price" and the search returns four.
    # OpenSearch 2.13's hybrid query dropped the size-th lexical hit of such
    # a sub-query and normalized over the worst match instead.
    units = [
        TextUnit(
            id=f"t{i}",
            text="widget price " * (5 - i) + "filler about the Buyer contract " * i,
        )
        for i in range(5)
    ] + [TextUnit(id="t-noise", text="Courier schedules nothing relevant here.")]
    with _indexed_pair(tmp_path, units, "standard") as (sqlite, container):
        (ours,), (theirs,) = (
            _search(h, SearchType.HYBRID, ["widget price"], 4)
            for h in (sqlite, container)
        )
    assert [r.source for r in theirs] == [r.source for r in ours]
    for mine, reference in zip(ours, theirs, strict=True):
        assert mine.score == pytest.approx(reference.score, abs=0.01)


# In-word punctuation, possessives, stop words and inflections, in several
# scripts (none of the Han, Hangul and Kana the analyzers bigram).
_ANALYZED = [
    "The Vendor's suppliers are shipping widgets",
    "don't 1,000.50 U.S.A. e-mail a:b it's VENDOR'S x＇s o’neil’s",
    "3.14 a1.b 1.a x_y co-op 'quoted' rock'n'roll e.g. 2024.01.15 $5,000.00",
    "rate: 5%; term:3 years. Fee:10 one..two a.,b 1.,2 end. 12:30 ab:cd:",
    "Interest on the bill amount is due; the fire system's computer name",
    "naïve café's résumé",
    # Combining marks (and NFD Latin), spaceless scripts, in-word punctuation
    # of other scripts, format characters, connectors and symbols.
    "हिन्दी भाषा में अनुबंध की शर्तें বাংলা চুক্তি தமிழ் ఒప్పందం",
    "العَرَبِيَّة الْعَقْد עִבְרִית חוֹזֶה צה\"ל ג'ורג' café résumé",
    "ภาษาไทย สัญญาที่ลงนาม ไทยabc ไทย๑๒ မြန်မာဘာသာ ខ្មែរ ລາວ",
    "a：b a‧b ١٬٠٠٠ 12؍34 co\u00adop a\u200bb 1\u202f000 _a_ __ foo‿bar",
    "x² H₂O ½ ① Ⓐb 𝐀b ﬁx pant-'s Exc.'s S&P500 us os Ds trekking contractibly",
    "١٢٣٫٤٥ ۳٫۵ ג׳ורג׳ İstanbul ΟΔΟΣ क:̪ a:́b",
]


@pytest.mark.parametrize("analyzer", ["standard", "english"])
def test_analyzers_tokenize_like_opensearch(tmp_path: Path, analyzer: str) -> None:
    opened = open_harness("container", tmp_path)
    try:
        client = next(opened).vector_indexer.opensearch_client.client
        for text in _ANALYZED:
            response = client.indices.analyze(body={"analyzer": analyzer, "text": text})
            expected = [token["token"] for token in response["tokens"]]
            assert get_analyzer(analyzer)(text) == expected, text
    finally:
        opened.close()
