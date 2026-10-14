# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ranking parity of the embedded SQLite engine with a real OpenSearch.

The same synthetic corpus, embedded by ``HashingEmbeddingFactory``, goes into
the SQLite engine and the OpenSearch container of ``docker/compose.local.yaml``
through the unchanged ``OpenSearchIndexer``; the same queries then go through
the unchanged ``OpenSearchRetriever``. Lexical search (our BM25 against
Lucene's) must rank the same top 10 with scores within 5 %; hybrid search
(our min-max fusion against neural-search's, over exact against approximate
kNN) must share 90 % of its top 10 with scores within 0.05.

The corpus is ASCII under the ``standard`` analyzer, which both engines
tokenize identically, and its words are at least three edits apart, so the
``fuzziness: AUTO`` that OpenSearch applies (and the embedded engine ignores)
expands no query term. A hybrid search fetches 50 hits, of which the top 10 are compared, and its
queries match at most 50 documents lexically: OpenSearch 2.13
(neural-search) was observed to min-max normalize a sub-query that matches more
documents than the request's ``size`` over hits outside its top ``size``, so
the two engines would normalize over different hits. Skipped unless
``LOCAL_STORES=1``.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.fixtures.store_backends import StoreHarness, open_harness
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchType,
    TextUnit,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("LOCAL_STORES") != "1",
        reason="set LOCAL_STORES=1 with docker/compose.local.yaml running",
    ),
]

_TOP = 10
# Hits a hybrid search fetches (and each sub-query normalizes over).
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


def _standard_analyzer(config: Config) -> None:
    config.indexing.opensearch.language_analyzers = {}


@pytest.fixture
def stores(tmp_path: Path) -> Iterator[tuple[StoreHarness, StoreHarness]]:
    units, _, _ = _corpus()
    sqlite = open_harness("sqlite", tmp_path)
    container = open_harness("container", tmp_path)
    pair = []
    try:
        for opened in (sqlite, container):
            harness = next(opened).derive(_standard_analyzer)
            harness.initialize()
            stats = harness.vector_indexer.index_text_units(units)
            assert stats.failed_items == 0, stats.errors
            pair.append(harness)
        yield pair[0], pair[1]
    finally:
        for opened in (sqlite, container):
            opened.close()


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


def test_lexical_ranking_matches_opensearch(
    stores: tuple[StoreHarness, StoreHarness],
) -> None:
    sqlite, container = stores
    _, common, _ = _corpus()
    for ours, theirs in zip(
        _search(sqlite, SearchType.LEXICAL, common),
        _search(container, SearchType.LEXICAL, common),
        strict=True,
    ):
        assert theirs, "a query without hits compares nothing"
        assert [r.source for r in ours] == [r.source for r in theirs]
        for mine, reference in zip(ours, theirs, strict=True):
            assert mine.score == pytest.approx(reference.score, rel=0.05)


def test_hybrid_ranking_matches_opensearch(
    stores: tuple[StoreHarness, StoreHarness],
) -> None:
    sqlite, container = stores
    _, _, mid = _corpus()
    assert len(mid) >= 8
    for ours, theirs in zip(
        _search(sqlite, SearchType.HYBRID, mid, _HYBRID_SIZE),
        _search(container, SearchType.HYBRID, mid, _HYBRID_SIZE),
        strict=True,
    ):
        mine = {r.source: r.score or 0.0 for r in ours[:_TOP]}
        reference = {r.source: r.score or 0.0 for r in theirs[:_TOP]}
        shared = mine.keys() & reference.keys()
        assert len(shared) >= 0.9 * _TOP
        for source in shared:
            assert mine[source] == pytest.approx(reference[source], abs=0.05)
