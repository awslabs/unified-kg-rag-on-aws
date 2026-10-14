# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The embedded engine's BM25 against values computed by hand from OpenSearch's
default similarity (Lucene's ``LegacyBM25Similarity``: k1 = 1.2, b = 0.75, with
the (k1 + 1) factor)."""

from __future__ import annotations

import math

import pytest

from unified_kg_rag.adapters.local.search_engine.bm25 import (
    FieldIndex,
    idf,
    quantized_length,
    term_score,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def field() -> FieldIndex:
    index = FieldIndex()
    index.add("d1", ["a", "b"])
    index.add("d2", ["a", "c", "c"])
    index.add("d3", ["b"])
    index.add("d4", [])  # no term in the field: not counted in N or avgdl
    return index


def test_idf_is_lucene_formula() -> None:
    # ln(1 + (N - n + 0.5) / (n + 0.5)) with N = 3, n = 2: ln(1.6)
    assert idf(2, 3) == pytest.approx(0.4700036292457356)
    # A term in every document still has a positive idf: ln(1 + 0.5/3.5)
    assert idf(3, 3) == pytest.approx(math.log(1 + 0.5 / 3.5))


def test_field_statistics(field: FieldIndex) -> None:
    assert field.doc_count == 3
    assert field.avg_length == pytest.approx(2.0)  # 6 tokens / 3 documents


def test_scores_match_hand_computed_values(field: FieldIndex) -> None:
    # "a": n = 2 -> idf = ln(1.6) = 0.470004
    # d1: tf 1, dl 2: 0.470004 * 2.2 * 1 / (1 + 1.2 * (0.25 + 0.75 * 2/2))
    #   = 1.034008 / 2.2 = 0.470004
    # d2: tf 1, dl 3: 1.034008 / (1 + 1.2 * (0.25 + 0.75 * 1.5)) = / 2.65
    scores = field.score(["a"])
    assert scores == {
        "d1": pytest.approx(0.47000363),
        "d2": pytest.approx(0.39019169),
    }
    # "c": n = 1 -> idf = ln(1 + 2.5/1.5) = 0.980829; d2: tf 2, dl 3:
    # 0.980829 * 2.2 * 2 / (2 + 1.2 * 1.375) = 4.315649 / 3.65 = 1.182369
    assert field.score(["c"]) == {"d2": pytest.approx(1.18236974)}
    # Terms add up; a repeated query term counts per repetition.
    both = field.score(["a", "c"])
    assert both["d2"] == pytest.approx(0.39019169 + 1.18236974)
    assert field.score(["a", "a"])["d1"] == pytest.approx(2 * 0.47000363)
    assert field.score(["missing"]) == {}


def test_term_score_keeps_the_k1_plus_one_factor() -> None:
    # At dl == avgdl: idf * (k1 + 1) * tf / (tf + k1), as OpenSearch scores
    # (Lucene 8+'s BM25Similarity alone would give 1 / 2.2).
    assert term_score(1, 10, 10.0, 1.0) == pytest.approx(1.0)


def test_short_lengths_are_exact_and_long_ones_round_down() -> None:
    # Lucene's one-byte norm: exact below 24, then 3 mantissa bits.
    assert [quantized_length(n) for n in range(24)] == list(range(24))
    assert quantized_length(100) == 96
    assert quantized_length(1000) == 984
    previous = 0
    for n in range(1, 5000):
        q = quantized_length(n)
        assert previous <= q <= n
        previous = q


def test_document_length_uses_the_quantized_norm() -> None:
    index = FieldIndex()
    index.add("long", ["x"] + ["filler"] * 99)  # 100 tokens -> norm 96
    index.add("short", ["y"])
    avg = 101 / 2
    expected = idf(1, 2) * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 96 / avg))
    assert index.score(["x"])["long"] == pytest.approx(expected)
