# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AUTO strategy-router response parsing.

Regression: the router parsed the LLM response with SearchStrategy(text.strip()
.lower()), which raised ValueError on anything but a bare enum word ("Local
search.", "I'd use local") and then fell back to DRIFT — the MOST expensive
strategy. Parsing now tolerates surrounding text and defaults to LOCAL.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import SearchStrategy

pytestmark = pytest.mark.unit

parse = GraphRAGChain._parse_routed_strategy


def test_exact_word() -> None:
    assert parse("local") == SearchStrategy.LOCAL
    assert parse("GLOBAL") == SearchStrategy.GLOBAL
    assert parse("  drift  ") == SearchStrategy.DRIFT
    assert parse("mix") == SearchStrategy.MIX


def test_tolerates_surrounding_text() -> None:
    assert parse("Local search.") == SearchStrategy.LOCAL
    assert parse("I'd recommend global") == SearchStrategy.GLOBAL


def test_unknown_defaults_to_local_not_drift() -> None:
    # The crux: an unparseable response must NOT land on the costly DRIFT.
    assert parse("banana") == SearchStrategy.LOCAL
    assert parse("") == SearchStrategy.LOCAL


def test_exact_match_precedence_over_substring() -> None:
    # "global" contains no other strategy name; ensure exact wins cleanly.
    assert parse("global") == SearchStrategy.GLOBAL


def test_first_mentioned_strategy_wins_not_list_order() -> None:
    # A substring scan returned whichever name it checked first.
    assert parse("drift, or else local") == SearchStrategy.DRIFT
    assert parse("local rather than drift") == SearchStrategy.LOCAL


def test_words_are_matched_whole() -> None:
    # "mixture" / "globally" are not strategy names.
    assert parse("a mixture, globally") == SearchStrategy.LOCAL


def test_simple_is_not_routable_by_default() -> None:
    # simple stays selectable explicitly but is not an AUTO target by default.
    assert parse("simple") == SearchStrategy.LOCAL
    assert parse("simple", (SearchStrategy.SIMPLE,)) == SearchStrategy.SIMPLE


def test_default_routable_set_and_fast_router_model() -> None:
    from unified_kg_rag.domain.models import Config, LanguageModelId

    search = Config().search
    assert search.auto_routable_strategies == [
        SearchStrategy.LOCAL,
        SearchStrategy.MIX,
        SearchStrategy.GLOBAL,
        SearchStrategy.DRIFT,
    ]
    assert search.strategy_selection_model_id == LanguageModelId.CLAUDE_V4_5_HAIKU
