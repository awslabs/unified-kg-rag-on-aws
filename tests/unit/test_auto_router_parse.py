# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""AUTO strategy-router response parsing.

Regression: the router parsed the LLM response with SearchStrategy(text.strip()
.lower()), which raised ValueError on anything but a bare enum word ("Local
search.", "I'd use local") and then fell back to DRIFT — the MOST expensive
strategy. Parsing now tolerates surrounding text and defaults to LOCAL.
"""

from __future__ import annotations

import pytest

from aws_graphrag.application.retrieval.rag_chain import GraphRAGChain
from aws_graphrag.domain.models import SearchStrategy

pytestmark = pytest.mark.unit

parse = GraphRAGChain._parse_routed_strategy


def test_exact_word() -> None:
    assert parse("local") == SearchStrategy.LOCAL
    assert parse("GLOBAL") == SearchStrategy.GLOBAL
    assert parse("  drift  ") == SearchStrategy.DRIFT
    assert parse("simple") == SearchStrategy.SIMPLE


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
