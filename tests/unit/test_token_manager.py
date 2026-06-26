# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TokenManager — context-window budgeting/optimization shared by
all search strategies (AWS-free: the Bedrock token counter is patched out).

These exercise the real budget math, PRIORITY_MULTIPLIERS-driven ordering, the
quality-score blend, the budget-exceeded short circuit, and the empty-context
fallback string. The token counter is replaced with a deterministic
word-count stub so assertions are exact and AWS is never touched.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.retrieval import token_manager as tm_module
from unified_kg_rag.adapters.retrieval.token_manager import (
    OptimizedContext,
    SectionType,
    TokenManager,
)
from unified_kg_rag.domain.models import Config, RetrievalResult

pytestmark = pytest.mark.unit


def _make_manager(mocker, max_context_tokens: int = 200000) -> TokenManager:
    """Build a TokenManager with Bedrock/boto wiring stubbed and a deterministic
    word-count token counter (1 token per whitespace-delimited word)."""
    mocker.patch.object(tm_module, "boto3")
    mocker.patch.object(tm_module, "get_assumed_role_boto_session")

    fake_counter = mocker.Mock()
    fake_counter.count_tokens.side_effect = lambda text: len(text.split())
    mocker.patch.object(tm_module, "BedrockTokenCounter", return_value=fake_counter)

    config = Config()
    config.search.token_manager.max_context_tokens = max_context_tokens
    return TokenManager(config)


def _r(content: str, score: float, retriever_type: str, source: str) -> RetrievalResult:
    return RetrievalResult(
        content=content, score=score, retriever_type=retriever_type, source=source
    )


class TestCountTokens:
    def test_empty_string_is_zero_without_calling_counter(self, mocker) -> None:
        mgr = _make_manager(mocker)
        assert mgr.count_tokens("") == 0

    def test_delegates_to_counter(self, mocker) -> None:
        mgr = _make_manager(mocker)
        assert mgr.count_tokens("one two three") == 3


class TestBudgeting:
    def test_budget_exceeded_returns_empty_context(self, mocker) -> None:
        # query alone (3 tokens) + buffer (512) exceeds a tiny target -> short circuit.
        mgr = _make_manager(mocker, max_context_tokens=1024)
        results = [_r("a b c d e", 0.9, "text", "s1")]
        out = mgr.optimize_context(results, query="one two three", max_tokens=100)
        assert out.sections == []
        assert out.sections_included == 0
        assert out.sections_excluded == 1
        assert out.quality_score == 0.0

    def test_fits_within_budget_includes_section(self, mocker) -> None:
        mgr = _make_manager(mocker, max_context_tokens=200000)
        results = [_r("alpha beta gamma", 0.9, "text", "s1")]
        out = mgr.optimize_context(results, query="q", max_tokens=10000)
        assert out.sections_included == 1
        assert out.total_tokens == 3

    def test_max_tokens_enforced_excludes_overflow(self, mocker) -> None:
        # Budget = max_tokens - query(1) - buffer(0); set buffer to 0 for exactness.
        mgr = _make_manager(mocker)
        # Two sections of 5 tokens each; budget allows only one.
        results = [
            _r("a b c d e", 0.9, "text", "s1"),
            _r("f g h i j", 0.8, "text", "s2"),
        ]
        out = mgr.optimize_context(
            results, query="q", max_tokens=6, max_context_tokens_buffer=0
        )
        # available = 6 - 1 - 0 = 5 tokens -> exactly one 5-token section fits.
        assert out.sections_included == 1
        assert out.total_tokens == 5

    def test_zero_token_sections_skipped(self, mocker) -> None:
        mgr = _make_manager(mocker)
        results = [
            _r("", 0.9, "text", "s1"),  # 0 tokens -> skipped
            _r("real content here", 0.5, "text", "s2"),
        ]
        out = mgr.optimize_context(results, query="q", max_tokens=10000)
        assert out.sections_included == 1
        assert out.sections[0].source_id == "s2"


class TestPriorityOrdering:
    def test_priority_multiplier_orders_equal_scores(self, mocker) -> None:
        # Equal base scores; TEXT(1.3) should outrank GENERAL(0.8).
        mgr = _make_manager(mocker)
        results = [
            _r("g g g g g", 0.5, "general", "gen"),
            _r("t t t t t", 0.5, "text", "txt"),
        ]
        # Budget for only one section -> the higher-priority TEXT wins.
        out = mgr.optimize_context(
            results, query="q", max_tokens=6, max_context_tokens_buffer=0
        )
        assert out.sections_included == 1
        assert out.sections[0].section_type == SectionType.TEXT

    def test_unknown_retriever_type_falls_back_to_general(self, mocker) -> None:
        mgr = _make_manager(mocker)
        # An out-of-enum retriever_type must degrade to GENERAL, not raise
        # ValueError out of SectionType(...) and abort context-building. A future
        # retriever or a malformed result would otherwise crash the whole query.
        section = mgr._create_context_section(
            _r("hello world", 0.5, "some_future_retriever", "s1"), index=0
        )
        # GENERAL multiplier is 0.8: 0.5 * 0.8 = 0.4.
        assert section.section_type == SectionType.GENERAL
        assert section.priority == pytest.approx(0.4)

    def test_unknown_retriever_type_does_not_crash_optimize_context(
        self, mocker
    ) -> None:
        # End-to-end guard: a result with an out-of-enum type flows through the
        # public optimize_context without raising.
        mgr = _make_manager(mocker)
        results = [_r("alpha beta gamma", 0.9, "mystery_type", "s1")]
        out = mgr.optimize_context(results, query="q", max_tokens=10000)
        assert out.sections_included == 1
        assert out.sections[0].section_type == SectionType.GENERAL
        assert out.sections[0].content == "alpha beta gamma"

    def test_claim_section_type_has_weight(self, mocker) -> None:
        # Claims are evidentiary; weighted alongside relationships (1.1).
        assert SectionType.CLAIM in TokenManager.PRIORITY_MULTIPLIERS
        assert TokenManager.PRIORITY_MULTIPLIERS[SectionType.CLAIM] == pytest.approx(
            1.1
        )

    def test_claim_retriever_type_maps_to_claim_section(self, mocker) -> None:
        mgr = _make_manager(mocker)
        section = mgr._create_context_section(
            _r("a claim", 0.5, "claim", "c1"), index=0
        )
        # CLAIM multiplier is 1.1: 0.5 * 1.1 = 0.55.
        assert section.section_type == SectionType.CLAIM
        assert section.priority == pytest.approx(0.55)

    def test_missing_score_defaults_to_half(self, mocker) -> None:
        mgr = _make_manager(mocker)
        result = RetrievalResult(
            content="x y", score=0.0, retriever_type="entity", source="s"
        )
        # score 0.0 is falsy -> base_score becomes 0.5; ENTITY multiplier 1.2.
        section = mgr._create_context_section(result, index=0)
        assert section.priority == pytest.approx(0.5 * 1.2)


class TestQualityScore:
    def test_empty_inputs_zero(self, mocker) -> None:
        mgr = _make_manager(mocker)
        assert mgr._calculate_quality_score([], []) == 0.0

    def test_full_selection_scores_one(self, mocker) -> None:
        # All sections selected -> priority_coverage 1.0 and type_diversity 1.0.
        mgr = _make_manager(mocker)
        results = [
            _r("a b", 0.5, "text", "s1"),
            _r("c d", 0.5, "entity", "s2"),
        ]
        out = mgr.optimize_context(results, query="q", max_tokens=10000)
        assert out.sections_included == 2
        assert out.quality_score == pytest.approx(1.0)

    def test_partial_selection_blends_coverage_and_diversity(self, mocker) -> None:
        # Two TEXT sections, only one fits. priority_coverage = 0.5 of total
        # priority; type_diversity = 1/1 (one type) = 1.0.
        # score = 0.5*0.7 + 1.0*0.3 = 0.65.
        mgr = _make_manager(mocker)
        results = [
            _r("a b c d e", 0.5, "text", "s1"),
            _r("f g h i j", 0.5, "text", "s2"),
        ]
        out = mgr.optimize_context(
            results, query="q", max_tokens=6, max_context_tokens_buffer=0
        )
        assert out.sections_included == 1
        assert out.quality_score == pytest.approx(0.65)


class TestBuildContextString:
    def test_empty_context_fallback_message(self) -> None:
        empty = OptimizedContext(
            sections=[],
            total_tokens=0,
            sections_included=0,
            sections_excluded=0,
            quality_score=0.0,
        )
        assert (
            TokenManager.build_context_string(empty) == "No relevant information found."
        )

    def test_string_contains_headers_and_separator(self, mocker) -> None:
        mgr = _make_manager(mocker)
        results = [
            _r("first chunk text", 0.9, "text", "src-a"),
            _r("second chunk text", 0.8, "entity", "src-b"),
        ]
        out = mgr.optimize_context(results, query="q", max_tokens=10000)
        rendered = TokenManager.build_context_string(out)
        assert "Source Type: TEXT" in rendered
        assert "Source ID: src-a" in rendered
        assert "\n\n---\n\n" in rendered  # section separator
        assert "first chunk text" in rendered
