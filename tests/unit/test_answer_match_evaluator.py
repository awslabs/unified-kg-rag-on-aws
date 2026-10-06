# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the deterministic answer-match evaluator (AWS-free)."""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.models import (
    Config,
    EvaluationQuery,
    EvaluationResult,
    EvaluatorType,
)
from unified_kg_rag.evaluation import AnswerMatchEvaluator, EvaluationManager
from unified_kg_rag.evaluation.answer_match_evaluator import (
    exact_match,
    normalize_answer,
    token_f1,
)

pytestmark = pytest.mark.unit


class TestNormalization:
    def test_squad_normalization(self) -> None:
        assert normalize_answer("The  Vendor, Inc.!") == "vendor inc"

    def test_unicode_punctuation_removed(self) -> None:
        assert normalize_answer("「공급사」는…") == "공급사는"

    @pytest.mark.parametrize(
        ("prediction", "reference"),
        [
            ("1,000", "1000"),  # punctuation deleted, not replaced by a space
            ("USD 1,000.", "usd 1000"),
            ("\uff11\uff10\uff10\uff10", "1000"),  # fullwidth digits (NFKC)
            ("Vendor\u2019s", "Vendors"),
            ("the U.S.", "US"),
        ],
    )
    def test_squad_equivalences(self, prediction: str, reference: str) -> None:
        assert exact_match(prediction, reference) == 1.0

    def test_decimal_point_deleted_like_squad(self) -> None:
        assert normalize_answer("3.5 days") == "35 days"

    def test_articles_only_as_whole_words(self) -> None:
        assert normalize_answer("Theater an Anchor") == "theater anchor"

    def test_exact_match_ignores_case_articles_punctuation(self) -> None:
        assert exact_match("the Buyer.", "Buyer") == 1.0
        assert exact_match("Buyer Ltd", "Buyer") == 0.0

    def test_token_f1(self) -> None:
        # pred tokens {vendor, ships, parts}, ref {vendor, ships, tools}
        assert token_f1("Vendor ships parts", "vendor ships tools") == pytest.approx(
            2 / 3
        )
        assert token_f1("unrelated", "vendor") == 0.0
        assert token_f1("", "") == 1.0
        assert token_f1("a", "vendor") == 0.0  # "a" normalizes to empty


def _report(config: Config, answer: str, ground_truth: str, metadata: dict):
    evaluator = AnswerMatchEvaluator(config, rag_chain=None)
    result = EvaluationResult(
        query_id="q",
        question="?",
        generated_answer=answer,
        ground_truth=ground_truth,
        metadata=metadata,
    )
    return evaluator.evaluate_single(
        EvaluationQuery(query_id="q", question="?"), result, ground_truth
    )


class TestEvaluator:
    def test_max_over_aliases(self, config: Config) -> None:
        report = _report(
            config,
            "Net 30 days",
            "thirty days",
            {"answer_aliases": ["net 30 days", "30 days"]},
        )
        values = {m.metric_type.value: m.value for m in report.metrics}
        assert values == {"exact_match": 1.0, "token_f1": 1.0}

    def test_single_string_alias_accepted(self, config: Config) -> None:
        report = _report(config, "USD 500", "", {"answer_aliases": "usd 500"})
        assert {m.metric_type.value: m.value for m in report.metrics} == {
            "exact_match": 1.0,
            "token_f1": 1.0,
        }

    def test_no_reference_skipped(self, config: Config) -> None:
        report = _report(config, "anything", "", {})
        assert report.metrics == []
        assert set(report.metadata["skipped_metrics"]) == {"exact_match", "token_f1"}

    def test_resolver_maps_type(self) -> None:
        assert (
            EvaluationManager._resolve_evaluator_class(EvaluatorType.ANSWER_MATCH)
            is AnswerMatchEvaluator
        )
