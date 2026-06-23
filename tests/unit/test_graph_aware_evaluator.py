# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for the graph-aware evaluator (M4)."""

from __future__ import annotations

import pytest

from aws_graphrag.domain.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationResult,
    EvaluatorType,
)
from aws_graphrag.evaluation import EvaluationManager, GraphAwareEvaluator

pytestmark = pytest.mark.unit


@pytest.fixture
def evaluator(config: Config) -> GraphAwareEvaluator:
    return GraphAwareEvaluator(config, rag_chain=None)


class TestCoverage:
    def test_full_match(self, evaluator: GraphAwareEvaluator) -> None:
        cov, m = evaluator._coverage(["Alice", "Acme"], "Alice and Acme")
        assert (cov, m) == (1.0, 2)

    def test_partial_match_case_insensitive(
        self, evaluator: GraphAwareEvaluator
    ) -> None:
        cov, m = evaluator._coverage(["Alice", "Bob"], "ALICE only")
        assert m == 1
        assert cov == 0.5

    def test_empty_expected_returns_none(self, evaluator: GraphAwareEvaluator) -> None:
        # None coverage so an absent dimension is excluded from the overall score.
        assert evaluator._coverage([], "anything") == (None, 0)

    def test_no_false_substring_match(self, evaluator: GraphAwareEvaluator) -> None:
        # Word-boundary matching: "AI" must NOT match inside "airport".
        cov, matched = evaluator._coverage(["AI"], "the airport is busy")
        assert matched == 0 and cov == 0.0

    def test_word_match_succeeds(self, evaluator: GraphAwareEvaluator) -> None:
        cov, matched = evaluator._coverage(["AI"], "we use AI models")
        assert matched == 1 and cov == 1.0

    def test_multiword_phrase_requires_contiguous_order(
        self, evaluator: GraphAwareEvaluator
    ) -> None:
        assert evaluator._coverage(["works at"], "she works at acme")[1] == 1
        # Same words, wrong order -> no match.
        assert evaluator._coverage(["at works"], "she works at acme")[1] == 0

    def test_punctuation_is_stripped(self, evaluator: GraphAwareEvaluator) -> None:
        # "Acme" should match despite surrounding punctuation.
        assert evaluator._coverage(["Acme"], "owned by Acme, Inc.")[1] == 1

    def test_cjk_coverage_via_substring(self, evaluator: GraphAwareEvaluator) -> None:
        # Space-less scripts have no word tokens; coverage falls back to a
        # substring check so CJK answers aren't always scored 0.
        cov, matched = evaluator._coverage(["東京電力"], "本契約は東京電力と締結された")
        assert matched == 1 and cov == 1.0
        # A CJK entity absent from the answer still scores 0.
        assert evaluator._coverage(["大阪ガス"], "本契約は東京電力と締結された")[1] == 0


class TestEvaluateSingle:
    def _result(self, answer: str, **md) -> EvaluationResult:
        return EvaluationResult(
            query_id="q1",
            question="?",
            generated_answer=answer,
            ground_truth="",
            metadata=md,
        )

    def test_emits_one_coverage_metric_per_dimension(
        self, evaluator: GraphAwareEvaluator
    ) -> None:
        report = evaluator.evaluate_single(
            EvaluationQuery(query_id="q1", question="?"),
            self._result(
                "Alice works at Acme",
                expected_entities=["Alice", "Acme"],
                expected_relationships=["works at"],
            ),
            ground_truth="",
        )
        types = {m.metric_type for m in report.metrics}
        # Coverage only (no duplicated precision/F1).
        assert types == {
            EvaluationMetricType.ENTITY_COVERAGE,
            EvaluationMetricType.RELATIONSHIP_COVERAGE,
        }
        assert report.evaluator_type is EvaluatorType.GRAPH_AWARE

    def test_perfect_coverage_scores_one(self, evaluator: GraphAwareEvaluator) -> None:
        report = evaluator.evaluate_single(
            EvaluationQuery(query_id="q1", question="?"),
            self._result(
                "Alice works at Acme",
                expected_entities=["Alice"],
                expected_relationships=["works at"],
            ),
            ground_truth="",
        )
        assert report.overall_score == 1.0
        assert report.metadata["matched_entity_count"] == 1
        assert report.metadata["matched_relationship_count"] == 1

    def test_missing_expectations_score_zero(
        self, evaluator: GraphAwareEvaluator
    ) -> None:
        report = evaluator.evaluate_single(
            EvaluationQuery(query_id="q1", question="?"),
            self._result("irrelevant answer", expected_entities=["Zeta"]),
            ground_truth="",
        )
        entity_coverage = next(
            m.value
            for m in report.metrics
            if m.metric_type == EvaluationMetricType.ENTITY_COVERAGE
        )
        assert entity_coverage == 0.0

    async def test_async_matches_sync(self, evaluator: GraphAwareEvaluator) -> None:
        query = EvaluationQuery(query_id="q1", question="?")
        result = self._result("Alice", expected_entities=["Alice"])
        sync = evaluator.evaluate_single(query, result, "")
        asyncr = await evaluator.aevaluate_single(query, result, "")
        assert sync.overall_score == asyncr.overall_score

    def test_missing_dimension_not_penalized(
        self, evaluator: GraphAwareEvaluator
    ) -> None:
        # Only entities expected; perfect entity match -> overall 1.0 (no
        # relationship dimension dragging it to ~0.5).
        report = evaluator.evaluate_single(
            EvaluationQuery(query_id="q1", question="?"),
            self._result("Alice", expected_entities=["Alice"]),
            ground_truth="",
        )
        assert report.overall_score == 1.0
        # No relationship metrics emitted when none are expected.
        assert all("relationship" not in m.metric_type.value for m in report.metrics)


class TestManagerThreading:
    """Verify EvaluationManager threads expected_* onto result.metadata."""

    def _manager(self, config: Config) -> EvaluationManager:
        config.evaluation.enabled_evaluators = [EvaluatorType.GRAPH_AWARE]
        # rag_chain is only needed for answer generation, not _evaluate_results.
        return EvaluationManager(config, rag_chain=object())

    async def test_threads_expectations_and_scores(self, config: Config) -> None:
        manager = self._manager(config)
        query = EvaluationQuery(query_id="q1", question="?")
        result = EvaluationResult(
            query_id="q1",
            question="?",
            generated_answer="Alice works at Acme",
            ground_truth="",
        )
        gt = EvaluationGroundTruth(
            query_id="q1",
            ground_truth="...",
            expected_entities=["Alice", "Acme"],
            expected_relationships=["works at"],
        )
        reports = await manager._evaluate_results([query], [result], [gt])
        # Manager copied the expectations onto the result metadata.
        assert result.metadata["expected_entities"] == ["Alice", "Acme"]
        assert reports and reports[0].overall_score == 1.0

    async def test_query_id_mismatch_is_safe(self, config: Config) -> None:
        manager = self._manager(config)
        query = EvaluationQuery(query_id="q1", question="?")
        result = EvaluationResult(
            query_id="q1", question="?", generated_answer="x", ground_truth=""
        )
        gt = EvaluationGroundTruth(query_id="OTHER", ground_truth="...")
        reports = await manager._evaluate_results([query], [result], [gt])
        # No matching ground truth -> no expectations -> zero coverage, no crash.
        assert "expected_entities" not in result.metadata
        assert reports and reports[0].overall_score == 0.0
