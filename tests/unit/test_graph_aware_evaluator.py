# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the graph-aware evaluator (M4)."""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationResult,
    EvaluatorType,
)
from unified_kg_rag.evaluation import EvaluationManager, GraphAwareEvaluator

pytestmark = pytest.mark.unit


@pytest.fixture
def evaluator(config: Config) -> GraphAwareEvaluator:
    return GraphAwareEvaluator(config)


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
        # Coverage only (no duplicated precision/F1), plus the answer length;
        # retrieved coverage is skipped without entity/relationship sources.
        assert types == {
            EvaluationMetricType.ENTITY_COVERAGE,
            EvaluationMetricType.RELATIONSHIP_COVERAGE,
            EvaluationMetricType.ANSWER_LENGTH,
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
        assert {m.metric_type.value: m.value for m in report.metrics} == {
            "entity_coverage": 1.0,
            "relationship_coverage": 1.0,
            "answer_length": float(len("Alice works at Acme")),
        }
        assert report.metadata["matched_entity_count"] == 1
        assert report.metadata["matched_relationship_count"] == 1
        assert report.metadata["skipped_metrics"] == {
            "retrieved_entity_coverage": "no_entity_sources",
            "retrieved_relationship_coverage": "no_relationship_sources",
        }

    def test_absent_dimension_recorded_as_skipped(
        self, evaluator: GraphAwareEvaluator
    ) -> None:
        report = evaluator.evaluate_single(
            EvaluationQuery(query_id="q1", question="?"),
            self._result("Alice", expected_entities=["Alice"]),
            ground_truth="",
        )
        assert report.metadata["skipped_metrics"] == {
            "relationship_coverage": "no_expected_relationships",
            "retrieved_relationship_coverage": "no_expected_relationships",
            "retrieved_entity_coverage": "no_entity_sources",
        }

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
        assert sync.metrics == asyncr.metrics

    def test_missing_dimension_not_penalized(
        self, evaluator: GraphAwareEvaluator
    ) -> None:
        # Only entities expected: the relationship dimension is not scored 0.
        report = evaluator.evaluate_single(
            EvaluationQuery(query_id="q1", question="?"),
            self._result("Alice", expected_entities=["Alice"]),
            ground_truth="",
        )
        coverage = [
            m.value
            for m in report.metrics
            if m.metric_type is not EvaluationMetricType.ANSWER_LENGTH
        ]
        assert coverage == [1.0]
        # No relationship metrics emitted when none are expected.
        assert all("relationship" not in m.metric_type.value for m in report.metrics)


class TestRelationshipCoverage:
    ANSWER = "The Vendor ships spare parts to the Buyer every month."

    def _cov(self, evaluator: GraphAwareEvaluator, expected: list) -> tuple:
        return evaluator._coverage(expected, self.ANSWER, relationships=True)

    def test_pair_counts_when_both_endpoints_mentioned(
        self, evaluator: GraphAwareEvaluator
    ) -> None:
        assert self._cov(evaluator, [{"source": "Vendor", "target": "Buyer"}]) == (
            1.0,
            1,
        )

    def test_pair_needs_both_endpoints(self, evaluator: GraphAwareEvaluator) -> None:
        assert self._cov(evaluator, [{"source": "Vendor", "target": "Auditor"}]) == (
            0.0,
            0,
        )

    def test_arrow_string_is_a_pair(self, evaluator: GraphAwareEvaluator) -> None:
        assert self._cov(evaluator, ["vendor -> buyer", "Vendor -> Auditor"]) == (
            0.5,
            1,
        )

    def test_plain_string_still_needs_the_phrase(
        self, evaluator: GraphAwareEvaluator
    ) -> None:
        assert self._cov(evaluator, ["ships spare parts"]) == (1.0, 1)
        assert self._cov(evaluator, ["Vendor supplies Buyer"]) == (0.0, 0)

    def test_arrow_not_parsed_for_entities(
        self, evaluator: GraphAwareEvaluator
    ) -> None:
        assert evaluator._coverage(["Vendor -> Buyer"], self.ANSWER) == (0.0, 0)

    def test_pair_without_target_rejected_at_load(self) -> None:
        with pytest.raises(ValueError, match="source.*target"):
            EvaluationGroundTruth(
                query_id="q",
                ground_truth="",
                expected_relationships=[{"source": "Vendor"}],
            )


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
        assert reports and {
            m.value
            for m in reports[0].metrics
            if m.metric_type is not EvaluationMetricType.ANSWER_LENGTH
        } == {1.0}

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
        assert reports and reports[0].metrics == []


class TestRetrievedCoverage:
    def _report(self, evaluator: GraphAwareEvaluator, answer: str, **fields):
        result = EvaluationResult(
            query_id="q1",
            question="?",
            generated_answer=answer,
            ground_truth="",
            metadata={
                "expected_entities": ["Vendor", "Auditor"],
                "expected_relationships": [{"source": "Buyer", "target": "Vendor"}],
            },
            **fields,
        )
        report = evaluator.evaluate_single(
            EvaluationQuery(query_id="q1", question="?"), result, ""
        )
        return {m.metric_type.value: m.value for m in report.metrics}, report.metadata

    def test_scores_the_retrieved_graph_sources(
        self, evaluator: GraphAwareEvaluator
    ) -> None:
        values, metadata = self._report(
            evaluator,
            "No.",
            retrieved_source_types=["entity", "relationship"],
            retrieved_entities=["VENDOR CORP"],
            retrieved_relationships=[
                {"source": "VENDOR", "target": "BUYER", "description": "ships to"}
            ],
        )
        # The answer names nothing, but the context had Vendor and the
        # Buyer-Vendor edge (direction-insensitive).
        assert values["entity_coverage"] == 0.0
        assert values["retrieved_entity_coverage"] == 0.5
        assert values["retrieved_relationship_coverage"] == 1.0
        assert values["answer_length"] == 3.0
        assert metadata["retrieved_matched_entity_count"] == 1

    def test_wordy_answer_does_not_raise_retrieved_coverage(
        self, evaluator: GraphAwareEvaluator
    ) -> None:
        sources = {
            "retrieved_source_types": ["entity"],
            "retrieved_entities": ["Vendor"],
        }
        short, _ = self._report(evaluator, "Vendor.", **sources)
        wordy, _ = self._report(
            evaluator, "Vendor, Auditor, Buyer and every other party.", **sources
        )
        assert wordy["entity_coverage"] > short["entity_coverage"]
        assert wordy["retrieved_entity_coverage"] == short["retrieved_entity_coverage"]
        assert wordy["answer_length"] > short["answer_length"]

    def test_skipped_without_sources_of_the_type(
        self, evaluator: GraphAwareEvaluator
    ) -> None:
        values, metadata = self._report(
            evaluator, "Vendor", retrieved_source_types=["community"]
        )
        assert "retrieved_entity_coverage" not in values
        assert metadata["skipped_metrics"] == {
            "retrieved_entity_coverage": "no_entity_sources",
            "retrieved_relationship_coverage": "no_relationship_sources",
        }

    def test_plain_relationship_string_matches_description(
        self, evaluator: GraphAwareEvaluator
    ) -> None:
        cov, matched = evaluator._retrieved_coverage(
            ["ships spare parts"],
            [{"source": "A", "target": "B", "description": "A ships spare parts to B"}],
        )
        assert (cov, matched) == (1.0, 1)
