# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for HybridScorer — the shared fusion/rerank/diversity core used by
every GraphRAG and LightRAG search strategy (AWS-free)."""

from __future__ import annotations

import pytest

from aws_graphrag.adapters.retrieval.hybrid_scorer import HybridScorer
from aws_graphrag.domain.models import Config, FusionMethod, RetrievalResult

pytestmark = pytest.mark.unit


def _scorer(config: Config | None = None) -> HybridScorer:
    config = config or Config()
    # Disable reranking so no Bedrock client is constructed.
    config.search.reranking.enabled = False
    return HybridScorer(config)


def _r(content: str, score: float | None, source: str) -> RetrievalResult:
    return RetrievalResult(
        content=content, score=score, source=source, retriever_type="test"
    )


class TestNormalizeScores:
    def test_empty_list(self) -> None:
        assert _scorer()._normalize_scores([]) == []

    def test_constant_scores_get_half(self) -> None:
        results = [_r("a", 3.0, "s1"), _r("b", 3.0, "s2")]
        out = _scorer()._normalize_scores(results)
        assert all(r.score == 0.5 for r in out)

    def test_min_max_normalized_to_unit_range(self) -> None:
        results = [_r("a", 10.0, "s1"), _r("b", 20.0, "s2"), _r("c", 15.0, "s3")]
        out = _scorer()._normalize_scores(results)
        by_content = {r.content: r.score for r in out}
        assert by_content["a"] == 0.0
        assert by_content["b"] == 1.0
        assert by_content["c"] == 0.5


class TestReciprocalRankFusion:
    def test_higher_rank_contributes_more(self) -> None:
        scorer = _scorer()
        k = scorer.fusion_config.rrf_k
        result_map = {"src": [_r("first", 0.9, "1"), _r("second", 0.8, "2")]}
        fused = scorer._reciprocal_rank_fusion(result_map)
        by_content = {r.content: r.score for r in fused}
        assert by_content["first"] == pytest.approx(1.0 / (k + 1))
        assert by_content["second"] == pytest.approx(1.0 / (k + 2))
        assert by_content["first"] > by_content["second"]

    def test_cross_source_contributions_accumulate(self) -> None:
        scorer = _scorer()
        k = scorer.fusion_config.rrf_k
        # Same content+source in two lists, both at rank 1.
        shared = _r("shared", 0.9, "x")
        result_map = {
            "a": [shared.model_copy()],
            "b": [shared.model_copy()],
        }
        fused = scorer._reciprocal_rank_fusion(result_map)
        assert len(fused) == 1
        assert fused[0].score == pytest.approx(2.0 / (k + 1))

    def test_fused_object_is_a_copy(self) -> None:
        scorer = _scorer()
        original = _r("c", 0.9, "x")
        fused = scorer._reciprocal_rank_fusion({"a": [original]})
        assert fused[0] is not original


class TestWeightedFusion:
    def test_per_source_weights_applied(self) -> None:
        config = Config()
        config.search.reranking.enabled = False
        config.search.fusion.fusion_weights = {"a": 2.0, "b": 0.5}
        scorer = HybridScorer(config)
        result_map = {"a": [_r("x", 1.0, "1")], "b": [_r("y", 1.0, "2")]}
        fused = scorer._weighted_fusion(result_map)
        by_content = {r.content: r.score for r in fused}
        assert by_content["x"] == pytest.approx(2.0)
        assert by_content["y"] == pytest.approx(0.5)

    def test_unweighted_source_defaults_to_one(self, caplog) -> None:
        config = Config()
        config.search.reranking.enabled = False
        config.search.fusion.fusion_weights = {"known": 2.0}
        scorer = HybridScorer(config)
        result_map = {"unknown_bucket": [_r("x", 1.0, "1")]}
        fused = scorer._weighted_fusion(result_map)
        assert fused[0].score == pytest.approx(1.0)


class TestResultKey:
    def test_same_content_and_source_collapse(self) -> None:
        scorer = _scorer()
        assert scorer._get_result_key(_r("same", 0.1, "s")) == scorer._get_result_key(
            _r("same", 0.9, "s")
        )

    def test_different_source_distinct(self) -> None:
        scorer = _scorer()
        assert scorer._get_result_key(_r("same", 0.1, "s1")) != scorer._get_result_key(
            _r("same", 0.1, "s2")
        )


class TestFuseAndRerank:
    def test_unknown_fusion_method_raises(self) -> None:
        scorer = _scorer()
        scorer.fusion_config.method = "nonexistent"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown fusion method"):
            scorer.fuse_and_rerank_results({"a": [_r("x", 1.0, "1")]}, top_k=5)

    def test_sorted_desc_and_truncated_to_top_k(self) -> None:
        config = Config()
        config.search.reranking.enabled = False
        config.search.fusion.method = FusionMethod.RRF
        config.search.fusion.diversity_lambda = 1.0  # skip diversity filtering
        scorer = HybridScorer(config)
        result_map = {"a": [_r("x", 0.5, "1"), _r("y", 0.4, "2"), _r("z", 0.3, "3")]}
        out = scorer.fuse_and_rerank_results(result_map, top_k=2)
        assert len(out) == 2
        assert out[0].score >= out[1].score

    def test_metrics_recorded(self) -> None:
        config = Config()
        config.search.reranking.enabled = False
        config.search.fusion.diversity_lambda = 1.0
        scorer = HybridScorer(config)
        scorer.fuse_and_rerank_results({"a": [_r("x", 1.0, "1")]}, top_k=5)
        metrics = scorer.get_metrics()
        assert metrics["metrics"]["final_fused_count"] == 1


class TestDiversityFiltering:
    def test_skipped_at_lambda_one(self) -> None:
        config = Config()
        config.search.reranking.enabled = False
        config.search.fusion.diversity_lambda = 1.0
        scorer = HybridScorer(config)
        results = [_r("a b c", 0.9, "1"), _r("a b c", 0.8, "2")]
        out = scorer._apply_diversity_filtering(list(results), top_k=10)
        assert len(out) == 2  # no filtering

    def test_penalizes_redundant_at_low_lambda(self) -> None:
        config = Config()
        config.search.reranking.enabled = False
        config.search.fusion.diversity_lambda = 0.1  # strong diversity
        scorer = HybridScorer(config)
        # Two near-duplicates + one distinct; MMR should prefer the distinct one
        # over the redundant duplicate for the 2nd slot.
        results = [
            _r("alpha beta gamma", 1.0, "1"),
            _r("alpha beta gamma delta", 0.95, "2"),
            _r("completely different content here", 0.5, "3"),
        ]
        out = scorer._apply_diversity_filtering(list(results), top_k=2)
        assert out[0].content == "alpha beta gamma"
        assert out[1].content == "completely different content here"


class TestRerankDegradation:
    def test_returns_original_when_rerank_model_raises(self) -> None:
        scorer = _scorer()

        class _BoomModel:
            top_n = 5

            def compress_documents(self, documents, query):  # noqa: ANN001
                raise RuntimeError("rerank backend down")

        scorer.rerank_model = _BoomModel()
        results = [_r("a", 0.9, "1"), _r("b", 0.8, "2")]
        out = scorer._apply_bedrock_reranking(results, query="q")
        assert out == results  # unchanged on failure
        assert _BoomModel.top_n == 5  # restored in finally
