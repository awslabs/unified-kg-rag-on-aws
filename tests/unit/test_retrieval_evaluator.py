# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the deterministic retrieval evaluator (AWS-free)."""

from __future__ import annotations

import pytest

from unified_kg_rag.application.retrieval.rag_chain import ProcessedQuery, RAGOutput
from unified_kg_rag.domain.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationQuery,
    EvaluationResult,
    EvaluatorType,
    SearchQuery,
    SearchResult,
)
from unified_kg_rag.evaluation import EvaluationManager, RetrievalEvaluator
from unified_kg_rag.evaluation.retrieval_evaluator import source_key

pytestmark = pytest.mark.unit


@pytest.fixture
def evaluator(config: Config) -> RetrievalEvaluator:
    config.evaluation.retrieval_k = 2
    return RetrievalEvaluator(config, rag_chain=None)


def _score(
    evaluator: RetrievalEvaluator, ranked: list[list[str]], refs: list[str]
) -> dict:
    result = EvaluationResult(
        query_id="q",
        question="?",
        generated_answer="a",
        ground_truth="",
        retrieved_source_ids=ranked,
        metadata={"reference_sources": refs},
    )
    report = evaluator.evaluate_single(
        EvaluationQuery(query_id="q", question="?"), result, ""
    )
    return {
        "metrics": {m.metric_type.value: m.value for m in report.metrics},
        "metadata": report.metadata,
    }


class TestSourceKey:
    @pytest.mark.parametrize(
        "identifier",
        [
            "Vendor-Terms.pdf",
            "docs/vendor-terms.pdf",
            "VENDOR-TERMS",
            "a\\vendor-terms.txt",
        ],
    )
    def test_matches_by_case_insensitive_file_stem(self, identifier: str) -> None:
        assert source_key(identifier) == "vendor-terms"

    def test_document_id_is_its_own_key(self) -> None:
        assert source_key("3f2a9c") == "3f2a9c"


class TestMetrics:
    def test_hit_recall_mrr(self, evaluator: RetrievalEvaluator) -> None:
        ranked = [["d0", "other.pdf"], ["d1", "vendor.pdf"], ["d2", "buyer.txt"]]
        out = _score(evaluator, ranked, ["Vendor.pdf", "buyer.pdf"])
        # k=2: vendor found at rank 2, buyer only at rank 3.
        assert out["metrics"] == {"hit_at_k": 1.0, "recall_at_k": 0.5, "mrr": 0.5}
        assert out["metadata"]["first_relevant_rank"] == 2
        assert out["metadata"]["k"] == 2

    def test_miss_scores_zero(self, evaluator: RetrievalEvaluator) -> None:
        out = _score(evaluator, [["a.pdf"], ["b.pdf"]], ["c.pdf"])
        assert out["metrics"] == {"hit_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}

    def test_match_by_document_id(self, evaluator: RetrievalEvaluator) -> None:
        out = _score(evaluator, [["DOC-7"]], ["doc-7"])
        assert out["metrics"]["mrr"] == 1.0

    def test_mrr_counts_beyond_k(self, evaluator: RetrievalEvaluator) -> None:
        out = _score(evaluator, [["a"], ["b"], ["c"], ["gold"]], ["gold"])
        assert out["metrics"] == {"hit_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.25}

    def test_no_references_skipped(self, evaluator: RetrievalEvaluator) -> None:
        out = _score(evaluator, [["a.pdf"]], [])
        assert out["metrics"] == {}
        assert set(out["metadata"]["skipped_metrics"].values()) == {
            "no_reference_sources"
        }

    def test_no_provenance_skipped(self, evaluator: RetrievalEvaluator) -> None:
        out = _score(evaluator, [[], []], ["a.pdf"])
        assert out["metrics"] == {}
        assert set(out["metadata"]["skipped_metrics"].values()) == {
            "no_source_provenance"
        }


def _output(sources: list[dict]) -> RAGOutput:
    return RAGOutput(
        answer="Vendor ships to Buyer.",
        sources=sources,
        search_results=SearchResult(
            query=SearchQuery(query="?"),
            results=[],
            total_results=0,
            search_strategy="local",
            processing_time=0.1,
        ),
        conversation_id=None,
        processed_query=ProcessedQuery(original_query="?", final_query="?"),
        metadata={"search_strategy": "local", "processing_time": 0.1},
    )


class _Chain:
    def __init__(self, output: RAGOutput) -> None:
        self.output = output

    async def ainvoke(self, inputs, config=None):
        return self.output

    async def abatch(self, inputs, config=None):
        return [self.output for _ in inputs]


class TestManagerIntegration:
    def test_source_ids_extracted_in_rank_order(self) -> None:
        out = _output(
            [
                {
                    "content": "chunk",
                    "metadata": {
                        "document_ids": ["doc-1"],
                        "attributes": {"file_name": "vendor.pdf"},
                    },
                },
                {"content": "report", "metadata": {"document_ids": []}},
                {"content": "x", "metadata": {"file_path": "/data/in/buyer.txt"}},
            ]
        )
        assert EvaluationManager._extract_source_ids(out) == [
            ["doc-1", "vendor.pdf"],
            [],
            ["buyer.txt"],
        ]

    async def test_reference_sources_scored_end_to_end(self, config: Config) -> None:
        config.evaluation.enabled_evaluators = [EvaluatorType.RETRIEVAL]
        chain = _Chain(
            _output(
                [
                    {"content": "c1", "metadata": {"file_name": "terms.pdf"}},
                    {"content": "c2", "metadata": {"file_name": "vendor.pdf"}},
                ]
            )
        )
        manager = EvaluationManager(config, rag_chain=chain)
        queries = [EvaluationQuery(query_id="q1", question="Who ships?")]
        gts = [
            EvaluationGroundTruth(
                query_id="q1", ground_truth="", reference_sources=["vendor.pdf"]
            )
        ]
        results, _, summary = await manager.evaluate_dataset(
            queries, gts, show_progress=False
        )
        assert results[0].retrieved_source_ids == [["terms.pdf"], ["vendor.pdf"]]
        assert summary.metric_statistics["mrr"]["mean"] == pytest.approx(0.5)
        assert summary.metric_statistics["hit_at_k"]["mean"] == 1.0
