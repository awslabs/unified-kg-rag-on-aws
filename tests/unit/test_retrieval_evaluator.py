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
from unified_kg_rag.evaluation.retrieval_evaluator import source_keys

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
    values = {m.metric_type.value: m.value for m in report.metrics}
    return {
        "fraction": values.pop("attributable_fraction", None),
        "metrics": values,
        "metadata": report.metadata,
    }


class TestSourceKeys:
    @pytest.mark.parametrize(
        "identifier",
        [
            "Vendor-Terms.pdf",
            "docs/vendor-terms.pdf",
            "VENDOR-TERMS",
            "a\\vendor-terms.txt",
            "s3://bucket/in/Vendor-Terms.md",
        ],
    )
    def test_matches_by_case_insensitive_file_stem(self, identifier: str) -> None:
        assert "vendor-terms" in source_keys(identifier)

    def test_full_name_and_stem(self) -> None:
        assert source_keys("Vendor.PDF") == {"vendor.pdf", "vendor"}

    @pytest.mark.parametrize(
        ("title", "key"),
        [
            ("St. Louis Cardinals", "st. louis cardinals"),
            ("U.S. Route 66", "u.s. route 66"),
            ("AC/DC", "ac/dc"),
            ("Version 2.0", "version 2.0"),
        ],
    )
    def test_titles_kept_whole(self, title: str, key: str) -> None:
        assert source_keys(title) == {key}

    def test_leading_slash_is_a_path(self) -> None:
        assert source_keys("/data/in/Buyer Notes") == {"buyer notes"}

    def test_blank_has_no_keys(self) -> None:
        assert source_keys("  ") == frozenset()


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

    def test_title_with_dots_matches_only_itself(
        self, evaluator: RetrievalEvaluator
    ) -> None:
        # The old stem rule reduced both to "st" and matched them.
        out = _score(evaluator, [["St. Paul Saints"]], ["St. Louis Cardinals"])
        assert out["metrics"]["hit_at_k"] == 0.0
        out = _score(evaluator, [["St. Louis Cardinals.txt"]], ["St. Louis Cardinals"])
        assert out["metrics"]["hit_at_k"] == 1.0

    def test_duplicate_references_counted_once(
        self, evaluator: RetrievalEvaluator
    ) -> None:
        out = _score(evaluator, [["vendor.pdf"]], ["vendor.pdf", "docs/Vendor.pdf"])
        assert out["metrics"]["recall_at_k"] == 1.0
        assert out["metadata"]["num_references"] == 1

    def test_mrr_counts_beyond_k(self, evaluator: RetrievalEvaluator) -> None:
        out = _score(evaluator, [["a"], ["b"], ["c"], ["gold"]], ["gold"])
        assert out["metrics"] == {"hit_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.25}

    def test_no_references_skipped(self, evaluator: RetrievalEvaluator) -> None:
        out = _score(evaluator, [["a.pdf"]], [])
        assert out["metrics"] == {}
        assert out["fraction"] == 1.0  # attribution is measured regardless
        assert out["metadata"]["skipped_metrics"] == {
            "hit_at_k": "no_reference_sources",
            "recall_at_k": "no_reference_sources",
            "mrr": "no_reference_sources",
        }

    def test_no_sources_is_a_miss(self, evaluator: RetrievalEvaluator) -> None:
        out = _score(evaluator, [], ["a.pdf"])
        assert out["metrics"] == {"hit_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}
        assert out["fraction"] is None
        assert out["metadata"]["num_sources"] == 0
        assert out["metadata"]["skipped_metrics"] == {
            "attributable_fraction": "no_sources"
        }

    def test_no_provenance_skipped(self, evaluator: RetrievalEvaluator) -> None:
        out = _score(evaluator, [[], []], ["a.pdf"])
        assert out["metrics"] == {}
        assert out["fraction"] == 0.0
        assert set(out["metadata"]["skipped_metrics"].values()) == {
            "no_source_provenance"
        }

    def test_unattributable_sources_do_not_take_top_k_slots(
        self, evaluator: RetrievalEvaluator
    ) -> None:
        # Two unattributable sources ahead of the gold one: k=2 still sees it.
        out = _score(evaluator, [[], [], ["other.pdf"], ["gold.pdf"]], ["gold.pdf"])
        assert out["metrics"] == {"hit_at_k": 1.0, "recall_at_k": 1.0, "mrr": 0.5}
        assert out["fraction"] == 0.5
        assert out["metadata"]["num_attributable_sources"] == 2


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
    def test_source_provenance_extracted_in_rank_order(self) -> None:
        out = _output(
            [
                {
                    "content": "chunk",
                    "metadata": {
                        "document_ids": ["doc-1"],
                        "attributes": {"file_name": "vendor.pdf"},
                    },
                },
                {
                    "content": "report",
                    "metadata": {"document_ids": ["h1"], "text_unit_ids": ["t1", "t2"]},
                },
                {"content": "x", "metadata": {"file_path": "/data/in/buyer.txt"}},
                {"content": "entity", "metadata": {"text_unit_ids": "t3"}},
                {
                    "content": "bare chunk",
                    "metadata": {"section_type": "text", "chunk_id": "t4"},
                },
            ]
        )
        assert EvaluationManager._extract_source_provenance(out) == [
            (["vendor.pdf"], []),
            ([], ["t1", "t2"]),  # document ids (content hashes) are not used
            (["buyer.txt"], []),
            ([], ["t3"]),  # Neptune unwraps a single-element list to a str
            ([], ["t4"]),
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


class _Resolver:
    def __init__(self, mapping: dict[str, str], fail: bool = False) -> None:
        self.mapping = mapping
        self.fail = fail
        self.calls: list[tuple[list[str], str | None]] = []

    async def aresolve(self, text_unit_ids, suffix):
        ids = list(text_unit_ids)
        self.calls.append((ids, suffix))
        if self.fail:
            raise RuntimeError("store unavailable")
        return {i: self.mapping[i] for i in ids if i in self.mapping}


class TestGraphSourceAttribution:
    def _chain(self) -> _Chain:
        return _Chain(
            _output(
                [
                    # A community report: no file, lineage only.
                    {"content": "report", "metadata": {"text_unit_ids": ["t1", "t2"]}},
                    {"content": "entity", "metadata": {"text_unit_ids": ["t9"]}},
                    {"content": "chunk", "metadata": {"file_name": "terms.pdf"}},
                ]
            )
        )

    async def _run(self, config: Config, resolver: _Resolver):
        config.evaluation.enabled_evaluators = [EvaluatorType.RETRIEVAL]
        manager = EvaluationManager(
            config, rag_chain=self._chain(), source_resolver=resolver
        )
        queries = [
            EvaluationQuery(
                query_id="q1", question="Who ships?", metadata={"suffix": "v2"}
            )
        ]
        gts = [
            EvaluationGroundTruth(
                query_id="q1", ground_truth="", reference_sources=["vendor.pdf"]
            )
        ]
        return await manager.evaluate_dataset(queries, gts, show_progress=False)

    async def test_lineage_resolved_to_files(self, config: Config) -> None:
        resolver = _Resolver({"t1": "vendor.pdf", "t2": "buyer.pdf"})
        results, _, summary = await self._run(config, resolver)
        assert results[0].retrieved_source_ids == [
            ["buyer.pdf", "vendor.pdf"],
            [],  # t9 unknown to the store: unattributable
            ["terms.pdf"],
        ]
        # One batched lookup per suffix, only for sources without a file name.
        assert resolver.calls == [(["t1", "t2", "t9"], "v2")]
        assert summary.metric_statistics["mrr"]["mean"] == 1.0
        assert summary.metric_statistics["attributable_fraction"][
            "mean"
        ] == pytest.approx(2 / 3)
        assert summary.grouped_statistics["search_strategy"]["local"][
            "attributable_fraction"
        ]["mean"] == pytest.approx(2 / 3)

    async def test_resolver_failure_leaves_sources_unattributable(
        self, config: Config
    ) -> None:
        results, _, summary = await self._run(config, _Resolver({}, fail=True))
        assert results[0].retrieved_source_ids == [[], [], ["terms.pdf"]]
        assert summary.metric_statistics["hit_at_k"]["mean"] == 0.0

    def test_default_resolver_only_for_graph_rag_chain(self, config: Config) -> None:
        manager = EvaluationManager(config, rag_chain=self._chain())
        assert manager.source_resolver is None
