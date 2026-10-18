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
    return RetrievalEvaluator(config)


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


def test_output_blocked_answer_is_skipped_not_a_miss(
    evaluator: RetrievalEvaluator,
) -> None:
    # Before, the sources the guardrail dropped scored as a retrieval miss (0).
    result = EvaluationResult(
        query_id="q",
        question="?",
        generated_answer="Answer blocked.",
        ground_truth="",
        retrieved_source_ids=[],
        metadata={"reference_sources": ["a.txt"]},
        guardrail_blocked="output",
    )
    report = evaluator.evaluate_single(
        EvaluationQuery(query_id="q", question="?"), result, ""
    )
    assert report.metrics == []
    assert set(report.metadata["skipped_metrics"].values()) == {"guardrail_blocked"}
    assert set(report.metadata["skipped_metrics"]) == {
        m.value for m in evaluator.metric_types()
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

    def test_file_name_matches_its_spaced_title(self) -> None:
        file_keys = source_keys("docs/Vendor_Terms_(2024).txt")
        assert source_keys("Vendor Terms (2024)") <= file_keys
        assert source_keys("Vendor  Terms (2024)") <= file_keys

    def test_leading_slash_is_a_path(self) -> None:
        assert source_keys("/data/in/Buyer Notes") == {"buyer notes"}

    def test_unicode_forms_compare_equal(self) -> None:
        # macOS and many PDF tools write decomposed (NFD) Hangul file names;
        # full-width Latin letters NFKC-fold to ASCII; casefold covers "ß".
        nfd = "\u1100\u1168\u110b\u1163\u11a8\u1109\u1165_\u1100\u1161\u1102\u1161\u1103\u1161\u1109\u1161\u11bc\u1109\u1161.pdf"
        assert nfd != "계약서_가나다상사.pdf"
        assert source_keys(nfd) == source_keys("계약서 가나다상사.pdf")
        assert source_keys("ＡＢＣ.pdf") == source_keys("abc.pdf")
        assert source_keys("Straße.txt") == source_keys("STRASSE.txt")

    def test_blank_has_no_keys(self) -> None:
        assert source_keys("  ") == frozenset()


class TestMetrics:
    def test_hit_recall_mrr(self, evaluator: RetrievalEvaluator) -> None:
        ranked = [["other.pdf"], ["vendor.pdf"], ["buyer.txt"]]
        out = _score(evaluator, ranked, ["Vendor.pdf", "buyer.pdf"])
        # k=2: vendor found at rank 2, buyer only at rank 3.
        assert out["metrics"] == {"hit_at_k": 1.0, "recall_at_k": 0.5, "mrr": 0.5}
        assert out["metadata"]["first_relevant_rank"] == 2
        assert out["metadata"]["k"] == 2

    def test_k_counts_files_not_sources(self, evaluator: RetrievalEvaluator) -> None:
        # A community report attributed to many files must not fill the top k
        # with all of them: k=2 counts the first two distinct files.
        ranked = [["a.pdf", "b.pdf", "c.pdf"], ["vendor.pdf"]]
        out = _score(evaluator, ranked, ["vendor.pdf"])
        assert out["metrics"] == {"hit_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.25}
        assert out["metadata"]["first_relevant_rank"] == 4
        assert out["metadata"]["num_ranked_files"] == 4
        assert out["fraction"] == 1.0

    def test_repeated_file_counts_once(self, evaluator: RetrievalEvaluator) -> None:
        ranked = [["a.pdf"], ["A.pdf"], ["vendor.pdf"]]
        out = _score(evaluator, ranked, ["vendor.pdf"])
        assert out["metrics"]["mrr"] == 0.5
        assert out["metrics"]["hit_at_k"] == 1.0

    def test_scores_per_source_type(self, config: Config) -> None:
        config.evaluation.retrieval_k = 1
        result = EvaluationResult(
            query_id="q",
            question="?",
            generated_answer="a",
            ground_truth="",
            retrieved_source_ids=[["a.pdf", "vendor.pdf"], ["vendor.pdf"], []],
            retrieved_source_types=["community", "text", "entity"],
            metadata={"reference_sources": ["vendor.pdf"]},
        )
        report = RetrievalEvaluator(config).evaluate_single(
            EvaluationQuery(query_id="q", question="?"), result, ""
        )
        # The unattributable entity source has no files: no entity row.
        assert report.metadata["by_source_type"] == {
            "community": {"hit_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.5},
            "text": {"hit_at_k": 1.0, "recall_at_k": 1.0, "mrr": 1.0},
        }

    def test_random_community_reports_score_like_random_text_units(
        self, config: Config
    ) -> None:
        # Measured regression (H2): five random reports attributed to 15 of 50
        # files each scored hit@5 ~0.98 when k counted sources, against ~0.19
        # for five random text units. Counting files, both are chance level.
        import random

        config.evaluation.retrieval_k = 5
        evaluator = RetrievalEvaluator(config)
        files = [f"doc{i}.pdf" for i in range(50)]
        rng = random.Random(0)

        def hit_rate(draw) -> float:
            hits = 0.0
            for _ in range(400):
                result = EvaluationResult(
                    query_id="q",
                    question="?",
                    generated_answer="a",
                    ground_truth="",
                    retrieved_source_ids=draw(),
                    metadata={"reference_sources": rng.sample(files, 2)},
                )
                report = evaluator.evaluate_single(
                    EvaluationQuery(query_id="q", question="?"), result, ""
                )
                hits += next(
                    m.value for m in report.metrics if m.metric_type.value == "hit_at_k"
                )
            return hits / 400

        units = hit_rate(lambda: [[f] for f in rng.sample(files, 5)])
        reports = hit_rate(lambda: [rng.sample(files, 15) for _ in range(5)])
        assert units < 0.3 and reports < 0.3

    def test_miss_scores_zero(self, evaluator: RetrievalEvaluator) -> None:
        out = _score(evaluator, [["a.pdf"], ["b.pdf"]], ["c.pdf"])
        assert out["metrics"] == {"hit_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}

    def test_match_by_document_id(self, evaluator: RetrievalEvaluator) -> None:
        out = _score(evaluator, [["DOC-7"]], ["doc-7"])
        assert out["metrics"]["mrr"] == 1.0

    @pytest.mark.parametrize(
        ("source", "reference", "hit"),
        [
            ("Report-A.pdf", "docs/Report-A.pdf", 1.0),
            ("Report-A.pdf", "report-a.pdf", 1.0),
            ("Report-A.pdf", "report-a", 1.0),
            # an extension in the reference must agree
            ("Report-A.docx", "Report-A.pdf", 0.0),
            # a version suffix is part of the name, not an extension
            ("notes.v2", "notes.v1", 0.0),
            ("notes.v2", "notes", 0.0),
            ("notes.v2", "Notes.V2", 1.0),
        ],
    )
    def test_stem_matches_only_a_reference_without_extension(
        self, evaluator: RetrievalEvaluator, source: str, reference: str, hit: float
    ) -> None:
        out = _score(evaluator, [[source]], [reference])
        assert out["metrics"]["hit_at_k"] == hit

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
        # Files rank: buyer (1), vendor (2), terms (3).
        assert summary.metric_statistics["mrr"]["mean"] == 0.5
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

    async def test_lineage_files_ordered_by_support(self, config: Config) -> None:
        # A report built mostly from vendor.pdf ranks vendor.pdf first.
        config.evaluation.enabled_evaluators = [EvaluatorType.RETRIEVAL]
        config.evaluation.retrieval_k = 1
        chain = _Chain(
            _output(
                [
                    {
                        "content": "report",
                        "metadata": {
                            "section_type": "community",
                            "text_unit_ids": ["t1", "t2", "t3"],
                        },
                    }
                ]
            )
        )
        resolver = _Resolver({"t1": "a.pdf", "t2": "vendor.pdf", "t3": "vendor.pdf"})
        manager = EvaluationManager(config, rag_chain=chain, source_resolver=resolver)
        results, _, summary = await manager.evaluate_dataset(
            [EvaluationQuery(query_id="q1", question="?")],
            [
                EvaluationGroundTruth(
                    query_id="q1", ground_truth="", reference_sources=["vendor.pdf"]
                )
            ],
            show_progress=False,
        )
        assert results[0].retrieved_source_ids == [["vendor.pdf", "a.pdf"]]
        assert results[0].retrieved_source_types == ["community"]
        assert summary.metric_statistics["hit_at_k"]["mean"] == 1.0
        by_type = summary.grouped_statistics["source_type"]
        assert by_type["community"]["hit_at_k"]["mean"] == 1.0


class TestSourceDetails:
    def test_types_entities_and_relationships_extracted(self) -> None:
        out = _output(
            [
                {"content": "c", "metadata": {"section_type": "text"}},
                {
                    "content": "e",
                    "metadata": {"section_type": "entity", "name": "VENDOR"},
                },
                {
                    "content": "r",
                    "metadata": {
                        "section_type": "relationship",
                        "source_name": "VENDOR",
                        "target_name": "BUYER",
                        "description": "ships parts to",
                    },
                },
                {"content": "no metadata"},
            ]
        )
        details = EvaluationManager._extract_source_details(out)
        assert details == {
            "retrieved_source_types": ["text", "entity", "relationship", ""],
            "retrieved_entities": ["VENDOR"],
            "retrieved_relationships": [
                {"source": "VENDOR", "target": "BUYER", "description": "ships parts to"}
            ],
        }
